# -*- coding: utf-8 -*-
import copy
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.datasets.mixup_data as mixup
import pcode.local_training.compressor as compressor
from pcode.models.generator import Generator
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker, LogitTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.timer import Timer
from pruneagent import PruneAgent


class Worker(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        if self.conf.nlp_input:
            total_devices = torch.cuda.device_count()
            self.device = torch.device(
                "cuda:" + str(self.conf.graph.worker_id % total_devices) if self.conf.graph.on_cuda else "cpu")
            info = "cuda:" + str(self.conf.graph.worker_id % total_devices)
            conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} use the device {info}"
            )
        else:
            self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=conf.logger.log_metric,
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )
        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # define the model compression operators.
        if conf.local_model_compression is not None:
            if conf.local_model_compression == "quantization":
                self.model_compression_fn = compressor.ModelQuantization(conf)

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )
        if self.conf.global_history or self.conf.local_history:
            self.global_models_buffer = []
            self.global_dicts = []
            self.global_tbs = []
        if self.conf.global_logits:
            self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
            self.logit_tracker = LogitTracker(unique_labels=self.conf.num_classes, device=self.device)
        elif self.conf.generator:
            self.generative_model = Generator(self.conf.data, self.device)

        self.prune_agent = PruneAgent(10, self.device)
        self.clients_comm_round = np.zeros(self.conf.n_clients)
        self.prune_agent.original_filters_number = 0

    def run(self):
        while True:

            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return
            if self.conf.global_logits:
                self._recv_logits_from_master()
            elif self.conf.generator:
                self._recv_generator_from_master()
            self._recv_model_from_master()

            # self._getFLOPs()

            self._train()

            self._prune()

            if self.conf.global_logits:
                self._send_logits_to_master()
            elif self.conf.generator:
                self._send_label_counts_to_master()
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return

    def _prune(self):

        if self.prune_agent.pruned_ratio.item() != 0:
            self.conf.logger.log(
                f'worker-{self.conf.graph.worker_id} pruned {int(self.prune_agent.pruned_ratio * self.prune_agent.retention_number)} filer')
            self.optimizer = self.prune_agent.prune(self.model, self.optimizer,
                                                    int(self.prune_agent.pruned_ratio * self.prune_agent.retention_number))

    def _getFLOPs(self):
        from thop import profile
        from thop import clever_format
        filename = './FLOPs/worker' + str(self.conf.graph.worker_id) + '.log'
        model = copy.deepcopy(self.model)
        with open(filename, 'a') as f:
            input = torch.randn(1, 3, 32, 32)
            flops, params = profile(model, inputs=(input,))
            flops, params = clever_format([flops, params], "%.3f")
            print(flops, params)
            f.write(f'comm: ' + str(self.comm_round) + ' FLOPs: ' + flops + ' params: ' + params)
        del model

    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg_len = 3
        if self.conf.global_history:
            msg_len += 1
        msg = torch.zeros((msg_len, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        # self.buffer_len
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        self.clients_comm_round[(msg[0].int() - 1).tolist()] += 1
        self.comm_round = self.clients_comm_round[self.conf.graph.client_id - 1]
        self.global_comm_round = self.conf.graph.comm_round
        self.conf.logger.log(f'client:{self.conf.graph.client_id} is activated')

        if self.conf.global_history:
            self.global_models_buffer_len = msg[3][self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()

        # once we receive the signal, we init for the local training.

        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        if self.global_comm_round == 1:
            self.prune_agent.get_original_filters_number(self.model)
            self.prune_agent.master_model = self.model

        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        if self.conf.generator:
            self.generator_state_dict = self.generative_model.state_dict()
            self.generator_tb = TensorBuffer(list(self.generator_state_dict.values()))
        if self.conf.global_history:
            while len(self.global_models_buffer) < self.global_models_buffer_len:
                self.global_models_buffer.append(copy.deepcopy(self.model))
                global_dict = copy.deepcopy(self.model_state_dict)
                self.global_dicts.append(global_dict)
                self.global_tbs.append(TensorBuffer(list(global_dict.values())))

        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

    def _recv_logits_from_master(self):
        dist.recv(tensor=self.global_logits, src=0)

        if self.conf.graph.on_cuda:
            self.global_logits = self.global_logits.to(self.device)
        dist.barrier()

    def _recv_generator_from_master(self):
        old_buffer = copy.deepcopy(self.generator_tb.buffer)
        dist.recv(tensor=self.generator_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.generator_tb.buffer)
        self.generator_tb.unpack(self.generator_state_dict.values())

        self.generative_model.load_state_dict(self.generator_state_dict)

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the generator ({self.arch}) from Master. The generator status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        retention_number = torch.tensor(0, dtype=int)
        dist.recv(retention_number, src=0)
        self.prune_agent.retention_number = int(retention_number)
        self.prune_agent.original_idx_len = len(self.model_tb.buffer)
        self.prune_agent.dense_chs_idx = {}
        for name, param in self.prune_agent.master_model.named_parameters():
            if 'weight' in name:
                if param.ndim == 1:
                    self.prune_agent.dense_chs_idx[name] = np.arange(param.shape[0])
                if param.ndim == 2:
                    self.prune_agent.dense_chs_idx[name] = np.arange(param.shape[1])
                if param.ndim == 4:
                    self.prune_agent.dense_chs_idx[name] = {'in_chs': np.arange(param.shape[1]),
                                                            'out_chs': np.arange(param.shape[0])}
        if self.prune_agent.retention_number != self.prune_agent.original_filters_number:
            self.prune_agent.prune(self.model, self.optimizer,
                                   0)
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        old_buffer = copy.deepcopy(self.model_tb.buffer)

        self.prune_agent.pruned_ratio = torch.tensor(0.0, dtype=float)
        if self.comm_round > 1:
            idx_len = torch.tensor(0, dtype=int)
            dist.recv(idx_len, src=0)
        if self.comm_round > 2:
            buffer = torch.zeros(self.prune_agent.original_idx_len)
            dist.recv(buffer, src=0)
            assert len(self.model_tb.buffer) == idx_len
            self.model_tb.buffer = buffer[:idx_len]
        else:
            dist.recv(self.model_tb.buffer, src=0)

        dist.recv(self.prune_agent.pruned_ratio, src=0)

        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())

        if self.comm_round > 1:
            self.prev_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))

        else:
            self.prev_model = None
            self.prune_agent.idx = torch.tensor(np.arange(len(new_buffer)))
            self.prune_agent.retention_ratio = 1

        self.model.load_state_dict(self.model_state_dict)

        if self.comm_round == 2:
            original_number = torch.tensor(0, dtype=int)
            dist.recv(original_number, src=0)
            self.prune_agent.original_number = original_number.item()
            self.prune_agent.original_idx_len = len(self.model_tb.buffer)
            self.prune_agent.original_score = torch.zeros(self.prune_agent.original_number)
            dist.recv(self.prune_agent.original_score, src=0)
            self.prune_agent.master_model = copy.deepcopy(self.model)
            self.prune_agent.dense_chs_idx = {}
            for name, param in self.prune_agent.master_model.named_parameters():

                if 'weight' in name:
                    if param.ndim == 1:
                        self.prune_agent.dense_chs_idx[name] = np.arange(param.shape[0])
                    if param.ndim == 2:
                        self.prune_agent.dense_chs_idx[name] = np.arange(param.shape[1])
                    if param.ndim == 4:
                        self.prune_agent.dense_chs_idx[name] = {'in_chs': np.arange(param.shape[1]),
                                                                'out_chs': np.arange(param.shape[0])}

            idx = 0
            self.prune_agent.original_score_list = {}
            self.prune_agent.score_list = {}
            self.prune_agent.bn_idx_list = {}
            for name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    score = self.model_tb.buffer[idx:idx + m.weight.shape[0]]
                    self.prune_agent.score_list.update({m.weight: score})
                    self.prune_agent.original_score_list.update({m.weight: score})
                    self.prune_agent.bn_idx_list.update({m.weight: np.arange(idx, idx + m.weight.shape[0]).tolist()})
                    idx += m.weight.shape[0]

        # random_reinit.random_reinit_model(self.conf, self.model)

        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))
        # self.aggregation = Aggregation(self.model.classifier.in_features).cuda()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        if self.conf.global_history:
            for i in range(self.global_models_buffer_len):
                old_buffer = copy.deepcopy(self.global_tbs[i].buffer)
                dist.recv(self.global_tbs[i].buffer, src=0)
                new_buffer = copy.deepcopy(self.global_tbs[i].buffer)
                self.global_tbs[i].unpack(self.global_dicts[i].values())
                self.global_models_buffer[i].load_state_dict(self.global_dicts[i])
                self.global_models_buffer[i] = self.global_models_buffer[i].to(self.device)

                self.conf.logger.log(
                    f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the history model:{i} from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
                )

        dist.barrier()

    def _train(self):
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            # localdata_id=self.conf.graph.client_id - 1,
            localdata_id=np.random.randint(100),
            is_train=True,
            data_partitioner=self.data_partitioner,
        )
        if self.global_comm_round == 10:
            self.prune_agent.get_score(self.model)
        # define optimizer, scheduler and runtime tracker.
        # models = [self.model]
        # if self.conf.mutual_history:
        #     for global_model in self.global_models_buffer:
        #         models.append(global_model)
        #     models.append(self.aggregation)
        if self.conf.graph.comm_round == 100:
            self.conf.lr *= 0.1
        # if self.conf.graph.comm_round == 150:
        #     self.conf.lr *= 0.1
        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )
        if self.conf.local_prox_term != 0:
            global_weight_collector = list(self.init_model.parameters())
        if self.conf.generator:
            self.clean_up_counts()
            self.generative_model.to(self.device)
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True, device=self.device
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss, output, feature = self._inference(data_batch)

                    if self.conf.local_prox_term != 0:
                        for param_index, param in enumerate(self.model.parameters()):
                            loss += ((self.conf.local_prox_term / 2) * torch.norm(
                                (param - global_weight_collector[param_index])) ** 2)

                    if self.conf.contrastive:
                        loss = self._local_training_with_contrastive(loss, feature, data_batch)
                    elif self.conf.generator:
                        if self.scheduler.epoch_ == 0:
                            samples = self.get_count_labels(data_batch['target'])
                            self.update_label_counts(samples['labels'], samples['counts'])
                        loss = self._local_training_with_generator(loss, output, data_batch, self.scheduler.epoch_)
                    elif self.conf.global_history or self.conf.local_history:
                        loss = self._local_training_with_historical_distillation(loss, output, data_batch)
                    elif self.conf.mutual_history:
                        loss = self._local_training_with_historical_mutual(loss, output, data_batch)
                    elif self.conf.global_logits:
                        self.logit_tracker.update(logits=output, Y=data_batch["target"])
                        loss = self._local_training_with_logits_distillation(loss, output, data_batch)
                    elif self.conf.grp_lasso_coeff:
                        self._local_training_with_group_lasso_global(loss, output)
                    else:
                        loss = self._local_training_with_self_distillation(
                            loss, output, data_batch
                        )

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward()

                    # if self.global_comm_round<50:
                    #     for m in self.model.modules():
                    #         if isinstance(m, nn.BatchNorm2d):
                    #             m.weight.grad.data.add_(0.00005 * torch.sign(m.weight.data))

                    # self._add_grad_from_prox_regularized_loss()
                    self.optimizer.step()
                    self.scheduler.step()

                # efficient local training.
                with self.timer("compress_model", epoch=self.scheduler.epoch_):
                    if hasattr(self, "model_compression_fn"):
                        self.model_compression_fn.compress_model(
                            param_groups=self.optimizer.param_groups
                        )

                # # display the logging info.
                # display_training_stat(self.conf, self.scheduler, self.tracker)

                # display tracking time.
                if (
                        self.conf.display_tracked_time
                        and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    self.conf.logger.log(self.timer.summary())

                # check divergence.
                # if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
                #         self.tracker.stat["loss"].vg
                # ):

                # if loss > 1e3 or torch.isnan(loss):
                #     self.conf.logger.log(
                #         f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                #     )
                #     self._terminate_comm_round()
                #     return

                # check stopping condition.
                if self._is_finished_one_comm_round():
                    self._terminate_comm_round()
                    return

            # display the logging info.
            display_training_stat(self.conf, self.scheduler, self.tracker)
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        feature, output = self.model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            bsz = data_batch["target"].size(0)
            self.tracker.update_local_metrics(
                loss.item(), 0, n_samples=bsz
            )
            for idx in range(1, 1 + len(performance)):
                self.tracker.update_local_metrics(
                    performance[idx - 1], idx, n_samples=bsz
                )
        return loss, output, feature

    def _add_grad_from_prox_regularized_loss(self):
        assert self.conf.local_prox_term >= 0
        if self.conf.local_prox_term != 0:
            assert self.conf.weight_decay == 0
            assert self.conf.optimizer == "sgd"
            assert self.conf.momentum_factor == 0

            for _param, _init_param in zip(
                    self.model.parameters(), self.init_model.parameters()
            ):
                if _param.grad is not None:
                    _param.grad.data.add_(
                        (_param.data - _init_param.data) * self.conf.local_prox_term
                    )

    def _local_training_with_generator(self, loss, output, data_batch, epoch):

        generative_alpha = 1
        generative_beta = 1
        gen_ratio = 1
        if self.conf.graph.comm_round > 1:
            self.generative_model.eval()
            generative_alpha = self.exp_lr_scheduler(self.conf.graph.comm_round, decay=0.98, init_lr=generative_alpha)
            generative_beta = self.exp_lr_scheduler(self.conf.graph.comm_round, decay=0.98, init_lr=generative_beta)

            gen_output = self.generative_model(data_batch["target"], latent_layer_idx=-1)['output']
            with torch.no_grad():
                _, logit_given_gen = self.model(gen_output, start_layer_idx=-1)
            user_latent_loss = generative_beta * self._divergence(output, logit_given_gen)

            sampled_y = np.random.choice(self.conf.num_classes, self.conf.batch_size)
            sampled_y = torch.tensor(sampled_y).to(self.device)
            gen_result = self.generative_model(sampled_y, latent_layer_idx=-1)
            gen_output = gen_result['output']  # latent representation when latent = True, x otherwise

            _, user_output = self.model(gen_output, start_layer_idx=-1)
            teacher_loss = generative_alpha * torch.mean(
                self.criterion(user_output, sampled_y)
            )
            # this is to further balance oversampled down-sampled synthetic data

            loss2 = gen_ratio * teacher_loss + user_latent_loss
            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=output.size(0)
                )
            loss = loss + loss2

        return loss

    def _local_training_with_logits_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:
            loss2 = self._divergence(output / self.conf.self_distillation_temperature,
                                     self.global_logits[data_batch["target"],
                                     :] / self.conf.self_distillation_temperature)
            loss2 *= self.conf.self_distillation
            loss = loss + loss2
            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=output.size(0)
                )

        return loss

    def _local_training_with_group_lasso_global(self, loss, output):
        lasso_in_ch = []
        lasso_out_ch = []
        model = self.model
        arch = self.arch
        for name, param in model.named_parameters():
            # Lasso added to only the neuronal layers
            if ('weight' in name) and any([i for i in ['conv', 'classifier'] if i in name]):
                if param.dim() == 4:

                    # Exclude depth-wise convolution layers from regularization

                    if 'conv1.' not in name:
                        _in = param.pow(2).sum(dim=[0, 2, 3])
                        lasso_in_ch.append(_in)

                    _out = param.pow(2).sum(dim=[1, 2, 3])
                    lasso_out_ch.append(_out)

                elif param.dim() == 2:
                    # Multi-FC-layer based classifier (only fc or fc3 are the last layers)
                    # if 'classifier' in name:
                    #     lasso_out_ch.append(param.pow(2).sum(dim=[1]))
                    lasso_in_ch.append(param.pow(2).sum(dim=[0]))

        _lasso_in_ch = torch.cat(lasso_in_ch).cuda()
        _lasso_out_ch = torch.cat(lasso_out_ch).cuda()

        lasso_penalty_in_ch = _lasso_in_ch.add(1.0e-8).sqrt().sum()
        lasso_penalty_out_ch = _lasso_out_ch.add(1.0e-8).sqrt().sum()

        lasso_penalty = lasso_penalty_in_ch + lasso_penalty_out_ch
        loss2 = lasso_penalty * self.conf.grp_lasso_coeff
        loss = loss + loss2.to(self.device)

        if self.tracker is not None:
            self.tracker.update_local_metrics(
                loss2.item(), -1, n_samples=output.size(0)
            )
        return loss

    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:
            with torch.no_grad():
                _, teacher_logits = self.init_model(data_batch["input"])
            loss2 = self.conf.self_distillation * self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=teacher_logits / self.conf.self_distillation_temperature,
            )
            loss = loss + loss2

            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=data_batch["target"].size(0)
                )
        return loss

    def _local_training_with_historical_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:

            _, teacher_logits = self.init_model(data_batch["input"])
            loss2 = self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=teacher_logits.detach() / self.conf.self_distillation_temperature,
            )
            extra_loss = 0
            if self.global_models_buffer_len > 0:
                _, teacher_logits = self.global_models_buffer[0](data_batch["input"])

                extra_loss += self._divergence(
                    student_logits=output / self.conf.self_distillation_temperature,
                    teacher_logits=teacher_logits.detach() / self.conf.self_distillation_temperature
                )
            if self.global_models_buffer_len > 1:
                _, teacher_logits = self.global_models_buffer[1](data_batch["input"])
                extra_loss += self._divergence(
                    student_logits=output / self.conf.self_distillation_temperature,
                    teacher_logits=teacher_logits.detach() / self.conf.self_distillation_temperature
                )
            loss2 += extra_loss / 2
            loss2 = self.conf.self_distillation * loss2
            loss = loss + loss2

            if self.tracker is not None and loss2 != 0:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=data_batch["target"].size(0)
                )
        return loss

    def _local_training_with_historical_mutual(self, loss, output, data_batch):
        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:
            _, teacher_logit = self.init_model(data_batch["input"])
            loss2 = self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=teacher_logit / self.conf.self_distillation_temperature
            )
            loss_true = 0
            if len(self.global_models_buffer) > 0 and loss < 0.5:
                teacher_logits = [output]
                for model in self.global_models_buffer:
                    _, teacher_logit = model(data_batch["input"])
                    loss_true += F.cross_entropy(teacher_logit, data_batch["target"])

                    teacher_logits.append(teacher_logit)

                teacher_logits = torch.stack(teacher_logits, dim=-1)
                # weighted_logits = torch.mean(teacher_logits,dim=-1)
                loss_group = 0
                for i in range(len(self.global_models_buffer) + 1):
                    for j in range(len(self.global_models_buffer) + 1):
                        if j != i:
                            loss_group += self._divergence(
                                student_logits=teacher_logits[:, :, i] / self.conf.self_distillation_temperature,
                                teacher_logits=teacher_logits[:, :, j] / self.conf.self_distillation_temperature,
                            ) / len(self.global_models_buffer)
                loss2 += loss_group
                # print(f"loss2:{loss2:.4f}, loss_true:{loss_true:.4f}, loss_group:{loss_group:.4f}")

            loss2 = self.conf.self_distillation * loss2
            loss = loss + loss2 + loss_true

            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=data_batch["target"].size(0)
                )
        return loss

    def _local_training_with_contrastive(self, loss, feature, data_batch):

        if self.conf.self_distillation > 0 and self.conf.graph.comm_round > 1:
            bsz = data_batch["target"].size(0)
            with torch.no_grad():
                teacher_feature, _ = self.init_model(data_batch["input"])
                prev_feature, _ = self.prev_model(data_batch["input"])
            logits = self.similarity(feature, teacher_feature).reshape(-1, 1)
            nega = self.similarity(feature, prev_feature).reshape(-1, 1)
            logits = torch.cat((logits, nega), dim=1)
            logits /= self.conf.self_distillation_temperature
            labels = torch.zeros(bsz).to(self.device).long()
            loss2 = self.conf.self_distillation * self.criterion(logits, labels)
            loss = loss + loss2

            if self.tracker is not None:
                self.tracker.update_local_metrics(
                    loss2.item(), -1, n_samples=bsz
                )
        return loss

    def _divergence(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def similarity(self, x1, x2):
        sim = F.cosine_similarity(x1, x2, dim=-1)
        return sim

    def _turn_off_grad(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_logits_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the logits back to Master."
        )
        logit_avg = self.logit_tracker.avg()
        logit_avg = logit_avg.clone().detach().cpu()
        dist.send(tensor=logit_avg, dst=0)
        dist.barrier()

    def _send_label_counts_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the label_counts back to Master."
        )
        label_counts = self.label_counts.detach().cpu()
        dist.send(tensor=label_counts, dst=0)
        dist.barrier()

    def _send_model_to_master(self):
        dist.barrier()

        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        self.prune_agent.get_idx()
        idx_len = torch.tensor(len(self.prune_agent.idx))
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        dist.send(tensor=idx_len, dst=0)
        dist.send(tensor=flatten_model.buffer.cpu(), dst=0)
        dist.send(tensor=self.prune_agent.idx, dst=0)
        dist.send(tensor=torch.tensor(time.time(), dtype=float), dst=0)
        dist.send(tensor=torch.tensor(self.prune_agent.retention_ratio, dtype=float), dst=0)
        dist.send(tensor=torch.tensor(self.prune_agent.retention_number, dtype=int), dst=0)
        self.conf.logger.log(
            f'worker-{self.conf.graph.worker_id},clinet-{self.conf.graph.client_id} retention_ratio is {self.prune_agent.retention_ratio}')
        dist.barrier()

    def _terminate_comm_round(self):

        # history_model = self._turn_off_grad(copy.deepcopy(self.model).cuda())
        self.model = self.model.cpu()

        if self.conf.global_logits:
            self.global_logits = self.global_logits.cpu()

        if self.conf.mutual_history:
            history_model = copy.deepcopy(self.init_model)
            history_model.train()
            history_model.requires_grad_(True)

        if self.conf.local_history:
            history_model = copy.deepcopy(self.init_model)
            if len(self.global_models_buffer) < 1:
                self.global_models_buffer.append(history_model)
            elif len(self.global_models_buffer) < 2:
                old_net = self.global_models_buffer[0]
                self.global_models_buffer[0] = history_model
                self.global_models_buffer.append(old_net)
            else:
                self.global_models_buffer[1] = self.global_models_buffer[0]
                self.global_models_buffer[0] = history_model

        del self.init_model
        if self.prev_model is not None:
            del self.prev_model
        if self.conf.global_history:
            for i in range(len(self.global_models_buffer)):
                self.global_models_buffer[i] = self.global_models_buffer[i].cpu()
        self.scheduler.clean()
        self.conf.logger.save_json()
        torch.cuda.empty_cache()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label) - 1] += count

    def clean_up_counts(self):
        self.label_counts = torch.ones(self.conf.num_classes)

    def get_count_labels(self, y):
        result = {}
        unique_y, counts = torch.unique(y, return_counts=True)
        unique_y = unique_y.detach().cpu().numpy()
        counts = counts.detach().cpu().numpy()
        result['labels'] = unique_y
        result['counts'] = counts
        return result
