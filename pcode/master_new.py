# -*- coding: utf-8 -*-
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
import pcode.utils.cross_entropy as cross_entropy
from pcode.models.generator import Generator
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pruned_rate_learning import PrunedRateLearning


class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        # create model as well as their corresponding state_dicts.
        _, self.master_model = create_model.define_model(
            conf, to_consistent_model=False
        )
        self.used_client_archs = set(
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")

        self.client_models = dict(
            create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch
        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
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
        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader]

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        # define the aggregators.
        self.aggregator = create_aggregator.Aggregator(
            conf,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch
        )
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.logger.log(f"Master initialized the aggregator/coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )
        if self.conf.global_history:
            self.global_models_buffer = []
            self.global_perfomances = [0, 0]
            self.global_best_performance = 0
        if self.conf.global_logits:
            self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
        if self.conf.generator:
            self.device = "cuda" if self.conf.graph.on_cuda else "cpu"
            self.generative_model = Generator(self.conf.data, self.device)

            self.generative_optimizer = torch.optim.Adam(
                params=self.generative_model.parameters(),
                lr=1e-4, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)

            self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.generative_optimizer, gamma=0.98)
        # save arguments to disk.
        conf.is_finished = False
        checkpoint.save_arguments(conf)

        # pruning ratio
        self.pruned_rate_learning = PrunedRateLearning(self.conf.min_retention, self.conf.min_pruned,
                                                       self.conf.max_pruned,
                                                       self.conf.init_cofe)
        self.heterogeneity = self.pruned_rate_learning.get_heterogeneity(self.conf.heterogeneity_ratio,
                                                                         self.conf.n_participated)

        original_filters_number = 0
        for m in self.master_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                original_filters_number += m.weight.shape[0]

        for selected_client_id in range(1, self.conf.n_clients + 1):
            if selected_client_id not in self.pruned_rate_learning.workers.keys():
                self.pruned_rate_learning.workers[selected_client_id] = {'retention_ratio': [], 'update_time': [],
                                                                         'idx': None, 'send_time': None,
                                                                         'recv_time': None,
                                                                         'retention_number': original_filters_number,
                                                                         'prune_time': 0}

    def run(self):
        performance, to_send_history = 0, False
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.

            if self.conf.random_select != None:
                # import random
                # selected_client_ids = self._random_select_clients()
                # random.shuffle(selected_client_ids)
                selected_client_ids = []
                for i in range(10):
                    selected_client_ids.append(np.random.randint(i * 10 + 1, (i + 1) * 10 + 1))
            else:
                if comm_round == 1:
                    selected_client_ids = self._random_select_clients()

            # detect early stopping.
            self._check_early_stopping()

            if self.conf.global_history:
                to_send_history = (self.global_perfomances[0] > performance) and (
                            self.global_perfomances[1] > performance)
                if to_send_history:
                    self.conf.logger.log(f"Master send history models to clients."
                                         f"history 1 performance:{self.global_perfomances[0]},"
                                         f"history 2 performance:{self.global_perfomances[1]},"
                                         f"current performance:{performance}")
            # init the activation tensor and broadcast to all clients (either start or stop).
            self._activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs, to_send_history
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                if self.conf.global_logits:
                    self._send_logit_to_selected_clients(selected_client_ids)
                elif self.conf.generator:
                    self._send_generator_to_selected_clients(selected_client_ids)

                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids, to_send_history)

                if self.conf.global_history:
                    self.buffer_global_models(performance)
            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            if self.conf.global_logits:
                self.aggregate_logits(selected_client_ids)
            elif self.conf.generator:
                label_weights, qualified_labels = self._receive_label_counts_from_selected_clients(selected_client_ids)

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            )

            if self.conf.generator:
                self.train_generator(selected_client_ids, label_weights, qualified_labels, flatten_local_models)

            # aggregate the local models and evaluate on the validation dataset.
            performance = self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def aggregate_logits(self, selected_client_ids):

        self.conf.logger.log(f"Master waits to receive the local logits.")
        dist.barrier()
        # init the placeholders to recv the local models from workers.
        logit_avgs = dict()
        for selected_client_id in selected_client_ids:
            logit_avgs[selected_client_id] = torch.zeros(self.conf.num_classes, self.conf.num_classes)

        # async to receive model from clients.
        self.global_logits = torch.zeros(self.conf.num_classes, self.conf.num_classes)
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=logit_avgs[client_id], src=world_id
            )

            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local logits.")
        for v in logit_avgs.values():
            self.global_logits += v
        self.global_logits /= len(selected_client_ids)
        print(self.global_logits)
        return self.global_logits

    def buffer_global_models(self, performance):
        same_arch = (
                len(self.client_models) == 1
                and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        )
        if not same_arch:
            return
        if performance > self.global_best_performance:
            self.global_best_performance = performance

        if len(self.global_models_buffer) == 0:
            self.global_models_buffer.append(self.master_model)
        # if performance > self.global_perfomances[0]:
        self.global_perfomances[1] = self.global_perfomances[0]
        self.global_perfomances[0] = performance

        if len(self.global_models_buffer) == 1:
            old_net = self.global_models_buffer[0]
            self.global_models_buffer[0] = self.master_model
            self.global_models_buffer.append(old_net)
        elif len(self.global_models_buffer) == 2:
            self.global_models_buffer[1] = self.global_models_buffer[0]
            self.global_models_buffer[0] = self.master_model

        # elif performance > self.global_perfomances[1]:
        #     self.global_perfomances[1] = performance
        #     if len(self.global_models_buffer) == 1:
        #         self.global_models_buffer.append(self.master_model)
        #     elif len(self.global_models_buffer) == 2:
        #         self.global_models_buffer[1] = self.master_model

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        msg_len = 3
        if self.conf.global_history:
            msg_len += 1
        activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
        activation_msg[0, :] = torch.tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.tensor(list_of_local_n_epochs)
        if self.conf.global_history:
            if to_send_history:
                activation_msg[3, :] = len(self.global_models_buffer)
            else:
                activation_msg[3, :] = 0
        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_logit_to_selected_clients(self, selected_client_ids):
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            dist.send(tensor=self.global_logits, dst=worker_rank)

        self.conf.logger.log(f"Master send the logits to clients")
        dist.barrier()

    def _send_generator_to_selected_clients(self, selected_client_ids):
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            generative_model = self.generative_model.state_dict()
            flatten_model = TensorBuffer(list(generative_model.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

        self.conf.logger.log(f"Master send the generator to workers.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids, to_send_history=False):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(self.master_model.state_dict().values()))
            self.pruned_rate_learning.workers[selected_client_id]['send_time'] = torch.tensor(time.time(), dtype=float)

            dist.send(tensor=torch.tensor(self.pruned_rate_learning.workers[selected_client_id]['retention_number'],
                                          dtype=int), dst=worker_rank)

            if not self.pruned_rate_learning.workers[selected_client_id]['recv_time']:
                # comm_round==1
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                dist.send(torch.tensor(0.0, dtype=float), dst=worker_rank)
            else:
                # comm_round>1
                dist.send(tensor=self.pruned_rate_learning.workers[selected_client_id]['idx_len'], dst=worker_rank)
                buffer = torch.tensor(
                    flatten_model.buffer[self.pruned_rate_learning.workers[selected_client_id]['idx']], )
                dist.send(tensor=buffer, dst=worker_rank)
                pruned_rate = self.pruned_rate_learning.get_default_pruned_rate(selected_client_id,
                                                                                self.conf.random_select)
                dist.send(tensor=torch.tensor(pruned_rate, dtype=float),
                          dst=worker_rank)
                self.conf.logger.log(
                    f"worker-{worker_rank},clinet-{selected_client_id} pruned rate is {pruned_rate},retention ratio is {self.pruned_rate_learning.workers[selected_client_id]['retention_ratio'][-1].item()}")
                if len(self.pruned_rate_learning.workers[selected_client_id]['update_time']) == 1:
                    # comm_round==2
                    bn, _ = self.pruned_rate_learning.get_bn_importance_order(self.master_model)
                    bn_num = torch.tensor(len(bn), dtype=int)
                    dist.send(tensor=bn_num, dst=worker_rank)
                    dist.send(tensor=bn, dst=worker_rank)
                    # sorted_bn, sorted_idx = torch.sort(flatten_model.buffer)
                    # dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                    # dist.send(tensor=sorted_bn, dst=worker_rank)
                    # dist.send(tensor=sorted_idx, dst=worker_rank)

            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )
            if to_send_history:
                for i, model in enumerate(self.global_models_buffer):
                    flatten_model = TensorBuffer(list(model.state_dict().values()))
                    dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                    self.conf.logger.log(
                        f"\tMaster send the history model id:{i} to process_id={worker_rank}."
                    )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb
        buffer_len = len(client_tb.buffer)
        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            self.pruned_rate_learning.workers[client_id]['world_id']=world_id
            self.pruned_rate_learning.workers[client_id]['idx_len'] = torch.tensor(0)
            reqs.append(dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['idx_len'], src=world_id))

            self.pruned_rate_learning.workers[client_id]['buffer'] = torch.zeros(buffer_len)
            reqs.append(dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['buffer'], src=world_id))

            self.pruned_rate_learning.workers[client_id]['idx'] = torch.zeros(buffer_len, dtype=int)
            reqs.append(dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['idx'], src=world_id))

            self.pruned_rate_learning.workers[client_id]['recv_time'] = torch.tensor(time.time(), dtype=float)
            reqs.append(dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['recv_time'], src=world_id))

            self.pruned_rate_learning.workers[client_id]['retention_ratio'].append(torch.tensor(0.0, dtype=float))
            reqs.append(
                dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['retention_ratio'][-1], src=world_id))

            self.pruned_rate_learning.workers[client_id]['retention_number'] = torch.tensor(0, dtype=int)
            reqs.append(
                dist.irecv(tensor=self.pruned_rate_learning.workers[client_id]['retention_number'], src=world_id))

        for req in reqs:
            req.wait()
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            # print(self.pruned_rate_learning.workers[client_id]['idx_len'])
            # print(self.pruned_rate_learning.workers[client_id]['buffer'])
            # print(self.pruned_rate_learning.workers[client_id]['idx'])
            # print(self.pruned_rate_learning.workers[client_id]['recv_time'])
            idx_len = self.pruned_rate_learning.workers[client_id]['idx_len']
            idx = self.pruned_rate_learning.workers[client_id]['idx'][:idx_len]
            self.pruned_rate_learning.workers[client_id]['idx']=idx
            buffer = self.pruned_rate_learning.workers[client_id]['buffer'][:idx_len]
            flatten_local_models[client_id].buffer[idx] = buffer
            # self.pruned_rate_learning.workers[client_id]['retention_ratio'].append(idx_len/buffer_len)
            self.pruned_rate_learning.workers[client_id]['update_time'].append(
                self.pruned_rate_learning.workers[client_id]['recv_time'] -
                self.pruned_rate_learning.workers[client_id]['send_time'])
        self.pruned_rate_learning.comm_round+=1
        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    def _receive_label_counts_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local label counts.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        label_counts = dict()
        for selected_client_id in selected_client_ids:
            label_count = torch.zeros(self.conf.num_classes)
            label_counts[selected_client_id] = label_count

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=label_counts[client_id], src=world_id
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local label counts.")

        MIN_SAMPLES_PER_LABEL = 1
        label_weights = []
        qualified_labels = []

        for label in range(self.conf.num_classes):
            weights = []
            for user in selected_client_ids:
                weights.append(label_counts[user][label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))  # obtain p(y)
        label_weights = np.array(label_weights).reshape((self.conf.num_classes, -1))
        print(label_weights)
        return label_weights, qualified_labels

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
                selected_client_ids=None
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        # uniformly averaged the model before the potential aggregation scheme.
        same_arch = (
                len(self.client_models) == 1
                and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        )

        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        if same_arch:
            fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        #            then self.aggregator.aggregate_fn = None.
        if self.aggregator.aggregate_fn is not None:
            # evaluate the uniformly averaged model.
            if fedavg_model is not None:
                performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator,
                    fedavg_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedag_test_loader",
                )
            else:
                assert "knowledge_transfer" in self.conf.fl_aggregate["scheme"]

                performance = None
                for _arch, _fedavg_model in fedavg_models.items():
                    master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator,
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedag_test_loader_{_arch}",
                    )

            # aggregate the local models.
            client_models = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                fedavg_model=fedavg_model,
                fedavg_models=fedavg_models,
                flatten_local_models=flatten_local_models,
                performance=performance,
                selected_client_ids=selected_client_ids
            )
            # here the 'client_models' are updated in-place.
            if same_arch:
                # here the 'master_model' is updated in-place only for 'same_arch is True'.
                self.master_model.load_state_dict(
                    list(client_models.values())[0].state_dict()
                )
            for arch, _client_model in client_models.items():
                self.client_models[arch].load_state_dict(_client_model.state_dict())
        else:
            # update self.master_model in place.
            if same_arch:
                self.master_model.load_state_dict(fedavg_model.state_dict())

            # update self.client_models in place.
            for arch, _fedavg_model in fedavg_models.items():
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())

        # evaluate the aggregated model on the test data.
        if same_arch:
            performance = master_utils.do_validation(
                self.conf,
                self.coordinator,
                self.master_model,
                self.criterion,
                self.metrics,
                self.test_loaders,
                label=f"aggregated_test_loader",
            )

        else:
            for arch, _client_model in self.client_models.items():
                master_utils.do_validation(
                    self.conf,
                    self.coordinator,
                    _client_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"aggregated_test_loader_{arch}",
                )
        torch.cuda.empty_cache()

        return performance.dictionary['top1']

    def train_generator(self, selected_client_ids, label_weights, qualified_labels, flatten_local_models, epochs=10,
                        verbose=True):
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        n_teacher_iters = 5
        ensemble_eta = 1

        local_models = {}
        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            _model = copy.deepcopy(self.client_models[_arch]).to(self.device)
            _model_state_dict = self.client_models[_arch].state_dict()
            flatten_local_model.unpack(_model_state_dict.values())
            _model.load_state_dict(_model_state_dict)
            local_models[client_idx] = _model

        import torch.nn.functional as F
        self.generative_model.train()
        self.generative_model = self.generative_model.cuda()
        for _ in range(epochs):
            for _ in range(n_teacher_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(qualified_labels, self.conf.batch_size)
                y_input = torch.tensor(y).to(self.device)
                ## feed to generator
                gen_result = self.generative_model(y_input, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps = gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss = 0
                for user_idx, user in enumerate(selected_client_ids):
                    user_model = local_models[user]
                    user_model.eval()
                    weight = label_weights[y][:, user_idx].reshape(-1, 1)

                    _, user_result_given_gen = user_model(gen_output, start_layer_idx=-1)

                    user_output_logp_ = F.log_softmax(user_result_given_gen, dim=1)
                    teacher_loss_ = torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss += teacher_loss_

                loss = teacher_loss + ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += teacher_loss
                DIVERSITY_LOSS += ensemble_eta * diversity_loss

        info = "Generator: Teacher Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS / (n_teacher_iters * epochs), DIVERSITY_LOSS / (n_teacher_iters * epochs))
        if verbose:
            self.conf.logger.log(info)
        self.generative_lr_scheduler.step()
        self.generative_model = self.generative_model.cpu()

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
