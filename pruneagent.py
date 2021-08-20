import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from pcode.utils.tensor_buffer import TensorBuffer


class PruneAgent():
    def __init__(self, num_classes, device, optimizer=False, minimal_ratio=0.1):
        self.num_classes = num_classes
        self.device = device
        self.optimizer = optimizer
        self.minimal_filter = []
        self.minimal_ratio=minimal_ratio

    def change_fc(self, model):
        model.add_module('fc', nn.Linear(model.fc.in_features, self.num_classes).to(self.device))

    def get_original_filters_number(self, model):
        original_filters_number = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                original_filters_number += m.weight.shape[0]
        self.original_filters_number = original_filters_number
        self.retention_number = original_filters_number

    def get_score(self, model):
        self.status = {}
        self.score = {}
        for layers_name, layers in model.named_children():
            if 'layer' in layers_name:
                masks = []
                if layers[-1].__class__.__name__ == 'Bottleneck':
                    score = torch.zeros(layers[-1].bn3.weight.shape[0]).to(self.device)
                    if layers == model.layer1:
                        self.status[model.bn1] = model.bn1.weight.abs()
                    for block in layers:
                        if block.downsample:
                            masks.append(block.downsample._modules['1'])
                            score += block.downsample._modules['1'].weight.abs()
                        masks.append(block.bn3)
                        score += block.bn3.weight.abs()
                        self.status[block.bn1] = block.bn1.weight.abs()
                        self.status[block.bn2] = block.bn2.weight.abs()

                elif layers[-1].__class__.__name__ == 'BasicBlock':
                    # score = torch.zeros(layers[-1].bn2.weight.shape[0]).to(self.device)
                    score = []
                    if layers == model.layer1:
                        masks.append(model.bn1)
                        # score += model.bn1.weight.abs()
                        score.append(model.bn1.weight.abs())
                    for block in layers:
                        if block.downsample:
                            masks.append(block.downsample._modules['1'])
                            # score += block.downsample._modules['1'].weight.abs()
                            score.append(block.downsample._modules['1'].weight.abs())
                        masks.append(block.bn2)
                        # score += block.bn2.weight.abs()
                        score.append(block.bn2.weight.abs())
                        self.status[block.bn1] = block.bn1.weight.abs()
                score = torch.vstack(score).max(0).values
                for bn in masks:
                    self.status[bn] = score

        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                self.score[name] = self.status[m]
                self.minimal_filter.append(int(self.minimal_ratio*len(self.score[name])))

    def prune(self, model, optimizer, num):
        # self.get_score(model)
        filtered_score_list = []
        for i, score in enumerate(self.score.values()):
            filtered_score_list.append(score.cpu().detach().numpy()[:-self.minimal_filter[i]])
        scores = np.concatenate(filtered_score_list)

        threshold = np.sort(scores)[self.original_filters_number - self.retention_number + num]
        to_prune = int((scores < threshold).sum())
        self.retention_number = self.original_filters_number - to_prune
        self.retention_ratio = self.retention_number / self.original_filters_number
        dense_chs = self.get_dense_chs(model, threshold)
        self.get_dense_model(dense_chs, model, optimizer)
        if self.optimizer:
            new_optimizer = optim.SGD(model.parameters(), lr=optimizer.defaults['lr'],
                                      momentum=optimizer.defaults['momentum'],
                                      weight_decay=optimizer.defaults['weight_decay'])
            for param in model.parameters():
                new_optimizer.state[param]['momentum_buffer'] = nn.Parameter(
                    optimizer.state[param]['momentum_buffer']).to(self.device)
            return new_optimizer

    def sparse_bn(self, dense_chs, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                with torch.no_grad():
                    d_chs = dense_chs[name + '.weight']
                    chs = torch.zeros(m.weight.shape[0]).to(self.device)
                    chs[d_chs] = 1
                    chs = chs
                    m.weight *= chs
                    m.bias *= chs

    def get_dense_model(self, dense_chs, model, optimizer):
        model.add_module('conv1', self.dense_model(model.conv1, optimizer, dense_chs['conv1.weight'], 'conv1'))
        model.add_module('bn1', self.dense_model(model.bn1, optimizer, dense_chs['bn1.weight'], 'bn1'))
        for layer_name, layer in model.named_children():
            if 'layer' in layer_name:
                for i, block in enumerate(layer):
                    for m_name, m in block.named_children():
                        if isinstance(m, nn.Sequential):
                            for mm_name, mm in m.named_children():
                                name = layer_name + '.' + str(i) + '.' + m_name + '.' + mm_name
                                m.add_module(mm_name,
                                             self.dense_model(mm, optimizer, dense_chs[name + '.weight'], name))
                        elif 'conv' in m_name or 'bn' in m_name:
                            name = layer_name + '.' + str(i) + '.' + m_name
                            block.add_module(m_name, self.dense_model(m, optimizer, dense_chs[name + '.weight'], name))
        model.add_module('classifier',
                         self.dense_model(model.classifier, optimizer, dense_chs['classifier.weight'], 'classifier'))


    def dense_model(self, model, optimizer, dense_chs, name='unknow'):
        dim = model.weight.dim()
        if dim == 4:

            in_chs = dense_chs['in_chs']
            out_chs = dense_chs['out_chs']
            num_in, num_out = len(in_chs), len(out_chs)

            new_model = nn.Conv2d(num_in, num_out, model.kernel_size, model.stride,
                                  model.padding, bias=False).to(self.device)
            new_model.weight.data = model.weight.data[out_chs][:, in_chs, :, :]
            if self.optimizer:
                optimizer.state[new_model.weight]['momentum_buffer'] = Parameter(
                    optimizer.state[model.weight]['momentum_buffer'][out_chs][:, in_chs, :, :]).to(self.device)

            # print("[{}]: {} >> {}".format(name, list(model.weight.shape), list(new_model.weight.shape)))

        # Generate a new dense tensor and replace (FC layer)
        elif dim == 2:

            num_in, num_out = len(dense_chs), self.num_classes
            new_model = nn.Linear(num_in, num_out).to(self.device)
            new_model.weight.data = model.weight[:, dense_chs]
            new_model.bias.data = model.bias
            if self.optimizer:
                optimizer.state[new_model.weight]['momentum_buffer'] = Parameter(
                    optimizer.state[model.weight]['momentum_buffer'][:, dense_chs]).to(self.device)
                optimizer.state[new_model.bias]['momentum_buffer'] = Parameter(
                    optimizer.state[model.bias]['momentum_buffer']).to(self.device)

            # self.status[new_model.weight] = self.status[model.weight][dense_chs]
            # del self.status[model.weight]
            # self.bn_idx_list[new_model.weight] = self.bn_idx_list[model.weight][dense_chs]
            # del self.bn_idx_list[model.weight]
            # print("[{}]: {} >> {}".format(name, list(model.weight.shape), list(new_model.weight.shape)))

        # Change parameters of non-neural computing layers (BN, biases)
        else:
            new_model = nn.BatchNorm2d(len(dense_chs)).to(self.device)
            new_model.weight.data = model.weight[dense_chs]
            new_model.bias.data = model.bias[dense_chs]
            new_model.running_mean.data = model.running_mean[dense_chs]
            new_model.running_var.data = model.running_var[dense_chs]
            if self.optimizer:
                optimizer.state[new_model.weight]['momentum_buffer'] = Parameter(
                    optimizer.state[model.weight]['momentum_buffer'][
                        dense_chs]).to(self.device)
                optimizer.state[new_model.bias]['momentum_buffer'] = Parameter(
                    optimizer.state[model.bias]['momentum_buffer'][
                        dense_chs]).to(self.device)

        return new_model.to(self.device)

    def get_dense_chs(self, model, threshold):
        dense_chs = {}
        n=0
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                dense_chs[name + '.weight'] = self.get_dense_bn(m, name, threshold,self.minimal_filter[n])
                n+=1

        for name, param in model.named_parameters():
            if (('conv' in name) or ('downsample.0' in name)) and ('weight' in name):

                rear_bn = name.replace('conv', 'bn').replace('downsample.0', 'downsample.1')

                if name == 'conv1.weight':
                    dense_chs[name] = {'in_chs': np.arange(param.shape[1]).tolist()}
                elif 'downsample.0' in name:
                    dense_chs[name] = {'in_chs': dense_chs[name.replace('downsample.0', 'conv1')]['in_chs']}
                else:
                    dense_chs[name] = {'in_chs': dense_chs[prior_bn]}
                dense_chs[name].update({'out_chs': dense_chs[rear_bn]})
                prior_bn = rear_bn

        dense_chs['classifier.weight'] = list(dense_chs.values())[-1]['out_chs']

        relative_dense_chs = {}
        for name, chs in dense_chs.items():
            chs_idx = self.dense_chs_idx[name]
            if isinstance(chs, list):
                relative_dense_chs[name] = np.where(np.in1d(chs_idx, chs))[0]
                self.dense_chs_idx[name] = np.array(chs)
            else:
                relative_dense_chs[name] = {'in_chs': np.where(np.in1d(chs_idx['in_chs'], chs['in_chs']))[0],
                                            'out_chs': np.where(np.in1d(chs_idx['out_chs'], chs['out_chs']))[0]}

                self.dense_chs_idx[name] = {'in_chs': np.array(chs['in_chs']), 'out_chs': np.array(chs['out_chs'])}
        self.dense_chs = relative_dense_chs
        return relative_dense_chs

    def get_dense_bn(self, m, name, threshold,minimal_filter):
        filter = self.score[name] >= threshold
        _, idex = self.score[name].topk(minimal_filter)
        filter[idex] = True
        chs = np.squeeze(np.argwhere(np.asarray((filter).cpu().detach().numpy())))
        if chs.size == 1:
            chs = np.resize(chs, (1,))
        return chs.tolist()

    def get_idx(self):
        with torch.no_grad():
            for name, m in self.master_model.named_modules():
                name += '.weight'
                if 'conv' in name or 'downsample.0' in name:
                    param = torch.zeros_like(m.weight, dtype=bool)
                    out_chs = self.dense_chs_idx[name]['out_chs']
                    in_chs = self.dense_chs_idx[name]['in_chs']
                    for i in in_chs:
                        for o in out_chs:
                            param[o, i, :, :] = True
                    m.weight.data = param
                elif 'classifier' in name:
                    param = torch.zeros_like(m.weight, dtype=bool)
                    param[:, self.dense_chs_idx[name]] = True
                    m.weight.data = param
                    m.bias.data = torch.ones_like(m.bias, dtype=bool)
                elif 'bn' in name or 'downsample.1' in name:
                    param = torch.zeros(m.weight.shape[0])
                    param[self.dense_chs_idx[name]] = True
                    m.weight.data = param
                    param = torch.zeros(m.weight.shape[0])
                    param[self.dense_chs_idx[name]] = True
                    m.bias.data = param
                    param = torch.zeros(m.weight.shape[0])
                    param[self.dense_chs_idx[name]] = True
                    m.running_mean.data = param
                    param = torch.zeros(m.weight.shape[0])
                    param[self.dense_chs_idx[name]] = True
                    m.running_var.data = param

        flatten_model = TensorBuffer(list(self.master_model.state_dict().values()))
        self.idx = torch.tensor(np.squeeze(np.argwhere(np.asarray(flatten_model.buffer.cpu().numpy()))), dtype=int)


if __name__ == '__main__':
    import torchvision.models as models
    import pruned_rate_learning

    model = models.resnet18()
    pruneagent = PruneAgent(model, 10)
    prune_rate = pruned_rate_learning.PrunedRateLearning(0.1, 0.02, 0.5, 2)
