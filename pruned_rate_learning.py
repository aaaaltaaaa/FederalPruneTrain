import numpy as np
import torch
import torch.nn as nn
class PrunedRateLearning():
    def __init__(self, min_retention, min_pruned, max_pruned, init_cofe):
        self.workers={}
        self.min_retention=min_retention
        self.min_pruned=min_pruned
        self.max_pruned=max_pruned
        self.init_cofe=init_cofe
        self.comm_round=0
        self.default_pruned_rate_list=[[0.5, 0.3, 0.2, 0.3, 0.3, 0.2, 0.3, 0.2, 0.2, 0.0],
                                       [0.3, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 0.0],
                                       [0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.0, 0.1, 0.0],
                                       [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]]

    def get_bn_importance_order(self,model):
        total = 0

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index + size)] = m.weight.data.abs().clone()
                index += size
        return bn,torch.sort(bn)


    def get_pruned_rate(self, worker, min_update_time):
        self.comm_round+=1
        worker=self.workers[worker]
        if len(worker['update_time'])>1:
            target_retention=self.newton_interpolate(worker['update_time'],worker['retention_ratio'], min_update_time)
            if  abs(worker['retention_ratio'][-1] - max(target_retention,self.min_retention))>self.min_pruned:
                pruned_ratio= (worker['retention_ratio'][-1] - target_retention)/ worker[
                    'retention_ratio'][-1]
            else:
                pruned_ratio=0
        else:
            pruned_ratio=abs(worker['update_time'][-1]-min_update_time)/(self.init_cofe* worker['update_time'][-1])
        self.pruned_rate=torch.tensor(max(min(pruned_ratio, self.max_pruned),0))
        return self.pruned_rate

    def get_default_pruned_rate(self, worker_rank):
        if self.comm_round%10==0:
            pruned_count=self.comm_round//10
            if pruned_count<=4:
                return self.default_pruned_rate_list[pruned_count-1][worker_rank-1]
        return 0



    def get_mini_update_time(self,workers):
        mini_update_time=self.workers[workers[0]]['update_time'][-1]
        for worker in workers:
            if self.workers[worker]['update_time'][-1] < mini_update_time:
                mini_update_time= self.workers[worker]['update_time'][-1]
        return mini_update_time


    def newton_interpolate(self,x_list, y_list, x):

        def difference_quotient_list(y_list, x_list=[]):
            if x_list == []:
                x_list = [i for i in range(len(y_list))]
            prev_list = y_list
            dq_list = []
            dq_list.append(prev_list[0])
            for t in range(1, len(y_list)):
                prev, curr = 0, 0
                m = []
                k = -1
                for i in prev_list:
                    curr = i
                    m.append((curr - prev) / (x_list[k + t] - x_list[k]))
                    prev = i
                    k += 1
                m.pop(0)
                prev_list = m
                dq_list.append(prev_list[0])
            return dq_list

        coef = difference_quotient_list(y_list, x_list)
        p = coef[0]
        for i in range(1, len(coef)):
            product = 1
            for j in range(i):
                product *= (x - x_list[j])
            p += coef[i] * product
        return p

    def get_heterogeneity(self, ratio, worker_num):
        H = 0
        for w in range(1, worker_num):
            H += 1 / (1 + ((ratio - 1) / (worker_num - 1)) * (worker_num - w))
        H = 1 - 1 / (worker_num - 1) * H
        return H

    def get_bandwidth(self, train_time, model_size, max_bandwidth, ratio, worker_num):
        update_time = np.zeros(worker_num)
        for w in range(worker_num):
            update_time[w] = ((2 * model_size) / max_bandwidth + train_time) * (
                    1 + ((ratio - 1) / (worker_num - 1)) * (worker_num - w - 1))
        bandwidth = 2 * model_size / (update_time - train_time)
        return bandwidth
