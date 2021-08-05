import numpy as np
import pruned_rate_learning
class HyperparametricCalculation():
    @classmethod
    def get_heterogeneity(cls, ratio, worker_num):
        H=0
        for w in range(1,worker_num):
            H+=1/(1+((ratio-1)/(worker_num-1))*(worker_num-w))
        H=1-1/(worker_num-1)*H
        return H

    @classmethod
    def get_bandwidth(cls,train_time,model_size,max_bandwidth,ratio,worker_num):
        update_time=np.zeros(worker_num)
        for w in range(worker_num):
            update_time[w]=((2*model_size)/max_bandwidth + train_time)* (
                        1 + ((ratio - 1) / (worker_num - 1)) * (worker_num - w-1))
        bandwidth=2*model_size / (update_time-train_time)
        return bandwidth

if __name__ == '__main__':
    import torch.distributed as dist

    dist.init_process_group('nccl', init_method='file:///home/wnpan/LTT/python_project/AdaptCL/sharedfile',
                            world_size=3, rank=0)
    print('OK')
    print('isdf')
