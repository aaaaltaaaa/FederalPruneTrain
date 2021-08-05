import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):

    if rank == 0:
        tensor = torch.zeros(1)
        tensor2 = torch.zeros(1)
        tensor += 1
        tensor2 +=2
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        dist.send(tensor=tensor2, dst=1)
    elif rank==2:
        tensor = torch.zeros(1)
        tensor2 = torch.zeros(1)
        time.sleep(2)
        tensor += 3
        tensor2 += 4
        dist.send(tensor=tensor, dst=1)
        dist.send(tensor=tensor2, dst=1)
    else:
        # Receive tensor from process 0
        tensor = torch.zeros(2)
        tensor2 = torch.zeros(2)
        time.sleep(1)
        reqs=[]
        r=dist.irecv(tensor=tensor, src=0)
        reqs.append(r)
        reqs.append(dist.irecv(tensor=tensor2, src=0))
        for r in reqs:
            r.wait()
        print('Rank ', rank, ' has data ', tensor[0])
        print('Rank ', rank, ' has data ', tensor2[0])
        reqs = []
        reqs.append(dist.irecv(tensor=tensor, src=2))
        reqs.append(dist.irecv(tensor=tensor2, src=2))
        for r in reqs:
            r.wait()
        print('Rank ', rank, ' has data ', tensor[0])
        print('Rank ', rank, ' has data ', tensor2[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
