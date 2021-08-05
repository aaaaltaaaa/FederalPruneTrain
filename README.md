#### parameters
- self_distillation: 蒸馏的系数(default=0, global蒸馏设置为0.1,model-contrastive-learning设置为5/1)
- self_distillation_temperature(model-contrastive-learning为0.5，default=1)
- group_norm_num_groups: groupnorm每一组的channel数(默认为batchnorm, gn设置为16)
- contrastive: 设为1使用model-contrastive-learning
- projection: 设为1添加投影层
- global_history: 设为1蒸馏前2轮global模型
- local_n_epochs: 本地训练epochs

#### CIFAR-10 with ResNet-8 (homogeneous)
The setup of the FedAvg/FedProx for resnet-8 with cifar10:

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --group_norm_num_groups 16\
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the FedDF for resnet-8 with cifar10:

```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --group_norm_num_groups 16 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the global distillation for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --self_distillation 0.1 --group_norm_num_groups 16\
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the model-contrastive-learning for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo \
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --self_distillation 5 --self_distillation_temperature 0.5 --projection 1 --contrastive 1\
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.1 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.1 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 0 --use_nesterov False --momentum_factor 0 \
    --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```