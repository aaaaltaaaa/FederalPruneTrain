python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data dbpedia --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 14\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.1 \
    --train_data_ratio 0.1 --nlp_input 1 --agg_data_ratio 0.97\
    --n_clients 20 --participation_ratio 0.2  --n_comm_rounds 10 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 5 --pn_normalize True --same_seed_process False

python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data dbpedia --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 14\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.1 \
    --train_data_ratio 0.1 --nlp_input 1 --agg_data_ratio 0.97\
    --n_clients 20 --participation_ratio 0.2  --n_comm_rounds 10 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False

python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data ag_news --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 4\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --contrastive 0\
    --train_data_ratio 0.5 --nlp_input 1 --agg_data_ratio 0.97\
    --n_clients 20 --participation_ratio 0.2  --n_comm_rounds 10 --local_n_epochs 1 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average,eval_ensemble=True,update_student_scheme=avg_logits,data_source=same,data_name=ag_news,total_n_server_pseudo_batches=5000,eval_batches_freq=20,early_stopping_server_batches=200 \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data sst --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 5\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.5 \
    --train_data_ratio 1 --nlp_input 1 --agg_data_ratio 0 --val_dataset 1 \
    --n_clients 10 --participation_ratio 0.4  --n_comm_rounds 10 --local_n_epochs 3 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 5 --pn_normalize True --same_seed_process False

python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data sst --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 5\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.5 \
    --train_data_ratio 1 --nlp_input 1 --agg_data_ratio 0 --val_dataset 1 \
    --n_clients 10 --participation_ratio 0.4  --n_comm_rounds 10 --local_n_epochs 3 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 6 --pn_normalize True --same_seed_process False

python  run.py \
    --arch distilbert --complex_arch master=distilbert,worker=distilbert --experiment feddistill \
    --data sst --pin_memory True --batch_size 64 --num_workers 6 --self_distillation 0.1 --global_logits 1 --num_classes 5\
    --partition_data non_iid_dirichlet --non_iid_alpha 0.5 \
    --train_data_ratio 1 --nlp_input 1 --agg_data_ratio 0 --val_dataset 1 \
    --n_clients 10 --participation_ratio 0.4  --n_comm_rounds 10 --local_n_epochs 3 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer 'adam' --lr 1e-5 --weight_decay 0 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
