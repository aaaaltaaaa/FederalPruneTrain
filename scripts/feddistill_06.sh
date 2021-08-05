python  run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddistill_06  --group_norm_num_groups=16\
	        --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
		    --self_distillation 0.1 --global_history 0 --projection=0 --global_logits 1 --num_classes 10 --generator 0\
		        --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
			    --n_clients 20 --participation_ratio 0.6 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			        --fl_aggregate scheme=federated_average \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize True --same_seed_process False

python  run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddistill_06  --group_norm_num_groups=16\
	        --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
		    --self_distillation 0.1 --global_history 0 --projection=0 --global_logits 1 --num_classes 10 --generator 0\
		        --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
			    --n_clients 20 --participation_ratio 0.6 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			        --fl_aggregate scheme=federated_average \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize True --same_seed_process False

python  run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddistill_06  --group_norm_num_groups=16\
	        --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
		    --self_distillation 0.1 --global_history 0 --projection=0 --global_logits 1 --num_classes 10 --generator 0\
		        --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
			    --n_clients 20 --participation_ratio 0.6 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			        --fl_aggregate scheme=federated_average \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py$PYTHON_VERSION/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize True --same_seed_process False
