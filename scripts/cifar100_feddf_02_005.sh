$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddf_02_005 \
	        --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
		    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
		        --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./imagenet32,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
			        --img_resolution 32 \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize False --same_seed_process False

$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddf_02_005 \
	        --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
		    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
		        --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./imagenet32,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
			        --img_resolution 32 \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize False --same_seed_process False

$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddf_02_005 \
	        --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
		    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
		        --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./imagenet32,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
			        --img_resolution 32 \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize False --same_seed_process False

$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddf_02_005 \
	        --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
		    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
		        --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./imagenet32,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
			        --img_resolution 32 \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize False --same_seed_process False

$HOME/conda/envs/pytorch-py3.6/bin/python run.py \
	    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment feddf_02_005 \
	        --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 \
		    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
		        --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
			    --fl_aggregate scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=class_selection,data_percentage=1.0,num_total_class=100,num_overlap_class=0,data_name=imagenet32,data_dir=./imagenet32,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000 \
			        --img_resolution 32 \
				    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
				        --lr_scheduler MultiStepLR --lr_decay 0.1 \
					    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
					        --track_time True --display_tracked_time True --python_path $HOME/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
						    --manual_seed 7 --pn_normalize False --same_seed_process False
