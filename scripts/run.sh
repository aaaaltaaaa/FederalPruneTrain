#!/usr/bin/env bash

usage="args:[-m,-d,-a,-r,-g]"

model=""
dataset=""
alpha=""
ratio=""
gpu=0

# 执行 getopt 来获得具名参数
GETOPT_ARGS=`getopt -o m:d:a:r:g:h -- "$@"`
eval set -- "$GETOPT_ARGS"

# 获取参数
while [ -n "$1" ]
do
  case "$1" in
    -m) model=$2; shift 2;;
    -d) dataset=$2; shift 2;;
    -a) alpha=$2; shift 2;;
    -r) ratio=$2; shift 2;;
    -g) gpu=$2; shift 2;;
    -h) echo $usage; break ;;
    --) break ;;
  esac
done

# 检查参数是否为空
if [[ -z $model || -z $dataset || -z $alpha || -z $ratio ]]; then
  echo "You should give all the arguments:"
  echo "model=$model, dataset=$dataset, alpha=$alpha, ratio=$ratio"
  exit 0
fi

# 检查模型是否支持
suppoert_models=("FedAvg" "FedProx" "FedDF" "FedGen" "FedDistill" "FedGKD" "FedPRJ" "MOON")
if ! (echo "${suppoert_models[@]}" | grep -wi -q "$model"); then
  echo "Unsupported Model: $model"
  echo "We now support: ${suppoert_models[*]}"
  exit 0
fi

experiment="$model"_alpha"$alpha"_ratio"$ratio"

# FedProx 的特殊设置
local_prox_term=0
if echo "FedProx" | grep -wi -q "$model"; then
  if [ $dataset == "cifar10" ]; then
    local_prox_term=0.01
  elif [ $dataset == "cifar100" ]; then
    local_prox_term=0.001
  fi
fi

# FedDF 的特殊设置
fl_aggregate="scheme=federated_average"
group_norm_num_groups=16
if echo "FedDF" | grep -wi -q "$model"; then
  fl_aggregate="scheme=noise_knowledge_transfer,update_student_scheme=avg_logits,data_source=other,data_type=train,data_scheme=random_sampling,data_name=cifar100,data_percentage=1.0,total_n_server_pseudo_batches=10000,eval_batches_freq=100,early_stopping_server_batches=1000"
  group_norm_num_groups=0
fi

# FedGKD/FedDistill/MOON/FedPRJ 的特殊设置
self_distillation=0
if echo "FedDistill|FedGKD|FedPRJ" | grep -wi -q "$model"; then
  self_distillation=0.1
elif echo "MOON" | grep -wi -q "$model"; then
  self_distillation=0.1
fi

# FedGen 的特殊设置
generator=0
if echo "FedGen" | grep -wi -q "$model"; then
  generator=1
fi

# FedDistill 的特殊设置
global_logits=0
if echo "FedDistill" | grep -wi -q "$model"; then
  global_logits=1
fi

# FedGen/FedDistill 的特殊设置
num_classes=10
if [ $dataset == "cifar100" ]; then
  num_classes=100
elif [ $dataset == "ag_news"]; then
  num_classes=4
elif [ $dataset == "sst"]; then
  num_classes=5
elif [ $dataset$ == "dbpedia"]; then
  num_classes=14
fi


# MOON 的特殊设置
projection=0
contrastive=0
self_distillation_temperature=1
if echo "MOON" | grep -wi -q "$model"; then
  projection=1
  contrastive=1
  self_distillation_temperature=0.5
fi

# FedPRJ 的特殊设置
if echo "FedPRJ" | grep -wi -q "$model"; then
  projection=1
fi

export "CUDA_VISIBLE_DEVICES=$gpu"

time=$(date "+%Y%m%d%H%M%S")
result_file=gresults/"$time"_"$experiment"_"$dataset".txt
# 执行 3 次训练
for num in {1..3..1}
do
  echo ============= Round $num Start =============
  tmp_file=/tmp/"$experiment"_"$dataset"_round"$num".txt

  python -W ignore run.py \
      --arch resnet8 --complex_arch master=resnet8,worker=resnet8 \
      --pin_memory True --batch_size 64 --num_workers 2 --global_history 0 \
      --partition_data non_iid_dirichlet --train_data_ratio $train_ratio --agg_data_ratio $ \
      --n_clients 20 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
      --optimizer sgd --lr 0.05 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
      --lr_scheduler MultiStepLR --lr_decay 0.1 \
      --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
      --track_time True --display_tracked_time True --python_path ~/conda/envs/pytorch-py3.6/bin/python --hostfile hostfile \
      --manual_seed 7 --pn_normalize True --same_seed_process False \
      --experiment $experiment --data $dataset --non_iid_alpha $alpha --participation_ratio $ratio \
      --fl_aggregate $fl_aggregate \
      --group_norm_num_groups $group_norm_num_groups \
      --local_prox_term $local_prox_term \
      --self_distillation $self_distillation \
      --generator $generator --global_logits $global_logits --num_classes $num_classes \
      --projection $projection --contrastive $contrastive --self_distillation_temperature $self_distillation_temperature \
      1>$tmp_file \
      2>/dev/null

  echo ============= Round $num Result =============
  echo "============= Round $num Result =============" >> $result_file

  grep "performance" $tmp_file | tail -n 5
  grep "performance" $tmp_file | tail -n 5 >> $result_file
done
