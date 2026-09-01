set -e

# 检查最后一个参数是否为 -e
last_arg="${@: -1}"
skip_to_eval=false
if [ "$last_arg" = "-e" ]; then
    skip_to_eval=true
    # 移除 -e 参数，避免影响其他参数
    set -- "${@:1:$(($#-1))}"
fi

task_name=${1}
task_config=${2}
gpu_id=${3}
train_config=${4:-"train_config"}
expert_data_num=${5:-50}
export TRAIN_CONFIG=$train_config

# 如果不是 -e 模式，执行完整流程
if [ "$skip_to_eval" = false ]; then
    if [ -d "./data/sim-$task_name/$task_config-$expert_data_num" ]; then
        echo "Processed data for $task_name already exists. Skipping data processing."
    else
        echo "Processing data for $task_name..."
        bash process_data.sh $task_name $task_config $expert_data_num
    fi
    bash train.sh $task_name $task_config $expert_data_num 0 $gpu_id $train_config
fi

# 执行评估部分

GREEN='\033[0;32m'
RESET='\033[0m'

echo -e "
${GREEN}=======================================
Task  Name  : $task_name
Run   Config: $task_config
Train Config: $train_config
GPU         : $gpu_id
=======================================${RESET}
"

cd ../../
# bash parallel_eval.sh $task_name $task_config Ablation/deploy $gpu_id
bash eval_policy.sh $task_name $task_config Ablation/deploy $gpu_id