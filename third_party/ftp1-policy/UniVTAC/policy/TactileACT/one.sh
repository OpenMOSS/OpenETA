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
expert_data_num=50
gpu_id=${3}

# 如果不是 -e 模式，执行完整流程
if [ "$skip_to_eval" = false ]; then
    if [ -d "./data/$task_name-$task_config-50" ]; then
        echo "Processed data for $task_name already exists. Skipping data processing."
    else
        echo "Processing data for $task_name..."
        bash process_data.sh $task_name $task_config $expert_data_num
    fi
    
    if [ -d "./clip_models/$task_name-$task_config-50" ]; then
        echo "CLIP model for $task_name already exists. Skipping CLIP pretraining."
    else
        echo "Pretraining CLIP model for $task_name..."
        export CUDA_VISIBLE_DEVICES=$gpu_id
        python clip_pretraining.py $task_name $task_config $expert_data_num
    fi

    bash train.sh $task_name $task_config $expert_data_num $gpu_id
fi

# 执行评估部分
cd ../../
bash eval_policy.sh $task_name $task_config TactileACT/deploy $gpu_id