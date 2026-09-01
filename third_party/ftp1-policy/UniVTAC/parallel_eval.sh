TASK_NAME=${1}
TASK_CONFIG=${2}
POLICY_CONFIG=${3}
GPU=${4:-0}
NUM_PROCESSES=${5:-2}
TOTAL_NUM=${6:-100}

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/parallel_eval_policy.py $TASK_NAME $TASK_CONFIG $POLICY_CONFIG \
    --total_num $TOTAL_NUM \
    --workers $NUM_PROCESSES