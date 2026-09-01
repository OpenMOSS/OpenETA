#!/bin/bash
# Run FTP1 evaluation on UniVTAC tasks.
# Usage (from UniVTAC/):
#   bash scripts/shell/eval_ftp1.sh
#   bash scripts/shell/eval_ftp1.sh lift_bottle,insert_hole
#   TOTAL_NUM=5 WORKERS=2 bash scripts/shell/eval_ftp1.sh
#   MODEL_NAME=FTP1 bash scripts/shell/eval_ftp1.sh pull_out_key
#   LOW_MEMORY=1 bash scripts/shell/eval_ftp1.sh   # FTP1 on CPU to save GPU VRAM
#   NO_VIDEO=1 bash scripts/shell/eval_ftp1.sh     # Disable video when ffmpeg not installed

set -e
cd "$(dirname "$0")/../.."

# Optional label used as the result directory suffix. If set, output becomes timestamp_RUN_SUFFIX.
RUN_SUFFIX=${RUN_SUFFIX:-"UniVTAC_insert_HDMI_expert_gsmall_ftp1"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/UniVTAC_insert_HDMI_expert_gsmall_ftp1/29999"}
DOMAIN_NAME=${DOMAIN_NAME:-"UniVTAC_insert_HDMI"}
# TASK_LIST="lift_bottle,lift_can,insert_HDMI,insert_hole,insert_tube,pull_out_key,put_bottle_in_shelf,grasp_classify"
TASK_LIST=${TASK_LIST:-${1:-"insert_HDMI"}}
TASK_CONFIG=${TASK_CONFIG:-"contact.yml"}
TOTAL_NUM=${TOTAL_NUM:-100}
WORKERS=${WORKERS:-1}
GPU=${GPU:-""}
LOW_MEMORY=${LOW_MEMORY:-0}
NO_VIDEO=${NO_VIDEO:-0}
MODEL_NAME=${MODEL_NAME:-FTP1}
SAVE_ROOT=${SAVE_ROOT:-"eval_results/${MODEL_NAME}"}
# Max steps per episode (0 = use task default from task_config). Default 160.
MAX_STEPS=${MAX_STEPS:-1000}
# Dir to save one sample of model input (224x224 BGR). Default: <save_root>/infer_input_sample (set to empty to disable).
SAVE_INFER_INPUT_DIR=${SAVE_INFER_INPUT_DIR:-"${SAVE_ROOT}/infer_input_sample"}
# Action representation: auto/relative/absolute/mix. 'auto' infers from checkpoint train_config.json.
ACTION_REP=${ACTION_REP:-auto}
# Only use the first N executable prediction steps from each chunk (0 = use all).
CHUNK_FIRST_N=${CHUNK_FIRST_N:-20}     # 20, aligned with TactileACT
VIDEO_FREQ=${VIDEO_FREQ:-2}
SUMMARY_ONLY_RUN_ROOTS=${SUMMARY_ONLY_RUN_ROOTS:-""}
RESUME=${RESUME:-0}

echo "[eval_ftp1.sh] model_name=$MODEL_NAME save_root=$SAVE_ROOT checkpoint_dir=$CHECKPOINT_DIR domain=$DOMAIN_NAME task_list=$TASK_LIST total_num=$TOTAL_NUM workers=$WORKERS gpu=$GPU low_memory=$LOW_MEMORY no_video=$NO_VIDEO max_steps=$MAX_STEPS action_rep=$ACTION_REP chunk_first_n=$CHUNK_FIRST_N video_freq=$VIDEO_FREQ run_suffix=${RUN_SUFFIX:-'(none)'} save_infer_input_dir=${SAVE_INFER_INPUT_DIR:-'(none)'} summary_only_run_roots=${SUMMARY_ONLY_RUN_ROOTS:-'(none)'} resume=$RESUME"

cmd=(
  python scripts/eval_ftp1.py
  --task_config "$TASK_CONFIG"
  --action_rep "$ACTION_REP"
  --chunk_first_n "$CHUNK_FIRST_N"
)

if [[ -n "$SUMMARY_ONLY_RUN_ROOTS" ]]; then
  cmd+=(
    --summary_only_run_roots "$SUMMARY_ONLY_RUN_ROOTS"
  )
  [[ -n "$CHECKPOINT_DIR" ]] && cmd+=(--checkpoint_dir "$CHECKPOINT_DIR")
  [[ -n "$DOMAIN_NAME" ]] && cmd+=(--domain_name "$DOMAIN_NAME")
else
  cmd+=(
    --checkpoint_dir "$CHECKPOINT_DIR"
    --domain_name "$DOMAIN_NAME"
    --task_list "$TASK_LIST"
    --save_root "$SAVE_ROOT"
    --total_num "$TOTAL_NUM"
    --workers "$WORKERS"
    --gpu "$GPU"
    --run_suffix "$RUN_SUFFIX"
    --temporal_ensemble
    --video_frequency "$VIDEO_FREQ"
  )
  [[ -n "$SAVE_INFER_INPUT_DIR" ]] && cmd+=(--save_infer_input_dir "$SAVE_INFER_INPUT_DIR")
  [[ "$LOW_MEMORY" == "1" ]] && cmd+=(--low_memory)
  [[ "$NO_VIDEO" == "1" ]] && cmd+=(--no_video)
  [[ "$RESUME" == "1" ]] && cmd+=(--resume)
  # cmd+=(--max_steps "$MAX_STEPS")
fi

"${cmd[@]}"
