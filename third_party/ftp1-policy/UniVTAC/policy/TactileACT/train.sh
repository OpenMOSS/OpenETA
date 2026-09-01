TASK_NAME=$1
CONFIG_NAME=$2
EXPERT_TRAJ_NUM=$3
GPU_ID=$4

SAVE_DIR=./data/$TASK_NAME-$CONFIG_NAME-$EXPERT_TRAJ_NUM
NAME=$TASK_NAME-$CONFIG_NAME

export CUDA_VISIBLE_DEVICES=$GPU_ID

CLIP_CKPT_PATH=clip_models/${TASK_NAME}-${CONFIG_NAME}-${EXPERT_TRAJ_NUM}/0/epoch_1499

python imitate_episodes.py --config config.json --save_dir $SAVE_DIR \
    --name $NAME --batch_size 64 --kl_weight 10 --z_dimension 32 \
    --num_epochs 4000 --dropout 0.025 --chunk_size 20 \
    --backbone clip_backbone \
    --gelsight_backbone_path ${CLIP_CKPT_PATH}_gelsight_encoder.pth \
    --vision_backbone_path ${CLIP_CKPT_PATH}_vision_encoder.pth