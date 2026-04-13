#!/bin/bash
# GRPO Fine-tuning Script for DiffusionDrive-AR
# Uses PDM Score as reward for RL fine-tuning

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp

# Ensure DiffusionDrive navsim is used
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Use best checkpoint from AR training for GRPO fine-tuning
PRETRAINED_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/owb1ii1t/checkpoints/epoch=27-val_loss=0.00.ckpt"

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Checkpoint not found at $PRETRAINED_CKPT"
    exit 1
fi

echo "=================================================="
echo "DiffusionDrive-AR GRPO Fine-tuning (v2 fixed)"
echo "=================================================="
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""
echo "Key fixes from v1:"
echo "  [Bug1] loss sum→mean (was causing 200↔-40 explosion)"
echo "  [Bug2] teacher-forced log prob (correct AR conditioning)"
echo "  [Bug3] dim handling for ego_tokens [B,M,T]"
echo "  [Bug4] PPO importance ratio + clipping"
echo "  [Bug5] std clamp + advantage clipping"
echo ""
echo "GRPO Settings v2:"
echo "  - Data   : navtrain (harder/more diverse than navtest)"
echo "  - Group  : 8 rollouts/scene  (was 4 → too few for variance)"
echo "  - Temp   : 1.0               (was 0.5 → too concentrated)"
echo "  - KL coef: 0.01              (was 0.1 → was too restrictive)"
echo "  - clip_eps: 0.2 (PPO)"
echo "  - LR     : 5e-6"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create symlink for checkpoint
SAFE_CKPT="/tmp/grpo_pretrained.ckpt"
if [ -L "$SAFE_CKPT" ]; then
    rm "$SAFE_CKPT"
fi
ln -s "$PRETRAINED_CKPT" "$SAFE_CKPT"

# Run GRPO training
# IMPORTANT changes vs v1:
#   train_test_split=navtrain  (was navtest — test data is too easy → all rewards ~1.0)
#   group_size=8               (was 4 → need more rollouts to get reward variance)
#   temperature=1.0            (was 0.5 → too peaked, all rollouts similar)
#   kl_coef=0.01               (was 0.1 → too high, blocks updates even with advantages)
#   ++clip_eps=0.2             (new PPO clipping parameter)
python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtrain \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_v2 \
    ++experiment_name=diffusiondrive_ar_grpo_v2 \
    ++trainer.params.max_epochs=20 \
    ++trainer.params.devices=4 \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++trainer.params.gradient_clip_val=1.0 \
    ++batch_size=1 \
    ++num_workers=0 \
    ++group_size=8 \
    ++kl_coef=0.01 \
    ++clip_eps=0.2 \
    ++lr=5e-6 \
    ++temperature=1.0 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_v2_navtrain_g8_t1.0"

echo "=================================================="
echo "GRPO Training Complete!"
echo "=================================================="
