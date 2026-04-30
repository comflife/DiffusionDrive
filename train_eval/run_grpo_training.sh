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
echo "DiffusionDrive-AR GRPO Fine-tuning (v3 - eval-matched)"
echo "=================================================="
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""
echo "Key fixes from v1:"
echo "GRPO Settings v3:"
echo "  - Data   : navtest (matching metric cache)"
echo "  - Group  : 16 rollouts/scene  (more samples → better selection)"
echo "  - Temp   : 0.3                 (train/eval matched)"
echo "  - KL coef: 0.1                 (prevent drift from 56.4% baseline)"
echo "  - clip_eps: 0.2 (PPO)"
echo "  - LR     : 1e-6                (slower, more stable)"
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
#   train_test_split=navtest  (matching metric cache)
python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtest \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_v3 \
    ++experiment_name=diffusiondrive_ar_grpo_v3 \
    ++trainer.params.max_epochs=20 \
    ++trainer.params.devices=4 \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++trainer.params.gradient_clip_val=1.0 \
    ++batch_size=1 \
    ++num_workers=0 \
    ++group_size=16 \
    ++kl_coef=0.1 \
    ++clip_eps=0.2 \
    ++lr=1e-6 \
    ++temperature=0.3 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_v3_navtrain_g16_t0.3"

echo "=================================================="
echo "GRPO Training Complete!"
echo "=================================================="
