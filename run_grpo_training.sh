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

# Find pretrained checkpoint (epoch 04 or last)
CHECKPOINT_DIR="/data2/byounggun/diffusiondrive_ar_output"
PRETRAINED_CKPT=$(ls -t $CHECKPOINT_DIR/diffusiondrive-ar/*/checkpoints/epoch=04*.ckpt 2>/dev/null | head -1)
if [ -z "$PRETRAINED_CKPT" ]; then
    PRETRAINED_CKPT=$(ls -t $CHECKPOINT_DIR/diffusiondrive-ar/*/checkpoints/last.ckpt 2>/dev/null | head -1)
fi

if [ -z "$PRETRAINED_CKPT" ]; then
    echo "ERROR: No pretrained checkpoint found"
    exit 1
fi

echo "=================================================="
echo "DiffusionDrive-AR GRPO Fine-tuning"
echo "=================================================="
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo "Codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"
echo "Output: /data2/byounggun/diffusiondrive_grpo_output"
echo "=================================================="
echo ""
echo "GRPO Settings:"
echo "  - Group size: 8 (rollouts per scene)"
echo "  - KL coef: 0.01"
echo "  - Learning rate: 1e-5"
echo "  - Reward: PDM Score"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create symlink for checkpoint
SAFE_CKPT="/tmp/grpo_pretrained.ckpt"
if [ -L "$SAFE_CKPT" ]; then
    rm "$SAFE_CKPT"
fi
ln -s "$PRETRAINED_CKPT" "$SAFE_CKPT"

# Run GRPO training
python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtrain \
    checkpoint_path="$SAFE_CKPT" \
    metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs" \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output \
    experiment_name=diffusiondrive_ar_grpo \
    trainer.max_epochs=50 \
    trainer.devices=4 \
    trainer.strategy=ddp_find_unused_parameters_true \
    batch_size=1 \
    group_size=8 \
    kl_coef=0.01 \
    lr=1e-5 \
    temperature=1.0

echo "=================================================="
echo "GRPO Training Complete!"
echo "=================================================="
