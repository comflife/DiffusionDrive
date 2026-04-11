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
echo "DiffusionDrive-AR GRPO Fine-tuning"
echo "=================================================="
echo "Pretrained checkpoint: $PRETRAINED_CKPT (epoch 27)"
echo "Codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"
echo "Output: /data2/byounggun/diffusiondrive_grpo_output"
echo "=================================================="
echo ""
echo "GRPO Settings:"
echo "  - Group size: 4 (rollouts per scene)"
echo "  - KL coef: 0.1"
echo "  - Temperature: 0.5"
echo "  - Learning rate: 1e-5"
echo "  - Reward: PDM Score"
echo "  - Wandb: enabled"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create symlink for checkpoint
SAFE_CKPT="/tmp/grpo_pretrained.ckpt"
if [ -L "$SAFE_CKPT" ]; then
    rm "$SAFE_CKPT"
fi
ln -s "$PRETRAINED_CKPT" "$SAFE_CKPT"

# Run GRPO training
# NOTE: Using navtest split with test data (consistent with eval script)
python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtest \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output \
    ++experiment_name=diffusiondrive_ar_grpo \
    ++trainer.params.max_epochs=50 \
    ++trainer.params.devices=4 \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++batch_size=1 \
    ++num_workers=0 \
    ++group_size=4 \
    ++kl_coef=0.1 \
    ++lr=1e-5 \
    ++temperature=0.5 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_navtest_g4_stable"

echo "=================================================="
echo "GRPO Training Complete!"
echo "=================================================="
