#!/bin/bash
# Custom Multi-Scale RL Training for DiffusionDrive-AR
# Combines GSPO + Multi-Scale Advantage

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp

export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Find pretrained checkpoint
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
echo "DiffusionDrive-AR Custom Multi-Scale RL Training"
echo "=================================================="
echo "Pretrained: $PRETRAINED_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_custom_rl_output"
echo ""
echo "Key Features:"
echo "  1. GSPO: Sequence-level importance ratio (low variance)"
echo "  2. Multi-Scale Advantage:"
echo "     - Global (70%): PDMS score (trajectory quality)"
echo "     - Local (30%): Token-level smoothness/progress"
echo "  3. Combined Loss: Global context + Local precision"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create symlink
SAFE_CKPT="/tmp/custom_rl_pretrained.ckpt"
if [ -L "$SAFE_CKPT" ]; then
    rm "$SAFE_CKPT"
fi
ln -s "$PRETRAINED_CKPT" "$SAFE_CKPT"

# Run training
python3 -m navsim.agents.diffusiondrive.custom_rl_train \
    train_test_split=navtrain \
    checkpoint_path="$SAFE_CKPT" \
    metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs" \
    output_dir=/data2/byounggun/diffusiondrive_custom_rl_output \
    experiment_name=diffusiondrive_ar_custom_rl \
    trainer.max_epochs=30 \
    trainer.devices=4 \
    trainer.strategy=ddp_find_unused_parameters_true \
    batch_size=1 \
    group_size=8 \
    lr=5e-6 \
    kl_coef=0.005 \
    temperature=0.8 \
    global_advantage_weight=0.7 \
    local_advantage_weight=0.3 \
    local_window_size=3 \
    gspo_aggregation="mean" \
    use_token_level_kl=true

echo "=================================================="
echo "Custom RL Training Complete!"
echo "=================================================="
