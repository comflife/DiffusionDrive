#!/bin/bash
# GRPO fine-tuning from the trained baseline AR checkpoint with temperature=0.8

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

BASE_AR_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt"

if [ ! -f "$BASE_AR_CKPT" ]; then
    echo "ERROR: Checkpoint not found at $BASE_AR_CKPT"
    exit 1
fi

echo "=================================================="
echo "DiffusionDrive-AR GRPO Fine-tuning from baseline AR"
echo "=================================================="
echo "Base checkpoint: $BASE_AR_CKPT"
echo "Group size     : 16"
echo "Temperature    : 0.8"
echo "KL coef        : 0.1"
echo "LR             : 1e-6"
echo "Output         : /data2/byounggun/diffusiondrive_grpo_output_base_t08"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

SAFE_CKPT="/tmp/grpo_base_ar_24l0pgz4.ckpt"
rm -f "$SAFE_CKPT"
ln -s "$BASE_AR_CKPT" "$SAFE_CKPT"

python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtest \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_base_t08 \
    ++experiment_name=diffusiondrive_ar_grpo_base_t08 \
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
    ++temperature=0.8 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_base_24l0pgz4_g16_t0.8"

echo "=================================================="
echo "GRPO Training Complete!"
echo "=================================================="
