#!/bin/bash
# Eval script for GRPO Latest checkpoint (direct loading, no conversion needed)

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Auto-detect latest GRPO checkpoint (v3 or v2)
GRPO_CKPT="/data2/byounggun/diffusiondrive_grpo_output_v3/checkpoints/last.ckpt"
if [ ! -f "$GRPO_CKPT" ]; then
    GRPO_CKPT="/data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/last.ckpt"
fi

if [ ! -f "$GRPO_CKPT" ]; then
    echo "ERROR: No GRPO checkpoint found"
    exit 1
fi

echo "=================================================="
echo "Evaluating GRPO Model (Latest)"
echo "=================================================="
echo "Checkpoint: $GRPO_CKPT"
echo "Modified: $(stat -c %y "$GRPO_CKPT" 2>/dev/null)"
echo "=================================================="

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path='$GRPO_CKPT'" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_grpo_latest_eval \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_v3/eval_latest

echo "=================================================="
echo "Eval Complete!"
echo "=================================================="
