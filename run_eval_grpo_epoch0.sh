#!/bin/bash
# Eval script for GRPO Epoch 0 model

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Use GRPO Epoch 0 checkpoint
GRPO_CKPT="/data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/grpo-epoch=00.ckpt"

if [ ! -f "$GRPO_CKPT" ]; then
    echo "ERROR: GRPO checkpoint not found at $GRPO_CKPT"
    echo "Checking available checkpoints:"
    ls -la /data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/ 2>/dev/null || echo "Checkpoint directory does not exist yet"
    exit 1
fi

echo "=================================================="
echo "Evaluating GRPO Model (Epoch 0 - First)"
echo "=================================================="
echo "Checkpoint: $GRPO_CKPT"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Use 1 GPU for evaluation
export CUDA_VISIBLE_DEVICES=0

# CRITICAL: Prevent Ray from overriding CUDA_VISIBLE_DEVICES in worker processes.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Set Ray GPU resources (important for Ray workers to see the GPU)
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path='$GRPO_CKPT'" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_grpo_epoch0_eval \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_v2/eval_epoch0

echo "=================================================="
echo "Eval Complete! Check output_dir for results"
echo "=================================================="
