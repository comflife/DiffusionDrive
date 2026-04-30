#!/bin/bash
# Evaluate the GRPO checkpoint trained from the baseline AR model with temperature=0.8.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

GRPO_CKPT="/data2/byounggun/diffusiondrive_grpo_output_base_t08/checkpoints/last.ckpt"

if [ ! -f "$GRPO_CKPT" ]; then
    echo "ERROR: GRPO checkpoint not found at $GRPO_CKPT"
    exit 1
fi

echo "=================================================="
echo "Evaluating GRPO Model (base AR, temp=0.8)"
echo "Checkpoint: $GRPO_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_grpo_output_base_t08/eval_latest"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$GRPO_CKPT" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_grpo_base_t08_eval \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_base_t08/eval_latest

echo "=================================================="
echo "Eval Complete!"
echo "=================================================="
