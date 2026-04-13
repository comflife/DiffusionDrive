#!/bin/bash
# Eval script for GRPO with checkpoint conversion

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

GRPO_CKPT="/data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/last.ckpt"
CONVERTED_CKPT="/data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/last_converted_ar.ckpt"

if [ ! -f "$GRPO_CKPT" ]; then
    echo "ERROR: GRPO checkpoint not found at $GRPO_CKPT"
    exit 1
fi

echo "=================================================="
echo "Converting GRPO checkpoint to AR-compatible..."
echo "=================================================="
python3 convert_grpo_ckpt_to_ar.py --input "$GRPO_CKPT" --output "$CONVERTED_CKPT"

echo ""
echo "=================================================="
echo "Evaluating converted GRPO checkpoint"
echo "=================================================="

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path='$CONVERTED_CKPT'" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_grpo_converted_eval \
    output_dir=/data2/byounggun/diffusiondrive_grpo_output_v2/eval_converted

echo "=================================================="
echo "Eval Complete!"
echo "=================================================="
