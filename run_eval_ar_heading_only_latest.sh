#!/bin/bash
# Evaluate the latest heading-only discrete AR model.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

AR_CKPT="${AR_CKPT:-/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/kjcgfbxh/checkpoints/last.ckpt}"

if [ ! -f "$AR_CKPT" ]; then
    echo "No heading-only checkpoint found at: $AR_CKPT"
    echo "Set AR_CKPT in this script or pass it inline before running."
    exit 1
fi

echo "=================================================="
echo "Evaluating DiffusionDrive-AR heading-only ablation"
echo "Checkpoint: $AR_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_ar_output/eval_heading_only_latest"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$AR_CKPT" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_heading_only_latest_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_heading_only_latest
