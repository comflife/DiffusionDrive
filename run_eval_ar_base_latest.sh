#!/bin/bash
# Evaluate the latest baseline discrete AR model.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

AR_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt"

echo "=================================================="
echo "Evaluating Baseline DiffusionDrive-AR"
echo "Checkpoint: $AR_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_ar_output/eval_base_latest"
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
    experiment_name=diffusiondrive_ar_base_latest_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_base_latest
