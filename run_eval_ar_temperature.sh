#!/bin/bash
# Eval script for AR decoder (latest model) with non-deterministic temperature=0.8

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Use latest AR checkpoint (from your best run)
AR_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/owb1ii1t/checkpoints/last.ckpt"

echo "=================================================="
echo "Evaluating AR Decoder Latest Model with Temperature=0.8 (non-deterministic)"
echo "=================================================="
echo "Checkpoint: $AR_CKPT"
echo "Temperature: 0.8 (for diversity in multinomial sampling)"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Use 1 GPU for evaluation
export CUDA_VISIBLE_DEVICES=0

# Prevent Ray CUDA issues
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$AR_CKPT" \
    ++agent.config.temperature=0.8 \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_latest_temp08_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_latest_temp08

echo "=================================================="
echo "Eval Complete! Check output_dir for results (PDM scores with stochastic sampling)"
echo "=================================================="
