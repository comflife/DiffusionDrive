#!/bin/bash
# Eval script for best trained model

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Use best checkpoint (last = epoch 27)
BEST_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/owb1ii1t/checkpoints/last.ckpt"

echo "=================================================="
echo "Evaluating Best Model (Epoch 27 - Final)"
echo "=================================================="
echo "Checkpoint: $BEST_CKPT"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Use 1 GPU for evaluation
export CUDA_VISIBLE_DEVICES=0

# CRITICAL: Prevent Ray from overriding CUDA_VISIBLE_DEVICES in worker processes.
# Without this, Ray sets CUDA_VISIBLE_DEVICES="" for tasks that don't explicitly
# request GPU resources (num_gpus=None in Task), making torch.cuda.is_available()=False.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Set Ray GPU resources (important for Ray workers to see the GPU)
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$BEST_CKPT" \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_epoch9_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_epoch9

echo "=================================================="
echo "Eval Complete! Check output_dir for results"
echo "=================================================="
