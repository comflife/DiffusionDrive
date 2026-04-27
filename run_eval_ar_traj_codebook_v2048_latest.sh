#!/bin/bash
# Evaluate the latest pure trajectory-codebook discrete AR model.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

AR_CKPT="${AR_CKPT:-}"

if [ -z "$AR_CKPT" ] || [ ! -f "$AR_CKPT" ]; then
    echo "No trajectory-codebook checkpoint found at: $AR_CKPT"
    echo "Set AR_CKPT inline after training, for example:"
    echo "AR_CKPT=/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/<run_id>/checkpoints/last.ckpt bash run_eval_ar_traj_codebook_v2048_latest.sh"
    exit 1
fi

echo "=================================================="
echo "Evaluating DiffusionDrive-AR trajectory-codebook v2048"
echo "Checkpoint: $AR_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_ar_output/eval_traj_codebook_v2048_latest"
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
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=trajectory_corners \
    agent.config.ar_use_residual_delta=false \
    agent.config.ar_use_heading_head=false \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_traj_codebook_v2048_latest_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_traj_codebook_v2048_latest
