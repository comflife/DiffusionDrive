#!/bin/bash
# Evaluate the most recent step-corner v2048 AR refine model using its last.ckpt.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

LATEST_CKPT="$(find /data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar -path '*/checkpoints/last.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
AR_CKPT="${AR_CKPT:-$LATEST_CKPT}"

if [ -z "$AR_CKPT" ] || [ ! -f "$AR_CKPT" ]; then
    echo "No refine checkpoint found."
    echo "Expected something like:"
    echo "/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/<run_id>/checkpoints/last.ckpt"
    exit 1
fi

# Hydra-safe alias because checkpoint filenames may contain '=' in some cases.
HYDRA_CKPT=/tmp/diffusiondrive_ar_step_corner_refine_latest_eval.ckpt
ln -sfn "$AR_CKPT" "$HYDRA_CKPT"

echo "=================================================="
echo "Evaluating latest DiffusionDrive-AR step-corner refine v2048"
echo "Latest checkpoint: $AR_CKPT"
echo "Hydra-safe checkpoint alias: $HYDRA_CKPT"
echo "Output: /data2/byounggun/diffusiondrive_ar_output/eval_step_corner_refine_latest"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$HYDRA_CKPT" \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=false \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=false \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_step_corner_refine_v2048_latest_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_step_corner_refine_latest
