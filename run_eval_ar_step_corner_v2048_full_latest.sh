#!/bin/bash
# Evaluate the FULL AR model (per-layer ego cross-attn + deformable BEV).
# Default: full/checkpoints/last.ckpt — override AR_CKPT to pick another epoch.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export RAY_IGNORE_UNHANDLED_ERRORS=1

AR_CKPT="${AR_CKPT:-/data2/byounggun/diffusiondrive_ar_output/full/checkpoints/last.ckpt}"

if [ ! -f "$AR_CKPT" ]; then
    echo "Checkpoint not found: $AR_CKPT"
    echo "Available checkpoints:"
    ls /data2/byounggun/diffusiondrive_ar_output/full/checkpoints/ 2>/dev/null
    echo "Override with: AR_CKPT=/path/to/ckpt bash $0"
    exit 1
fi

# Hydra-safe checkpoint alias (avoid '=' grammar issues in epoch=NN filenames)
HYDRA_CKPT=/tmp/diffusiondrive_ar_step_corner_v2048_full_eval.ckpt
ln -sfn "$AR_CKPT" "$HYDRA_CKPT"

echo "=================================================="
echo "Evaluating DiffusionDrive-AR FULL (ego-per-layer + deformable BEV)"
echo "Source ckpt : $AR_CKPT"
echo "Hydra alias : $HYDRA_CKPT"
echo "GPU         : $CUDA_VISIBLE_DEVICES"
echo "Output      : /data2/byounggun/diffusiondrive_ar_output/eval_full_latest"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

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
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.freeze_pretrained_trunk=false \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_step_corner_v2048_full_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_full_latest
