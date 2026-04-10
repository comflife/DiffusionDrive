#!/bin/bash
# DiffusionDrive-AR Evaluation Script
# Run PDMS scoring on trained AR model

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps

# Ensure DiffusionDrive navsim is used, not VAD
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Find latest checkpoint (use epoch 04 instead of last.ckpt to avoid corruption)
CHECKPOINT_DIR="/data2/byounggun/diffusiondrive_ar_output"
# Try epoch 04 first, then fallback to last.ckpt
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/diffusiondrive-ar/*/checkpoints/epoch=04*.ckpt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/diffusiondrive-ar/*/checkpoints/last.ckpt 2>/dev/null | head -1)
fi

# If no checkpoint found, exit
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $CHECKPOINT_DIR"
    echo "Please check if training has saved checkpoints."
    exit 1
fi

echo "=================================================="
echo "DiffusionDrive-AR PDMS Evaluation"
echo "=================================================="
echo "Checkpoint: $LATEST_CHECKPOINT"
echo "Codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"
echo "Output: /data2/byounggun/diffusiondrive_ar_output/eval_results"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create symlink without '=' in name for Hydra compatibility
CHECKPOINT_DIR=$(dirname "$LATEST_CHECKPOINT")
CHECKPOINT_BASE=$(basename "$LATEST_CHECKPOINT")
SAFE_CHECKPOINT="$CHECKPOINT_DIR/epoch04.ckpt"
if [ ! -L "$SAFE_CHECKPOINT" ]; then
    ln -s "$CHECKPOINT_BASE" "$SAFE_CHECKPOINT"
fi

# Run PDMS evaluation
# Note: SceneLoader will load navsim logs for metadata (required for proper evaluation)

# Create temp config with checkpoint path
cat > /tmp/ar_eval_config.yaml << EOF
# Override checkpoint
checkpoint_path: "$LATEST_CHECKPOINT"
EOF

python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$SAFE_CHECKPOINT" \
    worker=ray_distributed \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_ar_eval \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_results

echo "=================================================="
echo "Evaluation complete!"
echo "Results saved to: /data2/byounggun/diffusiondrive_ar_output/eval_results"
echo "=================================================="
