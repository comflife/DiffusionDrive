#!/bin/bash
# DiffusionDrive-AR Fast Evaluation Script
# Uses metric cache only (no scene loader overhead)

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps

# Ensure DiffusionDrive navsim is used
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Find latest checkpoint
CHECKPOINT_DIR="/data2/byounggun/diffusiondrive_ar_output"
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/diffusiondrive-ar/*/checkpoints/last.ckpt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi

echo "=================================================="
echo "DiffusionDrive-AR Fast PDMS Evaluation"
echo "=================================================="
echo "Checkpoint: $LATEST_CHECKPOINT"
echo "Metric Cache: /data2/byounggun/metric_cache"
echo "=================================================="

cd $NAVSIM_DEVKIT_ROOT

# Create a temporary config that uses cache only
cat > /tmp/ar_eval_config.yaml << EOF
# Minimal config for fast metric-based eval
hydra:
  run:
    dir: /data2/byounggun/diffusiondrive_ar_output/eval_results
  job:
    chdir: False

defaults:
  - override agent: diffusiondrive_ar_agent
  - override train_test_split: navtest

# Agent settings
agent:
  checkpoint_path: $LATEST_CHECKPOINT
  
# Metric cache only
metric_cache_path: /data2/byounggun/metric_cache
use_cache_without_dataset: true

# Worker
worker:
  _target_: navsim.planning.script.builders.worker_pool_builder.build_worker
  params:
    worker_type: ray_distributed
    max_workers: 64
EOF

# Run eval with metric cache only
python -m navsim.planning.script.run_pdm_score \
    --config-path=/tmp \
    --config-name=ar_eval_config \
    metric_cache_path=/data2/byounggun/metric_cache \
    agent.checkpoint_path=$LATEST_CHECKPOINT \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/eval_results

echo "=================================================="
echo "Evaluation complete!"
echo "=================================================="
