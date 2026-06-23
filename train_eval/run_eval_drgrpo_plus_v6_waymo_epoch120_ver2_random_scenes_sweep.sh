#!/bin/bash
# Sweep eval for Dr. GRPO+ random scenes (same count as assignment scenes)
# GPUs 1,2,3 run epochs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_NAME="${RL_NAME:-drgrpo_plus_ver2_random_scenes}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO+ v6_waymo epoch120 ver2 random scenes}"
export RL_DIR="${DRGRPO_PLUS_V6_WAYMO_VER2_RANDOM_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_random_scenes}"

run_gpu() {
    local gpu=$1
    shift
    for EPOCH in "$@"; do
        echo "=================================================="
        echo "GPU $gpu evaluating epoch $EPOCH ..."
        echo "=================================================="
        GPUS="$gpu" EPOCH_TAG="$EPOCH" "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh"
    done
}

# GPU 1: epochs 20, 25
run_gpu 1 20 25 &
# GPU 2: epochs 30, 35
run_gpu 2 30 35 &
# GPU 3: epoch 39
run_gpu 3 39 &

wait

echo "=================================================="
echo "All sweep evals complete!"
echo "=================================================="
