#!/bin/bash
# Sweep eval for Dr. GRPO+ train_full checkpoints on TEST assignment scenes.
# GPUs 1,2,3 run epochs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_DIR="${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_full}"
export EVAL_ROOT="${EVAL_ROOT:-/data2/byounggun/diffusiondrive_drgrpo_plus_eval_v6_waymo_epoch120_ver2_train_full_test_assignments}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-drgrpo_plus_v6_waymo_train_full}"

run_gpu() {
    local gpu=$1
    shift
    for EPOCH in "$@"; do
        echo "=================================================="
        echo "GPU $gpu evaluating epoch $EPOCH on test assignment scenes ..."
        echo "=================================================="
        GPUS="$gpu" EPOCH_TAG="$EPOCH" NUM_EPOCHS=1 \
            SCRIPT_DIR="$SCRIPT_DIR" RL_DIR="$RL_DIR" EVAL_ROOT="$EVAL_ROOT" EXPERIMENT_PREFIX="$EXPERIMENT_PREFIX" \
            bash "$SCRIPT_DIR/_run_eval_drgrpo_plus_test_assignments.sh"
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
