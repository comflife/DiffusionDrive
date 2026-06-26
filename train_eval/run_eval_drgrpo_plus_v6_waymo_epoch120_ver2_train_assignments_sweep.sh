#!/bin/bash
# Sweep eval for Dr. GRPO+ train_assignment_scenes on TEST assignment scenes.
# GPUs 1,2,3 run epochs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_gpu() {
    local gpu=$1
    shift
    for EPOCH in "$@"; do
        echo "=================================================="
        echo "GPU $gpu evaluating epoch $EPOCH on test assignment scenes ..."
        echo "=================================================="
        GPUS="$gpu" EPOCH_TAG="$EPOCH" NUM_EPOCHS=1 \
            "$SCRIPT_DIR/run_eval_drgrpo_plus_v6_waymo_epoch120_ver2_train_assignments_latest2.sh"
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
