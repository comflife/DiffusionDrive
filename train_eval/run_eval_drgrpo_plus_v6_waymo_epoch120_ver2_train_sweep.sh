#!/bin/bash
# Sweep eval for Dr. GRPO+ base model trained on train split (epochs 20,25,30,35,39)
# GPUs 1,2,3 run epochs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-train}"
ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
TRAIN_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

echo "Evaluating on train-split scene tokens (count: ${ASSIGNMENT_TOKEN_OUTPUT%% *})"

export RL_NAME="${RL_NAME:-drgrpo_plus_ver2_train}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO+ v6_waymo epoch120 ver2 train}"
export RL_DIR="${DRGRPO_PLUS_V6_WAYMO_VER2_TRAIN_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train}"

run_gpu() {
    local gpu=$1
    shift
    for EPOCH in "$@"; do
        echo "=================================================="
        echo "GPU $gpu evaluating epoch $EPOCH ..."
        echo "=================================================="
        GPUS="$gpu" EPOCH_TAG="$EPOCH" "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_train_latest.sh" \
            "++train_test_split.scene_filter.tokens=$TRAIN_SCENE_TOKENS"
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
