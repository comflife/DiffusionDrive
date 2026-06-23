#!/bin/bash
# Sweep eval for Dr. GRPO+ train-split random scenes (epochs 20,25,30,35,39)
# GPUs 1,2,3 run epochs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_SCENE_COUNT="${RANDOM_SCENE_COUNT:-4586}"
RANDOM_SCENE_DATA_SPLIT="${RANDOM_SCENE_DATA_SPLIT:-trainval}"
RANDOM_TOKEN_CACHE="$SCRIPT_DIR/.cache/random_scene_tokens_${RANDOM_SCENE_DATA_SPLIT}_seed${RANDOM_SEED}_n${RANDOM_SCENE_COUNT}.json"

if [ ! -f "$RANDOM_TOKEN_CACHE" ]; then
    echo "ERROR: random token cache not found: $RANDOM_TOKEN_CACHE" >&2
    exit 1
fi

TRAIN_SCENE_TOKENS="$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['tokens'], separators=(',',':')))" "$RANDOM_TOKEN_CACHE")"

echo "Evaluating on train-split random scene tokens (count: $RANDOM_SCENE_COUNT)"

export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_train_random}"
export RL_NAME="${RL_NAME:-drgrpo_plus_ver2_train_random_scenes}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO+ v6_waymo epoch120 ver2 train random scenes}"
export RL_DIR="${DRGRPO_PLUS_V6_WAYMO_VER2_TRAIN_RANDOM_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_random_scenes}"

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
