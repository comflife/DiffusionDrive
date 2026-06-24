#!/bin/bash
# Dr. GRPO+ v2 on val assignment scenes plus the same count of non-overlapping
# random val scenes (2x assignment count total).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"
source "$SCRIPT_DIR/_random_scene_tokens.sh"
source "$SCRIPT_DIR/_assign_plus_random_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-val}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_SCENE_DATA_SPLIT="${RANDOM_SCENE_DATA_SPLIT:-val}"

ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"

COMBINED_TOKEN_OUTPUT="$(assign_plus_random_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS" "$RANDOM_SEED" "$RANDOM_SCENE_DATA_SPLIT")"
COMBINED_SCENE_COUNT="${COMBINED_TOKEN_OUTPUT%% *}"
COMBINED_SCENE_TOKENS="${COMBINED_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_val_assign_plus_random_scenes}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"

echo "Assignment JSON      : $ASSIGNMENT_JSON"
echo "Assignment splits    : $ASSIGNMENT_SPLITS"
echo "Assignment scenes    : $ASSIGNMENT_SCENE_COUNT"
echo "Random scenes        : $ASSIGNMENT_SCENE_COUNT (disjoint)"
echo "Combined scenes      : $COMBINED_SCENE_COUNT"
echo "Random seed          : $RANDOM_SEED"
echo "Random data split    : $RANDOM_SCENE_DATA_SPLIT"
echo "Metric cache         : $METRIC_CACHE_PATH"
echo "Output override      : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgrpo_plus_training_v6_waymo_ver2_train.sh" \
    "++train_test_split.scene_filter.tokens=$COMBINED_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_val_assign_plus_random_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_val_assign_plus_random_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
