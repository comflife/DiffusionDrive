#!/bin/bash
# Dr. GRPO+ v2 waymo epoch-120 fine-tuning on TRAIN-split assignment scene_tokens
# (currently-cached ones only; ~1074 resolved against the merged val+navtrain cache).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-train}"
ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_assignment_scenes}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_train_merged}"

echo "Assignment JSON   : $ASSIGNMENT_JSON"
echo "Assignment splits : $ASSIGNMENT_SPLITS"
echo "Assignment scenes : $ASSIGNMENT_SCENE_COUNT"
echo "Metric cache      : $METRIC_CACHE_PATH"
echo "Output override   : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgrpo_plus_training_v6_waymo_ver2_train.sh" \
    "++train_test_split.scene_filter.tokens=$ASSIGNMENT_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_train_assignment_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_train_assignment_scenes_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
