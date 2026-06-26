#!/bin/bash
# Dr. GRPO+ v2 waymo epoch-120 fine-tuning on TRAIN-split assignment scene_tokens,
# intersected with the navtrain CURATED metric cache (~37k windows / ~3.5k scenes).
#
# Passes all 4,586 train-assignment scene_tokens; GRPO keeps only frames present in
# custom_cache (navtrain curation). Scenes with no curated frames are dropped
# automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" train)"
ASSIGNMENT_SCENE_COUNT="${TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/trained_model/drgrpoplus_train_assign}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/custom_cache}"

echo "Train assign scenes        : $ASSIGNMENT_SCENE_COUNT"
echo "Metric cache (curated)     : $METRIC_CACHE_PATH"
echo "Output override            : $OUTPUT_DIR"

exec "$SCRIPT_DIR/drgrpoplus_train_train.sh" \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$ASSIGNMENT_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_train_assignment_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_train_assignment_scenes_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
