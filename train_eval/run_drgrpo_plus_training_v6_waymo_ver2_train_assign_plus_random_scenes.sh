#!/bin/bash
# Dr. GRPO+ v2 on TRAIN-split assignment scenes plus 500 cache-backed random
# train scenes (disjoint from assignment). Resolves to ~1574 against the merged
# val+navtrain metric cache (1074 assignment + 500 random).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-train}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_COUNT="${RANDOM_COUNT:-500}"
# Pre-generated pool of cache-backed, non-assignment trainval scene_tokens.
# (Built from the merged val+navtrain metric cache so every entry is trainable.)
RANDOM_TOKENS_FILE="${RANDOM_TOKENS_FILE:-$SCRIPT_DIR/.cache/cached_random_train_seed${RANDOM_SEED}_n${RANDOM_COUNT}.json}"

ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

if [ ! -f "$RANDOM_TOKENS_FILE" ]; then
    echo "ERROR: cached random tokens file not found: $RANDOM_TOKENS_FILE" >&2
    echo "       (expected a JSON list of $RANDOM_COUNT cache-backed, non-assignment train scene_tokens)" >&2
    exit 1
fi

# Merge: assignment scene_tokens + random scene_tokens (drop any overlap).
COMBINED_TOKEN_OUTPUT="$(python3 - "$ASSIGNMENT_SCENE_TOKENS" "$RANDOM_TOKENS_FILE" <<'PY'
import json, sys
from pathlib import Path

assign = json.loads(sys.argv[1])
rand = json.loads(Path(sys.argv[2]).read_text())
assign_set = set(assign)
merged = assign + [t for t in rand if t not in assign_set]
print(f"{len(merged)} {json.dumps(merged, separators=(',', ':'))}")
PY
)"
COMBINED_SCENE_COUNT="${COMBINED_TOKEN_OUTPUT%% *}"
COMBINED_SCENE_TOKENS="${COMBINED_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_assign_plus_random_scenes}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_train_merged}"

echo "Assignment JSON      : $ASSIGNMENT_JSON"
echo "Assignment splits    : $ASSIGNMENT_SPLITS"
echo "Assignment scenes    : $ASSIGNMENT_SCENE_COUNT (requested)"
echo "Random tokens file   : $RANDOM_TOKENS_FILE"
echo "Random scenes        : $RANDOM_COUNT (cache-backed, disjoint)"
echo "Combined scenes      : $COMBINED_SCENE_COUNT (requested; ~1574 resolve against cache)"
echo "Metric cache         : $METRIC_CACHE_PATH"
echo "Output override      : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgrpo_plus_training_v6_waymo_ver2_train.sh" \
    "++train_test_split.scene_filter.tokens=$COMBINED_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_train_assign_plus_random_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_train_assign_plus_random_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
