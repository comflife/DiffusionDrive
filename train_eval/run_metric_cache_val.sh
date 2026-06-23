#!/bin/bash
# Build metric cache for the full navtrain val split.
# Also includes assignment val scenes (triage subset) so assignment/random RL
# can reuse the same cache path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_val_scene_tokens.sh"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"

ALL_VAL_TOKEN_OUTPUT="$(all_val_scene_tokens_and_count)"
ALL_VAL_SCENE_TOKENS="${ALL_VAL_TOKEN_OUTPUT#* }"

ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" val)"
ASSIGNMENT_VAL_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

read -r ALL_VAL_ONLY_COUNT ASSIGNMENT_VAL_ONLY_COUNT MERGED_SCENE_COUNT VAL_SCENE_TOKENS <<< "$(python3 - "$ALL_VAL_SCENE_TOKENS" "$ASSIGNMENT_VAL_SCENE_TOKENS" <<'PY'
import json
import sys

all_val = json.loads(sys.argv[1])
assignment_val = json.loads(sys.argv[2])
merged = sorted(set(all_val) | set(assignment_val))
print(len(all_val), len(assignment_val), len(merged), json.dumps(merged, separators=(',', ':')))
PY
)"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"

echo "=================================================="
echo "Building metric cache for full val split"
echo "Official val  : $ALL_VAL_ONLY_COUNT scenes"
echo "Assignment val: $ASSIGNMENT_VAL_ONLY_COUNT scenes (union for cache)"
echo "Cache total   : $MERGED_SCENE_COUNT scenes"
echo "Cache path    : $METRIC_CACHE_PATH"
echo "Log path      : $OPENSCENE_DATA_ROOT/navsim_logs/trainval"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.planning.script.run_metric_caching \
    train_test_split=navtrain \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$VAL_SCENE_TOKENS" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    cache.cache_path="$METRIC_CACHE_PATH" \
    cache.force_feature_computation=false \
    worker.threads_per_node=2
