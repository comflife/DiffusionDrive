#!/bin/bash
# Build the metric cache (PDM reward GT) for the 455 val-ASSIGNMENT scenes whose
# sensors were just downloaded. metric caching needs only logs + maps (no sensors),
# and applies NO train/val log-split restriction, so passing the val-assign
# scene_tokens is enough. Output goes to a dedicated cache dir on /data2.
#
# Run detached (it can take a while):
#   tmux new -s mc 'bash train_eval/run_metric_caching_val_assign.sh 2>&1 | tee /data2/byounggun/metric_cache_val_assign.log'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
cd "$NAVSIM_DEVKIT_ROOT"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
CACHE_PATH="${CACHE_PATH:-/data2/byounggun/metric_cache_val_assign}"

TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" val)"
SCENE_COUNT="${TOKEN_OUTPUT%% *}"
SCENE_TOKENS="${TOKEN_OUTPUT#* }"

echo "=================================================="
echo "Metric caching for val-assignment scenes"
echo "  assignment scenes : $SCENE_COUNT"
echo "  cache path        : $CACHE_PATH"
echo "  (resumable: existing metric_cache.pkl are skipped)"
echo "=================================================="

python3 navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$SCENE_TOKENS" \
    cache.cache_path="$CACHE_PATH" \
    metric_cache_path="$CACHE_PATH" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    "$@"

echo "=================================================="
echo "Done. Cached scenes:"
find "$CACHE_PATH" -name metric_cache.pkl 2>/dev/null | wc -l
echo "Cache path: $CACHE_PATH"
echo "=================================================="
