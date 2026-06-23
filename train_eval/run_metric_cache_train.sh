#!/bin/bash
# Build metric cache for train-split scenes (base / assignment scenes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-train}"
ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
TRAIN_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
TRAIN_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_train}"

echo "=================================================="
echo "Building metric cache for train-split scenes"
echo "Scene count   : $TRAIN_SCENE_COUNT"
echo "Cache path    : $METRIC_CACHE_PATH"
echo "Log path      : $OPENSCENE_DATA_ROOT/navsim_logs/trainval"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.planning.script.run_metric_caching \
    train_test_split=navtrain \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$TRAIN_SCENE_TOKENS" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    cache.cache_path="$METRIC_CACHE_PATH" \
    cache.force_feature_computation=false \
    worker.threads_per_node=2
