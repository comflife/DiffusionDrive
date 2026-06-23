#!/bin/bash
# Build metric cache for train-split random scenes.
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

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_train_random}"

echo "=================================================="
echo "Building metric cache for train-split random scenes"
echo "Scene count   : $RANDOM_SCENE_COUNT"
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
