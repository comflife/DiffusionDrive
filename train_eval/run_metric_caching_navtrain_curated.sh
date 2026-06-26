#!/bin/bash
# Build the metric cache (PDM reward GT) for the FULL navtrain CURATED set
# (= NAVSIM's official trainval-combined curation: 1,192 logs / 103,288 frame tokens,
#  which is exactly what the base model was trained on).
#
# One cache serves BOTH RL experiments:
#   - full trainset-curated RL : train_test_split=navtrain  (uses all 103,288)
#   - train-assign curated RL  : pass assign scene_tokens; GRPO auto-intersects the
#                                cache, yielding only the curated frames of those
#                                scenes (~37,210 windows / 3,524 scenes).
# val-assign has 0 curated frames, so it CANNOT use this cache.
#
# metric caching needs only logs + maps (no sensors). It is resumable: existing
# metric_cache.pkl are skipped. It is large (~103k tokens), so run detached:
#   tmux new -s mc 'bash train_eval/run_metric_caching_navtrain_curated.sh 2>&1 | tee /data2/byounggun/custom_cache.log'

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
cd "$NAVSIM_DEVKIT_ROOT"

CACHE_PATH="${CACHE_PATH:-/data2/byounggun/custom_cache}"

echo "=================================================="
echo "Metric caching for FULL navtrain CURATED set"
echo "  scene_filter      : navtrain (1,192 logs / 103,288 curated tokens)"
echo "  cache path        : $CACHE_PATH"
echo "  (resumable: existing metric_cache.pkl are skipped)"
echo "=================================================="

python3 navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain \
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
