#!/bin/bash
# Dr. GRPO+ v2 waymo epoch-120 fine-tuning on VAL-split assignment scene_tokens.
#
# Of the 455 val-split assignment scenes, only 132 have a route (roadblock_ids)
# in the raw trainval logs and therefore a PDM metric cache entry; the other 323
# have empty roadblock_ids and cannot be PDM-scored (navsim's own navtrain/navtest
# filters drop such scenes via has_route=true). We train on those 132 loadable
# scenes, which already live in metric_cache_val (the separate metric_cache_val_assign
# is a strict subset and is no longer needed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_val_assign_loadable_scene_tokens.sh"

ASSIGNMENT_TOKEN_OUTPUT="$(val_assign_loadable_scene_tokens_and_count)"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/trained_model/drgrpoplus_val_assign}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"

echo "Val assign scenes (loadable) : $ASSIGNMENT_SCENE_COUNT (of 455; 323 dropped: no route)"
echo "Metric cache                 : $METRIC_CACHE_PATH"
echo "Output override              : $OUTPUT_DIR"

exec "$SCRIPT_DIR/drgrpoplus_val_train.sh" \
    "++train_test_split.scene_filter.tokens=$ASSIGNMENT_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_val_assignment_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_val_assignment_scenes_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
