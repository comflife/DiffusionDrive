#!/bin/bash
# Dr. GRPO+ v2 on the loadable VAL assignment scenes plus the same count of
# non-overlapping random VAL scenes (1:1, 2x total).
#
# Only 132 of the 455 val-assignment scenes have a route (roadblock_ids) and a
# PDM metric cache entry; the other 323 cannot be PDM-scored. We therefore use
# those 132 loadable assignment scenes and sample 132 disjoint random official-val
# scenes (which all have routes), for 264 scenes total. Everything resolves from
# the single metric_cache_val (metric_cache_val_assign is a strict subset of it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_val_assign_loadable_scene_tokens.sh"
source "$SCRIPT_DIR/_random_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_SCENE_DATA_SPLIT="${RANDOM_SCENE_DATA_SPLIT:-val}"

# Loadable val-assignment scenes (132)
ASSIGNMENT_TOKEN_OUTPUT="$(val_assign_loadable_scene_tokens_and_count)"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

# Disjoint random official-val scenes, matched 1:1 to the loadable assign count.
export RANDOM_SCENE_DATA_SPLIT
RANDOM_TOKEN_OUTPUT="$(random_scene_tokens_and_count "$ASSIGNMENT_JSON" "$RANDOM_SEED" "$ASSIGNMENT_SCENE_COUNT")"
RANDOM_SCENE_COUNT="${RANDOM_TOKEN_OUTPUT%% *}"
RANDOM_SCENE_TOKENS="${RANDOM_TOKEN_OUTPUT#* }"

# Merge loadable-assign + random (drop any overlap defensively)
COMBINED_TOKEN_OUTPUT="$(python3 - "$ASSIGNMENT_SCENE_TOKENS" "$RANDOM_SCENE_TOKENS" <<'PY'
import json, sys
assign = json.loads(sys.argv[1])
rand = json.loads(sys.argv[2])
assign_set = set(assign)
overlap = assign_set & set(rand)
if overlap:
    print(f"ERROR: assign+random overlap ({len(overlap)} tokens)", file=sys.stderr)
    sys.exit(7)
merged = assign + [t for t in rand if t not in assign_set]
print(f"{len(merged)} {json.dumps(merged, separators=(',', ':'))}")
PY
)"
COMBINED_SCENE_COUNT="${COMBINED_TOKEN_OUTPUT%% *}"
COMBINED_SCENE_TOKENS="${COMBINED_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_val_assign_plus_random_scenes}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"

echo "Val assign scenes (loadable) : $ASSIGNMENT_SCENE_COUNT (of 455; 323 dropped: no route)"
echo "Random val scenes (disjoint) : $RANDOM_SCENE_COUNT"
echo "Combined scenes              : $COMBINED_SCENE_COUNT"
echo "Random seed                  : $RANDOM_SEED"
echo "Random data split            : $RANDOM_SCENE_DATA_SPLIT"
echo "Metric cache                 : $METRIC_CACHE_PATH"
echo "Output override              : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgrpo_plus_training_v6_waymo_ver2_val.sh" \
    "++train_test_split.scene_filter.tokens=$COMBINED_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_val_assign_plus_random_scenes \
    "wandb.name=drgrpo_plus_v6_waymo_ep120_ver2_val_assign_plus_random_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
