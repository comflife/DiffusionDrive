#!/bin/bash
# Dr. GSPO v2 waymo epoch-120 fine-tuning on a random scene_token subset
# with the same count as the assignment-triage export (excluding assignment scenes).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"
source "$SCRIPT_DIR/_random_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-all}"
ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"

RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_SCENE_COUNT="${RANDOM_SCENE_COUNT:-$ASSIGNMENT_SCENE_COUNT}"
RANDOM_TOKEN_OUTPUT="$(random_scene_tokens_and_count "$ASSIGNMENT_JSON" "$RANDOM_SEED" "$RANDOM_SCENE_COUNT")"
RANDOM_SCENE_TOKENS="${RANDOM_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_dr_gspo_output_v6_waymo_epoch120_ver2_random_scenes}"

echo "Assignment JSON      : $ASSIGNMENT_JSON"
echo "Assignment scene cnt : $ASSIGNMENT_SCENE_COUNT (reference count)"
echo "Random seed          : $RANDOM_SEED"
echo "Random scene count   : $RANDOM_SCENE_COUNT"
echo "Random token cache   : $SCRIPT_DIR/.cache/random_scene_tokens_seed${RANDOM_SEED}_n${RANDOM_SCENE_COUNT}.json"
echo "Output override      : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgspo_training_v6_waymo_120_ver2.sh" \
    "++train_test_split.scene_filter.tokens=$RANDOM_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgspo_v6_waymo_epoch120_ver2_random_scenes \
    "wandb.name=drgspo_v6_waymo_ep120_ver2_random_scenes_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"