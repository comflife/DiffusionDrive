#!/bin/bash
# Dr. GRPO v2 waymo epoch-120 fine-tuning, restricted to scene_token entries
# from an assignment-triage JSON export.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-all}"
ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"

export OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6_waymo_epoch120_ver2_assignment_scenes}"

echo "Assignment JSON   : $ASSIGNMENT_JSON"
echo "Assignment splits : $ASSIGNMENT_SPLITS"
echo "Assignment scenes : $ASSIGNMENT_SCENE_COUNT before current NAVSIM split/path filtering"
echo "Output override   : $OUTPUT_DIR"

exec "$SCRIPT_DIR/run_drgrpo_training_v6_waymo_120_ver2.sh" \
    "++train_test_split.scene_filter.tokens=$ASSIGNMENT_SCENE_TOKENS" \
    ++experiment_name=diffusiondrive_ar_drgrpo_v6_waymo_epoch120_ver2_assignment_scenes \
    "wandb.name=drgrpo_v6_waymo_ep120_ver2_assignment_scenes_g${GROUP_SIZE:-12}_lr${LR:-1e-5}_clip${CLIP_EPS:-0.25}_acc${ACCUMULATE_GRAD_BATCHES:-8}" \
    "$@"
