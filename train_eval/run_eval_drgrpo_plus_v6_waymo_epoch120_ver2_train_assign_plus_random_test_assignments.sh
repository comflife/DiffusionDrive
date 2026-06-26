#!/bin/bash
# Evaluate Dr. GRPO+ train_assign_plus_random checkpoints on TEST assignment scenes.
# Random scenes are training-only augmentation; eval uses assignment test split only.
#
# Usage:
#   EPOCH_TAG=22 GPUS=0 bash train_eval/run_eval_drgrpo_plus_v6_waymo_epoch120_ver2_train_assign_plus_random_test_assignments.sh
#   GPUS=0,1 NUM_EPOCHS=2 bash train_eval/run_eval_drgrpo_plus_v6_waymo_epoch120_ver2_train_assign_plus_random_test_assignments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_DIR="${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_assign_plus_random_scenes}"
export EVAL_ROOT="${EVAL_ROOT:-/data2/byounggun/diffusiondrive_drgrpo_plus_eval_v6_waymo_epoch120_ver2_train_assign_plus_random_test_assignments}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-drgrpo_plus_v6_waymo_train_assign_plus_random}"

source "$SCRIPT_DIR/_run_eval_drgrpo_plus_test_assignments.sh"
