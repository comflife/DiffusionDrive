#!/bin/bash
# Evaluate latest Dr. GRPO+ train_assignment_scenes checkpoints on TEST assignment scenes.
# Defaults to the latest two grpo-epoch=*.ckpt files, split across GPUs 0 and 1.
#
# Usage:
#   GPUS=0,1 NUM_EPOCHS=2 bash train_eval/run_eval_drgrpo_plus_v6_waymo_epoch120_ver2_train_assignments_latest2.sh
#   EPOCH_TAG=22 GPUS=0 bash train_eval/run_eval_drgrpo_plus_v6_waymo_epoch120_ver2_train_assignments_latest2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_DIR="${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_train_assignment_scenes}"
export EVAL_ROOT="${EVAL_ROOT:-/data2/byounggun/diffusiondrive_drgrpo_plus_eval_v6_waymo_epoch120_ver2_train_assignment_scenes_test_latest2}"
export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-drgrpo_plus_v6_waymo_train_assignment_scenes}"

source "$SCRIPT_DIR/_run_eval_drgrpo_plus_test_assignments.sh"
