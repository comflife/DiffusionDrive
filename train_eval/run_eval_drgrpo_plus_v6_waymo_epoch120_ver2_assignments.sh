#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_NAME="${RL_NAME:-drgrpo_plus_ver2_assignment_scenes}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO+ v6_waymo epoch120 ver2 assignment scenes}"
export RL_DIR="${DRGRPO_PLUS_V6_WAYMO_VER2_ASSIGNMENT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_assignment_scenes}"

exec "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh" "$@"
