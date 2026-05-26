#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_NAME="${RL_NAME:-drgrpo_ver2}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO v6_waymo epoch120 ver2}"
export RL_DIR="${DRGRPO_V6_WAYMO_VER2_DIR:-${DRGRPO_DIR:-${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6_waymo_epoch120_ver2}}}"

exec "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh" "$@"
