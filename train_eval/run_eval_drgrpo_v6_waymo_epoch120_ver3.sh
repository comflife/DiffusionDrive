#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_NAME="${RL_NAME:-drgrpo_ver3}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO v6_waymo epoch120 ver3}"
export RL_DIR="${DRGRPO_V6_WAYMO_VER3_DIR:-${DRGRPO_DIR:-${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6_waymo_epoch120_ver3}}}"

exec "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh" "$@"
