#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_NAME="${RL_NAME:-grpo}"
export RL_DIR="${GRPO_DIR:-${RL_DIR:-/data2/byounggun/diffusiondrive_grpo_output_v6}}"

exec "$SCRIPT_DIR/run_eval_rl_v6_all.sh" "$@"
