#!/bin/bash
# Sweep eval for Dr. GRPO+ assignment scenes (epochs 20,25,30,35,39)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_NAME="${RL_NAME:-drgrpo_plus_ver2_assignment_scenes}"
export EVAL_LABEL="${EVAL_LABEL:-Dr. GRPO+ v6_waymo epoch120 ver2 assignment scenes}"
export RL_DIR="${DRGRPO_PLUS_V6_WAYMO_VER2_ASSIGNMENT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_assignment_scenes}"

for EPOCH in 20 25 30 35 39; do
    echo "=================================================="
    echo "Evaluating epoch $EPOCH ..."
    echo "=================================================="
    EPOCH_TAG="$EPOCH" "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh" "$@"
done

echo "=================================================="
echo "All sweep evals complete!"
echo "=================================================="
