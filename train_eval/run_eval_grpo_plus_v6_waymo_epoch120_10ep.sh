#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RL_NAME="${RL_NAME:-grpo_plus_10ep}"
export EVAL_LABEL="${EVAL_LABEL:-GRPO+ v6_waymo epoch120 10ep}"
export RL_DIR="${GRPO_PLUS_V6_WAYMO_10EP_DIR:-${GRPO_PLUS_DIR:-${RL_DIR:-/data2/byounggun/diffusiondrive_grpo_plus_output_v6_waymo_epoch120_10ep}}}"
export GPUS="2"

CKPT_DIR="${RL_DIR}/checkpoints"
SUMMARY_CSV="/home/byounggun/DiffusionDrive/${RL_NAME}_eval_all_metrics.csv"

echo "epoch,score,valid,num_rows,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance" > "$SUMMARY_CSV"

epochs=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'grpo-epoch=*.ckpt' -printf '%f\n' \
    | sed -n 's/^grpo-epoch=\([0-9][0-9]*\)\.ckpt$/\1/p' \
    | sort -n)

for ep in $epochs; do
    echo "=================================================="
    echo "Evaluating epoch $ep on GPU ${GPUS} ..."
    echo "=================================================="

    EPOCH_TAG="$ep" GPUS="$GPUS" bash "$SCRIPT_DIR/run_eval_rl_v6_waymo_epoch120_latest.sh" || {
        echo "WARNING: eval failed for epoch $ep, skipping..."
        continue
    }

    eval_dir="${RL_DIR}/eval_epoch_$(printf '%02d' $((10#$ep)))"
    csv_file=$(find "$eval_dir" -maxdepth 1 -name '*.csv' -not -name 'summary_pdms.csv' | sort | tail -1)

    if [ -f "$csv_file" ]; then
        python3 - "$ep" "$csv_file" "$SUMMARY_CSV" <<'PY'
import sys, pandas as pd, csv
ep, csv_path, out_path = sys.argv[1:]
df = pd.read_csv(csv_path)
row = [
    int(ep),
    df['score'].mean(),
    df['valid'].mean(),
    len(df),
    df['no_at_fault_collisions'].mean(),
    df['drivable_area_compliance'].mean(),
    df['ego_progress'].mean(),
    df['time_to_collision_within_bound'].mean(),
    df['comfort'].mean(),
    df['driving_direction_compliance'].mean(),
]
with open(out_path, 'a', newline='') as f:
    csv.writer(f).writerow(row)
PY
    else
        echo "WARNING: no result CSV found in $eval_dir"
    fi
done

echo "=================================================="
echo "All epochs evaluated. Summary: $SUMMARY_CSV"
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo "=================================================="
