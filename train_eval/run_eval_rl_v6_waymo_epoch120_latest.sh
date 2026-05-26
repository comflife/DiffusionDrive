#!/bin/bash
# Evaluate the highest grpo-epoch checkpoint for v6_waymo epoch-120 RL runs.
#
# This is intended to be called by per-experiment wrappers, but can also be used
# directly:
#   RL_NAME=drgrpo RL_DIR=/path/to/output bash train_eval/run_eval_rl_v6_waymo_epoch120_latest.sh
#   EPOCH_TAG=25 RL_DIR=/path/to/output bash train_eval/run_eval_rl_v6_waymo_epoch120_latest.sh

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR="${TMPDIR:-/data2/byounggun/ray_tmp}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

RL_NAME="${RL_NAME:-rl}"
EVAL_LABEL="${EVAL_LABEL:-$RL_NAME}"
RL_DIR="${RL_DIR:?Set RL_DIR to the RL output directory}"
CKPT_DIR="${CKPT_DIR:-$RL_DIR/checkpoints}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
WORKER_THREADS="${WORKER_THREADS:-2}"
EPOCH_TAG="${EPOCH_TAG:-}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_DIR" >&2
    exit 1
fi

if [ -z "$EPOCH_TAG" ]; then
    EPOCH_TAG="$(
        find "$CKPT_DIR" -maxdepth 1 -type f -name 'grpo-epoch=*.ckpt' -printf '%f\n' \
            | sed -n 's/^grpo-epoch=\([0-9][0-9]*\)\.ckpt$/\1/p' \
            | sort -n \
            | tail -1
    )"
fi

if [ -z "$EPOCH_TAG" ]; then
    echo "ERROR: no grpo-epoch=*.ckpt files found under $CKPT_DIR" >&2
    exit 1
fi

if [ "$EPOCH_TAG" = "last" ]; then
    CKPT="$CKPT_DIR/last.ckpt"
    OUT_TAG="last"
else
    EPOCH_NUM="$((10#$EPOCH_TAG))"
    printf -v EPOCH_TAG "%02d" "$EPOCH_NUM"
    CKPT="$CKPT_DIR/grpo-epoch=${EPOCH_TAG}.ckpt"
    OUT_TAG="$EPOCH_TAG"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    echo "       Set EPOCH_TAG=<NN> or CKPT_DIR=/path/to/checkpoints to override." >&2
    exit 1
fi

EVAL_ROOT="${EVAL_ROOT:-$RL_DIR/eval_epoch_${OUT_TAG}}"
SUMMARY_CSV="$EVAL_ROOT/summary_pdms.csv"
SAFE_CKPT="/tmp/${RL_NAME}_v6_waymo_epoch120_epoch_${OUT_TAG}_$(date +%s).ckpt"
EXP_NAME="${RL_NAME}_v6_waymo_epoch120_eval_epoch_${OUT_TAG}"

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p "$EVAL_ROOT"
rm -f "$SUMMARY_CSV"
ln -sfn "$CKPT" "$SAFE_CKPT"

echo "=================================================="
echo "Evaluating $EVAL_LABEL latest/highest checkpoint"
echo "Checkpoint dir : $CKPT_DIR"
echo "Epoch tag      : $OUT_TAG"
echo "Source ckpt    : $CKPT"
echo "Hydra alias    : $SAFE_CKPT"
echo "Modified       : $(stat -c %y "$CKPT" 2>/dev/null)"
echo "GPUs           : $GPUS"
echo "Worker threads : $WORKER_THREADS"
echo "Metric cache   : $METRIC_CACHE_PATH"
echo "Output         : $EVAL_ROOT"
echo "Summary CSV    : $SUMMARY_CSV"
echo "=================================================="

CUDA_VISIBLE_DEVICES="$GPUS" \
RAY_CUDA_VISIBLE_DEVICES="$GPUS" \
python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$SAFE_CKPT" \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=false \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=true \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.agent_topk=30 \
    agent.config.freeze_pretrained_trunk=false \
    worker=ray_distributed \
    worker.threads_per_node="$WORKER_THREADS" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    experiment_name="$EXP_NAME" \
    output_dir="$EVAL_ROOT" \
    "$@"

latest_csv="$(find "$EVAL_ROOT" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [ -z "$latest_csv" ]; then
    echo "ERROR: no PDMS csv found in $EVAL_ROOT" >&2
    exit 1
fi

python3 - "$OUT_TAG" "$CKPT" "$EVAL_ROOT" "$latest_csv" "$SUMMARY_CSV" <<'PY'
import csv
import sys
import pandas as pd

epoch, ckpt, out_dir, csv_path, summary_path = sys.argv[1:]
df = pd.read_csv(csv_path)
summary_row = df.iloc[-1]
score = float(summary_row["score"])
valid_value = summary_row["valid"]
valid = str(valid_value).strip().lower() in ("true", "1", "yes")
num_rows = max(len(df) - 1, 0)

with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "checkpoint", "output_dir", "csv", "score", "valid", "num_rows"])
    writer.writerow([epoch, ckpt, out_dir, csv_path, f"{score:.8f}", valid, num_rows])

print(f"[epoch {epoch}] PDMS score={score:.8f} valid={valid} scenarios={num_rows}")
PY

echo "=================================================="
echo "PDMS summary"
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo "=================================================="
