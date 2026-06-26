#!/bin/bash
# Shared PDMS eval on TEST assignment scenes (navtest + assignment JSON test split).
# Training wrappers set RL_DIR / EVAL_ROOT / EXPERIMENT_PREFIX before sourcing this file.

set -euo pipefail

: "${SCRIPT_DIR:?SCRIPT_DIR must be set by the caller}"
: "${RL_DIR:?RL_DIR must be set by the caller}"
: "${EVAL_ROOT:?EVAL_ROOT must be set by the caller}"
: "${EXPERIMENT_PREFIX:?EXPERIMENT_PREFIX must be set by the caller}"

source "$SCRIPT_DIR/_assignment_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/home/byounggun/DiffusionDrive}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/navsim/exp/bg}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/navsim/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/navsim/dataset/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-test}"
CKPT_DIR="${CKPT_DIR:-$RL_DIR/checkpoints}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
GPUS="${GPUS:-0,1}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
EPOCH_TAG="${EPOCH_TAG:-}"
EPOCH_TAGS="${EPOCH_TAGS:-}"
CKPT="${CKPT:-}"
WORKER_THREADS="${WORKER_THREADS:-2}"
SUMMARY_CSV="$EVAL_ROOT/summary_pdms.csv"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
fi

ASSIGNMENT_TOKEN_OUTPUT="$(assignment_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS")"
ASSIGNMENT_SCENE_COUNT="${ASSIGNMENT_TOKEN_OUTPUT%% *}"
ASSIGNMENT_SCENE_TOKENS="${ASSIGNMENT_TOKEN_OUTPUT#* }"
TOKENS_FILE="$SCRIPT_DIR/.cache/eval_assignment_scene_tokens_${ASSIGNMENT_SPLITS}.json"
mkdir -p "$(dirname "$TOKENS_FILE")"
python3 - "$ASSIGNMENT_SCENE_TOKENS" "$TOKENS_FILE" <<'PY'
import json
import sys
from pathlib import Path

tokens = json.loads(sys.argv[1])
Path(sys.argv[2]).write_text(json.dumps({"tokens": tokens}, indent=2))
PY

mapfile -t CKPTS < <(
    python3 - "$CKPT_DIR" "$NUM_EPOCHS" "$EPOCH_TAG" "$EPOCH_TAGS" "$CKPT" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
num_epochs = int(sys.argv[2])
epoch_tag = sys.argv[3].strip()
epoch_tags = [part.strip() for part in sys.argv[4].split(",") if part.strip()]
explicit_ckpt = sys.argv[5].strip()

if explicit_ckpt:
    path = Path(explicit_ckpt)
    if not path.is_file():
        raise SystemExit(f"ERROR: CKPT not found: {path}")
    print(path)
    raise SystemExit

if epoch_tag:
    path = ckpt_dir / f"grpo-epoch={epoch_tag}.ckpt"
    if not path.is_file():
        raise SystemExit(f"ERROR: checkpoint not found: {path}")
    print(path)
    raise SystemExit

if epoch_tags:
    for tag in epoch_tags:
        path = ckpt_dir / f"grpo-epoch={tag}.ckpt"
        if not path.is_file():
            raise SystemExit(f"ERROR: checkpoint not found: {path}")
        print(path)
    raise SystemExit

items = []
for path in ckpt_dir.glob("grpo-epoch=*.ckpt"):
    match = re.search(r"grpo-epoch=(\d+)\.ckpt$", path.name)
    if match:
        items.append((int(match.group(1)), path))
if not items:
    raise SystemExit(f"ERROR: no grpo-epoch=*.ckpt files found in {ckpt_dir}")
for _, path in sorted(items)[-num_epochs:]:
    print(path)
PY
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoints selected from $CKPT_DIR" >&2
    exit 1
fi

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p "$EVAL_ROOT"
rm -f "$EVAL_ROOT"/summary_epoch_*.csv
echo "epoch,checkpoint,output_dir,csv,NC,DAC,TTC,Comfort,EP,PDMS,valid,num_rows" > "$SUMMARY_CSV"

echo "Evaluating test assignment scenes: $ASSIGNMENT_SCENE_COUNT"
echo "Token file   : $TOKENS_FILE"
echo "Checkpoint   : $CKPT_DIR"
echo "Eval root    : $EVAL_ROOT"
echo "Metric cache : $METRIC_CACHE_PATH"
echo "GPUs         : $GPUS"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

run_eval_one() {
    local ckpt="$1"
    local gpu="$2"
    local epoch
    epoch="$(basename "$ckpt" | sed -E 's/^grpo-epoch=([0-9]+)\.ckpt$/\1/')"
    printf -v epoch_tag "%03d" "$epoch"

    local out_dir="$EVAL_ROOT/epoch_${epoch_tag}"
    local hydra_ckpt="/tmp/${EXPERIMENT_PREFIX}_test_assignment_epoch_${epoch_tag}.ckpt"
    local epoch_summary="$EVAL_ROOT/summary_epoch_${epoch_tag}.csv"

    ln -sfn "$ckpt" "$hydra_ckpt"
    mkdir -p "$out_dir"

    echo "--------------------------------------------------"
    echo "[epoch $epoch] GPU $gpu"
    echo "Checkpoint : $ckpt"
    echo "Output     : $out_dir"
    echo "--------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$gpu" \
    RAY_CUDA_VISIBLE_DEVICES="$gpu" \
    python3 -m navsim.planning.script.run_pdm_score \
        train_test_split=navtest \
        "++train_test_split.scene_filter.log_names=null" \
        "++train_test_split.scene_filter.tokens_file=$TOKENS_FILE" \
        agent=diffusiondrive_ar_agent \
        "agent.checkpoint_path=$hydra_ckpt" \
        agent.config.ego_vocab_size=2048 \
        agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy \
        agent.config.ar_codebook_mode=step_corners \
        agent.config.ar_teacher_forcing=false \
        agent.config.ar_num_modes=1 \
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
        navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
        sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
        experiment_name="${EXPERIMENT_PREFIX}_test_assignment_eval_epoch_${epoch_tag}" \
        output_dir="$out_dir"

    local latest_csv
    latest_csv="$(find "$out_dir" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [ -z "$latest_csv" ]; then
        echo "[epoch $epoch] ERROR: no csv found in $out_dir" >&2
        return 1
    fi

    python3 - "$epoch" "$ckpt" "$out_dir" "$latest_csv" "$epoch_summary" <<'PY'
import csv
import sys
import pandas as pd

epoch, ckpt, out_dir, csv_path, summary_path = sys.argv[1:]
df = pd.read_csv(csv_path)
row = df.iloc[-1]
vals = {
    "NC": float(row["no_at_fault_collisions"]) * 100,
    "DAC": float(row["drivable_area_compliance"]) * 100,
    "TTC": float(row["time_to_collision_within_bound"]) * 100,
    "Comfort": float(row["comfort"]) * 100,
    "EP": float(row["ego_progress"]) * 100,
    "PDMS": float(row["score"]) * 100,
}
valid = str(row["valid"]).strip().lower() in ("true", "1", "yes")
num_rows = max(len(df) - 1, 0)

with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        epoch, ckpt, out_dir, csv_path,
        f"{vals['NC']:.4f}", f"{vals['DAC']:.4f}",
        f"{vals['TTC']:.4f}", f"{vals['Comfort']:.4f}",
        f"{vals['EP']:.4f}", f"{vals['PDMS']:.4f}",
        valid, num_rows,
    ])

print(
    f"[epoch {epoch}] NC={vals['NC']:.1f} DAC={vals['DAC']:.1f} "
    f"TTC={vals['TTC']:.1f} Comfort={vals['Comfort']:.1f} "
    f"EP={vals['EP']:.1f} PDMS={vals['PDMS']:.1f} valid={valid} rows={num_rows}"
)
PY
}

gpu_index=0
for ckpt in "${CKPTS[@]}"; do
    gpu="${GPU_LIST[$gpu_index]}"
    gpu_index=$(( (gpu_index + 1) % ${#GPU_LIST[@]} ))
    run_eval_one "$ckpt" "$gpu" &
done

wait

for summary in $(find "$EVAL_ROOT" -maxdepth 1 -type f -name 'summary_epoch_*.csv' | sort); do
    cat "$summary" >> "$SUMMARY_CSV"
done

echo "=================================================="
echo "PDMS summary"
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo "=================================================="
