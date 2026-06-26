#!/bin/bash
# Evaluate the latest Dr. GRPO+ test assignment + random scene checkpoints.
# Defaults to the latest two grpo-epoch=*.ckpt files, split across GPUs 0 and 1.

set -euo pipefail

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/home/byounggun/DiffusionDrive}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/navsim/exp/bg}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/navsim/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/navsim/dataset/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

RL_DIR="${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_test_assign_plus_random_scenes}"
CKPT_DIR="${CKPT_DIR:-$RL_DIR/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-/data2/byounggun/diffusiondrive_drgrpo_plus_eval_v6_waymo_epoch120_ver2_test_assign_plus_random_latest2}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
GPUS="${GPUS:-0,1}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
WORKER_THREADS="${WORKER_THREADS:-2}"
SUMMARY_CSV="$EVAL_ROOT/summary_pdms.csv"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
fi

mapfile -t CKPTS < <(
    python3 - "$CKPT_DIR" "$NUM_EPOCHS" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
num_epochs = int(sys.argv[2])
items = []
for path in ckpt_dir.glob("grpo-epoch=*.ckpt"):
    match = re.search(r"grpo-epoch=(\d+)\.ckpt$", path.name)
    if match:
        items.append((int(match.group(1)), path))
for _, path in sorted(items)[-num_epochs:]:
    print(path)
PY
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no grpo-epoch=*.ckpt files found in $CKPT_DIR" >&2
    exit 1
fi

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p "$EVAL_ROOT"
rm -f "$EVAL_ROOT"/summary_epoch_*.csv
echo "epoch,checkpoint,output_dir,csv,NC,DAC,TTC,Comfort,EP,PDMS,valid,num_rows" > "$SUMMARY_CSV"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

run_eval_one() {
    local ckpt="$1"
    local gpu="$2"
    local epoch
    epoch="$(basename "$ckpt" | sed -E 's/^grpo-epoch=([0-9]+)\.ckpt$/\1/')"
    printf -v epoch_tag "%03d" "$epoch"

    local out_dir="$EVAL_ROOT/epoch_${epoch_tag}"
    local hydra_ckpt="/tmp/drgrpo_plus_v6_waymo_test_assign_plus_random_epoch_${epoch_tag}.ckpt"
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
        experiment_name="drgrpo_plus_v6_waymo_test_assign_plus_random_eval_epoch_${epoch_tag}" \
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
