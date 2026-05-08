#!/bin/bash
# Evaluate DiffusionDrive-AR joint_v9_scratch milestone checkpoints and print PDMS.
# v9_scratch = v9 architecture (bev_first ordering, single-call agent, deformable
#       BEV ON, ego cross-attn ON, BEV pos enc ON) trained from scratch — NO 88.1
#       PDMS DiffusionDrive checkpoint warm-start (only ImageNet ResNet34).
# Defaults to epoch 80, 90, ..., 150. Missing checkpoints fail the script unless
# SKIP_MISSING=1 is set.

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

CKPT_DIR="${CKPT_DIR:-/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v9_scratch/checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-/data2/byounggun/diffusiondrive_ar_output/eval_step_corner_v2048_joint_v9_scratch_milestones}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
START_EPOCH="${START_EPOCH:-60}"
END_EPOCH="${END_EPOCH:-150}"
EPOCH_STEP="${EPOCH_STEP:-10}"
WORKER_THREADS="${WORKER_THREADS:-2}"
SKIP_MISSING="${SKIP_MISSING:-0}"
PARALLEL_EVAL="${PARALLEL_EVAL:-1}"
SUMMARY_CSV="$EVAL_ROOT/summary_pdms.csv"

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p "$EVAL_ROOT"
rm -f "$EVAL_ROOT"/summary_epoch_*.csv
echo "epoch,checkpoint,output_dir,csv,score,valid,num_rows" > "$SUMMARY_CSV"

echo "=================================================="
echo "Evaluating DiffusionDrive-AR joint_v9_scratch milestones (bev_first ordering, scratch)"
echo "Checkpoint dir : $CKPT_DIR"
echo "Epochs         : $START_EPOCH..$END_EPOCH step $EPOCH_STEP"
echo "GPUs           : $GPUS"
echo "Parallel eval  : $PARALLEL_EVAL"
echo "Metric cache   : $METRIC_CACHE_PATH"
echo "Eval root      : $EVAL_ROOT"
echo "Summary CSV    : $SUMMARY_CSV"
echo "=================================================="

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no GPUs configured. Set GPUS=0,1,2,3"
    exit 1
fi

run_eval_one() {
    local epoch="$1"
    local gpu="$2"
    printf -v epoch_tag "%03d" "$epoch"
    local ckpt="$CKPT_DIR/milestone_epoch_${epoch_tag}.ckpt"
    local out_dir="$EVAL_ROOT/epoch_${epoch_tag}"
    local hydra_ckpt="/tmp/diffusiondrive_ar_step_corner_v2048_joint_v9_scratch_epoch_${epoch_tag}.ckpt"
    local epoch_summary="$EVAL_ROOT/summary_epoch_${epoch_tag}.csv"

    if [ ! -f "$ckpt" ]; then
        echo "[epoch $epoch_tag] Missing checkpoint: $ckpt"
        if [ "$SKIP_MISSING" = "1" ]; then
            return 0
        fi
        return 1
    fi

    ln -sfn "$ckpt" "$hydra_ckpt"

    echo "--------------------------------------------------"
    echo "[epoch $epoch_tag] Evaluating"
    echo "Source ckpt : $ckpt"
    echo "Hydra alias : $hydra_ckpt"
    echo "GPU         : $gpu"
    echo "Output      : $out_dir"
    echo "--------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$gpu" \
    RAY_CUDA_VISIBLE_DEVICES="$gpu" \
    python3 -m navsim.planning.script.run_pdm_score \
        train_test_split=navtest \
        agent=diffusiondrive_ar_agent \
        "agent.checkpoint_path=$hydra_ckpt" \
        agent.config.ego_vocab_size=2048 \
        agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
        agent.config.ar_codebook_mode=step_corners \
        agent.config.ar_teacher_forcing=false \
        agent.config.ar_num_modes=1 \
        agent.config.ar_token_loss_weight=1.0 \
        agent.config.ar_traj_loss_weight=8.0 \
        agent.config.ar_heading_loss_weight=2.0 \
        agent.config.ar_use_residual_delta=true \
        agent.config.ar_use_heading_head="${HEADING_HEAD:-false}" \
        agent.config.ar_step_aware_agent=false \
        agent.config.ar_use_ego_cross_attn=true \
        agent.config.ar_use_deformable_bev=true \
        agent.config.ar_use_bev_pos_enc=true \
        agent.config.ar_attn_stack_ordering=bev_first \
        agent.config.freeze_pretrained_trunk=false \
        worker=ray_distributed \
        worker.threads_per_node="$WORKER_THREADS" \
        metric_cache_path="$METRIC_CACHE_PATH" \
        experiment_name="diffusiondrive_ar_step_corner_v2048_joint_v9_scratch_eval_epoch_${epoch_tag}" \
        output_dir="$out_dir"

    local latest_csv
    latest_csv="$(find "$out_dir" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [ -z "$latest_csv" ]; then
        echo "[epoch $epoch_tag] ERROR: no PDMS csv found in $out_dir"
        return 1
    fi

    python3 - "$epoch_tag" "$ckpt" "$out_dir" "$latest_csv" "$epoch_summary" <<'PY'
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
    writer.writerow([epoch, ckpt, out_dir, csv_path, f"{score:.8f}", valid, num_rows])

print(f"[epoch {epoch}] PDMS score={score:.8f} valid={valid} scenarios={num_rows}")
PY
}

running_jobs=0
gpu_index=0

for epoch in $(seq "$START_EPOCH" "$EPOCH_STEP" "$END_EPOCH"); do
    gpu="${GPU_LIST[$gpu_index]}"
    gpu_index=$(( (gpu_index + 1) % ${#GPU_LIST[@]} ))

    if [ "$PARALLEL_EVAL" = "1" ]; then
        run_eval_one "$epoch" "$gpu" &
        running_jobs=$((running_jobs + 1))
        if [ "$running_jobs" -ge "${#GPU_LIST[@]}" ]; then
            wait -n
            running_jobs=$((running_jobs - 1))
        fi
    else
        run_eval_one "$epoch" "$gpu"
    fi
done

wait

for summary in $(find "$EVAL_ROOT" -maxdepth 1 -type f -name 'summary_epoch_*.csv' | sort); do
    cat "$summary" >> "$SUMMARY_CSV"
done

echo "=================================================="
echo "PDMS summary"
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo "=================================================="
