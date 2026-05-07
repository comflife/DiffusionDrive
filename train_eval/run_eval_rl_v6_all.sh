#!/bin/bash
# Bulk-eval every v6 RL checkpoint on navtest and summarize PDMS.
#
# Intended to be called by algorithm-specific wrappers, but can also be used
# directly:
#   RL_NAME=grpo RL_DIR=/path/to/output ./train_eval/run_eval_rl_v6_all.sh

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

RL_NAME="${RL_NAME:-rl}"
RL_DIR="${RL_DIR:?Set RL_DIR to the RL output directory}"
CKPT_DIR="$RL_DIR/checkpoints"
GPUS_LIST="${GPUS:-0,1,2,3}"
WORKER_THREADS="${WORKER_THREADS:-2}"
RESULT_ROOT="${RESULT_ROOT:-$RL_DIR/eval_all}"
LOG_ROOT="${LOG_ROOT:-$RL_DIR/eval_all_logs}"
SUMMARY_FILE="${SUMMARY_FILE:-$RL_DIR/eval_all_summary.txt}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"
BASE_DEFAULT="${BASE_CKPT:-/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6/checkpoints/milestone_epoch_080.ckpt}"
STEP_MIN="${STEP_MIN:-0}"
EPOCH_MIN="${EPOCH_MIN:-0}"
METRIC_CACHE="${METRIC_CACHE:-/data2/byounggun/metric_cache}"

[ -d "$CKPT_DIR" ] || { echo "ERROR: $CKPT_DIR not found" >&2; exit 1; }
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

# Match both old step-based and new epoch-based Lightning checkpoint names.
mapfile -t ALL_CKPTS < <(
    find "$CKPT_DIR" -maxdepth 1 -type f \
        \( -name 'grpo-step=*.ckpt' -o -name 'grpo-epoch=*.ckpt' -o -name 'epoch=*.ckpt' \) \
        | sort -V
)

CKPTS=()
for C in "${ALL_CKPTS[@]}"; do
    NAME="$(basename "$C" .ckpt)"
    NUM="$(echo "$NAME" | grep -oE '[0-9]+' | tail -1 || true)"
    if [[ "$NAME" == *step* ]]; then
        [ -n "$NUM" ] && [ "$NUM" -ge "$STEP_MIN" ] && CKPTS+=("$C")
    elif [[ "$NAME" == *epoch* ]]; then
        [ -n "$NUM" ] && [ "$NUM" -ge "$EPOCH_MIN" ] && CKPTS+=("$C")
    else
        CKPTS+=("$C")
    fi
done

if [ "$INCLUDE_BASE" = "1" ]; then
    if [ -f "$BASE_DEFAULT" ]; then
        CKPTS=("$BASE_DEFAULT" "${CKPTS[@]}")
        echo "Including base ckpt for reference: $BASE_DEFAULT"
    else
        echo "WARN: INCLUDE_BASE=1 but base ckpt not found: $BASE_DEFAULT" >&2
    fi
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: no ckpts to eval under $CKPT_DIR" >&2
    echo "       looked for grpo-step=*.ckpt, grpo-epoch=*.ckpt, epoch=*.ckpt" >&2
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=================================================="
echo "Bulk eval $RL_NAME v6 ckpts"
echo "=================================================="
echo "Source dir    : $RL_DIR"
echo "Ckpts to eval : ${#CKPTS[@]}  (STEP_MIN=$STEP_MIN, EPOCH_MIN=$EPOCH_MIN, INCLUDE_BASE=$INCLUDE_BASE)"
echo "Parallel GPUs : $NUM_GPUS  (${GPU_ARRAY[*]})"
echo "Worker thrds  : $WORKER_THREADS per eval"
echo "Result root   : $RESULT_ROOT"
echo "Log root      : $LOG_ROOT"
echo "Summary file  : $SUMMARY_FILE"
echo "=================================================="
echo "Eval queue:"
for i in "${!CKPTS[@]}"; do
    G="${GPU_ARRAY[$((i % NUM_GPUS))]}"
    echo "  [$((i+1))/${#CKPTS[@]}] gpu=$G  $(basename "${CKPTS[$i]}")"
done
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"
PIDS=()

run_eval_for() {
    local CKPT="$1"
    local GPU="$2"
    local NAME_SAFE
    NAME_SAFE=$(basename "$CKPT" .ckpt | tr '=' '_')
    local OUT="$RESULT_ROOT/$NAME_SAFE"
    local LOG="$LOG_ROOT/${NAME_SAFE}.log"
    local SAFE="/tmp/${RL_NAME}_v6_evalall_${NAME_SAFE}_$$.ckpt"

    ln -sfn "$CKPT" "$SAFE"
    mkdir -p "$OUT"

    local ID
    ID=$(basename "$CKPT" .ckpt | grep -oE '[0-9]+' | tail -1 || true)
    local RAY_TMP="/tmp/${RL_NAME:0:3}v6_${ID:-x}_$$"
    mkdir -p "$RAY_TMP"

    {
        echo "=== $(date) eval start ==="
        echo "ckpt   : $CKPT"
        echo "alias  : $SAFE"
        echo "gpu    : $GPU"
        echo "tmpdir : $RAY_TMP"
        echo "outdir : $OUT"
        echo "==="
    } > "$LOG"

    TMPDIR="$RAY_TMP" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    RAY_CUDA_VISIBLE_DEVICES="$GPU" \
    python3 -m navsim.planning.script.run_pdm_score \
        train_test_split=navtest \
        agent=diffusiondrive_ar_agent \
        "agent.checkpoint_path=$SAFE" \
        agent.config.ego_vocab_size=2048 \
        agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
        agent.config.ar_codebook_mode=step_corners \
        agent.config.ar_teacher_forcing=false \
        agent.config.ar_num_modes=1 \
        agent.config.ar_use_residual_delta=true \
        agent.config.ar_use_heading_head=false \
        agent.config.ar_step_aware_agent=true \
        agent.config.ar_use_ego_cross_attn=true \
        agent.config.ar_use_deformable_bev=true \
        agent.config.ar_use_bev_pos_enc=true \
        agent.config.agent_topk=30 \
        agent.config.freeze_pretrained_trunk=false \
        worker=ray_distributed \
        worker.threads_per_node="$WORKER_THREADS" \
        metric_cache_path="$METRIC_CACHE" \
        experiment_name="${RL_NAME}_v6_evalall_${NAME_SAFE}" \
        output_dir="$OUT" \
        >> "$LOG" 2>&1

    local EXIT=$?
    echo "=== $(date) eval end (exit $EXIT) ===" >> "$LOG"
    rm -rf "$RAY_TMP"
    rm -f "$SAFE"
    return $EXIT
}

i=0
for CKPT in "${CKPTS[@]}"; do
    GPU="${GPU_ARRAY[$((i % NUM_GPUS))]}"
    while [ "$(jobs -r -p | wc -l)" -ge "$NUM_GPUS" ]; do
        sleep 10
    done

    NAME=$(basename "$CKPT")
    echo "[$(date '+%H:%M:%S')] LAUNCH gpu=$GPU  $((i+1))/${#CKPTS[@]}  $NAME"
    ( run_eval_for "$CKPT" "$GPU" ) &
    PIDS+=($!)
    i=$((i+1))
    sleep 3
done

echo "All ${#PIDS[@]} evals launched. Waiting for completion..."
FAIL_COUNT=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done
echo "All done. failed=${FAIL_COUNT}/${#PIDS[@]}"

{
    echo "=================================================="
    echo "Bulk eval summary - $(date)"
    echo "Source dir : $RL_DIR"
    echo "Total ckpts: ${#CKPTS[@]}    failed=${FAIL_COUNT}"
    echo "=================================================="
    printf "%-44s  %-8s  %-12s  %s\n" "ckpt" "id" "PDMS" "log"
    printf "%-44s  %-8s  %-12s  %s\n" "----" "--" "----" "---"
    for CKPT in "${CKPTS[@]}"; do
        NAME_RAW=$(basename "$CKPT" .ckpt)
        NAME_SAFE=$(basename "$CKPT" .ckpt | tr '=' '_')
        LOG="$LOG_ROOT/${NAME_SAFE}.log"
        SCORE=$(grep -h "Final average score of valid results" "$LOG" 2>/dev/null \
                | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        ID=$(echo "$NAME_RAW" | grep -oE '[0-9]+' | tail -1)
        printf "%-44s  %-8s  %-12s  %s\n" "$NAME_RAW" "${ID:-?}" "${SCORE:-FAIL}" "$LOG"
    done
} | tee "$SUMMARY_FILE"

echo
echo "Summary saved to: $SUMMARY_FILE"
echo "Per-ckpt logs   : $LOG_ROOT"
echo "Per-ckpt CSVs   : $RESULT_ROOT/<ckpt_name>/"
