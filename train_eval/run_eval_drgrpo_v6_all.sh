#!/bin/bash
# Bulk-eval EVERY dr_grpo v6 checkpoint on navtest in parallel.
#
# Layout: round-robin assign each ckpt to one GPU, run NUM_GPUS evals in
# parallel, throttle so at most NUM_GPUS evals run at any time. When all
# are done, aggregate per-ckpt "Final average score" into a sorted summary
# table (also written to a file).
#
# Each parallel eval:
#   - uses CUDA_VISIBLE_DEVICES=<one_gpu>
#   - uses an isolated TMPDIR so per-driver ray clusters don't fight
#   - logs to its own file under $LOG_ROOT
#   - writes per-scenario CSV under $RESULT_ROOT/<ckpt_name>/
#
# Usage:
#   ./run_eval_drgrpo_v6_all.sh
#   GPUS=0,1,2,3 ./run_eval_drgrpo_v6_all.sh             # explicit GPU set
#   DRGRPO_DIR=/path/to/output_v6 ./run_eval_drgrpo_v6_all.sh
#   INCLUDE_BASE=1 ./run_eval_drgrpo_v6_all.sh           # also eval the SFT base
#   STEP_MIN=300 ./run_eval_drgrpo_v6_all.sh             # skip ckpts before step 300
#   WORKER_THREADS=2 ./run_eval_drgrpo_v6_all.sh         # ray workers per eval
#
# v6 config (must match training):
#   - V=2048 step_corners, residual_delta=true, heading_head=false
#   - step_aware_agent=true, ego_cross_attn=true
#   - deformable_bev=TRUE   (v6 difference vs v5)
#   - bev_pos_enc=true, agent_topk=30

set -euo pipefail

# ── Env ──────────────────────────────────────────────────────────────────
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── Config ───────────────────────────────────────────────────────────────
DRGRPO_DIR="${DRGRPO_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6}"
CKPT_DIR="$DRGRPO_DIR/checkpoints"
GPUS_LIST="${GPUS:-0,1,2,3}"
WORKER_THREADS="${WORKER_THREADS:-2}"
RESULT_ROOT="${RESULT_ROOT:-$DRGRPO_DIR/eval_all}"
LOG_ROOT="${LOG_ROOT:-$DRGRPO_DIR/eval_all_logs}"
SUMMARY_FILE="${SUMMARY_FILE:-$DRGRPO_DIR/eval_all_summary.txt}"
INCLUDE_BASE="${INCLUDE_BASE:-0}"     # 1 → also eval the SFT base for reference
STEP_MIN="${STEP_MIN:-0}"             # skip ckpts whose step < STEP_MIN
METRIC_CACHE="${METRIC_CACHE:-/data2/byounggun/metric_cache}"

# ── Validation ───────────────────────────────────────────────────────────
[ -d "$CKPT_DIR" ] || { echo "ERROR: $CKPT_DIR not found" >&2; exit 1; }
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

# ── Discover ckpts ───────────────────────────────────────────────────────
# Match grpo-step=*.ckpt; sort -V handles zero-padded step numbers naturally.
# `|| true` guards against empty glob under set -e + pipefail.
mapfile -t ALL_CKPTS < <(ls -1 "$CKPT_DIR"/grpo-step=*.ckpt 2>/dev/null | sort -V || true)

# Filter by STEP_MIN
CKPTS=()
for C in "${ALL_CKPTS[@]}"; do
    STEP=$(basename "$C" .ckpt | grep -oE '[0-9]+' | tail -1)
    if [ -n "$STEP" ] && [ "$STEP" -ge "$STEP_MIN" ]; then
        CKPTS+=("$C")
    fi
done

# Optional: prepend the SFT base for reference comparison
if [ "$INCLUDE_BASE" = "1" ]; then
    BASE_DEFAULT="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6/checkpoints/manual_epoch_068.ckpt"
    BASE="${BASE_CKPT:-$BASE_DEFAULT}"
    if [ -f "$BASE" ]; then
        CKPTS=("$BASE" "${CKPTS[@]}")
        echo "Including base ckpt for reference: $BASE"
    else
        echo "WARN: INCLUDE_BASE=1 but base ckpt not found: $BASE" >&2
    fi
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: no ckpts to eval (looked under $CKPT_DIR matching grpo-step=*.ckpt, STEP_MIN=$STEP_MIN)" >&2
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=================================================="
echo "Bulk eval dr_grpo v6 ckpts"
echo "=================================================="
echo "Source dir    : $DRGRPO_DIR"
echo "Ckpts to eval : ${#CKPTS[@]}  (STEP_MIN=$STEP_MIN, INCLUDE_BASE=$INCLUDE_BASE)"
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

# ── Parallel launch with throttling ──────────────────────────────────────
cd "$NAVSIM_DEVKIT_ROOT"
PIDS=()

run_eval_for() {
    local CKPT="$1"
    local GPU="$2"
    local NAME_SAFE
    NAME_SAFE=$(basename "$CKPT" .ckpt | tr '=' '_')   # filesystem-safe
    local OUT="$RESULT_ROOT/$NAME_SAFE"
    local LOG="$LOG_ROOT/${NAME_SAFE}.log"
    local SAFE="/tmp/drgrpo_v6_evalall_${NAME_SAFE}_$$.ckpt"

    ln -sfn "$CKPT" "$SAFE"
    mkdir -p "$OUT"

    # Isolated TMPDIR so per-driver ray clusters don't fight over /tmp/ray
    local RAY_TMP="/data2/byounggun/ray_tmp_evalall/${NAME_SAFE}_$$"
    mkdir -p "$RAY_TMP"

    {
        echo "=== $(date) — eval start ==="
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
        experiment_name="drgrpo_v6_evalall_${NAME_SAFE}" \
        output_dir="$OUT" \
        >> "$LOG" 2>&1

    local EXIT=$?
    echo "=== $(date) — eval end (exit $EXIT) ===" >> "$LOG"
    rm -rf "$RAY_TMP"
    rm -f "$SAFE"
    return $EXIT
}

i=0
for CKPT in "${CKPTS[@]}"; do
    GPU="${GPU_ARRAY[$((i % NUM_GPUS))]}"

    # Throttle: wait until at least one GPU slot is free
    while [ "$(jobs -r -p | wc -l)" -ge "$NUM_GPUS" ]; do
        sleep 10
    done

    NAME=$(basename "$CKPT")
    echo "[$(date '+%H:%M:%S')] LAUNCH gpu=$GPU  $((i+1))/${#CKPTS[@]}  $NAME"
    ( run_eval_for "$CKPT" "$GPU" ) &
    PIDS+=($!)
    i=$((i+1))
    # Tiny stagger to avoid simultaneous ray init thrashing
    sleep 3
done

echo "All ${#PIDS[@]} evals launched. Waiting for completion..."
FAIL_COUNT=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done
echo "All done.  failed=${FAIL_COUNT}/${#PIDS[@]}"

# ── Aggregate summary ────────────────────────────────────────────────────
{
    echo "=================================================="
    echo "Bulk eval summary — $(date)"
    echo "Source dir : $DRGRPO_DIR"
    echo "Total ckpts: ${#CKPTS[@]}    failed=${FAIL_COUNT}"
    echo "=================================================="
    printf "%-44s  %-8s  %-12s  %s\n" "ckpt" "step" "PDMS" "log"
    printf "%-44s  %-8s  %-12s  %s\n" "----" "----" "----" "---"
    for CKPT in "${CKPTS[@]}"; do
        NAME_RAW=$(basename "$CKPT" .ckpt)
        NAME_SAFE=$(basename "$CKPT" .ckpt | tr '=' '_')
        LOG="$LOG_ROOT/${NAME_SAFE}.log"
        # Final score line looks like:
        #   Final average score of valid results: 0.7999389831566607.
        SCORE=$(grep -h "Final average score of valid results" "$LOG" 2>/dev/null \
                | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        STEP=$(echo "$NAME_RAW" | grep -oE '[0-9]+' | tail -1)
        printf "%-44s  %-8s  %-12s  %s\n" "$NAME_RAW" "${STEP:-?}" "${SCORE:-FAIL}" "$LOG"
    done
} | tee "$SUMMARY_FILE"

echo
echo "Summary saved to: $SUMMARY_FILE"
echo "Per-ckpt logs   : $LOG_ROOT"
echo "Per-ckpt CSVs   : $RESULT_ROOT/<ckpt_name>/"
