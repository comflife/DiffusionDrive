#!/bin/bash
# Eval a v5-style GRPO/GSPO/GRPO+ checkpoint on navtest with PDM scoring.
#
# Defaults to the GRPO+ v5 output dir (matches run_grpo_plus_training_v5.sh).
# Override via env to point at a different RL run:
#
#   # default (grpo_plus v5)
#   ./run_eval_grpo_v5.sh
#
#   # specific algorithm output
#   GRPO_DIR=/data2/byounggun/diffusiondrive_grpo_output_v5      ./run_eval_grpo_v5.sh
#   GRPO_DIR=/data2/byounggun/diffusiondrive_gspo_output_v5      ./run_eval_grpo_v5.sh
#
#   # specific ckpt (overrides GRPO_DIR's auto-pick)
#   GRPO_CKPT=/.../checkpoints/grpo-05.ckpt ./run_eval_grpo_v5.sh
#
#   # different GPUs / output dir
#   GPUS=0,1,2,3 OUTPUT_DIR=/path/eval_dir ./run_eval_grpo_v5.sh
#
# v5 config (mirrored to match the trained model — must agree with the
# settings in run_*_training_v5.sh):
#   - V=2048 step_corners, residual_delta=true, heading_head=false
#   - step_aware_agent=true, ego_cross_attn=true
#   - deformable_bev=FALSE  ← v5 difference vs v6 (flat global MHA)
#   - bev_pos_enc=true, agent_topk=30

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

DEFAULT_GRPO_DIR="${GRPO_DIR:-/data2/byounggun/diffusiondrive_grpo_plus_output_v5}"

# ── Resolve checkpoint ────────────────────────────────────────────────────
if [ -z "${GRPO_CKPT:-}" ]; then
    if [ -f "$DEFAULT_GRPO_DIR/checkpoints/last.ckpt" ]; then
        GRPO_CKPT="$DEFAULT_GRPO_DIR/checkpoints/last.ckpt"
    else
        GRPO_CKPT=$(ls -t "$DEFAULT_GRPO_DIR/checkpoints/"*.ckpt 2>/dev/null | head -1 || true)
    fi
fi

if [ -z "${GRPO_CKPT:-}" ] || [ ! -f "$GRPO_CKPT" ]; then
    echo "ERROR: No GRPO ckpt found." >&2
    echo "       Tried: $DEFAULT_GRPO_DIR/checkpoints/last.ckpt" >&2
    echo "       Set GRPO_CKPT=/path/to/grpo-XX.ckpt or GRPO_DIR=/path/to/output_dir." >&2
    exit 1
fi

# Hydra-safe alias (avoids '=' in checkpoint paths confusing parsing)
SAFE_CKPT="/tmp/grpo_v5_eval_$(date +%s).ckpt"
ln -sfn "$GRPO_CKPT" "$SAFE_CKPT"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
WORKER_THREADS="${WORKER_THREADS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_GRPO_DIR/eval_latest}"

echo "=================================================="
echo "Evaluating v5-base GRPO checkpoint"
echo "=================================================="
echo "Source ckpt  : $GRPO_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Modified     : $(stat -c %y "$GRPO_CKPT" 2>/dev/null)"
echo "GPUs         : $GPUS"
echo "Worker thrds : $WORKER_THREADS"
echo "Output       : $OUTPUT_DIR"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

CUDA_VISIBLE_DEVICES="$GPUS" \
RAY_CUDA_VISIBLE_DEVICES="$GPUS" \
python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$SAFE_CKPT" \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=false \
    agent.config.ar_num_modes=1 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=false \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=false \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.agent_topk=30 \
    agent.config.freeze_pretrained_trunk=false \
    worker=ray_distributed \
    worker.threads_per_node="$WORKER_THREADS" \
    metric_cache_path=/data2/byounggun/metric_cache \
    experiment_name=diffusiondrive_grpo_v5_eval \
    output_dir="$OUTPUT_DIR" \
    "$@"

echo "=================================================="
echo "Eval complete. Output: $OUTPUT_DIR"
echo "=================================================="
