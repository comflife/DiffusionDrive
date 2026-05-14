#!/bin/bash
# Eval a v6-base Dr. GRPO+ checkpoint on navtest with PDM scoring.
#
# Matches run_drgrpo_plus_training_v6.sh:
#   - algorithm: dr_grpo_plus
#   - output: /data2/byounggun/diffusiondrive_dr_grpo_plus_output_v6
#   - V=2048 step_corners, residual_delta=true, heading_head=false
#   - step_aware_agent=true, ego_cross_attn=true
#   - deformable_bev=true, bev_pos_enc=true, agent_topk=30
#
# Override via env:
#   DRGRPO_PLUS_DIR=/data2/byounggun/diffusiondrive_dr_grpo_plus_output_v6 \
#       ./train_eval/run_eval_drgrpo_plus_v6.sh
#
#   DRGRPO_PLUS_CKPT=/data2/byounggun/diffusiondrive_dr_grpo_plus_output_v6/checkpoints/grpo-epoch=09.ckpt \
#       OUTPUT_DIR=/data2/byounggun/diffusiondrive_dr_grpo_plus_output_v6/eval_epoch_09 \
#       GPUS=0 \
#       ./train_eval/run_eval_drgrpo_plus_v6.sh

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR="${TMPDIR:-/data2/byounggun/ray_tmp}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

DEFAULT_DRGRPO_PLUS_DIR="${DRGRPO_PLUS_DIR:-/data2/byounggun/diffusiondrive_dr_grpo_plus_output_v6}"

# Resolve checkpoint. Prefer Lightning's last.ckpt, otherwise newest ckpt.
USER_SUPPLIED_CKPT=0
if [ -n "${DRGRPO_PLUS_CKPT:-}" ]; then
    USER_SUPPLIED_CKPT=1
else
    if [ -f "$DEFAULT_DRGRPO_PLUS_DIR/checkpoints/last.ckpt" ]; then
        DRGRPO_PLUS_CKPT="$DEFAULT_DRGRPO_PLUS_DIR/checkpoints/last.ckpt"
    else
        DRGRPO_PLUS_CKPT=$(ls -t "$DEFAULT_DRGRPO_PLUS_DIR/checkpoints/"*.ckpt 2>/dev/null | head -1 || true)
    fi
fi

if [ -z "${DRGRPO_PLUS_CKPT:-}" ] || [ ! -f "$DRGRPO_PLUS_CKPT" ]; then
    echo "ERROR: Dr. GRPO+ ckpt not found." >&2
    if [ "$USER_SUPPLIED_CKPT" = "1" ]; then
        echo "       Path you specified does not exist:" >&2
        echo "         $DRGRPO_PLUS_CKPT" >&2
        if [ -d "$(dirname "$DRGRPO_PLUS_CKPT")" ]; then
            echo "       Files actually present in $(dirname "$DRGRPO_PLUS_CKPT"):" >&2
            ls -1 "$(dirname "$DRGRPO_PLUS_CKPT")"/*.ckpt 2>&1 | sed 's/^/         /' >&2
        fi
    else
        echo "       Auto-pick failed under: $DEFAULT_DRGRPO_PLUS_DIR/checkpoints/" >&2
        echo "       Set DRGRPO_PLUS_CKPT=/path/to/grpo-epoch=XX.ckpt or DRGRPO_PLUS_DIR=/path/to/output_dir." >&2
    fi
    exit 1
fi

# Hydra-safe alias, since checkpoint filenames can contain '='.
SAFE_CKPT="/tmp/drgrpo_plus_v6_eval_$(date +%s).ckpt"
ln -sfn "$DRGRPO_PLUS_CKPT" "$SAFE_CKPT"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
WORKER_THREADS="${WORKER_THREADS:-2}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_DRGRPO_PLUS_DIR/eval_latest}"

echo "=================================================="
echo "Evaluating v6-base Dr. GRPO+ checkpoint"
echo "=================================================="
echo "Source ckpt  : $DRGRPO_PLUS_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Modified     : $(stat -c %y "$DRGRPO_PLUS_CKPT" 2>/dev/null)"
echo "GPUs         : $GPUS"
echo "Worker thrds : $WORKER_THREADS"
echo "Metric cache : $METRIC_CACHE_PATH"
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
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.agent_topk=30 \
    agent.config.freeze_pretrained_trunk=false \
    worker=ray_distributed \
    worker.threads_per_node="$WORKER_THREADS" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    experiment_name=diffusiondrive_drgrpo_plus_v6_eval \
    output_dir="$OUTPUT_DIR" \
    "$@"

echo "=================================================="
echo "Eval complete. Output: $OUTPUT_DIR"
echo "=================================================="
