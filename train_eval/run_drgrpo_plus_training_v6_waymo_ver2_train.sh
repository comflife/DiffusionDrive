#!/bin/bash
# Dr. GRPO+ v2 (Aggressive) fine-tuning on v6_waymo epoch-120 SFT, full VAL split.
#
# Uses all navtrain val-log scenes (~2.6k). Assignment/random wrappers override
# tokens to smaller subsets from the same metric cache.
#
# Usage:
#   bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_train.sh
#   DEVICES=1 bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_train.sh
#   Run four experiments in parallel (one GPU each):
#     CUDA_VISIBLE_DEVICES=0 bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_train.sh
#     CUDA_VISIBLE_DEVICES=1 bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_train_assignments.sh
#     CUDA_VISIBLE_DEVICES=2 bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_train_assign_plus_random_scenes.sh
#     CUDA_VISIBLE_DEVICES=3 bash train_eval/run_drgrpo_plus_training_v6_waymo_ver2_test_assign_plus_random_scenes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_val_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"

VAL_TOKEN_OUTPUT="$(all_val_scene_tokens_and_count)"
VAL_SCENE_COUNT="${VAL_TOKEN_OUTPUT%% *}"
VAL_SCENE_TOKENS="${VAL_TOKEN_OUTPUT#* }"

echo "Using full val-split scene tokens (count: $VAL_SCENE_COUNT)"

DEFAULT_V6_WAYMO_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V6_WAYMO_DIR/milestone_epoch_120.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v6_waymo epoch-120 ckpt not found at: $BASE_CKPT" >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_ver2_val_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Aggressive v2 tunables ───────────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-12}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.25}"
LR="${LR:-1e-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
DEVICES="${DEVICES:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_val_full}"

# Simple resume support (auto-continue if last.ckpt exists)
RESUME_CKPT="${RESUME_CKPT-__auto__}"
if [ "$RESUME_CKPT" = "__auto__" ]; then
    if [ -f "$OUTPUT_DIR/checkpoints/last.ckpt" ]; then
        RESUME_CKPT="$OUTPUT_DIR/checkpoints/last.ckpt"
    else
        RESUME_CKPT=""
    fi
elif [ "$RESUME_CKPT" = "none" ] || [ "$RESUME_CKPT" = "NONE" ] || [ "$RESUME_CKPT" = "0" ]; then
    RESUME_CKPT=""
fi

RESUME_ARGS=()
RESUME_DISPLAY="none"
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "ERROR: RESUME_CKPT not found at: $RESUME_CKPT" >&2
        exit 1
    fi
    RESUME_SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_ver2_val_resume_$(date +%s).ckpt"
    ln -sfn "$RESUME_CKPT" "$RESUME_SAFE_CKPT"
    RESUME_ARGS=(++resume_ckpt_path="$RESUME_SAFE_CKPT")
    RESUME_DISPLAY="$RESUME_CKPT"
fi

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GRPO+ v2 (Aggressive) - v6_waymo epoch-120 VAL"
echo "=================================================="
echo "Algorithm    : dr_grpo_plus (v2 aggressive)"
echo "Base ckpt    : $BASE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "LR           : $LR"
echo "Clip eps     : $CLIP_EPS"
echo "Accum grads  : $ACCUMULATE_GRAD_BATCHES"
echo "Eff. batch   : $EFFECTIVE_BATCH"
echo "Max epochs   : $MAX_EPOCHS"
echo "Output       : $OUTPUT_DIR"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtrain \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$VAL_SCENE_TOKENS" \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path="$METRIC_CACHE_PATH" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    output_dir="$OUTPUT_DIR" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_val_full \
    ++config.ego_vocab_size=2048 \
    ++config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy \
    ++config.ar_codebook_mode=step_corners \
    ++config.ar_use_residual_delta=true \
    ++config.ar_use_heading_head=true \
    ++config.ar_step_aware_agent=true \
    ++config.ar_use_ego_cross_attn=true \
    ++config.ar_use_deformable_bev=true \
    ++config.ar_use_bev_pos_enc=true \
    ++config.agent_topk=30 \
    ++trainer.params.max_epochs="$MAX_EPOCHS" \
    ++trainer.params.devices="$DEVICES" \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++trainer.params.gradient_clip_val=1.0 \
    ++trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    ++batch_size=1 \
    ++num_workers=0 \
    ++algorithm=dr_grpo_plus \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    "${RESUME_ARGS[@]}" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_plus_v6_waymo_ep120_ver2_val_full_g${GROUP_SIZE}_lr${LR}_clip${CLIP_EPS}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GRPO+ v2 VAL (full) Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
