#!/bin/bash
# Dr. GRPO fine-tuning on top of the v6_waymo SFT checkpoint at epoch 120,
# using the same NoRD-style recipe as run_drgrpo_training_v6.sh.
#
# Base:
#   /data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints/milestone_epoch_120.ckpt
#
# Matches run_training_ar_step_corner_v2048_joint_v6_waymo.sh:
#   - V=2048 step_corners, Waymo-derived DiffusionDrive codebook
#   - residual_delta=true, heading_head=true
#   - step_aware_agent=true, ego_cross_attn=true
#   - deformable_bev=true, bev_pos_enc=true, agent_topk=30
#
# Usage:
#   bash train_eval/run_drgrpo_training_v6_waymo_120.sh
#   DEVICES=2 ACCUMULATE_GRAD_BATCHES=8 bash train_eval/run_drgrpo_training_v6_waymo_120.sh
#   BASE_CKPT=/path/to/ckpt.ckpt bash train_eval/run_drgrpo_training_v6_waymo_120.sh
#   RESUME_CKPT=none bash train_eval/run_drgrpo_training_v6_waymo_120.sh  # fresh run

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

# Pick a base checkpoint.
DEFAULT_V6_WAYMO_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V6_WAYMO_DIR/milestone_epoch_120.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v6_waymo epoch-120 ckpt not found at: $BASE_CKPT" >&2
        echo "       Set BASE_CKPT=/path/to/v6_waymo.ckpt to override." >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/drgrpo_v6_waymo_epoch120_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# Tunables. Defaults mirror the existing v6 NoRD recipe.
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LR="${LR:-5e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6_waymo_epoch120}"

# If this output already has a Lightning checkpoint, continue it by default.
# Set RESUME_CKPT=none or RESUME_CKPT= to force a fresh run from BASE_CKPT.
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
    RESUME_SAFE_CKPT="/tmp/drgrpo_v6_waymo_epoch120_resume_$(date +%s).ckpt"
    ln -sfn "$RESUME_CKPT" "$RESUME_SAFE_CKPT"
    RESUME_ARGS=(++resume_ckpt_path="$RESUME_SAFE_CKPT")
    RESUME_DISPLAY="$RESUME_CKPT"
fi

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GRPO Fine-tuning (v6_waymo epoch-120 base)"
echo "=================================================="
echo "Algorithm    : dr_grpo"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Resume ckpt  : $RESUME_DISPLAY"
echo "Codebook     : codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy"
echo "Heading head : true"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps     : $CLIP_EPS"
echo "LR           : $LR"
echo "Devices      : $DEVICES"
echo "Accum grads  : $ACCUMULATE_GRAD_BATCHES"
echo "Eff. batch   : $EFFECTIVE_BATCH trajectories / opt step"
echo "Target epochs: $MAX_EPOCHS"
echo "Save every   : 1 epoch  (keep last $KEEP_LAST_N)"
echo "Output       : $OUTPUT_DIR"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtest \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    output_dir="$OUTPUT_DIR" \
    ++experiment_name=diffusiondrive_ar_drgrpo_v6_waymo_epoch120 \
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
    ++algorithm=dr_grpo \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    "${RESUME_ARGS[@]}" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_v6_waymo_ep120_g${GROUP_SIZE}_t${TEMPERATURE}_lr${LR}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GRPO Training Complete (v6_waymo epoch-120)! Output: $OUTPUT_DIR"
echo "=================================================="
