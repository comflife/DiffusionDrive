#!/bin/bash
# Dr. GRPO+ (improved) fine-tuning on top of the v6_waymo SFT checkpoint at epoch 120.
# Improvements applied to the dr_grpo_plus algorithm:
#   - pg_tok uses MEAN over T (fixes scale mismatch with pg_seq)
#   - Reward-weighted centroid for divergence (instead of simple mean)
#   - Temporal decay on token weights (early tokens matter more causally)
#   - Adaptive alpha: ramps from 0.3 -> 0.7 over 80% of training
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
#   bash train_eval/run_drgrpo_plus_training_v6_waymo.sh
#   DEVICES=2 ACCUMULATE_GRAD_BATCHES=8 bash train_eval/run_drgrpo_plus_training_v6_waymo.sh
#   BASE_CKPT=/path/to/ckpt.ckpt bash train_eval/run_drgrpo_plus_training_v6_waymo.sh

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

SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# Tunables. Defaults mirror the existing v6 NoRD recipe.
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LR="${LR:-5e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120}"

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GRPO+ Fine-tuning (v6_waymo epoch-120 base)"
echo "=================================================="
echo "Algorithm    : dr_grpo_plus  (improved)"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
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
echo "Epochs       : $MAX_EPOCHS"
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
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120 \
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
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_plus_v6_waymo_ep120_g${GROUP_SIZE}_t${TEMPERATURE}_lr${LR}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GRPO+ Training Complete (v6_waymo epoch-120)! Output: $OUTPUT_DIR"
echo "=================================================="
