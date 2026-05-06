#!/bin/bash
# GRPO+ : GSPO sequence-level importance ratio + per-token advantage
# (group-divergence-weighted) on top of the v5 SFT ckpt.
#
# v5 ≠ v6: deformable BEV OFF, 2D BEV pos enc ON. Everything else mirrors v6.
#
# α (token_attention_alpha) = blend between GSPO and per-token attention:
#   α = 0    → pure GSPO
#   α = 0.5  → balanced (recommended)
#   α = 1    → pure token-attention (no flat sequence baseline)
#
# Usage:
#   ./run_grpo_plus_training_v5.sh
#   ALPHA=0.7 ./run_grpo_plus_training_v5.sh
#   BASE_CKPT=/path/to/v5_milestone_epoch_080.ckpt ./run_grpo_plus_training_v5.sh

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

# ── Pick a base checkpoint ────────────────────────────────────────────────
DEFAULT_V5_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v5/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    LATEST_MS=$(ls -1 "$DEFAULT_V5_DIR"/milestone_epoch_*.ckpt 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_MS" ]; then
        BASE_CKPT="$LATEST_MS"
    elif [ -f "$DEFAULT_V5_DIR/last.ckpt" ]; then
        BASE_CKPT="$DEFAULT_V5_DIR/last.ckpt"
    else
        echo "ERROR: No v5 ckpt found under $DEFAULT_V5_DIR. Set BASE_CKPT=..." >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/grpo_plus_v5_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (override via env) ───────────────────────────────────────────
ALPHA="${ALPHA:-0.5}"
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS_SEQ="${CLIP_EPS_SEQ:-4e-4}"   # GSPO sequence-level
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
DEVICES="${DEVICES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_grpo_plus_output_v5}"

echo "=================================================="
echo "DiffusionDrive-AR GRPO+ Fine-tuning  (v5 base)"
echo "  GSPO sequence-level ratio + group-divergence token advantage"
echo "=================================================="
echo "Algorithm    : grpo_plus"
echo "α (blend)    : $ALPHA   (0=pure GSPO, 1=pure token-attention)"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps_seq : $CLIP_EPS_SEQ  (sequence-level)"
echo "LR           : $LR"
echo "Epochs       : $MAX_EPOCHS"
echo "Devices      : $DEVICES"
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
    ++experiment_name="diffusiondrive_ar_grpo_plus_v5" \
    ++config.ego_vocab_size=2048 \
    ++config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    ++config.ar_codebook_mode=step_corners \
    ++config.ar_use_residual_delta=true \
    ++config.ar_use_heading_head=false \
    ++config.ar_step_aware_agent=true \
    ++config.ar_use_ego_cross_attn=true \
    ++config.ar_use_deformable_bev=false \
    ++config.ar_use_bev_pos_enc=true \
    ++config.agent_topk=30 \
    ++trainer.params.max_epochs="$MAX_EPOCHS" \
    ++trainer.params.devices="$DEVICES" \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++trainer.params.gradient_clip_val=1.0 \
    ++batch_size=1 \
    ++num_workers=0 \
    ++algorithm=grpo_plus \
    ++token_attention_alpha="$ALPHA" \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps_seq="$CLIP_EPS_SEQ" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_plus_v5_a${ALPHA}_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "GRPO+ Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
