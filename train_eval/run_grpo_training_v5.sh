#!/bin/bash
# Vanilla GRPO fine-tuning on top of the v5 SFT checkpoint.
# (Token-level importance ratio, PPO clipping ε ≈ 0.2.)
#
# v5 config (mirrored here so the loaded ckpt structurally matches):
#   - V=2048 step_corners codebook
#   - residual delta ON, heading head OFF
#   - step-aware agent ON, per-layer ego cross-attn ON
#   - deformable BEV OFF (flat global MHA), 2D sin-cos BEV pos enc ON
#   - backbone joint training, uniform LR
#
# For sequence-level alternatives, use:
#   - run_gspo_training_v5.sh         (Qwen GSPO sequence-level ratio)
#   - run_grpo_plus_training_v5.sh    (GSPO ratio + token-attention advantage)
#
# Usage:
#   ./run_grpo_training_v5.sh
#   BASE_CKPT=/path/to/v5_milestone_epoch_080.ckpt ./run_grpo_training_v5.sh
#   GROUP_SIZE=16 KL_COEF=0.1 LR=1e-6 TEMPERATURE=0.3 ./run_grpo_training_v5.sh

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

SAFE_CKPT="/tmp/grpo_v5_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (override via env) ───────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS="${CLIP_EPS:-0.2}"            # GRPO token-level PPO clip
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
DEVICES="${DEVICES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_grpo_output_v5}"

echo "=================================================="
echo "DiffusionDrive-AR GRPO Fine-tuning  (v5 base, token-level)"
echo "=================================================="
echo "Algorithm    : grpo  (token-level importance ratio)"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps     : $CLIP_EPS  (PPO token-level)"
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
    ++experiment_name=diffusiondrive_ar_grpo_v5 \
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
    ++algorithm=grpo \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_v5_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "GRPO Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
