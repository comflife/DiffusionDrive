#!/bin/bash
# GSPO (Qwen Group Sequence Policy Optimization) fine-tuning on the v5 SFT ckpt.
#
# v5 ≠ v6: deformable BEV OFF (flat global MHA), 2D BEV pos enc ON.
# Everything else mirrors v6.
#
# Algorithm options (env-controlled):
#   ALGORITHM=gspo        → sequence-level importance ratio (default)
#   ALGORITHM=gspo_token  → GSPO value, per-token gradient routing
#   ALGORITHM=grpo        → token-level (debug / baseline comparison)
#
# CLIP_EPS_SEQ tuning note: paper uses 4e-4 for LLM-scale T (hundreds-thousands
# of tokens). Our T = num_poses = 8, so per-token noise contributes ~125× more
# to the sequence ratio. If clip_frac stays > 30% in practice, raise
# CLIP_EPS_SEQ to ~5e-3.

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

# ── Tunables (override via env) ───────────────────────────────────────────
ALGORITHM="${ALGORITHM:-gspo}"
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS_SEQ="${CLIP_EPS_SEQ:-4e-4}"   # GSPO sequence-level (paper: 3e-4 ~ 4e-4)
CLIP_EPS="${CLIP_EPS:-0.2}"            # only used if ALGORITHM=grpo
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
DEVICES="${DEVICES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_${ALGORITHM}_output_v5}"

SAFE_CKPT="/tmp/${ALGORITHM}_v5_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

echo "=================================================="
echo "DiffusionDrive-AR ${ALGORITHM^^} Fine-tuning  (v5 base, sequence-level)"
echo "=================================================="
echo "Algorithm    : $ALGORITHM"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps_seq : $CLIP_EPS_SEQ  (sequence-level, used by gspo / gspo_token)"
echo "Clip eps     : $CLIP_EPS  (token-level, used only if algorithm=grpo)"
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
    ++experiment_name="diffusiondrive_ar_${ALGORITHM}_v5" \
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
    ++algorithm="$ALGORITHM" \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++clip_eps_seq="$CLIP_EPS_SEQ" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="${ALGORITHM}_v5_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "${ALGORITHM^^} Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
