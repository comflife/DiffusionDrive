#!/bin/bash
# GSPO (Group Sequence Policy Optimization, Qwen 2025-07) fine-tuning on
# top of the v6 SFT checkpoint.
#
# GSPO vs GRPO (one-line summary):
#   - GRPO: per-token importance ratio, PPO clip ε ≈ 0.2
#   - GSPO: per-sequence importance ratio with length normalization
#           s_i = exp((1/T) · Σ_t [log π_new(a_t|s, a_<t) − log π_old(a_t|s, a_<t)])
#           Clip applied to s_i directly, ε_seq ≈ 3e-4 ~ 4e-4
#           (~500x tighter than GRPO because s_i is concentrated near 1).
#   Why: token-level ratios are noisier and produce wasted off-policy gradient,
#        especially for long sequences. Sequence-level ratio aligns the
#        importance correction with the granularity of the reward (per-rollout
#        PDM score in our case).
#
# Reference: arXiv:2507.18071  (Zheng et al., Qwen team)
#
# Default base = milestone_epoch_080.ckpt from joint_v6 SFT training.
#
# Usage:
#   ./run_gspo_training_v6.sh
#   BASE_CKPT=/path/to/v6.ckpt ./run_gspo_training_v6.sh
#   ALGORITHM=gspo_token ./run_gspo_training_v6.sh   # (gradient routes per-token, value matches GSPO)

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

# ── Pick a base checkpoint ────────────────────────────────────────────────
DEFAULT_V6_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V6_DIR/milestone_epoch_080.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v6 epoch-80 ckpt not found at: $BASE_CKPT" >&2
        echo "       Set BASE_CKPT=/path/to/v6.ckpt to override." >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/gspo_v6_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (override via env) ───────────────────────────────────────────
ALGORITHM="${ALGORITHM:-gspo}"            # gspo | gspo_token | grpo (debug)
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS_SEQ="${CLIP_EPS_SEQ:-4e-4}"      # GSPO sequence-level clip (paper: 3e-4 ~ 4e-4)
CLIP_EPS="${CLIP_EPS:-0.2}"               # only used if ALGORITHM=grpo
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
DEVICES="${DEVICES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_${ALGORITHM}_output_v6}"

echo "=================================================="
echo "DiffusionDrive-AR ${ALGORITHM^^} Fine-tuning  (v6 base, sequence-level)"
echo "=================================================="
echo "Algorithm    : $ALGORITHM"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE   (rollouts per scene)"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps_seq : $CLIP_EPS_SEQ  (GSPO; ignored if algorithm=grpo)"
echo "Clip eps     : $CLIP_EPS      (GRPO; ignored if algorithm=gspo*)"
echo "LR           : $LR"
echo "Epochs       : $MAX_EPOCHS"
echo "Save every   : 1 epoch  (keep last $KEEP_LAST_N)"
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
    ++experiment_name="diffusiondrive_ar_${ALGORITHM}_v6" \
    ++config.ego_vocab_size=2048 \
    ++config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    ++config.ar_codebook_mode=step_corners \
    ++config.ar_use_residual_delta=true \
    ++config.ar_use_heading_head=false \
    ++config.ar_step_aware_agent=true \
    ++config.ar_use_ego_cross_attn=true \
    ++config.ar_use_deformable_bev=true \
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
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="${ALGORITHM}_v6_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "${ALGORITHM^^} Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
