#!/bin/bash
# GRPO fine-tuning on top of the v6 SFT checkpoint.
#
# Compatible with the new GRPO trainer (grpo_trainer.py v3 fixes):
#   - Hydra-driven TransfuserConfig (V=2048 step_corners + v6 modules)
#   - diff_decoder→AR weight remap on load (warm-start AR cross-attns)
#   - Reference model mirrored from policy → KL=0 at init
#   - Sampling temperature consistent across old/new/ref log_probs (PPO unbiased)
#   - Model heading trusted in PDM reward (no atan2 override for step_corners)
#   - PDM failure streak tracking (raises after PDM_FAIL_RAISE_AFTER consecutive)
#
# Default base = milestone_epoch_080.ckpt from joint_v6 SFT training.
#
# Usage:
#   ./run_grpo_training_v6.sh
#   BASE_CKPT=/path/to/v6.ckpt ./run_grpo_training_v6.sh
#   (or override with env vars below)

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

# Hydra-safe alias (avoids '=' / spaces in actual ckpt paths confusing parsing)
SAFE_CKPT="/tmp/grpo_v6_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (override via env) ───────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
DEVICES="${DEVICES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_grpo_output_v6}"

echo "=================================================="
echo "DiffusionDrive-AR GRPO Fine-tuning  (v6 base, fixed trainer)"
echo "=================================================="
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE   (rollouts per scene)"
echo "Temperature  : $TEMPERATURE  (sample + log_prob consistent)"
echo "KL coef      : $KL_COEF      (categorical KL on π(·|T))"
echo "PPO clip eps : $CLIP_EPS"
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
    ++experiment_name=diffusiondrive_ar_grpo_v6 \
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
    ++algorithm=grpo \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_v6_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "GRPO Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
