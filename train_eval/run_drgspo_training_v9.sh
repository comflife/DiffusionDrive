#!/bin/bash
# Dr. GSPO — NoRD recipe applied to Qwen GSPO.
#   - No /std in advantage  (same fix as Dr. GRPO)
#   - Token-level PPO clip ε = 0.2 instead of tiny sequence-level clip (4e-4)
#     which is designed for LLM-scale T. T=8 is far too short for 4e-4.
#   - Sequence-level importance ratio s_i = exp(mean_t Δlog π_t) is kept.
#
# Base: v9 SFT checkpoint (milestone_epoch_110.ckpt, navtest PDMS=0.80241).
#
# NoRD recipe (uniform across all RL algos):
#   LR = 5e-6 (constant), KL = 0, temperature = 1.0,
#   batch = 128 trajectories / opt step (4 GPUs × 1 × accum=4 × group=8).

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

# ── Pick a base checkpoint ────────────────────────────────────────────────
DEFAULT_V9_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v9/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V9_DIR/milestone_epoch_110.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v9 epoch-110 ckpt not found at: $BASE_CKPT" >&2
        echo "       Set BASE_CKPT=/path/to/v9.ckpt to override." >&2
        exit 1
    fi
fi
if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/drgspo_v9_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (NoRD recipe defaults; override via env) ─────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
LR="${LR:-5e-6}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_dr_gspo_output_v9}"

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GSPO Fine-tuning  (v9 ep110 base, NoRD recipe)"
echo "=================================================="
echo "Algorithm    : dr_gspo"
echo "Base ckpt    : $BASE_CKPT       (PDMS=0.80241)"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE     (NoRD = 1.0)"
echo "KL coef      : $KL_COEF       (NoRD = 0.0)"
echo "Clip eps     : $CLIP_EPS     (token-level PPO, used for seq ratio)"
echo "LR           : $LR     (NoRD = 5e-6)"
echo "Devices      : $DEVICES"
echo "Accum grads  : $ACCUMULATE_GRAD_BATCHES"
echo "Eff. batch   : $EFFECTIVE_BATCH traj/step  (NoRD = 128)"
echo "Epochs       : $MAX_EPOCHS"
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
    ++experiment_name=diffusiondrive_ar_dr_gspo_v9 \
    ++config.ego_vocab_size=2048 \
    ++config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    ++config.ar_codebook_mode=step_corners \
    ++config.ar_use_residual_delta=true \
    ++config.ar_use_heading_head=false \
    ++config.ar_step_aware_agent=false \
    ++config.ar_use_ego_cross_attn=true \
    ++config.ar_use_deformable_bev=true \
    ++config.ar_use_bev_pos_enc=true \
    ++config.ar_attn_stack_ordering=bev_first \
    ++config.agent_topk=30 \
    ++trainer.params.max_epochs="$MAX_EPOCHS" \
    ++trainer.params.devices="$DEVICES" \
    ++trainer.params.strategy=ddp_find_unused_parameters_true \
    ++trainer.params.gradient_clip_val=1.0 \
    ++trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    ++batch_size=1 \
    ++num_workers=0 \
    ++algorithm=dr_gspo \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgspo_v9_nord_g${GROUP_SIZE}_t${TEMPERATURE}_lr${LR}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GSPO (v9 base) Training Complete!  Output: $OUTPUT_DIR"
echo "=================================================="
