#!/bin/bash
# Dr. GRPO (arXiv:2503.20783) fine-tuning on top of the v6 SFT checkpoint,
# using the NoRD NAVSIM recipe (arxiv 2602.21172, §5.2 + §11.2).
#
# Why NoRD-style: prior recipe (lr=1e-6, KL=0.05, batch=4, T=0.7) produced
# KL≈0 over 200 steps → policy did not move. NoRD's NAVSIM run uses much
# larger gradient steps and no KL trust region, which is what actually
# moves a saturated SFT policy.
#
#   NoRD NAVSIM Dr. GRPO (their numbers):
#     LR = 5e-6 (constant), KL = 0, temperature = 1.0,
#     batch = 128 trajectories / opt step, group = 8, ~160 steps total
#
#   Our matched recipe (4 GPUs, can't scale to NoRD's 30):
#     LR = 5e-6, KL = 0, temperature = 1.0, group = 8,
#     accumulate_grad_batches = 4
#       ⇒ effective = 4 GPUs · 1 batch · 4 accum · 8 group
#                  = 128 trajectories / opt step  (matches NoRD)
#     1 epoch on navtest ≈ 3037 batches / 4 accum ≈ 760 opt steps
#     save every 60 opt steps → ~12 ckpts per epoch
#
# Reward stays at PDMS only — NoRD's length/format reward terms exist
# because they output a token sequence whose length/format can be malformed;
# our AR head emits fixed-length T=8 valid codebook indices, no-ops here.
#
# v6 ≠ v5: deformable BEV ON (GridSample cross-attn for K/V), bev_pos_enc ON.
#
# Default base = manual_epoch_068.ckpt (a copy of last-v1.ckpt at epoch 68
# from the still-running joint_v6 SFT training, snapshotted manually so the
# running SFT job's overwrites won't clobber it).
#
# Usage:
#   ./run_drgrpo_training_v6.sh
#   BASE_CKPT=/path/to/v6_milestone_epoch_080.ckpt ./run_drgrpo_training_v6.sh
#   ACCUMULATE_GRAD_BATCHES=8 ./run_drgrpo_training_v6.sh   # 256 / opt step
#   LR=1e-5 KL_COEF=0.01 ./run_drgrpo_training_v6.sh        # tweak if needed

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
    # Prefer manual_epoch_*.ckpt first (frozen snapshots), then automatic
    # milestone_epoch_*.ckpt (only created from epoch 80 onward), then
    # last.ckpt as last resort. NEVER pick last-v1.ckpt — that's the live
    # file the SFT job is still overwriting.
    #
    # Note: `|| true` is required — `ls` exits non-zero when no glob match,
    # which under `set -euo pipefail` would silently kill the script.
    LATEST_MANUAL=$(ls -1 "$DEFAULT_V6_DIR"/manual_epoch_*.ckpt 2>/dev/null | sort | tail -1 || true)
    LATEST_MS=$(ls -1 "$DEFAULT_V6_DIR"/milestone_epoch_*.ckpt 2>/dev/null | sort | tail -1 || true)
    if [ -n "$LATEST_MANUAL" ]; then
        BASE_CKPT="$LATEST_MANUAL"
    elif [ -n "$LATEST_MS" ]; then
        BASE_CKPT="$LATEST_MS"
    elif [ -f "$DEFAULT_V6_DIR/last.ckpt" ]; then
        BASE_CKPT="$DEFAULT_V6_DIR/last.ckpt"
    else
        echo "ERROR: No v6 ckpt found under $DEFAULT_V6_DIR. Set BASE_CKPT=..." >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/drgrpo_v6_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (NoRD recipe defaults; override via env) ─────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.0}"              # NoRD: no KL trust region
CLIP_EPS="${CLIP_EPS:-0.2}"            # Dr. GRPO token-level PPO clip
LR="${LR:-5e-6}"                       # NoRD NAVSIM constant LR
TEMPERATURE="${TEMPERATURE:-1.0}"      # NoRD rollout temperature
MAX_EPOCHS="${MAX_EPOCHS:-1}"          # 1 epoch ≈ ~760 opt steps @ accum=4
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"  # effective batch = 128 traj/step
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-60}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"     # keep all ckpts within an epoch
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_output_v6}"

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GRPO Fine-tuning  (v6 base, NoRD recipe)"
echo "  no std-norm, no length-norm, token-level PPO clip, no KL"
echo "=================================================="
echo "Algorithm    : dr_grpo"
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "Temperature  : $TEMPERATURE     (NoRD = 1.0)"
echo "KL coef      : $KL_COEF       (NoRD = 0.0)"
echo "Clip eps     : $CLIP_EPS     (PPO token-level)"
echo "LR           : $LR     (NoRD = 5e-6)"
echo "Devices      : $DEVICES"
echo "Accum grads  : $ACCUMULATE_GRAD_BATCHES"
echo "Eff. batch   : $EFFECTIVE_BATCH trajectories / opt step  (NoRD = 128)"
echo "Epochs       : $MAX_EPOCHS"
echo "Save every   : $SAVE_EVERY_N_STEPS opt steps  (keep last $KEEP_LAST_N)"
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
    ++experiment_name=diffusiondrive_ar_drgrpo_v6 \
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
    ++trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    ++batch_size=1 \
    ++num_workers=0 \
    ++algorithm=dr_grpo \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++save_every_n_steps="$SAVE_EVERY_N_STEPS" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_v6_nord_g${GROUP_SIZE}_t${TEMPERATURE}_lr${LR}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GRPO Training Complete (v6)! Output: $OUTPUT_DIR"
echo "=================================================="
