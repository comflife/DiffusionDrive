#!/bin/bash
# GRPO+ : GSPO sequence-level importance ratio + per-token advantage
# (group-divergence-weighted) on top of the v6 SFT checkpoint.
#
# Motivation
# ──────────
# Vanilla GRPO with PDM reward suffers two problems:
#   1. PDMS is a SCALAR per-trajectory reward, but GRPO computes the
#      importance ratio per token. Token-level ratio noise piles up across
#      T tokens, inflating policy-gradient variance.
#   2. PDMS reward equally weights every token in the trajectory. But the
#      decisions that actually matter (e.g. obstacle avoidance, sharp turns)
#      live in only a few waypoints. Sequence-only advantage cannot focus
#      the gradient on those decisive tokens.
#
# GRPO+ addresses both:
#   1. Importance ratio  →  GSPO sequence-level (length-normalized):
#         s_i = exp((1/T) · Σ_t [log π_new(a_t|s, a_<t) − log π_old(a_t|s, a_<t)])
#      Clip at ε_seq ≈ 4e-4 (sequence ratio is concentrated near 1).
#   2. Advantage         →  hybrid sequence + token:
#         A_seq[i]   = (r_i − mean(r)) / std(r)
#         w[i, t]    = ||pos_xy[i, t] − group_mean_pos_xy[t]|| / mean_t(...)
#         A_tok[i,t] = A_seq[i] · w[i, t]            (mean over t = A_seq)
#         loss = (1−α)·GSPO_term(s_i, A_seq) + α·TokenAttn_term(s_it, A_tok)
#
# w[i, t] is large at waypoints where rollout i diverged from the group mean
# (the "decision points" of this scene). Combined with the SIGN of A_seq, it
# concentrates positive gradient on good-rollout-divergent tokens and negative
# gradient on bad-rollout-divergent tokens — i.e. exactly the tokens where the
# policy actually made the choice that drove the reward differential.
#
# α = 0  →  pure GSPO
# α = 1  →  pure token-attention (no flat sequence baseline)
# α ≈ .5 →  balanced; recommended starting point
#
# Default base = milestone_epoch_080.ckpt from joint_v6 SFT training.
#
# Usage:
#   ./run_grpo_plus_training_v6.sh
#   BASE_CKPT=/path/to/v6.ckpt ./run_grpo_plus_training_v6.sh
#   ALPHA=0.7 KL_COEF=0.05 ./run_grpo_plus_training_v6.sh

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

SAFE_CKPT="/tmp/grpo_plus_v6_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Tunables (override via env) ───────────────────────────────────────────
ALPHA="${ALPHA:-0.5}"                       # blend (1−α)·GSPO + α·TokenAttn
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_COEF="${KL_COEF:-0.05}"
CLIP_EPS_SEQ="${CLIP_EPS_SEQ:-4e-4}"        # sequence-level clip (GSPO range)
LR="${LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
DEVICES="${DEVICES:-4}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_grpo_plus_output_v6}"

echo "=================================================="
echo "DiffusionDrive-AR GRPO+ Fine-tuning  (v6 base)"
echo "  GSPO sequence-level ratio + group-divergence token advantage"
echo "=================================================="
echo "Base ckpt    : $BASE_CKPT"
echo "Hydra alias  : $SAFE_CKPT"
echo "α (blend)    : $ALPHA   (0=pure GSPO, 1=pure token-attention)"
echo "Group size   : $GROUP_SIZE   (rollouts per scene)"
echo "Temperature  : $TEMPERATURE"
echo "KL coef      : $KL_COEF"
echo "Clip eps_seq : $CLIP_EPS_SEQ  (sequence-level)"
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
    ++experiment_name="diffusiondrive_ar_grpo_plus_v6" \
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
    ++algorithm=grpo_plus \
    ++token_attention_alpha="$ALPHA" \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps_seq="$CLIP_EPS_SEQ" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="grpo_plus_v6_a${ALPHA}_g${GROUP_SIZE}_t${TEMPERATURE}_kl${KL_COEF}" \
    "$@"

echo "=================================================="
echo "GRPO+ Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
