#!/bin/bash
# Dr. GRPO+ v3 (ULTRA AGGRESSIVE + Smart SFT Anchor) on v6_waymo epoch-120

set -euo pipefail

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

DEFAULT_V6_WAYMO_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V6_WAYMO_DIR/milestone_epoch_120.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v6_waymo epoch-120 ckpt not found at: $BASE_CKPT" >&2
        exit 1
    fi
fi

SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_ver3_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

GROUP_SIZE="${GROUP_SIZE:-16}"
KL_COEF="${KL_COEF:-0.001}"
CLIP_EPS="${CLIP_EPS:-0.3}"
LR="${LR:-2e-5}"
TEMPERATURE="${TEMPERATURE:-0.95}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
KEEP_LAST_N="${KEEP_LAST_N:-9999}"
SFT_AUX_COEF="${SFT_AUX_COEF:-0.25}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver3}"

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
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then echo "ERROR: RESUME_CKPT not found"; exit 1; fi
    RESUME_SAFE="/tmp/drgrpo_plus_v6_waymo_ver3_resume_$(date +%s).ckpt"
    ln -sfn "$RESUME_CKPT" "$RESUME_SAFE"
    RESUME_ARGS=(++resume_ckpt_path="$RESUME_SAFE")
fi

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "Dr. GRPO+ v3 ULTRA (Smart SFT Anchor) - v6_waymo ep120"
echo "=================================================="
echo "LR=$LR  G=$GROUP_SIZE  clip=$CLIP_EPS  acc=$ACCUMULATE_GRAD_BATCHES  sft_aux=$SFT_AUX_COEF"
echo "Eff.batch=$EFFECTIVE_BATCH  epochs=$MAX_EPOCHS"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtest \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path=/data2/byounggun/metric_cache \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    output_dir="$OUTPUT_DIR" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver3 \
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
    ++trainer.params.gradient_clip_val=1.5 \
    ++trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    ++batch_size=1 \
    ++num_workers=0 \
    ++algorithm=dr_grpo_plus \
    ++group_size="$GROUP_SIZE" \
    ++kl_coef="$KL_COEF" \
    ++clip_eps="$CLIP_EPS" \
    ++lr="$LR" \
    ++temperature="$TEMPERATURE" \
    ++sft_aux_coef="$SFT_AUX_COEF" \
    ++keep_last_n_ckpts="$KEEP_LAST_N" \
    "${RESUME_ARGS[@]}" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_plus_v6_waymo_ver3_g${GROUP_SIZE}_lr${LR}_sft${SFT_AUX_COEF}" \
    "$@"

echo "Dr. GRPO+ v3 ULTRA complete. Output: $OUTPUT_DIR"
