#!/bin/bash
# Dr. GRPO+ v2 (Aggressive) fine-tuning on v6_waymo epoch-120 SFT, full VAL split.
#
# Trains on the official navtrain VAL scenes (~2,632) PLUS the loadable val-split
# assignment scenes (132 of 455). Only 132 assignment scenes have a route
# (roadblock_ids) and thus a PDM metric cache entry; the other 323 have empty
# roadblock_ids and cannot be PDM-scored (navsim's navtrain/navtest filters drop
# such scenes via has_route=true). All scenes resolve from a single cache,
# metric_cache_val (metric_cache_val_assign is a strict subset of it).
#
# Usage (4 GPUs, default):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_eval/drgrpoplus_val_train.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_eval/drgrpoplus_val_assign_train.sh
#   Override GPU count with DEVICES=N (e.g. DEVICES=1 for a single GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_val_scene_tokens.sh"
source "$SCRIPT_DIR/_val_assign_loadable_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

# Official navtrain VAL scene tokens (~2,632)
VAL_TOKEN_OUTPUT="$(all_val_scene_tokens_and_count)"
VAL_SCENE_COUNT="${VAL_TOKEN_OUTPUT%% *}"
VAL_SCENE_TOKENS="${VAL_TOKEN_OUTPUT#* }"

# Loadable val-split assignment scene tokens (132 of 455; route-having + cached)
ASSIGN_TOKEN_OUTPUT="$(val_assign_loadable_scene_tokens_and_count)"
ASSIGN_SCENE_COUNT="${ASSIGN_TOKEN_OUTPUT%% *}"
ASSIGN_SCENE_TOKENS="${ASSIGN_TOKEN_OUTPUT#* }"

# Merge official val + loadable val assignment scene tokens (drop overlap)
FULL_TOKEN_OUTPUT="$(python3 - "$VAL_SCENE_TOKENS" "$ASSIGN_SCENE_TOKENS" <<'PY'
import json, sys
val = json.loads(sys.argv[1])
assign = json.loads(sys.argv[2])
val_set = set(val)
merged = val + [t for t in assign if t not in val_set]
print(f"{len(merged)} {json.dumps(merged, separators=(',', ':'))}")
PY
)"
FULL_SCENE_COUNT="${FULL_TOKEN_OUTPUT%% *}"
FULL_SCENE_TOKENS="${FULL_TOKEN_OUTPUT#* }"

echo "Official val scenes      : $VAL_SCENE_COUNT"
echo "Val assign scenes (load) : $ASSIGN_SCENE_COUNT (of 455; 323 dropped: no route)"
echo "Full val scenes          : $FULL_SCENE_COUNT"

DEFAULT_V6_WAYMO_DIR="/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6_waymo/checkpoints"
if [ -z "${BASE_CKPT:-}" ]; then
    BASE_CKPT="$DEFAULT_V6_WAYMO_DIR/milestone_epoch_120.ckpt"
    if [ ! -f "$BASE_CKPT" ]; then
        echo "ERROR: Default v6_waymo epoch-120 ckpt not found at: $BASE_CKPT" >&2
        exit 1
    fi
fi

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: BASE_CKPT not found at: $BASE_CKPT" >&2
    exit 1
fi

SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_ver2_val_base_$(date +%s).ckpt"
ln -sfn "$BASE_CKPT" "$SAFE_CKPT"

# ── Aggressive v2 tunables ───────────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-12}"
KL_COEF="${KL_COEF:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.25}"
LR="${LR:-1e-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-35}"
DEVICES="${DEVICES:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
# Save full checkpoints only at these (1-indexed) epochs; last.ckpt is always kept for resume.
SAVE_EPOCHS="${SAVE_EPOCHS:-[20,25,30,35]}"
# Single val cache; it already contains the 132 loadable assignment scenes.
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache_val}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/byounggun/trained_model/drgrpoplus_val}"

# Simple resume support (auto-continue if last.ckpt exists)
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
RESUME_DISPLAY="none"
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "ERROR: RESUME_CKPT not found at: $RESUME_CKPT" >&2
        exit 1
    fi
    RESUME_SAFE_CKPT="/tmp/drgrpo_plus_v6_waymo_epoch120_ver2_val_resume_$(date +%s).ckpt"
    ln -sfn "$RESUME_CKPT" "$RESUME_SAFE_CKPT"
    RESUME_ARGS=(++resume_ckpt_path="$RESUME_SAFE_CKPT")
    RESUME_DISPLAY="$RESUME_CKPT"
fi

EFFECTIVE_BATCH=$(( DEVICES * ACCUMULATE_GRAD_BATCHES * GROUP_SIZE ))

echo "=================================================="
echo "DiffusionDrive-AR Dr. GRPO+ v2 (Aggressive) - v6_waymo epoch-120 VAL (full)"
echo "=================================================="
echo "Algorithm    : dr_grpo_plus (v2 aggressive)"
echo "Base ckpt    : $BASE_CKPT"
echo "Group size   : $GROUP_SIZE"
echo "LR           : $LR"
echo "Clip eps     : $CLIP_EPS"
echo "Accum grads  : $ACCUMULATE_GRAD_BATCHES"
echo "Eff. batch   : $EFFECTIVE_BATCH"
echo "Max epochs   : $MAX_EPOCHS"
echo "Save epochs  : $SAVE_EPOCHS (+ last.ckpt)"
echo "Metric cache : $METRIC_CACHE_PATH"
echo "Output       : $OUTPUT_DIR"
echo "=================================================="

cd "$NAVSIM_DEVKIT_ROOT"

python3 -m navsim.agents.diffusiondrive.grpo_train \
    train_test_split=navtrain \
    ++train_test_split.scene_filter.log_names=null \
    "++train_test_split.scene_filter.tokens=$FULL_SCENE_TOKENS" \
    ++checkpoint_path="$SAFE_CKPT" \
    ++metric_cache_path="$METRIC_CACHE_PATH" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
    output_dir="$OUTPUT_DIR" \
    ++experiment_name=diffusiondrive_ar_drgrpo_plus_v6_waymo_epoch120_ver2_val_full \
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
    "++save_epochs=$SAVE_EPOCHS" \
    "${RESUME_ARGS[@]}" \
    wandb.enabled=true \
    wandb.project="diffusiondrive-grpo" \
    wandb.name="drgrpo_plus_v6_waymo_ep120_ver2_val_full_g${GROUP_SIZE}_lr${LR}_clip${CLIP_EPS}_acc${ACCUMULATE_GRAD_BATCHES}" \
    "$@"

echo "=================================================="
echo "Dr. GRPO+ v2 VAL (full) Training Complete! Output: $OUTPUT_DIR"
echo "=================================================="
