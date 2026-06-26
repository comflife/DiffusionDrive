#!/bin/bash
# Evaluate the last checkpoint from Dr. GRPO+ v6_waymo epoch120 ver2
# test assignment + random scenes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_assignment_scene_tokens.sh"
source "$SCRIPT_DIR/_random_scene_tokens.sh"
source "$SCRIPT_DIR/_assign_plus_random_scene_tokens.sh"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/home/byounggun/DiffusionDrive}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/data/navsim/exp/bg}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/navsim/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/data/navsim/dataset/maps}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_IGNORE_UNHANDLED_ERRORS=1

ASSIGNMENT_JSON="${ASSIGNMENT_JSON:-/home/byounggun/DiffusionDrive/assignments_navsim_all-splits_keep_all-queue_all-scope_all-labelers_2026-05-26T1239.json}"
ASSIGNMENT_SPLITS="${ASSIGNMENT_SPLITS:-test}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_SCENE_DATA_SPLIT="${RANDOM_SCENE_DATA_SPLIT:-test}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/data2/byounggun/metric_cache}"
WORKER_THREADS="${WORKER_THREADS:-2}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

RL_DIR="${RL_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_output_v6_waymo_epoch120_ver2_test_assign_plus_random_scenes}"
CKPT="${CKPT:-$RL_DIR/checkpoints/last.ckpt}"
HYDRA_CKPT="${HYDRA_CKPT:-/tmp/drgrpo_plus_v6_waymo_test_assign_plus_random_last.ckpt}"
OUT_DIR="${OUT_DIR:-/data2/byounggun/diffusiondrive_drgrpo_plus_eval_v6_waymo_epoch120_ver2_test_assign_plus_random_scenes/last}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

COMBINED_TOKEN_OUTPUT="$(assign_plus_random_scene_tokens_and_count "$ASSIGNMENT_JSON" "$ASSIGNMENT_SPLITS" "$RANDOM_SEED" "$RANDOM_SCENE_DATA_SPLIT")"
COMBINED_SCENE_COUNT="${COMBINED_TOKEN_OUTPUT%% *}"
COMBINED_SCENE_TOKENS="${COMBINED_TOKEN_OUTPUT#* }"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU="${GPU_LIST[0]}"

cd "$NAVSIM_DEVKIT_ROOT"
mkdir -p "$OUT_DIR"
ln -sfn "$CKPT" "$HYDRA_CKPT"

echo "Checkpoint      : $CKPT"
echo "Output dir      : $OUT_DIR"
echo "GPU             : $GPU"
echo "Eval scenes     : $COMBINED_SCENE_COUNT"
echo "Metric cache    : $METRIC_CACHE_PATH"

CUDA_VISIBLE_DEVICES="$GPU" \
RAY_CUDA_VISIBLE_DEVICES="$GPU" \
python3 -m navsim.planning.script.run_pdm_score \
    train_test_split=navtest \
    agent=diffusiondrive_ar_agent \
    "agent.checkpoint_path=$HYDRA_CKPT" \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/waymo_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=false \
    agent.config.ar_num_modes=1 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=true \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.agent_topk=30 \
    agent.config.freeze_pretrained_trunk=false \
    worker=ray_distributed \
    worker.threads_per_node="$WORKER_THREADS" \
    metric_cache_path="$METRIC_CACHE_PATH" \
    navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
    sensor_blobs_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
    experiment_name=drgrpo_plus_v6_waymo_test_assign_plus_random_last_eval \
    output_dir="$OUT_DIR" \
    "++train_test_split.scene_filter.tokens=$COMBINED_SCENE_TOKENS" \
    "$@"

LATEST_CSV="$(find "$OUT_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [ -z "$LATEST_CSV" ]; then
    echo "ERROR: no csv found in $OUT_DIR" >&2
    exit 1
fi

python3 - "$LATEST_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1]) as f:
    row = list(csv.DictReader(f))[-1]

print("csv:", sys.argv[1])
print("score:", row["score"])
print("valid:", row["valid"])
PY
