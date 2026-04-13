#!/bin/bash
# Visualization script for GRPO model

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

# Default paths
CKPT="/data2/byounggun/diffusiondrive_grpo_output_v2/checkpoints/grpo-epoch=00.ckpt"
CSV="/data2/byounggun/diffusiondrive_grpo_output_v2/eval_results/grpo_results.csv"
METRIC_CACHE="/data2/byounggun/metric_cache"
OUTPUT_DIR="/home/byounggun/DiffusionDrive/plots/grpo_visualization"

# Allow override via environment variables
[ -n "$GRPO_CKPT" ] && CKPT="$GRPO_CKPT"
[ -n "$GRPO_CSV" ] && CSV="$GRPO_CSV"

# Parse command line arguments
MODE="mixed"
NUM_SCENES=20
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --num_scenes)
            NUM_SCENES="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --csv)
            CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== GRPO Visualization ==="
echo "Checkpoint: $CKPT"
echo "CSV: $CSV"
echo "Mode: $MODE"
echo "Num scenes: $NUM_SCENES"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo ""

python3 visualize_grpo_inference.py \
    --ckpt "$CKPT" \
    --csv "$CSV" \
    --metric_cache "$METRIC_CACHE" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --num_scenes "$NUM_SCENES" \
    --gpu "$GPU" \
    --split test
