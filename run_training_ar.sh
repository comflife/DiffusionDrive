#!/bin/bash
# DiffusionDrive-AR Training Script
# Uses pretrained DiffusionDrive backbone + discrete AR head
# Reuses existing dataset cache

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp

# Ensure DiffusionDrive navsim is used, not VAD
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR training..."
echo "Using pretrained backbone from: /data/navsim/dataset/checkpoints/diffusiondrive_navsim_88p1_PDMS"
echo "Using dataset cache: /data2/byounggun/training_cache"
echo "Using ego codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"

# Run training with default_training as base, override agent and other params
python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_v512 \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=1 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output

# For multi-GPU training, uncomment below:
# python -m navsim.planning.script.run_training \
#     agent=diffusiondrive_ar_agent \
#     train_test_split=navtrain \
#     cache_path="/data2/byounggun/training_cache" \
#     +experiment_name=diffusiondrive_ar_v512_distributed \
#     trainer.params.max_epochs=100 \
#     trainer.params.devices=4 \
#     dataloader.params.batch_size=1 \
#     +output_dir=/data2/byounggun/diffusiondrive_ar_output
