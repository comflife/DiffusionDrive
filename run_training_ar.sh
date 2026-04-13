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
echo "Decoder: single-mode discrete AR + residual trajectory reconstruction"
echo "Initializing from checkpoint: /home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS"

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
    dataloader.params.batch_size=32 \
    agent.lr=1e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.freeze_pretrained_trunk=true \
    output_dir=/data2/byounggun/diffusiondrive_ar_output \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_v512"

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
