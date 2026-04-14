#!/bin/bash
# Continue tuning the baseline discrete AR decoder from the trained AR checkpoint.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp

export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

BASE_AR_CKPT="/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR decoder tuning from trained AR base..."
echo "Base AR checkpoint: $BASE_AR_CKPT"
echo "Using dataset cache: /data2/byounggun/training_cache"
echo "Using ego codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy"
echo "Trainable parameters: discrete AR trajectory head only"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_decoder_tune_from_base \
    trainer.params.max_epochs=40 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp \
    dataloader.params.batch_size=64 \
    agent.lr=1e-4 \
    agent.checkpoint_path="$BASE_AR_CKPT" \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.freeze_pretrained_trunk=true \
    output_dir=/data2/byounggun/diffusiondrive_ar_output \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_decoder_tune_from_base"
