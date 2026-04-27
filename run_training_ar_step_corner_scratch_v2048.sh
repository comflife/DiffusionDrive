#!/bin/bash
# Train DiffusionDrive-AR from scratch with the new step-corner v2048 codebook.
# No pretrained checkpoint loading, no trunk freezing, no residual refine,
# no separate heading head. Pure discrete AR token training.

export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 training from scratch..."
echo "Pretrained checkpoint loading: OFF"
echo "Freeze pretrained trunk: OFF"
echo "Using dataset cache: /data2/byounggun/training_cache"
echo "Using step-corner codebook: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy"
echo "Decoder: 8 autoregressive tokens, each token includes local dx/dy + heading delta"
echo "Refinement: residual delta OFF, heading head OFF"
echo "Teacher forcing: OFF"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_scratch_v2048 \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=null \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=false \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=0.0 \
    agent.config.ar_heading_loss_weight=0.0 \
    agent.config.ar_use_residual_delta=false \
    agent.config.ar_use_heading_head=false \
    agent.config.freeze_pretrained_trunk=false \
    output_dir=/data2/byounggun/diffusiondrive_ar_output \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_scratch_v2048"
