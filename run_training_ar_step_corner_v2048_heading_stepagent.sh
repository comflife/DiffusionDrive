#!/bin/bash
# Train DiffusionDrive-AR (regular BOS, NOT ar_plus) with:
#   - new step-corner v2048 codebook (includes heading)
#   - residual delta head ON
#   - heading refine head ON
#   - step-aware agent fusion (fix for the time-invariant agent context issue)

export CUDA_VISIBLE_DEVICES=0,1
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 (heading head + step-aware agent)..."
echo "Pretrained checkpoint: /home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS"
echo "Codebook: codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, [V,3])"
echo "Refinement: residual delta ON, heading head ON"
echo "Agent context: step-aware nonlinear fusion (ar_step_aware_agent=true)"
echo "Trunk freeze: OFF (backbone joint training)"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_heading_stepagent \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=2 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=true \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=true \
    agent.config.ar_step_aware_agent=true \
    agent.config.freeze_pretrained_trunk=false \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/heading_stepagent \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_heading_stepagent"
