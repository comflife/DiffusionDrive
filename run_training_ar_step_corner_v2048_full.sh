#!/bin/bash
# Train DiffusionDrive-AR with the FULL conditioning recovery:
#   - step-corner v2048 codebook (heading included)
#   - residual delta head ON
#   - heading refine head ON
#   - step-aware agent fusion ON
#   - per-layer ego cross-attention ON   (recovers original diffusion conditioning)
#   - deformable BEV cross-attention ON  (waypoint-aware spatial sampling)
#   - backbone joint training (no freeze)
#
# Override CUDA_VISIBLE_DEVICES if you want different GPUs.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR FULL (per-layer ego + deformable BEV)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048)"
echo "Refinement  : residual delta ON, heading head OFF (codebook dtheta used as-is to avoid override)"
echo "Agent       : step-aware nonlinear fusion"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON"
echo "Trunk       : joint training (freeze=false)"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_full \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=2 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=32 \
    agent.lr=2e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=true \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=0.0 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=false \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.freeze_pretrained_trunk=false \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/full \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_full"
