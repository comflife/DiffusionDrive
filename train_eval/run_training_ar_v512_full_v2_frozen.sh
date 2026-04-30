#!/bin/bash
# Train DiffusionDrive-AR — v2 ablation (FROZEN trunk variant):
#   - identical to run_training_ar_v512_full_v2.sh EXCEPT:
#   - freeze_pretrained_trunk=true → only _trajectory_head.* is trained.
#   - SFT-style: trunk locked at pretrained DiffusionDrive weights.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR v2 FROZEN (v512 step_delta + heading refine + full conditioning)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v512/ego.npy  (V=512, [V,2] xy-only)"
echo "Mode        : step_delta"
echo "Refinement  : residual delta ON, heading head ON"
echo "Agent       : step-aware nonlinear fusion"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON"
echo "Trunk       : FROZEN (only _trajectory_head.* trains)"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_v512_full_v2_frozen \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ego_vocab_size=512 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy \
    agent.config.ar_codebook_mode=step_delta \
    agent.config.ar_teacher_forcing=true \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=true \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.freeze_pretrained_trunk=true \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/v512_full_v2_frozen \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_v512_full_v2_frozen"
