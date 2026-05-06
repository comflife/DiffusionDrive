#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v6 (DiffusionDrive-style BEV conditioning):
#   - based on joint_v4 (v2048 step_corners + ego cross-attn ON)
#   - v6 change: deformable BEV -> ON
#   - AR deformable BEV now attends to the causal prefix trajectory, preserving
#     the original DiffusionDrive intent of trajectory-conditioned spatial BEV lookup
#   - backbone joint training (freeze=false), trunk_lr_mult=0.05
# Hypothesis: keeping the original DiffusionDrive conditioning style while swapping
# only the trajectory decoder to a discrete AR decoder improves planning fidelity.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v6 (DiffusionDrive-style deformable BEV + ego cross-attn)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, corner-> [V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta ON, heading head OFF"
echo "Agent       : step-aware nonlinear fusion"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON"
echo "BEV path    : causal prefix trajectory-conditioned deformable sampling"
echo "Trunk       : joint training (freeze=false), lr x 0.05 (= 1e-5)"
echo "Head        : full lr 2e-4"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 80"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v6 \
    trainer.params.max_epochs=150 \
    +trainer.params.devices=4 \
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
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.ar_use_residual_delta=true \
    agent.config.ar_use_heading_head=false \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.trunk_lr_mult=0.05 \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=80 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v6 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v6" \
    "$@"
