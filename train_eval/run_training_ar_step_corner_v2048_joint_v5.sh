#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v5 (uniform LR ablation):
#   - based on joint_v4 (v2048 step_corners + flat BEV + 2D pos enc + ego cross-attn)
#   - v5 change: backbone/head use the same lr (no trunk lr multiplier)
#   - deformable BEV  → OFF (back to global flat MHA)
#   - 2D sin-cos BEV positional encoding → ON  (UniAD/DETR-style)
#   - backbone joint training (freeze=false), uniform lr=2e-4
# Hypothesis: removing the low-lr trunk split tests whether joint full-rate
# optimization helps once ego cross-attn is already restored.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v5 (flat BEV + 2D pos enc + ego cross-attn + uniform lr)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, corner→[V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta ON, heading head OFF"
echo "Agent       : step-aware nonlinear fusion"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV OFF"
echo "BEV pos enc : 2D sin-cos ON"
echo "Trunk       : joint training (freeze=false), same lr as head"
echo "Head        : full lr 2e-4"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 80"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v5 \
    trainer.params.max_epochs=150 \
    +trainer.params.devices=4 \
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
    agent.config.ar_use_heading_head=false \
    agent.config.ar_step_aware_agent=true \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=false \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=80 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v5 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v5" \
    "$@"
