#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v7_scratch (scratch init):
#   - same architecture as v7 (single-call agent attention, deformable BEV,
#     ego cross-attn, 2D BEV pos enc, agent_topk=30, step_aware OFF)
#   - DIFFERENCE from v7: NO 88.1 PDMS DiffusionDrive checkpoint warm-start.
#       agent.checkpoint_path=null  (overrides yaml default)
#     The image/lidar backbones still load ImageNet ResNet34 (resnet34.a1_in1k)
#     via timm's bkb_path — this matches what the original DiffusionDrive paper
#     used as its starting point (ImageNet-pretrained backbone, head from
#     scratch). The AR trajectory head, BEV decoders, agent/ego heads, etc.
#     all start from random init.
#   - Purpose: clean baseline measuring how well the AR head learns from
#     scratch with the v7 architecture, without the (possibly imperfect)
#     warm-start from the diffusion-trained cross-attentions.
#   - Comparing v7 vs v7_scratch tells us how much of v7's PDMS comes from
#     warm-start vs from architecture choices.
# Hypothesis: v7_scratch will be lower than v7 at the same epoch count, but
# the gap quantifies the warm-start contribution. May need more epochs to
# converge from scratch (consider --max_epochs 200 if 150 isn't enough).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v7_scratch (no DiffusionDrive warm-start)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, corner-> [V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta ON, heading head OFF"
echo "Agent       : SINGLE-CALL (step_aware=false), K/V shared across T, no step_emb on K/V"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON"
echo "BEV path    : causal prefix trajectory-conditioned deformable sampling"
echo "Init        : SCRATCH (no DiffusionDrive 88.1 PDMS checkpoint)"
echo "Backbone    : ImageNet ResNet34 (resnet34.a1_in1k, via timm bkb_path) — same as original DiffusionDrive"
echo "Agent K/V   : agent_topk=30 (matches original cross_agent_attention)"
echo "LR          : uniform 2e-4 across head + trunk (no trunk lr multiplier)"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 60"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v7_scratch \
    trainer.params.max_epochs=150 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=null \
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
    agent.config.ar_step_aware_agent=false \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=60 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v7_scratch \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v7_scratch" \
    "$@"
