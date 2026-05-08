#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v9_scratch (bev_first, scratch init):
#   - same architecture as v9 (bev_first attn ordering, single-call agent attention,
#     deformable BEV ON, ego cross-attn ON, 2D BEV pos enc ON, step_aware OFF)
#   - DIFFERENCE from v9: NO 88.1 PDMS DiffusionDrive checkpoint warm-start.
#       agent.checkpoint_path=null  (overrides yaml default)
#     Image/lidar backbones still load ImageNet ResNet34 (resnet34.a1_in1k)
#     via timm bkb_path — matches the original DiffusionDrive starting point.
#     The AR trajectory head, BEV decoders, agent/ego heads, etc. all start
#     from random init.
#   - Comparing v9 vs v9_scratch (and against v7 / v7_scratch) tells us
#     whether the bev_first ordering matters mainly because of warm-start
#     alignment (gain in v9 but not v9_scratch) or because of the inductive
#     bias itself (gain in both).

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v9_scratch (bev_first ordering, no DiffusionDrive warm-start)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, corner-> [V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta ON, heading head OFF"
echo "Agent       : SINGLE-CALL (step_aware=false), K/V shared across T"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON, BEV pos enc ON"
echo "Op order    : BEV -> Agent -> SelfAttn(causal) -> Ego -> FFN  (bev_first)"
echo "Init        : SCRATCH (no DiffusionDrive 88.1 PDMS checkpoint)"
echo "Backbone    : ImageNet ResNet34 (resnet34.a1_in1k, via timm bkb_path)"
echo "Agent K/V   : agent_topk=30"
echo "LR          : uniform 2e-4 across head + trunk"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 60"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v9_scratch \
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
    agent.config.ar_attn_stack_ordering=bev_first \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=60 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v9_scratch \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v9_scratch" \
    "$@"
