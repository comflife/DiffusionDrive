#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v10 (deeper AR head):
#   - based on v9 (bev_first attn ordering, single-call agent attention,
#     deformable BEV ON, ego cross-attn ON, 2D BEV pos enc ON, agent_topk=30,
#     step_aware OFF)
#   - v10 change: ar_num_layers 2 -> 12
#   - Hypothesis: v6..v9 plateau at PDMS ~0.80 partly because the AR head is
#     only 2 layers deep. Increasing depth gives more iterations of
#     (BEV -> Agent -> SelfAttn -> Ego -> FFN) refinement on the discrete
#     token sequence before logits / residual delta.
#   - Warm-start: DiffusionDrive diff_decoder has only 2 layers, so AR
#     layers 0..1 receive the warm-started weights via the existing remap;
#     AR layers 2..11 stay randomly initialized (strict=False, harmless).
#     The extra layers may need a few extra epochs to settle; the cosine
#     schedule is unchanged for direct comparison with v9.
#   - Risk: extra random-init layers near the input may disrupt the
#     warm-start signal. If PDMS regresses vs v9 in early epochs, try
#     reducing depth (e.g. 6) or using a smaller LR for the new layers.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v10 (deeper AR head, num_layers=6)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive_v2/ego.npy  (V=2048, corner-> [V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta OFF, heading head OFF"
echo "Agent       : SINGLE-CALL (step_aware=false), K/V shared across T"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON, BEV pos enc OFF"
echo "Op order    : BEV -> Agent -> SelfAttn(causal) -> Ego -> FFN  (bev_first, same as v9)"
echo "AR depth    : 6 layers (v9 had 2)"
echo "Init        : warm-start from 88.1 PDMS DiffusionDrive checkpoint (only layers 0..1)"
echo "Agent K/V   : agent_topk=30 (matches original cross_agent_attention)"
echo "LR          : uniform 2e-4 across head + trunk"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 60"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v10 \
    trainer.params.max_epochs=150 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ego_vocab_size=2048 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048_diffusiondrive_v2/ego.npy \
    agent.config.ar_codebook_mode=step_corners \
    agent.config.ar_teacher_forcing=true \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=8.0 \
    agent.config.ar_heading_loss_weight=2.0 \
    agent.config.ar_use_residual_delta=false \
    agent.config.ar_use_heading_head=false \
    agent.config.ar_step_aware_agent=false \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=false \
    agent.config.ar_attn_stack_ordering=bev_first \
    agent.config.ar_num_layers=6 \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=60 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v10 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v10" \
    "$@"
