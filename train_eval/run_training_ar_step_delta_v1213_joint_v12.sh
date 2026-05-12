#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v12 (step_delta, token-CE-only):
#   - based on v9 (bev_first attn ordering, single-call agent, deformable BEV ON,
#     ego cross-attn ON, BEV pos enc ON, agent_topk=30, step_aware OFF)
#   - v12 changes:
#       1. Codebook: navsim_kdisk_v2048/ego.npy  (actual V=1213, step_delta [V,2])
#       2. Mode: step_delta (per-step displacement tokens, cumsum trajectory)
#       3. Loss: token CE ONLY — no residual delta, no traj regression, no heading loss
#          Matches VAD discrete-AR training recipe (pure cross-entropy on tokens).
#       4. Heading: head disabled; heading derived from atan2(deltas) if needed.
#
# Rationale: VAD showed that pure token CE with a well-built codebook is enough
# for strong AR trajectory generation. Removing auxiliary regression losses
# simplifies the training signal and prevents the model from "cheating" via
# continuous regression heads instead of learning the discrete vocabulary.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-delta v1213 JOINT v12 (token-CE-only)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048/ego.npy  (V=1213, delta-> [V,2])"
echo "Mode        : step_delta"
echo "Loss        : token CE ONLY (traj_loss=0, heading_loss=0, residual_delta=OFF)"
echo "Refinement  : residual delta OFF, heading head OFF"
echo "Agent       : SINGLE-CALL (step_aware=false), K/V shared across T"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON, BEV pos enc ON"
echo "Op order    : BEV -> Agent -> SelfAttn(causal) -> Ego -> FFN  (bev_first)"
echo "Init        : warm-start from 88.1 PDMS DiffusionDrive checkpoint"
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
    +experiment_name=diffusiondrive_ar_step_delta_v1213_joint_v12 \
    trainer.params.max_epochs=150 \
    +trainer.params.devices=4 \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    dataloader.params.batch_size=64 \
    agent.lr=2e-4 \
    agent.checkpoint_path=/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    agent.config.ego_vocab_size=1213 \
    agent.config.ego_vocab_path=/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v2048/ego.npy \
    agent.config.ar_codebook_mode=step_delta \
    agent.config.ar_teacher_forcing=true \
    agent.config.ar_num_modes=1 \
    agent.config.ar_token_loss_weight=1.0 \
    agent.config.ar_traj_loss_weight=0.0 \
    agent.config.ar_heading_loss_weight=0.0 \
    agent.config.ar_use_residual_delta=false \
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
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_delta_v1213_joint_v12 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_delta_v1213_joint_v12" \
    "$@"
