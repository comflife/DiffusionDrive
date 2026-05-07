#!/bin/bash
# Train DiffusionDrive-AR — JOINT trunk training v7 (single-call agent attention):
#   - based on v6 (deformable BEV + ego cross-attn + 2D BEV pos enc + agent_topk=30)
#   - v7 change: ar_step_aware_agent -> OFF
#   - Effect: ego-agent cross-attention is now a SINGLE MHA call per layer
#       Q=[B*M, T, D], K=V=[B*M, K, D]   (K/V shared across T, no step_emb on K/V)
#     instead of a T-step Python loop with per-step K/V.
#   - This call shape is identical to the original DiffusionDrive
#     cross_agent_attention, so the warm-started diff_decoder.cross_agent_attention
#     weights map onto the same Q/K/V interaction with no input-distribution
#     shift. AR temporal dependency is still enforced by the causal self-attn.
#   - K/V no longer carries step_emb; step info lives on the ego query side via
#     step_emb (matches v2 conditioning style).
#   - Compute drops accordingly: agent attention is now O(layers) MHA calls
#     per forward instead of O(layers*T).
# Hypothesis: removing the per-timestep agent loop + dropping step_emb on the
# agent K/V side makes the warm-started cross_agent_attention weights apply
# more cleanly, recovering more of the 88.1 PDMS pretrain signal.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NAVSIM_DEVKIT_ROOT=/home/byounggun/DiffusionDrive
export NAVSIM_EXP_ROOT=/data/navsim/exp/bg
export OPENSCENE_DATA_ROOT=/data/navsim/dataset
export NUPLAN_MAPS_ROOT=/data/navsim/dataset/maps
export TMPDIR=/data2/byounggun/ray_tmp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

cd $NAVSIM_DEVKIT_ROOT

echo "Starting DiffusionDrive-AR step-corner v2048 JOINT v7 (single-call agent attention)..."
echo "Codebook    : codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy  (V=2048, corner-> [V,3])"
echo "Mode        : step_corners"
echo "Refinement  : residual delta ON, heading head OFF"
echo "Agent       : SINGLE-CALL (step_aware=false), K/V shared across T, no step_emb on K/V"
echo "Conditioning: per-layer ego cross-attn ON, deformable BEV ON"
echo "BEV path    : causal prefix trajectory-conditioned deformable sampling"
echo "Warm-start  : diff_decoder.cross_{bev,agent,ego}_attn / ffn / norms -> AR head"
echo "Agent K/V   : agent_topk=30 (matches original cross_agent_attention)"
echo "LR          : uniform 2e-4 across head + trunk (no trunk lr multiplier)"
echo "Schedule    : 150 epochs, cosine LR matched to 150"
echo "Snapshots   : milestone every 10 epochs starting from epoch 80"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"

python -m navsim.planning.script.run_training \
    agent=diffusiondrive_ar_agent \
    train_test_split=navtrain \
    cache_path="/data2/byounggun/training_cache" \
    force_cache_computation=false \
    +experiment_name=diffusiondrive_ar_step_corner_v2048_joint_v7 \
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
    agent.config.ar_step_aware_agent=false \
    agent.config.ar_use_ego_cross_attn=true \
    agent.config.ar_use_deformable_bev=true \
    agent.config.ar_use_bev_pos_enc=true \
    agent.config.freeze_pretrained_trunk=false \
    agent.config.cos_lr_epochs=150 \
    agent.config.ckpt_milestone_start=80 \
    agent.config.ckpt_milestone_every=10 \
    output_dir=/data2/byounggun/diffusiondrive_ar_output/step_corner_v2048_joint_v7 \
    wandb.enabled=true \
    wandb.project="diffusiondrive-ar" \
    wandb.name="diffusiondrive_ar_step_corner_v2048_joint_v7" \
    "$@"
