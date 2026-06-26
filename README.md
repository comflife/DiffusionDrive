# DiffusionDrive with Discrete Autoregressive Decoder

This repository implements a discrete autoregressive planner on top of the DiffusionDrive backbone.

The original diffusion decoder is kept as a baseline, but the current main workflow replaces the trajectory head with a **discrete token-based autoregressive decoder**:

1. Load the pretrained DiffusionDrive NAVSIM checkpoint.
2. Either freeze the pretrained trunk and train only the AR head (SFT-frozen), or jointly fine-tune the trunk with a smaller learning rate (SFT-joint).
3. Optionally continue with GRPO using PDMS as the reward.

## What We Changed

We keep the original DiffusionDrive backbone but replace the diffusion-based trajectory head with a discrete autoregressive decoder. At a high level the planner now behaves more like a sequence model over motion primitives than a continuous trajectory regressor:

- the policy predicts **discrete token IDs**
- the token sequence is generated **autoregressively**
- the final reported `(x, y, heading)` trajectory is a **continuous refinement** of that token sequence

The planning decision lives in token space, but the final output is reconstructed from those tokens with continuous residual heads.

## Architecture

The Transfuser backbone (image/lidar encoder + transformer decoder + agent head + BEV semantic head) is reused. Only the trajectory head is swapped for a `DiscreteARTrajectoryHead` defined in [navsim/agents/diffusiondrive/transfuser_model_ar.py](navsim/agents/diffusiondrive/transfuser_model_ar.py).

```
Camera / Lidar
       │
       ▼
┌──────────────────────────────┐
│  Transfuser Backbone         │
│  (image/lidar encoder + BEV) │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│ Transformer Decoder                          │
│  queries → [Trajectory Query, Agents Query]  │
└──────────┬─────────────────┬─────────────────┘
           │                 │
           │          ┌──────┴──────┐
           │          │  AgentHead  │
           │          │(bbox+score) │
           │          └──────┬──────┘
           │                 │
           ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│           Discrete Autoregressive Decoder                   │
│         (DiscreteARTrajectoryHead)                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Input: BOS + ego context (trajectory query)        │   │
│   └────────────────────────┬────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  For t = 0 ... T-1  (autoregressive)                │   │
│   │                                                     │   │
│   │   ┌─────────────┐                                   │   │
│   │   │ Temporal    │  ← causal self-attention          │   │
│   │   │ Self-Attn   │     (only attends 0..t)           │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐ (optional) per-layer ego          │   │
│   │   │ Ego CrossA  │  cross-attn to ego_base           │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐  agent features at step t         │   │
│   │   │ Ego-Agent   │  ← top-K continuous agents        │   │
│   │   │ Cross-Attn  │    (optional step-aware fusion)   │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐  BEV features                     │   │
│   │   │ BEV Cross-  │  flat global, OR                  │   │
│   │   │ Attn        │  waypoint-aware deformable        │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐                                   │   │
│   │   │ FFN         │                                   │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   [Token Logit_t] ──► argmax/sampling ──► token_t   │   │
│   │          │                                          │   │
│   │   token_t ──► embedding ──► input_{t+1}             │   │
│   └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Trajectory Reconstruction                          │   │
│   │   (depends on codebook mode — see below)            │   │
│   │                                                     │   │
│   │   = Final Trajectory (x, y, θ)                      │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

The `ego_token_head` classifies a codebook index at each timestep:

```python
logits = self.ego_token_head(ego_q)   # [B, M, T, V]
tokens = logits.argmax(-1)            # [B, M, T]
```

The token is **not used directly as coordinates**. Instead, a coarse-to-fine post-process refines the trajectory: discrete token decides the coarse motion primitive, residual head fixes the quantization gap, heading head produces orientation.

### Codebook Modes

`DiscreteARTrajectoryHead` supports three codebook modes, switched by `agent.config.ar_codebook_mode`:

| Mode | Codebook shape | Token meaning | Reconstruction |
| --- | --- | --- | --- |
| `step_delta` | `[V, 2]` | per-step `(dx, dy)` displacement in ego frame | `cumsum(token_deltas + residual_deltas)` |
| `step_corners` | `[V, 3]` | per-step local `(dx, dy, dθ)` action | rolled forward through running pose; residual delta on `(dx, dy)` |
| `trajectory_corners` | `[V, T, 3]` | full `(x, y, θ)` trajectory in one token | direct lookup; single-step decode |

In `step_delta` and `step_corners` the residual head adds a small continuous correction on top of the codebook lookup. In `trajectory_corners` the codebook entry is the entire trajectory and there is no residual on top.

### Optional Decoder Components

The AR head exposes several flags to control conditioning strength:

- `ar_use_residual_delta` (default `true`): add residual `(dx, dy)` correction from hidden state.
- `ar_use_heading_head` (default `true`): predict heading from hidden state. When `false` (only meaningful in `step_corners`), the codebook's discrete `dθ` is used directly — heading then has no gradient signal except via token CE.
- `ar_step_aware_agent` (default `false`): nonlinear `(agent, step_emb)` fusion so the agent K/V varies per step.
- `ar_use_ego_cross_attn` (default `false`): per-layer cross-attention to a length-1 ego context, mirroring the original diffusion conditioning.
- `ar_use_deformable_bev` (default `false`): waypoint-aware grid-sample BEV cross-attention instead of global flat attention. Reference points are derived causally from already-decoded tokens.
- `ar_teacher_forcing` (default `true`): teacher forcing for SFT. When `false`, the model is supervised on its own AR rollout.

## Loss

`TransfuserAgentAR.compute_loss` ([navsim/agents/diffusiondrive/transfuser_agent_ar.py](navsim/agents/diffusiondrive/transfuser_agent_ar.py)) builds the total loss in two parts:

1. **Trajectory loss** (always on): the AR head's internal weighted sum
   ```
   trajectory_loss = ar_token_loss_weight   * token_CE
                   + ar_traj_loss_weight    * traj_smoothL1
                   + ar_heading_loss_weight * heading_smoothL1
   ```
2. **Auxiliary loss** (only when `freeze_pretrained_trunk=false`): the original Transfuser supervision on the trunk-side heads
   ```
   aux = agent_class_weight * agent_class_CE
       + agent_box_weight   * agent_box_L1
       + bev_semantic_weight * bev_semantic_CE
   total = trajectory_loss + aux
   ```
   This is required during joint trunk fine-tuning. Without it, `agent_head` (whose output is consumed by the AR head as `agent_kv`) and `bev_semantic_head` would drift, since trajectory_loss alone gives them no direct gradient.

When the trunk is fully frozen, only the AR head trains and the auxiliary terms are skipped (their parameters have `requires_grad=False`, so they would contribute nothing).

## Base Checkpoint

All AR experiments start from the pretrained DiffusionDrive NAVSIM checkpoint:

- `diffusiondrive_navsim_88p1_PDMS`

The trunk weights (backbone, transformer decoder, agent head, BEV semantic head) are loaded from this file. The new AR trajectory head is initialized fresh — its keys are reported as "missing" by `init_from_pretrained`, while the old diffusion head's keys are reported as "unexpected" and silently dropped.

## Codebook

Codebooks live under [codebook_cache/](codebook_cache/). The active ones:

| Path | Mode | Shape | Notes |
| --- | --- | --- | --- |
| `codebook_cache/navsim_kdisk_v512/ego.npy` | `step_delta` | `(512, 2)` | Single-step displacements |
| `codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy` | `step_corners` | `(2048, 4, 2)` | Loaded as `(V, 3)` after corner→`(x, y, θ)` reduction |
| `codebook_cache/navsim_kdisk_v512_diffusiondrive/ego.npy` | `trajectory_corners` | `(512, 6, 4, 2)` | Loaded as `(V, T, 3)` after corner→`(x, y, θ)` reduction |

You can generate a new codebook with the scripts under [create_codebook/](create_codebook/), e.g.:

```bash
python create_codebook/navsim_create_codebook_diffusiondrive.py \
    --data_path /path/to/navsim/logs/trainval \
    --output codebook_cache/navsim_kdisk_v2048_diffusiondrive/ego.npy \
    --vocab_size 2048
```

The exact CLI varies between scripts — read the `argparse` block at the top of each one.

## Training

All scripts live under [train_eval/](train_eval/). Edit the path/env vars in a script first, then run it from the repo root:

```bash
bash train_eval/<script>.sh
```

### SFT (V=2048, step_corners) — current main recipe

[train_eval/ardecoder_train.sh](train_eval/ardecoder_train.sh) jointly fine-tunes the trunk and trains the AR head from the pretrained `diffusiondrive_navsim_88p1_PDMS` checkpoint.

- `freeze_pretrained_trunk=false` (joint), uniform `lr=2e-4`
- `ego_vocab_size=2048`, `ar_codebook_mode=step_corners`, waymo codebook
- residual delta + heading head + step-aware agent + ego cross-attn + deformable BEV + BEV pos enc, `agent_topk=30`
- 150 epochs, milestone checkpoint every 10 epochs from epoch 80

In joint mode the auxiliary `agent_class_loss`, `agent_box_loss`, and `bev_semantic_loss` are added automatically (see **Loss**).

## Evaluation

[train_eval/ardecoder_eval.sh](train_eval/ardecoder_eval.sh) evaluates milestone checkpoints on `navtest` and writes PDMS into a summary CSV. Set `CKPT_DIR`, `START_EPOCH`, `END_EPOCH` to match your training run. The same harness evaluates GRPO checkpoints — point `CKPT_DIR` at the GRPO output and keep the `agent.config.ar_*` overrides matching the checkpoint.

## GRPO Fine-Tuning

After SFT, the AR model can be tuned with RL using PDM Score (PDMS) as the sequence-level reward. The trainer ([navsim/agents/diffusiondrive/grpo_trainer.py](navsim/agents/diffusiondrive/grpo_trainer.py)) keeps a trainable policy and a frozen reference, samples `G` AR rollouts per scene, scores them with PDMS, and optimizes a PPO-clipped policy-gradient loss with optional KL.

Supported algorithms (`++algorithm=<name>`): `grpo`, `dr_grpo`, `gspo`, `dr_gspo`, `gspo_token`, `grpo_plus`, `dr_grpo_plus` — current default is `dr_grpo_plus` (NoRD recipe on GRPO+ with per-token attention weighting from rollout divergence).

### Training

- [train_eval/drgrpoplus_val_train.sh](train_eval/drgrpoplus_val_train.sh) — Dr. GRPO+ on the navtrain VAL split
- [train_eval/drgrpoplus_val_assign_train.sh](train_eval/drgrpoplus_val_assign_train.sh) — same, restricted to loadable val-split assignment scenes

Default setup: base = `v6_waymo` SFT `milestone_epoch_120.ckpt`, group size 12, temperature 1.0, KL 0.0, clip 0.25, lr 1e-5, 35 epochs. AR config matches the SFT base. Override via env vars (`GROUP_SIZE`, `LR`, `MAX_EPOCHS`, `DEVICES`, ...); auto-resumes from `last.ckpt` unless `RESUME_CKPT=none`.

### Evaluation

GRPO checkpoints are standard AR checkpoints — evaluate them with [train_eval/ardecoder_eval.sh](train_eval/ardecoder_eval.sh) (see **Evaluation** above), keeping the `agent.config.ar_*` overrides matching the SFT base.

## Main Files

Core files for the current AR workflow:

- [navsim/agents/diffusiondrive/transfuser_config.py](navsim/agents/diffusiondrive/transfuser_config.py) — config dataclass (AR options live here)
- [navsim/agents/diffusiondrive/transfuser_model_ar.py](navsim/agents/diffusiondrive/transfuser_model_ar.py) — `V2TransfuserModelAR` + `DiscreteARTrajectoryHead`
- [navsim/agents/diffusiondrive/transfuser_agent_ar.py](navsim/agents/diffusiondrive/transfuser_agent_ar.py) — Lightning agent wrapper, optimizer / loss / checkpoint policy
- [navsim/agents/diffusiondrive/transfuser_loss.py](navsim/agents/diffusiondrive/transfuser_loss.py) — agent / BEV auxiliary losses (reused for joint training)
- [navsim/agents/diffusiondrive/grpo_trainer.py](navsim/agents/diffusiondrive/grpo_trainer.py)
- [navsim/agents/diffusiondrive/grpo_train.py](navsim/agents/diffusiondrive/grpo_train.py)
- [navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml](navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml) — default agent config

## Notes

- The repository still contains the original DiffusionDrive codebase (`transfuser_model_v2.py`, `transfuser_agent.py`) for reference.
- When `freeze_pretrained_trunk=false` you must keep the auxiliary losses on (they are added automatically by `compute_loss`); otherwise the agent and BEV heads drift and degrade the AR head's `agent_kv` input.
- `trunk_lr_mult < 1.0` triggers a head/trunk LR split inside `get_coslr_optimizers`: the AR head keeps `agent.lr`, while everything else (backbone, transformer decoder, agent head, BEV semantic head) gets `agent.lr × trunk_lr_mult`. Use this for joint fine-tuning to protect the pretrained trunk.
