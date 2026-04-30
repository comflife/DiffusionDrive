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

All training scripts live under [train_eval/](train_eval/). Run them from the repo root:

```bash
cd DiffusionDrive
bash train_eval/<script>.sh
```

Edit the script first to point `NAVSIM_EXP_ROOT`, `OPENSCENE_DATA_ROOT`, `cache_path`, and `output_dir` at your storage.

### Frozen-trunk SFT (V=512, step_delta)

[train_eval/run_training_ar_v512_full_v2_frozen.sh](train_eval/run_training_ar_v512_full_v2_frozen.sh) — minimal config: trunk is frozen, only the AR head trains.

Key overrides:

- `agent.config.freeze_pretrained_trunk=true`
- `agent.config.ego_vocab_size=512`
- `agent.config.ar_codebook_mode=step_delta`

### Joint-trunk SFT (V=2048, step_corners) — recommended

[train_eval/run_training_ar_step_corner_v2048_joint_v2.sh](train_eval/run_training_ar_step_corner_v2048_joint_v2.sh) — current main recipe.

Key overrides:

- `agent.config.freeze_pretrained_trunk=false`
- `agent.config.trunk_lr_mult=0.05` → trunk gets `lr × 0.05` while the AR head keeps full lr
- `agent.config.ego_vocab_size=2048`
- `agent.config.ar_codebook_mode=step_corners`
- `agent.config.ar_use_residual_delta=true`, `ar_use_heading_head=false` (heading comes from the discrete token's `dθ`)
- `agent.config.ar_step_aware_agent=true`
- `agent.config.ar_use_ego_cross_attn=true`
- `agent.config.ar_use_deformable_bev=true`

In joint mode the auxiliary `agent_class_loss`, `agent_box_loss`, and `bev_semantic_loss` are added automatically — see the **Loss** section.

### Other available recipes

[train_eval/](train_eval/) contains a handful of related ablations:

- `run_training_ar_v512_full_v2.sh` — V=512 step_delta, joint trunk
- `run_training_ar_step_corner_v2048_full.sh` — V=2048 step_corners with default heading head
- `run_training_ar_step_corner_v2048_heading_stepagent.sh` — heading head + step-aware agent
- `run_training_ar_step_corner_v2048_heading_stepagent_notf.sh` — same but with `teacher_forcing=false`
- `run_training_ar_step_corner_v2048_scratch.sh` — train from scratch (no pretrained trunk)

## Evaluation

Eval scripts mirror the training recipes and live in the same folder:

- [train_eval/run_eval_ar_v512_full_v2_frozen_latest.sh](train_eval/run_eval_ar_v512_full_v2_frozen_latest.sh)
- [train_eval/run_eval_ar_v512_full_v2_latest.sh](train_eval/run_eval_ar_v512_full_v2_latest.sh)
- [train_eval/run_eval_ar_step_corner_v2048_full_latest.sh](train_eval/run_eval_ar_step_corner_v2048_full_latest.sh)
- [train_eval/run_eval_ar_step_corner_v2048_heading_stepagent_latest.sh](train_eval/run_eval_ar_step_corner_v2048_heading_stepagent_latest.sh)
- [train_eval/run_eval_ar_step_corner_v2048_heading_stepagent_notf_latest.sh](train_eval/run_eval_ar_step_corner_v2048_heading_stepagent_notf_latest.sh)

Edit the `agent.checkpoint_path` line in the script to point at your `last.ckpt`, then:

```bash
bash train_eval/run_eval_ar_step_corner_v2048_full_latest.sh
```

## GRPO Fine-Tuning

After SFT, the AR model can be further tuned with GRPO using PDMS as the reward.

### Training

- [train_eval/run_grpo_training.sh](train_eval/run_grpo_training.sh)
- [train_eval/run_grpo_training_base_t08.sh](train_eval/run_grpo_training_base_t08.sh)

Default GRPO setup (in `run_grpo_training_base_t08.sh`):

- train split for rollouts: `navtest`
- reward: PDMS
- group size: `16`
- sampling temperature: `0.8`
- KL coefficient: `0.1`
- PPO clip epsilon: `0.2`
- learning rate: `1e-6`

Implementation notes:

- rollouts are AR sampling, not teacher forcing
- the loss recomputes log-probs in **teacher-forced** mode on the rollout tokens (`V2TransfuserModelAR.compute_token_log_probs`) so that `log π_θ(a_t | s, a_{<t})` conditions on the rollout — not the model's own AR predictions
- reward is sequence-level PDMS
- KL is computed against a frozen reference model

### Evaluation

- [train_eval/run_eval_grpo_base_t08.sh](train_eval/run_eval_grpo_base_t08.sh)
- [train_eval/run_eval_grpo_latest.sh](train_eval/run_eval_grpo_latest.sh)
- [train_eval/run_eval_grpo_epoch0.sh](train_eval/run_eval_grpo_epoch0.sh)
- [train_eval/run_eval_grpo_converted.sh](train_eval/run_eval_grpo_converted.sh)

Edit the script's `agent.checkpoint_path` to your GRPO `last.ckpt`, then run.

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
