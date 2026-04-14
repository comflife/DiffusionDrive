# DiffusionDrive with Discrete Autoregressive Decoder

This repository implements a discrete autoregressive planner on top of the DiffusionDrive backbone.

The original diffusion decoder is kept as a baseline, but the current main workflow replaces the trajectory head with a **discrete token-based autoregressive decoder**:

1. Load the pretrained DiffusionDrive NAVSIM checkpoint.
2. Freeze the pretrained trunk.
3. Train a discrete autoregressive decoder with supervised fine-tuning.
4. Optionally continue with GRPO using PDMS as the reward.

## What We Changed

We maintain the original DiffusionDrive backbone, but the planning head has been replaced with discrete autoregressive decoders.

At a high level, the planner now behaves more like a sequence model over motion primitives than a direct continuous trajectory regressor.

The important distinction is:

- the policy predicts **discrete token IDs**
- the token sequence is generated **autoregressively**
- the final reported trajectory is a **continuous refinement** of that token sequence

So yes, this is a discrete autoregressive decoder. The planning decision itself lives in token space, but the final `(x, y, heading)` trajectory is reconstructed from those tokens with additional continuous refinement heads.

## Baseline AR vs AR+

| Item | Baseline AR | AR+ |
| --- | --- | --- |
| Agent name | `diffusiondrive_ar_agent` | `diffusiondrive_ar_plus_agent` |
| Core planner type | Single-policy discrete AR | Single-policy discrete AR |
| Token space | Ego codebook tokens | Ego codebook tokens |
| Final output | Token-based plan + continuous refinement | Token-based plan + continuous refinement |
| BOS | Ego-centered BOS | Rich BOS from ego + BEV + agent summary + status |
| Agent conditioning | Basic top-k agent context | Stronger agent-state-aware context |
| Time dependency in agent context | Simpler repeated agent context | Step-aware agent context |
| Agent usage during decoding | Cross-attention | Cross-attention + ego-conditioned gating |
| Main goal | Stable discrete AR baseline | Stronger conditioning without diffusion |
| Main files | `transfuser_model_ar.py`, `transfuser_agent_ar.py` | `transfuser_model_ar_plus.py`, `transfuser_agent_ar_plus.py` |

### Baseline AR

The baseline AR model is a single-policy discrete autoregressive decoder.

- `agent=diffusiondrive_ar_agent`
- model file: [navsim/agents/diffusiondrive/transfuser_model_ar.py](navsim/agents/diffusiondrive/transfuser_model_ar.py)
- agent file: [navsim/agents/diffusiondrive/transfuser_agent_ar.py](navsim/agents/diffusiondrive/transfuser_agent_ar.py)
- config file: [navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml](navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml)

Key properties:

- single-mode AR policy: `ar_num_modes=1`
- ego action space is a discrete codebook loaded from `codebook_cache/navsim_kdisk_v512/ego.npy`
- token prediction is autoregressive
- training uses teacher forcing only for SFT next-token prediction
- trajectory reconstruction is not token-only:
  token coarse motion is refined by residual continuous heads
- training loss is:
  token cross-entropy + trajectory SmoothL1 + heading SmoothL1
- pretrained trunk is frozen and only the AR trajectory head is updated

### AR+

AR+ is a stronger discrete AR decoder variant intended to improve conditioning without bringing diffusion back into the planner.

- `agent=diffusiondrive_ar_plus_agent`
- model file: [navsim/agents/diffusiondrive/transfuser_model_ar_plus.py](navsim/agents/diffusiondrive/transfuser_model_ar_plus.py)
- agent file: [navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py](navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py)
- config file: [navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml](navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml)

Compared with baseline AR, AR+ adds:

- richer BOS construction from ego query, BEV summary, agent summary, and status encoding
- stronger agent-state-aware conditioning
- step-aware agent context instead of simple static repetition
- ego-conditioned gating over agent context

This is still a discrete autoregressive planner, not a diffusion planner.

## Architecture

The backbone (Transfuser) is frozen; only the Trajectory Head is replaced by a **Discrete Autoregressive Decoder**.

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
│         (RichDiscreteARTrajectoryHead)                      │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Input: BOS token + Rich Ego Context                │   │
│   │         (ego + BEV summary + agent summary + status)│   │
│   └────────────────────────┬────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  For t = 0 ... T-1  (autoregressive)                │   │
│   │                                                     │   │
│   │   ┌─────────────┐                                   │   │
│   │   │ Temporal    │  ← causal self-attention         │   │
│   │   │ Self-Attn   │     (only attends 0..t)          │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐      agent features at t          │   │
│   │   │ Ego-Agent   │  ←  [query + bbox state + score]  │   │
│   │   │ Cross-Attn  │     with ego-conditioned gating   │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐      BEV grid features            │   │
│   │   │ BEV         │  ←  cross-attention               │   │
│   │   │ Cross-Attn  │                                   │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   ┌─────────────┐                                   │   │
│   │   │ FFN         │                                   │   │
│   │   └──────┬──────┘                                   │   │
│   │          │                                          │   │
│   │          ▼                                          │   │
│   │   [Token Logit_t] ──► argmax ──► token_t           │   │
│   │          │                                          │   │
│   │          └─────────────────────────────────────┐    │   │
│   │                                                │    │   │
│   │   token_t ──► embedding ──► input_{t+1}       │    │   │
│   └────────────────────────────────────────────────┘    │   │
│                            │                              │
│                            ▼                              │
│   ┌─────────────────────────────────────────────────────┐ │
│   │  Output Construction                                │ │
│   │                                                     │ │
│   │   tokens ──► ego_codebook[tokens]   = token_deltas  │ │
│   │            + ego_delta_head(hidden) = residuals     │ │
│   │            ──────────────────────────────────────   │ │
│   │            = final_deltas  ──► cumsum ──► (x, y)    │ │
│   │                                                     │ │
│   │   hidden ──► ego_heading_head()     = heading       │ │
│   │                                                     │ │
│   │            ──────────────────────────────────────   │ │
│   │            = Final Trajectory (x, y, θ)             │ │
│   └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Idea: Codebook Lookup + Residual Head

This is the easiest way to think about the current planner:

1. The decoder predicts a motion token at each future step.
2. Each token maps to a coarse motion primitive in the ego codebook.
3. The decoder hidden state predicts a small residual correction on top of that primitive.
4. A separate head predicts heading.
5. The corrected per-step motions are accumulated into the final trajectory.

So the final output is not "raw coordinates directly from a regression head", and it is also not "token lookup only".

It is:

- discrete motion planning first
- continuous geometric refinement second

That design keeps the planner token-based, while avoiding the large quantization error of a pure token-only rollout.

The decoder's `ego_token_head` classifies a codebook index at each timestep:

```python
logits = self.ego_token_head(ego_q)   # [B, M, T, V]
tokens = logits.argmax(-1)            # [B, M, T]
```

However, the token is **not used directly as coordinates**. Instead, a two-stage post-processing refines the trajectory:

```python
# 1. Codebook Lookup: base displacement for the token
token_deltas = self.ego_codebook[tokens]          # [B, M, T, 2]

# 2. Residual Head: fine adjustment predicted from hidden state
residual_deltas = self.ego_delta_head(hidden)     # [B, M, T, 2]

# 3. Sum and cumsum → absolute coordinates
deltas_xy = token_deltas + residual_deltas
pos_xy = deltas_xy.cumsum(dim=2)                  # [B, M, T, 2]

# 4. Heading is predicted by a separate continuous head
heading = self.ego_heading_head(hidden)           # [B, M, T, 1]
```

This creates a **coarse-to-fine** collaboration: the discrete token decides the coarse direction, and the residual head handles fine-grained correction.

Another way to read this is:

- token sequence = the planner's motion language
- codebook lookup = convert that language into coarse step motions
- residual head = fix the parts the finite codebook cannot represent precisely
- heading head = make the final trajectory orientation-aware

This is why the model is still a discrete AR planner even though the final trajectory is continuous.

## Base Checkpoint

All AR experiments are initialized from the pretrained DiffusionDrive NAVSIM checkpoint:

- `diffusiondrive_navsim_88p1_PDMS`

This checkpoint is used to initialize the trunk. The AR head is newly initialized, and the pretrained trunk is frozen during AR training.

## Codebook

The ego codebook currently used by the AR models is:

- `codebook_cache/navsim_kdisk_v512/ego.npy`

Current assumption:

- shape is effectively `[512, 2]`
- each token represents a single-step local displacement primitive

The AR decoder predicts token sequences over this codebook and then reconstructs continuous trajectories with residual refinement.

Example intuition:

- one token may correspond to "small forward step"
- another may correspond to "forward plus slight left offset"
- another may correspond to "shorter curved step"

The exact vectors are learned by the codebook creation process, but conceptually they behave like a vocabulary of local motion primitives.

You can generate a new codebook with:

```bash
python navsim_create_codebook_diffusiondrive.py \
    --data_path /path/to/navsim/logs/trainval \
    --output codebook_cache/navsim_kdisk_v512/ego.npy \
    --vocab_size 512 \
    --n_trajs 100000 \
    --tol_dist 0.05 \
    --seed 42
```

## Training

### 1. Baseline AR SFT

Script: [run_training_ar.sh](run_training_ar.sh)

Run:

```bash
cd DiffusionDrive
bash run_training_ar.sh
```

Current default settings:

- pretrained checkpoint: `diffusiondrive_navsim_88p1_PDMS`
- trunk freeze: `true`
- batch size: `64` per GPU
- devices: `4`
- learning rate: `2e-4`
- loss weights:
  - token: `1.0`
  - trajectory: `8.0`
  - heading: `2.0`

### 2. AR+ SFT

Script: [run_training_ar_plus.sh](run_training_ar_plus.sh)

Run:

```bash
cd DiffusionDrive
bash run_training_ar_plus.sh
```

This uses the same pretrained trunk and similar hyperparameters, but swaps the decoder to AR+.

## Evaluation

### Baseline AR Evaluation

Script: [run_eval_ar_base_latest.sh](run_eval_ar_base_latest.sh)

Run:

```bash
cd DiffusionDrive
bash run_eval_ar_base_latest.sh
```

Current evaluated checkpoint (edit the script to match your path):

- `/path/to/diffusiondrive-ar/checkpoints/last.ckpt`

### AR+ Evaluation

Script: [run_eval_ar_plus_latest.sh](run_eval_ar_plus_latest.sh)

Run:

```bash
cd DiffusionDrive
bash run_eval_ar_plus_latest.sh
```

Current evaluated checkpoint (edit the script to match your path):

- `/path/to/diffusiondrive-ar-plus/checkpoints/last.ckpt`

## GRPO Fine-Tuning

After SFT, the baseline AR model can be further tuned with GRPO using PDMS as the reward.

### GRPO Setup

Script: [run_grpo_training_base_t08.sh](run_grpo_training_base_t08.sh)

Run:

```bash
cd DiffusionDrive
bash run_grpo_training_base_t08.sh
```

Current GRPO setup (edit the script to match your checkpoint path):

- base checkpoint: `/path/to/diffusiondrive-ar/checkpoints/last.ckpt`
- train split for GRPO rollouts: `navtest`
- reward: PDMS
- group size: `16`
- sampling temperature: `0.8`
- KL coefficient: `0.1`
- PPO clip epsilon: `0.2`
- learning rate: `1e-6`

Implementation notes:

- rollout generation is autoregressive sampling, not GT teacher forcing
- token log-probs are recomputed teacher-forced on sampled trajectories for PPO/GRPO loss
- reward is sequence-level PDMS
- reference KL is computed from the token distributions

### GRPO Evaluation

Script: [run_eval_grpo_base_t08.sh](run_eval_grpo_base_t08.sh)

Run:

```bash
cd DiffusionDrive
bash run_eval_grpo_base_t08.sh
```

Current evaluated checkpoint (edit the script to match your path):

- `/path/to/diffusiondrive_grpo_output/checkpoints/last.ckpt`

Output directory:

- `/path/to/diffusiondrive_grpo_output/eval_latest`

## Main Files

Core files for the current AR workflow:

- [navsim/agents/diffusiondrive/transfuser_config.py](navsim/agents/diffusiondrive/transfuser_config.py)
- [navsim/agents/diffusiondrive/transfuser_model_ar.py](navsim/agents/diffusiondrive/transfuser_model_ar.py)
- [navsim/agents/diffusiondrive/transfuser_agent_ar.py](navsim/agents/diffusiondrive/transfuser_agent_ar.py)
- [navsim/agents/diffusiondrive/transfuser_model_ar_plus.py](navsim/agents/diffusiondrive/transfuser_model_ar_plus.py)
- [navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py](navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py)
- [navsim/agents/diffusiondrive/grpo_trainer.py](navsim/agents/diffusiondrive/grpo_trainer.py)
- [navsim/agents/diffusiondrive/grpo_train.py](navsim/agents/diffusiondrive/grpo_train.py)

## Notes

- The repository still contains the original DiffusionDrive codebase, but this README documents the current discrete AR training flow we are actively using.
- Baseline AR and AR+ are intentionally separated into different agents and scripts for clean experiment tracking.
- GRPO is currently wired to the baseline AR model path. If AR+ becomes the preferred SFT model, the GRPO trainer should be connected to AR+ explicitly.
