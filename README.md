# DiffusionDrive AR Experiments

This repository is currently organized around a discrete autoregressive planner built on top of the DiffusionDrive backbone.

The original diffusion decoder is not the main training target here. The current workflow is:

1. Load the pretrained DiffusionDrive NAVSIM checkpoint.
2. Freeze the pretrained trunk.
3. Train a discrete autoregressive decoder with supervised fine-tuning.
4. Optionally continue with GRPO using PDMS as the reward.

## What We Changed

We maintain the original DiffusionDrive backbone, but the planning head has been replaced or extended with discrete autoregressive decoders.

### Baseline AR

The baseline AR model is a single-policy discrete autoregressive decoder.

- `agent=diffusiondrive_ar_agent`
- model file: [navsim/agents/diffusiondrive/transfuser_model_ar.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_ar.py)
- config file: [navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml](/home/byounggun/DiffusionDrive/navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml)

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
- model file: [navsim/agents/diffusiondrive/transfuser_model_ar_plus.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_ar_plus.py)
- agent file: [navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py)
- config file: [navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml](/home/byounggun/DiffusionDrive/navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml)

Compared with baseline AR, AR+ adds:

- richer BOS construction from ego query, BEV summary, agent summary, and status encoding
- stronger agent-state-aware conditioning
- step-aware agent context instead of simple static repetition
- ego-conditioned gating over agent context

This is still a discrete autoregressive planner, not a diffusion planner.

## Base Checkpoint

All AR experiments are initialized from the pretrained DiffusionDrive NAVSIM checkpoint:

- `/home/byounggun/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS`

This checkpoint is used to initialize the trunk. The AR head is newly initialized, and the pretrained trunk is frozen during AR training.

## Codebook

The ego codebook currently used by the AR models is:

- `/home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy`

Current assumption:

- shape is effectively `[512, 2]`
- each token represents a single-step local displacement primitive

The AR decoder predicts token sequences over this codebook and then reconstructs continuous trajectories with residual refinement.

## Training

### 1. Baseline AR SFT

Script:

- [run_training_ar.sh](/home/byounggun/DiffusionDrive/run_training_ar.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
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

Script:

- [run_training_ar_plus.sh](/home/byounggun/DiffusionDrive/run_training_ar_plus.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_training_ar_plus.sh
```

This uses the same pretrained trunk and similar hyperparameters, but swaps the decoder to AR+.

### 3. Decoder Tuning From a Trained Baseline AR Checkpoint

Script:

- [run_training_ar_from_base.sh](/home/byounggun/DiffusionDrive/run_training_ar_from_base.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_training_ar_from_base.sh
```

This starts from the trained baseline AR checkpoint:

- `/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt`

and continues tuning the AR decoder while keeping the trunk frozen.

## Evaluation

### Baseline AR Evaluation

Script:

- [run_eval_ar_base_latest.sh](/home/byounggun/DiffusionDrive/run_eval_ar_base_latest.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_eval_ar_base_latest.sh
```

Current evaluated checkpoint:

- `/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt`

### AR+ Evaluation

Script:

- [run_eval_ar_plus_latest.sh](/home/byounggun/DiffusionDrive/run_eval_ar_plus_latest.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_eval_ar_plus_latest.sh
```

Current evaluated checkpoint:

- `/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/77qd0ttw/checkpoints/last.ckpt`

### Stochastic AR Evaluation

Script:

- [run_eval_ar_temperature.sh](/home/byounggun/DiffusionDrive/run_eval_ar_temperature.sh)

This is useful when checking how stochastic sampling changes AR behavior under a nonzero decoding temperature.

## GRPO Fine-Tuning

After SFT, the baseline AR model can be further tuned with GRPO using PDMS as the reward.

### GRPO Setup

Script:

- [run_grpo_training_base_t08.sh](/home/byounggun/DiffusionDrive/run_grpo_training_base_t08.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_grpo_training_base_t08.sh
```

Current GRPO setup:

- base checkpoint:
  `/data2/byounggun/diffusiondrive_ar_output/diffusiondrive-ar/24l0pgz4/checkpoints/last.ckpt`
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

Script:

- [run_eval_grpo_base_t08.sh](/home/byounggun/DiffusionDrive/run_eval_grpo_base_t08.sh)

Run:

```bash
cd /home/byounggun/DiffusionDrive
bash run_eval_grpo_base_t08.sh
```

Current evaluated checkpoint:

- `/data2/byounggun/diffusiondrive_grpo_output_base_t08/checkpoints/last.ckpt`

Output directory:

- `/data2/byounggun/diffusiondrive_grpo_output_base_t08/eval_latest`

## Main Files

Core files for the current AR workflow:

- [navsim/agents/diffusiondrive/transfuser_config.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_config.py)
- [navsim/agents/diffusiondrive/transfuser_model_ar.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_ar.py)
- [navsim/agents/diffusiondrive/transfuser_agent_ar.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_agent_ar.py)
- [navsim/agents/diffusiondrive/transfuser_model_ar_plus.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_model_ar_plus.py)
- [navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/transfuser_agent_ar_plus.py)
- [navsim/agents/diffusiondrive/grpo_trainer.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/grpo_trainer.py)
- [navsim/agents/diffusiondrive/grpo_train.py](/home/byounggun/DiffusionDrive/navsim/agents/diffusiondrive/grpo_train.py)

## Notes

- The repository still contains the original DiffusionDrive codebase, but this README documents the current discrete AR training flow we are actively using.
- Baseline AR and AR+ are intentionally separated into different agents and scripts for clean experiment tracking.
- GRPO is currently wired to the baseline AR model path. If AR+ becomes the preferred SFT model, the GRPO trainer should be connected to AR+ explicitly.
