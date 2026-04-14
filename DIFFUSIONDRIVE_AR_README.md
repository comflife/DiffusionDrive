# DiffusionDrive with Discrete Autoregressive Decoder

VAD의 Discrete AR Decoder 방식을 DiffusionDrive에 적용한 구현입니다.

기존 DiffusionDrive의 Diffusion-based Trajectory Head를 **Discrete Token 기반 Autoregressive Decoder**로 교체했습니다.
Perception backbone (TransfuserBackbone)과 BEV representation은 그대로 유지하며, 궤적 생성 부분만 AR로 전환합니다.

## 변경 사항 요약

### 1. 기존 DiffusionDrive (Diffusion-based)
- **Trajectory Representation**: Continuous coordinates (x, y, θ)
- **Generation**: DDIM diffusion process (denoising)
- **Loss**: MSE on noise / trajectory
- **Training**: Predict noise or direct coordinates at multiple timesteps

### 2. 새로운 DiffusionDrive-AR (Discrete Token-based)
- **Trajectory Representation**: Discrete tokens (codebook indices)
- **Generation**: Autoregressive token prediction (step-by-step)
- **Loss**: CrossEntropy on token classification + Smooth L1 on continuous refinement
- **Training**: Teacher forcing with BOS-shifted input

## 구현 파일 구조

```
navsim/agents/diffusiondrive/
├── discrete_ar_head.py              # Standalone 초기 버전 AR Head
├── transfuser_model_ar.py           # 기본 AR 모델 (V2TransfuserModelAR)
├── transfuser_agent_ar.py           # 기본 AR Agent (TransfuserAgentAR)
├── transfuser_model_ar_plus.py      # 향상된 AR 모델 (V2TransfuserModelARPlus)
├── transfuser_agent_ar_plus.py      # 향상된 AR Agent (TransfuserAgentARPlus)
└── __init__.py

navsim/planning/script/config/common/agent/
├── diffusiondrive_ar_agent.yaml     # 기본 AR Agent 설정
└── diffusiondrive_ar_plus_agent.yaml # AR Plus Agent 설정

codebook_cache/
├── navsim_kdisk_v256/
├── navsim_kdisk_v512/
├── navsim_kdisk_v1024/
├── navsim_kdisk_v2048/
└── v2/
```

> **Note**: 현재 메인으로 사용하는 버전은 **AR Plus** (`transfuser_model_ar_plus.py`)입니다. 기본 AR (`transfuser_model_ar.py`)은 이전 버전이며, AR Plus는 더 풍부한 BOS seed와 ego-conditioned agent gating을 포함합니다.

## 전체 아키텍처 도식

```
Camera / Lidar
       │
       ▼
┌─────────────────┐
│ Transfuser      │
│ Backbone        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ BEV Feature Map │      │ Vehicle Status   │
│ [B, D, H, W]    │      │ Encoding         │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│ Transformer Decoder                         │
│  (key/val = BEV + status, query = learned)  │
└──────────────┬──────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
[Trajectory Query]  [Agents Query]
      │                 │
      ▼                 ▼
      │          ┌──────────────┐
      │          │ AgentHead    │
      │          │ (bbox + score│
      │          │  prediction) │
      │          └──────┬───────┘
      │                 │
      ▼                 ▼
┌─────────────────────────────────────────────┐
│ RichDiscreteARTrajectoryHead (AR Plus)      │
│                                               │
│  1. Select top-K agents by distance/score   │
│  2. Build rich agent KV (query + state emb) │
│  3. Build rich BOS (ego + BEV + agent       │
│     summary + status)                        │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │ AR Attention Stack (repeat L times) │    │
│  │                                     │    │
│  │  a) Temporal Self-Attention         │    │
│  │     (causal mask)                   │    │
│  │                                     │    │
│  │  b) Ego-Agent Cross-Attention       │    │
│  │     with Ego-Conditioned Gating     │    │
│  │     q=ego_t, kv=agent_t             │    │
│  │                                     │    │
│  │  c) BEV Cross-Attention             │    │
│  │     q=ego_seq, kv=BEV_grid          │    │
│  │                                     │    │
│  │  d) FFN                             │    │
│  └─────────────────────────────────────┘    │
│               │                               │
│               ▼                               │
│      [Ego Token Logits]  [B, M, T, V]        │
│               │                               │
│               ▼ (argmax / sampling)           │
│      [Discrete Tokens]   [B, M, T]           │
│               │                               │
│     ┌─────────┴──────────┐                    │
│     ▼                    ▼                    │
│ Codebook Lookup     Residual Head            │
│ (ego_codebook)      (ego_delta_head)         │
│     ▼                    ▼                    │
│ token_deltas       residual_deltas           │
│     └─────────┬──────────┘                    │
│               ▼ (+ cumsum)                    │
│      (x, y) trajectory                       │
│               │                               │
│     ┌─────────┴──────────┐                    │
│     ▼                    ▼                    │
│ Heading Head          Final Trajectory       │
│ (ego_heading_head)    (x, y, θ)              │
└─────────────────────────────────────────────┘
```

### 핵심 아이디어: Codebook Lookup + Residual Head

Decoder의 `ego_token_head`는 각 timestep마다 코드북 인덱스를 분류합니다:
```python
logits = self.ego_token_head(ego_q)   # [B, M, T, V]
tokens = logits.argmax(-1)            # [B, M, T]
```

하지만 토큰을 바로 좌표로 쓰는 것이 아니라, **2단계 후처리**를 거칩니다:

```python
# 1. Codebook Lookup: 토큰에 해당하는 기본 displacement
 token_deltas = self.ego_codebook[tokens]          # [B, M, T, 2]

# 2. Residual Head: hidden state로 미세 조정을 예측
 residual_deltas = self.ego_delta_head(hidden)     # [B, M, T, 2]

# 3. 합산 후 누적 → 절대 좌표
 deltas_xy = token_deltas + residual_deltas
 pos_xy = deltas_xy.cumsum(dim=2)                  # [B, M, T, 2]

# 4. Heading은 별도 Head로 직접 예측
 heading = self.ego_heading_head(hidden)           # [B, M, T, 1]
```

이렇게 하면 discrete token이 **대략적인 방향(coarse)**을 결정하고, residual head가 **미세 조정(fine)**을 담당하는 협업 구조가 됩니다.

## 설치 및 환경 설정

### 1. 기존 DiffusionDrive 환경 사용
```bash
cd /home/byounggun/DiffusionDrive
conda activate navsim  # 기존 환경
```

### 2. 필요한 패키지 확인
```bash
pip install numpy torch matplotlib tqdm
```

## Codebook 생성

### NavSim 데이터에서 Codebook 생성
```bash
python navsim_create_codebook_diffusiondrive.py \
    --data_path /path/to/navsim/logs/trainval \
    --output codebook_cache/navsim_kdisk_v512/ego.npy \
    --vocab_size 512 \
    --n_trajs 100000 \
    --tol_dist 0.05 \
    --seed 42
```

### 파라미터 설명
- `--vocab_size`: Codebook 크기 (default: 512)
  - 작을수록: 더 제한적인 동작, 빠른 학습
  - 클수록: 더 다양한 동작, 느린 학습
- `--tol_dist`: K-disk clustering tolerance (default: 0.05m)
  - 작을수록: 더 정밀한 클러스터링, 더 많은 고유 토큰
  - 클수록: 더 넓은 클러스터링, 더 적은 토큰

## 훈련 방법

### 1. 설정 파일 수정
`navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml`에서 codebook 경로를 확인/수정합니다:

```yaml
config:
  ego_vocab_path: /home/byounggun/DiffusionDrive/codebook_cache/navsim_kdisk_v512/ego.npy
```

### 2. 훈련 실행
```bash
cd /home/byounggun/DiffusionDrive
python -m navsim.planning.script.run_training \
    training=default_training \
    agent=diffusiondrive_ar_plus_agent \
    split=trainval \
    experiment_name=diffusiondrive_ar_plus_experiment
```

### 3. 분산 학습 (Multi-GPU)
```bash
python -m navsim.planning.script.run_training \
    training=default_training \
    agent=diffusiondrive_ar_plus_agent \
    split=trainval \
    experiment_name=diffusiondrive_ar_plus_distributed \
    trainer.params.num_nodes=1 \
    trainer.params.devices=4
```

## 평가 (Evaluation) 방법

프로젝트 루트에 준비된 스크립트를 실행합니다:

```bash
# AR Plus 평가
bash run_eval_ar_plus_latest.sh

# 기본 AR 평가
bash run_eval_ar_base_latest.sh
```

각 스크립트는 `navsim.planning.script.run_pdm_score`를 호출하여 PDMS 점수를 계산합니다.

## 모델 아키텍처 비교

### Original DiffusionDrive
```
Camera/Lidar → Backbone → BEV Features → Transformer Decoder
                                               │
                    Plan Anchors → Diffusion Decoder → Trajectory
                                               │
                                         Time Embedding
```

### DiffusionDrive-AR (기본)
```
Camera/Lidar → Backbone → BEV Features → Transformer Decoder
                                               │
                    BOS Token → AR Decoder → Token 1 → Token 2 → ... → Token T
                                     │          │
                              Agent KV Cache    │
                                     │      Codebook Lookup
                              BEV Features      │
                                         Trajectory (cumsum + residual)
```

### DiffusionDrive-AR Plus (현재 메인)
AR Plus는 기본 AR에 다음 향상 기능을 추가합니다:
- **Rich BOS Seed**: `ego_query`뿐만 아니라 BEV summary, agent summary, vehicle status를 fuse하여 더 풍부한 초기 임베딩 생성
- **Rich Agent KV**: agent query feature에 실제 bounding box 상태(`[x, y, heading, length, width]`)와 detection score를 추가 인코딩
- **Ego-Conditioned Agent Gating**: cross-attention 직전에 ego query와 agent key의 관련성을 계산하여 중요한 agent만 선택적으로 강조

## 주요 구현 세부사항

### 1. DiscreteARTrajectoryHead / RichDiscreteARTrajectoryHead
- **Temporal Self-Attention**: Causal masking으로 AR 속성 보장
- **Ego-Agent Cross-Attention**: 각 timestep에서 agent token과 상호작용 (AR Plus는 gating 추가)
- **BEV Cross-Attention**: BEV features와 상호작용
- **Token Embedding**: Discrete token indices → continuous embeddings

### 2. Teacher Forcing (Training)
```python
# BOS-shifted input
input[t=0] = BOS embedding + rich ego context
input[t>0] = embedding(GT[t-1])  # Previous GT token
target[t] = GT[t]                 # Current GT token
```

### 3. Autoregressive Inference
```python
# Step-by-step generation
for t in range(T):
    logits = model(input[:, :, :t+1, :])
    token[t] = argmax(logits[:, :, t, :])
    input[:, :, t+1, :] = embedding(token[t])
```

### 4. AR Plus의 손실 함수
```python
loss = (
    token_loss_weight   * token_loss    # CrossEntropy
  + traj_loss_weight    * traj_loss     # Smooth L1 on (x,y)
  + heading_loss_weight * heading_loss  # Smooth L1 on heading
)
```

## 성능 비교 (예상)

| Metric | DiffusionDrive | DiffusionDrive-AR | DiffusionDrive-AR Plus |
|--------|----------------|-------------------|------------------------|
| Inference Speed | ~10 steps (DDIM) | 8 steps (T tokens) | 8 steps (T tokens) |
| Memory Usage | High (multiple timesteps) | Lower | Lower |
| Trajectory Diversity | High | Medium (constrained by codebook) | Medium |
| Training Stability | Moderate | Higher (CE loss) | Higher (CE + L1) |
| Long-horizon Consistency | Moderate | Better (AR nature) | Better (AR nature) |

## 디버깅 및 문제 해결

### 1. NaN loss 발생 시
```python
# Learning rate 감소
lr: 1e-4  # instead of 6e-4

# Gradient clipping 확인
gradient_clip_val: 1.0
```

### 2. Poor trajectory quality
- Codebook 크기 증가: `ego_vocab_size: 1024`
- Tolerance 조정: `tol_dist: 0.03` (더 정밀한 클러스터링)

### 3. Slow convergence
- Agent KV cache 사이즈 조정: `agent_topk: 4` (instead of 8)
- Backbone frozen 여부 확인: `freeze_pretrained_trunk: true`로 trunk 고정하고 AR head만 학습

## 확장 가능성

### 1. Multi-modal Ego Planning
현재 `ego_fut_mode`로 여러 trajectory 모드 생성 가능. 각 모드별로 별도의 AR sequence 생성.

### 2. Hierarchical AR
```
High-level: Goal token (long-term)
Mid-level: Waypoint tokens (medium-term)
Low-level: Control tokens (short-term)
```

### 3. Agent Trajectory AR
현재는 ego trajectory만 AR로 생성. Agent trajectories도 AR로 생성 가능 (더 느리지만 더 정확).

## 참고 자료

- VAD Discrete AR: [`VAD/projects/mmdet3d_plugin/VAD/VAD_discrete_ar.py`](VAD/projects/mmdet3d_plugin/VAD/VAD_discrete_ar.py)
- AutoVLA Tokenization: [`action_token_cluster.py`](action_token_cluster.py)
- Original DiffusionDrive: [`navsim/agents/diffusiondrive/transfuser_model_v2.py`](navsim/agents/diffusiondrive/transfuser_model_v2.py)
- 기본 AR Model: [`navsim/agents/diffusiondrive/transfuser_model_ar.py`](navsim/agents/diffusiondrive/transfuser_model_ar.py)
- AR Plus Model: [`navsim/agents/diffusiondrive/transfuser_model_ar_plus.py`](navsim/agents/diffusiondrive/transfuser_model_ar_plus.py)
- AR Plus Agent Config: [`navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml`](navsim/planning/script/config/common/agent/diffusiondrive_ar_plus_agent.yaml)

## Citation

```bibtex
@article{diffusiondrive_ar,
  title={DiffusionDrive with Discrete Autoregressive Decoder},
  year={2025}
}
```
