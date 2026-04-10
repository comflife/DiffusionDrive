# DiffusionDrive with Discrete Autoregressive Decoder

VAD의 Discrete AR Decoder 방식을 DiffusionDrive에 적용한 구현입니다.

## 변경 사항 요약

### 1. 기존 DiffusionDrive (Diffusion-based)
- **Trajectory Representation**: Continuous coordinates (x, y, θ)
- **Generation**: DDIM diffusion process (denoising)
- **Loss**: MSE on noise / trajectory
- **Training**: Predict noise or direct coordinates at multiple timesteps

### 2. 새로운 DiffusionDrive-AR (Discrete Token-based)
- **Trajectory Representation**: Discrete tokens (codebook indices)
- **Generation**: Autoregressive token prediction (step-by-step)
- **Loss**: CrossEntropy on token classification
- **Training**: Teacher forcing with BOS-shifted input

## 구현 파일 구조

```
VAD/navsim/navsim/agents/diffusiondrive/
├── discrete_ar_head.py           # Discrete AR Head 구현
├── transfuser_model_ar.py        # AR 모델 (V2TransfuserModelAR)
├── transfuser_agent_ar.py        # AR Agent (TransfuserAgentAR)
└── __init__.py                   # 모듈 export 업데이트

VAD/navsim/navsim/planning/script/config/common/agent/
└── diffusiondrive_ar_agent.yaml  # Agent 설정

VAD/navsim/navsim/planning/script/config/training/
└── diffusiondrive_ar_training.yaml  # Training 설정

codebook_cache/                   # Codebook 저장 위치
├── diffusiondrive_ego_vocab.npy
├── vehicle_vocab.npy
├── cyclist_vocab.npy
└── pedestrian_vocab.npy
```

## 설치 및 환경 설정

### 1. 기존 DiffusionDrive 환경 사용
```bash
cd VAD/navsim
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
    --output codebook_cache/diffusiondrive_ego_vocab.npy \
    --vocab_size 256 \
    --n_trajs 100000 \
    --tol_dist 0.05 \
    --seed 42
```

### 파라미터 설명
- `--vocab_size`: Codebook 크기 (default: 256)
  - 작을수록: 더 제한적인 동작, 빠른 학습
  - 클수록: 더 다양한 동작, 느린 학습
- `--tol_dist`: K-disk clustering tolerance (default: 0.05m)
  - 작을수록: 더 정밀한 클러스터링, 더 많은 고유 토큰
  - 클수록: 더 넓은 클러스터링, 더 적은 토큰

## 훈련 방법

### 1. 설정 파일 수정
`VAD/navsim/navsim/planning/script/config/common/agent/diffusiondrive_ar_agent.yaml`에서 codebook 경로 설정:

```yaml
config:
  ego_vocab_path: /actual/path/to/ego_vocab.npy
  vehicle_vocab_path: /actual/path/to/vehicle_vocab.npy
  cyclist_vocab_path: /actual/path/to/cyclist_vocab.npy
  pedestrian_vocab_path: /actual/path/to/pedestrian_vocab.npy
```

### 2. 훈련 실행
```bash
cd VAD/navsim
python -m navsim.planning.script.run_training \
    training=diffusiondrive_ar_training \
    agent=diffusiondrive_ar_agent \
    split=trainval \
    experiment_name=diffusiondrive_ar_experiment \
    +output_dir=${NAVSIM_EXP_ROOT}/diffusiondrive_ar_output
```

### 3. 분산 학습 (Multi-GPU)
```bash
python -m navsim.planning.script.run_training \
    training=diffusiondrive_ar_training \
    agent=diffusiondrive_ar_agent \
    split=trainval \
    experiment_name=diffusiondrive_ar_distributed \
    trainer.params.num_nodes=1 \
    trainer.params.devices=4  # 4 GPUs
```

## 모델 아키텍처 비교

### Original DiffusionDrive
```
Camera/Lidar → Backbone → BEV Features → Transformer Decoder
                                               ↓
                    Plan Anchors → Diffusion Decoder → Trajectory
                                               ↑
                                         Time Embedding
```

### DiffusionDrive-AR
```
Camera/Lidar → Backbone → BEV Features → Transformer Decoder
                                               ↓
                    BOS Token → AR Decoder → Token 1 → Token 2 → ... → Token T
                                     ↑          ↓
                              Agent KV Cache    ↓
                                     ↑      Codebook Lookup
                              BEV Features      ↓
                                         Trajectory (cumsum)
```

## 주요 구현 세부사항

### 1. DiscreteARTrajectoryHead
- **Temporal Self-Attention**: Causal masking으로 AR 속성 보장
- **Ego-Agent Cross-Attention**: 각 timestep에서 agent token과 상호작용
- **BEV Cross-Attention**: BEV features와 상호작용
- **Token Embedding**: Discrete token indices → continuous embeddings

### 2. Teacher Forcing (Training)
```python
# BOS-shifted input
input[t=0] = BOS embedding + ego context
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

## 성능 비교 (예상)

| Metric | DiffusionDrive | DiffusionDrive-AR |
|--------|----------------|-------------------|
| Inference Speed | ~10 steps (DDIM) | 8 steps (T tokens) |
| Memory Usage | High (multiple timesteps) | Lower |
| Trajectory Diversity | High | Medium (constrained by codebook) |
| Training Stability | Moderate | Higher (CE loss) |
| Long-horizon Consistency | Moderate | Better (AR nature) |

## 디버깅 및 문제 해결

### 1. NaN loss 발생 시
```python
# Learning rate 감소
lr: 1e-4  # instead of 6e-4

# Gradient clipping 확인
gradient_clip_val: 1.0
```

### 2. Poor trajectory quality
- Codebook 크기 증가: `vocab_size: 512` or `1024`
- Tolerance 조정: `tol_dist: 0.03` (더 정밀한 클러스터링)

### 3. Slow convergence
- Agent KV cache 사이즈 조정: `agent_topk: 4` (instead of 8)
- Layer 수 조정: `num_layers: 1` (faster but less expressive)

## 확장 가능성

### 1. Multi-modal Ego Planning
현재 `ego_fut_mode=20`로 여러 trajectory 모드 생성 가능. 각 모드별로 별도의 AR sequence 생성.

### 2. Hierarchical AR
```
High-level: Goal token (long-term)
Mid-level: Waypoint tokens (medium-term)
Low-level: Control tokens (short-term)
```

### 3. Agent Trajectory AR
현재는 ego trajectory만 AR로 생성. Agent trajectories도 AR로 생성 가능 (더 느리지만 더 정확).

## 참고 자료

- VAD Discrete AR: `VAD/projects/mmdet3d_plugin/VAD/VAD_discrete_ar.py`
- AutoVLA Tokenization: `action_token_cluster.py`
- Original DiffusionDrive: `transfuser_model_v2.py`

## Citation

```bibtex
@article{diffusiondrive_ar,
  title={DiffusionDrive with Discrete Autoregressive Decoder},
  year={2025}
}
```
