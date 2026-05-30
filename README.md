# PPO (Proximal Policy Optimization)

LunarLander, Pendulum, Ant 환경에 대한 PPO 구현 및 학습/평가.

## 환경 세팅

### 1. Conda 환경 생성

```bash
conda create -n ppo python=3.10 -y
```

### 2. 패키지 설치

```bash
conda activate ppo
pip install torch gymnasium[box2d] matplotlib numpy moviepy
```

### 설치된 주요 패키지 버전

| 패키지 | 버전 |
|--------|------|
| Python | 3.10 |
| PyTorch | 2.12.0 |
| Gymnasium | 1.3.0 |
| NumPy | 2.2.6 |
| Matplotlib | 3.10.9 |
| MoviePy | (영상 저장용) |

## 파일 구조

```
ppo/
├── README.md
├── ppo.py                  # PPO 알고리즘 (Actor-Critic, GAE, Buffer)
├── train_lunarlander.py    # LunarLander-v3 학습
└── evaluate.py             # 학습된 모델 평가 + 영상 저장
```

## 사용법

### 1. LunarLander-v3 (Discrete action)

#### 학습

```bash
conda activate ppo
python train_lunarlander.py
```

- 총 1,000,000 step 학습
- 학습 완료 시 저장:
  - `ppo_lunarlander.pt` — 모델 가중치
  - `lunarlander_training.png` — 학습 곡선

#### 평가

```bash
python evaluate.py --env LunarLander-v3 --model ppo_lunarlander.pt --episodes 100
```

#### 영상 저장

```bash
python evaluate.py --env LunarLander-v3 --model ppo_lunarlander.pt --record --episodes 5
```

`videos/` 폴더에 에피소드별 mp4 파일 저장. `--video-dir` 옵션으로 저장 경로 변경 가능.

### 2. Pendulum (Continuous action)

TODO

### 3. Ant (Continuous action, high-dim)

TODO
