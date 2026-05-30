# PPO (Proximal Policy Optimization)

PPO implementation for LunarLander, Pendulum, and Ant environments with training and evaluation.

## Setup

```bash
conda create -n ppo python=3.10 -y
conda activate ppo
pip install torch gymnasium[box2d] matplotlib numpy moviepy
```


## Usage

- LunarLander-v3 (Discrete action)

```bash
python train_lunarlander.py

python evaluate.py --env LunarLander-v3 --model ppo_lunarlander.pt --episodes 100

# save video
python evaluate.py --env LunarLander-v3 --model ppo_lunarlander.pt --record --episodes 5
```


- Pendulum-v1 (Continuous action)


```bash
python train_pendulum.py

python evaluate.py --env Pendulum-v1 --model ppo_pendulum.pt --episodes 100

# save video
python evaluate.py --env Pendulum-v1 --model ppo_pendulum.pt --record --episodes 5
```

- Ant (Continuous action, high-dim)

TODO
