import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo import ActorCritic, PPO

# Hyperparameters
SEED = 42
TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_STEPS = 2048
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 1.0
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 64
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, hidden_dim=HIDDEN_DIM)
    agent = PPO(
        model,
        lr=LR,
        gamma=GAMMA,
        lam=LAM,
        clip_eps=CLIP_EPS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        update_epochs=UPDATE_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        device=DEVICE,
    )

    all_rewards = []
    avg_rewards = []
    total_steps = 0
    iteration = 0

    print(f"Training PPO on LunarLander-v3 | Device: {DEVICE}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print("-" * 60)

    while total_steps < TOTAL_TIMESTEPS:
        episode_rewards, next_value = agent.collect_rollout(env, ROLLOUT_STEPS)
        total_steps += ROLLOUT_STEPS
        iteration += 1

        policy_loss, value_loss, entropy = agent.update(next_value)

        all_rewards.extend(episode_rewards)
        if len(all_rewards) > 0:
            recent_avg = np.mean(all_rewards[-100:])
            avg_rewards.append(recent_avg)
        else:
            avg_rewards.append(0)

        if iteration % 5 == 0 or len(episode_rewards) > 0:
            n_eps = len(episode_rewards)
            ep_avg = np.mean(episode_rewards) if episode_rewards else float("nan")
            print(
                f"Step {total_steps:>7d} | Iter {iteration:>3d} | "
                f"Episodes: {n_eps:>3d} | Avg: {ep_avg:>8.1f} | "
                f"Last100: {np.mean(all_rewards[-100:]):>8.1f} | "
                f"PL: {policy_loss:.4f} | VL: {value_loss:.4f} | Ent: {entropy:.4f}"
            )

    env.close()

    # Save model
    torch.save(model.state_dict(), "ppo_lunarlander.pt")
    print(f"\nModel saved to ppo_lunarlander.pt")

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, alpha=0.3, label="Episode Reward")
    window = min(100, len(all_rewards))
    if len(all_rewards) >= window:
        smoothed = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(all_rewards)), smoothed, label=f"{window}-ep Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO - LunarLander-v3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lunarlander_training.png", dpi=150)
    print("Training curve saved to lunarlander_training.png")


if __name__ == "__main__":
    train()
