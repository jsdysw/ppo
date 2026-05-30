import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from ppo import ActorCritic, ActorCriticContinuous


def evaluate(env_name, model_path, n_episodes=100, render=False, record=False, video_dir="videos"):
    render_mode = "human" if render else ("rgb_array" if record else None)
    env = gym.make(env_name, render_mode=render_mode)
    if record:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)
    obs_dim = env.observation_space.shape[0]

    if hasattr(env.action_space, "n"):
        act_dim = env.action_space.n
        model = ActorCritic(obs_dim, act_dim)
    else:
        act_dim = env.action_space.shape[0]
        model = ActorCriticContinuous(obs_dim, act_dim)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if isinstance(model, ActorCritic):
                    logits, _ = model(obs_tensor)
                    action = logits.argmax(dim=-1).item()
                else:
                    mean, _, _ = model(obs_tensor)
                    action = mean.cpu().numpy().flatten()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:>3d} | Reward: {total_reward:>8.1f} | Avg: {np.mean(rewards):>8.1f}")

    env.close()
    print(f"\n{'='*40}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  Mean:   {np.mean(rewards):.1f}")
    print(f"  Std:    {np.std(rewards):.1f}")
    print(f"  Min:    {np.min(rewards):.1f}")
    print(f"  Max:    {np.max(rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--model", default="ppo_lunarlander.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record", action="store_true", help="Save episodes as mp4 videos")
    parser.add_argument("--video-dir", default="videos", help="Directory to save videos")
    args = parser.parse_args()
    evaluate(args.env, args.model, args.episodes, args.render, args.record, args.video_dir)
