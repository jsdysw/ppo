import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    """Actor-Critic network for discrete action spaces."""

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value.squeeze(-1), entropy


class ActorCriticContinuous(nn.Module):
    """Actor-Critic network for continuous action spaces."""

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.actor_net(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp()
        value = self.critic(obs)
        return mean, std, value

    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy().flatten(), log_prob, value.squeeze(-1)

    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, value.squeeze(-1), entropy


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def get(self, device):
        return (
            torch.tensor(np.array(self.observations), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.actions), dtype=torch.float32).to(device),
            torch.tensor(self.log_probs, dtype=torch.float32).to(device),
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.tensor(self.values, dtype=torch.float32).to(device),
            torch.tensor(self.dones, dtype=torch.float32).to(device),
        )


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    values_ext = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values
    return advantages, returns


class PPO:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        mini_batch_size=64,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.buffer = RolloutBuffer()

    def collect_rollout(self, env, rollout_steps):
        """Collect rollout_steps of experience."""
        obs, _ = env.reset()
        episode_rewards = []
        current_ep_reward = 0

        for _ in range(rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.buffer.store(obs, action, log_prob.item(), reward, value.item(), float(done))
            current_ep_reward += reward

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Compute next value for GAE
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if isinstance(self.model, ActorCriticContinuous):
                _, _, next_value = self.model(obs_tensor)
            else:
                _, next_value = self.model(obs_tensor)
            next_value = next_value.squeeze().item()

        return episode_rewards, next_value

    def update(self, next_value):
        """Perform PPO update."""
        obs, actions, old_log_probs, rewards, values, dones = self.buffer.get(self.device)

        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, dones, next_value, self.gamma, self.lam
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # If discrete, actions should be long
        if isinstance(self.model, ActorCritic):
            actions = actions.long()

        total_samples = len(obs)
        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(self.update_epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                new_log_probs, new_values, entropy = self.model.evaluate(b_obs, b_actions)

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (new_values - b_returns).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        self.buffer.clear()
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)
