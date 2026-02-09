"""
PPO Agent Implementation for SE-RL Framework
============================================

This module implements the complete PPO (Proximal Policy Optimization) algorithm
with Generalized Advantage Estimation (GAE) as described in the paper.

Key components:
- PPO with clipped objective
- GAE (Generalized Advantage Estimation) - Equation (3)
- Actor-Critic architecture
- Proper value function loss

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    clip_value: float = 0.2  # Value function clipping

    # Training parameters
    ppo_epochs: int = 10  # Number of PPO update epochs
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5  # Gradient clipping

    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 2

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorNetwork(nn.Module):
    """
    Actor network for continuous action space.
    Outputs mean and log_std for Gaussian policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, log_std_min: float = -20, log_std_max: float = 2):
        super(ActorNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build MLP layers
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Final layer initialization for policy
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            mean: Action mean [batch_size, action_dim]
            log_std: Log standard deviation [batch_size, action_dim]
        """
        features = self.feature_network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Args:
            state: State tensor
            deterministic: If True, return mean action

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            entropy: Policy entropy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create normal distribution
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # Reparameterization trick

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        # Squash action to [-1, 1] using tanh
        action = torch.tanh(action)

        # Correct log_prob for tanh squashing
        log_prob -= torch.sum(
            torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True
        )

        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given state-action pairs.

        Args:
            state: State tensor
            action: Action tensor (must be in [-1, 1])

        Returns:
            log_prob: Log probability
            entropy: Entropy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)

        # Inverse tanh to get pre-squashed action
        action_pre_tanh = torch.atanh(torch.clamp(action, -0.999, 0.999))

        log_prob = dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)

        # Correct for tanh squashing
        log_prob -= torch.sum(
            torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True
        )

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super(CriticNetwork, self).__init__()

        # Build MLP layers
        layers = []
        input_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            value: Value estimate [batch_size, 1]
        """
        return self.network(state)


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    Implements GAE computation as per Equation (3) in the paper.
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.reset()

    def reset(self):
        """Reset the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.advantages = None
        self.returns = None

        self.ptr = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            done: bool, log_prob: float, value: float):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """
        Compute Generalized Advantage Estimation (GAE).

        Implements Equation (3) from the paper:
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Compute GAE in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]

            # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            last_gae = advantages[t]

        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, mini_batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get mini-batches for PPO update.

        Args:
            mini_batch_size: Size of each mini-batch

        Returns:
            List of mini-batch dictionaries
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device).unsqueeze(-1)
        advantages = torch.FloatTensor(self.advantages).to(self.device).unsqueeze(-1)
        returns = torch.FloatTensor(self.returns).to(self.device).unsqueeze(-1)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device).unsqueeze(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices
        batch_size = len(self.states)
        indices = np.random.permutation(batch_size)

        # Create mini-batches
        batches = []
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            batch_indices = indices[start:end]

            batch = {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_log_probs': old_log_probs[batch_indices],
                'advantages': advantages[batch_indices],
                'returns': returns[batch_indices],
                'old_values': old_values[batch_indices]
            }
            batches.append(batch)

        return batches


class PPOAgent:
    """
    Complete PPO Agent implementation with GAE.

    Implements Algorithm 1 inner loop from the paper with:
    - PPO clipped objective
    - Generalized Advantage Estimation
    - Value function clipping
    - Entropy bonus
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.device

        # Initialize actor and critic networks
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=2048,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device
        )

        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': []
        }

        logger.info(f"PPO Agent initialized: state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state: np.ndarray, deterministic: bool = False
                      ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state.

        Args:
            state: Current state
            deterministic: If True, return mean action

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            action, log_prob, _ = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)

        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().numpy().item(),
            value.cpu().numpy().item()
        )

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, done: bool, log_prob: float, value: float):
        """Store transition in buffer"""
        self.buffer.add(state, action, reward, done, log_prob, value)

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.

        Returns:
            Dictionary of training statistics
        """
        # Compute GAE
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_value = self.critic(last_state).cpu().numpy().item()

        self.buffer.compute_gae(last_value)

        # Get mini-batches
        batches = self.buffer.get_batches(self.config.mini_batch_size)

        # Training statistics for this update
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []

        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            for batch in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                old_values = batch['old_values']

                # Get current policy log probs and entropy
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)

                # Compute ratio: r_t = pi(a|s) / pi_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                new_values = self.critic(states)

                # Clipped value loss
                value_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.clip_value,
                    self.config.clip_value
                )
                value_loss1 = F.mse_loss(new_values, returns)
                value_loss2 = F.mse_loss(value_clipped, returns)
                value_loss = torch.max(value_loss1, value_loss2)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                        policy_loss +
                        self.config.value_loss_coef * value_loss +
                        self.config.entropy_coef * entropy_loss
                )

                # Update actor
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Compute statistics
                with torch.no_grad():
                    # KL divergence approximation
                    kl_div = (old_log_probs - new_log_probs).mean().item()

                    # Clip fraction
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
                kl_divs.append(kl_div)
                clip_fractions.append(clip_frac)

        # Clear buffer
        self.buffer.reset()

        # Aggregate statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions)
        }

        # Store in history
        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats

    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Agent loaded from {path}")

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
        return value.cpu().numpy().item()


class PPOTrainer:
    """
    Trainer class for PPO agent in financial environments.

    Implements the inner loop training from Algorithm 1.
    """

    def __init__(self, agent: PPOAgent, env, config: PPOConfig):
        self.agent = agent
        self.env = env
        self.config = config

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []

        logger.info("PPO Trainer initialized")

    def train(self, num_episodes: int, steps_per_update: int = 2048,
              log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Train the agent for specified number of episodes.

        Args:
            num_episodes: Number of training episodes
            steps_per_update: Steps to collect before each PPO update
            log_interval: Episodes between logging

        Returns:
            Training history
        """
        total_steps = 0
        episode_reward = 0
        episode_length = 0

        state = self.env.reset()

        for episode in range(num_episodes):
            # Collect rollouts
            for _ in range(steps_per_update):
                # Select action
                action, log_prob, value = self.agent.select_action(state)

                # Take step
                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.agent.store_transition(state, action, reward, done, log_prob, value)

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                state = next_state

                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    episode_reward = 0
                    episode_length = 0
                    state = self.env.reset()

            # Perform PPO update
            update_stats = self.agent.update()

            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-log_interval:]) if self.episode_lengths else 0

                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                    f"Value Loss: {update_stats['value_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.4f}"
                )

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': self.agent.training_stats
        }

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                state = next_state

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards)
        }


# Unit tests
if __name__ == "__main__":
    print("Testing PPO Agent...")

    # Create dummy environment for testing
    class DummyEnv:
        def __init__(self):
            self.state_dim = 10
            self.action_dim = 1
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return np.random.randn(self.state_dim)

        def step(self, action):
            self.step_count += 1
            next_state = np.random.randn(self.state_dim)
            reward = -np.abs(action).sum()
            done = self.step_count >= 100
            return next_state, reward, done, {}

    env = DummyEnv()
    config = PPOConfig()

    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config
    )

    # Test action selection
    state = env.reset()
    action, log_prob, value = agent.select_action(state)
    print(f"Action shape: {action.shape}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")

    # Test training
    trainer = PPOTrainer(agent, env, config)
    history = trainer.train(num_episodes=5, steps_per_update=200)
    print(f"Training completed. Final reward: {history['episode_rewards'][-1] if history['episode_rewards'] else 0}")

    # Test evaluation
    eval_metrics = trainer.evaluate(num_episodes=3)
    print(f"Evaluation metrics: {eval_metrics}")

    print("PPO Agent tests passed!")
