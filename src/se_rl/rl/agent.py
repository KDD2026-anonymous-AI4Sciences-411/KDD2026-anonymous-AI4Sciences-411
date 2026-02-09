"""
RL Agent for SE-RL Framework
==========================

This module implements reinforcement learning agents for financial trading.

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for RL agent"""
    state_dim: int = 64
    action_dim: int = 4
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100
    device: str = "auto"

class DQNNetwork(nn.Module):
    """Deep Q-Network for RL agent"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.shared_layers(state)
        
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self) -> int:
        return len(self.buffer)

class RLAgent:
    """Reinforcement learning agent for financial trading"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize networks
        self.q_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_network = DQNNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize actor-critic network
        self.actor_critic = ActorCriticNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        
        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.memory_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.training_mode = True
        
        logger.info(f"RL Agent initialized on device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.config.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def select_action_policy(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action using policy network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.actor_critic(state_tensor)
            
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return action.item(), log_prob.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_dqn(self) -> Dict[str, float]:
        """Train DQN agent"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # Update target network
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, 
                          self.config.epsilon_decay * self.epsilon)
        
        self.step_count += 1
        
        return {'dqn_loss': loss.item(), 'epsilon': self.epsilon}
    
    def train_actor_critic(self, states: np.ndarray, actions: np.ndarray, 
                          rewards: np.ndarray, next_states: np.ndarray, 
                          dones: np.ndarray) -> Dict[str, float]:
        """Train actor-critic agent"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Forward pass
        action_probs, state_values = self.actor_critic(states)
        _, next_state_values = self.actor_critic(next_states)
        
        # Compute advantages
        advantages = rewards + (self.config.gamma * next_state_values.squeeze() * ~dones) - state_values.squeeze()
        
        # Actor loss (policy gradient)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(state_values.squeeze(), 
                                rewards + self.config.gamma * next_state_values.squeeze() * ~dones)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.ac_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        self.ac_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().squeeze()
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.actor_critic(state_tensor)
            return action_probs.cpu().numpy().squeeze()
    
    def save_model(self, filepath: str):
        """Save agent model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'ac_optimizer_state_dict': self.ac_optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.ac_optimizer.load_state_dict(checkpoint['ac_optimizer_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.step_count = checkpoint.get('step_count', 0)
        
        logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training: bool):
        """Set training mode"""
        self.training_mode = training
        self.q_network.train(training)
        self.target_network.train(training)
        self.actor_critic.train(training)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'device': str(self.device),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_mode': self.training_mode,
            'replay_buffer_size': len(self.replay_buffer),
            'config': self.config.__dict__
        } 