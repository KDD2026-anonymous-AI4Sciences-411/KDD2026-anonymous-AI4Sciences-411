import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
import random
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_episodes: int = 1000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FinancialMetrics:
    """Calculates financial performance metrics"""
    
    @staticmethod
    def calculate_pa(execution_prices: List[float], vwap_prices: List[float]) -> float:
        """Calculate Price Advantage (PA) over VWAP"""
        if len(execution_prices) != len(vwap_prices) or len(execution_prices) == 0:
            return 0.0
        
        price_advantages = []
        for exec_price, vwap_price in zip(execution_prices, vwap_prices):
            if vwap_price > 0:
                advantage = (vwap_price - exec_price) / vwap_price
                price_advantages.append(advantage)
        
        if not price_advantages:
            return 0.0
        
        pa_bps = np.mean(price_advantages) * 10000
        return pa_bps
    
    @staticmethod
    def calculate_wr(returns: List[float]) -> float:
        """Calculate Win Ratio (WR)"""
        if not returns:
            return 0.0
        
        positive_returns = sum(1 for r in returns if r > 0)
        win_ratio = positive_returns / len(returns)
        return win_ratio
    
    @staticmethod
    def calculate_glr(returns: List[float]) -> float:
        """Calculate Gain-Loss Ratio (GLR)"""
        if not returns:
            return 0.0
        
        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        
        if not gains or not losses:
            return 0.0
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.0
        
        return avg_gain / avg_loss
    
    @staticmethod
    def calculate_afi(final_inventories: List[float]) -> float:
        """Calculate Average Final Inventory (AFI)"""
        if not final_inventories:
            return 0.0
        
        return np.mean([abs(inv) for inv in final_inventories])

class RLAgent(nn.Module):
    """Base RL agent class"""
    
    def __init__(self, state_dim: int, action_dim: int, config: TrainingConfig):
        super(RLAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.device
        
        # Initialize networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.total_steps = 0
        
        self.to(self.device)
        
    def _build_actor(self) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()
        )
    
    def _build_critic(self) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def get_action(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from current state"""
        state = state.to(self.device)
        
        if training and random.random() < self.epsilon:
            # Epsilon-greedy exploration
            action = torch.rand(self.action_dim, device=self.device) * 2 - 1
            log_prob = torch.zeros(1, device=self.device)
        else:
            # Policy-based action
            action_mean = self.actor(state)
            
            # Add exploration noise
            if training:
                noise = torch.randn_like(action_mean) * 0.1
                action = action_mean + noise
            else:
                action = action_mean
            
            # Clip action to valid range
            action = torch.clamp(action, -1, 1)
            log_prob = torch.zeros(1, device=self.device)
        
        return action, log_prob
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate for state"""
        state = state.to(self.device)
        return self.critic(state)
    
    def update_epsilon(self):
        """Update exploration epsilon"""
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)

class StaticEnvironment:
    """Static market environment using historical data"""
    
    def __init__(self, data: pd.DataFrame, config: TrainingConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Trading state
        self.cash = 1000000.0
        self.inventory = 0.0
        self.total_value = self.cash
        
        # Transaction costs
        self.transaction_cost = 0.001
        self.slippage = 0.0005
        
        logger.info(f"Static environment initialized with {len(data)} data points")
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = 1000000.0
        self.inventory = 0.0
        self.total_value = self.cash
        
        return self._get_state()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take action in environment"""
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        
        # Parse action
        order_size = action.item() * self.cash
        
        # Execute trade
        execution_price = current_data['close']
        
        # Apply slippage
        if order_size > 0:
            execution_price *= (1 + self.slippage)
        else:
            execution_price *= (1 - self.slippage)
        
        # Calculate shares to trade
        shares = order_size / execution_price
        
        # Update portfolio
        old_value = self.cash + self.inventory * execution_price
        
        # Apply transaction costs
        transaction_cost = abs(order_size) * self.transaction_cost
        
        self.cash -= order_size + transaction_cost
        self.inventory += shares
        
        # Calculate new total value
        new_value = self.cash + self.inventory * execution_price
        
        # Calculate reward
        reward = (new_value - old_value) / old_value
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'execution_price': execution_price,
            'order_size': order_size,
            'transaction_cost': transaction_cost,
            'total_value': new_value,
            'cash': self.cash,
            'inventory': self.inventory
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current state"""
        if self.current_step >= len(self.data):
            return {}
        
        current_data = self.data.iloc[self.current_step]
        
        state = {
            'current_price': current_data['close'],
            'open_price': current_data['open'],
            'high_price': current_data['high'],
            'low_price': current_data['low'],
            'volume': current_data['volume'],
            'returns': current_data.get('returns', 0.0),
            'volatility': current_data.get('volatility', 0.01),
            'rsi': current_data.get('rsi', 50.0),
            'macd': current_data.get('macd', 0.0),
            'bb_position': current_data.get('bb_position', 0.5),
            'volume_ratio': current_data.get('volume_ratio_20', 1.0),
            'cash': self.cash,
            'inventory': self.inventory,
            'total_value': self.total_value,
            'step': self.current_step
        }
        
        return state

class RLTrainer:
    """Main RL trainer for SE-RL framework"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Training components
        self.agent = None
        self.static_env = None
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        
        logger.info(f"RL Trainer initialized with config: {config}")
    
    def initialize_agent(self, state_dim: int, action_dim: int):
        """Initialize RL agent"""
        self.agent = RLAgent(state_dim, action_dim, self.config)
        logger.info(f"Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def set_static_environment(self, data: pd.DataFrame):
        """Set static training environment"""
        self.static_env = StaticEnvironment(data, self.config)
        logger.info("Static environment set")
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.static_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_info = []
        
        while episode_steps < 1000:  # Max steps per episode
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            
            # Get action
            action, log_prob = self.agent.get_action(state_tensor, training=True)
            
            # Take action
            next_state, reward, done, info = self.static_env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            episode_info.append(info)
            
            state = next_state
            
            if done:
                break
        
        # Update exploration
        self.agent.update_epsilon()
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_info': episode_info
        }
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy"""
        episode_rewards = []
        execution_prices = []
        vwap_prices = []
        returns = []
        final_inventories = []
        
        for episode in range(num_episodes):
            state = self.static_env.reset()
            episode_reward = 0.0
            episode_execution_prices = []
            episode_vwap_prices = []
            
            while True:
                state_tensor = self._state_to_tensor(state)
                action, _ = self.agent.get_action(state_tensor, training=False)
                
                next_state, reward, done, info = self.static_env.step(action)
                
                episode_reward += reward
                episode_execution_prices.append(info.get('execution_price', 100.0))
                episode_vwap_prices.append(state.get('current_price', 100.0))
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            execution_prices.extend(episode_execution_prices)
            vwap_prices.extend(episode_vwap_prices)
            returns.append(episode_reward)
            final_inventories.append(self.static_env.inventory)
        
        # Calculate metrics
        metrics = {
            'PA': FinancialMetrics.calculate_pa(execution_prices, vwap_prices),
            'WR': FinancialMetrics.calculate_wr(returns),
            'GLR': FinancialMetrics.calculate_glr(returns),
            'AFI': FinancialMetrics.calculate_afi(final_inventories),
            'MeanReward': np.mean(episode_rewards),
            'StdReward': np.std(episode_rewards)
        }
        
        return metrics
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dict to tensor"""
        state_values = []
        
        for key in ['current_price', 'returns', 'volatility', 'rsi', 'macd', 
                   'bb_position', 'volume_ratio', 'cash', 'inventory']:
            state_values.append(state.get(key, 0.0))
        
        return torch.FloatTensor(state_values).unsqueeze(0)
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        self.evaluation_history = checkpoint.get('evaluation_history', [])
        
        logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    config = TrainingConfig()
    trainer = RLTrainer(config)
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'close': np.random.randn(1000).cumsum() + 100,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'volume': np.random.randint(100000, 1000000, 1000),
        'returns': np.random.randn(1000) * 0.02,
        'volatility': np.random.rand(1000) * 0.05,
        'rsi': np.random.rand(1000) * 100,
        'macd': np.random.randn(1000) * 0.1,
        'bb_position': np.random.rand(1000),
        'volume_ratio_20': np.random.rand(1000) * 2
    })
    
    # Initialize agent
    state_dim = 9
    action_dim = 1
    trainer.initialize_agent(state_dim, action_dim)
    
    # Set environment
    trainer.set_static_environment(dummy_data)
    
    # Train for a few episodes
    for episode in range(10):
        result = trainer.train_episode()
        print(f"Episode {episode}: Reward = {result['episode_reward']:.4f}")
    
    # Evaluate policy
    metrics = trainer.evaluate_policy(num_episodes=5)
    print(f"Evaluation metrics: {metrics}") 