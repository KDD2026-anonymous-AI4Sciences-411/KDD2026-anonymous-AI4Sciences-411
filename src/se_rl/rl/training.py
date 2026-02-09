import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass
import time
import random
from collections import deque
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    
    # RL parameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Target network update rate
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # PPO parameters
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # SAC parameters
    sac_alpha: float = 0.2
    sac_auto_alpha: bool = True
    
    # Training environment
    static_env_weight: float = 0.5
    dynamic_env_weight: float = 0.5
    rebalance_iterations: int = 10
    
    # Evaluation
    eval_frequency: int = 100
    eval_episodes: int = 50
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FinancialMetrics:
    """Calculates financial performance metrics"""
    
    @staticmethod
    def calculate_pa(execution_prices: List[float], vwap_prices: List[float]) -> float:
        """
        Calculate Price Advantage (PA) over VWAP
        
        Args:
            execution_prices: List of execution prices
            vwap_prices: List of VWAP prices
            
        Returns:
            PA value in basis points
        """
        if len(execution_prices) != len(vwap_prices) or len(execution_prices) == 0:
            return 0.0
        
        # Calculate average price advantage
        price_advantages = []
        for exec_price, vwap_price in zip(execution_prices, vwap_prices):
            if vwap_price > 0:
                advantage = (vwap_price - exec_price) / vwap_price
                price_advantages.append(advantage)
        
        if not price_advantages:
            return 0.0
        
        # Convert to basis points
        pa_bps = np.mean(price_advantages) * 10000
        return pa_bps
    
    @staticmethod
    def calculate_wr(returns: List[float]) -> float:
        """
        Calculate Win Ratio (WR)
        
        Args:
            returns: List of returns
            
        Returns:
            Win ratio between 0 and 1
        """
        if not returns:
            return 0.0
        
        positive_returns = sum(1 for r in returns if r > 0)
        win_ratio = positive_returns / len(returns)
        return win_ratio
    
    @staticmethod
    def calculate_glr(returns: List[float]) -> float:
        """
        Calculate Gain-Loss Ratio (GLR)
        
        Args:
            returns: List of returns
            
        Returns:
            Gain-loss ratio
        """
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
        """
        Calculate Average Final Inventory (AFI)
        
        Args:
            final_inventories: List of final inventory levels
            
        Returns:
            Average final inventory
        """
        if not final_inventories:
            return 0.0
        
        return np.mean([abs(inv) for inv in final_inventories])
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0.0
        
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(252)  # Annualized
        return sharpe_ratio
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Maximum drawdown as percentage
        """
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

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
            nn.Tanh()  # Output in [-1, 1] for continuous actions
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
        """
        Get action from current state
        
        Args:
            state: Current state tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob)
        """
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
            
            # Calculate log probability (simplified)
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

class PPOTrainer:
    """PPO trainer for RL agents"""
    
    def __init__(self, agent: RLAgent, config: TrainingConfig):
        self.agent = agent
        self.config = config
        self.device = config.device
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        logger.info("PPO Trainer initialized")
    
    def collect_experience(self, state: torch.Tensor, action: torch.Tensor, 
                          reward: float, next_state: torch.Tensor, done: bool,
                          log_prob: torch.Tensor):
        """Collect experience for training"""
        self.states.append(state.cpu().numpy())
        self.actions.append(action.cpu().numpy())
        self.rewards.append(reward)
        self.next_states.append(next_state.cpu().numpy())
        self.dones.append(done)
        self.log_probs.append(log_prob.cpu().numpy())
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.states) < self.config.batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(self.dones)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(states, rewards, next_states, dones)
        
        # PPO training
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        for _ in range(self.config.ppo_epochs):
            # Sample mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current policy
                action_mean = self.agent.actor(batch_states)
                noise = torch.randn_like(action_mean) * 0.1
                batch_actions_new = action_mean + noise
                batch_actions_new = torch.clamp(batch_actions_new, -1, 1)
                
                # Calculate new log probabilities
                new_log_probs = torch.zeros_like(batch_old_log_probs)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.agent.get_value(batch_states)
                value_loss = F.mse_loss(values.squeeze(), batch_advantages + values.squeeze().detach())
                
                # Entropy loss
                entropy_loss = -torch.mean(torch.sum(-action_mean * torch.log(action_mean + 1e-8), dim=1))
                
                # Total loss
                total_loss = (actor_loss + 
                            self.config.value_loss_coef * value_loss + 
                            self.config.entropy_coef * entropy_loss)
                
                # Update networks
                self.agent.actor_optimizer.zero_grad()
                self.agent.critic_optimizer.zero_grad()
                total_loss.backward()
                self.agent.actor_optimizer.step()
                self.agent.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear buffer
        self._clear_buffer()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def _calculate_advantages(self, states: torch.Tensor, rewards: torch.Tensor,
                            next_states: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using TD(0)"""
        values = self.agent.get_value(states).squeeze()
        next_values = self.agent.get_value(next_states).squeeze()
        
        # TD target
        td_target = rewards + self.config.gamma * next_values * (~dones).float()
        
        # Advantage
        advantages = td_target - values
        
        return advantages
    
    def _clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()

class StaticEnvironment:
    """Static market environment using historical data"""
    
    def __init__(self, data: pd.DataFrame, config: TrainingConfig):
        self.data = data
        self.config = config
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Trading state
        self.cash = 1000000.0  # Initial cash
        self.inventory = 0.0   # Current inventory
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
        """
        Take action in environment
        
        Args:
            action: Agent action (order size as fraction of available cash)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        next_data = self.data.iloc[self.current_step + 1]
        
        # Parse action
        order_size = action.item() * self.cash  # Convert to cash amount
        
        # Execute trade
        execution_price = current_data['close']
        
        # Apply slippage
        if order_size > 0:  # Buy
            execution_price *= (1 + self.slippage)
        else:  # Sell
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
        
        # Market state
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

class DynamicEnvironment:
    """Dynamic multi-agent market environment"""
    
    def __init__(self, data: pd.DataFrame, config: TrainingConfig, num_agents: int = 3):
        self.data = data
        self.config = config
        self.num_agents = num_agents
        
        # Market state
        self.order_book = {
            'bids': [],
            'asks': []
        }
        self.market_price = 100.0
        self.market_volume = 1000000
        
        # Agents
        self.agents = self._initialize_agents()
        
        # Trading state
        self.cash = 1000000.0
        self.inventory = 0.0
        self.total_value = self.cash
        
        logger.info(f"Dynamic environment initialized with {num_agents} agents")
    
    def _initialize_agents(self) -> List[Dict[str, Any]]:
        """Initialize market agents"""
        agents = []
        
        for i in range(self.num_agents):
            agent_type = ['market_maker', 'informed_trader', 'noise_trader'][i % 3]
            
            agent = {
                'id': i,
                'type': agent_type,
                'cash': 1000000.0,
                'inventory': 0.0,
                'risk_aversion': random.uniform(0.1, 0.5),
                'information_quality': random.uniform(0.1, 1.0),
                'trading_frequency': random.uniform(0.1, 1.0)
            }
            
            agents.append(agent)
        
        return agents
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment"""
        self.cash = 1000000.0
        self.inventory = 0.0
        self.total_value = self.cash
        
        # Reset agents
        for agent in self.agents:
            agent['cash'] = 1000000.0
            agent['inventory'] = 0.0
        
        # Reset order book
        self.order_book = {'bids': [], 'asks': []}
        
        return self._get_state()
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take action in dynamic environment"""
        
        # Get agent actions
        agent_actions = self._get_agent_actions()
        
        # Execute agent actions
        for agent_action in agent_actions:
            self._execute_agent_action(agent_action)
        
        # Execute main agent action
        execution_price = self._execute_main_action(action)
        
        # Update market state
        self._update_market_state()
        
        # Calculate reward
        reward = self._calculate_reward(execution_price)
        
        # Get next state
        next_state = self._get_state()
        
        # Check if done
        done = False  # Simplified - would check time limits
        
        info = {
            'execution_price': execution_price,
            'market_price': self.market_price,
            'order_book_depth': len(self.order_book['bids']) + len(self.order_book['asks'])
        }
        
        return next_state, reward, done, info
    
    def _get_agent_actions(self) -> List[Dict[str, Any]]:
        """Get actions from market agents"""
        actions = []
        
        for agent in self.agents:
            if random.random() < agent['trading_frequency']:
                action = self._generate_agent_action(agent)
                actions.append(action)
        
        return actions
    
    def _generate_agent_action(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action for a market agent"""
        action_type = random.choice(['market_order', 'limit_order'])
        
        if action_type == 'market_order':
            # Market order
            order_size = random.uniform(-1000, 1000) * agent['risk_aversion']
            action = {
                'agent_id': agent['id'],
                'type': 'market_order',
                'size': order_size,
                'price': None
            }
        else:
            # Limit order
            spread = random.uniform(0.001, 0.01)
            if random.random() < 0.5:
                # Bid
                price = self.market_price * (1 - spread)
                size = random.uniform(100, 1000)
            else:
                # Ask
                price = self.market_price * (1 + spread)
                size = random.uniform(-1000, -100)
            
            action = {
                'agent_id': agent['id'],
                'type': 'limit_order',
                'size': size,
                'price': price
            }
        
        return action
    
    def _execute_agent_action(self, action: Dict[str, Any]):
        """Execute agent action"""
        if action['type'] == 'market_order':
            # Execute market order
            execution_price = self.market_price
            order_size = action['size']
            
            # Update agent portfolio
            agent = self.agents[action['agent_id']]
            agent['cash'] -= order_size * execution_price
            agent['inventory'] += order_size
            
        elif action['type'] == 'limit_order':
            # Add to order book
            if action['size'] > 0:
                self.order_book['bids'].append({
                    'price': action['price'],
                    'size': action['size'],
                    'agent_id': action['agent_id']
                })
            else:
                self.order_book['asks'].append({
                    'price': action['price'],
                    'size': abs(action['size']),
                    'agent_id': action['agent_id']
                })
    
    def _execute_main_action(self, action: torch.Tensor) -> float:
        """Execute main agent action"""
        order_size = action.item() * self.cash
        
        # Find best execution price
        if order_size > 0:  # Buy
            if self.order_book['asks']:
                execution_price = min(ask['price'] for ask in self.order_book['asks'])
            else:
                execution_price = self.market_price * 1.001  # Slight premium
        else:  # Sell
            if self.order_book['bids']:
                execution_price = max(bid['price'] for bid in self.order_book['bids'])
            else:
                execution_price = self.market_price * 0.999  # Slight discount
        
        # Update portfolio
        shares = order_size / execution_price
        self.cash -= order_size
        self.inventory += shares
        
        return execution_price
    
    def _update_market_state(self):
        """Update market state based on order book"""
        # Simple market price update
        if self.order_book['bids'] and self.order_book['asks']:
            best_bid = max(bid['price'] for bid in self.order_book['bids'])
            best_ask = min(ask['price'] for ask in self.order_book['asks'])
            self.market_price = (best_bid + best_ask) / 2
        
        # Clear some orders (simplified)
        if random.random() < 0.1:
            if self.order_book['bids']:
                self.order_book['bids'].pop(random.randint(0, len(self.order_book['bids'])-1))
            if self.order_book['asks']:
                self.order_book['asks'].pop(random.randint(0, len(self.order_book['asks'])-1))
    
    def _calculate_reward(self, execution_price: float) -> float:
        """Calculate reward for the main agent"""
        # Simple reward based on execution quality
        market_impact = abs(self.inventory) * 0.001  # Simplified market impact
        reward = -market_impact
        
        # Add profit/loss component
        if self.inventory != 0:
            price_change = (execution_price - self.market_price) / self.market_price
            reward += self.inventory * price_change
        
        return reward
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'market_price': self.market_price,
            'order_book_bids': len(self.order_book['bids']),
            'order_book_asks': len(self.order_book['asks']),
            'cash': self.cash,
            'inventory': self.inventory,
            'total_value': self.cash + self.inventory * self.market_price
        }

class RLTrainer:
    """Main RL trainer for SE-RL framework"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Training components
        self.agent = None
        self.trainer = None
        self.static_env = None
        self.dynamic_env = None
        
        # Performance tracking
        self.training_history = []
        self.evaluation_history = []
        
        logger.info(f"RL Trainer initialized with config: {config}")
    
    def initialize_agent(self, state_dim: int, action_dim: int):
        """Initialize RL agent"""
        self.agent = RLAgent(state_dim, action_dim, self.config)
        self.trainer = PPOTrainer(self.agent, self.config)
        
        logger.info(f"Agent initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def set_environments(self, static_data: pd.DataFrame, dynamic_data: pd.DataFrame):
        """Set training environments"""
        self.static_env = StaticEnvironment(static_data, self.config)
        self.dynamic_env = DynamicEnvironment(dynamic_data, self.config)
        
        logger.info("Environments set")
    
    def train_episode(self, env_type: str = "static") -> Dict[str, Any]:
        """Train for one episode"""
        
        if env_type == "static":
            env = self.static_env
        else:
            env = self.dynamic_env
        
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_info = []
        
        while episode_steps < self.config.max_steps_per_episode:
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            
            # Get action
            action, log_prob = self.agent.get_action(state_tensor, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.trainer.collect_experience(
                state_tensor, action, reward, 
                self._state_to_tensor(next_state), done, log_prob
            )
            
            episode_reward += reward
            episode_steps += 1
            episode_info.append(info)
            
            state = next_state
            
            if done:
                break
        
        # Train on collected experience
        training_losses = self.trainer.train_step()
        
        # Update exploration
        self.agent.update_epsilon()
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'training_losses': training_losses,
            'episode_info': episode_info
        }
    
    def evaluate_policy(self, env_type: str = "static", num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy"""
        
        if env_type == "static":
            env = self.static_env
        else:
            env = self.dynamic_env
        
        episode_rewards = []
        execution_prices = []
        vwap_prices = []
        returns = []
        final_inventories = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_execution_prices = []
            episode_vwap_prices = []
            
            while True:
                state_tensor = self._state_to_tensor(state)
                action, _ = self.agent.get_action(state_tensor, training=False)
                
                next_state, reward, done, info = env.step(action)
                
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
            final_inventories.append(env.inventory)
        
        # Calculate metrics
        metrics = {
            'PA': FinancialMetrics.calculate_pa(execution_prices, vwap_prices),
            'WR': FinancialMetrics.calculate_wr(returns),
            'GLR': FinancialMetrics.calculate_glr(returns),
            'AFI': FinancialMetrics.calculate_afi(final_inventories),
            'Sharpe': FinancialMetrics.calculate_sharpe_ratio(returns),
            'MaxDD': FinancialMetrics.calculate_max_drawdown(episode_rewards),
            'MeanReward': np.mean(episode_rewards),
            'StdReward': np.std(episode_rewards)
        }
        
        return metrics
    
    def hybrid_training(self, num_episodes: int = 100) -> Dict[str, Any]:
        """Perform hybrid environment training"""
        
        static_results = []
        dynamic_results = []
        
        for episode in range(num_episodes):
            # Train in static environment
            static_result = self.train_episode("static")
            static_results.append(static_result)
            
            # Train in dynamic environment
            dynamic_result = self.train_episode("dynamic")
            dynamic_results.append(dynamic_result)
            
            # Rebalance if needed
            if episode % self.config.rebalance_iterations == 0:
                self._rebalance_environments(static_results, dynamic_results)
        
        return {
            'static_results': static_results,
            'dynamic_results': dynamic_results
        }
    
    def _rebalance_environments(self, static_results: List[Dict], dynamic_results: List[Dict]):
        """Rebalance environment weights based on performance"""
        if not static_results or not dynamic_results:
            return
        
        # Calculate average rewards
        static_avg_reward = np.mean([r['episode_reward'] for r in static_results[-10:]])
        dynamic_avg_reward = np.mean([r['episode_reward'] for r in dynamic_results[-10:]])
        
        # Adjust weights
        total_reward = static_avg_reward + dynamic_avg_reward
        if total_reward > 0:
            self.config.static_env_weight = static_avg_reward / total_reward
            self.config.dynamic_env_weight = dynamic_avg_reward / total_reward
        
        logger.info(f"Environment weights rebalanced: static={self.config.static_env_weight:.3f}, dynamic={self.config.dynamic_env_weight:.3f}")
    
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dict to tensor"""
        # Extract numerical values
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
    state_dim = 9  # Number of state features
    action_dim = 1  # Single action (order size)
    trainer.initialize_agent(state_dim, action_dim)
    
    # Set environments
    trainer.set_environments(dummy_data, dummy_data)
    
    # Train for a few episodes
    for episode in range(10):
        result = trainer.train_episode("static")
        print(f"Episode {episode}: Reward = {result['episode_reward']:.4f}")
    
    # Evaluate policy
    metrics = trainer.evaluate_policy("static", num_episodes=5)
    print(f"Evaluation metrics: {metrics}") 