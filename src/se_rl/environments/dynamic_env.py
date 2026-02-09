"""
Dynamic Environment for SE-RL Framework
====================================

This module implements a dynamic environment for training RL agents
with market simulation and multiple agent interactions.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class DynamicEnvironmentConfig:
    """Configuration for dynamic environment"""
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position: float = 0.1
    min_trade_size: float = 1000.0
    max_trade_size: float = 100000.0
    window_size: int = 20
    num_agents: int = 3
    agent_types: List[str] = None
    market_impact_factor: float = 0.001
    volatility_scale: float = 1.0
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = ["market_maker", "informed_trader", "noise_trader"]

class MarketAgent:
    """Base class for market agents"""
    
    def __init__(self, agent_id: int, agent_type: str, config: DynamicEnvironmentConfig):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.capital = config.initial_capital
        self.position = 0.0
        self.trades = []
        self.portfolio_values = []
        
    def get_action(self, state: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Get action based on current state and market data"""
        # This will be overridden by specific agent types
        return np.array([0.33, 0.33, 0.34, 0.5])  # Default: hold
    
    def update_portfolio(self, price: float):
        """Update portfolio value"""
        portfolio_value = self.capital + (self.position * price)
        self.portfolio_values.append(portfolio_value)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if not self.portfolio_values:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values) / np.max(portfolio_values) - 1
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'num_trades': len(self.trades)
        }

class MarketMakerAgent(MarketAgent):
    """Market maker agent that provides liquidity"""
    
    def __init__(self, agent_id: int, config: DynamicEnvironmentConfig):
        super().__init__(agent_id, "market_maker", config)
        self.spread = 0.002  # 0.2% spread
        self.inventory_target = 0.0
        self.inventory_penalty = 0.001
    
    def get_action(self, state: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Market maker action: provide liquidity and manage inventory"""
        current_price = market_data['current_price']
        volatility = market_data.get('volatility', 0.02)
        
        # Adjust spread based on volatility
        adjusted_spread = self.spread * (1 + volatility * 10)
        
        # Calculate bid and ask prices
        bid_price = current_price * (1 - adjusted_spread / 2)
        ask_price = current_price * (1 + adjusted_spread / 2)
        
        # Inventory management
        inventory_imbalance = self.position - self.inventory_target
        inventory_adjustment = -inventory_imbalance * self.inventory_penalty
        
        # Determine action based on inventory and market conditions
        if inventory_imbalance > 0.1:  # Too much inventory
            action = np.array([0.1, 0.8, 0.1, 0.7])  # Sell
        elif inventory_imbalance < -0.1:  # Too little inventory
            action = np.array([0.8, 0.1, 0.1, 0.7])  # Buy
        else:
            # Balanced inventory, provide liquidity
            action = np.array([0.4, 0.4, 0.2, 0.5])  # Balanced
        
        return action

class InformedTraderAgent(MarketAgent):
    """Informed trader agent with superior information"""
    
    def __init__(self, agent_id: int, config: DynamicEnvironmentConfig):
        super().__init__(agent_id, "informed_trader", config)
        self.information_advantage = 0.1  # 10% information advantage
        self.confidence_threshold = 0.6
    
    def get_action(self, state: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Informed trader action: trade based on superior information"""
        current_price = market_data['current_price']
        future_price = market_data.get('future_price', current_price)
        
        # Calculate expected return
        expected_return = (future_price - current_price) / current_price
        
        # Add information advantage
        if expected_return > 0:
            expected_return += self.information_advantage
        else:
            expected_return -= self.information_advantage
        
        # Determine action based on expected return
        if abs(expected_return) > self.confidence_threshold:
            if expected_return > 0:
                action = np.array([0.9, 0.05, 0.05, 0.8])  # Strong buy
            else:
                action = np.array([0.05, 0.9, 0.05, 0.8])  # Strong sell
        else:
            action = np.array([0.2, 0.2, 0.6, 0.3])  # Hold
        
        return action

class NoiseTraderAgent(MarketAgent):
    """Noise trader agent with random behavior"""
    
    def __init__(self, agent_id: int, config: DynamicEnvironmentConfig):
        super().__init__(agent_id, "noise_trader", config)
        self.noise_level = 0.3
        self.momentum_factor = 0.2
    
    def get_action(self, state: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Noise trader action: random trading with some momentum"""
        # Random component
        random_action = np.random.random(4)
        random_action = random_action / np.sum(random_action[:3])  # Normalize probabilities
        
        # Momentum component
        recent_returns = market_data.get('recent_returns', [0])
        momentum = np.mean(recent_returns) if recent_returns else 0
        
        if abs(momentum) > 0.01:  # Significant momentum
            if momentum > 0:
                momentum_action = np.array([0.7, 0.1, 0.2, 0.6])  # Follow momentum
            else:
                momentum_action = np.array([0.1, 0.7, 0.2, 0.6])  # Follow momentum
        else:
            momentum_action = np.array([0.33, 0.33, 0.34, 0.5])  # No momentum
        
        # Combine random and momentum components
        action = (1 - self.momentum_factor) * random_action + self.momentum_factor * momentum_action
        
        return action

class DynamicEnvironment:
    """Dynamic environment with market simulation and multiple agents"""
    
    def __init__(self, data: pd.DataFrame, config: DynamicEnvironmentConfig, 
                 num_agents: int = 3, agent_types: List[str] = None):
        self.data = data.copy()
        self.config = config
        self.num_agents = num_agents
        self.agent_types = agent_types or config.agent_types
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Market state
        self.current_step = 0
        self.current_price = self.data.iloc[0]['close']
        self.price_history = [self.current_price]
        self.volume_history = [0]
        self.order_book = {'bids': [], 'asks': []}
        
        # Preprocess data
        self._preprocess_data()
        
        logger.info(f"Dynamic environment initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> List[MarketAgent]:
        """Initialize market agents"""
        agents = []
        
        for i in range(self.num_agents):
            agent_type = self.agent_types[i % len(self.agent_types)]
            
            if agent_type == "market_maker":
                agent = MarketMakerAgent(i, self.config)
            elif agent_type == "informed_trader":
                agent = InformedTraderAgent(i, self.config)
            elif agent_type == "noise_trader":
                agent = NoiseTraderAgent(i, self.config)
            else:
                # Default to noise trader
                agent = NoiseTraderAgent(i, self.config)
            
            agents.append(agent)
        
        return agents
    
    def _preprocess_data(self):
        """Preprocess the financial data"""
        # Calculate technical indicators
        self.data['returns'] = self.data['close'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window=20).std()
        
        # Calculate future prices for informed traders
        self.data['future_price'] = self.data['close'].shift(-1)
        
        # Calculate recent returns for momentum
        self.data['recent_returns'] = self.data['returns'].rolling(window=5).mean()
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(0)
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.current_price = self.data.iloc[0]['close']
        self.price_history = [self.current_price]
        self.volume_history = [0]
        self.order_book = {'bids': [], 'asks': []}
        
        # Reset agents
        for agent in self.agents:
            agent.capital = self.config.initial_capital
            agent.position = 0.0
            agent.trades = []
            agent.portfolio_values = [self.config.initial_capital]
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current market state"""
        if self.current_step >= len(self.data):
            return np.zeros((self.config.window_size, 10))
        
        # Get market data
        market_data = self.data.iloc[self.current_step]
        
        # Create state vector
        state = np.array([
            market_data['close'],
            market_data['volume'],
            market_data['returns'],
            market_data['volatility'],
            self.current_price,
            len(self.order_book['bids']),
            len(self.order_book['asks']),
            np.mean(self.price_history[-10:]) if len(self.price_history) >= 10 else self.current_price,
            np.std(self.price_history[-10:]) if len(self.price_history) >= 10 else 0,
            len(self.agents)
        ])
        
        return state
    
    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], bool, Dict[str, Any]]:
        """Take a step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), [0.0] * len(self.agents), True, {'message': 'Episode ended'}
        
        # Get market data
        market_data = self._get_market_data()
        
        # Collect actions from all agents
        agent_actions = []
        for i, agent in enumerate(self.agents):
            if i < len(actions):
                action = actions[i]
            else:
                action = agent.get_action(self._get_state(), market_data)
            agent_actions.append(action)
        
        # Execute trades and update market
        rewards = self._execute_trades(agent_actions)
        
        # Update market price based on supply and demand
        self._update_market_price()
        
        # Update step
        self.current_step += 1
        
        # Update agent portfolios
        for agent in self.agents:
            agent.update_portfolio(self.current_price)
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Create info dictionary
        info = {
            'step': self.current_step,
            'current_price': self.current_price,
            'agent_metrics': [agent.get_portfolio_metrics() for agent in self.agents],
            'market_volume': sum(self.volume_history[-10:]) if self.volume_history else 0
        }
        
        return next_state, rewards, done, info
    
    def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for agents"""
        if self.current_step >= len(self.data):
            return {}
        
        market_data = self.data.iloc[self.current_step].to_dict()
        market_data.update({
            'current_price': self.current_price,
            'recent_returns': self.data['returns'].iloc[max(0, self.current_step-5):self.current_step+1].tolist(),
            'price_history': self.price_history[-20:],
            'volume_history': self.volume_history[-20:]
        })
        
        return market_data
    
    def _execute_trades(self, actions: List[np.ndarray]) -> List[float]:
        """Execute trades for all agents"""
        rewards = []
        
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            # Parse action
            action_type, action_size = self._parse_action(action)
            
            # Execute trade
            reward, trade_info = self._execute_agent_trade(agent, action_type, action_size)
            rewards.append(reward)
            
            # Record trade
            if trade_info['executed']:
                agent.trades.append(trade_info)
        
        return rewards
    
    def _parse_action(self, action: np.ndarray) -> Tuple[str, float]:
        """Parse action from agent output"""
        action_probs = action[:3]
        action_size = action[3] if len(action) > 3 else 0.5
        
        action_type_idx = np.argmax(action_probs)
        action_types = ['buy', 'sell', 'hold']
        action_type = action_types[action_type_idx]
        
        action_size = np.clip(action_size, 0.0, 1.0)
        
        return action_type, action_size
    
    def _execute_agent_trade(self, agent: MarketAgent, action_type: str, action_size: float) -> Tuple[float, Dict[str, Any]]:
        """Execute a trade for a specific agent"""
        trade_info = {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type,
            'action_type': action_type,
            'action_size': action_size,
            'current_price': self.current_price,
            'executed': False
        }
        
        reward = 0.0
        
        if action_type == 'buy' and action_size > 0:
            # Calculate trade size
            max_trade_value = agent.capital * self.config.max_position
            trade_value = max_trade_value * action_size
            trade_value = np.clip(trade_value, self.config.min_trade_size, self.config.max_trade_size)
            
            # Calculate shares to buy
            shares = trade_value / self.current_price
            
            # Apply transaction costs and slippage
            total_cost = trade_value * (1 + self.config.transaction_cost + self.config.slippage)
            
            if total_cost <= agent.capital:
                # Execute trade
                agent.capital -= total_cost
                agent.position += shares
                
                # Update market impact
                self._apply_market_impact('buy', trade_value)
                
                trade_info.update({
                    'executed': True,
                    'shares': shares,
                    'trade_value': trade_value,
                    'total_cost': total_cost
                })
                
                # Calculate reward
                next_price = self.data.iloc[min(self.current_step + 1, len(self.data) - 1)]['close']
                price_change = (next_price - self.current_price) / self.current_price
                reward = price_change * shares * self.current_price
                
        elif action_type == 'sell' and action_size > 0:
            # Calculate shares to sell
            max_shares = agent.position
            shares_to_sell = max_shares * action_size
            
            if shares_to_sell > 0:
                # Calculate trade value
                trade_value = shares_to_sell * self.current_price
                
                # Apply transaction costs and slippage
                net_proceeds = trade_value * (1 - self.config.transaction_cost - self.config.slippage)
                
                # Execute trade
                agent.capital += net_proceeds
                agent.position -= shares_to_sell
                
                # Update market impact
                self._apply_market_impact('sell', trade_value)
                
                trade_info.update({
                    'executed': True,
                    'shares': shares_to_sell,
                    'trade_value': trade_value,
                    'net_proceeds': net_proceeds
                })
                
                # Calculate reward
                next_price = self.data.iloc[min(self.current_step + 1, len(self.data) - 1)]['close']
                price_change = (self.current_price - next_price) / self.current_price
                reward = price_change * shares_to_sell * self.current_price
        
        else:  # hold
            reward = -0.0001  # Small penalty for holding
        
        return reward, trade_info
    
    def _apply_market_impact(self, trade_type: str, trade_value: float):
        """Apply market impact to price"""
        impact = trade_value * self.config.market_impact_factor
        
        if trade_type == 'buy':
            self.current_price *= (1 + impact)
        else:  # sell
            self.current_price *= (1 - impact)
    
    def _update_market_price(self):
        """Update market price based on supply and demand"""
        # Get base price from data
        if self.current_step < len(self.data):
            base_price = self.data.iloc[self.current_step]['close']
        else:
            base_price = self.current_price
        
        # Add some noise and trend
        noise = np.random.normal(0, base_price * 0.001)
        trend = base_price * 0.0001 * np.random.choice([-1, 1])
        
        self.current_price = base_price + noise + trend
        self.price_history.append(self.current_price)
    
    def get_market_metrics(self) -> Dict[str, float]:
        """Calculate market performance metrics"""
        if not self.price_history:
            return {}
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        total_return = (prices[-1] - prices[0]) / prices[0]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'final_price': prices[-1],
            'price_range': np.max(prices) - np.min(prices)
        }
    
    def render(self):
        """Render the current market state"""
        print(f"Step: {self.current_step}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Market Volume: {sum(self.volume_history[-10:]):.0f}")
        print(f"Number of Agents: {len(self.agents)}")
        print("-" * 50)
        
        for i, agent in enumerate(self.agents):
            portfolio_value = agent.capital + (agent.position * self.current_price)
            print(f"Agent {i} ({agent.agent_type}): ${portfolio_value:.2f}, Trades: {len(agent.trades)}")
        print("-" * 50) 