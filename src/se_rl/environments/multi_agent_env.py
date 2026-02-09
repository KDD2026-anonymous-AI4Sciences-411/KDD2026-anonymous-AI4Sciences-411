"""
Multi-Agent Environment for SE-RL Framework
========================================

This module implements a multi-agent environment for training RL agents
with complex agent interactions and communication protocols.

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
import time

logger = logging.getLogger(__name__)

@dataclass
class MultiAgentEnvironmentConfig:
    """Configuration for multi-agent environment"""
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position: float = 0.1
    min_trade_size: float = 1000.0
    max_trade_size: float = 100000.0
    window_size: int = 20
    num_agents: int = 3
    agent_types: List[str] = None
    communication_enabled: bool = True
    market_impact_factor: float = 0.001
    information_sharing: bool = True
    coordination_enabled: bool = True
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = ["market_maker", "informed_trader", "noise_trader"]

class CommunicationProtocol:
    """Communication protocol for multi-agent interactions"""
    
    def __init__(self, num_agents: int, agent_types: List[str]):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.message_history = []
        self.trust_scores = {i: 1.0 for i in range(num_agents)}
        self.communication_network = self._initialize_network()
    
    def _initialize_network(self) -> Dict[int, List[int]]:
        """Initialize communication network"""
        network = {}
        for i in range(self.num_agents):
            # Each agent can communicate with all other agents
            network[i] = [j for j in range(self.num_agents) if j != i]
        return network
    
    def broadcast_message(self, sender_id: int, message: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to all connected agents"""
        message['sender_id'] = sender_id
        message['timestamp'] = time.time()
        message['trust_score'] = self.trust_scores[sender_id]
        
        self.message_history.append(message)
        
        # Update trust scores based on message quality
        self._update_trust_score(sender_id, message)
        
        return message
    
    def _update_trust_score(self, sender_id: int, message: Dict[str, Any]):
        """Update trust score based on message quality"""
        # Simple trust update based on message type
        if message.get('type') == 'market_info':
            # Market information is generally trustworthy
            self.trust_scores[sender_id] = min(1.0, self.trust_scores[sender_id] + 0.01)
        elif message.get('type') == 'trade_signal':
            # Trade signals need verification
            self.trust_scores[sender_id] = max(0.0, self.trust_scores[sender_id] - 0.005)
    
    def get_recent_messages(self, agent_id: int, num_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for a specific agent"""
        recent_messages = []
        for message in self.message_history[-num_messages:]:
            if message['sender_id'] != agent_id:
                recent_messages.append(message)
        return recent_messages
    
    def get_aggregated_info(self, agent_id: int) -> Dict[str, Any]:
        """Get aggregated information from recent messages"""
        recent_messages = self.get_recent_messages(agent_id, 20)
        
        if not recent_messages:
            return {}
        
        # Aggregate market information
        market_info = []
        trade_signals = []
        
        for message in recent_messages:
            if message.get('type') == 'market_info':
                market_info.append(message.get('data', {}))
            elif message.get('type') == 'trade_signal':
                trade_signals.append(message.get('signal', {}))
        
        return {
            'market_info': market_info,
            'trade_signals': trade_signals,
            'num_messages': len(recent_messages)
        }

class MultiAgentEnvironment:
    """Multi-agent environment with complex interactions"""
    
    def __init__(self, data: pd.DataFrame, config: MultiAgentEnvironmentConfig, 
                 agent_types: List[str] = None):
        self.data = data.copy()
        self.config = config
        self.agent_types = agent_types or config.agent_types
        self.num_agents = config.num_agents
        
        # Initialize communication protocol
        if config.communication_enabled:
            self.communication = CommunicationProtocol(self.num_agents, self.agent_types)
        else:
            self.communication = None
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Market state
        self.current_step = 0
        self.current_price = self.data.iloc[0]['close']
        self.price_history = [self.current_price]
        self.volume_history = [0]
        self.order_book = {'bids': [], 'asks': []}
        
        # Agent interactions
        self.interaction_history = []
        self.coordination_events = []
        
        # Preprocess data
        self._preprocess_data()
        
        logger.info(f"Multi-agent environment initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> List[Dict[str, Any]]:
        """Initialize multi-agent system"""
        agents = []
        
        for i in range(self.num_agents):
            agent_type = self.agent_types[i % len(self.agent_types)]
            
            agent = {
                'id': i,
                'type': agent_type,
                'capital': self.config.initial_capital,
                'position': 0.0,
                'trades': [],
                'portfolio_values': [self.config.initial_capital],
                'communication_buffer': [],
                'coordination_state': {},
                'performance_history': []
            }
            
            agents.append(agent)
        
        return agents
    
    def _preprocess_data(self):
        """Preprocess the financial data"""
        # Calculate technical indicators
        self.data['returns'] = self.data['close'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window=20).std()
        
        # Calculate market microstructure features
        self.data['bid_ask_spread'] = self.data['high'] - self.data['low']
        self.data['volume_imbalance'] = self.data['volume'].rolling(window=10).std()
        
        # Calculate agent-specific features
        for agent_type in self.agent_types:
            self.data[f'{agent_type}_activity'] = np.random.random(len(self.data))
        
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
            agent['capital'] = self.config.initial_capital
            agent['position'] = 0.0
            agent['trades'] = []
            agent['portfolio_values'] = [self.config.initial_capital]
            agent['communication_buffer'] = []
            agent['coordination_state'] = {}
            agent['performance_history'] = []
        
        # Reset communication
        if self.communication:
            self.communication.message_history = []
            self.communication.trust_scores = {i: 1.0 for i in range(self.num_agents)}
        
        # Reset interactions
        self.interaction_history = []
        self.coordination_events = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current market state"""
        if self.current_step >= len(self.data):
            return np.zeros((self.config.window_size, 15))
        
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
            len(self.agents),
            market_data['bid_ask_spread'],
            market_data['volume_imbalance'],
            len(self.interaction_history),
            len(self.coordination_events)
        ])
        
        return state
    
    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], bool, Dict[str, Any]]:
        """Take a step in the multi-agent environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), [0.0] * len(self.agents), True, {'message': 'Episode ended'}
        
        # Phase 1: Communication and Information Sharing
        if self.config.communication_enabled:
            self._process_communication_phase()
        
        # Phase 2: Coordination and Strategy Alignment
        if self.config.coordination_enabled:
            self._process_coordination_phase()
        
        # Phase 3: Action Execution
        rewards = self._execute_actions(actions)
        
        # Phase 4: Market Update
        self._update_market_state()
        
        # Phase 5: Performance Evaluation and Learning
        self._evaluate_performance()
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Create info dictionary
        info = {
            'step': self.current_step,
            'current_price': self.current_price,
            'agent_metrics': [self._get_agent_metrics(agent) for agent in self.agents],
            'communication_stats': self._get_communication_stats(),
            'coordination_stats': self._get_coordination_stats()
        }
        
        return next_state, rewards, done, info
    
    def _process_communication_phase(self):
        """Process communication between agents"""
        for i, agent in enumerate(self.agents):
            # Generate market information
            market_info = self._generate_market_info(agent)
            
            # Broadcast information
            if self.communication:
                message = self.communication.broadcast_message(i, {
                    'type': 'market_info',
                    'data': market_info,
                    'agent_type': agent['type']
                })
                
                # Store in agent's communication buffer
                agent['communication_buffer'].append(message)
    
    def _process_coordination_phase(self):
        """Process coordination between agents"""
        if not self.config.coordination_enabled:
            return
        
        # Identify coordination opportunities
        coordination_opportunities = self._identify_coordination_opportunities()
        
        for opportunity in coordination_opportunities:
            # Execute coordination
            coordination_result = self._execute_coordination(opportunity)
            
            if coordination_result:
                self.coordination_events.append(coordination_result)
    
    def _execute_actions(self, actions: List[np.ndarray]) -> List[float]:
        """Execute actions for all agents"""
        rewards = []
        
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            # Parse action
            action_type, action_size = self._parse_action(action)
            
            # Apply coordination effects
            if self.config.coordination_enabled:
                action_type, action_size = self._apply_coordination_effects(agent, action_type, action_size)
            
            # Execute trade
            reward, trade_info = self._execute_agent_trade(agent, action_type, action_size)
            rewards.append(reward)
            
            # Record trade
            if trade_info['executed']:
                agent['trades'].append(trade_info)
                
                # Broadcast trade signal
                if self.communication and self.config.information_sharing:
                    self.communication.broadcast_message(i, {
                        'type': 'trade_signal',
                        'signal': {
                            'action_type': action_type,
                            'action_size': action_size,
                            'price': self.current_price,
                            'timestamp': time.time()
                        }
                    })
        
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
    
    def _execute_agent_trade(self, agent: Dict[str, Any], action_type: str, action_size: float) -> Tuple[float, Dict[str, Any]]:
        """Execute a trade for a specific agent"""
        trade_info = {
            'agent_id': agent['id'],
            'agent_type': agent['type'],
            'action_type': action_type,
            'action_size': action_size,
            'current_price': self.current_price,
            'executed': False
        }
        
        reward = 0.0
        
        if action_type == 'buy' and action_size > 0:
            # Calculate trade size
            max_trade_value = agent['capital'] * self.config.max_position
            trade_value = max_trade_value * action_size
            trade_value = np.clip(trade_value, self.config.min_trade_size, self.config.max_trade_size)
            
            # Calculate shares to buy
            shares = trade_value / self.current_price
            
            # Apply transaction costs and slippage
            total_cost = trade_value * (1 + self.config.transaction_cost + self.config.slippage)
            
            if total_cost <= agent['capital']:
                # Execute trade
                agent['capital'] -= total_cost
                agent['position'] += shares
                
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
            max_shares = agent['position']
            shares_to_sell = max_shares * action_size
            
            if shares_to_sell > 0:
                # Calculate trade value
                trade_value = shares_to_sell * self.current_price
                
                # Apply transaction costs and slippage
                net_proceeds = trade_value * (1 - self.config.transaction_cost - self.config.slippage)
                
                # Execute trade
                agent['capital'] += net_proceeds
                agent['position'] -= shares_to_sell
                
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
    
    def _update_market_state(self):
        """Update market state"""
        # Get base price from data
        if self.current_step < len(self.data):
            base_price = self.data.iloc[self.current_step]['close']
        else:
            base_price = self.current_price
        
        # Add noise and trend
        noise = np.random.normal(0, base_price * 0.001)
        trend = base_price * 0.0001 * np.random.choice([-1, 1])
        
        self.current_price = base_price + noise + trend
        self.price_history.append(self.current_price)
        
        # Update agent portfolios
        for agent in self.agents:
            portfolio_value = agent['capital'] + (agent['position'] * self.current_price)
            agent['portfolio_values'].append(portfolio_value)
    
    def _evaluate_performance(self):
        """Evaluate agent performance and update learning"""
        for agent in self.agents:
            # Calculate performance metrics
            metrics = self._get_agent_metrics(agent)
            agent['performance_history'].append(metrics)
            
            # Update coordination state based on performance
            if metrics['total_return'] > 0.05:  # Good performance
                agent['coordination_state']['confidence'] = agent['coordination_state'].get('confidence', 0) + 0.1
            else:  # Poor performance
                agent['coordination_state']['confidence'] = max(0, agent['coordination_state'].get('confidence', 0) - 0.05)
    
    def _generate_market_info(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market information for an agent"""
        market_data = self.data.iloc[self.current_step]
        
        return {
            'price': self.current_price,
            'volume': market_data['volume'],
            'volatility': market_data['volatility'],
            'returns': market_data['returns'],
            'agent_type': agent['type'],
            'position': agent['position'],
            'confidence': agent['coordination_state'].get('confidence', 0.5)
        }
    
    def _identify_coordination_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for agent coordination"""
        opportunities = []
        
        # Check for market making opportunities
        if len(self.order_book['bids']) < 2 or len(self.order_book['asks']) < 2:
            opportunities.append({
                'type': 'market_making',
                'agents': [i for i, agent in enumerate(self.agents) if agent['type'] == 'market_maker'],
                'priority': 'high'
            })
        
        # Check for information sharing opportunities
        if self.communication:
            recent_messages = self.communication.get_recent_messages(0, 5)
            if len(recent_messages) < 3:
                opportunities.append({
                    'type': 'information_sharing',
                    'agents': list(range(self.num_agents)),
                    'priority': 'medium'
                })
        
        return opportunities
    
    def _execute_coordination(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute coordination based on opportunity"""
        if opportunity['type'] == 'market_making':
            return self._execute_market_making_coordination(opportunity)
        elif opportunity['type'] == 'information_sharing':
            return self._execute_information_sharing_coordination(opportunity)
        
        return None
    
    def _execute_market_making_coordination(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market making coordination"""
        return {
            'type': 'market_making',
            'agents': opportunity['agents'],
            'timestamp': time.time(),
            'success': True
        }
    
    def _execute_information_sharing_coordination(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Execute information sharing coordination"""
        return {
            'type': 'information_sharing',
            'agents': opportunity['agents'],
            'timestamp': time.time(),
            'success': True
        }
    
    def _apply_coordination_effects(self, agent: Dict[str, Any], action_type: str, action_size: float) -> Tuple[str, float]:
        """Apply coordination effects to agent actions"""
        # Simple coordination effects
        confidence = agent['coordination_state'].get('confidence', 0.5)
        
        # Adjust action size based on confidence
        adjusted_size = action_size * (0.5 + confidence)
        adjusted_size = np.clip(adjusted_size, 0.0, 1.0)
        
        return action_type, adjusted_size
    
    def _get_agent_metrics(self, agent: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for an agent"""
        if not agent['portfolio_values']:
            return {}
        
        portfolio_values = np.array(agent['portfolio_values'])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values) / np.max(portfolio_values) - 1
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'num_trades': len(agent['trades'])
        }
    
    def _get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        if not self.communication:
            return {}
        
        return {
            'total_messages': len(self.communication.message_history),
            'trust_scores': self.communication.trust_scores.copy(),
            'recent_messages': len(self.communication.message_history[-10:])
        }
    
    def _get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        return {
            'total_events': len(self.coordination_events),
            'market_making_events': len([e for e in self.coordination_events if e['type'] == 'market_making']),
            'information_sharing_events': len([e for e in self.coordination_events if e['type'] == 'information_sharing'])
        }
    
    def get_market_metrics(self) -> Dict[str, float]:
        """Calculate market performance metrics"""
        if not self.price_history:
            return {}
        
        prices = np.array(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        total_return = (prices[-1] - prices[0]) / prices[0]
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'final_price': prices[-1],
            'price_range': np.max(prices) - np.min(prices)
        }
    
    def render(self):
        """Render the current multi-agent state"""
        print(f"Step: {self.current_step}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Number of Agents: {len(self.agents)}")
        print(f"Communication Enabled: {self.config.communication_enabled}")
        print(f"Coordination Enabled: {self.config.coordination_enabled}")
        print("-" * 50)
        
        for i, agent in enumerate(self.agents):
            portfolio_value = agent['capital'] + (agent['position'] * self.current_price)
            confidence = agent['coordination_state'].get('confidence', 0.5)
            print(f"Agent {i} ({agent['type']}): ${portfolio_value:.2f}, Trades: {len(agent['trades'])}, Confidence: {confidence:.2f}")
        
        if self.communication:
            print(f"Total Messages: {len(self.communication.message_history)}")
            print(f"Trust Scores: {self.communication.trust_scores}")
        
        print(f"Coordination Events: {len(self.coordination_events)}")
        print("-" * 50) 