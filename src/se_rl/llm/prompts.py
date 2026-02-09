"""
LLM Prompt Templates for SE-RL Framework
=======================================

This module contains sophisticated prompt templates for generating
RL components using LLMs, including examples and advanced prompting techniques.

Author: AI Research Engineer
Date: 2024
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PromptTemplates:
    """Advanced prompt templates with sophisticated engineering techniques"""
    
    @staticmethod
    def reward_function_prompt(market_context: str, agent_type: str = "single") -> str:
        """Generate sophisticated prompt for reward function creation"""
        
        examples = PromptTemplates._get_reward_function_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in Reinforcement Learning (RL)
and quantitative finance. Your task is to design a reward function for optimal trade
execution in financial markets.

CONTEXT:
- Market Environment: {market_context}
- Agent Type: {agent_type}

TASK: Create a reward function that optimizes trading performance by considering:
1. Execution Quality: Minimize market impact and slippage
2. Risk Management: Control position size and exposure
3. Profit Maximization: Optimize for total return
4. Transaction Costs: Account for fees and spreads
5. Market Conditions: Adapt to volatility and liquidity

REQUIREMENTS:
1. Function must return a scalar reward value
2. Consider both immediate and long-term effects
3. Balance exploration vs exploitation
4. Handle edge cases (no trades, extreme market conditions)
5. Include risk-adjusted metrics
6. Consider market microstructure effects

EXAMPLES OF REWARD FUNCTIONS:

{examples}

OUTPUT FORMAT:
```python
def reward_function(state, action, next_state, market_data):
    """
    Reward function for {agent_type} agent in {market_context}
    
    Args:
        state: Current market state
        action: Agent's action (position size, order type)
        next_state: Next market state
        market_data: Additional market information
        
    Returns:
        float: Reward value
    """
    # Your implementation here
    pass
```

Generate a reward function that follows the examples above and meets all requirements.
"""

    @staticmethod
    def network_architecture_prompt(state_dim: int, action_dim: int, agent_type: str = "single") -> str:
        """Generate prompt for network architecture with examples"""
        
        examples = PromptTemplates._get_network_architecture_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in deep learning architectures
for reinforcement learning. Design a neural network architecture for financial trading.

CONTEXT:
- State Dimension: {state_dim}
- Action Dimension: {action_dim}
- Agent Type: {agent_type}

TASK: Create a neural network architecture that:
1. Processes financial time-series data efficiently
2. Captures temporal dependencies and patterns
3. Handles multi-dimensional state representations
4. Outputs appropriate action distributions
5. Balances complexity with computational efficiency

REQUIREMENTS:
1. Use PyTorch nn.Module as base class
2. Include appropriate activation functions
3. Consider attention mechanisms for time-series
4. Implement residual connections where beneficial
5. Handle variable input sequences
6. Include dropout for regularization
7. Optimize for GPU memory usage

EXAMPLES OF NETWORK ARCHITECTURES:

{examples}

OUTPUT FORMAT:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TradingNetwork(nn.Module):
    def __init__(self, state_dim={state_dim}, action_dim={action_dim}):
        super().__init__()
        # Your implementation here
        
    def forward(self, state):
        # Your implementation here
        pass
```

Generate a network architecture that follows the examples above and meets all requirements.
"""

    @staticmethod
    def multi_agent_communication_prompt(num_agents: int, agent_types: List[str]) -> str:
        """Generate prompt for multi-agent communication with examples"""
        
        examples = PromptTemplates._get_communication_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in multi-agent systems and
financial market simulation. Design a communication protocol for multi-agent trading.

CONTEXT:
- Number of agents: {num_agents}
- Agent types: {', '.join(agent_types)}

TASK: Create a communication protocol that enables agents to:
1. Share market information efficiently
2. Coordinate trading strategies
3. Avoid conflicts and market manipulation
4. Maintain market stability
5. Learn from each other's behavior

REQUIREMENTS:
1. Define message formats and protocols
2. Implement information sharing mechanisms
3. Handle different agent types appropriately
4. Include coordination strategies
5. Provide conflict resolution
6. Ensure market efficiency
7. Support learning and adaptation

EXAMPLES OF COMMUNICATION PROTOCOLS:

{examples}

OUTPUT FORMAT:
```python
class MultiAgentCommunication:
    def __init__(self, num_agents, agent_types):
        # Your implementation here
        pass
        
    def broadcast_message(self, sender_id, message):
        # Your implementation here
        pass
```

Generate a communication protocol that follows the examples above and meets all requirements.
"""

    @staticmethod
    def llm4profiling_prompt(agent_types: List[str], market_context: str) -> str:
        """Generate prompt for LLM4Profiling - intelligent entity configuration"""
        
        examples = PromptTemplates._get_profiling_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in multi-agent systems and
intelligent entity profiling. Design intelligent entity configurations for multi-agent trading.

CONTEXT:
- Agent types: {', '.join(agent_types)}
- Market context: {market_context}

TASK: Create intelligent entity configurations that define the behavior, capabilities, 
and characteristics of each agent type in the multi-agent trading system.

REQUIREMENTS:
1. Define personality traits and behavioral patterns for each agent type
2. Specify trading strategies and decision-making logic
3. Configure risk tolerance and investment preferences
4. Set communication styles and information sharing patterns
5. Define learning capabilities and adaptation mechanisms
6. Specify market impact and order execution preferences
7. Configure collaboration and competition dynamics

EXAMPLES OF INTELLIGENT ENTITY PROFILES:

{examples}

OUTPUT FORMAT:
```python
class IntelligentEntityProfiles:
    def __init__(self, agent_types):
        # Your implementation here
        pass
        
    def get_agent_profile(self, agent_type):
        # Your implementation here
        pass
```

Generate intelligent entity profiles that follow the examples above and meet all requirements.
"""

    @staticmethod
    def imagination_module_prompt(market_context: str) -> str:
        """Generate prompt for imagination module with examples"""
        
        examples = PromptTemplates._get_imagination_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in model-based reinforcement
learning and financial modeling. Design an imagination module for predicting
future market states and rewards.

CONTEXT:
- Market Environment: {market_context}

TASK: Create an imagination module that:
1. Predicts future market states based on current conditions
2. Estimates potential rewards for different actions
3. Models market dynamics and interactions
4. Handles uncertainty and stochasticity
5. Provides multiple plausible futures

REQUIREMENTS:
1. Implement probabilistic state prediction
2. Include reward estimation mechanisms
3. Handle multi-step prediction horizons
4. Consider market microstructure effects
5. Provide uncertainty quantification
6. Support parallel imagination rollouts
7. Optimize for computational efficiency

EXAMPLES OF IMAGINATION MODULES:

{examples}

OUTPUT FORMAT:
```python
class ImaginationModule:
    def __init__(self, state_dim, action_dim, horizon=10):
        # Your implementation here
        pass

    def imagine_future(self, current_state, action, num_samples=100):
        # Your implementation here
        pass
```

Generate an imagination module that follows the examples above and meets all requirements.
"""

    @staticmethod
    def _get_reward_function_examples() -> str:
        """Get examples of reward functions"""
        return """
EXAMPLE 1 - Basic Execution Quality Reward:
```python
def basic_execution_reward(state, action, next_state, market_data):
    '''Basic reward focusing on execution quality'''
    # Extract key metrics
    price_impact = abs(next_state['price'] - state['price']) / state['price']
    volume_ratio = action['volume'] / market_data['avg_volume']
    spread_cost = market_data['bid_ask_spread'] * action['volume']
    
    # Calculate reward components
    execution_quality = 1.0 / (1.0 + price_impact)
    volume_penalty = 1.0 / (1.0 + volume_ratio)
    cost_penalty = 1.0 / (1.0 + spread_cost)
    
    # Combine rewards
    reward = execution_quality * volume_penalty * cost_penalty
    return reward
```

EXAMPLE 2 - Risk-Aware Reward Function:
```python
def risk_aware_reward(state, action, next_state, market_data):
    '''Reward function with risk management'''
    # Calculate returns
    returns = (next_state['portfolio_value'] - state['portfolio_value']) / state['portfolio_value']
    
    # Risk metrics
    position_size = abs(action['position'])
    volatility = market_data['volatility']
    var_penalty = max(0, position_size * volatility - 0.02)  # 2% VaR limit
    
    # Risk-adjusted reward
    risk_adjusted_return = returns / (1.0 + var_penalty)
    
    # Transaction cost penalty
    transaction_cost = market_data['transaction_cost'] * abs(action['position'])
    
    reward = risk_adjusted_return - transaction_cost
    return reward
```

EXAMPLE 3 - Multi-Objective Reward Function:
```python
def multi_objective_reward(state, action, next_state, market_data):
    '''Multi-objective reward balancing multiple goals'''
    # Performance metrics
    returns = (next_state['portfolio_value'] - state['portfolio_value']) / state['portfolio_value']
    sharpe_ratio = returns / (market_data['volatility'] + 1e-6)
    
    # Execution metrics
    slippage = abs(next_state['execution_price'] - state['mid_price']) / state['mid_price']
    market_impact = action['volume'] / market_data['avg_volume']
    
    # Risk metrics
    drawdown = (state['peak_value'] - next_state['portfolio_value']) / state['peak_value']
    position_concentration = action['position'] / state['total_assets']
    
    # Weighted combination
    reward = (
        0.4 * returns +
        0.2 * sharpe_ratio +
        0.2 * (1.0 - slippage) +
        0.1 * (1.0 - market_impact) +
        0.1 * (1.0 - drawdown)
    )
    
    return reward
```
"""

    @staticmethod
    def _get_network_architecture_examples() -> str:
        """Get examples of network architectures"""
        return """
EXAMPLE 1 - LSTM-based Architecture:
```python
class LSTMTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        # state shape: (batch_size, seq_len, state_dim)
        lstm_out, _ = self.lstm(state)
        x = lstm_out[:, -1, :]  # Take last timestep
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
```

EXAMPLE 2 - Attention-based Architecture:
```python
class AttentionTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=8):
        super().__init__()
        self.embedding = nn.Linear(state_dim, 128)
        self.attention = nn.MultiheadAttention(128, num_heads)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.layer_norm = nn.LayerNorm(128)
        
    def forward(self, state):
        # state shape: (batch_size, seq_len, state_dim)
        x = self.embedding(state)
        x = x.transpose(0, 1)  # (seq_len, batch_size, 128)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.transpose(0, 1)  # (batch_size, seq_len, 128)
        
        x = x.mean(dim=1)  # Global average pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
```

EXAMPLE 3 - Residual Architecture:
```python
class ResidualTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input_fc = nn.Linear(state_dim, 128)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(3)
        ])
        self.output_fc = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.input_fc(state))
        
        for block in self.residual_blocks:
            x = block(x)
            
        x = self.output_fc(x)
        return F.softmax(x, dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm(x + residual)
        return F.relu(x)
```
"""

    @staticmethod
    def _get_communication_examples() -> str:
        """Get examples of communication protocols"""
        return """
EXAMPLE 1 - Information Sharing Protocol:
```python
class InformationSharingProtocol:
    def __init__(self, num_agents, agent_types):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.message_history = []
        self.trust_scores = {i: 1.0 for i in range(num_agents)}
        
    def broadcast_market_info(self, sender_id, market_data):
        message = {
            'type': 'market_info',
            'sender': sender_id,
            'data': market_data,
            'timestamp': time.time(),
            'confidence': self.trust_scores[sender_id]
        }
        self.message_history.append(message)
        return message
        
    def get_aggregated_info(self, agent_id):
        recent_messages = [m for m in self.message_history[-10:] 
                          if m['sender'] != agent_id]
        return self._aggregate_messages(recent_messages)
```

EXAMPLE 2 - Coordination Protocol:
```python
class CoordinationProtocol:
    def __init__(self, num_agents, agent_types):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.coordination_state = {}
        
    def propose_action(self, agent_id, action, reasoning):
        proposal = {
            'agent_id': agent_id,
            'action': action,
            'reasoning': reasoning,
            'timestamp': time.time()
        }
        return proposal
        
    def coordinate_actions(self, proposals):
        # Implement voting or consensus mechanism
        votes = {}
        for proposal in proposals:
            action_key = str(proposal['action'])
            votes[action_key] = votes.get(action_key, 0) + 1
            
        best_action = max(votes.items(), key=lambda x: x[1])[0]
        return best_action
```

EXAMPLE 3 - Learning Protocol:
```python
class LearningProtocol:
    def __init__(self, num_agents, agent_types):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.experience_buffer = []
        
    def share_experience(self, agent_id, experience):
        shared_exp = {
            'agent_id': agent_id,
            'state': experience['state'],
            'action': experience['action'],
            'reward': experience['reward'],
            'next_state': experience['next_state'],
            'success': experience['reward'] > 0
        }
        self.experience_buffer.append(shared_exp)
        return shared_exp
        
    def get_peer_experiences(self, agent_id, num_experiences=10):
        peer_experiences = [exp for exp in self.experience_buffer[-100:]
                           if exp['agent_id'] != agent_id and exp['success']]
        return random.sample(peer_experiences, min(num_experiences, len(peer_experiences)))
```
"""

    @staticmethod
    def _get_profiling_examples() -> str:
        """Get examples of intelligent entity profiles"""
        return """
EXAMPLE 1 - Market Maker Profile:
```python
class MarketMakerProfile:
    def __init__(self):
        self.personality_traits = {
            'risk_tolerance': 'conservative',
            'decision_speed': 'fast',
            'information_sharing': 'high',
            'collaboration_style': 'cooperative',
            'adaptation_rate': 'medium'
        }
        
        self.trading_strategy = {
            'primary_goal': 'liquidity_provision',
            'order_type_preference': 'limit_orders',
            'spread_management': 'dynamic',
            'inventory_control': 'aggressive',
            'risk_management': 'strict'
        }
        
        self.communication_style = {
            'message_frequency': 'high',
            'information_detail': 'comprehensive',
            'response_time': 'immediate',
            'trust_level': 'high'
        }
        
        self.learning_capabilities = {
            'pattern_recognition': 'excellent',
            'market_adaptation': 'good',
            'strategy_evolution': 'moderate',
            'collaborative_learning': 'high'
        }
    
    def get_behavioral_pattern(self):
        return {
            'order_placement': 'frequent_small_orders',
            'price_quoting': 'competitive_spreads',
            'risk_management': 'continuous_monitoring',
            'market_analysis': 'real_time'
        }
```

EXAMPLE 2 - Informed Trader Profile:
```python
class InformedTraderProfile:
    def __init__(self):
        self.personality_traits = {
            'risk_tolerance': 'moderate',
            'decision_speed': 'medium',
            'information_sharing': 'selective',
            'collaboration_style': 'competitive',
            'adaptation_rate': 'high'
        }
        
        self.trading_strategy = {
            'primary_goal': 'alpha_generation',
            'order_type_preference': 'market_orders',
            'timing_strategy': 'opportunistic',
            'position_sizing': 'dynamic',
            'risk_management': 'sophisticated'
        }
        
        self.communication_style = {
            'message_frequency': 'low',
            'information_detail': 'strategic',
            'response_time': 'deliberate',
            'trust_level': 'conditional'
        }
        
        self.learning_capabilities = {
            'pattern_recognition': 'outstanding',
            'market_adaptation': 'excellent',
            'strategy_evolution': 'rapid',
            'information_processing': 'superior'
        }
    
    def get_behavioral_pattern(self):
        return {
            'order_placement': 'large_opportunistic_orders',
            'information_gathering': 'extensive',
            'risk_management': 'sophisticated',
            'market_analysis': 'deep_fundamental'
        }
```

EXAMPLE 3 - Multi-Agent Profile Manager:
```python
class IntelligentEntityProfiles:
    def __init__(self, agent_types):
        self.agent_types = agent_types
        self.profiles = {
            'market_maker': MarketMakerProfile(),
            'informed_trader': InformedTraderProfile(),
            'noise_trader': NoiseTraderProfile()
        }
        
        self.interaction_patterns = {
            'market_maker_informed': 'cooperative_competitive',
            'market_maker_noise': 'liquidity_provision',
            'informed_noise': 'information_asymmetry',
            'all_agents': 'market_efficiency'
        }
    
    def get_agent_profile(self, agent_type):
        '''Get profile for specific agent type'''
        return self.profiles.get(agent_type, None)
    
    def get_interaction_pattern(self, agent_type1, agent_type2):
        '''Get interaction pattern between two agent types'''
        key = f"{agent_type1}_{agent_type2}"
        reverse_key = f"{agent_type2}_{agent_type1}"
        return self.interaction_patterns.get(key, 
                                           self.interaction_patterns.get(reverse_key, 'neutral'))
    
    def get_market_dynamics(self):
        '''Get overall market dynamics from agent interactions'''
        return {
            'liquidity_level': 'high',
            'price_efficiency': 'moderate',
            'volatility': 'variable',
            'information_flow': 'asymmetric'
        }
```
"""

    @staticmethod
    def _get_imagination_examples() -> str:
        """Get examples of imagination modules"""
        return """
EXAMPLE 1 - Probabilistic Imagination Module:
```python
class ProbabilisticImaginationModule:
    def __init__(self, state_dim, action_dim, horizon=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # State transition model
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim * 2)  # Mean and variance
        )
        
        # Reward model
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def imagine_future(self, current_state, action, num_samples=100):
        imagined_states = []
        imagined_rewards = []
        
        for _ in range(num_samples):
            # Predict next state
            combined_input = torch.cat([current_state, action], dim=-1)
            transition_output = self.transition_model(combined_input)
            
            mean, log_var = torch.chunk(transition_output, 2, dim=-1)
            std = torch.exp(0.5 * log_var)
            next_state = mean + std * torch.randn_like(std)
            
            # Predict reward
            reward = self.reward_model(combined_input)
            
            imagined_states.append(next_state)
            imagined_rewards.append(reward)
        
        return torch.stack(imagined_states), torch.stack(imagined_rewards)
```

EXAMPLE 2 - Multi-Step Imagination Module:
```python
class MultiStepImaginationModule:
    def __init__(self, state_dim, action_dim, horizon=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # LSTM-based transition model
        self.transition_lstm = nn.LSTM(
            state_dim + action_dim, 128, batch_first=True
        )
        self.transition_head = nn.Linear(128, state_dim)
        
        # Reward prediction
        self.reward_head = nn.Linear(128, 1)
    
    def imagine_future(self, current_state, action_sequence, num_samples=100):
        batch_size = current_state.shape[0]
        imagined_trajectories = []
        
        for _ in range(num_samples):
            trajectory = [current_state]
            rewards = []
            
            hidden = None
            for t in range(self.horizon):
                if t < len(action_sequence):
                    action = action_sequence[t]
                else:
                    action = torch.zeros(batch_size, self.action_dim)
                
                combined_input = torch.cat([trajectory[-1], action], dim=-1)
                combined_input = combined_input.unsqueeze(1)  # Add time dimension
                
                lstm_out, hidden = self.transition_lstm(combined_input, hidden)
                
                next_state = self.transition_head(lstm_out.squeeze(1))
                reward = self.reward_head(lstm_out.squeeze(1))
                
                trajectory.append(next_state)
                rewards.append(reward)
            
            imagined_trajectories.append({
                'states': torch.stack(trajectory),
                'rewards': torch.stack(rewards)
            })
        
        return imagined_trajectories
```

EXAMPLE 3 - Uncertainty-Aware Imagination Module:
```python
class UncertaintyAwareImaginationModule:
    def __init__(self, state_dim, action_dim, horizon=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Ensemble of transition models
        self.transition_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            ) for _ in range(5)
        ])
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(state_dim + action_dim, 1)
    
    def imagine_future(self, current_state, action, num_samples=100):
        imagined_states = []
        uncertainties = []
        
        for _ in range(num_samples):
            # Get predictions from ensemble
            combined_input = torch.cat([current_state, action], dim=-1)
            ensemble_predictions = []
            
            for model in self.transition_models:
                pred = model(combined_input)
                ensemble_predictions.append(pred)
            
            ensemble_predictions = torch.stack(ensemble_predictions)
            
            # Calculate mean and uncertainty
            mean_prediction = ensemble_predictions.mean(dim=0)
            uncertainty = ensemble_predictions.std(dim=0)
            
            # Sample from distribution
            next_state = mean_prediction + uncertainty * torch.randn_like(uncertainty)
            
            imagined_states.append(next_state)
            uncertainties.append(uncertainty.mean())
        
        return torch.stack(imagined_states), torch.stack(uncertainties)
```
""" 