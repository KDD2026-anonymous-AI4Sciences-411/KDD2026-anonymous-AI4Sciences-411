import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import ast
import inspect
import logging
from dataclasses import dataclass
import re
from pathlib import Path
import time
import random

logger = logging.getLogger(__name__)

@dataclass
class ComponentConfig:
    """Configuration for LLM component generation"""
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3
    code_validation: bool = True

class PromptTemplates:
    """Advanced prompt templates with examples and sophisticated engineering"""
    
    @staticmethod
    def reward_function_prompt(market_context: str, agent_type: str = "single") -> str:
        """Generate sophisticated prompt for reward function creation with examples"""
        
        examples = PromptTemplates._get_reward_function_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in Reinforcement Learning (RL) 
and quantitative finance. Your task is to design a reward function for optimal trade 
execution in financial markets.

CONTEXT:
{market_context}

AGENT TYPE: {agent_type}

TASK: Generate a complete Python reward function that balances execution speed with 
market impact while considering transaction costs and risk management.

REQUIREMENTS:
1. The reward function must be differentiable and suitable for gradient-based optimization
2. Consider market impact: larger orders should have higher execution costs
3. Balance speed vs. impact: faster execution may have higher costs
4. Include transaction costs and slippage
5. Penalize excessive inventory risk
6. Reward successful order completion
7. Adapt to market volatility

EXAMPLES OF REWARD FUNCTIONS:

{examples}

CONSTRAINTS:
- Function must return a scalar reward value
- Must handle both positive and negative rewards
- Should be normalized appropriately
- Must be computationally efficient

OUTPUT FORMAT:
```python
def reward_function(state, action, next_state, market_data):
    """
    Reward function for trade execution.
    
    Args:
        state: Current market state (dict)
        action: Agent's action (dict)
        next_state: Next market state (dict)
        market_data: Historical market data (DataFrame)
    
    Returns:
        float: Reward value
    """
    # Your implementation here
```

Generate a reward function that follows the examples above and meets all requirements.
"""

    @staticmethod
    def _get_reward_function_examples() -> str:
        """Get examples of reward functions"""
        return """
EXAMPLE 1 - Basic Execution Reward:
```python
def basic_execution_reward(state, action, next_state, market_data):
    '''Basic reward function focusing on execution quality'''
    current_price = state['current_price']
    execution_price = next_state['execution_price']
    order_size = action['order_size']
    
    # Price improvement reward
    price_improvement = (current_price - execution_price) / current_price
    
    # Market impact penalty
    market_impact = abs(order_size) * 0.001  # 0.1% per unit
    
    # Transaction cost penalty
    transaction_cost = abs(order_size) * 0.0005  # 0.05% per unit
    
    # Inventory risk penalty
    inventory_risk = abs(state['inventory']) * 0.0001
    
    reward = price_improvement - market_impact - transaction_cost - inventory_risk
    return reward
```

EXAMPLE 2 - Advanced Risk-Aware Reward:
```python
def risk_aware_reward(state, action, next_state, market_data):
    '''Advanced reward function with risk management'''
    # Execution quality
    execution_quality = (state['target_price'] - next_state['execution_price']) / state['target_price']
    
    # Market impact (non-linear)
    order_size = abs(action['order_size'])
    market_impact = 0.001 * order_size + 0.0001 * order_size**2
    
    # Volatility-adjusted risk
    volatility = state.get('volatility', 0.02)
    risk_penalty = volatility * abs(state['inventory']) * 0.1
    
    # Time pressure reward
    time_remaining = state.get('time_remaining', 1.0)
    time_pressure = 1.0 / (1.0 + time_remaining)
    
    # Completion bonus
    completion_bonus = 0.1 if abs(state['remaining_order']) < 0.01 else 0.0
    
    reward = execution_quality - market_impact - risk_penalty + time_pressure + completion_bonus
    return reward
```

EXAMPLE 3 - Multi-Objective Reward:
```python
def multi_objective_reward(state, action, next_state, market_data):
    '''Multi-objective reward balancing multiple goals'''
    # Price efficiency
    price_efficiency = (state['vwap'] - next_state['execution_price']) / state['vwap']
    
    # Speed efficiency
    speed_efficiency = state.get('execution_speed', 0.5)
    
    # Risk efficiency
    risk_efficiency = 1.0 / (1.0 + abs(state['inventory']))
    
    # Market efficiency
    market_efficiency = 1.0 / (1.0 + state.get('market_impact', 0.0))
    
    # Weighted combination
    weights = {'price': 0.4, 'speed': 0.2, 'risk': 0.2, 'market': 0.2}
    reward = (weights['price'] * price_efficiency + 
              weights['speed'] * speed_efficiency + 
              weights['risk'] * risk_efficiency + 
              weights['market'] * market_efficiency)
    
    return reward
```
"""

    @staticmethod
    def network_architecture_prompt(state_dim: int, action_dim: int, agent_type: str = "single") -> str:
        """Generate prompt for network architecture with examples"""
        
        examples = PromptTemplates._get_network_architecture_examples()
        
        return f"""
You are an expert AI Research Engineer specializing in deep learning architectures 
for reinforcement learning. Design a neural network architecture for financial trading.

CONTEXT:
- State dimension: {state_dim}
- Action dimension: {action_dim}
- Agent type: {agent_type}

TASK: Create a neural network architecture suitable for RL-based trading that can:
1. Process market state information effectively
2. Learn optimal trading policies
3. Handle temporal dependencies in market data
4. Provide both policy and value estimates
5. Be computationally efficient for real-time trading

REQUIREMENTS:
1. Use actor-critic architecture
2. Include attention mechanisms for market data
3. Handle variable-length sequences
4. Provide uncertainty estimates
5. Include residual connections
6. Use appropriate activation functions
7. Consider gradient flow and stability

EXAMPLES OF NETWORK ARCHITECTURES:

{examples}

OUTPUT FORMAT:
```python
class TradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Your implementation here
        
    def forward(self, state):
        # Your implementation here
        pass
```

Generate a network architecture that follows the examples above and meets all requirements.
"""

    @staticmethod
    def _get_network_architecture_examples() -> str:
        """Get examples of network architectures"""
        return """
EXAMPLE 1 - LSTM-Based Trading Network:
```python
class LSTMTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state_sequence):
        lstm_out, _ = self.lstm(state_sequence)
        last_hidden = lstm_out[:, -1, :]
        action_probs = self.actor(last_hidden)
        value = self.critic(last_hidden)
        return action_probs, value
```

EXAMPLE 2 - Attention-Based Network:
```python
class AttentionTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(state_dim, num_heads)
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.ReLU(),
            nn.Linear(state_dim * 4, state_dim)
        )
        self.actor = nn.Linear(state_dim, action_dim)
        self.critic = nn.Linear(state_dim, 1)
    
    def forward(self, state_sequence):
        # Self-attention
        attn_out, _ = self.attention(state_sequence, state_sequence, state_sequence)
        attn_out = self.norm1(state_sequence + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(attn_out + ffn_out)
        
        # Output heads
        pooled = torch.mean(ffn_out, dim=1)
        action_probs = torch.tanh(self.actor(pooled))
        value = self.critic(pooled)
        return action_probs, value
```

EXAMPLE 3 - Residual Network:
```python
class ResidualTradingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, 256)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(4)
        ])
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        x = self.input_proj(state)
        for block in self.residual_blocks:
            x = block(x)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.layers(x))
```
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
    def _get_communication_examples() -> str:
        """Get examples of communication protocols"""
        return """
EXAMPLE 1 - Market Information Sharing:
```python
class MarketInformationProtocol:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.message_history = []
        self.market_views = {}
    
    def broadcast_market_view(self, agent_id, market_data):
        '''Broadcast market view to all agents'''
        message = {
            'type': 'market_view',
            'sender': agent_id,
            'timestamp': time.time(),
            'data': {
                'price_forecast': market_data['price_forecast'],
                'volatility_estimate': market_data['volatility'],
                'confidence': market_data['confidence']
            }
        }
        self.message_history.append(message)
        return message
    
    def get_consensus_view(self):
        '''Get consensus market view from all agents'''
        if not self.message_history:
            return None
        
        recent_messages = [m for m in self.message_history 
                          if time.time() - m['timestamp'] < 60]
        
        if not recent_messages:
            return None
        
        # Aggregate views
        price_forecasts = [m['data']['price_forecast'] for m in recent_messages]
        avg_forecast = np.mean(price_forecasts)
        
        return {
            'consensus_price': avg_forecast,
            'confidence': np.mean([m['data']['confidence'] for m in recent_messages])
        }
```

EXAMPLE 2 - Coordination Protocol:
```python
class CoordinationProtocol:
    def __init__(self, agent_types):
        self.agent_types = agent_types
        self.coordination_rules = {
            'market_maker': self._market_maker_rules,
            'informed_trader': self._informed_trader_rules,
            'noise_trader': self._noise_trader_rules
        }
    
    def coordinate_actions(self, agent_id, agent_type, proposed_action):
        '''Coordinate actions to avoid conflicts'''
        rules = self.coordination_rules.get(agent_type, lambda x: x)
        coordinated_action = rules(proposed_action)
        
        message = {
            'type': 'coordination',
            'sender': agent_id,
            'original_action': proposed_action,
            'coordinated_action': coordinated_action,
            'reason': f'Applied {agent_type} coordination rules'
        }
        
        return coordinated_action, message
    
    def _market_maker_rules(self, action):
        '''Rules for market maker coordination'''
        # Market makers should maintain liquidity
        if abs(action['order_size']) > 1000:
            action['order_size'] = np.sign(action['order_size']) * 1000
        return action
    
    def _informed_trader_rules(self, action):
        '''Rules for informed trader coordination'''
        # Informed traders should not front-run too aggressively
        if action.get('urgency', 0) > 0.8:
            action['urgency'] = 0.8
        return action
    
    def _noise_trader_rules(self, action):
        '''Rules for noise trader coordination'''
        # Noise traders should have limited impact
        if abs(action['order_size']) > 500:
            action['order_size'] = np.sign(action['order_size']) * 500
        return action
```

EXAMPLE 3 - Learning Protocol:
```python
class LearningProtocol:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.experience_buffer = []
        self.performance_history = {}
    
    def share_experience(self, agent_id, experience):
        '''Share learning experience with other agents'''
        message = {
            'type': 'experience',
            'sender': agent_id,
            'experience': {
                'state': experience['state'],
                'action': experience['action'],
                'reward': experience['reward'],
                'next_state': experience['next_state'],
                'performance': experience.get('performance', 0)
            }
        }
        
        self.experience_buffer.append(message)
        return message
    
    def get_shared_knowledge(self, agent_id):
        '''Get shared knowledge from other agents'''
        # Filter experiences from other agents
        other_experiences = [exp for exp in self.experience_buffer 
                           if exp['sender'] != agent_id]
        
        # Sort by performance
        other_experiences.sort(key=lambda x: x['experience']['performance'], 
                              reverse=True)
        
        # Return top experiences
        return other_experiences[:10]
```

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

EXAMPLE 3 - Noise Trader Profile:
```python
class NoiseTraderProfile:
    def __init__(self):
        self.personality_traits = {
            'risk_tolerance': 'high',
            'decision_speed': 'variable',
            'information_sharing': 'minimal',
            'collaboration_style': 'independent',
            'adaptation_rate': 'low'
        }
        
        self.trading_strategy = {
            'primary_goal': 'momentum_trading',
            'order_type_preference': 'mixed',
            'timing_strategy': 'reactive',
            'position_sizing': 'emotional',
            'risk_management': 'basic'
        }
        
        self.communication_style = {
            'message_frequency': 'irregular',
            'information_detail': 'superficial',
            'response_time': 'delayed',
            'trust_level': 'low'
        }
        
        self.learning_capabilities = {
            'pattern_recognition': 'basic',
            'market_adaptation': 'poor',
            'strategy_evolution': 'slow',
            'emotional_control': 'variable'
        }
    
    def get_behavioral_pattern(self):
        return {
            'order_placement': 'emotional_reactive',
            'information_processing': 'superficial',
            'risk_management': 'inconsistent',
            'market_analysis': 'technical_simple'
        }
```

EXAMPLE 4 - Multi-Agent Profile Manager:
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
        """Get profile for specific agent type"""
        return self.profiles.get(agent_type, None)
    
    def get_interaction_pattern(self, agent_type1, agent_type2):
        """Get interaction pattern between two agent types"""
        key = f"{agent_type1}_{agent_type2}"
        reverse_key = f"{agent_type2}_{agent_type1}"
        return self.interaction_patterns.get(key, 
                                           self.interaction_patterns.get(reverse_key, 'neutral'))
    
    def get_market_dynamics(self):
        """Get overall market dynamics from agent interactions"""
        return {
            'liquidity_level': 'high',
            'price_efficiency': 'moderate',
            'volatility': 'variable',
            'information_flow': 'asymmetric'
        }
```
"""

class CodeValidator:
    """Validates generated code for syntax and functionality"""
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, str]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    @staticmethod
    def validate_function_signature(code: str, expected_function: str) -> Tuple[bool, str]:
        """Validate function signature"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == expected_function:
                    return True, f"Found function {expected_function}"
            return False, f"Function {expected_function} not found"
        except Exception as e:
            return False, f"Error parsing function: {str(e)}"
    
    @staticmethod
    def validate_imports(code: str, required_imports: List[str]) -> Tuple[bool, str]:
        """Validate required imports"""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            missing = [imp for imp in required_imports if imp not in imports]
            if missing:
                return False, f"Missing imports: {missing}"
            return True, "All required imports present"
        except Exception as e:
            return False, f"Error checking imports: {str(e)}"

class LLMComponentGenerator:
    """Main class for LLM-powered component generation"""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.validator = CodeValidator()
        self.generation_history = []
        self.llm_model = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM model using production-ready integration"""
        try:
            from llm_integration import LLMFactory
            
            # Create LLM configuration
            llm_config = {
                "provider": "openai",  # Can be configured via config
                "model_name": self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            # Create LLM integration instance
            llm_integration = LLMFactory.create_llm(llm_config)
            logger.info(f"Successfully initialized LLM model: {self.config.model_name}")
            return llm_integration
            
        except ImportError:
            logger.warning("LLM integration module not available, using placeholder")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            return None
    
    def generate_reward_function(self, market_context: str, agent_type: str = "single") -> Dict[str, Any]:
        """Generate reward function using LLM"""
        prompt = PromptTemplates.reward_function_prompt(market_context, agent_type)
        
        for attempt in range(self.config.max_retries):
            try:
                code = self._generate_code_with_llm(prompt)
                validation_result = self._validate_reward_function(code)
                
                if validation_result['valid']:
                    result = {
                        'code': code,
                        'valid': True,
                        'type': 'reward_function',
                        'agent_type': agent_type,
                        'market_context': market_context,
                        'validation_message': validation_result['message']
                    }
                    self.generation_history.append(result)
                    return result
                else:
                    logger.warning(f"Reward function validation failed: {validation_result['message']}")
                    
            except Exception as e:
                logger.error(f"Error generating reward function (attempt {attempt + 1}): {str(e)}")
        
        # Fallback to template
        logger.warning("Using fallback reward function template")
        return self._get_fallback_reward_function(agent_type)
    
    def generate_network_architecture(self, state_dim: int, action_dim: int, agent_type: str = "single") -> Dict[str, Any]:
        """Generate network architecture using LLM"""
        prompt = PromptTemplates.network_architecture_prompt(state_dim, action_dim, agent_type)
        
        for attempt in range(self.config.max_retries):
            try:
                code = self._generate_code_with_llm(prompt)
                validation_result = self._validate_network_architecture(code)
                
                if validation_result['valid']:
                    result = {
                        'code': code,
                        'valid': True,
                        'type': 'network_architecture',
                        'state_dim': state_dim,
                        'action_dim': action_dim,
                        'agent_type': agent_type,
                        'validation_message': validation_result['message']
                    }
                    self.generation_history.append(result)
                    return result
                else:
                    logger.warning(f"Network architecture validation failed: {validation_result['message']}")
                    
            except Exception as e:
                logger.error(f"Error generating network architecture (attempt {attempt + 1}): {str(e)}")
        
        # Fallback to template
        logger.warning("Using fallback network architecture template")
        return self._get_fallback_network_architecture(state_dim, action_dim, agent_type)
    
    def generate_multi_agent_communication(self, num_agents: int, agent_types: List[str]) -> Dict[str, Any]:
        """Generate multi-agent communication protocol using LLM"""
        prompt = PromptTemplates.multi_agent_communication_prompt(num_agents, agent_types)
        
        for attempt in range(self.config.max_retries):
            try:
                code = self._generate_code_with_llm(prompt)
                validation_result = self._validate_communication_protocol(code)
                
                if validation_result['valid']:
                    result = {
                        'code': code,
                        'valid': True,
                        'type': 'multi_agent_communication',
                        'num_agents': num_agents,
                        'agent_types': agent_types,
                        'validation_message': validation_result['message']
                    }
                    self.generation_history.append(result)
                    return result
                else:
                    logger.warning(f"Communication protocol validation failed: {validation_result['message']}")
                    
            except Exception as e:
                logger.error(f"Error generating communication protocol (attempt {attempt + 1}): {str(e)}")
        
        # Fallback to template
        logger.warning("Using fallback communication protocol template")
        return self._get_fallback_communication_protocol(num_agents, agent_types)
    
    def generate_llm4profiling(self, agent_types: List[str], market_context: str) -> Dict[str, Any]:
        """Generate LLM4Profiling - intelligent entity configurations using LLM"""
        prompt = PromptTemplates.llm4profiling_prompt(agent_types, market_context)
        
        for attempt in range(self.config.max_retries):
            try:
                code = self._generate_code_with_llm(prompt)
                validation_result = self._validate_profiling_config(code)
                
                if validation_result['valid']:
                    result = {
                        'code': code,
                        'valid': True,
                        'type': 'llm4profiling',
                        'agent_types': agent_types,
                        'market_context': market_context,
                        'validation_message': validation_result['message']
                    }
                    self.generation_history.append(result)
                    return result
                else:
                    logger.warning(f"LLM4Profiling validation failed: {validation_result['message']}")
                    
            except Exception as e:
                logger.error(f"Error generating LLM4Profiling (attempt {attempt + 1}): {str(e)}")
        
        # Fallback to template
        logger.warning("Using fallback LLM4Profiling template")
        return self._get_fallback_profiling_config(agent_types)
    
    def _generate_code_with_llm(self, prompt: str) -> str:
        """Generate code using LLM (production implementation)"""
        if self.llm_model is None:
            logger.warning("LLM model not available, using fallback")
            return self._get_fallback_code(prompt)
        
        try:
            logger.info("Generating code with LLM")
            
            # Generate code using the LLM integration
            generated_code = self.llm_model.generate_code(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            logger.info(f"Generated code length: {len(generated_code)}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code with LLM: {str(e)}")
            return self._get_fallback_code(prompt)
    
    def _get_fallback_code(self, prompt: str) -> str:
        """Get fallback code when LLM is not available"""
        # Extract code block from prompt if it exists
        code_match = re.search(r'```python\n(.*?)\n```', prompt, re.DOTALL)
        if code_match:
            return code_match.group(1)
        

    def _validate_reward_function(self, code: str) -> Dict[str, Any]:
        """Validate generated reward function"""
        # Check syntax
        syntax_valid, syntax_msg = self.validator.validate_python_syntax(code)
        if not syntax_valid:
            return {'valid': False, 'message': syntax_msg}
        
        # Check function signature
        signature_valid, signature_msg = self.validator.validate_function_signature(code, 'reward_function')
        if not signature_valid:
            return {'valid': False, 'message': signature_msg}
        
        return {'valid': True, 'message': 'Valid reward function'}
    
    def _validate_network_architecture(self, code: str) -> Dict[str, Any]:
        """Validate generated network architecture"""
        # Check syntax
        syntax_valid, syntax_msg = self.validator.validate_python_syntax(code)
        if not syntax_valid:
            return {'valid': False, 'message': syntax_msg}
        
        # Check for class definition
        if 'class' not in code:
            return {'valid': False, 'message': 'No class definition found'}
        
        # Check for forward method
        if 'def forward' not in code:
            return {'valid': False, 'message': 'No forward method found'}
        
        return {'valid': True, 'message': 'Valid network architecture'}
    
    def _validate_communication_protocol(self, code: str) -> Dict[str, Any]:
        """Validate generated communication protocol"""
        # Check syntax
        syntax_valid, syntax_msg = self.validator.validate_python_syntax(code)
        if not syntax_valid:
            return {'valid': False, 'message': syntax_msg}
        
        # Check for class definition
        if 'class' not in code:
            return {'valid': False, 'message': 'No class definition found'}
        
        return {'valid': True, 'message': 'Valid communication protocol'}
    
    def _validate_profiling_config(self, code: str) -> Dict[str, Any]:
        """Validate generated LLM4Profiling configuration"""
        # Check syntax
        syntax_valid, syntax_msg = self.validator.validate_python_syntax(code)
        if not syntax_valid:
            return {'valid': False, 'message': syntax_msg}
        
        # Check for class definition
        if 'class' not in code:
            return {'valid': False, 'message': 'No class definition found'}
        
        # Check for profile-related methods
        if 'get_agent_profile' not in code:
            return {'valid': False, 'message': 'No get_agent_profile method found'}
        
        return {'valid': True, 'message': 'Valid LLM4Profiling configuration'}
    
    def _get_fallback_reward_function(self, agent_type: str) -> Dict[str, Any]:
        """Get fallback reward function template"""
        code = f"""
def reward_function(state, action, next_state, market_data):
    '''Fallback reward function for {agent_type} agent'''
    # Basic execution quality reward
    execution_price = next_state.get('execution_price', state.get('current_price', 100))
    current_price = state.get('current_price', 100)
    
    # Price improvement
    price_improvement = (current_price - execution_price) / current_price
    
    # Market impact penalty
    order_size = abs(action.get('order_size', 0))
    market_impact = order_size * 0.001
    
    # Transaction cost
    transaction_cost = order_size * 0.0005
    
    reward = price_improvement - market_impact - transaction_cost
    return reward
"""
        return {
            'code': code,
            'valid': True,
            'type': 'reward_function',
            'agent_type': agent_type,
            'is_fallback': True
        }
    
    def _get_fallback_network_architecture(self, state_dim: int, action_dim: int, agent_type: str) -> Dict[str, Any]:
        """Get fallback network architecture template"""
        code = f"""
class TradingNetwork(nn.Module):
    def __init__(self, state_dim={state_dim}, action_dim={action_dim}):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
"""
        return {
            'code': code,
            'valid': True,
            'type': 'network_architecture',
            'state_dim': state_dim,
            'action_dim': action_dim,
            'agent_type': agent_type,
            'is_fallback': True
        }
    
    def _get_fallback_communication_protocol(self, num_agents: int, agent_types: List[str]) -> Dict[str, Any]:
        """Get fallback communication protocol template"""
        code = f"""
class MultiAgentCommunication:
    def __init__(self, num_agents={num_agents}, agent_types={agent_types}):
        self.num_agents = num_agents
        self.agent_types = agent_types
        self.message_history = []
    
    def broadcast_message(self, sender_id, message):
        '''Broadcast message to all agents'''
        message_record = {{
            'sender': sender_id,
            'message': message,
            'timestamp': time.time()
        }}
        self.message_history.append(message_record)
        return message_record
    
    def get_messages(self, agent_id):
        '''Get messages for specific agent'''
        return [msg for msg in self.message_history if msg['sender'] != agent_id]
"""
        return {
            'code': code,
            'valid': True,
            'type': 'multi_agent_communication',
            'num_agents': num_agents,
            'agent_types': agent_types,
            'is_fallback': True
        }
    
    def _get_fallback_profiling_config(self, agent_types: List[str]) -> Dict[str, Any]:
        """Get fallback LLM4Profiling configuration template"""
        code = f"""
class IntelligentEntityProfiles:
    def __init__(self, agent_types={agent_types}):
        self.agent_types = agent_types
        self.profiles = {{
            'market_maker': {{
                'personality_traits': {{
                    'risk_tolerance': 'conservative',
                    'decision_speed': 'fast',
                    'information_sharing': 'high'
                }},
                'trading_strategy': {{
                    'primary_goal': 'liquidity_provision',
                    'order_type_preference': 'limit_orders'
                }}
            }},
            'informed_trader': {{
                'personality_traits': {{
                    'risk_tolerance': 'moderate',
                    'decision_speed': 'medium',
                    'information_sharing': 'selective'
                }},
                'trading_strategy': {{
                    'primary_goal': 'alpha_generation',
                    'order_type_preference': 'market_orders'
                }}
            }},
            'noise_trader': {{
                'personality_traits': {{
                    'risk_tolerance': 'high',
                    'decision_speed': 'variable',
                    'information_sharing': 'minimal'
                }},
                'trading_strategy': {{
                    'primary_goal': 'momentum_trading',
                    'order_type_preference': 'mixed'
                }}
            }}
        }}
    
    def get_agent_profile(self, agent_type):
        '''Get profile for specific agent type'''
        return self.profiles.get(agent_type, None)
    
    def get_interaction_pattern(self, agent_type1, agent_type2):
        '''Get interaction pattern between two agent types'''
        patterns = {{
            'market_maker_informed': 'cooperative_competitive',
            'market_maker_noise': 'liquidity_provision',
            'informed_noise': 'information_asymmetry'
        }}
        key = f"{{agent_type1}}_{{agent_type2}}"
        reverse_key = f"{{agent_type2}}_{{agent_type1}}"
        return patterns.get(key, patterns.get(reverse_key, 'neutral'))
"""
        return {
            'code': code,
            'valid': True,
            'type': 'llm4profiling',
            'agent_types': agent_types,
            'is_fallback': True
        } 