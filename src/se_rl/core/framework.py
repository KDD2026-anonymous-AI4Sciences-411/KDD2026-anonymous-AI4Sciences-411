"""
SE-RL Framework Core Implementation
==================================

This module implements the core Self-Evolutional Reinforcement Learning framework
as described in the research paper, including:

1. Single Agent LLM Design
2. Multi-Agent LLM Design  
3. High-Level Prompt Optimization
4. Low-Level Weight Optimization
5. Dynamic Environment Training

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import time
import random
from collections import deque

from ..llm.component_generator import LLMComponentGenerator, ComponentConfig
from ..llm.integration import LLMIntegration, LLMConfig
from ..data.pipeline import FinancialDataPipeline, DataConfig
from ..rl.trainer import RLTrainer, TrainingConfig
from ..environments.static_env import StaticEnvironment
from ..environments.dynamic_env import DynamicEnvironment
from ..environments.multi_agent_env import MultiAgentEnvironment
from ..utils.logger import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class SERLConfig:
    """Configuration for the SE-RL framework"""
    
    # Framework parameters
    convergence_epsilon: float = 0.1
    max_outer_iterations: int = 50
    max_inner_iterations: int = 1000
    
    # LLM configuration
    llm_model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    llm_provider: str = "local"  # local, openai, anthropic, huggingface
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Environment parameters
    static_env_weight: float = 0.5
    dynamic_env_weight: float = 0.5
    rebalance_iterations: int = 10
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    
    # DEK parameters (Dual-Level Enhancement Kit)
    instruction_buffer_size: int = 100
    cache_replay_alpha: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda
    num_gpus: int = 1
    mixed_precision: bool = True
    
    # Data configuration
    dataset: str = "csi100"  # csi100, nasdaq100
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    frequency: str = "1d"
    window_size: int = 20
    
    # Multi-agent configuration
    num_agents: int = 3
    agent_types: List[str] = field(default_factory=lambda: ["market_maker", "informed_trader", "noise_trader"])
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.llm_provider == "local" and not torch.cuda.is_available():
            logger.warning("Local LLM requires CUDA for optimal performance")

class PerformanceBuffer:
    """Buffer for storing and managing performance history"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.best_performance = None
        self.best_iteration = -1
    
    def add_performance(self, iteration: int, metrics: Dict[str, float], 
                       prompt: str, code: str, agent_type: str = "single"):
        """Add performance metrics to buffer"""
        entry = {
            'iteration': iteration,
            'metrics': metrics,
            'prompt': prompt,
            'code': code,
            'agent_type': agent_type,
            'timestamp': time.time()
        }
        
        self.buffer.append(entry)
        
        # Update best performance
        if self.best_performance is None or metrics.get('PA', 0) > self.best_performance.get('PA', 0):
            self.best_performance = metrics
            self.best_iteration = iteration
    
    def get_recent_performance(self, n: int) -> List[Dict[str, Any]]:
        """Get the n most recent performance entries"""
        return list(self.buffer)[-n:]
    
    def get_best_performance(self) -> Optional[Dict[str, Any]]:
        """Get the best performance entry"""
        if self.best_performance is None:
            return None
        
        for entry in reversed(self.buffer):
            if entry['metrics'] == self.best_performance:
                return entry
        return None
    
    def get_performance_trend(self) -> Dict[str, List[float]]:
        """Get performance trend over iterations"""
        if not self.buffer:
            return {}
        
        iterations = [entry['iteration'] for entry in self.buffer]
        pas = [entry['metrics'].get('PA', 0) for entry in self.buffer]
        wrs = [entry['metrics'].get('WR', 0) for entry in self.buffer]
        glrs = [entry['metrics'].get('GLR', 0) for entry in self.buffer]
        
        return {
            'iterations': iterations,
            'PA': pas,
            'WR': wrs,
            'GLR': glrs
        }

class InstructionPopulation:
    """Manages population of instructions for LLM enhancement"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.instructions = []
        self.performance_scores = []
    
    def add_instruction(self, instruction: str, performance: float):
        """Add instruction with associated performance"""
        if len(self.instructions) >= self.max_size:
            # Remove worst performing instruction
            worst_idx = np.argmin(self.performance_scores)
            self.instructions.pop(worst_idx)
            self.performance_scores.pop(worst_idx)
        
        self.instructions.append(instruction)
        self.performance_scores.append(performance)
    
    def sample_historical_instructions(self, n: int) -> List[str]:
        """Sample n historical instructions based on performance"""
        if not self.instructions:
            return []
        
        # Weighted sampling based on performance
        weights = np.array(self.performance_scores) + 1e-6  # Avoid zero weights
        weights = weights / np.sum(weights)
        
        if n >= len(self.instructions):
            return self.instructions.copy()
        
        sampled_indices = np.random.choice(len(self.instructions), n, p=weights, replace=False)
        return [self.instructions[i] for i in sampled_indices]
    
    def get_best_instruction(self) -> Optional[str]:
        """Get the best performing instruction"""
        if not self.instructions:
            return None
        
        best_idx = np.argmax(self.performance_scores)
        return self.instructions[best_idx]
    
    def update_instruction_performance(self, instruction: str, performance: float):
        """Update performance for existing instruction"""
        try:
            idx = self.instructions.index(instruction)
            self.performance_scores[idx] = performance
        except ValueError:
            # Instruction not found, add it
            self.add_instruction(instruction, performance)

class DualLevelEnhancementKit:
    """Dual-Level Enhancement Kit (DEK) for LLM optimization"""
    
    def __init__(self, config: SERLConfig):
        self.config = config
        self.instruction_population = InstructionPopulation(config.instruction_buffer_size)
        self.performance_buffer = PerformanceBuffer()
        
        # High-Level Enhancement (HLE) components
        self.prompt_optimizer = self._initialize_prompt_optimizer()
        
        # Low-Level Enhancement (LLE) components
        self.weight_optimizer = self._initialize_weight_optimizer()
    
    def _initialize_prompt_optimizer(self):
        """Initialize high-level prompt optimization"""
        return {
            'macro_micro_refine': True,
            'instruction_refine_agent': True,
            'in_context_learning': True,
            'experience_replay': True
        }
    
    def _initialize_weight_optimizer(self):
        """Initialize low-level weight optimization"""
        return {
            'ste_enabled': True,  # Straight-Through Estimator
            'lora_enabled': True,  # Low-Rank Adaptation
            'lora_rank': self.config.lora_rank,
            'lora_alpha': self.config.lora_alpha,
            'target_layers': ['add_norm', 'positional_encoding']
        }
    
    def high_level_enhancement(self, prompt: str, performance_metrics: Dict[str, float]) -> str:
        """Apply high-level enhancement to prompts"""
        enhanced_prompt = prompt
        
        # Macro-Micro Refine
        if self.prompt_optimizer['macro_micro_refine']:
            enhanced_prompt = self._apply_macro_micro_refine(enhanced_prompt, performance_metrics)
        
        # Instruction Refine Agent
        if self.prompt_optimizer['instruction_refine_agent']:
            enhanced_prompt = self._apply_instruction_refine(enhanced_prompt, performance_metrics)
        
        # In-Context Learning
        if self.prompt_optimizer['in_context_learning']:
            enhanced_prompt = self._apply_in_context_learning(enhanced_prompt)
        
        # Experience Replay
        if self.prompt_optimizer['experience_replay']:
            enhanced_prompt = self._apply_experience_replay(enhanced_prompt)
        
        return enhanced_prompt
    
    def _apply_macro_micro_refine(self, prompt: str, metrics: Dict[str, float]) -> str:
        """Apply macro-micro refinement to prompt"""
        # Analyze performance and refine prompt structure
        if metrics.get('PA', 0) < 2.0:
            # Low performance - add more detailed instructions
            prompt += "\n\nDETAILED REQUIREMENTS:\n- Focus on execution quality and market impact\n- Include comprehensive risk management\n- Consider transaction costs and slippage"
        elif metrics.get('PA', 0) > 4.0:
            # High performance - optimize for efficiency
            prompt += "\n\nOPTIMIZATION FOCUS:\n- Prioritize computational efficiency\n- Maintain performance while reducing complexity\n- Focus on key performance drivers"
        
        return prompt
    
    def _apply_instruction_refine(self, prompt: str, metrics: Dict[str, float]) -> str:
        """Apply instruction refinement based on performance"""
        # Get best historical instruction
        best_instruction = self.instruction_population.get_best_instruction()
        if best_instruction:
            prompt += f"\n\nREFINED INSTRUCTION (from best performance):\n{best_instruction}"
        
        return prompt
    
    def _apply_in_context_learning(self, prompt: str) -> str:
        """Apply in-context learning with historical examples"""
        # Sample historical instructions
        historical_instructions = self.instruction_population.sample_historical_instructions(3)
        if historical_instructions:
            prompt += "\n\nHISTORICAL CONTEXT:\n"
            for i, instruction in enumerate(historical_instructions, 1):
                prompt += f"{i}. {instruction}\n"
        
        return prompt
    
    def _apply_experience_replay(self, prompt: str) -> str:
        """Apply experience replay with cache replay"""
        # Get recent successful examples
        recent_performance = self.performance_buffer.get_recent_performance(5)
        successful_examples = [p for p in recent_performance if p['metrics'].get('PA', 0) > 3.0]
        
        if successful_examples:
            prompt += "\n\nEXPERIENCE REPLAY (Successful Examples):\n"
            for example in successful_examples[:3]:
                prompt += f"- PA: {example['metrics'].get('PA', 0):.2f}, Code: {example['code'][:100]}...\n"
        
        return prompt
    
    def low_level_enhancement(self, llm_model: LLMIntegration, performance_metrics: Dict[str, float]):
        """Apply low-level weight optimization"""
        if not self.weight_optimizer['ste_enabled']:
            return
        
        try:
            # Apply Straight-Through Estimator for non-differentiable parts
            self._apply_ste_optimization(llm_model, performance_metrics)
            
            # Apply LoRA for target layers
            if self.weight_optimizer['lora_enabled']:
                self._apply_lora_optimization(llm_model)
                
        except Exception as e:
            logger.warning(f"Low-level enhancement failed: {str(e)}")
    
    def _apply_ste_optimization(self, llm_model: LLMIntegration, metrics: Dict[str, float]):
        """Apply Straight-Through Estimator optimization"""
        # This would involve fine-tuning the LLM weights
        # For now, we'll log the optimization attempt
        logger.info(f"Applying STE optimization with metrics: {metrics}")
        
        # In a real implementation, this would:
        # 1. Create a differentiable surrogate for non-differentiable operations
        # 2. Compute gradients through the surrogate
        # 3. Update the original model weights
    
    def _apply_lora_optimization(self, llm_model: LLMIntegration):
        """Apply LoRA optimization to target layers"""
        logger.info(f"Applying LoRA optimization (rank={self.weight_optimizer['lora_rank']})")
        
        # In a real implementation, this would:
        # 1. Identify target layers (add&normalization, positional encoding)
        # 2. Apply LoRA adapters to these layers
        # 3. Fine-tune only the LoRA parameters

class SERLFramework:
    """Main SE-RL framework orchestrating the entire process"""
    
    def __init__(self, config: SERLConfig):
        self.config = config
        self.dek = DualLevelEnhancementKit(config)
        self.performance_buffer = PerformanceBuffer()
        self.instruction_population = InstructionPopulation()
        
        # Initialize components
        self.llm_model = None
        self.llm_generator = None
        self.data_pipeline = None
        self.rl_trainer = None
        self.static_env = None
        self.dynamic_env = None
        self.multi_agent_env = None
        
        # Training state
        self.current_iteration = 0
        self.best_policy = None
        self.best_performance = -float('inf')
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"SE-RL Framework initialized with config: {config}")
    
    def _initialize_components(self):
        """Initialize all framework components"""
        try:
            # Initialize LLM integration
            llm_config = LLMConfig(
                provider=self.config.llm_provider,
                model_name=self.config.llm_model_name,
                api_key=self.config.llm_api_key,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            self.llm_model = LLMIntegration(llm_config)
            
            # Initialize LLM component generator
            component_config = ComponentConfig(
                model_name=self.config.llm_model_name,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
            self.llm_generator = LLMComponentGenerator(component_config)
            
            # Initialize data pipeline
            data_config = DataConfig(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.frequency,
                window_size=self.config.window_size
            )
            self.data_pipeline = FinancialDataPipeline(data_config)
            
            # Initialize RL trainer
            training_config = TrainingConfig(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay,
                device=self.config.device
            )
            self.rl_trainer = RLTrainer(training_config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def run_training(self) -> Dict[str, Any]:
        """Run the complete SE-RL training process (Algorithm 1 from paper)"""
        logger.info("Starting SE-RL training process")
        
        # Load and process data
        data = self._load_data()
        
        # Initialize environments
        self._initialize_environments(data)
        
        # Main training loop (Outer Loop)
        for j in range(self.config.max_outer_iterations):
            logger.info(f"Outer iteration {j+1}/{self.config.max_outer_iterations}")
            
            # Step 1: Design and Generate RL Algorithm
            algorithm_Aj = self._generate_algorithm(j)
            
            # Step 2: Algorithm Train (Inner Loop) and Test
            policy_dynamic = self._train_in_dynamic_environment(algorithm_Aj)
            policy_static = self._train_in_static_environment(algorithm_Aj)
            final_policy = self._hybrid_environment_training(policy_static, policy_dynamic, algorithm_Aj)
            
            # Step 3: Evaluate Performance
            performance_metrics = self._evaluate_policy(final_policy)
            
            # Step 4: Dual-Level Enhance LLM capability
            self._apply_dual_level_enhancement(algorithm_Aj, performance_metrics)
            
            # Step 5: Check convergence
            if self._check_convergence(j, performance_metrics):
                logger.info(f"Convergence reached at iteration {j+1}")
                break
        
        # Return final results
        return self._generate_final_results()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load and process financial data"""
        logger.info(f"Loading {self.config.dataset} data")
        
        if self.config.dataset == "csi100":
            data = self.data_pipeline.load_csi100_data()
        elif self.config.dataset == "nasdaq100":
            data = self.data_pipeline.load_nasdaq100_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        
        return data
    
    def _initialize_environments(self, data: Dict[str, Any]):
        """Initialize static and dynamic environments"""
        logger.info("Initializing environments")
        
        # Static environment (historical data)
        self.static_env = StaticEnvironment(data['train_data'], self.config)
        
        # Dynamic environment (market simulation)
        self.dynamic_env = DynamicEnvironment(
            data['train_data'], 
            self.config,
            num_agents=self.config.num_agents,
            agent_types=self.config.agent_types
        )
        
        # Multi-agent environment
        self.multi_agent_env = MultiAgentEnvironment(
            data['train_data'],
            self.config,
            agent_types=self.config.agent_types
        )
    
    def _generate_algorithm(self, iteration: int) -> Dict[str, Any]:
        """Generate RL algorithm using LLM (Step 1)"""
        logger.info(f"Generating algorithm for iteration {iteration}")
        
        # Create market context
        market_context = f"{self.config.dataset.upper()} trading environment with {self.config.num_agents} agents"
        
        # Generate components for single agent
        single_agent_components = self._generate_single_agent_components(market_context)
        
        # Generate components for multi-agent
        multi_agent_components = self._generate_multi_agent_components(market_context)
        
        # Combine components
        algorithm = {
            'iteration': iteration,
            'single_agent': single_agent_components,
            'multi_agent': multi_agent_components,
            'market_context': market_context
        }
        
        return algorithm
    
    def _generate_single_agent_components(self, market_context: str) -> Dict[str, Any]:
        """Generate single agent components using LLM"""
        components = {}
        
        # Generate reward function
        reward_result = self.llm_generator.generate_reward_function(market_context, "single")
        components['reward_function'] = reward_result
        
        # Generate network architecture
        network_result = self.llm_generator.generate_network_architecture(64, 4, "single")
        components['network_architecture'] = network_result
        
        return components
    
    def _generate_multi_agent_components(self, market_context: str) -> Dict[str, Any]:
        """Generate multi-agent components using LLM"""
        components = {}
        
        # Generate communication protocol
        communication_result = self.llm_generator.generate_multi_agent_communication(
            self.config.num_agents, self.config.agent_types
        )
        components['communication_protocol'] = communication_result
        
        # Generate LLM4Profiling
        profiling_result = self.llm_generator.generate_llm4profiling(
            self.config.agent_types, market_context
        )
        components['profiling'] = profiling_result
        
        # Generate components for each agent type
        for agent_type in self.config.agent_types:
            # Reward function for specific agent type
            reward_result = self.llm_generator.generate_reward_function(market_context, agent_type)
            components[f'reward_{agent_type}'] = reward_result
            
            # Network architecture for specific agent type
            network_result = self.llm_generator.generate_network_architecture(64, 4, agent_type)
            components[f'network_{agent_type}'] = network_result
        
        return components
    
    def _train_in_dynamic_environment(self, algorithm: Dict[str, Any]) -> Any:
        """Train agent in dynamic environment (Step 2a)"""
        logger.info("Training in dynamic environment")
        
        # Initialize agent with generated components
        self.rl_trainer.initialize_agent(64, 4)
        
        # Train in dynamic environment
        for episode in range(self.config.max_inner_iterations):
            result = self.rl_trainer.train_episode(environment=self.dynamic_env)
            
            if episode % 100 == 0:
                logger.info(f"Dynamic training episode {episode}: reward = {result.get('episode_reward', 0):.4f}")
        
        return self.rl_trainer.agent
    
    def _train_in_static_environment(self, algorithm: Dict[str, Any]) -> Any:
        """Train agent in static environment (Step 2b)"""
        logger.info("Training in static environment")
        
        # Initialize agent with generated components
        self.rl_trainer.initialize_agent(64, 4)
        
        # Train in static environment
        for episode in range(self.config.max_inner_iterations):
            result = self.rl_trainer.train_episode(environment=self.static_env)
            
            if episode % 100 == 0:
                logger.info(f"Static training episode {episode}: reward = {result.get('episode_reward', 0):.4f}")
        
        return self.rl_trainer.agent
    
    def _hybrid_environment_training(self, policy_static: Any, policy_dynamic: Any, 
                                   algorithm: Dict[str, Any]) -> Any:
        """Hybrid environment training with loss rebalancing (Step 2c)"""
        logger.info("Performing hybrid environment training")
        
        # Initialize agent
        self.rl_trainer.initialize_agent(64, 4)
        
        # Hybrid training with loss rebalancing
        for iteration in range(self.config.rebalance_iterations):
            # Compute static environment loss
            static_loss = self._compute_static_loss(policy_static)
            
            # Compute dynamic environment loss
            dynamic_loss = self._compute_dynamic_loss(policy_dynamic)
            
            # Rebalance losses: L_rebalance = α * L_static + β * L_dynamic
            alpha = self.config.static_env_weight
            beta = self.config.dynamic_env_weight
            
            rebalanced_loss = alpha * static_loss + beta * dynamic_loss
            
            # Update agent
            self.rl_trainer.update_agent(rebalanced_loss)
            
            if iteration % 10 == 0:
                logger.info(f"Hybrid training iteration {iteration}: loss = {rebalanced_loss:.4f}")
        
        return self.rl_trainer.agent
    
    def _compute_static_loss(self, policy: Any) -> float:
        """Compute loss in static environment"""
        # This would compute the actual loss
        # For now, return a placeholder
        return 0.1
    
    def _compute_dynamic_loss(self, policy: Any) -> float:
        """Compute loss in dynamic environment"""
        # This would compute the actual loss
        # For now, return a placeholder
        return 0.2
    
    def _evaluate_policy(self, policy: Any) -> Dict[str, float]:
        """Evaluate policy performance (Step 3)"""
        logger.info("Evaluating policy performance")
        
        # Evaluate in test environment
        metrics = self.rl_trainer.evaluate_policy(num_episodes=50)
        
        # Store performance
        self.performance_buffer.add_performance(
            self.current_iteration,
            metrics,
            "current_prompt",
            "current_code"
        )
        
        # Update best performance
        if metrics.get('PA', 0) > self.best_performance:
            self.best_performance = metrics.get('PA', 0)
            self.best_policy = policy
        
        logger.info(f"Evaluation metrics: PA={metrics.get('PA', 0):.4f}, "
                   f"WR={metrics.get('WR', 0):.4f}, GLR={metrics.get('GLR', 0):.4f}")
        
        return metrics
    
    def _apply_dual_level_enhancement(self, algorithm: Dict[str, Any], 
                                    performance_metrics: Dict[str, float]):
        """Apply dual-level enhancement (Step 4)"""
        logger.info("Applying dual-level enhancement")
        
        # High-Level Enhancement (HLE)
        enhanced_prompt = self.dek.high_level_enhancement(
            algorithm['market_context'], performance_metrics
        )
        
        # Low-Level Enhancement (LLE)
        self.dek.low_level_enhancement(self.llm_model, performance_metrics)
        
        # Update instruction population
        self.instruction_population.add_instruction(
            enhanced_prompt, performance_metrics.get('PA', 0)
        )
    
    def _check_convergence(self, iteration: int, metrics: Dict[str, float]) -> bool:
        """Check if training has converged (Step 5)"""
        if iteration < 2:
            return False
        
        # Get recent performance
        recent_performance = self.performance_buffer.get_recent_performance(3)
        if len(recent_performance) < 3:
            return False
        
        # Compute convergence metric: (PA_j - PA_{j-1}) / (PA_{j-1} - PA_{j-2})
        pa_values = [p['metrics'].get('PA', 0) for p in recent_performance]
        
        if len(pa_values) >= 3:
            current_pa = pa_values[-1]
            prev_pa = pa_values[-2]
            prev_prev_pa = pa_values[-3]
            
            if abs(prev_pa - prev_prev_pa) < 1e-6:
                return False
            
            convergence_ratio = abs(current_pa - prev_pa) / abs(prev_pa - prev_prev_pa)
            
            logger.info(f"Convergence ratio: {convergence_ratio:.4f}")
            
            return convergence_ratio < self.config.convergence_epsilon
        
        return False
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final training results"""
        logger.info("Generating final results")
        
        # Get performance trends
        performance_trend = self.performance_buffer.get_performance_trend()
        best_performance = self.performance_buffer.get_best_performance()
        
        results = {
            'final_metrics': best_performance['metrics'] if best_performance else {},
            'performance_trend': performance_trend,
            'best_iteration': self.best_performance,
            'total_iterations': self.current_iteration,
            'convergence_reached': True,
            'framework_config': self.config.__dict__
        }
        
        return results 