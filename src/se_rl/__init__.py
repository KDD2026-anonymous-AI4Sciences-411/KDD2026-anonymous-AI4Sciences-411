"""
SE-RL Framework: Self-Evolutional Reinforcement Learning
=======================================================

A comprehensive framework for LLM-powered reinforcement learning in financial trading.

This framework implements:
- Bi-level optimization (outer loop: LLM evolution, inner loop: agent training)
- LLM-powered RL algorithm design (LLM4Reward, LLM4Agent, LLM4Imagine, etc.)
- Dual-Level Enhancement Kit (DEK) with STE and LoRA
- Hybrid environment training with adaptive loss rebalancing
- Complete Limit Order Book (LOB) simulation

Author: AI Research Engineer
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI Research Engineer"

# Core framework
from .core.framework import SERLFramework, SERLConfig, PerformanceBuffer, InstructionPopulation, DualLevelEnhancementKit
from .core.training_loop import (
    SERLTrainingLoop,
    TrainingLoopConfig,
    EMASmoothing,
    AdaptiveLossRebalancer,
    ConvergenceChecker,
    FinancialMetricsCalculator
)

# Environments
from .environments.limit_order_book import LimitOrderBook, Order, OrderSide, OrderType, Trade, LOBSnapshot
from .environments.execution_env import ExecutionEnvironment, ExecutionEnvConfig, HybridExecutionEnvironment
from .environments.static_env import StaticEnvironment, StaticEnvironmentConfig
from .environments.dynamic_env import DynamicEnvironment, DynamicEnvironmentConfig

# RL components
from .rl.ppo_agent import PPOAgent, PPOConfig, PPOTrainer, ActorNetwork, CriticNetwork, RolloutBuffer
from .rl.trainer import RLTrainer, TrainingConfig

# LLM components
from .llm.imagination import LLM4Imagine, ImaginationConfig
from .llm.low_level_enhancement import LowLevelEnhancement, LoRAConfig, STELayer, GumbelSoftmaxSTE
from .llm.code_validator import CodeValidator, ValidatorConfig, ValidationResult

# Data pipeline
from .data.pipeline import FinancialDataPipeline, DataConfig

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Core Framework
    'SERLFramework',
    'SERLConfig',
    'PerformanceBuffer',
    'InstructionPopulation',
    'DualLevelEnhancementKit',

    # Training Loop
    'SERLTrainingLoop',
    'TrainingLoopConfig',
    'EMASmoothing',
    'AdaptiveLossRebalancer',
    'ConvergenceChecker',
    'FinancialMetricsCalculator',

    # Limit Order Book
    'LimitOrderBook',
    'Order',
    'OrderSide',
    'OrderType',
    'Trade',
    'LOBSnapshot',

    # Execution Environment
    'ExecutionEnvironment',
    'ExecutionEnvConfig',
    'HybridExecutionEnvironment',

    # Static Environment
    'StaticEnvironment',
    'StaticEnvironmentConfig',

    # Dynamic Environment
    'DynamicEnvironment',
    'DynamicEnvironmentConfig',

    # PPO Agent
    'PPOAgent',
    'PPOConfig',
    'PPOTrainer',
    'ActorNetwork',
    'CriticNetwork',
    'RolloutBuffer',

    # RL Trainer
    'RLTrainer',
    'TrainingConfig',

    # LLM4Imagine
    'LLM4Imagine',
    'ImaginationConfig',

    # Low-Level Enhancement
    'LowLevelEnhancement',
    'LoRAConfig',
    'STELayer',
    'GumbelSoftmaxSTE',

    # Code Validator
    'CodeValidator',
    'ValidatorConfig',
    'ValidationResult',

    # Data Pipeline
    'FinancialDataPipeline',
    'DataConfig',
]
