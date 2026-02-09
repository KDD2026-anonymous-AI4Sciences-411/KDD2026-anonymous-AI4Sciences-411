"""
SE-RL Reinforcement Learning Module
===================================

This module provides RL algorithms and training utilities:
- PPO Agent with GAE (Generalized Advantage Estimation)
- Actor-Critic networks
- Rollout buffer with advantage computation
- Training and evaluation utilities
"""

from .ppo_agent import (
    PPOAgent,
    PPOConfig,
    PPOTrainer,
    ActorNetwork,
    CriticNetwork,
    RolloutBuffer
)

from .trainer import (
    RLTrainer,
    TrainingConfig,
    FinancialMetrics,
    RLAgent
)

__all__ = [
    # PPO
    'PPOAgent',
    'PPOConfig',
    'PPOTrainer',
    'ActorNetwork',
    'CriticNetwork',
    'RolloutBuffer',
    # Trainer
    'RLTrainer',
    'TrainingConfig',
    'FinancialMetrics',
    'RLAgent',
]
