"""
SE-RL Core Module
=================

This module provides the core SE-RL framework components:
- Main SE-RL Framework (bi-level optimization)
- Training Loop (with EMA convergence)
- Dual-Level Enhancement Kit (DEK)
- Adaptive Loss Rebalancing
- Financial Metrics Calculator
"""

from .framework import (
    SERLFramework,
    SERLConfig,
    PerformanceBuffer,
    InstructionPopulation,
    DualLevelEnhancementKit
)

from .training_loop import (
    SERLTrainingLoop,
    TrainingLoopConfig,
    EMASmoothing,
    AdaptiveLossRebalancer,
    ConvergenceChecker,
    FinancialMetricsCalculator
)

__all__ = [
    # Framework
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
]
