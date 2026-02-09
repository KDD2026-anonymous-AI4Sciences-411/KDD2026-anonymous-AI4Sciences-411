"""
SE-RL LLM Module
================

This module provides LLM integration for automated RL algorithm design:
- LLM Component Generator (LLM4Reward, LLM4Agent, etc.)
- LLM4Imagine (50% mixed sampling for training data)
- Low-Level Enhancement (STE and LoRA)
- Code Validator (two-stage validation)
- Prompt templates
"""

from .imagination import (
    LLM4Imagine,
    ImaginationConfig,
    MarketScenarioGenerator,
    ImaginaryDataGenerator
)

from .low_level_enhancement import (
    LowLevelEnhancement,
    LoRAConfig,
    LoRALinear,
    LoRALayerNorm,
    LoRAPositionalEncoding,
    StraightThroughEstimator,
    STELayer,
    GumbelSoftmaxSTE,
    CodeGenerationSTE
)

from .code_validator import (
    CodeValidator,
    ValidatorConfig,
    ValidationResult,
    SyntaxValidator,
    SecurityValidator,
    RuntimeValidator,
    SemanticValidator
)

__all__ = [
    # Imagination
    'LLM4Imagine',
    'ImaginationConfig',
    'MarketScenarioGenerator',
    'ImaginaryDataGenerator',
    # Low-Level Enhancement
    'LowLevelEnhancement',
    'LoRAConfig',
    'LoRALinear',
    'LoRALayerNorm',
    'LoRAPositionalEncoding',
    'StraightThroughEstimator',
    'STELayer',
    'GumbelSoftmaxSTE',
    'CodeGenerationSTE',
    # Code Validator
    'CodeValidator',
    'ValidatorConfig',
    'ValidationResult',
    'SyntaxValidator',
    'SecurityValidator',
    'RuntimeValidator',
    'SemanticValidator',
]
