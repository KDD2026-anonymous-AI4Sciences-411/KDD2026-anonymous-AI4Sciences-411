"""
SE-RL Environments Module
========================

This module provides various environments for training RL agents:
- Static environment (historical data replay)
- Dynamic environment (multi-agent market simulation)
- Execution environment (MDP-based order execution)
- Limit Order Book (complete LOB with price-time priority)
"""

from .limit_order_book import (
    LimitOrderBook,
    Order,
    OrderSide,
    OrderType,
    Trade,
    LOBSnapshot,
    PriceLevel
)

from .execution_env import (
    ExecutionEnvironment,
    ExecutionEnvConfig,
    HybridExecutionEnvironment
)

from .static_env import (
    StaticEnvironment,
    StaticEnvironmentConfig
)

from .dynamic_env import (
    DynamicEnvironment,
    DynamicEnvironmentConfig,
    MarketAgent,
    MarketMakerAgent,
    InformedTraderAgent,
    NoiseTraderAgent
)

__all__ = [
    # LOB
    'LimitOrderBook',
    'Order',
    'OrderSide',
    'OrderType',
    'Trade',
    'LOBSnapshot',
    'PriceLevel',
    # Execution
    'ExecutionEnvironment',
    'ExecutionEnvConfig',
    'HybridExecutionEnvironment',
    # Static
    'StaticEnvironment',
    'StaticEnvironmentConfig',
    # Dynamic
    'DynamicEnvironment',
    'DynamicEnvironmentConfig',
    'MarketAgent',
    'MarketMakerAgent',
    'InformedTraderAgent',
    'NoiseTraderAgent',
]
