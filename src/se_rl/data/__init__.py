"""
SE-RL Data Module
=================

This module provides data processing utilities:
- Financial data pipeline
- Feature engineering
- Data normalization
"""

from .pipeline import (
    FinancialDataPipeline,
    DataConfig
)

__all__ = [
    'FinancialDataPipeline',
    'DataConfig',
]
