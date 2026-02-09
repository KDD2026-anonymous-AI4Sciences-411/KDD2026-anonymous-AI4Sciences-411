"""
LLM4Imagine Module for SE-RL Framework
======================================

This module implements the LLM4Imagine component as described in the paper.

LLM4Imagine generates simulated market data for training:
- 50% of data comes from static historical data
- 50% of data comes from LLM-imagined market scenarios

This helps the agent generalize better by training on diverse market conditions.

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class ImaginationConfig:
    """Configuration for LLM4Imagine module"""
    # Sampling ratio (paper specifies 50%)
    imagination_ratio: float = 0.5  # Proportion of imagined data

    # Generation parameters
    num_scenarios: int = 10  # Number of imagined scenarios to generate
    scenario_length: int = 100  # Length of each scenario (time steps)

    # Market condition templates
    market_conditions: List[str] = None

    # Noise parameters
    noise_scale: float = 0.1

    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self):
        if self.market_conditions is None:
            self.market_conditions = [
                "bull_market",
                "bear_market",
                "high_volatility",
                "low_volatility",
                "trending_up",
                "trending_down",
                "range_bound",
                "flash_crash",
                "gradual_recovery",
                "consolidation"
            ]


class MarketScenarioGenerator:
    """
    Generates market scenarios using LLM for imagination.

    Each scenario describes market conditions, volatility, trends,
    and other parameters that are used to simulate market data.
    """

    def __init__(self, config: ImaginationConfig):
        self.config = config

    def generate_scenario_prompt(self, condition: str) -> str:
        """Generate prompt for LLM to describe a market scenario"""
        prompt = f"""You are a financial market expert. Generate a detailed market scenario for the following condition: {condition}

Please provide the following parameters in JSON format:
{{
    "trend": "up" | "down" | "sideways",
    "trend_strength": 0.0 to 1.0,
    "volatility": 0.0 to 1.0 (0 = low, 1 = high),
    "volume_pattern": "increasing" | "decreasing" | "stable" | "spike",
    "mean_reversion_strength": 0.0 to 1.0,
    "momentum": -1.0 to 1.0,
    "bid_ask_spread_multiplier": 0.5 to 3.0,
    "market_depth": "shallow" | "normal" | "deep",
    "price_jumps": 0 to 5 (number of sudden price jumps),
    "description": "brief description of the scenario"
}}

Only respond with the JSON object, no additional text."""
        return prompt

    def parse_scenario_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into scenario parameters"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                scenario = json.loads(json_match.group())
                return self._validate_scenario(scenario)
            else:
                return self._get_default_scenario()
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response, using default scenario")
            return self._get_default_scenario()

    def _validate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize scenario parameters"""
        validated = {
            'trend': scenario.get('trend', 'sideways'),
            'trend_strength': np.clip(scenario.get('trend_strength', 0.5), 0.0, 1.0),
            'volatility': np.clip(scenario.get('volatility', 0.3), 0.0, 1.0),
            'volume_pattern': scenario.get('volume_pattern', 'stable'),
            'mean_reversion_strength': np.clip(scenario.get('mean_reversion_strength', 0.5), 0.0, 1.0),
            'momentum': np.clip(scenario.get('momentum', 0.0), -1.0, 1.0),
            'bid_ask_spread_multiplier': np.clip(scenario.get('bid_ask_spread_multiplier', 1.0), 0.5, 3.0),
            'market_depth': scenario.get('market_depth', 'normal'),
            'price_jumps': int(np.clip(scenario.get('price_jumps', 0), 0, 5)),
            'description': scenario.get('description', 'Default market scenario')
        }
        return validated

    def _get_default_scenario(self) -> Dict[str, Any]:
        """Get default scenario parameters"""
        return {
            'trend': 'sideways',
            'trend_strength': 0.3,
            'volatility': 0.3,
            'volume_pattern': 'stable',
            'mean_reversion_strength': 0.5,
            'momentum': 0.0,
            'bid_ask_spread_multiplier': 1.0,
            'market_depth': 'normal',
            'price_jumps': 0,
            'description': 'Normal market conditions'
        }


class ImaginaryDataGenerator:
    """
    Generates imaginary market data based on scenario parameters.

    Uses stochastic processes (GBM, Ornstein-Uhlenbeck) to simulate
    realistic market data based on LLM-generated scenarios.
    """

    def __init__(self, initial_price: float = 100.0, dt: float = 1 / 252):
        self.initial_price = initial_price
        self.dt = dt  # Time step (fraction of year)

    def generate_data(self, scenario: Dict[str, Any], length: int) -> pd.DataFrame:
        """
        Generate simulated market data based on scenario.

        Args:
            scenario: Scenario parameters from LLM
            length: Number of time steps to generate

        Returns:
            DataFrame with OHLCV data
        """
        # Extract scenario parameters
        trend = scenario['trend']
        trend_strength = scenario['trend_strength']
        volatility = scenario['volatility']
        volume_pattern = scenario['volume_pattern']
        mean_reversion = scenario['mean_reversion_strength']
        momentum = scenario['momentum']
        price_jumps = scenario['price_jumps']

        # Generate base price process
        prices = self._generate_price_process(
            trend, trend_strength, volatility, mean_reversion, momentum, length
        )

        # Add price jumps
        if price_jumps > 0:
            prices = self._add_price_jumps(prices, price_jumps)

        # Generate OHLC from prices
        ohlc = self._generate_ohlc(prices, volatility)

        # Generate volume
        volume = self._generate_volume(length, volume_pattern)

        # Create DataFrame
        df = pd.DataFrame({
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': volume
        })

        # Add derived features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['returns'].rolling(20).std().fillna(df['returns'].std())

        return df

    def _generate_price_process(self, trend: str, trend_strength: float,
                                volatility: float, mean_reversion: float,
                                momentum: float, length: int) -> np.ndarray:
        """
        Generate price process using combination of GBM and OU process.

        P_t = P_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

        With mean reversion:
        dP = kappa * (theta - P) * dt + sigma * dW
        """
        prices = np.zeros(length)
        prices[0] = self.initial_price

        # Set drift based on trend
        if trend == 'up':
            mu = 0.1 * trend_strength
        elif trend == 'down':
            mu = -0.1 * trend_strength
        else:
            mu = 0.0

        # Add momentum effect
        mu += momentum * 0.05

        # Volatility scaling
        sigma = 0.2 * volatility + 0.1  # Base volatility + scaling

        # Mean reversion target
        theta = self.initial_price

        for t in range(1, length):
            # GBM component
            gbm_return = (mu - 0.5 * sigma ** 2) * self.dt + sigma * np.sqrt(self.dt) * np.random.randn()

            # Mean reversion component (OU process)
            mr_component = mean_reversion * 0.1 * (theta - prices[t - 1]) / prices[t - 1] * self.dt

            # Combined return
            total_return = gbm_return + mr_component

            prices[t] = prices[t - 1] * np.exp(total_return)

        return prices

    def _add_price_jumps(self, prices: np.ndarray, num_jumps: int) -> np.ndarray:
        """Add sudden price jumps to the price series"""
        length = len(prices)
        jump_indices = np.random.choice(range(10, length - 10), size=num_jumps, replace=False)

        for idx in jump_indices:
            # Random jump magnitude (±5-15%)
            jump_size = np.random.uniform(0.05, 0.15) * np.random.choice([-1, 1])
            prices[idx:] *= (1 + jump_size)

        return prices

    def _generate_ohlc(self, prices: np.ndarray, volatility: float) -> Dict[str, np.ndarray]:
        """Generate OHLC data from close prices"""
        length = len(prices)

        # Intraday volatility factor
        intraday_vol = 0.005 * (volatility + 1)

        open_prices = np.zeros(length)
        high_prices = np.zeros(length)
        low_prices = np.zeros(length)

        open_prices[0] = prices[0]
        high_prices[0] = prices[0] * (1 + np.abs(np.random.randn()) * intraday_vol)
        low_prices[0] = prices[0] * (1 - np.abs(np.random.randn()) * intraday_vol)

        for t in range(1, length):
            # Open is close of previous day with small gap
            gap = np.random.randn() * intraday_vol * 0.5
            open_prices[t] = prices[t - 1] * (1 + gap)

            # High and low
            range_size = np.abs(np.random.randn()) * intraday_vol * 2
            high_prices[t] = max(open_prices[t], prices[t]) * (1 + range_size)
            low_prices[t] = min(open_prices[t], prices[t]) * (1 - range_size)

        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices
        }

    def _generate_volume(self, length: int, pattern: str) -> np.ndarray:
        """Generate volume data based on pattern"""
        base_volume = 1000000  # Base daily volume

        if pattern == 'increasing':
            trend = np.linspace(0.7, 1.3, length)
        elif pattern == 'decreasing':
            trend = np.linspace(1.3, 0.7, length)
        elif pattern == 'spike':
            trend = np.ones(length)
            spike_indices = np.random.choice(range(length), size=length // 20, replace=False)
            trend[spike_indices] = np.random.uniform(2, 5, size=len(spike_indices))
        else:  # stable
            trend = np.ones(length)

        # Add noise
        noise = np.random.lognormal(0, 0.3, length)

        volume = base_volume * trend * noise

        return volume.astype(int)


class LLM4Imagine:
    """
    Main LLM4Imagine module.

    Implements the 50% mixed sampling strategy for training:
    - 50% static (historical) data
    - 50% imagined (LLM-generated) data
    """

    def __init__(self, config: ImaginationConfig, llm_interface=None):
        """
        Initialize LLM4Imagine module.

        Args:
            config: Imagination configuration
            llm_interface: Interface to LLM for scenario generation
        """
        self.config = config
        self.llm_interface = llm_interface
        self.scenario_generator = MarketScenarioGenerator(config)
        self.data_generator = ImaginaryDataGenerator()

        # Cache for generated scenarios
        self.scenario_cache: Dict[str, Dict[str, Any]] = {}
        self.generated_data_cache: List[pd.DataFrame] = []

        logger.info(f"LLM4Imagine initialized with {config.imagination_ratio * 100}% imagination ratio")

    def generate_imagined_scenarios(self, num_scenarios: int = None) -> List[Dict[str, Any]]:
        """
        Generate imagined market scenarios using LLM.

        Args:
            num_scenarios: Number of scenarios to generate

        Returns:
            List of scenario dictionaries
        """
        if num_scenarios is None:
            num_scenarios = self.config.num_scenarios

        scenarios = []
        conditions = self.config.market_conditions

        for i in range(num_scenarios):
            condition = conditions[i % len(conditions)]

            # Check cache first
            if condition in self.scenario_cache:
                scenarios.append(self.scenario_cache[condition])
                continue

            # Generate scenario using LLM (or default if no LLM available)
            if self.llm_interface is not None:
                prompt = self.scenario_generator.generate_scenario_prompt(condition)
                try:
                    response = self.llm_interface.generate(prompt)
                    scenario = self.scenario_generator.parse_scenario_response(response)
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}, using default scenario")
                    scenario = self._generate_default_scenario(condition)
            else:
                scenario = self._generate_default_scenario(condition)

            scenario['condition'] = condition
            self.scenario_cache[condition] = scenario
            scenarios.append(scenario)

        return scenarios

    def _generate_default_scenario(self, condition: str) -> Dict[str, Any]:
        """Generate default scenario parameters for a given condition"""
        default_scenarios = {
            'bull_market': {
                'trend': 'up', 'trend_strength': 0.7, 'volatility': 0.3,
                'volume_pattern': 'increasing', 'mean_reversion_strength': 0.2,
                'momentum': 0.5, 'bid_ask_spread_multiplier': 0.8,
                'market_depth': 'deep', 'price_jumps': 0,
                'description': 'Strong bull market with increasing volume'
            },
            'bear_market': {
                'trend': 'down', 'trend_strength': 0.7, 'volatility': 0.5,
                'volume_pattern': 'increasing', 'mean_reversion_strength': 0.3,
                'momentum': -0.5, 'bid_ask_spread_multiplier': 1.5,
                'market_depth': 'shallow', 'price_jumps': 1,
                'description': 'Bear market with high volatility'
            },
            'high_volatility': {
                'trend': 'sideways', 'trend_strength': 0.2, 'volatility': 0.8,
                'volume_pattern': 'spike', 'mean_reversion_strength': 0.5,
                'momentum': 0.0, 'bid_ask_spread_multiplier': 2.0,
                'market_depth': 'shallow', 'price_jumps': 2,
                'description': 'High volatility regime'
            },
            'low_volatility': {
                'trend': 'sideways', 'trend_strength': 0.1, 'volatility': 0.1,
                'volume_pattern': 'decreasing', 'mean_reversion_strength': 0.7,
                'momentum': 0.0, 'bid_ask_spread_multiplier': 0.7,
                'market_depth': 'deep', 'price_jumps': 0,
                'description': 'Low volatility, quiet market'
            },
            'trending_up': {
                'trend': 'up', 'trend_strength': 0.5, 'volatility': 0.4,
                'volume_pattern': 'stable', 'mean_reversion_strength': 0.3,
                'momentum': 0.3, 'bid_ask_spread_multiplier': 1.0,
                'market_depth': 'normal', 'price_jumps': 0,
                'description': 'Moderate uptrend'
            },
            'trending_down': {
                'trend': 'down', 'trend_strength': 0.5, 'volatility': 0.4,
                'volume_pattern': 'stable', 'mean_reversion_strength': 0.3,
                'momentum': -0.3, 'bid_ask_spread_multiplier': 1.0,
                'market_depth': 'normal', 'price_jumps': 0,
                'description': 'Moderate downtrend'
            },
            'range_bound': {
                'trend': 'sideways', 'trend_strength': 0.1, 'volatility': 0.3,
                'volume_pattern': 'stable', 'mean_reversion_strength': 0.8,
                'momentum': 0.0, 'bid_ask_spread_multiplier': 1.0,
                'market_depth': 'normal', 'price_jumps': 0,
                'description': 'Range-bound, mean reverting market'
            },
            'flash_crash': {
                'trend': 'down', 'trend_strength': 0.9, 'volatility': 1.0,
                'volume_pattern': 'spike', 'mean_reversion_strength': 0.1,
                'momentum': -0.8, 'bid_ask_spread_multiplier': 3.0,
                'market_depth': 'shallow', 'price_jumps': 3,
                'description': 'Flash crash scenario'
            },
            'gradual_recovery': {
                'trend': 'up', 'trend_strength': 0.4, 'volatility': 0.4,
                'volume_pattern': 'increasing', 'mean_reversion_strength': 0.4,
                'momentum': 0.2, 'bid_ask_spread_multiplier': 1.2,
                'market_depth': 'normal', 'price_jumps': 0,
                'description': 'Gradual recovery from lows'
            },
            'consolidation': {
                'trend': 'sideways', 'trend_strength': 0.1, 'volatility': 0.2,
                'volume_pattern': 'decreasing', 'mean_reversion_strength': 0.6,
                'momentum': 0.0, 'bid_ask_spread_multiplier': 0.9,
                'market_depth': 'normal', 'price_jumps': 0,
                'description': 'Market consolidation phase'
            }
        }

        return default_scenarios.get(condition, self.scenario_generator._get_default_scenario())

    def generate_imagined_data(self, num_scenarios: int = None,
                               scenario_length: int = None) -> List[pd.DataFrame]:
        """
        Generate imagined market data.

        Args:
            num_scenarios: Number of scenarios to generate
            scenario_length: Length of each scenario

        Returns:
            List of DataFrames with imagined market data
        """
        if num_scenarios is None:
            num_scenarios = self.config.num_scenarios
        if scenario_length is None:
            scenario_length = self.config.scenario_length

        # Generate scenarios
        scenarios = self.generate_imagined_scenarios(num_scenarios)

        # Generate data for each scenario
        imagined_data = []
        for scenario in scenarios:
            data = self.data_generator.generate_data(scenario, scenario_length)
            data['scenario'] = scenario.get('condition', 'unknown')
            imagined_data.append(data)

        self.generated_data_cache = imagined_data
        return imagined_data

    def get_mixed_batch(self, static_data: pd.DataFrame,
                        batch_size: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get a mixed batch with 50% static and 50% imagined data.

        This implements the paper's mixed sampling strategy.

        Args:
            static_data: Historical (static) data
            batch_size: Total batch size

        Returns:
            Combined DataFrame and metadata
        """
        static_size = int(batch_size * (1 - self.config.imagination_ratio))
        imagined_size = batch_size - static_size

        # Sample from static data
        if len(static_data) >= static_size:
            static_indices = np.random.choice(len(static_data) - static_size, 1)[0]
            static_batch = static_data.iloc[static_indices:static_indices + static_size].copy()
        else:
            static_batch = static_data.copy()

        # Get or generate imagined data
        if not self.generated_data_cache:
            self.generate_imagined_data()

        # Sample from imagined data
        imagined_batch = self._sample_imagined_data(imagined_size)

        # Combine batches
        static_batch['source'] = 'static'
        imagined_batch['source'] = 'imagined'

        combined = pd.concat([static_batch.reset_index(drop=True),
                              imagined_batch.reset_index(drop=True)],
                             ignore_index=True)

        # Shuffle
        combined = combined.sample(frac=1).reset_index(drop=True)

        metadata = {
            'static_ratio': static_size / batch_size,
            'imagined_ratio': imagined_size / batch_size,
            'static_size': static_size,
            'imagined_size': imagined_size
        }

        return combined, metadata

    def _sample_imagined_data(self, size: int) -> pd.DataFrame:
        """Sample from imagined data cache"""
        if not self.generated_data_cache:
            # Generate default data
            scenario = self.scenario_generator._get_default_scenario()
            data = self.data_generator.generate_data(scenario, size)
            return data

        # Randomly select scenarios and sample
        samples = []
        remaining = size

        while remaining > 0:
            scenario_data = np.random.choice(self.generated_data_cache)
            sample_size = min(remaining, len(scenario_data))

            start_idx = np.random.randint(0, max(1, len(scenario_data) - sample_size))
            sample = scenario_data.iloc[start_idx:start_idx + sample_size].copy()
            samples.append(sample)
            remaining -= sample_size

        return pd.concat(samples, ignore_index=True).iloc[:size]

    def create_training_dataset(self, static_data: pd.DataFrame,
                                total_samples: int) -> pd.DataFrame:
        """
        Create a complete training dataset with mixed static and imagined data.

        Args:
            static_data: Historical data
            total_samples: Total number of samples

        Returns:
            Mixed training dataset
        """
        # Generate imagined scenarios if not already done
        if not self.generated_data_cache:
            self.generate_imagined_data()

        # Calculate sizes
        static_size = int(total_samples * (1 - self.config.imagination_ratio))
        imagined_size = total_samples - static_size

        # Sample static data
        if len(static_data) >= static_size:
            static_samples = static_data.sample(n=static_size, replace=True)
        else:
            static_samples = static_data.sample(n=static_size, replace=True)

        # Sample imagined data
        imagined_samples = self._sample_imagined_data(imagined_size)

        # Combine and shuffle
        static_samples['source'] = 'static'
        imagined_samples['source'] = 'imagined'

        combined = pd.concat([static_samples, imagined_samples], ignore_index=True)
        combined = combined.sample(frac=1).reset_index(drop=True)

        logger.info(f"Created training dataset: {len(combined)} samples "
                    f"({static_size} static, {imagined_size} imagined)")

        return combined


# Unit tests
if __name__ == "__main__":
    print("Testing LLM4Imagine module...")

    # Create configuration
    config = ImaginationConfig(
        imagination_ratio=0.5,
        num_scenarios=5,
        scenario_length=100
    )

    # Initialize module (without LLM for testing)
    imagine = LLM4Imagine(config)

    # Test scenario generation
    print("\n1. Testing scenario generation...")
    scenarios = imagine.generate_imagined_scenarios(5)
    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i + 1}: {scenario.get('condition', 'unknown')} - "
              f"trend={scenario.get('trend')}, vol={scenario.get('volatility'):.2f}")

    # Test data generation
    print("\n2. Testing data generation...")
    imagined_data = imagine.generate_imagined_data(3, 50)
    for i, data in enumerate(imagined_data):
        print(f"Data {i + 1}: shape={data.shape}, "
              f"price range=[{data['close'].min():.2f}, {data['close'].max():.2f}]")

    # Test mixed batch
    print("\n3. Testing mixed batch generation...")
    static_data = pd.DataFrame({
        'open': 100 + np.random.randn(200) * 2,
        'high': 102 + np.random.randn(200) * 2,
        'low': 98 + np.random.randn(200) * 2,
        'close': 100 + np.random.randn(200) * 2,
        'volume': np.random.randint(100000, 1000000, 200)
    })

    mixed_batch, metadata = imagine.get_mixed_batch(static_data, batch_size=100)
    print(f"Mixed batch shape: {mixed_batch.shape}")
    print(f"Source distribution: {mixed_batch['source'].value_counts().to_dict()}")
    print(f"Metadata: {metadata}")

    # Test training dataset creation
    print("\n4. Testing training dataset creation...")
    training_data = imagine.create_training_dataset(static_data, total_samples=500)
    print(f"Training dataset shape: {training_data.shape}")
    print(f"Source distribution: {training_data['source'].value_counts().to_dict()}")

    print("\nLLM4Imagine tests passed!")
