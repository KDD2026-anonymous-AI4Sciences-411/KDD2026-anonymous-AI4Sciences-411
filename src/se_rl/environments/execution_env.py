"""
Order Execution Environment for SE-RL Framework
===============================================

This module implements the complete MDP-based environment for order execution
as described in Section A.5 of the paper.

MDP Definition:
- State: s = (LOB_data, private_state) where private_state = (normalized_remaining_time, normalized_inventory)
- Action: a ∈ [0, 1] - continuous action representing execution ratio
- Transition: Determined by LOB dynamics and agent interactions

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from enum import Enum

from .limit_order_book import LimitOrderBook, Order, OrderSide, OrderType, Trade

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEnvConfig:
    """Configuration for order execution environment"""
    # Initial conditions
    initial_inventory: float = 10000.0  # Shares to sell
    initial_price: float = 100.0
    total_time: int = 240  # Total execution horizon (e.g., 240 minutes = 4 hours)

    # Transaction costs
    transaction_cost: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps

    # Market impact parameters
    permanent_impact: float = 0.1  # Permanent market impact coefficient
    temporary_impact: float = 0.1  # Temporary market impact coefficient

    # LOB parameters
    tick_size: float = 0.01
    lob_depth_levels: int = 5

    # Reward parameters
    vwap_benchmark: bool = True  # Use VWAP as benchmark
    inventory_penalty: float = 0.1  # Penalty for remaining inventory

    # State space configuration
    state_dim: int = 32  # Total state dimension
    action_dim: int = 1  # Continuous action [0, 1]

    # Normalization
    max_inventory: float = 10000.0
    price_scale: float = 100.0


class ExecutionEnvironment:
    """
    Complete MDP environment for order execution.

    Implements the environment dynamics described in Section A.5:
    - State: s = (LOB features, normalized remaining time, normalized inventory)
    - Action: Continuous ratio a ∈ [0, 1] determining execution amount
    - Reward: Based on price advantage vs VWAP benchmark
    """

    def __init__(self, data: pd.DataFrame, config: ExecutionEnvConfig):
        """
        Initialize execution environment.

        Args:
            data: Historical market data with OHLCV columns
            config: Environment configuration
        """
        self.data = data.copy()
        self.config = config

        # Initialize LOB
        self.lob = LimitOrderBook(
            initial_price=config.initial_price,
            tick_size=config.tick_size
        )

        # Environment state
        self.current_step = 0
        self.remaining_inventory = config.initial_inventory
        self.initial_inventory = config.initial_inventory
        self.total_time = config.total_time

        # Execution tracking
        self.execution_prices = []
        self.execution_quantities = []
        self.execution_times = []
        self.trades_history = []

        # Performance tracking
        self.episode_reward = 0.0
        self.cumulative_cost = 0.0

        # Preprocess data
        self._preprocess_data()

        logger.info(f"Execution environment initialized with {len(self.data)} time steps")

    def _preprocess_data(self):
        """Preprocess market data and compute features"""
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute additional features
        self.data['returns'] = self.data['close'].pct_change().fillna(0)
        self.data['volatility'] = self.data['returns'].rolling(20).std().fillna(0.01)
        self.data['vwap'] = (
                (self.data['close'] * self.data['volume']).cumsum() /
                self.data['volume'].cumsum()
        )

        # Compute volume profile
        self.data['volume_ma'] = self.data['volume'].rolling(20).mean().fillna(self.data['volume'].mean())
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']

        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(0)

        # Compute TWAP schedule for reference
        self._compute_twap_schedule()

    def _compute_twap_schedule(self):
        """Compute TWAP (Time-Weighted Average Price) execution schedule"""
        total_steps = min(len(self.data), self.total_time)
        self.twap_schedule = np.ones(total_steps) * (self.initial_inventory / total_steps)

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.remaining_inventory = self.initial_inventory

        # Reset LOB
        self.lob.reset()
        self._initialize_lob()

        # Reset tracking
        self.execution_prices = []
        self.execution_quantities = []
        self.execution_times = []
        self.trades_history = []
        self.episode_reward = 0.0
        self.cumulative_cost = 0.0

        return self._get_state()

    def _initialize_lob(self):
        """Initialize LOB with initial orders based on market data"""
        current_price = self.data.iloc[0]['close']

        # Add some initial liquidity
        for i in range(5):
            # Bid side
            bid_price = current_price * (1 - 0.001 * (i + 1))
            self.lob.submit_order(
                agent_id=-1,  # Market maker
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=np.random.uniform(100, 500),
                timestamp=0.0
            )

            # Ask side
            ask_price = current_price * (1 + 0.001 * (i + 1))
            self.lob.submit_order(
                agent_id=-1,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=np.random.uniform(100, 500),
                timestamp=0.0
            )

    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.

        State components as per paper:
        1. LOB features (bid/ask prices, depths, imbalance, etc.)
        2. Private state:
           - Normalized remaining time: (T - t) / T
           - Normalized remaining inventory: I_t / I_0
        """
        # LOB state vector
        lob_state = self.lob.get_state_vector(self.config.lob_depth_levels)

        # Private state
        remaining_time_ratio = (self.total_time - self.current_step) / self.total_time
        inventory_ratio = self.remaining_inventory / self.initial_inventory

        private_state = np.array([
            remaining_time_ratio,
            inventory_ratio
        ], dtype=np.float32)

        # Market features
        if self.current_step < len(self.data):
            market_data = self.data.iloc[self.current_step]
            market_features = np.array([
                market_data['returns'],
                market_data['volatility'],
                market_data['volume_ratio'],
                market_data['close'] / self.config.price_scale,
            ], dtype=np.float32)
        else:
            market_features = np.zeros(4, dtype=np.float32)

        # Combine all state components
        state = np.concatenate([lob_state, private_state, market_features])

        # Pad or truncate to match state_dim
        if len(state) < self.config.state_dim:
            state = np.pad(state, (0, self.config.state_dim - len(state)))
        else:
            state = state[:self.config.state_dim]

        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Continuous action in [0, 1] representing execution ratio

        Returns:
            next_state: Next state observation
            reward: Step reward
            done: Episode termination flag
            info: Additional information
        """
        # Clip action to valid range
        action_ratio = np.clip(action, 0.0, 1.0)
        if isinstance(action_ratio, np.ndarray):
            action_ratio = action_ratio.item() if action_ratio.size == 1 else action_ratio[0]

        # Calculate execution quantity
        execution_qty = self.remaining_inventory * action_ratio

        # Execute the order
        execution_price, execution_cost, trades = self._execute_order(execution_qty)

        # Update state
        executed_qty = sum(t.quantity for t in trades) if trades else 0
        self.remaining_inventory -= executed_qty

        if executed_qty > 0:
            self.execution_prices.append(execution_price)
            self.execution_quantities.append(executed_qty)
            self.execution_times.append(self.current_step)
            self.trades_history.extend(trades)

        # Calculate reward
        reward = self._calculate_reward(execution_price, executed_qty, execution_cost)
        self.episode_reward += reward
        self.cumulative_cost += execution_cost

        # Update LOB with market dynamics
        self._update_lob_dynamics()

        # Move to next step
        self.current_step += 1

        # Check termination
        done = (
                self.current_step >= min(len(self.data) - 1, self.total_time) or
                self.remaining_inventory <= 0
        )

        # Apply terminal penalty for remaining inventory
        if done and self.remaining_inventory > 0:
            terminal_penalty = self._calculate_terminal_penalty()
            reward += terminal_penalty

        # Get next state
        next_state = self._get_state()

        # Info dictionary
        info = {
            'step': self.current_step,
            'action_ratio': action_ratio,
            'execution_qty': executed_qty,
            'execution_price': execution_price,
            'execution_cost': execution_cost,
            'remaining_inventory': self.remaining_inventory,
            'episode_reward': self.episode_reward,
            'lob_mid_price': self.lob.get_mid_price(),
            'lob_spread': self.lob.get_spread() or 0.0
        }

        return next_state, reward, done, info

    def _execute_order(self, quantity: float) -> Tuple[float, float, List[Trade]]:
        """
        Execute a sell order through the LOB.

        Args:
            quantity: Quantity to sell

        Returns:
            avg_execution_price: Volume-weighted average execution price
            total_cost: Total execution cost (transaction costs + market impact)
            trades: List of executed trades
        """
        if quantity <= 0:
            return self.lob.get_mid_price(), 0.0, []

        timestamp = float(self.current_step)

        # Submit market order
        order, trades = self.lob.submit_order(
            agent_id=0,  # Execution agent
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=0,  # Market order
            quantity=quantity,
            timestamp=timestamp
        )

        if not trades:
            # No execution, try limit order at best bid
            best_bid = self.lob.get_best_bid() or self.lob.get_mid_price()
            order, trades = self.lob.submit_order(
                agent_id=0,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=best_bid,
                quantity=quantity,
                timestamp=timestamp
            )

        if not trades:
            return self.lob.get_mid_price(), 0.0, []

        # Calculate average execution price
        total_value = sum(t.price * t.quantity for t in trades)
        total_qty = sum(t.quantity for t in trades)
        avg_price = total_value / total_qty if total_qty > 0 else self.lob.get_mid_price()

        # Calculate costs
        transaction_cost = total_value * self.config.transaction_cost
        slippage_cost = total_value * self.config.slippage

        # Market impact (temporary + permanent)
        impact_cost = (
                self.config.temporary_impact * (total_qty / self.initial_inventory) * avg_price * total_qty +
                self.config.permanent_impact * (total_qty / self.initial_inventory) ** 2 * avg_price * total_qty
        )

        total_cost = transaction_cost + slippage_cost + impact_cost

        return avg_price, total_cost, trades

    def _calculate_reward(self, execution_price: float, quantity: float,
                          cost: float) -> float:
        """
        Calculate step reward based on execution quality.

        Reward is based on price advantage vs VWAP benchmark:
        r = (P_vwap - P_execution) * quantity - cost

        Args:
            execution_price: Average execution price
            quantity: Executed quantity
            cost: Execution costs

        Returns:
            Step reward
        """
        if quantity <= 0:
            # Small penalty for not executing when should
            return -0.001 * self.remaining_inventory / self.initial_inventory

        # Get VWAP benchmark
        if self.current_step < len(self.data):
            vwap = self.data.iloc[self.current_step]['vwap']
        else:
            vwap = self.lob.get_mid_price()

        # Price advantage (selling above VWAP is good)
        # Note: For sell orders, higher execution price is better
        price_advantage = (execution_price - vwap) / vwap * 10000  # In basis points

        # Reward = price advantage * quantity - cost
        reward = price_advantage * (quantity / self.initial_inventory) - cost / (self.initial_inventory * self.config.initial_price)

        return reward

    def _calculate_terminal_penalty(self) -> float:
        """Calculate penalty for remaining inventory at episode end"""
        inventory_ratio = self.remaining_inventory / self.initial_inventory
        penalty = -self.config.inventory_penalty * inventory_ratio ** 2
        return penalty

    def _update_lob_dynamics(self):
        """Update LOB with market dynamics (other traders, market makers, etc.)"""
        if self.current_step >= len(self.data) - 1:
            return

        current_data = self.data.iloc[self.current_step]
        next_data = self.data.iloc[self.current_step + 1]

        # Update mid price based on market data
        new_mid_price = next_data['close']

        # Add new orders to simulate market activity
        timestamp = float(self.current_step)

        # Clear some old orders and add new ones
        # Simulate market makers updating quotes
        for i in range(3):
            # New bid
            bid_price = new_mid_price * (1 - 0.001 * (i + 1) + np.random.uniform(-0.0005, 0.0005))
            self.lob.submit_order(
                agent_id=-1,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=np.random.uniform(50, 200),
                timestamp=timestamp
            )

            # New ask
            ask_price = new_mid_price * (1 + 0.001 * (i + 1) + np.random.uniform(-0.0005, 0.0005))
            self.lob.submit_order(
                agent_id=-1,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=np.random.uniform(50, 200),
                timestamp=timestamp
            )

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate execution performance metrics.

        Returns:
            Dictionary of metrics including PA, WR, GLR, AFI
        """
        if not self.execution_prices:
            return {
                'PA': 0.0,
                'PA_std': 0.0,
                'WR': 0.0,
                'GLR': 0.0,
                'AFI': self.remaining_inventory / self.initial_inventory,
                'VWAP': 0.0,
                'avg_execution_price': 0.0,
                'total_executed': 0.0,
                'num_trades': 0
            }

        # Calculate VWAP of executions
        total_value = sum(p * q for p, q in zip(self.execution_prices, self.execution_quantities))
        total_qty = sum(self.execution_quantities)
        avg_execution_price = total_value / total_qty if total_qty > 0 else 0

        # Get market VWAP for comparison
        if self.execution_times:
            market_vwap = self.data.iloc[self.execution_times[-1]]['vwap']
        else:
            market_vwap = self.data['vwap'].iloc[-1]

        # Price Advantage (PA) in basis points
        pa = (avg_execution_price - market_vwap) / market_vwap * 10000

        # PA standard deviation (for sell orders)
        if len(self.execution_prices) > 1:
            pa_values = [(p - market_vwap) / market_vwap * 10000 for p in self.execution_prices]
            pa_std = np.std(pa_values)
        else:
            pa_std = 0.0

        # Calculate returns for WR and GLR
        returns = []
        for i in range(1, len(self.execution_prices)):
            ret = (self.execution_prices[i] - self.execution_prices[i - 1]) / self.execution_prices[i - 1]
            returns.append(ret)

        # Win Ratio
        wins = sum(1 for r in returns if r > 0)
        wr = wins / len(returns) if returns else 0.0

        # Gain-Loss Ratio
        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        if gains and losses:
            glr = np.mean(gains) / np.mean(losses)
        else:
            glr = 0.0

        # Average Final Inventory
        afi = self.remaining_inventory / self.initial_inventory

        return {
            'PA': pa,
            'PA_std': pa_std,
            'WR': wr,
            'GLR': glr,
            'AFI': afi,
            'VWAP': market_vwap,
            'avg_execution_price': avg_execution_price,
            'total_executed': total_qty,
            'num_trades': len(self.trades_history),
            'episode_reward': self.episode_reward,
            'cumulative_cost': self.cumulative_cost
        }

    def render(self, mode: str = 'human'):
        """Render current environment state"""
        snapshot = self.lob.get_snapshot()

        print(f"\n{'=' * 60}")
        print(f"Step: {self.current_step}/{self.total_time}")
        print(f"Remaining Inventory: {self.remaining_inventory:.2f} ({self.remaining_inventory / self.initial_inventory * 100:.1f}%)")
        print(f"{'=' * 60}")
        print(f"LOB Mid Price: ${snapshot.mid_price:.2f}")
        print(f"Best Bid: ${snapshot.best_bid:.2f} | Best Ask: ${snapshot.best_ask:.2f}")
        print(f"Spread: ${snapshot.spread:.4f} ({snapshot.spread / snapshot.mid_price * 10000:.2f} bps)")
        print(f"Imbalance: {snapshot.imbalance:.4f}")
        print(f"{'=' * 60}")
        print(f"Executions: {len(self.execution_prices)}")
        if self.execution_prices:
            print(f"Avg Execution Price: ${np.mean(self.execution_prices):.2f}")
        print(f"Episode Reward: {self.episode_reward:.4f}")
        print(f"{'=' * 60}")

    @property
    def observation_space_dim(self) -> int:
        """Get observation space dimension"""
        return self.config.state_dim

    @property
    def action_space_dim(self) -> int:
        """Get action space dimension"""
        return self.config.action_dim


class HybridExecutionEnvironment:
    """
    Hybrid environment combining static (historical) and dynamic (simulated) components.

    Implements the hybrid training approach from the paper with adaptive loss rebalancing.
    """

    def __init__(self, static_env: ExecutionEnvironment,
                 dynamic_env: 'DynamicExecutionEnvironment',
                 alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize hybrid environment.

        Args:
            static_env: Static environment (historical data)
            dynamic_env: Dynamic environment (multi-agent simulation)
            alpha: Weight for static environment loss
            beta: Weight for dynamic environment loss
        """
        self.static_env = static_env
        self.dynamic_env = dynamic_env
        self.alpha = alpha
        self.beta = beta

        # Current environment
        self.current_env = static_env
        self.is_static = True

        # Metrics history for adaptive rebalancing
        self.static_metrics_history = []
        self.dynamic_metrics_history = []

    def reset(self, use_static: bool = None) -> np.ndarray:
        """Reset environment"""
        if use_static is None:
            # 50% probability of each environment (as per paper)
            self.is_static = np.random.random() < 0.5
        else:
            self.is_static = use_static

        self.current_env = self.static_env if self.is_static else self.dynamic_env
        return self.current_env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in current environment"""
        next_state, reward, done, info = self.current_env.step(action)
        info['is_static'] = self.is_static
        return next_state, reward, done, info

    def adapt_weights(self, static_loss: float, dynamic_loss: float):
        """
        Adaptive loss rebalancing as per Equation (5).

        L_rebalance = α * L_static + β * L_dynamic

        Adjusts α and β based on relative performance.
        """
        self.static_metrics_history.append(static_loss)
        self.dynamic_metrics_history.append(dynamic_loss)

        if len(self.static_metrics_history) < 10:
            return

        # Calculate average losses
        avg_static = np.mean(self.static_metrics_history[-10:])
        avg_dynamic = np.mean(self.dynamic_metrics_history[-10:])

        # Rebalance: Give more weight to environment with higher loss
        total_loss = avg_static + avg_dynamic + 1e-8
        self.alpha = avg_static / total_loss
        self.beta = avg_dynamic / total_loss

        # Normalize to sum to 1
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total

        logger.debug(f"Adapted weights: alpha={self.alpha:.4f}, beta={self.beta:.4f}")

    def get_rebalanced_loss(self, static_loss: float, dynamic_loss: float) -> float:
        """Calculate rebalanced loss"""
        return self.alpha * static_loss + self.beta * dynamic_loss

    def get_metrics(self) -> Dict[str, float]:
        """Get current environment metrics"""
        return self.current_env.get_metrics()


# Unit tests
if __name__ == "__main__":
    print("Testing Execution Environment...")

    # Create dummy data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(500) * 0.1)
    volume = np.random.uniform(1000, 10000, 500)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(500) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(500)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(500)) * 0.002),
        'close': prices,
        'volume': volume
    }, index=dates)

    config = ExecutionEnvConfig(
        initial_inventory=1000.0,
        initial_price=100.0,
        total_time=100
    )

    env = ExecutionEnvironment(data, config)

    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    # Test steps
    total_reward = 0
    for i in range(50):
        action = np.array([np.random.uniform(0.05, 0.2)])
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    print(f"Total reward: {total_reward:.4f}")
    print(f"Remaining inventory: {env.remaining_inventory:.2f}")

    # Get metrics
    metrics = env.get_metrics()
    print(f"Metrics: {metrics}")

    env.render()

    print("Execution Environment tests passed!")
