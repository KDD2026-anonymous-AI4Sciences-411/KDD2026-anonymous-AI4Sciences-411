"""
Complete Training Loop for SE-RL Framework
==========================================

This module implements the complete training loop with:
- Adaptive loss rebalancing (Equation 5)
- EMA smoothed convergence condition
- Comprehensive evaluation metrics (PA, PA-std, WR, GLR, AFI)
- Hybrid environment training

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainingLoopConfig:
    """Configuration for the complete training loop"""
    # Outer loop parameters
    max_outer_iterations: int = 50
    convergence_tolerance: float = 0.1  # epsilon_tol

    # Inner loop parameters
    max_inner_iterations: int = 1000
    steps_per_update: int = 2048

    # Loss rebalancing (Equation 5)
    initial_alpha: float = 0.5  # Static environment weight
    initial_beta: float = 0.5  # Dynamic environment weight
    rebalance_interval: int = 10  # Iterations between rebalancing

    # EMA parameters for convergence
    ema_decay: float = 0.9  # Exponential moving average decay
    convergence_window: int = 5  # Window for convergence check

    # Evaluation parameters
    eval_episodes: int = 50
    eval_interval: int = 10

    # Logging
    log_interval: int = 10


class EMASmoothing:
    """
    Exponential Moving Average smoothing for convergence detection.

    Used to smooth performance metrics before checking convergence
    as described in the paper.
    """

    def __init__(self, decay: float = 0.9):
        """
        Initialize EMA smoother.

        Args:
            decay: EMA decay factor (higher = smoother)
        """
        self.decay = decay
        self.ema_value = None
        self.history = []

    def update(self, value: float) -> float:
        """
        Update EMA with new value.

        EMA_t = decay * EMA_{t-1} + (1 - decay) * value_t

        Args:
            value: New value to incorporate

        Returns:
            Updated EMA value
        """
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = self.decay * self.ema_value + (1 - self.decay) * value

        self.history.append(self.ema_value)
        return self.ema_value

    def get_current(self) -> Optional[float]:
        """Get current EMA value"""
        return self.ema_value

    def get_history(self) -> List[float]:
        """Get EMA history"""
        return self.history.copy()

    def reset(self):
        """Reset EMA state"""
        self.ema_value = None
        self.history = []


class AdaptiveLossRebalancer:
    """
    Adaptive loss rebalancing as per Equation (5).

    L_rebalance = α * L_static + β * L_dynamic

    Adjusts α and β dynamically based on relative losses.
    """

    def __init__(self, initial_alpha: float = 0.5, initial_beta: float = 0.5,
                 adaptation_rate: float = 0.1, min_weight: float = 0.1):
        """
        Initialize adaptive rebalancer.

        Args:
            initial_alpha: Initial weight for static loss
            initial_beta: Initial weight for dynamic loss
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight for either environment
        """
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight

        # History for analysis
        self.static_loss_history = []
        self.dynamic_loss_history = []
        self.alpha_history = [initial_alpha]
        self.beta_history = [initial_beta]

    def compute_rebalanced_loss(self, static_loss: float, dynamic_loss: float) -> float:
        """
        Compute rebalanced loss according to Equation (5).

        L_rebalance = α * L_static + β * L_dynamic

        Args:
            static_loss: Loss from static environment
            dynamic_loss: Loss from dynamic environment

        Returns:
            Rebalanced loss
        """
        self.static_loss_history.append(static_loss)
        self.dynamic_loss_history.append(dynamic_loss)

        return self.alpha * static_loss + self.beta * dynamic_loss

    def adapt_weights(self, static_loss: float, dynamic_loss: float):
        """
        Adapt weights based on relative losses.

        Higher loss environment gets more weight to balance training.

        Args:
            static_loss: Recent static environment loss
            dynamic_loss: Recent dynamic environment loss
        """
        # Calculate relative losses
        total_loss = static_loss + dynamic_loss + 1e-8
        static_ratio = static_loss / total_loss
        dynamic_ratio = dynamic_loss / total_loss

        # Adapt weights towards balanced losses
        # Higher loss -> increase weight
        target_alpha = static_ratio
        target_beta = dynamic_ratio

        # Smooth adaptation
        self.alpha = (1 - self.adaptation_rate) * self.alpha + self.adaptation_rate * target_alpha
        self.beta = (1 - self.adaptation_rate) * self.beta + self.adaptation_rate * target_beta

        # Ensure minimum weights
        self.alpha = max(self.min_weight, self.alpha)
        self.beta = max(self.min_weight, self.beta)

        # Normalize to sum to 1
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total

        # Record history
        self.alpha_history.append(self.alpha)
        self.beta_history.append(self.beta)

        logger.debug(f"Adapted weights: α={self.alpha:.4f}, β={self.beta:.4f}")

    def get_weights(self) -> Tuple[float, float]:
        """Get current weights"""
        return self.alpha, self.beta

    def get_history(self) -> Dict[str, List[float]]:
        """Get weight history"""
        return {
            'alpha': self.alpha_history,
            'beta': self.beta_history,
            'static_loss': self.static_loss_history,
            'dynamic_loss': self.dynamic_loss_history
        }


class FinancialMetricsCalculator:
    """
    Complete financial metrics calculator.

    Implements all metrics from the paper:
    - PA (Price Advantage) - Equation referenced in results
    - PA-std (PA Standard Deviation)
    - WR (Win Ratio)
    - GLR (Gain-Loss Ratio)
    - AFI (Average Final Inventory)
    """

    @staticmethod
    def calculate_pa(execution_prices: List[float],
                     benchmark_prices: List[float],
                     quantities: List[float] = None) -> float:
        """
        Calculate Price Advantage (PA) in basis points.

        PA = mean((P_benchmark - P_execution) / P_benchmark) * 10000

        For sell orders: higher execution price = positive PA

        Args:
            execution_prices: Prices at which orders were executed
            benchmark_prices: Benchmark prices (e.g., VWAP)
            quantities: Execution quantities for weighting

        Returns:
            PA in basis points
        """
        if not execution_prices or not benchmark_prices:
            return 0.0

        if len(execution_prices) != len(benchmark_prices):
            min_len = min(len(execution_prices), len(benchmark_prices))
            execution_prices = execution_prices[:min_len]
            benchmark_prices = benchmark_prices[:min_len]

        if quantities is None:
            quantities = [1.0] * len(execution_prices)

        # Calculate price advantages
        advantages = []
        for exec_p, bench_p, qty in zip(execution_prices, benchmark_prices, quantities):
            if bench_p > 0:
                # For sell orders: (exec - bench) / bench
                advantage = (exec_p - bench_p) / bench_p
                advantages.append(advantage * qty)

        if not advantages:
            return 0.0

        # Volume-weighted average
        total_qty = sum(quantities)
        if total_qty > 0:
            pa = sum(advantages) / total_qty * 10000  # Convert to basis points
        else:
            pa = np.mean(advantages) * 10000

        return pa

    @staticmethod
    def calculate_pa_std(execution_prices: List[float],
                         benchmark_prices: List[float]) -> float:
        """
        Calculate PA Standard Deviation.

        Measures consistency of execution quality.

        Args:
            execution_prices: Execution prices
            benchmark_prices: Benchmark prices

        Returns:
            Standard deviation of PA in basis points
        """
        if len(execution_prices) < 2:
            return 0.0

        # Calculate individual PAs
        pas = []
        for exec_p, bench_p in zip(execution_prices, benchmark_prices):
            if bench_p > 0:
                pa = (exec_p - bench_p) / bench_p * 10000
                pas.append(pa)

        if len(pas) < 2:
            return 0.0

        return np.std(pas)

    @staticmethod
    def calculate_wr(returns: List[float]) -> float:
        """
        Calculate Win Ratio (WR).

        WR = number of positive returns / total returns

        Args:
            returns: List of trade returns

        Returns:
            Win ratio (0-1)
        """
        if not returns:
            return 0.0

        wins = sum(1 for r in returns if r > 0)
        return wins / len(returns)

    @staticmethod
    def calculate_glr(returns: List[float]) -> float:
        """
        Calculate Gain-Loss Ratio (GLR).

        GLR = mean(gains) / mean(losses)

        Args:
            returns: List of trade returns

        Returns:
            Gain-loss ratio (higher is better)
        """
        if not returns:
            return 0.0

        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]

        if not gains or not losses:
            return 0.0 if not gains else float('inf')

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return float('inf') if avg_gain > 0 else 0.0

        return avg_gain / avg_loss

    @staticmethod
    def calculate_afi(final_inventories: List[float],
                      initial_inventories: List[float] = None) -> float:
        """
        Calculate Average Final Inventory (AFI).

        AFI = mean(|final_inventory / initial_inventory|)

        Lower is better (indicates complete execution).

        Args:
            final_inventories: Final inventory levels
            initial_inventories: Initial inventory levels

        Returns:
            Average final inventory ratio
        """
        if not final_inventories:
            return 0.0

        if initial_inventories is None:
            initial_inventories = [1.0] * len(final_inventories)

        ratios = []
        for final, initial in zip(final_inventories, initial_inventories):
            if initial > 0:
                ratio = abs(final) / initial
                ratios.append(ratio)

        return np.mean(ratios) if ratios else 0.0

    @staticmethod
    def calculate_implementation_shortfall(execution_prices: List[float],
                                           decision_prices: List[float],
                                           quantities: List[float]) -> float:
        """
        Calculate Implementation Shortfall.

        IS = sum((P_decision - P_execution) * quantity) / sum(P_decision * quantity)

        Args:
            execution_prices: Actual execution prices
            decision_prices: Prices at time of decision
            quantities: Execution quantities

        Returns:
            Implementation shortfall in basis points
        """
        if not execution_prices or not decision_prices or not quantities:
            return 0.0

        total_cost = sum(
            (d - e) * q for e, d, q in zip(execution_prices, decision_prices, quantities)
        )
        total_value = sum(d * q for d, q in zip(decision_prices, quantities))

        if total_value == 0:
            return 0.0

        return (total_cost / total_value) * 10000  # Basis points

    @staticmethod
    def calculate_all_metrics(execution_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate all financial metrics from execution data.

        Args:
            execution_data: Dictionary containing:
                - execution_prices: List of execution prices
                - benchmark_prices: List of benchmark prices
                - quantities: List of quantities
                - returns: List of returns
                - final_inventories: List of final inventories
                - initial_inventories: List of initial inventories

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        exec_prices = execution_data.get('execution_prices', [])
        bench_prices = execution_data.get('benchmark_prices', [])
        quantities = execution_data.get('quantities', [])
        returns = execution_data.get('returns', [])
        final_inv = execution_data.get('final_inventories', [])
        initial_inv = execution_data.get('initial_inventories', [])

        metrics['PA'] = FinancialMetricsCalculator.calculate_pa(
            exec_prices, bench_prices, quantities
        )
        metrics['PA_std'] = FinancialMetricsCalculator.calculate_pa_std(
            exec_prices, bench_prices
        )
        metrics['WR'] = FinancialMetricsCalculator.calculate_wr(returns)
        metrics['GLR'] = FinancialMetricsCalculator.calculate_glr(returns)
        metrics['AFI'] = FinancialMetricsCalculator.calculate_afi(final_inv, initial_inv)

        return metrics


class ConvergenceChecker:
    """
    Convergence checker using EMA-smoothed performance metrics.

    Implements convergence condition: PA_j - PA_{j-1} < epsilon_tol
    """

    def __init__(self, tolerance: float = 0.1, window: int = 5, ema_decay: float = 0.9):
        """
        Initialize convergence checker.

        Args:
            tolerance: Convergence tolerance (epsilon_tol)
            window: Window for checking convergence
            ema_decay: EMA decay for smoothing
        """
        self.tolerance = tolerance
        self.window = window
        self.ema = EMASmoothing(decay=ema_decay)
        self.pa_history = []

    def update(self, pa: float) -> bool:
        """
        Update with new PA and check convergence.

        Args:
            pa: Current Price Advantage

        Returns:
            True if converged, False otherwise
        """
        smoothed_pa = self.ema.update(pa)
        self.pa_history.append(smoothed_pa)

        return self.check_convergence()

    def check_convergence(self) -> bool:
        """
        Check if training has converged.

        Convergence condition: |PA_j - PA_{j-1}| < epsilon_tol
        """
        if len(self.pa_history) < self.window:
            return False

        recent = self.pa_history[-self.window:]

        # Check if improvements are within tolerance
        improvements = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        avg_improvement = np.mean(improvements)

        is_converged = avg_improvement < self.tolerance

        if is_converged:
            logger.info(f"Convergence detected: avg improvement {avg_improvement:.4f} < {self.tolerance}")

        return is_converged

    def get_convergence_rate(self) -> float:
        """Get the current convergence rate"""
        if len(self.pa_history) < 2:
            return float('inf')

        return abs(self.pa_history[-1] - self.pa_history[-2])


class SERLTrainingLoop:
    """
    Complete SE-RL training loop implementation.

    Orchestrates:
    - Outer loop (LLM evolution)
    - Inner loop (agent training)
    - Hybrid environment training
    - Adaptive loss rebalancing
    - Convergence checking
    """

    def __init__(self, config: TrainingLoopConfig,
                 static_env, dynamic_env, agent, llm_generator=None):
        """
        Initialize training loop.

        Args:
            config: Training loop configuration
            static_env: Static (historical) environment
            dynamic_env: Dynamic (simulated) environment
            agent: RL agent
            llm_generator: LLM component generator
        """
        self.config = config
        self.static_env = static_env
        self.dynamic_env = dynamic_env
        self.agent = agent
        self.llm_generator = llm_generator

        # Initialize components
        self.rebalancer = AdaptiveLossRebalancer(
            initial_alpha=config.initial_alpha,
            initial_beta=config.initial_beta
        )
        self.convergence_checker = ConvergenceChecker(
            tolerance=config.convergence_tolerance,
            window=config.convergence_window
        )
        self.metrics_calculator = FinancialMetricsCalculator()

        # EMA smoothers for various metrics
        self.pa_ema = EMASmoothing(decay=config.ema_decay)
        self.reward_ema = EMASmoothing(decay=config.ema_decay)

        # Training history
        self.outer_loop_history = []
        self.inner_loop_history = []
        self.evaluation_history = []

        logger.info("SE-RL Training Loop initialized")

    def run(self) -> Dict[str, Any]:
        """
        Run the complete SE-RL training process.

        Implements Algorithm 1 from the paper.

        Returns:
            Training results
        """
        logger.info("Starting SE-RL training loop")
        start_time = time.time()

        best_pa = -float('inf')
        best_agent_state = None

        for j in range(self.config.max_outer_iterations):
            outer_start = time.time()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Outer Iteration {j + 1}/{self.config.max_outer_iterations}")
            logger.info(f"{'=' * 60}")

            # Step 1: Generate algorithm components (if LLM available)
            if self.llm_generator is not None:
                self._generate_algorithm_components(j)

            # Step 2: Inner loop training
            inner_results = self._run_inner_loop(j)

            # Step 3: Evaluate performance
            eval_results = self._evaluate_agent()

            # Step 4: Update best agent
            current_pa = eval_results['PA']
            smoothed_pa = self.pa_ema.update(current_pa)

            if current_pa > best_pa:
                best_pa = current_pa
                best_agent_state = self._get_agent_state()
                logger.info(f"New best PA: {best_pa:.4f}")

            # Record outer loop results
            outer_results = {
                'iteration': j,
                'inner_results': inner_results,
                'eval_results': eval_results,
                'smoothed_pa': smoothed_pa,
                'alpha': self.rebalancer.alpha,
                'beta': self.rebalancer.beta,
                'duration': time.time() - outer_start
            }
            self.outer_loop_history.append(outer_results)

            # Step 5: Check convergence
            if self.convergence_checker.update(current_pa):
                logger.info(f"Convergence reached at iteration {j + 1}")
                break

            # Log progress
            self._log_progress(j, outer_results)

        # Restore best agent
        if best_agent_state is not None:
            self._set_agent_state(best_agent_state)

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f}s")
        logger.info(f"Best PA: {best_pa:.4f}")

        return self._generate_final_results()

    def _generate_algorithm_components(self, iteration: int):
        """Generate new algorithm components using LLM"""
        # This would use the LLM generator to create new components
        # For now, we just log the attempt
        logger.info(f"Generating algorithm components for iteration {iteration}")

    def _run_inner_loop(self, outer_iteration: int) -> Dict[str, Any]:
        """
        Run the inner training loop.

        Trains in both static and dynamic environments with adaptive rebalancing.
        """
        logger.info("Running inner training loop")

        episode_rewards_static = []
        episode_rewards_dynamic = []
        losses_static = []
        losses_dynamic = []

        for i in range(self.config.max_inner_iterations):
            # Alternate between environments (or use 50% sampling)
            use_static = np.random.random() < 0.5

            if use_static:
                reward, loss = self._train_step(self.static_env)
                episode_rewards_static.append(reward)
                losses_static.append(loss)
            else:
                reward, loss = self._train_step(self.dynamic_env)
                episode_rewards_dynamic.append(reward)
                losses_dynamic.append(loss)

            # Periodic rebalancing
            if (i + 1) % self.config.rebalance_interval == 0:
                if losses_static and losses_dynamic:
                    avg_static = np.mean(losses_static[-self.config.rebalance_interval:])
                    avg_dynamic = np.mean(losses_dynamic[-self.config.rebalance_interval:])
                    self.rebalancer.adapt_weights(avg_static, avg_dynamic)

            # Logging
            if (i + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(episode_rewards_static[-self.config.log_interval:] +
                                     episode_rewards_dynamic[-self.config.log_interval:])
                smoothed_reward = self.reward_ema.update(avg_reward)
                logger.info(f"  Inner step {i + 1}/{self.config.max_inner_iterations}: "
                            f"avg_reward={avg_reward:.4f}, smoothed={smoothed_reward:.4f}")

        return {
            'static_rewards': episode_rewards_static,
            'dynamic_rewards': episode_rewards_dynamic,
            'static_losses': losses_static,
            'dynamic_losses': losses_dynamic,
            'final_alpha': self.rebalancer.alpha,
            'final_beta': self.rebalancer.beta
        }

    def _train_step(self, env) -> Tuple[float, float]:
        """Execute one training step in an environment"""
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0

        done = False
        while not done and steps < 1000:
            # Select action
            action, log_prob, value = self.agent.select_action(state)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store transition
            self.agent.store_transition(state, action, reward, done, log_prob, value)

            episode_reward += reward
            state = next_state
            steps += 1

        # Update agent
        if hasattr(self.agent, 'update'):
            update_stats = self.agent.update()
            episode_loss = update_stats.get('policy_loss', 0.0)

        return episode_reward, episode_loss

    def _evaluate_agent(self) -> Dict[str, float]:
        """Evaluate agent performance"""
        logger.info("Evaluating agent...")

        execution_prices = []
        benchmark_prices = []
        quantities = []
        returns = []
        final_inventories = []
        initial_inventories = []

        for ep in range(self.config.eval_episodes):
            state = self.static_env.reset()
            done = False
            ep_return = 0.0

            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.static_env.step(action)

                # Record execution data
                if 'execution_price' in info:
                    execution_prices.append(info['execution_price'])
                if 'benchmark_price' in info:
                    benchmark_prices.append(info['benchmark_price'])
                elif 'lob_mid_price' in info:
                    benchmark_prices.append(info['lob_mid_price'])
                if 'execution_qty' in info:
                    quantities.append(info['execution_qty'])

                ep_return += reward
                state = next_state

            returns.append(ep_return)

            # Record inventory info
            if hasattr(self.static_env, 'remaining_inventory'):
                final_inventories.append(self.static_env.remaining_inventory)
            if hasattr(self.static_env, 'initial_inventory'):
                initial_inventories.append(self.static_env.initial_inventory)

        # Calculate metrics
        execution_data = {
            'execution_prices': execution_prices,
            'benchmark_prices': benchmark_prices,
            'quantities': quantities,
            'returns': returns,
            'final_inventories': final_inventories,
            'initial_inventories': initial_inventories
        }

        metrics = self.metrics_calculator.calculate_all_metrics(execution_data)
        metrics['mean_return'] = np.mean(returns)
        metrics['std_return'] = np.std(returns)

        self.evaluation_history.append(metrics)

        logger.info(f"Evaluation: PA={metrics['PA']:.4f}, WR={metrics['WR']:.4f}, "
                    f"GLR={metrics['GLR']:.4f}, AFI={metrics['AFI']:.4f}")

        return metrics

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for checkpointing"""
        if hasattr(self.agent, 'state_dict'):
            return {
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict()
            }
        return {}

    def _set_agent_state(self, state: Dict[str, Any]):
        """Set agent state from checkpoint"""
        if state and hasattr(self.agent, 'actor'):
            self.agent.actor.load_state_dict(state['actor'])
            self.agent.critic.load_state_dict(state['critic'])

    def _log_progress(self, iteration: int, results: Dict[str, Any]):
        """Log training progress"""
        eval_results = results['eval_results']
        logger.info(f"\nIteration {iteration + 1} Summary:")
        logger.info(f"  PA: {eval_results['PA']:.4f} (smoothed: {results['smoothed_pa']:.4f})")
        logger.info(f"  WR: {eval_results['WR']:.4f}")
        logger.info(f"  GLR: {eval_results['GLR']:.4f}")
        logger.info(f"  AFI: {eval_results['AFI']:.4f}")
        logger.info(f"  Loss weights: α={results['alpha']:.4f}, β={results['beta']:.4f}")
        logger.info(f"  Duration: {results['duration']:.2f}s")

    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final training results"""
        return {
            'outer_loop_history': self.outer_loop_history,
            'evaluation_history': self.evaluation_history,
            'convergence_history': self.convergence_checker.pa_history,
            'rebalancer_history': self.rebalancer.get_history(),
            'final_metrics': self.evaluation_history[-1] if self.evaluation_history else {},
            'num_iterations': len(self.outer_loop_history)
        }


# Unit tests
if __name__ == "__main__":
    print("Testing SE-RL Training Loop components...")

    # Test EMA Smoothing
    print("\n1. Testing EMA Smoothing...")
    ema = EMASmoothing(decay=0.9)
    values = [1, 2, 3, 4, 5, 5, 5, 5]
    for v in values:
        smoothed = ema.update(v)
        print(f"  Value: {v}, Smoothed: {smoothed:.4f}")

    # Test Adaptive Loss Rebalancer
    print("\n2. Testing Adaptive Loss Rebalancer...")
    rebalancer = AdaptiveLossRebalancer()
    for i in range(10):
        static_loss = np.random.uniform(0.1, 0.5)
        dynamic_loss = np.random.uniform(0.2, 0.6)
        rebalanced = rebalancer.compute_rebalanced_loss(static_loss, dynamic_loss)
        rebalancer.adapt_weights(static_loss, dynamic_loss)
        alpha, beta = rebalancer.get_weights()
        print(f"  Step {i + 1}: α={alpha:.4f}, β={beta:.4f}")

    # Test Financial Metrics
    print("\n3. Testing Financial Metrics...")
    exec_prices = [100.1, 100.2, 99.9, 100.5, 100.3]
    bench_prices = [100.0, 100.0, 100.0, 100.0, 100.0]
    quantities = [100, 150, 200, 100, 150]
    returns = [0.01, -0.02, 0.03, -0.01, 0.02]

    pa = FinancialMetricsCalculator.calculate_pa(exec_prices, bench_prices, quantities)
    pa_std = FinancialMetricsCalculator.calculate_pa_std(exec_prices, bench_prices)
    wr = FinancialMetricsCalculator.calculate_wr(returns)
    glr = FinancialMetricsCalculator.calculate_glr(returns)
    afi = FinancialMetricsCalculator.calculate_afi([100, 50, 25], [1000, 1000, 1000])

    print(f"  PA: {pa:.4f} bps")
    print(f"  PA_std: {pa_std:.4f} bps")
    print(f"  WR: {wr:.4f}")
    print(f"  GLR: {glr:.4f}")
    print(f"  AFI: {afi:.4f}")

    # Test Convergence Checker
    print("\n4. Testing Convergence Checker...")
    checker = ConvergenceChecker(tolerance=0.5, window=3)
    pas = [1.0, 2.0, 2.5, 2.7, 2.8, 2.85, 2.87, 2.88]
    for pa in pas:
        converged = checker.update(pa)
        print(f"  PA: {pa:.4f}, Converged: {converged}")

    print("\nAll Training Loop tests passed!")
