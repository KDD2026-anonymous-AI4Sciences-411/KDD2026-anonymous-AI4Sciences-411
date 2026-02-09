"""
Financial Metrics for SE-RL Framework
==================================

This module provides financial metrics for evaluating trading performance.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for financial metrics"""
    risk_free_rate: float = 0.02  # Annual risk-free rate
    trading_days_per_year: int = 252
    confidence_level: float = 0.95  # For VaR calculations

class FinancialMetrics:
    """Financial metrics calculator for trading performance evaluation"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
    
    def calculate_all_metrics(self, portfolio_values: List[float], 
                            benchmark_values: Optional[List[float]] = None,
                            trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Calculate all financial metrics"""
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(portfolio_values))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(portfolio_values))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(portfolio_values))
        
        # Trading metrics
        if trades:
            metrics.update(self._calculate_trading_metrics(trades))
        
        # Benchmark comparison
        if benchmark_values:
            metrics.update(self._calculate_benchmark_metrics(portfolio_values, benchmark_values))
        
        # Additional metrics
        metrics.update(self._calculate_additional_metrics(portfolio_values))
        
        return metrics
    
    def _calculate_return_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized return
        num_periods = len(portfolio_values) - 1
        annualized_return = (1 + total_return) ** (self.config.trading_days_per_year / num_periods) - 1
        
        # Average return
        avg_return = np.mean(returns)
        
        # Cumulative return
        cumulative_return = (1 + returns).prod() - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_return': avg_return,
            'cumulative_return': cumulative_return
        }
    
    def _calculate_risk_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate risk-based metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Volatility
        volatility = np.std(returns)
        annualized_volatility = volatility * np.sqrt(self.config.trading_days_per_year)
        
        # Maximum drawdown
        max_drawdown, drawdown_duration = self._calculate_max_drawdown(portfolio_values)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, (1 - self.config.confidence_level) * 100)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation
        }
    
    def _calculate_risk_adjusted_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe ratio
        excess_returns = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(self.config.trading_days_per_year)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = np.mean(excess_returns) / downside_deviation
        sortino_ratio_annualized = sortino_ratio * np.sqrt(self.config.trading_days_per_year)
        
        # Calmar ratio
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        max_dd, _ = self._calculate_max_drawdown(portfolio_values)
        calmar_ratio = total_return / abs(max_dd) if max_dd != 0 else 0
        
        # Information ratio (assuming zero benchmark)
        information_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_annualized': sharpe_ratio_annualized,
            'sortino_ratio': sortino_ratio,
            'sortino_ratio_annualized': sortino_ratio_annualized,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_trading_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        if not trades:
            return {}
        
        # Extract trade information
        trade_returns = []
        winning_trades = []
        losing_trades = []
        
        for trade in trades:
            if 'return' in trade:
                trade_returns.append(trade['return'])
                if trade['return'] > 0:
                    winning_trades.append(trade['return'])
                else:
                    losing_trades.append(trade['return'])
        
        if not trade_returns:
            return {}
        
        trade_returns = np.array(trade_returns)
        
        # Win rate
        win_rate = len(winning_trades) / len(trade_returns)
        
        # Average win/loss
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Profit factor
        total_wins = np.sum(winning_trades) if winning_trades else 0
        total_losses = abs(np.sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Gain-loss ratio
        gain_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        
        # Expected value
        expected_value = np.mean(trade_returns)
        
        # Trade frequency
        trade_frequency = len(trades) / len(trade_returns) if trade_returns else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'gain_loss_ratio': gain_loss_ratio,
            'expected_value': expected_value,
            'trade_frequency': trade_frequency,
            'num_trades': len(trades)
        }
    
    def _calculate_benchmark_metrics(self, portfolio_values: List[float], 
                                   benchmark_values: List[float]) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        if len(portfolio_values) != len(benchmark_values) or len(portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        benchmark_values = np.array(benchmark_values)
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        # Alpha (excess return)
        portfolio_total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        benchmark_total_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        alpha = portfolio_total_return - benchmark_total_return
        
        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Tracking error
        tracking_error = np.std(portfolio_returns - benchmark_returns)
        tracking_error_annualized = tracking_error * np.sqrt(self.config.trading_days_per_year)
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        information_ratio_annualized = information_ratio * np.sqrt(self.config.trading_days_per_year)
        
        # Correlation
        correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'tracking_error_annualized': tracking_error_annualized,
            'information_ratio': information_ratio,
            'information_ratio_annualized': information_ratio_annualized,
            'correlation': correlation
        }
    
    def _calculate_additional_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # VaR and CVaR using different methods
        var_historical = np.percentile(returns, 5)
        cvar_historical = np.mean(returns[returns <= var_historical])
        
        # Omega ratio
        threshold = 0
        positive_returns = returns[returns > threshold]
        negative_returns = returns[returns < threshold]
        
        if len(negative_returns) > 0:
            omega_ratio = np.sum(positive_returns) / abs(np.sum(negative_returns))
        else:
            omega_ratio = float('inf')
        
        # Sterling ratio
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        max_dd, _ = self._calculate_max_drawdown(portfolio_values)
        sterling_ratio = (total_return - self.config.risk_free_rate) / abs(max_dd) if max_dd != 0 else 0
        
        # Burke ratio
        burke_ratio = (total_return - self.config.risk_free_rate) / np.sqrt(np.sum(returns[returns < 0]**2)) if np.sum(returns[returns < 0]**2) > 0 else 0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_historical': var_historical,
            'cvar_historical': cvar_historical,
            'omega_ratio': omega_ratio,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration"""
        peak = portfolio_values[0]
        max_dd = 0
        max_dd_duration = 0
        current_duration = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_duration = current_duration
        
        return max_dd, max_dd_duration
    
    def calculate_pa_metric(self, portfolio_values: List[float], 
                          vwap_values: List[float]) -> float:
        """Calculate Price Advantage (PA) over VWAP"""
        if len(portfolio_values) != len(vwap_values) or len(portfolio_values) < 2:
            return 0.0
        
        portfolio_values = np.array(portfolio_values)
        vwap_values = np.array(vwap_values)
        
        # Calculate price advantage
        price_advantage = (portfolio_values - vwap_values) / vwap_values
        pa_metric = np.mean(price_advantage)
        
        return pa_metric
    
    def calculate_wr_metric(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Win Ratio (WR)"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('return', 0) > 0)
        total_trades = len(trades)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def calculate_glr_metric(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate Gain-Loss Ratio (GLR)"""
        if not trades:
            return 0.0
        
        winning_trades = [trade['return'] for trade in trades if trade.get('return', 0) > 0]
        losing_trades = [trade['return'] for trade in trades if trade.get('return', 0) < 0]
        
        avg_gain = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1.0
        
        return avg_gain / avg_loss if avg_loss > 0 else 0.0
    
    def calculate_afi_metric(self, final_inventory: float, 
                           initial_inventory: float) -> float:
        """Calculate Averaged Final Inventory (AFI)"""
        if initial_inventory == 0:
            return 0.0
        
        return (final_inventory - initial_inventory) / initial_inventory
    
    def generate_performance_report(self, metrics: Dict[str, float]) -> str:
        """Generate a formatted performance report"""
        report = "Financial Performance Report\n"
        report += "=" * 50 + "\n\n"
        
        # Return metrics
        report += "RETURN METRICS:\n"
        report += f"Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)\n"
        report += f"Annualized Return: {metrics.get('annualized_return', 0):.4f} ({metrics.get('annualized_return', 0)*100:.2f}%)\n"
        report += f"Average Return: {metrics.get('avg_return', 0):.4f} ({metrics.get('avg_return', 0)*100:.2f}%)\n\n"
        
        # Risk metrics
        report += "RISK METRICS:\n"
        report += f"Volatility: {metrics.get('volatility', 0):.4f} ({metrics.get('volatility', 0)*100:.2f}%)\n"
        report += f"Annualized Volatility: {metrics.get('annualized_volatility', 0):.4f} ({metrics.get('annualized_volatility', 0)*100:.2f}%)\n"
        report += f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)\n"
        report += f"VaR (95%): {metrics.get('var_95', 0):.4f} ({metrics.get('var_95', 0)*100:.2f}%)\n\n"
        
        # Risk-adjusted metrics
        report += "RISK-ADJUSTED METRICS:\n"
        report += f"Sharpe Ratio: {metrics.get('sharpe_ratio_annualized', 0):.4f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio_annualized', 0):.4f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}\n\n"
        
        # Trading metrics
        if 'win_rate' in metrics:
            report += "TRADING METRICS:\n"
            report += f"Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.2f}%)\n"
            report += f"Profit Factor: {metrics.get('profit_factor', 0):.4f}\n"
            report += f"Gain-Loss Ratio: {metrics.get('gain_loss_ratio', 0):.4f}\n"
            report += f"Number of Trades: {metrics.get('num_trades', 0)}\n\n"
        
        return report 