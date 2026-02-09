"""
Static Environment for SE-RL Framework
===================================

This module implements a static environment for training RL agents
using historical financial data.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StaticEnvironmentConfig:
    """Configuration for static environment"""
    initial_capital: float = 1000000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position: float = 0.1  # Maximum position as fraction of capital
    min_trade_size: float = 1000.0
    max_trade_size: float = 100000.0
    window_size: int = 20
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'
            ]

class StaticEnvironment:
    """Static environment for training RL agents on historical data"""
    
    def __init__(self, data: pd.DataFrame, config: StaticEnvironmentConfig):
        self.data = data.copy()
        self.config = config
        self.current_step = 0
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.position = 0.0
        self.trades = []
        self.portfolio_values = []
        
        # Preprocess data
        self._preprocess_data()
        
        # Initialize state
        self.reset()
        
        logger.info(f"Static environment initialized with {len(self.data)} data points")
    
    def _preprocess_data(self):
        """Preprocess the financial data"""
        # Ensure all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate technical indicators
        self._calculate_technical_indicators()
        
        # Normalize features
        self._normalize_features()
        
        # Create state windows
        self._create_state_windows()
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators"""
        # Returns
        self.data['returns'] = self.data['close'].pct_change()
        
        # Volatility (rolling standard deviation of returns)
        self.data['volatility'] = self.data['returns'].rolling(window=20).std()
        
        # RSI
        self.data['rsi'] = self._calculate_rsi(self.data['close'])
        
        # MACD
        self.data['macd'], self.data['macd_signal'] = self._calculate_macd(self.data['close'])
        
        # Bollinger Bands
        self.data['bollinger_upper'], self.data['bollinger_lower'] = self._calculate_bollinger_bands(self.data['close'])
        
        # Volume indicators
        self.data['volume_ma'] = self.data['volume'].rolling(window=20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_ma']
        
        # Price indicators
        self.data['price_ma_short'] = self.data['close'].rolling(window=5).mean()
        self.data['price_ma_long'] = self.data['close'].rolling(window=20).mean()
        self.data['price_ratio'] = self.data['price_ma_short'] / self.data['price_ma_long']
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def _normalize_features(self):
        """Normalize features using z-score normalization"""
        feature_columns = [col for col in self.config.feature_columns if col in self.data.columns]
        
        for col in feature_columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                # Price and volume normalization
                self.data[f'{col}_norm'] = self.data[col] / self.data[col].rolling(window=20).mean()
            else:
                # Z-score normalization for other features
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val > 0:
                    self.data[f'{col}_norm'] = (self.data[col] - mean_val) / std_val
                else:
                    self.data[f'{col}_norm'] = 0
    
    def _create_state_windows(self):
        """Create state windows for the environment"""
        feature_columns = [col for col in self.config.feature_columns if col in self.data.columns]
        norm_columns = [f'{col}_norm' for col in feature_columns if f'{col}_norm' in self.data.columns]
        
        # Create state windows
        self.state_data = []
        for i in range(self.config.window_size, len(self.data)):
            window_data = self.data.iloc[i-self.config.window_size:i][norm_columns].values
            self.state_data.append(window_data)
        
        self.state_data = np.array(self.state_data)
        logger.info(f"Created {len(self.state_data)} state windows with shape {self.state_data.shape}")
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        
        # Return initial state
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        if self.current_step >= len(self.state_data):
            return np.zeros((self.config.window_size, len(self.config.feature_columns)))
        
        state = self.state_data[self.current_step].copy()
        
        # Add portfolio information
        portfolio_info = np.array([
            self.current_capital / self.initial_capital,  # Normalized capital
            self.position,  # Current position
            len(self.trades),  # Number of trades
        ])
        
        # Pad or truncate portfolio info to match feature dimension
        if len(portfolio_info) < state.shape[1]:
            portfolio_info = np.pad(portfolio_info, (0, state.shape[1] - len(portfolio_info)))
        else:
            portfolio_info = portfolio_info[:state.shape[1]]
        
        # Add portfolio info to state
        state[0, :len(portfolio_info)] = portfolio_info
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        if self.current_step >= len(self.state_data) - 1:
            return self._get_state(), 0.0, True, {'message': 'Episode ended'}
        
        # Parse action
        action_type, action_size = self._parse_action(action)
        
        # Execute trade
        reward, trade_info = self._execute_trade(action_type, action_size)
        
        # Update step
        self.current_step += 1
        
        # Update portfolio value
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.current_capital + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Check if episode is done
        done = self.current_step >= len(self.state_data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Create info dictionary
        info = {
            'step': self.current_step,
            'capital': self.current_capital,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'trade_info': trade_info,
            'current_price': current_price
        }
        
        return next_state, reward, done, info
    
    def _parse_action(self, action: np.ndarray) -> Tuple[str, float]:
        """Parse action from agent output"""
        # Action format: [buy_probability, sell_probability, hold_probability, size]
        action_probs = action[:3]
        action_size = action[3] if len(action) > 3 else 0.5
        
        # Determine action type
        action_type_idx = np.argmax(action_probs)
        action_types = ['buy', 'sell', 'hold']
        action_type = action_types[action_type_idx]
        
        # Normalize action size to [0, 1]
        action_size = np.clip(action_size, 0.0, 1.0)
        
        return action_type, action_size
    
    def _execute_trade(self, action_type: str, action_size: float) -> Tuple[float, Dict[str, Any]]:
        """Execute a trade and calculate reward"""
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        trade_info = {
            'action_type': action_type,
            'action_size': action_size,
            'current_price': current_price,
            'next_price': next_price,
            'executed': False
        }
        
        reward = 0.0
        
        if action_type == 'buy' and action_size > 0:
            # Calculate trade size
            max_trade_value = self.current_capital * self.config.max_position
            trade_value = max_trade_value * action_size
            trade_value = np.clip(trade_value, self.config.min_trade_size, self.config.max_trade_size)
            
            # Calculate shares to buy
            shares = trade_value / current_price
            
            # Apply transaction costs and slippage
            total_cost = trade_value * (1 + self.config.transaction_cost + self.config.slippage)
            
            if total_cost <= self.current_capital:
                # Execute trade
                self.current_capital -= total_cost
                self.position += shares
                
                # Record trade
                trade_info.update({
                    'executed': True,
                    'shares': shares,
                    'trade_value': trade_value,
                    'total_cost': total_cost
                })
                
                self.trades.append(trade_info.copy())
                
                # Calculate reward based on price movement
                price_change = (next_price - current_price) / current_price
                reward = price_change * shares * current_price
                
        elif action_type == 'sell' and action_size > 0:
            # Calculate shares to sell
            max_shares = self.position
            shares_to_sell = max_shares * action_size
            
            if shares_to_sell > 0:
                # Calculate trade value
                trade_value = shares_to_sell * current_price
                
                # Apply transaction costs and slippage
                net_proceeds = trade_value * (1 - self.config.transaction_cost - self.config.slippage)
                
                # Execute trade
                self.current_capital += net_proceeds
                self.position -= shares_to_sell
                
                # Record trade
                trade_info.update({
                    'executed': True,
                    'shares': shares_to_sell,
                    'trade_value': trade_value,
                    'net_proceeds': net_proceeds
                })
                
                self.trades.append(trade_info.copy())
                
                # Calculate reward based on price movement
                price_change = (current_price - next_price) / current_price
                reward = price_change * shares_to_sell * current_price
        
        else:  # hold
            # Small penalty for holding to encourage trading
            reward = -0.0001
        
        return reward, trade_info
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if not self.portfolio_values:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        max_drawdown = np.min(portfolio_values) / np.max(portfolio_values) - 1
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'num_trades': len(self.trades)
        }
    
    def render(self):
        """Render the current state (for debugging)"""
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.current_capital + (self.position * current_price)
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Capital: ${self.current_capital:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Trades: {len(self.trades)}")
        print("-" * 50) 