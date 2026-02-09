"""
Feature Engineering for SE-RL Framework
====================================

This module provides feature engineering functionality for financial time-series data.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    window_sizes: List[int] = None
    include_technical_indicators: bool = True
    include_microstructure_features: bool = True
    include_time_features: bool = True
    include_volume_features: bool = True
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [5, 10, 20, 50]

class FeatureEngineer:
    """Feature engineering for financial time-series data"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_names = []
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw financial data"""
        logger.info("Starting feature engineering")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Technical indicators
        if self.config.include_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Volume features
        if self.config.include_volume_features:
            df = self._add_volume_features(df)
        
        # Market microstructure features
        if self.config.include_microstructure_features:
            df = self._add_microstructure_features(df)
        
        # Time-based features
        if self.config.include_time_features:
            df = self._add_time_features(df)
        
        # Remove NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        self.feature_names = list(df.columns)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_pct'] = df['hl_spread'] * 100
        
        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        df['oc_spread_pct'] = df['oc_spread'] * 100
        
        # Price ranges
        for window in self.config.window_sizes:
            df[f'price_range_{window}'] = (df['high'].rolling(window).max() - df['low'].rolling(window).min()) / df['close']
            df[f'price_volatility_{window}'] = df['returns'].rolling(window).std()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # Moving averages
        for window in self.config.window_sizes:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
            df[f'price_to_ema_{window}'] = df['close'] / df[f'ema_{window}']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'], df['bb_middle'] = self._calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # Average True Range (ATR)
        df['atr'] = self._calculate_atr(df)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for window in self.config.window_sizes:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Volume indicators
        df['volume_price_trend'] = (df['volume'] * df['returns']).cumsum()
        df['on_balance_volume'] = self._calculate_on_balance_volume(df)
        
        # Volume rate of change
        df['volume_roc'] = df['volume'].pct_change()
        
        # Money Flow Index
        df['mfi'] = self._calculate_money_flow_index(df)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread approximation
        df['spread_estimate'] = (df['high'] - df['low']) / df['close']
        
        # Price impact
        df['price_impact'] = df['volume'] * df['returns'].abs()
        
        # Order flow imbalance (approximation)
        df['order_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Tick size effects
        df['tick_size_ratio'] = df['price_change'].abs() / df['close']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        # Month
        df['month'] = df.index.month
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Day of year
        df['day_of_year'] = df.index.dayofyear
        
        # Week of year
        df['week_of_year'] = df.index.isocalendar().week
        
        # Time since market open (approximation)
        df['time_since_open'] = (df.index.hour * 60 + df.index.minute) / (24 * 60)
        
        # Market session (morning/afternoon)
        df['market_session'] = (df.index.hour >= 12).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_on_balance_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names"""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category"""
        groups = {
            'price': [col for col in self.feature_names if any(x in col for x in ['returns', 'price', 'close', 'open', 'high', 'low'])],
            'volume': [col for col in self.feature_names if 'volume' in col],
            'technical': [col for col in self.feature_names if any(x in col for x in ['rsi', 'macd', 'bb_', 'stoch', 'williams', 'atr', 'mfi'])],
            'time': [col for col in self.feature_names if any(x in col for x in ['day', 'month', 'quarter', 'week', 'time', 'session'])],
            'microstructure': [col for col in self.feature_names if any(x in col for x in ['spread', 'impact', 'imbalance', 'tick'])]
        }
        return groups 