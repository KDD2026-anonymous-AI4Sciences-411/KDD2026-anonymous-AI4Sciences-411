import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
import requests
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for financial data pipeline"""
    # Data sources
    data_source: str = "CSI100" 
    cache_dir: str = "./data_cache"
    
    # Time parameters
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    frequency: str = "1d"  # 1m, 5m, 15m, 1h, 1d
    
    # Feature engineering
    window_size: int = 20  # Lookback window for features
    prediction_horizon: int = 5  # Prediction horizon
    
    # Data processing
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Normalization
    normalize_method: str = "zscore"  # zscore, minmax, robust
    scale_features: bool = True
    
    # Market data
    transaction_cost: float = 0.001
    slippage: float = 0.0005

class FinancialDataDownloader:
    """Downloads financial data from various sources"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Financial Data Downloader initialized with config: {config}")
    
    def download_csi100_data(self) -> pd.DataFrame:
        """
        Download CSI100 index data (China A-share market)
        
        Returns:
            DataFrame with CSI100 data
        """
        logger.info("Downloading CSI100 data...")
        
        # CSI100 components (simplified list - in practice would be full 100 stocks)
        csi100_symbols = [
            '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '002594.SZ',
            '300059.SZ', '300760.SZ', '600000.SH', '600036.SH', '600519.SH',
            '600887.SH', '000858.SZ', '002415.SZ', '002594.SZ', '300059.SZ'
        ]
        
        all_data = {}
        
        for symbol in csi100_symbols:
            try:
                data = self._download_stock_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Downloaded data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No CSI100 data downloaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = self._combine_stock_data(all_data)
        logger.info(f"CSI100 data download completed. Shape: {combined_data.shape}")
        
        return combined_data
    
    def download_nasdaq100_data(self) -> pd.DataFrame:
        """
        Download NASDAQ100 index data
        
        Returns:
            DataFrame with NASDAQ100 data
        """
        logger.info("Downloading NASDAQ100 data...")
        
        # NASDAQ100 components (simplified list)
        nasdaq100_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO'
        ]
        
        all_data = {}
        
        for symbol in nasdaq100_symbols:
            try:
                data = self._download_stock_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Downloaded data for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No NASDAQ100 data downloaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = self._combine_stock_data(all_data)
        logger.info(f"NASDAQ100 data download completed. Shape: {combined_data.shape}")
        
        return combined_data
    
    def _download_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download data for a single stock"""
        
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_{self.config.frequency}.csv"
        if cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded {symbol} from cache")
                return data
            except Exception as e:
                logger.warning(f"Cache load failed for {symbol}: {str(e)}")
        
        # Download from source
        try:
            if self.config.data_source == "yfinance":
                data = self._download_from_yfinance(symbol)
            else:
                logger.error(f"Unsupported data source: {self.config.data_source}")
                return None
            
            # Save to cache
            if data is not None and not data.empty:
                data.to_csv(cache_file)
                logger.info(f"Saved {symbol} to cache")
            
            return data
            
        except Exception as e:
            logger.error(f"Download failed for {symbol}: {str(e)}")
            return None
    
    def _download_from_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date,
                interval=self.config.frequency
            )
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            logger.error(f"yfinance download failed for {symbol}: {str(e)}")
            return None
    
    def _combine_stock_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple stock datasets"""
        combined_data = []
        
        for symbol, data in stock_data.items():
            # Ensure all datasets have the same date range
            data = data.sort_index()
            combined_data.append(data)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(combined_data, axis=0, ignore_index=False)
        combined = combined.sort_index()
        
        return combined

class FeatureEngineer:
    """Engineers features for financial time series"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_columns = []
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for financial data
        
        Args:
            data: Raw financial data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Basic price features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Time features
        df = self._add_time_features(df)
        
        # Remove NaN values
        df = df.dropna()
        
        logger.info(f"Feature engineering completed. Features: {len(self.feature_columns)}")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        
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
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_ma_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'price_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_std_{window}'] = df['close'].rolling(window).std()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_annualized'] = df['volatility'] * np.sqrt(252)
        
        self.feature_columns.extend([
            'returns', 'log_returns', 'price_change', 'price_change_pct',
            'hl_spread', 'hl_spread_pct', 'oc_spread', 'volatility', 'volatility_annualized'
        ])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(5).sum() / df['volume'].rolling(5).sum()
        
        # Volume ratios
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        
        self.feature_columns.extend([
            'volume_change', 'volume_ma_5', 'volume_ma_20', 'volume_price_trend',
            'volume_weighted_price', 'volume_ratio_5', 'volume_ratio_20'
        ])
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], window=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Price position relative to moving averages
        df['price_vs_sma_20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_ema_20'] = df['close'] / df['ema_20'] - 1
        
        self.feature_columns.extend([
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
            'price_vs_sma_20', 'price_vs_ema_20'
        ])
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Bid-ask spread approximation (using high-low as proxy)
        df['spread_estimate'] = df['hl_spread']
        
        # Order flow imbalance (approximation)
        df['order_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Price efficiency
        df['price_efficiency'] = abs(df['returns']) / df['volatility']
        
        # Market impact (simplified)
        df['market_impact'] = df['volume'] * df['returns'] / df['volume'].rolling(20).mean()
        
        self.feature_columns.extend([
            'spread_estimate', 'order_imbalance', 'price_efficiency', 'market_impact'
        ])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        # Month
        df['month'] = df.index.month
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Day of year
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        self.feature_columns.extend([
            'day_of_week', 'month', 'quarter', 'day_of_year',
            'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
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
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

class DataNormalizer:
    """Normalizes financial data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scalers = {}
        self.feature_means = {}
        self.feature_stds = {}
        
    def fit_transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Fit normalizer and transform data
        
        Args:
            data: Input data
            feature_columns: Columns to normalize
            
        Returns:
            Normalized data
        """
        logger.info("Fitting and transforming data...")
        
        df = data.copy()
        
        if self.config.scale_features:
            for col in feature_columns:
                if col in df.columns:
                    if self.config.normalize_method == "zscore":
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        df[col] = (df[col] - mean_val) / (std_val + 1e-8)
                        
                        # Store for later use
                        self.feature_means[col] = mean_val
                        self.feature_stds[col] = std_val
                        
                    elif self.config.normalize_method == "minmax":
                        min_val = df[col].min()
                        max_val = df[col].max()
                        df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                        
                        # Store for later use
                        self.feature_means[col] = min_val
                        self.feature_stds[col] = max_val - min_val
                    
                    elif self.config.normalize_method == "robust":
                        median_val = df[col].median()
                        mad_val = df[col].mad()
                        df[col] = (df[col] - median_val) / (mad_val + 1e-8)
                        
                        # Store for later use
                        self.feature_means[col] = median_val
                        self.feature_stds[col] = mad_val
        
        logger.info(f"Data normalization completed using {self.config.normalize_method}")
        return df
    
    def transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted normalizer
        
        Args:
            data: Input data
            feature_columns: Columns to normalize
            
        Returns:
            Normalized data
        """
        df = data.copy()
        
        if self.config.scale_features:
            for col in feature_columns:
                if col in df.columns and col in self.feature_means:
                    mean_val = self.feature_means[col]
                    std_val = self.feature_stds[col]
                    df[col] = (df[col] - mean_val) / (std_val + 1e-8)
        
        return df

class FinancialDataset(Dataset):
    """PyTorch dataset for financial time series"""
    
    def __init__(self, data: pd.DataFrame, feature_columns: List[str], 
                 target_column: str = 'returns', window_size: int = 20,
                 prediction_horizon: int = 5):
        self.data = data
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Financial dataset created with {len(self.sequences)} sequences")
    
    def _create_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """Create sequences for training"""
        sequences = []
        
        for i in range(self.window_size, len(self.data) - self.prediction_horizon):
            # Input sequence
            input_data = self.data[self.feature_columns].iloc[i-self.window_size:i].values
            input_tensor = torch.FloatTensor(input_data)
            
            # Target (future returns)
            target_data = self.data[self.target_column].iloc[i:i+self.prediction_horizon].values
            target_tensor = torch.FloatTensor(target_data)
            
            # Market state (current state)
            current_state = self.data.iloc[i].to_dict()
            
            sequences.append({
                'input': input_tensor,
                'target': target_tensor,
                'state': current_state,
                'index': i
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]

class FinancialDataPipeline:
    """Main data pipeline for SE-RL framework"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.downloader = FinancialDataDownloader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.normalizer = DataNormalizer(config)
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_columns = []
        
        logger.info(f"Financial Data Pipeline initialized with config: {config}")
    
    def load_csi100_data(self) -> Dict[str, Any]:
        """Load and process CSI100 data"""
        logger.info("Loading CSI100 data...")
        
        # Download data
        raw_data = self.downloader.download_csi100_data()
        
        if raw_data.empty:
            logger.error("No CSI100 data available")
            return {}
        
        # Process data
        processed_data = self._process_data(raw_data)
        
        return processed_data
    
    def load_nasdaq100_data(self) -> Dict[str, Any]:
        """Load and process NASDAQ100 data"""
        logger.info("Loading NASDAQ100 data...")
        
        # Download data
        raw_data = self.downloader.download_nasdaq100_data()
        
        if raw_data.empty:
            logger.error("No NASDAQ100 data available")
            return {}
        
        # Process data
        processed_data = self._process_data(raw_data)
        
        return processed_data
    
    def _process_data(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Process raw financial data"""
        
        # Engineer features
        feature_data = self.feature_engineer.engineer_features(raw_data)
        self.feature_columns = self.feature_engineer.feature_columns
        
        # Split data
        train_data, val_data, test_data = self._split_data(feature_data)
        
        # Normalize data
        train_data_norm = self.normalizer.fit_transform(train_data, self.feature_columns)
        val_data_norm = self.normalizer.transform(val_data, self.feature_columns)
        test_data_norm = self.normalizer.transform(test_data, self.feature_columns)
        
        # Create datasets
        train_dataset = FinancialDataset(
            train_data_norm, self.feature_columns, 
            window_size=self.config.window_size,
            prediction_horizon=self.config.prediction_horizon
        )
        
        val_dataset = FinancialDataset(
            val_data_norm, self.feature_columns,
            window_size=self.config.window_size,
            prediction_horizon=self.config.prediction_horizon
        )
        
        test_dataset = FinancialDataset(
            test_data_norm, self.feature_columns,
            window_size=self.config.window_size,
            prediction_horizon=self.config.prediction_horizon
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return {
            'train_data': train_data_norm,
            'val_data': val_data_norm,
            'test_data': test_data_norm,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'feature_columns': self.feature_columns,
            'normalizer': self.normalizer
        }
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets"""
        
        total_len = len(data)
        train_len = int(total_len * self.config.train_split)
        val_len = int(total_len * self.config.val_split)
        
        train_data = data.iloc[:train_len]
        val_data = data.iloc[train_len:train_len + val_len]
        test_data = data.iloc[train_len + val_len:]
        
        logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def get_market_state(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Get market state for RL environment"""
        
        if index >= len(data):
            return {}
        
        current_row = data.iloc[index]
        
        # Basic market state
        market_state = {
            'current_price': current_row.get('close', 100.0),
            'open_price': current_row.get('open', 100.0),
            'high_price': current_row.get('high', 100.0),
            'low_price': current_row.get('low', 100.0),
            'volume': current_row.get('volume', 1000000),
            'returns': current_row.get('returns', 0.0),
            'volatility': current_row.get('volatility', 0.01),
            'rsi': current_row.get('rsi', 50.0),
            'macd': current_row.get('macd', 0.0),
            'bb_position': current_row.get('bb_position', 0.5),
            'volume_ratio': current_row.get('volume_ratio_20', 1.0),
            'timestamp': data.index[index] if index < len(data.index) else datetime.now()
        }
        
        # Add feature vector
        feature_vector = []
        for col in self.feature_columns:
            if col in current_row:
                feature_vector.append(float(current_row[col]))
            else:
                feature_vector.append(0.0)
        
        market_state['feature_vector'] = feature_vector
        
        return market_state

# Example usage
if __name__ == "__main__":
    # Initialize data pipeline
    config = DataConfig()
    pipeline = FinancialDataPipeline(config)
    
    # Load CSI100 data
    csi100_data = pipeline.load_csi100_data()
    
    if csi100_data:
        print("CSI100 data loaded successfully!")
        print(f"Feature columns: {len(csi100_data['feature_columns'])}")
        print(f"Train samples: {len(csi100_data['train_dataset'])}")
        print(f"Val samples: {len(csi100_data['val_dataset'])}")
        print(f"Test samples: {len(csi100_data['test_dataset'])}")
        
        # Test data loader
        for batch in csi100_data['train_loader']:
            print(f"Batch input shape: {batch['input'].shape}")
            print(f"Batch target shape: {batch['target'].shape}")
            break
    
    # Load NASDAQ100 data
    nasdaq100_data = pipeline.load_nasdaq100_data()
    
    if nasdaq100_data:
        print("\nNASDAQ100 data loaded successfully!")
        print(f"Feature columns: {len(nasdaq100_data['feature_columns'])}")
        print(f"Train samples: {len(nasdaq100_data['train_dataset'])}")
        print(f"Val samples: {len(nasdaq100_data['val_dataset'])}")
        print(f"Test samples: {len(nasdaq100_data['test_dataset'])}") 