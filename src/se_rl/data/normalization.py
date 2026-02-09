"""
Data Normalization for SE-RL Framework
===================================

This module provides data normalization functionality for financial time-series data.

Author: AI Research Engineer
Date: 2024
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """Configuration for data normalization"""
    method: str = "zscore"  # zscore, minmax, robust, custom
    feature_groups: Dict[str, str] = None  # Different methods for different feature groups
    window_size: int = 252  # For rolling normalization
    clip_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier clipping
    
    def __post_init__(self):
        if self.feature_groups is None:
            self.feature_groups = {
                'price': 'zscore',
                'volume': 'robust',
                'technical': 'minmax',
                'time': 'minmax',
                'microstructure': 'zscore'
            }

class DataNormalizer:
    """Data normalization for financial time-series data"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.scalers = {}
        self.feature_stats = {}
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, feature_groups: Optional[Dict[str, List[str]]] = None):
        """Fit the normalizer to the data"""
        logger.info("Fitting data normalizer")
        
        if feature_groups is None:
            feature_groups = self._auto_detect_feature_groups(data)
        
        for group_name, features in feature_groups.items():
            if not features:
                continue
            
            # Get method for this group
            method = self.config.feature_groups.get(group_name, self.config.method)
            
            # Create scaler
            scaler = self._create_scaler(method)
            
            # Fit scaler
            group_data = data[features].dropna()
            if len(group_data) > 0:
                scaler.fit(group_data)
                self.scalers[group_name] = scaler
                
                # Store feature statistics
                self.feature_stats[group_name] = {
                    'method': method,
                    'features': features,
                    'mean': group_data.mean().to_dict() if hasattr(scaler, 'mean_') else None,
                    'std': group_data.std().to_dict() if hasattr(scaler, 'scale_') else None,
                    'min': group_data.min().to_dict() if hasattr(scaler, 'data_min_') else None,
                    'max': group_data.max().to_dict() if hasattr(scaler, 'data_max_') else None
                }
        
        self.is_fitted = True
        logger.info(f"Data normalizer fitted with {len(self.scalers)} feature groups")
    
    def transform(self, data: pd.DataFrame, feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Transform data using fitted normalizer"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transformation")
        
        logger.info("Transforming data")
        
        if feature_groups is None:
            feature_groups = self._auto_detect_feature_groups(data)
        
        normalized_data = data.copy()
        
        for group_name, features in feature_groups.items():
            if group_name not in self.scalers or not features:
                continue
            
            scaler = self.scalers[group_name]
            available_features = [f for f in features if f in data.columns]
            
            if available_features:
                # Transform the features
                group_data = data[available_features].dropna()
                if len(group_data) > 0:
                    normalized_group = scaler.transform(group_data)
                    normalized_data.loc[group_data.index, available_features] = normalized_group
                    
                    # Clip outliers if enabled
                    if self.config.clip_outliers:
                        normalized_data[available_features] = self._clip_outliers(
                            normalized_data[available_features]
                        )
        
        logger.info("Data transformation completed")
        return normalized_data
    
    def fit_transform(self, data: pd.DataFrame, feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Fit the normalizer and transform the data"""
        self.fit(data, feature_groups)
        return self.transform(data, feature_groups)
    
    def inverse_transform(self, data: pd.DataFrame, feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Inverse transform normalized data"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transformation")
        
        logger.info("Inverse transforming data")
        
        if feature_groups is None:
            feature_groups = self._auto_detect_feature_groups(data)
        
        original_data = data.copy()
        
        for group_name, features in feature_groups.items():
            if group_name not in self.scalers or not features:
                continue
            
            scaler = self.scalers[group_name]
            available_features = [f for f in features if f in data.columns]
            
            if available_features:
                group_data = data[available_features].dropna()
                if len(group_data) > 0:
                    original_group = scaler.inverse_transform(group_data)
                    original_data.loc[group_data.index, available_features] = original_group
        
        logger.info("Inverse transformation completed")
        return original_data
    
    def _create_scaler(self, method: str) -> BaseEstimator:
        """Create scaler based on method"""
        if method == "zscore":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "rolling":
            return RollingNormalizer(window_size=self.config.window_size)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _auto_detect_feature_groups(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Auto-detect feature groups based on column names"""
        groups = {
            'price': [],
            'volume': [],
            'technical': [],
            'time': [],
            'microstructure': []
        }
        
        for col in data.columns:
            if any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low', 'return']):
                groups['price'].append(col)
            elif 'volume' in col.lower():
                groups['volume'].append(col)
            elif any(x in col.lower() for x in ['rsi', 'macd', 'bb_', 'stoch', 'williams', 'atr', 'mfi', 'sma', 'ema']):
                groups['technical'].append(col)
            elif any(x in col.lower() for x in ['day', 'month', 'quarter', 'week', 'time', 'session']):
                groups['time'].append(col)
            elif any(x in col.lower() for x in ['spread', 'impact', 'imbalance', 'tick']):
                groups['microstructure'].append(col)
            else:
                # Default to price group
                groups['price'].append(col)
        
        return groups
    
    def _clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers in normalized data"""
        clipped_data = data.copy()
        
        for col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            lower_bound = mean_val - self.config.outlier_threshold * std_val
            upper_bound = mean_val + self.config.outlier_threshold * std_val
            
            clipped_data[col] = clipped_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return clipped_data
    
    def get_feature_statistics(self) -> Dict[str, Dict]:
        """Get feature statistics for each group"""
        return self.feature_stats.copy()
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file"""
        import pickle
        
        scaler_data = {
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scalers = scaler_data['scalers']
        self.feature_stats = scaler_data['feature_stats']
        self.config = scaler_data['config']
        self.is_fitted = True
        
        logger.info(f"Scalers loaded from {filepath}")

class RollingNormalizer(BaseEstimator, TransformerMixin):
    """Rolling normalization for time-series data"""
    
    def __init__(self, window_size: int = 252):
        self.window_size = window_size
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the rolling normalizer"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform data using rolling normalization"""
        if not self.is_fitted:
            raise ValueError("RollingNormalizer must be fitted before transformation")
        
        if isinstance(X, pd.DataFrame):
            return self._transform_dataframe(X)
        else:
            return self._transform_array(X)
    
    def _transform_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using rolling normalization"""
        normalized = X.copy()
        
        for col in X.columns:
            # Calculate rolling statistics
            rolling_mean = X[col].rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = X[col].rolling(window=self.window_size, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1)
            
            # Normalize
            normalized[col] = (X[col] - rolling_mean) / rolling_std
        
        return normalized
    
    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform numpy array using rolling normalization"""
        normalized = X.copy()
        
        for i in range(X.shape[1]):
            col_data = X[:, i]
            
            # Calculate rolling statistics
            rolling_mean = np.convolve(col_data, np.ones(self.window_size)/self.window_size, mode='same')
            rolling_var = np.convolve(col_data**2, np.ones(self.window_size)/self.window_size, mode='same') - rolling_mean**2
            rolling_std = np.sqrt(np.maximum(rolling_var, 1e-8))  # Avoid division by zero
            
            # Normalize
            normalized[:, i] = (col_data - rolling_mean) / rolling_std
        
        return normalized
    
    def inverse_transform(self, X):
        """Inverse transform is not implemented for rolling normalization"""
        raise NotImplementedError("Inverse transform not implemented for rolling normalization")

class AdaptiveNormalizer:
    """Adaptive normalization that adjusts to market regimes"""
    
    def __init__(self, regime_detector=None, base_normalizer=None):
        self.regime_detector = regime_detector
        self.base_normalizer = base_normalizer or DataNormalizer(NormalizationConfig())
        self.regime_normalizers = {}
        self.current_regime = None
    
    def fit(self, data: pd.DataFrame, regime_labels: Optional[List[str]] = None):
        """Fit normalizers for different market regimes"""
        if regime_labels is None and self.regime_detector:
            regime_labels = self.regime_detector.detect_regimes(data)
        
        if regime_labels:
            # Fit separate normalizers for each regime
            for regime in set(regime_labels):
                regime_data = data[regime_labels == regime]
                if len(regime_data) > 0:
                    normalizer = DataNormalizer(NormalizationConfig())
                    normalizer.fit(regime_data)
                    self.regime_normalizers[regime] = normalizer
        
        # Also fit base normalizer
        self.base_normalizer.fit(data)
    
    def transform(self, data: pd.DataFrame, regime_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """Transform data using regime-specific normalizers"""
        if regime_labels and self.regime_normalizers:
            # Use regime-specific normalizers
            normalized_data = data.copy()
            
            for regime in set(regime_labels):
                regime_mask = regime_labels == regime
                if regime in self.regime_normalizers:
                    regime_data = data[regime_mask]
                    normalized_regime = self.regime_normalizers[regime].transform(regime_data)
                    normalized_data.loc[regime_mask] = normalized_regime
            
            return normalized_data
        else:
            # Use base normalizer
            return self.base_normalizer.transform(data) 