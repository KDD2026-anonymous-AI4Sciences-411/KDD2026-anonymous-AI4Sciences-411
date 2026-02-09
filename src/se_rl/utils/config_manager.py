"""
Config Manager for SE-RL Framework
================================

This module provides configuration management functionality for the SE-RL framework.

Author: AI Research Engineer
Date: 2024
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for SE-RL framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        if self.config_path and os.path.exists(self.config_path):
            self._load_from_file()
        else:
            self._load_defaults()
    
    def _load_from_file(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {str(e)}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config = {
            'framework': {
                'convergence_epsilon': 0.1,
                'max_outer_iterations': 50,
                'max_inner_iterations': 1000
            },
            'llm': {
                'provider': 'local',
                'model_name': 'meta-llama/Llama-3.3-70B-Instruct',
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9
            },
            'training': {
                'learning_rate': 3e-4,
                'batch_size': 64,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995
            },
            'environment': {
                'static_env_weight': 0.5,
                'dynamic_env_weight': 0.5,
                'rebalance_iterations': 10,
                'initial_capital': 1000000.0,
                'transaction_cost': 0.001,
                'slippage': 0.0005
            },
            'dek': {
                'instruction_buffer_size': 100,
                'cache_replay_alpha': 0.1,
                'lora_rank': 16,
                'lora_alpha': 32
            },
            'hardware': {
                'device': 'auto',
                'num_gpus': 1,
                'mixed_precision': True
            },
            'data': {
                'dataset': 'csi100',
                'start_date': '2020-01-01',
                'end_date': '2024-01-01',
                'frequency': '1d',
                'window_size': 20
            },
            'multi_agent': {
                'num_agents': 3,
                'agent_types': ['market_maker', 'informed_trader', 'noise_trader']
            },
            'logging': {
                'level': 'INFO',
                'file': None,
                'console_output': True
            }
        }
        
        logger.info("Default configuration loaded")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to file"""
        save_path = file_path or self.config_path
        
        if not save_path:
            logger.warning("No file path specified for saving configuration")
            return
        
        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    # Default to YAML
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {str(e)}")
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary of updates"""
        for key, value in updates.items():
            self.set(key, value)
    
    def get_framework_config(self) -> Dict[str, Any]:
        """Get framework configuration"""
        return self.get('framework', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.get('llm', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.get('training', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return self.get('environment', {})
    
    def get_dek_config(self) -> Dict[str, Any]:
        """Get DEK configuration"""
        return self.get('dek', {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.get('hardware', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get('data', {})
    
    def get_multi_agent_config(self) -> Dict[str, Any]:
        """Get multi-agent configuration"""
        return self.get('multi_agent', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        required_sections = ['framework', 'llm', 'training', 'environment']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate specific values
        if self.get('framework.max_outer_iterations', 0) <= 0:
            logger.error("max_outer_iterations must be positive")
            return False
        
        if self.get('llm.temperature', 0) < 0 or self.get('llm.temperature', 0) > 2:
            logger.error("LLM temperature must be between 0 and 2")
            return False
        
        if self.get('training.learning_rate', 0) <= 0:
            logger.error("Learning rate must be positive")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def merge_configs(self, *configs: Dict[str, Any]):
        """Merge multiple configuration dictionaries"""
        merged = self.config.copy()
        
        for config in configs:
            self._deep_merge(merged, config)
        
        self.config = merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary"""
        self.config = config_dict.copy()
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 50)
        yaml.dump(self.config, sys.stdout, default_flow_style=False, indent=2) 