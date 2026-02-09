# SE-RL Framework API Reference

## Overview

This document provides comprehensive API documentation for the Self-Evolutional Reinforcement Learning (SE-RL) framework. The framework is designed for financial order execution using LLM-powered RL agent generation and optimization.

## Table of Contents

1. [Core Framework](#core-framework)
2. [LLM Component Generation](#llm-component-generation)
3. [Financial Data Pipeline](#financial-data-pipeline)
4. [RL Training System](#rl-training-system)
5. [Configuration Classes](#configuration-classes)
6. [Utility Classes](#utility-classes)

## Core Framework

### SERLConfig

Configuration class for the main SE-RL framework.

```python
class SERLConfig:
    def __init__(self, 
                 convergence_epsilon: float = 0.1,
                 max_outer_iterations: int = 50,
                 max_inner_iterations: int = 1000,
                 learning_rate: float = 3e-4,
                 batch_size: int = 64,
                 device: str = "auto",
                 **kwargs):
```

**Parameters:**
- `convergence_epsilon` (float): Convergence threshold for PA improvement
- `max_outer_iterations` (int): Maximum number of outer loop iterations
- `max_inner_iterations` (int): Maximum number of inner loop iterations
- `learning_rate` (float): Learning rate for RL training
- `batch_size` (int): Batch size for training
- `device` (str): Device to use ("cpu", "cuda", or "auto")

### SERLFramework

Main framework class that orchestrates the entire SE-RL process.

```python
class SERLFramework:
    def __init__(self, config: SERLConfig):
        """
        Initialize the SE-RL framework.
        
        Args:
            config: Configuration object
        """
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the complete SE-RL training process.
        
        Returns:
            Dictionary containing training results and metrics
        """
    
    def initialize_components(self):
        """Initialize all framework components."""
    
    def _generate_algorithm(self, prompt: str) -> Dict[str, Any]:
        """Generate RL algorithm using LLM."""
    
    def _train_in_dynamic_environment(self, algorithm: Dict[str, Any]) -> Any:
        """Train agent in dynamic environment."""
    
    def _train_in_static_environment(self, algorithm: Dict[str, Any]) -> Any:
        """Train agent in static environment."""
    
    def _hybrid_environment_training(self, policy_static: Any, policy_dynamic: Any, algorithm: Dict[str, Any]) -> Any:
        """Combine static and dynamic environment training."""
    
    def _evaluate_policy(self, policy: Any) -> Dict[str, float]:
        """Evaluate policy performance."""
    
    def _apply_dual_level_enhancement(self, prompt: str, buffer: Any, metrics: Dict[str, float]):
        """Apply dual-level enhancement to LLM."""
    
    def _check_convergence(self, iteration: int) -> bool:
        """Check if training has converged."""
```

### PerformanceBuffer

Buffer for storing and managing performance history.

```python
class PerformanceBuffer:
    def __init__(self, max_size: int = 1000):
        """
        Initialize performance buffer.
        
        Args:
            max_size: Maximum number of entries to store
        """
    
    def add_performance(self, iteration: int, metrics: Dict[str, float], prompt: str, code: str):
        """Add performance metrics to buffer."""
    
    def get_recent_performance(self, n: int) -> List[Dict[str, Any]]:
        """Get the n most recent performance entries."""
    
    def get_best_performance(self) -> Optional[Dict[str, Any]]:
        """Get the best performance entry."""
    
    def get_performance_trend(self) -> Dict[str, List[float]]:
        """Get performance trend over iterations."""
```

### InstructionPopulation

Manages population of instructions for LLM enhancement.

```python
class InstructionPopulation:
    def __init__(self, max_size: int = 100):
        """
        Initialize instruction population.
        
        Args:
            max_size: Maximum number of instructions to store
        """
    
    def add_instruction(self, instruction: str, performance: float):
        """Add instruction with associated performance."""
    
    def sample_historical_instructions(self, n: int) -> List[str]:
        """Sample n historical instructions."""
    
    def get_best_instruction(self) -> Optional[str]:
        """Get the best performing instruction."""
    
    def update_instruction_performance(self, instruction: str, performance: float):
        """Update performance for existing instruction."""
```

## LLM Component Generation

### ComponentConfig

Configuration for LLM component generation.

```python
class ComponentConfig:
    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-7b-hf",
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 top_p: float = 0.9,
                 max_retries: int = 3,
                 **kwargs):
```

**Parameters:**
- `model_name` (str): Name of the LLM model to use
- `temperature` (float): Sampling temperature for generation
- `max_tokens` (int): Maximum tokens to generate
- `top_p` (float): Top-p sampling parameter
- `max_retries` (int): Maximum retries for failed generations

### LLMComponentGenerator

Generates RL components using LLM.

```python
class LLMComponentGenerator:
    def __init__(self, config: ComponentConfig):
        """
        Initialize LLM component generator.
        
        Args:
            config: Configuration object
        """
    
    def generate_reward_function(self, market_context: str, agent_type: str = "single") -> Dict[str, Any]:
        """
        Generate reward function using LLM.
        
        Args:
            market_context: Description of market environment
            agent_type: Type of agent ("single" or "multi")
            
        Returns:
            Dictionary containing generated code and metadata
        """
    
    def generate_network_architecture(self, state_dim: int, action_dim: int, agent_type: str = "single") -> Dict[str, Any]:
        """
        Generate network architecture using LLM.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            agent_type: Type of agent ("single" or "multi")
            
        Returns:
            Dictionary containing generated code and metadata
        """
    
    def generate_imagination_module(self, market_context: str) -> Dict[str, Any]:
        """
        Generate imagination module using LLM.
        
        Args:
            market_context: Description of market environment
            
        Returns:
            Dictionary containing generated code and metadata
        """
    
    def generate_multi_agent_system(self, num_agents: int, agent_types: List[str]) -> Dict[str, Any]:
        """
        Generate multi-agent system using LLM.
        
        Args:
            num_agents: Number of agents
            agent_types: List of agent types
            
        Returns:
            Dictionary containing generated code and metadata
        """
```

### CodeValidator

Validates generated code for syntax and functionality.

```python
class CodeValidator:
    def __init__(self):
        """Initialize code validator."""
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python syntax.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
    
    def validate_function_signature(self, code: str, function_name: str) -> Tuple[bool, str]:
        """
        Validate function signature.
        
        Args:
            code: Python code containing function
            function_name: Name of function to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
    
    def validate_imports(self, code: str) -> Tuple[bool, str]:
        """
        Validate import statements.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
```

## Financial Data Pipeline

### DataConfig

Configuration for data processing.

```python
class DataConfig:
    def __init__(self,
                 start_date: str = "2020-01-01",
                 end_date: str = "2024-01-01",
                 frequency: str = "1d",
                 window_size: int = 20,
                 normalize_method: str = "zscore",
                 **kwargs):
```

**Parameters:**
- `start_date` (str): Start date for data collection
- `end_date` (str): End date for data collection
- `frequency` (str): Data frequency ("1m", "5m", "15m", "1h", "1d")
- `window_size` (int): Window size for sequential data
- `normalize_method` (str): Normalization method ("zscore", "minmax", "robust")

### FinancialDataPipeline

Main data processing pipeline.

```python
class FinancialDataPipeline:
    def __init__(self, config: DataConfig):
        """
        Initialize data pipeline.
        
        Args:
            config: Configuration object
        """
    
    def load_csi100_data(self) -> Dict[str, Any]:
        """
        Load CSI100 data.
        
        Returns:
            Dictionary containing train/val/test datasets and loaders
        """
    
    def load_nasdaq100_data(self) -> Dict[str, Any]:
        """
        Load NASDAQ100 data.
        
        Returns:
            Dictionary containing train/val/test datasets and loaders
        """
    
    def process_custom_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process custom financial data.
        
        Args:
            data: Raw financial data
            
        Returns:
            Dictionary containing processed datasets and loaders
        """
```

### FeatureEngineer

Engineers features from raw financial data.

```python
class FeatureEngineer:
    def __init__(self, config: DataConfig):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object
        """
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Args:
            data: Raw financial data
            
        Returns:
            DataFrame with engineered features
        """
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data."""
    
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features to data."""
    
    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features to data."""
    
    def add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features to data."""
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to data."""
```

### DataNormalizer

Normalizes data for training.

```python
class DataNormalizer:
    def __init__(self, config: DataConfig):
        """
        Initialize data normalizer.
        
        Args:
            config: Configuration object
        """
    
    def fit_transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Fit normalizer and transform data.
        
        Args:
            data: Data to normalize
            feature_columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
    
    def transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Transform data using fitted normalizer.
        
        Args:
            data: Data to normalize
            feature_columns: Columns to normalize
            
        Returns:
            Normalized DataFrame
        """
    
    def inverse_transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            feature_columns: Columns to inverse transform
            
        Returns:
            Original scale DataFrame
        """
```

## RL Training System

### TrainingConfig

Configuration for RL training.

```python
class TrainingConfig:
    def __init__(self,
                 learning_rate: float = 3e-4,
                 batch_size: int = 64,
                 max_episodes: int = 1000,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 device: str = "auto",
                 **kwargs):
```

**Parameters:**
- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `max_episodes` (int): Maximum training episodes
- `gamma` (float): Discount factor
- `epsilon_start` (float): Initial exploration rate
- `epsilon_end` (float): Final exploration rate
- `epsilon_decay` (float): Exploration rate decay
- `device` (str): Device to use for training

### RLTrainer

Main RL training system.

```python
class RLTrainer:
    def __init__(self, config: TrainingConfig):
        """
        Initialize RL trainer.
        
        Args:
            config: Configuration object
        """
    
    def initialize_agent(self, state_dim: int, action_dim: int):
        """
        Initialize RL agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
        """
    
    def set_static_environment(self, data: pd.DataFrame):
        """
        Set static environment for training.
        
        Args:
            data: Historical market data
        """
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Returns:
            Dictionary containing episode results
        """
    
    def evaluate_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
    
    def save_model(self, path: str):
        """Save trained model."""
    
    def load_model(self, path: str):
        """Load trained model."""
```

### RLAgent

RL agent with actor-critic architecture.

```python
class RLAgent:
    def __init__(self, state_dim: int, action_dim: int, config: TrainingConfig):
        """
        Initialize RL agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            config: Training configuration
        """
    
    def get_action(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for given state.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_probability)
        """
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent parameters.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary containing loss values
        """
    
    def save(self, path: str):
        """Save agent parameters."""
    
    def load(self, path: str):
        """Load agent parameters."""
```

### StaticEnvironment

Static market environment for training.

```python
class StaticEnvironment:
    def __init__(self, data: pd.DataFrame, config: TrainingConfig):
        """
        Initialize static environment.
        
        Args:
            data: Historical market data
            config: Training configuration
        """
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state dictionary
        """
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
    
    def get_available_actions(self) -> List[float]:
        """Get available actions."""
```

### FinancialMetrics

Calculates financial performance metrics.

```python
class FinancialMetrics:
    @staticmethod
    def calculate_pa(execution_prices: List[float], vwap_prices: List[float]) -> float:
        """
        Calculate Price Advantage (PA).
        
        Args:
            execution_prices: List of execution prices
            vwap_prices: List of VWAP prices
            
        Returns:
            Price advantage value
        """
    
    @staticmethod
    def calculate_wr(returns: List[float]) -> float:
        """
        Calculate Win Ratio (WR).
        
        Args:
            returns: List of returns
            
        Returns:
            Win ratio value
        """
    
    @staticmethod
    def calculate_glr(returns: List[float]) -> float:
        """
        Calculate Gain-Loss Ratio (GLR).
        
        Args:
            returns: List of returns
            
        Returns:
            Gain-loss ratio value
        """
    
    @staticmethod
    def calculate_afi(final_inventories: List[float]) -> float:
        """
        Calculate Average Final Inventory (AFI).
        
        Args:
            final_inventories: List of final inventory positions
            
        Returns:
            Average final inventory value
        """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio value
        """
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Maximum drawdown value
        """
```

## Configuration Classes

All configuration classes support the following methods:

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""

def from_dict(self, config_dict: Dict[str, Any]):
    """Load configuration from dictionary."""

def save(self, path: str):
    """Save configuration to file."""

def load(self, path: str):
    """Load configuration from file."""

def validate(self) -> bool:
    """Validate configuration parameters."""
```

## Utility Classes

### Logger

Centralized logging system.

```python
class Logger:
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: str = None):
        """Setup logging configuration."""
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get logger instance."""
```

### ExperimentTracker

Tracks experiment progress and results.

```python
class ExperimentTracker:
    def __init__(self, experiment_name: str, output_dir: str):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            output_dir: Output directory
        """
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics at given step."""
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
    
    def save_results(self, results: Dict[str, Any]):
        """Save final results."""
    
    def generate_report(self) -> str:
        """Generate experiment report."""
```

## Error Handling

The framework includes comprehensive error handling with custom exceptions:

```python
class SERLFrameworkError(Exception):
    """Base exception for SE-RL framework."""

class LLMGenerationError(SERLFrameworkError):
    """Exception for LLM generation failures."""

class DataProcessingError(SERLFrameworkError):
    """Exception for data processing failures."""

class TrainingError(SERLFrameworkError):
    """Exception for training failures."""

class ConfigurationError(SERLFrameworkError):
    """Exception for configuration errors."""
```

## Usage Examples

### Basic Usage

```python
from se_rl_framework import SERLConfig, SERLFramework

# Initialize configuration
config = SERLConfig(
    convergence_epsilon=0.1,
    max_outer_iterations=50,
    device="cuda"
)

# Initialize framework
framework = SERLFramework(config)

# Run training
results = framework.run_training()
```

### Custom Configuration

```python
from llm_generator import ComponentConfig, LLMComponentGenerator

# Custom LLM configuration
llm_config = ComponentConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    temperature=0.8,
    max_retries=5
)

# Initialize generator
generator = LLMComponentGenerator(llm_config)

# Generate components
reward_function = generator.generate_reward_function(
    market_context="High-frequency trading environment",
    agent_type="single"
)
```

### Data Processing

```python
from financial_data_pipeline import DataConfig, FinancialDataPipeline

# Data configuration
data_config = DataConfig(
    start_date="2020-01-01",
    end_date="2024-01-01",
    frequency="1d",
    window_size=20
)

# Initialize pipeline
pipeline = FinancialDataPipeline(data_config)

# Load data
data = pipeline.load_csi100_data()
```

### RL Training

```python
from rl_trainer import TrainingConfig, RLTrainer

# Training configuration
train_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=128,
    max_episodes=2000
)

# Initialize trainer
trainer = RLTrainer(train_config)

# Initialize agent
trainer.initialize_agent(state_dim=64, action_dim=1)

# Train and evaluate
for episode in range(100):
    result = trainer.train_episode()
    
    if episode % 10 == 0:
        metrics = trainer.evaluate_policy(num_episodes=5)
        print(f"Episode {episode}: PA={metrics['PA']:.4f}")
``` 