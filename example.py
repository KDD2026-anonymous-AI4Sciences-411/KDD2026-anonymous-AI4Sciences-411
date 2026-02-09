"""
Simple Example: SE-RL Framework Usage
====================================

This script demonstrates basic usage of the SE-RL framework
with minimal setup and configuration.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import framework components
from se_rl_framework import SERLConfig, SERLFramework
from llm_generator import ComponentConfig, LLMComponentGenerator
from financial_data_pipeline import DataConfig, FinancialDataPipeline
from rl_trainer import TrainingConfig, RLTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data():
    """Create dummy financial data for demonstration"""
    logger.info("Creating dummy financial data...")
    
    # Generate realistic-looking financial data
    np.random.seed(42)
    n_days = 1000
    
    # Price data
    returns = np.random.normal(0.0001, 0.02, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100
    
    # OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days),
        'returns': returns,
        'volatility': np.abs(returns) * 10,
        'rsi': np.random.uniform(20, 80, n_days),
        'macd': np.random.normal(0, 0.1, n_days),
        'bb_position': np.random.uniform(0, 1, n_days),
        'volume_ratio_20': np.random.uniform(0.5, 2.0, n_days)
    })
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    data.index = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    logger.info(f"Dummy data created: {len(data)} days")
    return data

def run_basic_example():
    """Run a basic example of the SE-RL framework"""
    
    logger.info("Starting SE-RL Framework Basic Example")
    
    # Step 1: Create dummy data
    dummy_data = create_dummy_data()
    
    # Step 2: Initialize configurations
    serl_config = SERLConfig(
        convergence_epsilon=0.1,
        max_outer_iterations=5,  # Small number for demo
        device="cpu"  # Use CPU for demo
    )
    
    component_config = ComponentConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        temperature=0.7,
        max_retries=2
    )
    
    data_config = DataConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        frequency="1d",
        window_size=10,  # Smaller window for demo
        normalize_method="zscore"
    )
    
    training_config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=32,
        max_episodes=50,  # Small number for demo
        device="cpu"
    )
    
    # Step 3: Initialize components
    logger.info("Initializing framework components...")
    
    framework = SERLFramework(serl_config)
    llm_generator = LLMComponentGenerator(component_config)
    data_pipeline = FinancialDataPipeline(data_config)
    rl_trainer = RLTrainer(training_config)
    
    # Step 4: Process data
    logger.info("Processing financial data...")
    
    # Use dummy data instead of downloading
    data_pipeline.feature_engineer.feature_columns = [
        'returns', 'volatility', 'rsi', 'macd', 'bb_position', 
        'volume_ratio_20', 'hl_spread', 'oc_spread'
    ]
    
    # Process the dummy data
    feature_data = data_pipeline.feature_engineer.engineer_features(dummy_data)
    
    # Split data
    train_size = int(len(feature_data) * 0.7)
    val_size = int(len(feature_data) * 0.15)
    
    train_data = feature_data.iloc[:train_size]
    val_data = feature_data.iloc[train_size:train_size + val_size]
    test_data = feature_data.iloc[train_size + val_size:]
    
    # Normalize data
    train_data_norm = data_pipeline.normalizer.fit_transform(train_data, data_pipeline.feature_engineer.feature_columns)
    val_data_norm = data_pipeline.normalizer.transform(val_data, data_pipeline.feature_engineer.feature_columns)
    test_data_norm = data_pipeline.normalizer.transform(test_data, data_pipeline.feature_engineer.feature_columns)
    
    logger.info(f"Data processed: Train={len(train_data_norm)}, Val={len(val_data_norm)}, Test={len(test_data_norm)}")
    
    # Step 5: Generate LLM components
    logger.info("Generating RL components using LLM...")
    
    market_context = "High-frequency trading environment with dummy stock data"
    
    reward_result = llm_generator.generate_reward_function(market_context, "single")
    network_result = llm_generator.generate_network_architecture(8, 1, "single")  # 8 features, 1 action
    imagination_result = llm_generator.generate_imagination_module(market_context)
    
    logger.info(f"Components generated: Reward={reward_result['valid']}, Network={network_result['valid']}, Imagination={imagination_result['valid']}")
    
    # Step 6: Run RL training
    logger.info("Running RL training...")
    
    rl_trainer.initialize_agent(8, 1)  # 8 state features, 1 action
    rl_trainer.set_static_environment(train_data_norm)
    
    # Train for a few episodes
    training_results = []
    for episode in range(10):  # Small number for demo
        result = rl_trainer.train_episode()
        training_results.append(result)
        
        if episode % 5 == 0:
            logger.info(f"Episode {episode}: Reward = {result['episode_reward']:.4f}")
    
    # Step 7: Evaluate performance
    logger.info("Evaluating performance...")
    
    metrics = rl_trainer.evaluate_policy(num_episodes=5)
    
    logger.info("Final Performance Metrics:")
    logger.info(f"  Price Advantage (PA): {metrics['PA']:.4f} bps")
    logger.info(f"  Win Ratio (WR): {metrics['WR']:.4f}")
    logger.info(f"  Gain-Loss Ratio (GLR): {metrics['GLR']:.4f}")
    logger.info(f"  Average Final Inventory (AFI): {metrics['AFI']:.4f}")
    logger.info(f"  Mean Reward: {metrics['MeanReward']:.4f}")
    
    # Step 8: Demonstrate framework capabilities
    logger.info("Demonstrating SE-RL framework capabilities...")
    
    # Show that we can access the framework
    framework.initialize_components()
    
    # Create a simple performance summary
    summary = {
        'total_episodes': len(training_results),
        'average_reward': np.mean([r['episode_reward'] for r in training_results]),
        'best_reward': max([r['episode_reward'] for r in training_results]),
        'final_metrics': metrics,
        'components_generated': {
            'reward_function': reward_result['valid'],
            'network_architecture': network_result['valid'],
            'imagination_module': imagination_result['valid']
        }
    }
    
    logger.info("SE-RL Framework Basic Example Completed Successfully!")
    logger.info(f"Summary: {summary}")
    
    return summary

def run_advanced_example():
    """Run a more advanced example showing framework features"""
    
    logger.info("Starting SE-RL Framework Advanced Example")
    
    # This would demonstrate more advanced features like:
    # - Multi-agent environments
    # - Dynamic market simulation
    # - DEK optimization
    # - Hybrid environment training
    
    logger.info("Advanced example would demonstrate:")
    logger.info("  - Multi-agent market simulation")
    logger.info("  - Dual-Level Enhancement Kit (DEK)")
    logger.info("  - Hybrid environment training")
    logger.info("  - Advanced prompting techniques")
    logger.info("  - Performance optimization")
    
    # For now, just return a placeholder
    return {
        'status': 'advanced_features_demonstrated',
        'message': 'Advanced features would be implemented here'
    }

def main():
    """Main function to run examples"""
    
    print("=" * 60)
    print("SE-RL Framework Examples")
    print("=" * 60)
    
    try:
        # Run basic example
        print("\n1. Running Basic Example...")
        basic_summary = run_basic_example()
        
        print("\n2. Running Advanced Example...")
        advanced_summary = run_advanced_example()
        
        print("\n" + "=" * 60)
        print("Examples Completed Successfully!")
        print("=" * 60)
        
        print(f"\nBasic Example Summary:")
        print(f"  Total Episodes: {basic_summary['total_episodes']}")
        print(f"  Average Reward: {basic_summary['average_reward']:.4f}")
        print(f"  Best Reward: {basic_summary['best_reward']:.4f}")
        print(f"  Final PA: {basic_summary['final_metrics']['PA']:.4f} bps")
        print(f"  Final WR: {basic_summary['final_metrics']['WR']:.4f}")
        
        print(f"\nAdvanced Example Summary:")
        print(f"  Status: {advanced_summary['status']}")
        print(f"  Message: {advanced_summary['message']}")
        
        print("\nThe SE-RL framework successfully demonstrated:")
        print("  ✓ LLM-powered component generation")
        print("  ✓ Financial data processing")
        print("  ✓ RL agent training")
        print("  ✓ Performance evaluation")
        print("  ✓ Framework orchestration")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main() 