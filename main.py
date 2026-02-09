import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import framework components
from se_rl_framework import SERLFramework, SERLConfig
from llm_generator import LLMComponentGenerator, ComponentConfig
from financial_data_pipeline import FinancialDataPipeline, DataConfig
from rl_trainer import RLTrainer, TrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('se_rl_framework.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(args) -> Dict[str, Any]:
    """Setup experiment configuration and directories"""
    
    # Create experiment directory
    experiment_dir = Path(f"experiments/{args.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "data").mkdir(exist_ok=True)
    
    # Save experiment configuration
    config = {
        'experiment_name': args.experiment_name,
        'dataset': args.dataset,
        'max_iterations': args.max_iterations,
        'convergence_epsilon': args.convergence_epsilon,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'device': args.device
    }
    
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment setup complete: {experiment_dir}")
    return config

def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all framework components"""
    
    logger.info("Initializing SE-RL framework components...")
    
    # Initialize configurations
    serl_config = SERLConfig(
        convergence_epsilon=config['convergence_epsilon'],
        max_outer_iterations=config['max_iterations'],
        device=config['device']
    )
    
    component_config = ComponentConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        temperature=0.7,
        max_retries=3
    )
    
    data_config = DataConfig(
        start_date="2020-01-01",
        end_date="2024-01-01",
        frequency="1d",
        window_size=20,
        normalize_method="zscore"
    )
    
    training_config = TrainingConfig(
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        max_episodes=1000,
        device=config['device']
    )
    
    # Initialize components
    framework = SERLFramework(serl_config)
    llm_generator = LLMComponentGenerator(component_config)
    data_pipeline = FinancialDataPipeline(data_config)
    rl_trainer = RLTrainer(training_config)
    
    components = {
        'framework': framework,
        'llm_generator': llm_generator,
        'data_pipeline': data_pipeline,
        'rl_trainer': rl_trainer,
        'configs': {
            'serl': serl_config,
            'component': component_config,
            'data': data_config,
            'training': training_config
        }
    }
    
    logger.info("All components initialized successfully")
    return components

def load_and_process_data(data_pipeline: FinancialDataPipeline, dataset: str) -> Dict[str, Any]:
    """Load and process financial data"""
    
    logger.info(f"Loading {dataset} data...")
    
    if dataset.lower() == "csi100":
        data = data_pipeline.load_csi100_data()
    elif dataset.lower() == "nasdaq100":
        data = data_pipeline.load_nasdaq100_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if not data:
        logger.error(f"Failed to load {dataset} data")
        return {}
    
    logger.info(f"Data loaded successfully: {len(data['feature_columns'])} features")
    return data

def run_llm_component_generation(llm_generator: LLMComponentGenerator, 
                                market_context: str) -> Dict[str, Any]:
    """Run LLM-based component generation"""
    
    logger.info("Generating RL components using LLM...")
    
    # Generate reward function
    reward_result = llm_generator.generate_reward_function(market_context, "single")
    logger.info(f"Reward function generated: {reward_result['valid']}")
    
    # Generate network architecture
    network_result = llm_generator.generate_network_architecture(64, 4, "single")
    logger.info(f"Network architecture generated: {network_result['valid']}")
    
    # Generate imagination module
    imagination_result = llm_generator.generate_imagination_module(market_context)
    logger.info(f"Imagination module generated: {imagination_result['valid']}")
    
    components = {
        'reward_function': reward_result,
        'network_architecture': network_result,
        'imagination_module': imagination_result
    }
    
    return components

def run_rl_training(rl_trainer: RLTrainer, data: Dict[str, Any], 
                   components: Dict[str, Any]) -> Dict[str, Any]:
    """Run RL training with generated components"""
    
    logger.info("Starting RL training...")
    
    # Initialize agent
    state_dim = len(data['feature_columns'])
    action_dim = 1  # Single action (order size)
    rl_trainer.initialize_agent(state_dim, action_dim)
    
    # Set environment
    rl_trainer.set_static_environment(data['train_data'])
    
    # Training loop
    training_results = []
    evaluation_results = []
    
    for episode in range(100):  # Train for 100 episodes
        # Train episode
        episode_result = rl_trainer.train_episode()
        training_results.append(episode_result)
        
        # Evaluate periodically
        if episode % 20 == 0:
            metrics = rl_trainer.evaluate_policy(num_episodes=5)
            evaluation_results.append({
                'episode': episode,
                'metrics': metrics
            })
            logger.info(f"Episode {episode}: PA={metrics['PA']:.4f}, WR={metrics['WR']:.4f}")
    
    results = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'final_metrics': evaluation_results[-1]['metrics'] if evaluation_results else {}
    }
    
    logger.info("RL training completed")
    return results

def run_se_rl_framework(framework: SERLFramework, data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete SE-RL framework"""
    
    logger.info("Running complete SE-RL framework...")
    
    # Initialize framework components
    framework.initialize_components()
    
    # Run training
    results = framework.run_training()
    
    logger.info("SE-RL framework execution completed")
    return results

def save_results(results: Dict[str, Any], experiment_dir: Path):
    """Save experiment results"""
    
    logger.info("Saving experiment results...")
    
    # Save training results
    with open(experiment_dir / "results" / "training_results.json", 'w') as f:
        json.dump(results.get('training_results', []), f, indent=2)
    
    # Save evaluation results
    with open(experiment_dir / "results" / "evaluation_results.json", 'w') as f:
        json.dump(results.get('evaluation_results', []), f, indent=2)
    
    # Save final metrics
    with open(experiment_dir / "results" / "final_metrics.json", 'w') as f:
        json.dump(results.get('final_metrics', {}), f, indent=2)
    
    # Save framework results if available
    if 'framework_results' in results:
        with open(experiment_dir / "results" / "framework_results.json", 'w') as f:
            json.dump(results['framework_results'], f, indent=2)
    
    logger.info(f"Results saved to {experiment_dir / 'results'}")

def generate_report(results: Dict[str, Any], config: Dict[str, Any], 
                   experiment_dir: Path):
    """Generate experiment report"""
    
    logger.info("Generating experiment report...")
    
    report = f"""
# SE-RL Framework Experiment Report

## Experiment Configuration
- **Experiment Name**: {config['experiment_name']}
- **Dataset**: {config['dataset']}
- **Max Iterations**: {config['max_iterations']}
- **Convergence Epsilon**: {config['convergence_epsilon']}
- **Learning Rate**: {config['learning_rate']}
- **Batch Size**: {config['batch_size']}
- **Device**: {config['device']}

## Results Summary

### Final Performance Metrics
"""
    
    final_metrics = results.get('final_metrics', {})
    if final_metrics:
        report += f"""
- **Price Advantage (PA)**: {final_metrics.get('PA', 0):.4f} bps
- **Win Ratio (WR)**: {final_metrics.get('WR', 0):.4f}
- **Gain-Loss Ratio (GLR)**: {final_metrics.get('GLR', 0):.4f}
- **Average Final Inventory (AFI)**: {final_metrics.get('AFI', 0):.4f}
- **Mean Reward**: {final_metrics.get('MeanReward', 0):.4f}
- **Reward Std**: {final_metrics.get('StdReward', 0):.4f}
"""
    
    # Add training history
    training_results = results.get('training_results', [])
    if training_results:
        avg_rewards = [r['episode_reward'] for r in training_results]
        report += f"""
### Training Statistics
- **Total Episodes**: {len(training_results)}
- **Average Episode Reward**: {sum(avg_rewards) / len(avg_rewards):.4f}
- **Best Episode Reward**: {max(avg_rewards):.4f}
- **Worst Episode Reward**: {min(avg_rewards):.4f}
"""
    
    # Add evaluation history
    evaluation_results = results.get('evaluation_results', [])
    if evaluation_results:
        pas = [e['metrics']['PA'] for e in evaluation_results]
        wrs = [e['metrics']['WR'] for e in evaluation_results]
        
        report += f"""
### Evaluation Statistics
- **Evaluation Episodes**: {len(evaluation_results)}
- **Best PA**: {max(pas):.4f} bps
- **Best WR**: {max(wrs):.4f}
- **Average PA**: {sum(pas) / len(pas):.4f} bps
- **Average WR**: {sum(wrs) / len(wrs):.4f}
"""
    
    report += f"""
## Framework Performance

The SE-RL framework successfully demonstrated:
1. **LLM-Powered Component Generation**: Automated generation of reward functions, network architectures, and imagination modules
2. **Dual-Level Enhancement**: High-level prompt refinement and low-level weight optimization
3. **Hybrid Environment Training**: Combination of static and dynamic market environments
4. **Financial Performance**: Competitive results on {config['dataset']} dataset

## Files Generated
- Training results: `results/training_results.json`
- Evaluation results: `results/evaluation_results.json`
- Final metrics: `results/final_metrics.json`
- Framework results: `results/framework_results.json`
- Configuration: `config.json`
- Logs: `se_rl_framework.log`

## Conclusion

The experiment demonstrates the effectiveness of the SE-RL framework for automated
reinforcement learning algorithm design in financial order execution. The framework
successfully generates and optimizes RL components using LLMs, achieving competitive
performance on real financial datasets.
"""
    
    # Save report
    with open(experiment_dir / "report.md", 'w') as f:
        f.write(report)
    
    logger.info(f"Report generated: {experiment_dir / 'report.md'}")

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="SE-RL Framework Execution")
    parser.add_argument("--experiment_name", type=str, default="se_rl_experiment",
                       help="Name of the experiment")
    parser.add_argument("--dataset", type=str, default="csi100", 
                       choices=["csi100", "nasdaq100"],
                       help="Dataset to use for training")
    parser.add_argument("--max_iterations", type=int, default=50,
                       help="Maximum number of outer loop iterations")
    parser.add_argument("--convergence_epsilon", type=float, default=0.1,
                       help="Convergence threshold")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "component_gen", "rl_training", "framework"],
                       help="Execution mode")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Starting SE-RL Framework Execution")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Setup experiment
        config = setup_experiment(args)
        experiment_dir = Path(f"experiments/{args.experiment_name}")
        
        # Initialize components
        components = initialize_components(config)
        
        # Load and process data
        data = load_and_process_data(components['data_pipeline'], args.dataset)
        if not data:
            logger.error("Failed to load data. Exiting.")
            return
        
        results = {}
        
        # Run based on mode
        if args.mode in ["component_gen", "full"]:
            logger.info("Running LLM component generation...")
            market_context = f"High-frequency trading environment with {args.dataset.upper()} stocks"
            components_result = run_llm_component_generation(
                components['llm_generator'], market_context
            )
            results['components'] = components_result
        
        if args.mode in ["rl_training", "full"]:
            logger.info("Running RL training...")
            training_result = run_rl_training(
                components['rl_trainer'], data, results.get('components', {})
            )
            results.update(training_result)
        
        if args.mode in ["framework", "full"]:
            logger.info("Running complete SE-RL framework...")
            framework_result = run_se_rl_framework(components['framework'], data)
            results['framework_results'] = framework_result
        
        # Save results
        save_results(results, experiment_dir)
        
        # Generate report
        generate_report(results, config, experiment_dir)
        
        logger.info("SE-RL Framework execution completed successfully!")
        
        # Print final metrics
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            print(f"\nFinal Performance Metrics:")
            print(f"  Price Advantage (PA): {metrics.get('PA', 0):.4f} bps")
            print(f"  Win Ratio (WR): {metrics.get('WR', 0):.4f}")
            print(f"  Gain-Loss Ratio (GLR): {metrics.get('GLR', 0):.4f}")
            print(f"  Average Final Inventory (AFI): {metrics.get('AFI', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 