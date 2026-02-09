"""
Test Suite for SE-RL Framework
=============================

Comprehensive tests for the Self-Evolutional Reinforcement Learning framework.
"""

import unittest
import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from se_rl_framework import SERLConfig, SERLFramework, PerformanceBuffer, InstructionPopulation, DualLevelEnhancementKit
from llm_generator import ComponentConfig, LLMComponentGenerator, CodeValidator
from financial_data_pipeline import DataConfig, FinancialDataPipeline, FeatureEngineer, DataNormalizer
from rl_trainer import TrainingConfig, RLTrainer, RLAgent, StaticEnvironment, FinancialMetrics

class TestSERLFramework(unittest.TestCase):
    """Test cases for the main SE-RL framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SERLConfig(
            convergence_epsilon=0.1,
            max_outer_iterations=5,
            device="cpu"
        )
        self.framework = SERLFramework(self.config)
    
    def test_config_initialization(self):
        """Test SERLConfig initialization"""
        config = SERLConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.convergence_epsilon, 0.1)
        self.assertEqual(config.max_outer_iterations, 50)
    
    def test_framework_initialization(self):
        """Test SERLFramework initialization"""
        self.assertIsNotNone(self.framework)
        self.assertIsNotNone(self.framework.dek)
        self.assertIsNotNone(self.framework.performance_buffer)
        self.assertIsNotNone(self.framework.instruction_population)
    
    def test_performance_buffer(self):
        """Test PerformanceBuffer functionality"""
        buffer = PerformanceBuffer()
        
        # Test adding performance
        metrics = {'PA': 4.25, 'WR': 0.99}
        buffer.add_performance(1, metrics, "test prompt", "test code")
        
        # Test getting recent performance
        recent = buffer.get_recent_performance(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]['metrics']['PA'], 4.25)
        
        # Test getting best performance
        best = buffer.get_best_performance()
        self.assertIsNotNone(best)
        self.assertEqual(best['metrics']['PA'], 4.25)
    
    def test_instruction_population(self):
        """Test InstructionPopulation functionality"""
        population = InstructionPopulation()
        
        # Test adding instruction
        population.add_instruction("test instruction", 4.25)
        
        # Test sampling historical instructions
        sampled = population.sample_historical_instructions(1)
        self.assertEqual(len(sampled), 1)
        self.assertIn("test instruction", sampled[0])
        
        # Test getting best instruction
        best = population.get_best_instruction()
        self.assertIsNotNone(best)
        self.assertIn("test instruction", best)

class TestLLMGenerator(unittest.TestCase):
    """Test cases for LLM component generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ComponentConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            temperature=0.7,
            max_retries=2
        )
        self.generator = LLMComponentGenerator(self.config)
    
    def test_config_initialization(self):
        """Test ComponentConfig initialization"""
        config = ComponentConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_retries, 3)
    
    def test_generator_initialization(self):
        """Test LLMComponentGenerator initialization"""
        self.assertIsNotNone(self.generator)
        self.assertIsNotNone(self.generator.validator)
    
    def test_reward_function_generation(self):
        """Test reward function generation"""
        market_context = "High-frequency trading environment"
        result = self.generator.generate_reward_function(market_context, "single")
        
        self.assertIsNotNone(result)
        self.assertIn('code', result)
        self.assertIn('valid', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'reward_function')
    
    def test_network_architecture_generation(self):
        """Test network architecture generation"""
        result = self.generator.generate_network_architecture(64, 4, "single")
        
        self.assertIsNotNone(result)
        self.assertIn('code', result)
        self.assertIn('valid', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'network_architecture')
    
    def test_imagination_module_generation(self):
        """Test imagination module generation"""
        market_context = "High-frequency trading environment"
        result = self.generator.generate_imagination_module(market_context)
        
        self.assertIsNotNone(result)
        self.assertIn('code', result)
        self.assertIn('valid', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'imagination_module')
    
    def test_code_validation(self):
        """Test code validation functionality"""
        validator = CodeValidator()
        
        # Test valid Python syntax
        valid_code = "def test_function():\n    return True"
        is_valid, message = validator.validate_python_syntax(valid_code)
        self.assertTrue(is_valid)
        
        # Test invalid Python syntax
        invalid_code = "def test_function():\n    return True\n    invalid syntax"
        is_valid, message = validator.validate_python_syntax(invalid_code)
        self.assertFalse(is_valid)
        
        # Test function signature validation
        code_with_function = "def reward_function(state, action):\n    return 0.0"
        is_valid, message = validator.validate_function_signature(code_with_function, 'reward_function')
        self.assertTrue(is_valid)

class TestDataPipeline(unittest.TestCase):
    """Test cases for financial data pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DataConfig(
            start_date="2020-01-01",
            end_date="2022-12-31",
            frequency="1d",
            window_size=10
        )
        self.pipeline = FinancialDataPipeline(self.config)
    
    def test_config_initialization(self):
        """Test DataConfig initialization"""
        config = DataConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.frequency, "1d")
        self.assertEqual(config.window_size, 20)
    
    def test_pipeline_initialization(self):
        """Test FinancialDataPipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.downloader)
        self.assertIsNotNone(self.pipeline.feature_engineer)
        self.assertIsNotNone(self.pipeline.normalizer)
    
    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        # Create dummy data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        feature_engineer = FeatureEngineer(self.config)
        result = feature_engineer.engineer_features(data)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.columns), len(data.columns))
        self.assertIn('returns', result.columns)
        self.assertIn('volatility', result.columns)
    
    def test_data_normalization(self):
        """Test data normalization functionality"""
        # Create dummy data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        normalizer = DataNormalizer(self.config)
        feature_columns = ['feature1', 'feature2']
        
        # Test fit_transform
        normalized_data = normalizer.fit_transform(data, feature_columns)
        self.assertIsNotNone(normalized_data)
        
        # Test transform
        new_data = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70]
        })
        transformed_data = normalizer.transform(new_data, feature_columns)
        self.assertIsNotNone(transformed_data)

class TestRLTrainer(unittest.TestCase):
    """Test cases for RL training system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=32,
            max_episodes=10,
            device="cpu"
        )
        self.trainer = RLTrainer(self.config)
    
    def test_config_initialization(self):
        """Test TrainingConfig initialization"""
        config = TrainingConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.batch_size, 64)
    
    def test_trainer_initialization(self):
        """Test RLTrainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.device, "cpu")
    
    def test_agent_initialization(self):
        """Test RLAgent initialization"""
        state_dim = 8
        action_dim = 1
        agent = RLAgent(state_dim, action_dim, self.config)
        
        self.assertIsNotNone(agent)
        self.assertEqual(agent.state_dim, state_dim)
        self.assertEqual(agent.action_dim, action_dim)
        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.critic)
    
    def test_agent_action_generation(self):
        """Test agent action generation"""
        state_dim = 8
        action_dim = 1
        agent = RLAgent(state_dim, action_dim, self.config)
        
        # Create dummy state
        state = torch.randn(1, state_dim)
        
        # Test action generation
        action, log_prob = agent.get_action(state, training=True)
        self.assertIsNotNone(action)
        self.assertIsNotNone(log_prob)
        self.assertEqual(action.shape, (1, action_dim))
    
    def test_static_environment(self):
        """Test static environment functionality"""
        # Create dummy data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'returns': [0.01, 0.01, 0.01, 0.01, 0.01],
            'volatility': [0.02, 0.02, 0.02, 0.02, 0.02],
            'rsi': [50, 50, 50, 50, 50],
            'macd': [0, 0, 0, 0, 0],
            'bb_position': [0.5, 0.5, 0.5, 0.5, 0.5],
            'volume_ratio_20': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
        
        env = StaticEnvironment(data, self.config)
        
        # Test reset
        state = env.reset()
        self.assertIsNotNone(state)
        self.assertIn('current_price', state)
        self.assertIn('cash', state)
        
        # Test step
        action = torch.tensor([0.1])  # Small buy order
        next_state, reward, done, info = env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIsNotNone(reward)
        self.assertIsInstance(done, bool)
        self.assertIsNotNone(info)
    
    def test_financial_metrics(self):
        """Test financial metrics calculation"""
        # Test PA calculation
        execution_prices = [100, 101, 102]
        vwap_prices = [100.5, 101.5, 102.5]
        pa = FinancialMetrics.calculate_pa(execution_prices, vwap_prices)
        self.assertIsInstance(pa, float)
        
        # Test WR calculation
        returns = [0.01, -0.01, 0.02, -0.005, 0.015]
        wr = FinancialMetrics.calculate_wr(returns)
        self.assertIsInstance(wr, float)
        self.assertGreaterEqual(wr, 0.0)
        self.assertLessEqual(wr, 1.0)
        
        # Test GLR calculation
        glr = FinancialMetrics.calculate_glr(returns)
        self.assertIsInstance(glr, float)
        
        # Test AFI calculation
        final_inventories = [0.1, -0.2, 0.05, -0.1]
        afi = FinancialMetrics.calculate_afi(final_inventories)
        self.assertIsInstance(afi, float)
        self.assertGreaterEqual(afi, 0.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy data for integration tests
        self.dummy_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100000, 1000000, 100),
            'returns': np.random.randn(100) * 0.02,
            'volatility': np.random.rand(100) * 0.05,
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.1,
            'bb_position': np.random.uniform(0, 1, 100),
            'volume_ratio_20': np.random.uniform(0.5, 2.0, 100)
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Initialize components
        serl_config = SERLConfig(device="cpu")
        component_config = ComponentConfig()
        data_config = DataConfig()
        training_config = TrainingConfig(device="cpu")
        
        framework = SERLFramework(serl_config)
        llm_generator = LLMComponentGenerator(component_config)
        data_pipeline = FinancialDataPipeline(data_config)
        rl_trainer = RLTrainer(training_config)
        
        # Process data
        feature_data = data_pipeline.feature_engineer.engineer_features(self.dummy_data)
        train_data = feature_data.iloc[:70]
        
        # Generate components
        market_context = "Test trading environment"
        reward_result = llm_generator.generate_reward_function(market_context, "single")
        network_result = llm_generator.generate_network_architecture(8, 1, "single")
        
        # Initialize and train agent
        rl_trainer.initialize_agent(8, 1)
        rl_trainer.set_static_environment(train_data)
        
        # Train for a few episodes
        for episode in range(3):
            result = rl_trainer.train_episode()
            self.assertIsNotNone(result)
            self.assertIn('episode_reward', result)
        
        # Evaluate performance
        metrics = rl_trainer.evaluate_policy(num_episodes=2)
        self.assertIsNotNone(metrics)
        self.assertIn('PA', metrics)
        self.assertIn('WR', metrics)
    
    def test_framework_initialization(self):
        """Test complete framework initialization"""
        config = SERLConfig(device="cpu")
        framework = SERLFramework(config)
        
        # Test component initialization
        framework.initialize_components()
        self.assertIsNotNone(framework.llm_model)
        self.assertIsNotNone(framework.static_env)
        self.assertIsNotNone(framework.dynamic_env)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSERLFramework))
    test_suite.addTest(unittest.makeSuite(TestLLMGenerator))
    test_suite.addTest(unittest.makeSuite(TestDataPipeline))
    test_suite.addTest(unittest.makeSuite(TestRLTrainer))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 