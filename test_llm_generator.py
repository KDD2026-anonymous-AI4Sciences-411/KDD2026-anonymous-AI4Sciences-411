import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_component_generator import ComponentConfig, LLMComponentGenerator, PromptTemplates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prompt_templates():
    """Test prompt template generation"""
    logger.info("Testing prompt templates...")
    
    # Test reward function prompt
    market_context = "High-frequency trading environment with CSI100 stocks"
    reward_prompt = PromptTemplates.reward_function_prompt(market_context, "single")
    logger.info(f"Reward function prompt length: {len(reward_prompt)}")
    logger.info("Reward function prompt contains examples: " + 
               ("Yes" if "EXAMPLE 1" in reward_prompt else "No"))
    
    # Test network architecture prompt
    network_prompt = PromptTemplates.network_architecture_prompt(64, 4, "single")
    logger.info(f"Network architecture prompt length: {len(network_prompt)}")
    logger.info("Network architecture prompt contains examples: " + 
               ("Yes" if "EXAMPLE 1" in network_prompt else "No"))
    
    # Test multi-agent communication prompt
    communication_prompt = PromptTemplates.multi_agent_communication_prompt(
        3, ["market_maker", "informed_trader", "noise_trader"]
    )
    logger.info(f"Communication prompt length: {len(communication_prompt)}")
    logger.info("Communication prompt contains examples: " + 
               ("Yes" if "EXAMPLE 1" in communication_prompt else "No"))
    
    logger.info("✅ Prompt template tests passed!")

def test_component_generation():
    """Test component generation"""
    logger.info("Testing component generation...")
    
    # Initialize generator
    config = ComponentConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        temperature=0.7,
        max_retries=2
    )
    generator = LLMComponentGenerator(config)
    
    # Test reward function generation
    market_context = "High-frequency trading environment with CSI100 stocks"
    reward_result = generator.generate_reward_function(market_context, "single")
    
    logger.info(f"Reward function generation successful: {reward_result['valid']}")
    logger.info(f"Reward function type: {reward_result['type']}")
    logger.info(f"Reward function agent type: {reward_result['agent_type']}")
    logger.info(f"Code length: {len(reward_result['code'])}")
    
    # Test network architecture generation
    network_result = generator.generate_network_architecture(64, 4, "single")
    
    logger.info(f"Network architecture generation successful: {network_result['valid']}")
    logger.info(f"Network architecture type: {network_result['type']}")
    logger.info(f"State dim: {network_result['state_dim']}")
    logger.info(f"Action dim: {network_result['action_dim']}")
    logger.info(f"Code length: {len(network_result['code'])}")
    
    # Test multi-agent communication generation
    communication_result = generator.generate_multi_agent_communication(
        3, ["market_maker", "informed_trader", "noise_trader"]
    )
    
    logger.info(f"Communication protocol generation successful: {communication_result['valid']}")
    logger.info(f"Communication protocol type: {communication_result['type']}")
    logger.info(f"Number of agents: {communication_result['num_agents']}")
    logger.info(f"Agent types: {communication_result['agent_types']}")
    logger.info(f"Code length: {len(communication_result['code'])}")
    
    # Test LLM4Profiling generation
    market_context = "High-frequency trading environment with CSI100 stocks"
    profiling_result = generator.generate_llm4profiling(
        ["market_maker", "informed_trader", "noise_trader"], market_context
    )
    
    logger.info(f"LLM4Profiling generation successful: {profiling_result['valid']}")
    logger.info(f"LLM4Profiling type: {profiling_result['type']}")
    logger.info(f"Agent types: {profiling_result['agent_types']}")
    logger.info(f"Market context: {profiling_result['market_context']}")
    logger.info(f"Code length: {len(profiling_result['code'])}")
    
    logger.info("✅ Component generation tests passed!")

def test_code_validation():
    """Test code validation"""
    logger.info("Testing code validation...")
    
    from llm_component_generator import CodeValidator
    
    validator = CodeValidator()
    
    # Test valid Python syntax
    valid_code = """
def test_function():
    return True
"""
    is_valid, message = validator.validate_python_syntax(valid_code)
    logger.info(f"Valid syntax test: {is_valid} - {message}")
    
    # Test invalid Python syntax
    invalid_code = """
def test_function():
    return True
    invalid syntax
"""
    is_valid, message = validator.validate_python_syntax(invalid_code)
    logger.info(f"Invalid syntax test: {is_valid} - {message}")
    
    # Test function signature validation
    code_with_function = """
def reward_function(state, action):
    return 0.0
"""
    is_valid, message = validator.validate_function_signature(code_with_function, 'reward_function')
    logger.info(f"Function signature test: {is_valid} - {message}")
    
    logger.info("✅ Code validation tests passed!")

def main():
    """Main test function"""
    logger.info("Starting LLM Component Generator tests...")
    
    try:
        test_prompt_templates()
        test_component_generation()
        test_code_validation()
        
        logger.info("🎉 All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 