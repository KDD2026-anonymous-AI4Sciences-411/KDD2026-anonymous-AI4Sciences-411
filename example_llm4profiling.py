#!/usr/bin/env python3
"""
LLM4Profiling Example for SE-RL Framework
=========================================

This script demonstrates the production-ready LLM4Profiling functionality
for generating intelligent entity configurations in multi-agent trading systems.

Author: AI Research Engineer
Date: 2024
"""

import sys
import os
import logging
import json
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_component_generator import ComponentConfig, LLMComponentGenerator
from llm_integration import LLMFactory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_llm_configuration():
    """Setup LLM configuration for different providers"""
    
    # Configuration options for different LLM providers
    configs = {
        "openai": {
            "provider": "openai",
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "anthropic": {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "huggingface": {
            "provider": "huggingface",
            "model_name": "meta-llama/Llama-2-7b-hf",
            "api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "local": {
            "provider": "local",
            "model_name": "meta-llama/Llama-2-7b-hf",
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    # Try to find available configuration
    for provider, config in configs.items():
        if config["api_key"] or provider == "local":
            logger.info(f"Using {provider} LLM configuration")
            return config
    
    # Fallback to OpenAI with warning
    logger.warning("No API keys found, using fallback configuration")
    return configs["openai"]

def demonstrate_llm4profiling():
    """Demonstrate LLM4Profiling functionality"""
    
    logger.info("🚀 Starting LLM4Profiling Demonstration")
    
    # Setup LLM configuration
    llm_config = setup_llm_configuration()
    
    # Initialize LLM component generator
    component_config = ComponentConfig(
        model_name=llm_config["model_name"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        max_retries=3
    )
    
    generator = LLMComponentGenerator(component_config)
    
    # Define different market contexts and agent types
    market_contexts = [
        "High-frequency trading environment with CSI100 stocks",
        "Algorithmic trading in NASDAQ100 with market makers and institutional traders",
        "Cryptocurrency trading with DeFi protocols and arbitrage bots",
        "Options trading with market makers, institutional traders, and retail investors"
    ]
    
    agent_type_combinations = [
        ["market_maker", "informed_trader", "noise_trader"],
        ["institutional_trader", "retail_trader", "arbitrage_bot"],
        ["liquidity_provider", "momentum_trader", "mean_reversion_trader"],
        ["market_maker", "informed_trader", "noise_trader", "arbitrage_bot"]
    ]
    
    results = []
    
    for i, (market_context, agent_types) in enumerate(zip(market_contexts, agent_type_combinations)):
        logger.info(f"\n📊 Scenario {i+1}: {market_context}")
        logger.info(f"Agent Types: {', '.join(agent_types)}")
        
        try:
            # Generate LLM4Profiling configuration
            profiling_result = generator.generate_llm4profiling(agent_types, market_context)
            
            if profiling_result['valid']:
                logger.info("✅ LLM4Profiling generation successful")
                logger.info(f"Code length: {len(profiling_result['code'])} characters")
                
                # Save the generated code
                filename = f"llm4profiling_scenario_{i+1}.py"
                with open(filename, 'w') as f:
                    f.write(profiling_result['code'])
                logger.info(f"💾 Saved to: {filename}")
                
                # Parse and analyze the generated profiles
                analyze_generated_profiles(profiling_result['code'], agent_types)
                
                results.append({
                    'scenario': i+1,
                    'market_context': market_context,
                    'agent_types': agent_types,
                    'success': True,
                    'filename': filename,
                    'code_length': len(profiling_result['code'])
                })
            else:
                logger.error(f"❌ LLM4Profiling generation failed: {profiling_result.get('validation_message', 'Unknown error')}")
                results.append({
                    'scenario': i+1,
                    'market_context': market_context,
                    'agent_types': agent_types,
                    'success': False,
                    'error': profiling_result.get('validation_message', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"❌ Error in scenario {i+1}: {str(e)}")
            results.append({
                'scenario': i+1,
                'market_context': market_context,
                'agent_types': agent_types,
                'success': False,
                'error': str(e)
            })
    
    # Generate summary report
    generate_summary_report(results)
    
    return results

def analyze_generated_profiles(code: str, agent_types: List[str]):
    """Analyze the generated LLM4Profiling code"""
    
    logger.info("🔍 Analyzing generated profiles...")
    
    # Check for key components
    analysis = {
        'has_class_definition': 'class' in code,
        'has_init_method': '__init__' in code,
        'has_get_agent_profile': 'get_agent_profile' in code,
        'has_interaction_patterns': 'interaction' in code.lower(),
        'agent_types_covered': []
    }
    
    # Check if all agent types are covered
    for agent_type in agent_types:
        if agent_type in code:
            analysis['agent_types_covered'].append(agent_type)
    
    # Log analysis results
    logger.info(f"  - Class definition: {'✅' if analysis['has_class_definition'] else '❌'}")
    logger.info(f"  - Init method: {'✅' if analysis['has_init_method'] else '❌'}")
    logger.info(f"  - Get agent profile method: {'✅' if analysis['has_get_agent_profile'] else '❌'}")
    logger.info(f"  - Interaction patterns: {'✅' if analysis['has_interaction_patterns'] else '❌'}")
    logger.info(f"  - Agent types covered: {len(analysis['agent_types_covered'])}/{len(agent_types)}")
    
    if analysis['agent_types_covered']:
        logger.info(f"    Covered: {', '.join(analysis['agent_types_covered'])}")

def generate_summary_report(results: List[Dict[str, Any]]):
    """Generate a summary report of the LLM4Profiling demonstration"""
    
    logger.info("\n📋 LLM4Profiling Demonstration Summary")
    logger.info("=" * 50)
    
    total_scenarios = len(results)
    successful_scenarios = sum(1 for r in results if r['success'])
    failed_scenarios = total_scenarios - successful_scenarios
    
    logger.info(f"Total scenarios: {total_scenarios}")
    logger.info(f"Successful: {successful_scenarios} ✅")
    logger.info(f"Failed: {failed_scenarios} ❌")
    logger.info(f"Success rate: {successful_scenarios/total_scenarios*100:.1f}%")
    
    if successful_scenarios > 0:
        avg_code_length = sum(r['code_length'] for r in results if r['success']) / successful_scenarios
        logger.info(f"Average code length: {avg_code_length:.0f} characters")
    
    # List successful scenarios
    if successful_scenarios > 0:
        logger.info("\n✅ Successful scenarios:")
        for result in results:
            if result['success']:
                logger.info(f"  - Scenario {result['scenario']}: {result['market_context'][:50]}...")
                logger.info(f"    Agent types: {', '.join(result['agent_types'])}")
                logger.info(f"    File: {result['filename']}")
    
    # List failed scenarios
    if failed_scenarios > 0:
        logger.info("\n❌ Failed scenarios:")
        for result in results:
            if not result['success']:
                logger.info(f"  - Scenario {result['scenario']}: {result['market_context'][:50]}...")
                logger.info(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Save detailed report
    report_filename = "llm4profiling_report.json"
    with open(report_filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Detailed report saved to: {report_filename}")

def demonstrate_multi_component_generation():
    """Demonstrate generation of multiple components together"""
    
    logger.info("\n🔧 Multi-Component Generation Demonstration")
    
    # Setup
    llm_config = setup_llm_configuration()
    component_config = ComponentConfig(
        model_name=llm_config["model_name"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"]
    )
    generator = LLMComponentGenerator(component_config)
    
    # Generate complete multi-agent system
    market_context = "High-frequency trading environment with CSI100 stocks"
    agent_types = ["market_maker", "informed_trader", "noise_trader"]
    
    logger.info("Generating complete multi-agent system components...")
    
    components = {}
    
    # 1. Generate reward functions for each agent type
    logger.info("1. Generating reward functions...")
    for agent_type in agent_types:
        reward_result = generator.generate_reward_function(market_context, agent_type)
        components[f'reward_{agent_type}'] = reward_result
        logger.info(f"   ✅ {agent_type} reward function generated")
    
    # 2. Generate network architectures
    logger.info("2. Generating network architectures...")
    for agent_type in agent_types:
        network_result = generator.generate_network_architecture(64, 4, agent_type)
        components[f'network_{agent_type}'] = network_result
        logger.info(f"   ✅ {agent_type} network architecture generated")
    
    # 3. Generate communication protocol
    logger.info("3. Generating communication protocol...")
    communication_result = generator.generate_multi_agent_communication(len(agent_types), agent_types)
    components['communication'] = communication_result
    logger.info("   ✅ Communication protocol generated")
    
    # 4. Generate LLM4Profiling
    logger.info("4. Generating LLM4Profiling...")
    profiling_result = generator.generate_llm4profiling(agent_types, market_context)
    components['profiling'] = profiling_result
    logger.info("   ✅ LLM4Profiling generated")
    
    # Save all components
    logger.info("5. Saving all components...")
    for name, result in components.items():
        if result['valid']:
            filename = f"component_{name}.py"
            with open(filename, 'w') as f:
                f.write(result['code'])
            logger.info(f"   💾 Saved: {filename}")
    
    logger.info("✅ Multi-component generation completed!")
    return components

def main():
    """Main demonstration function"""
    
    logger.info("🎯 SE-RL Framework LLM4Profiling Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demonstrate LLM4Profiling
        profiling_results = demonstrate_llm4profiling()
        
        # Demonstrate multi-component generation
        multi_components = demonstrate_multi_component_generation()
        
        logger.info("\n🎉 Demonstration completed successfully!")
        logger.info("Check the generated files for the complete multi-agent system components.")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 