# SE-RL Framework: Self-Evolutional Reinforcement Learning for Financial Order Execution

A comprehensive implementation of the Self-Evolutional Reinforcement Learning (SE-RL) framework for automated RL algorithm design and optimization in financial order execution, as described in the research paper "Large Language Model (LLM) as an Excellent Reinforcement Learning Researcher in both Single Agent and Multi-Agent Scenarios".


## Table of Contents

1. [Overview](#overview)
2. [Key Innovations](#key-innovations)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Core Components](#core-components)
7. [Algorithm Details](#algorithm-details)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Experiments](#experiments)

## Overview

The SE-RL framework addresses two fundamental challenges in RL-based financial trading:

1. **Slow research pace** - Traditional RL methods evolve slowly compared to other AI domains
2. **Idealistic market assumptions** - Existing methods ignore the impact of orders on market dynamics

The framework uses Large Language Models (LLMs) to automatically design, train, and iteratively optimize RL algorithms through a **bi-level optimization architecture**:

- **Outer Loop**: LLM research capability evolution through Dual-Level Enhancement Kit (DEK)
- **Inner Loop**: Execution agent training in hybrid environments (static + dynamic)

## Key Innovations

### 1. LLM-Powered RL Design

Five specialized LLM modules for automated RL algorithm design:

| Module | Function | Description |
|--------|----------|-------------|
| **LLM4Reward** | Reward Function Design | Designs reward functions with domain knowledge |
| **LLM4Communication** | Multi-Agent Communication | Designs communication protocols for multi-agent coordination |
| **LLM4Agent** | Network Architecture | Designs neural network architectures for policies |
| **LLM4Profiling** | Agent Profiling | Creates profiles for heterogeneous agent behaviors |
| **LLM4Imagine** | Market Imagination | Generates synthetic market scenarios (50% mixed sampling) |

### 2. Dual-Level Enhancement Kit (DEK)

Two-level optimization for LLM improvement:

- **High-Level Enhancement (HLE)**: Prompt optimization with Macro-Micro refinement, In-Context Learning, and Cache Replay
- **Low-Level Enhancement (LLE)**: Weight optimization using Straight-Through Estimator (STE) and LoRA fine-tuning

### 3. Hybrid Environment Training

Combines static and dynamic environments with adaptive loss rebalancing:

```
L_rebalance = α * L_static + β * L_dynamic
```

Where α and β are dynamically adjusted based on relative losses.

### 4. Complete Limit Order Book (LOB) Simulation

Full LOB implementation with:
- Price-time priority matching
- Multi-agent order submission
- Midpoint price updates: P(t+1) = (p*_bid + p*_ask) / 2

## Architecture

```
SE-RL Framework
├── Outer Loop (LLM Evolution)
│   ├── Algorithm Generation (LLM4*)
│   ├── Performance Evaluation
│   └── Dual-Level Enhancement (DEK)
│       ├── High-Level: Prompt Optimization
│       └── Low-Level: STE + LoRA
│
└── Inner Loop (Agent Training)
    ├── Static Environment (Historical Data)
    ├── Dynamic Environment (Multi-Agent Simulation)
    │   ├── Market Makers
    │   ├── Informed Traders
    │   └── Noise Traders
    └── Hybrid Training with Adaptive Rebalancing
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (recommended for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/se-rl-framework.git
cd se-rl-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
transformers>=4.20.0
peft>=0.5.0  # For LoRA
gymnasium>=0.26.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Project Structure

```
se-rl-framework/
├── src/
│   └── se_rl/
│       ├── core/
│       │   ├── framework.py          # Main SE-RL framework
│       │   └── training_loop.py      # Complete training loop with EMA convergence
│       │
│       ├── environments/
│       │   ├── limit_order_book.py   # Complete LOB implementation
│       │   ├── execution_env.py      # MDP-based execution environment
│       │   ├── static_env.py         # Static (historical) environment
│       │   ├── dynamic_env.py        # Dynamic (multi-agent) environment
│       │   └── multi_agent_env.py    # Multi-agent coordination environment
│       │
│       ├── rl/
│       │   ├── ppo_agent.py          # PPO with GAE implementation
│       │   └── trainer.py            # RL training utilities
│       │
│       ├── llm/
│       │   ├── generator.py          # LLM component generator
│       │   ├── prompts.py            # Prompt templates for LLM4*
│       │   ├── imagination.py        # LLM4Imagine module (50% mixed sampling)
│       │   ├── low_level_enhancement.py  # STE and LoRA implementation
│       │   ├── code_validator.py     # Two-stage code validation
│       │   └── integration.py        # LLM API integration
│       │
│       ├── data/
│       │   └── pipeline.py           # Financial data pipeline
│       │
│       └── utils/
│           └── logger.py             # Logging utilities
│
├── configs/                          # Configuration files
├── scripts/                          # Training and evaluation scripts
├── tests/                            # Unit tests
└── README.md
```

## Core Components

### 1. Limit Order Book (`limit_order_book.py`)

Complete LOB implementation with price-time priority matching:

```python
from se_rl.environments.limit_order_book import LimitOrderBook, OrderSide, OrderType

# Initialize LOB
lob = LimitOrderBook(initial_price=100.0, tick_size=0.01)

# Submit orders
order, trades = lob.submit_order(
    agent_id=1,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=99.5,
    quantity=100,
    timestamp=1.0
)

# Get LOB state for RL agent
state_vector = lob.get_state_vector(depth_levels=5)
```

### 2. PPO Agent with GAE (`ppo_agent.py`)

Complete PPO implementation with Generalized Advantage Estimation:

```python
from se_rl.rl.ppo_agent import PPOAgent, PPOConfig

config = PPOConfig(
    actor_lr=3e-4,
    critic_lr=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=10
)

agent = PPOAgent(state_dim=32, action_dim=1, config=config)

# Training loop
action, log_prob, value = agent.select_action(state)
agent.store_transition(state, action, reward, done, log_prob, value)
update_stats = agent.update()  # PPO update with GAE
```

### 3. Execution Environment (`execution_env.py`)

MDP-based environment with proper state/action spaces:

**State Space:**
- LOB features (bid/ask prices, depths, imbalance)
- Private state: normalized remaining time (T-t)/T
- Private state: normalized inventory I_t/I_0

**Action Space:**
- Continuous action a ∈ [0, 1] representing execution ratio

```python
from se_rl.environments.execution_env import ExecutionEnvironment, ExecutionEnvConfig

config = ExecutionEnvConfig(
    initial_inventory=10000.0,
    total_time=240,  # 4-hour execution window
    transaction_cost=0.001
)

env = ExecutionEnvironment(data, config)
state = env.reset()
next_state, reward, done, info = env.step(action)
metrics = env.get_metrics()  # PA, WR, GLR, AFI
```

### 4. LLM4Imagine (`imagination.py`)

50% mixed sampling strategy for training data:

```python
from se_rl.llm.imagination import LLM4Imagine, ImaginationConfig

config = ImaginationConfig(
    imagination_ratio=0.5,  # 50% imagined data
    num_scenarios=10,
    scenario_length=100
)

imagine = LLM4Imagine(config, llm_interface)

# Generate mixed training batch
mixed_batch, metadata = imagine.get_mixed_batch(static_data, batch_size=256)
# metadata: {'static_ratio': 0.5, 'imagined_ratio': 0.5}
```

### 5. Low-Level Enhancement (`low_level_enhancement.py`)

STE and LoRA for LLM fine-tuning:

```python
from se_rl.llm.low_level_enhancement import LowLevelEnhancement, LoRAConfig

config = LoRAConfig(
    rank=16,
    alpha=32,
    target_modules=['q_proj', 'v_proj', 'layer_norm']
)

lle = LowLevelEnhancement(config)

# Apply LoRA to LLM
model_with_lora = lle.apply_lora(llm_model)

# Get trainable parameters
lora_params = lle.get_lora_parameters()
```

### 6. Training Loop (`training_loop.py`)

Complete training with EMA convergence:

```python
from se_rl.core.training_loop import SERLTrainingLoop, TrainingLoopConfig

config = TrainingLoopConfig(
    max_outer_iterations=50,
    max_inner_iterations=1000,
    convergence_tolerance=0.1,  # epsilon_tol
    ema_decay=0.9,
    initial_alpha=0.5,
    initial_beta=0.5
)

trainer = SERLTrainingLoop(config, static_env, dynamic_env, agent)
results = trainer.run()  # Returns training history and metrics
```

## Algorithm Details

### Algorithm 1: SE-RL Main Loop

```
Input: Historical data D_h, LLM model M, convergence threshold ε
Output: Optimal execution policy π*

1: Initialize instruction population I, performance buffer B
2: for j = 1 to J_max do
3:     // Step 1: Generate Algorithm
4:     A_j ← LLM4Design(M, I, B)  // LLM4Reward, LLM4Agent, etc.
5:
6:     // Step 2: Train in Hybrid Environment
7:     π_static ← Train(A_j, D_h)
8:     π_dynamic ← Train(A_j, D_dynamic)
9:     π_j ← HybridTrain(π_static, π_dynamic, α, β)
10:
11:    // Step 3: Evaluate
12:    PA_j ← Evaluate(π_j)
13:
14:    // Step 4: Dual-Level Enhancement
15:    M ← DEK_Enhance(M, PA_j, I, B)
16:
17:    // Step 5: Check Convergence (EMA smoothed)
18:    PA_ema ← EMA_Update(PA_j)
19:    if |PA_ema_j - PA_ema_{j-1}| < ε then
20:        break
21: return π_j
```

### Generalized Advantage Estimation (GAE)

Equation (3) from the paper:

```
A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### Adaptive Loss Rebalancing

Equation (5) from the paper:

```
L_rebalance = α * L_static + β * L_dynamic

α, β are adapted based on relative losses:
α ← α * L_static / (L_static + L_dynamic)
β ← β * L_dynamic / (L_static + L_dynamic)
```

### Convergence Condition

EMA-smoothed convergence check:

```
PA_ema_t = decay * PA_ema_{t-1} + (1-decay) * PA_t

Converged if: |PA_ema_j - PA_ema_{j-1}| < ε_tol
```

## Usage

### Basic Training

```python
from se_rl.core.framework import SERLFramework, SERLConfig

config = SERLConfig(
    dataset="csi100",
    llm_model_name="meta-llama/Llama-3.3-70B-Instruct",
    max_outer_iterations=50,
    max_inner_iterations=1000,
    convergence_epsilon=0.1
)

framework = SERLFramework(config)
results = framework.run_training()

print(f"Best PA: {results['final_metrics']['PA']:.4f} bps")
```

### Command Line

```bash
# Train on CSI100 dataset
python -m se_rl.main --dataset csi100 --epochs 50

# Train on NASDAQ100 with specific LLM
python -m se_rl.main --dataset nasdaq100 --llm meta-llama/Llama-3.3-70B-Instruct

# Evaluate trained model
python -m se_rl.main --mode eval --checkpoint path/to/model.pt
```

## Configuration

### SERLConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_outer_iterations` | 50 | Maximum outer loop iterations |
| `max_inner_iterations` | 1000 | Maximum inner loop steps |
| `convergence_epsilon` | 0.1 | Convergence threshold |
| `learning_rate` | 3e-4 | Learning rate for PPO |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clip ratio |
| `static_env_weight` | 0.5 | Initial α for static loss |
| `dynamic_env_weight` | 0.5 | Initial β for dynamic loss |
| `lora_rank` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |

## Evaluation Metrics

### Financial Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **PA** | (P_exec - P_vwap) / P_vwap × 10000 | Price Advantage (bps) |
| **PA-std** | std(PA) | Consistency of execution |
| **WR** | #wins / #trades | Win Ratio |
| **GLR** | mean(gains) / mean(losses) | Gain-Loss Ratio |
| **AFI** | mean(\|final_inv\| / initial_inv) | Average Final Inventory |

### Usage

```python
from se_rl.core.training_loop import FinancialMetricsCalculator

calc = FinancialMetricsCalculator()

metrics = calc.calculate_all_metrics({
    'execution_prices': exec_prices,
    'benchmark_prices': vwap_prices,
    'quantities': quantities,
    'returns': returns,
    'final_inventories': final_inv,
    'initial_inventories': initial_inv
})

print(f"PA: {metrics['PA']:.2f} bps")
print(f"WR: {metrics['WR']:.2%}")
print(f"GLR: {metrics['GLR']:.2f}")
print(f"AFI: {metrics['AFI']:.2%}")
```

## Experiments

### Datasets

- **CSI100**: Top 100 stocks from China Securities Index
- **NASDAQ100**: Top 100 NASDAQ stocks

### Results Summary

Based on paper experiments:

| Dataset | Method | PA (bps) | WR | GLR |
|---------|--------|----------|-----|-----|
| CSI100 | SE-RL | **3.57±0.08** | **0.62** | **1.45** |
| CSI100 | TWAP | 0.00 | 0.50 | 1.00 |
| CSI100 | PPO | 2.31±0.15 | 0.55 | 1.21 |
| NASDAQ100 | SE-RL | **2.89±0.10** | **0.59** | **1.38** |

### Running Experiments

```bash
# CSI100 experiment
python scripts/run_experiment.py --dataset csi100 --seeds 5

# NASDAQ100 experiment
python scripts/run_experiment.py --dataset nasdaq100 --seeds 5

# Ablation study (without DEK)
python scripts/run_experiment.py --dataset csi100 --no-dek
```

## Code Validation

The framework includes a two-stage code validation pipeline for LLM-generated code:

1. **Syntax Check**: AST parsing and validation
2. **Runtime Verification**: Safe sandbox execution

```python
from se_rl.llm.code_validator import CodeValidator

validator = CodeValidator()

# Validate reward function code
result = validator.validate(code, code_type='reward')
if result.is_valid:
    print("Code validation passed!")
else:
    print(f"Validation failed at {result.stage}: {result.error_message}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{serl2024,
  title={Large Language Model (LLM) as an Excellent Reinforcement Learning Researcher in both Single Agent and Multi-Agent Scenarios},
  author={Anonymous},
  journal={KDD},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work builds upon advances in LLM-based code generation and automated ML
- Financial data processing inspired by industry-standard execution algorithms
- Multi-agent simulation based on realistic market microstructure models
