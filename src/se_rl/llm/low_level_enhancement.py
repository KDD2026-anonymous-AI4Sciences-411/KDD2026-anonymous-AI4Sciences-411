"""
Low-Level Enhancement (LLE) for SE-RL Framework
================================================

This module implements the low-level weight optimization components
of the Dual-Level Enhancement Kit (DEK) as described in the paper:

1. Straight-Through Estimator (STE) - For non-differentiable code generation
2. LoRA (Low-Rank Adaptation) - For efficient LLM fine-tuning

The LLE targets specific layers:
- Add & Normalization layers
- Positional encoding layers

Author: AI Research Engineer
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    rank: int = 16  # LoRA rank (r)
    alpha: int = 32  # LoRA scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Modules to apply LoRA

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                'add_norm',
                'layer_norm',
                'positional_encoding',
                'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention layers
            ]


class StraightThroughEstimator(Function):
    """
    Straight-Through Estimator (STE) for non-differentiable operations.

    Enables gradient flow through non-differentiable code generation
    by using a differentiable surrogate in the backward pass.

    Forward: Uses the actual non-differentiable operation
    Backward: Uses identity or surrogate gradient
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Forward pass with hard thresholding (non-differentiable).

        Args:
            input_tensor: Input logits/probabilities
            threshold: Threshold for binarization

        Returns:
            Binary output tensor
        """
        ctx.save_for_backward(input_tensor)
        ctx.threshold = threshold

        # Hard thresholding (non-differentiable)
        output = (input_tensor > threshold).float()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass using straight-through gradient.

        The gradient flows through as if the forward pass was identity.
        """
        input_tensor, = ctx.saved_tensors

        # Straight-through: gradient passes through unchanged
        # Optionally, we can use a surrogate gradient
        grad_input = grad_output.clone()

        # Apply gradient scaling based on distance from threshold
        # This helps with gradient magnitude
        scale = torch.exp(-torch.abs(input_tensor - ctx.threshold))
        grad_input = grad_input * scale

        return grad_input, None


class STELayer(nn.Module):
    """
    Straight-Through Estimator layer for code generation.

    Wraps the STE function in a module for easy integration.
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0):
        super(STELayer, self).__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, x: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (logits)
            hard: If True, use hard thresholding with STE
                  If False, use soft (differentiable) approximation

        Returns:
            Processed tensor
        """
        if hard:
            # Apply temperature scaling
            x_scaled = x / self.temperature
            return StraightThroughEstimator.apply(torch.sigmoid(x_scaled), self.threshold)
        else:
            # Soft approximation (fully differentiable)
            return torch.sigmoid(x / self.temperature)


class GumbelSoftmaxSTE(nn.Module):
    """
    Gumbel-Softmax with Straight-Through gradient.

    Enables differentiable sampling from categorical distributions,
    useful for discrete code choices.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = True):
        super(GumbelSoftmaxSTE, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Gumbel-Softmax.

        Args:
            logits: Unnormalized log probabilities [batch, num_classes]

        Returns:
            Sampled one-hot vectors (or soft approximation)
        """
        # Sample from Gumbel distribution
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Add Gumbel noise and apply temperature
        y_soft = F.softmax((logits + gumbels) / self.temperature, dim=-1)

        if self.hard:
            # Straight-through: forward uses hard, backward uses soft
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.

    Implements: h = Wx + (BA)x * (alpha / r)

    Where:
    - W: Original frozen weights
    - B: Low-rank matrix [out_features, r]
    - A: Low-rank matrix [r, in_features]
    - alpha: Scaling factor
    - r: Rank
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 16,
                 alpha: int = 32, dropout: float = 0.1):
        super(LoRALinear, self).__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original frozen weights
        self.weight = nn.Parameter(original_layer.weight.data.clone(), requires_grad=False)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize LoRA matrices
        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights"""
        # A is initialized with Gaussian, B is initialized with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        h = Wx + b + (BA)x * scaling
        """
        # Original linear transformation
        result = F.linear(x, self.weight, self.bias)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        result = result + lora_out

        return result

    def merge_weights(self):
        """Merge LoRA weights into the original weights"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def unmerge_weights(self):
        """Unmerge LoRA weights from the original weights"""
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling


class LoRALayerNorm(nn.Module):
    """
    LoRA-adapted Layer Normalization.

    Applies low-rank adaptation to the scale (gamma) and shift (beta) parameters.
    """

    def __init__(self, original_layer: nn.LayerNorm, rank: int = 8, alpha: int = 16):
        super(LoRALayerNorm, self).__init__()

        self.normalized_shape = original_layer.normalized_shape
        self.eps = original_layer.eps
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original parameters (frozen)
        self.weight = nn.Parameter(original_layer.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)

        # LoRA adaptation for gamma (scale)
        self.lora_gamma_A = nn.Parameter(torch.zeros(rank))
        self.lora_gamma_B = nn.Parameter(torch.zeros(self.normalized_shape[0], rank))

        # LoRA adaptation for beta (shift)
        self.lora_beta_A = nn.Parameter(torch.zeros(rank))
        self.lora_beta_B = nn.Parameter(torch.zeros(self.normalized_shape[0], rank))

        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights"""
        nn.init.normal_(self.lora_gamma_A, std=0.01)
        nn.init.zeros_(self.lora_gamma_B)
        nn.init.normal_(self.lora_beta_A, std=0.01)
        nn.init.zeros_(self.lora_beta_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA-adapted layer norm"""
        # Compute adapted gamma and beta
        gamma_adapt = (self.lora_gamma_B @ self.lora_gamma_A) * self.scaling
        beta_adapt = (self.lora_beta_B @ self.lora_beta_A) * self.scaling

        gamma = self.weight + gamma_adapt
        beta = self.bias + beta_adapt

        # Apply layer normalization
        return F.layer_norm(x, self.normalized_shape, gamma, beta, self.eps)


class LoRAPositionalEncoding(nn.Module):
    """
    LoRA-adapted Positional Encoding.

    Allows fine-tuning of positional representations through low-rank adaptation.
    """

    def __init__(self, d_model: int, max_len: int = 5000, rank: int = 8, alpha: int = 16):
        super(LoRAPositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original positional encoding (frozen)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

        # LoRA adaptation matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, d_model))
        self.lora_B = nn.Parameter(torch.zeros(max_len, rank))

        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights"""
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA-adapted positional encoding"""
        seq_len = x.size(1)

        # Original PE
        pe = self.pe[:, :seq_len, :]

        # LoRA adaptation
        lora_pe = (self.lora_B[:seq_len, :] @ self.lora_A).unsqueeze(0) * self.scaling

        return x + pe + lora_pe


class LowLevelEnhancement:
    """
    Low-Level Enhancement (LLE) manager for DEK.

    Handles the application of STE and LoRA to the LLM model.
    """

    def __init__(self, config: LoRAConfig):
        self.config = config
        self.lora_layers: Dict[str, nn.Module] = {}
        self.ste_layer = STELayer()
        self.gumbel_ste = GumbelSoftmaxSTE()

    def apply_lora(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA adaptation to target modules in the model.

        Args:
            model: The LLM model to adapt

        Returns:
            Model with LoRA layers applied
        """
        for name, module in model.named_modules():
            # Check if this module should be adapted
            should_adapt = any(target in name.lower() for target in self.config.target_modules)

            if should_adapt:
                if isinstance(module, nn.Linear):
                    # Replace with LoRA Linear
                    lora_layer = LoRALinear(
                        module,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout
                    )
                    self._replace_module(model, name, lora_layer)
                    self.lora_layers[name] = lora_layer
                    logger.info(f"Applied LoRA to Linear layer: {name}")

                elif isinstance(module, nn.LayerNorm):
                    # Replace with LoRA LayerNorm
                    lora_layer = LoRALayerNorm(
                        module,
                        rank=self.config.rank // 2,  # Smaller rank for LayerNorm
                        alpha=self.config.alpha // 2
                    )
                    self._replace_module(model, name, lora_layer)
                    self.lora_layers[name] = lora_layer
                    logger.info(f"Applied LoRA to LayerNorm layer: {name}")

        logger.info(f"Total LoRA layers applied: {len(self.lora_layers)}")
        return model

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model by name"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization"""
        params = []
        for layer in self.lora_layers.values():
            if isinstance(layer, LoRALinear):
                params.extend([layer.lora_A, layer.lora_B])
            elif isinstance(layer, LoRALayerNorm):
                params.extend([
                    layer.lora_gamma_A, layer.lora_gamma_B,
                    layer.lora_beta_A, layer.lora_beta_B
                ])
            elif isinstance(layer, LoRAPositionalEncoding):
                params.extend([layer.lora_A, layer.lora_B])
        return params

    def merge_lora_weights(self):
        """Merge all LoRA weights into original weights"""
        for layer in self.lora_layers.values():
            if hasattr(layer, 'merge_weights'):
                layer.merge_weights()
        logger.info("Merged all LoRA weights")

    def unmerge_lora_weights(self):
        """Unmerge all LoRA weights from original weights"""
        for layer in self.lora_layers.values():
            if hasattr(layer, 'unmerge_weights'):
                layer.unmerge_weights()
        logger.info("Unmerged all LoRA weights")

    def apply_ste_optimization(self, logits: torch.Tensor,
                               reward_signal: float) -> torch.Tensor:
        """
        Apply STE optimization for code generation.

        Args:
            logits: Code generation logits
            reward_signal: Performance reward for gradient scaling

        Returns:
            Processed logits with STE gradients
        """
        # Scale logits by reward signal
        scaled_logits = logits * (1 + reward_signal)

        # Apply STE
        output = self.ste_layer(scaled_logits, hard=self.training if hasattr(self, 'training') else True)

        return output

    def compute_ste_gradient(self, loss: torch.Tensor,
                             code_choices: torch.Tensor,
                             performance_metric: float) -> torch.Tensor:
        """
        Compute gradient through STE for non-differentiable code generation.

        Uses the performance metric to weight the gradients.

        Args:
            loss: Current loss
            code_choices: Discrete code choices made
            performance_metric: Performance improvement (PA)

        Returns:
            Estimated gradient
        """
        # The gradient is scaled by the performance metric
        # Positive performance -> reinforce choices
        # Negative performance -> discourage choices

        # Compute surrogate gradient
        with torch.enable_grad():
            # Create soft version for gradient computation
            soft_choices = torch.sigmoid(code_choices.float())
            surrogate_loss = loss * soft_choices.sum()
            surrogate_loss.backward(retain_graph=True)

        # Scale gradient by performance
        for param in self.get_lora_parameters():
            if param.grad is not None:
                param.grad *= performance_metric

        return loss

    def save_lora_weights(self, path: str):
        """Save LoRA weights to file"""
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[name] = layer.state_dict()

        torch.save({
            'lora_weights': state_dict,
            'config': self.config
        }, path)
        logger.info(f"LoRA weights saved to {path}")

    def load_lora_weights(self, path: str):
        """Load LoRA weights from file"""
        checkpoint = torch.load(path)
        state_dict = checkpoint['lora_weights']

        for name, layer in self.lora_layers.items():
            if name in state_dict:
                layer.load_state_dict(state_dict[name])

        logger.info(f"LoRA weights loaded from {path}")


class CodeGenerationSTE(nn.Module):
    """
    Complete STE module for LLM code generation.

    Handles the non-differentiable sampling of discrete code tokens
    while maintaining gradient flow for optimization.
    """

    def __init__(self, vocab_size: int, temperature: float = 1.0):
        super(CodeGenerationSTE, self).__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.gumbel_ste = GumbelSoftmaxSTE(temperature=temperature, hard=True)

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code token generation.

        Args:
            logits: Token logits [batch, seq_len, vocab_size]

        Returns:
            token_ids: Selected token IDs [batch, seq_len]
            soft_probs: Soft probabilities for gradient computation
        """
        batch_size, seq_len, _ = logits.shape

        # Reshape for Gumbel-Softmax
        flat_logits = logits.view(-1, self.vocab_size)

        # Apply Gumbel-Softmax with STE
        one_hot = self.gumbel_ste(flat_logits)

        # Get token IDs
        token_ids = one_hot.argmax(dim=-1).view(batch_size, seq_len)

        # Get soft probabilities for gradient
        soft_probs = F.softmax(flat_logits / self.temperature, dim=-1).view(batch_size, seq_len, -1)

        return token_ids, soft_probs

    def compute_reinforce_loss(self, logits: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute REINFORCE-style loss for code generation.

        L = -sum(log_prob(action) * reward)

        Args:
            logits: Token logits
            actions: Sampled token IDs
            rewards: Reward for each token

        Returns:
            Policy gradient loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        loss = -(selected_log_probs * rewards).mean()
        return loss


# Unit tests
if __name__ == "__main__":
    print("Testing Low-Level Enhancement components...")

    # Test STE
    print("\n1. Testing Straight-Through Estimator...")
    ste = STELayer(threshold=0.5, temperature=1.0)
    x = torch.randn(10, requires_grad=True)
    y = ste(x, hard=True)
    loss = y.sum()
    loss.backward()
    print(f"Input: {x[:5].detach()}")
    print(f"Output: {y[:5].detach()}")
    print(f"Gradient: {x.grad[:5]}")

    # Test Gumbel-Softmax STE
    print("\n2. Testing Gumbel-Softmax STE...")
    gumbel = GumbelSoftmaxSTE(temperature=0.5, hard=True)
    logits = torch.randn(5, 10, requires_grad=True)
    samples = gumbel(logits)
    loss = samples.sum()
    loss.backward()
    print(f"Samples shape: {samples.shape}")
    print(f"One-hot check: {samples[0]}")
    print(f"Gradient norm: {logits.grad.norm()}")

    # Test LoRA Linear
    print("\n3. Testing LoRA Linear...")
    original = nn.Linear(64, 128)
    lora = LoRALinear(original, rank=8, alpha=16)
    x = torch.randn(32, 64)
    y_orig = original(x)
    y_lora = lora(x)
    print(f"Original output norm: {y_orig.norm():.4f}")
    print(f"LoRA output norm: {y_lora.norm():.4f}")
    print(f"Difference norm: {(y_orig - y_lora).norm():.4f}")

    # Test LoRA LayerNorm
    print("\n4. Testing LoRA LayerNorm...")
    original_ln = nn.LayerNorm(64)
    lora_ln = LoRALayerNorm(original_ln, rank=4, alpha=8)
    x = torch.randn(32, 64)
    y_orig = original_ln(x)
    y_lora = lora_ln(x)
    print(f"LayerNorm output difference: {(y_orig - y_lora).abs().mean():.6f}")

    # Test Low-Level Enhancement manager
    print("\n5. Testing LLE Manager...")
    config = LoRAConfig(rank=8, alpha=16)
    lle = LowLevelEnhancement(config)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

    # Apply LoRA
    model_with_lora = lle.apply_lora(model)
    params = lle.get_lora_parameters()
    print(f"Number of LoRA parameters: {len(params)}")
    print(f"Total trainable params: {sum(p.numel() for p in params)}")

    # Test CodeGenerationSTE
    print("\n6. Testing CodeGenerationSTE...")
    code_ste = CodeGenerationSTE(vocab_size=50000, temperature=1.0)
    logits = torch.randn(2, 10, 50000, requires_grad=True)
    token_ids, soft_probs = code_ste(logits)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Soft probs shape: {soft_probs.shape}")

    print("\nAll Low-Level Enhancement tests passed!")
