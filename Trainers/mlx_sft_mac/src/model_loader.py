"""
Model loading and LoRA application for MLX SFT training.
Uses mlx_lm for model loading and custom LoRA implementation.
"""

from typing import List, Tuple, Any, Optional
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Implements: output = base_output + (x @ A @ B) * scale
    where A and B are trainable low-rank matrices.
    """

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ) -> "LoRALinear":
        """Create LoRALinear from an existing Linear layer."""
        output_dims, input_dims = linear.weight.shape

        # Handle quantized layers
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = linear.weight.shape[1] * 32 // linear.bits

        lora_layer = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        lora_layer.linear = linear
        return lora_layer

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout_p = dropout

        # Base linear layer (will be set from existing layer)
        self.linear = None

        # LoRA matrices
        # A: (input_dims, rank) - initialized with small random values
        # B: (rank, output_dims) - initialized with zeros
        scale = 1.0 / rank
        self.lora_a = mx.random.normal((input_dims, rank)) * scale
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: base output + LoRA adaptation."""
        # Base layer output
        base_output = self.linear(x)

        # LoRA adaptation: x @ A @ B * scale
        lora_output = (x @ self.lora_a) @ self.lora_b
        lora_output = lora_output * self.scale

        return base_output + lora_output

    def to_linear(self) -> nn.Linear:
        """Merge LoRA weights back into a regular Linear layer."""
        # Get base weight
        base_weight = self.linear.weight

        # Compute LoRA contribution
        lora_weight = (self.lora_a @ self.lora_b).T * self.scale

        # Merge weights
        merged_weight = base_weight + lora_weight

        # Create new linear layer
        merged = nn.Linear(self.input_dims, self.output_dims)
        merged.weight = merged_weight

        if hasattr(self.linear, 'bias') and self.linear.bias is not None:
            merged.bias = self.linear.bias

        return merged


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer using mlx_lm.

    Args:
        model_name: HuggingFace model name (e.g., "mlx-community/Qwen3-0.6B-4bit")
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError(
            "mlx_lm not installed. Install with: pip install mlx-lm"
        )

    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")

    # Load model and tokenizer
    model, tokenizer = load(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[WARN] Set pad_token to eos_token: {tokenizer.eos_token}")

    # Print model info
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"[OK] Model loaded with {total_params:,} parameters")

    return model, tokenizer


def apply_lora(
    model: Any,
    rank: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: List[str] = None
) -> Any:
    """
    Apply LoRA adapters to target modules in the model.

    Args:
        model: MLX model
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        target_modules: List of module names to apply LoRA to

    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

    print(f"Applying LoRA with rank={rank}, alpha={alpha}")
    print(f"Target modules: {target_modules}")

    # Count modules before
    total_replaced = 0

    def apply_lora_to_model(model, prefix=""):
        """Recursively apply LoRA to matching modules."""
        nonlocal total_replaced

        if hasattr(model, 'layers'):
            # Transformer layers
            for i, layer in enumerate(model.layers):
                apply_lora_to_layer(layer, f"{prefix}layers.{i}.")

        return model

    def apply_lora_to_layer(layer, prefix):
        """Apply LoRA to a single transformer layer."""
        nonlocal total_replaced

        # Self attention
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if name in target_modules and hasattr(attn, name):
                    linear = getattr(attn, name)
                    if isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
                        lora_layer = LoRALinear.from_linear(
                            linear, rank=rank, alpha=alpha, dropout=dropout
                        )
                        setattr(attn, name, lora_layer)
                        total_replaced += 1

        # MLP
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if name in target_modules and hasattr(mlp, name):
                    linear = getattr(mlp, name)
                    if isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
                        lora_layer = LoRALinear.from_linear(
                            linear, rank=rank, alpha=alpha, dropout=dropout
                        )
                        setattr(mlp, name, lora_layer)
                        total_replaced += 1

    # Apply LoRA
    apply_lora_to_model(model)

    # Freeze base model, keep LoRA trainable
    model.freeze()

    # Unfreeze LoRA parameters
    def unfreeze_lora(model):
        """Unfreeze only LoRA parameters."""
        if hasattr(model, 'layers'):
            for layer in model.layers:
                unfreeze_lora_in_layer(layer)

    def unfreeze_lora_in_layer(layer):
        """Unfreeze LoRA in a layer."""
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, name):
                    module = getattr(attn, name)
                    if isinstance(module, LoRALinear):
                        module.lora_a = module.lora_a
                        module.lora_b = module.lora_b

        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, name):
                    module = getattr(mlp, name)
                    if isinstance(module, LoRALinear):
                        module.lora_a = module.lora_a
                        module.lora_b = module.lora_b

    unfreeze_lora(model)

    # Count trainable parameters
    trainable_params = 0
    total_params = 0

    for name, param in tree_flatten(model.parameters()):
        total_params += param.size
        if 'lora' in name.lower():
            trainable_params += param.size

    print(f"[OK] Applied LoRA to {total_replaced} modules")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable: {100 * trainable_params / total_params:.2f}%")

    return model


def get_lora_parameters(model) -> dict:
    """Extract only LoRA parameters from the model."""
    lora_params = {}

    for name, param in tree_flatten(model.parameters()):
        if 'lora_a' in name or 'lora_b' in name:
            lora_params[name] = param

    return lora_params


def save_lora_adapters(model, save_path: str):
    """Save only LoRA adapter weights."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    lora_params = get_lora_parameters(model)

    # Convert to numpy and save
    lora_params_np = {k: np.array(v) for k, v in lora_params.items()}

    adapter_path = save_path / "lora_adapters.npz"
    np.savez(adapter_path, **lora_params_np)

    print(f"[OK] Saved {len(lora_params)} LoRA tensors to {adapter_path}")

    return adapter_path


def load_lora_adapters(model, load_path: str):
    """Load LoRA adapter weights."""
    adapter_file = Path(load_path)
    if adapter_file.is_dir():
        adapter_file = adapter_file / "lora_adapters.npz"

    if not adapter_file.exists():
        raise FileNotFoundError(f"LoRA adapters not found: {adapter_file}")

    # Load numpy arrays
    loaded = np.load(adapter_file)

    # Update model parameters
    model_params = dict(tree_flatten(model.parameters()))

    for name in loaded.files:
        if name in model_params:
            model_params[name] = mx.array(loaded[name])

    # Update model
    model.update(tree_unflatten(list(model_params.items())))

    print(f"[OK] Loaded {len(loaded.files)} LoRA tensors from {adapter_file}")

    return model
