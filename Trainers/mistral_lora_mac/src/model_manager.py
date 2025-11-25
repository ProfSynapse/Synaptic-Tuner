"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/src/model_manager.py

Model Manager with LoRA Integration for MLX Fine-Tuning System

This module handles:
- Loading Mistral-7B-Instruct-v0.3 from Hugging Face
- Converting model to MLX format
- Applying LoRA adapters to target modules
- Managing trainable vs frozen parameters
- Saving and loading LoRA adapters
- Parameter counting and statistics

Dependencies:
- mlx: Core ML framework for Apple Silicon
- mlx.nn: Neural network modules
- transformers: Model downloading
- huggingface_hub: Model hub access

Related Files:
- config/config_manager.py: Model and LoRA configuration
- src/utils.py: Logging and parameter counting
- src/trainer.py: Uses model for training
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np


@dataclass
class ParameterStats:
    """Model parameter statistics."""
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_percent: float


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.

    Implements: h = Wx + (B @ A)x where W is frozen, A and B are trainable.
    - A: (input_dim, rank)
    - B: (rank, output_dim)
    - scaling: alpha / rank
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        scale: Optional[float] = None
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA alpha scaling parameter
            dropout: Dropout probability
            scale: Optional manual scaling factor (overrides alpha/rank)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA scaling
        self.scale = scale if scale is not None else alpha / rank

        # LoRA matrices
        # Initialize A with small random values (Gaussian)
        self.lora_A = mx.random.normal((in_features, rank), scale=0.01)
        # Initialize B with zeros (as per LoRA paper)
        self.lora_B = mx.zeros((rank, out_features))

        # Dropout
        self.dropout_prob = dropout

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            LoRA adaptation: (B @ A) @ x * scale
        """
        # Apply dropout during training
        if self.dropout_prob > 0 and self.training:
            mask = mx.random.bernoulli(1 - self.dropout_prob, x.shape)
            x = x * mask / (1 - self.dropout_prob)

        # LoRA forward: x @ A @ B * scale
        result = x @ self.lora_A  # (*, rank)
        result = result @ self.lora_B  # (*, out_features)
        result = result * self.scale

        return result


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0
    ):
        """
        Initialize LoRA Linear layer.

        Args:
            base_layer: Original frozen linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability
        """
        super().__init__()

        self.base_layer = base_layer
        self.in_features = base_layer.weight.shape[1]
        self.out_features = base_layer.weight.shape[0]

        # Freeze base layer
        self.base_layer.freeze()

        # Create LoRA adapter
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: base output + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Base layer output (frozen)
        base_output = self.base_layer(x)

        # LoRA adaptation
        lora_output = self.lora(x)

        return base_output + lora_output


class ModelManager:
    """
    Manages model loading, LoRA application, and adapter save/load.

    Workflow:
    1. Load base model from Hugging Face (or use MLX pre-converted models)
    2. Apply LoRA adapters to target modules
    3. Freeze base parameters, keep LoRA trainable
    4. Provide interfaces for training and inference
    """

    def __init__(self, config, logger):
        """
        Initialize model manager.

        Args:
            config: Config object with model and LoRA settings
            logger: StructuredLogger
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.reference_model = None
        self.tokenizer = None

    def load_base_model(self):
        """
        Load Mistral-7B base model.

        For MLX, we use the mlx-community converted models or load via transformers
        and convert. This is a placeholder for the actual MLX model loading.

        Returns:
            MLX model
        """
        self.logger.info(f"Loading model: {self.config.model.name}")
        self.logger.info("Note: Using MLX-specific model loading")

        try:
            # Try to load from mlx-community (pre-converted models)
            from mlx_lm import load

            model_path = self.config.model.name
            # For mlx-community models, use: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            # For now, we'll document that users should use MLX-converted models

            self.logger.info(f"Loading MLX model from: {model_path}")
            model, tokenizer = load(model_path)

            self.model = model
            self.tokenizer = tokenizer

            self.logger.info("Model loaded successfully")
            return model

        except ImportError:
            self.logger.error("mlx_lm not installed. Install with: pip install mlx-lm")
            raise

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Falling back to manual model construction")
            # Fallback: construct model manually (simplified for demonstration)
            self.model = self._create_dummy_model()
            return self.model

    def _create_dummy_model(self):
        """
        Create a simplified model structure for demonstration.
        In production, this would be the full Mistral model.
        """
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified: just a few transformer layers
                self.layers = []
                for i in range(4):  # Reduced from 32 for demo
                    layer = nn.TransformerEncoderLayer(
                        d_model=4096,
                        num_heads=32,
                        dim_feedforward=14336,
                        dropout=0.0
                    )
                    self.layers.append(layer)

                self.embed_tokens = nn.Embedding(32000, 4096)
                self.lm_head = nn.Linear(4096, 32000)

            def __call__(self, input_ids, attention_mask=None):
                x = self.embed_tokens(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        self.logger.warning("Using simplified model structure for demonstration")
        return SimpleTransformer()

    def apply_lora(self):
        """
        Apply LoRA adapters to target modules in the model.

        Replaces specified modules (e.g., q_proj, v_proj) with LoRA-wrapped versions.

        Returns:
            Model with LoRA adapters applied
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        self.logger.info("Applying LoRA adapters...")
        self.logger.info(f"Target modules: {self.config.lora.target_modules}")
        self.logger.info(f"LoRA rank: {self.config.lora.rank}")
        self.logger.info(f"LoRA alpha: {self.config.lora.alpha}")
        self.logger.info(f"LoRA dropout: {self.config.lora.dropout}")

        # Apply LoRA to target modules
        lora_modules_count = self._apply_lora_to_modules(
            self.model,
            target_modules=self.config.lora.target_modules,
            rank=self.config.lora.rank,
            alpha=self.config.lora.alpha,
            dropout=self.config.lora.dropout
        )

        self.logger.info(f"Applied LoRA to {lora_modules_count} modules")

        # Get parameter statistics
        stats = self.count_parameters()
        self.logger.info(f"Total parameters: {stats.total_params:,}")
        self.logger.info(f"Trainable parameters: {stats.trainable_params:,}")
        self.logger.info(f"Frozen parameters: {stats.frozen_params:,}")
        self.logger.info(f"Trainable: {stats.trainable_percent:.2f}%")

        return self.model

    def _apply_lora_to_modules(
        self,
        module: nn.Module,
        target_modules: list,
        rank: int,
        alpha: int,
        dropout: float,
        prefix: str = ""
    ) -> int:
        """
        Recursively apply LoRA to target modules.

        Args:
            module: Current module
            target_modules: List of module names to target
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
            prefix: Current module path

        Returns:
            Count of modules with LoRA applied
        """
        count = 0

        # Get all child modules
        for name, child in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this module should have LoRA
            should_apply = any(target in name for target in target_modules)

            if should_apply and isinstance(child, nn.Linear):
                self.logger.debug(f"Applying LoRA to: {full_name}")

                # Replace with LoRA version
                lora_layer = LoRALinear(
                    base_layer=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )

                # Set the new layer (this is MLX-specific)
                setattr(module, name, lora_layer)
                count += 1

        return count

    def get_trainable_params(self) -> Dict[str, mx.array]:
        """
        Get only trainable (LoRA) parameters.

        Returns:
            Dictionary of parameter names to arrays
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        trainable = {}
        for name, param in tree_flatten(self.model.parameters()):
            # LoRA parameters will have 'lora' in their name
            if 'lora' in name.lower():
                trainable[name] = param

        return trainable

    def count_parameters(self) -> ParameterStats:
        """
        Count total, trainable, and frozen parameters.

        Returns:
            ParameterStats object
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        total_params = 0
        trainable_params = 0

        for name, param in tree_flatten(self.model.parameters()):
            num_params = param.size

            total_params += num_params

            # Check if trainable (LoRA parameters)
            if 'lora' in name.lower():
                trainable_params += num_params

        frozen_params = total_params - trainable_params

        return ParameterStats(
            total_params=total_params,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            trainable_percent=(trainable_params / total_params * 100) if total_params > 0 else 0
        )

    def save_adapters(self, save_path: str):
        """
        Save only LoRA adapter weights.

        Args:
            save_path: Path to save adapters (npz format)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.logger.info(f"Saving LoRA adapters to: {save_path}")

        # Get only LoRA parameters
        lora_params = self.get_trainable_params()

        # Convert to numpy for saving
        lora_params_np = {}
        for name, param in lora_params.items():
            lora_params_np[name] = np.array(param)

        # Save to npz file
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **lora_params_np)

        self.logger.info(f"Saved {len(lora_params)} LoRA parameter tensors")

    def load_adapters(self, load_path: str):
        """
        Load LoRA adapter weights.

        Args:
            load_path: Path to load adapters from (npz format)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.logger.info(f"Loading LoRA adapters from: {load_path}")

        # Load from npz file
        loaded = np.load(load_path)

        # Update model parameters
        model_params = dict(tree_flatten(self.model.parameters()))

        for name in loaded.files:
            if name in model_params:
                # Convert numpy to MLX array and update
                model_params[name] = mx.array(loaded[name])
            else:
                self.logger.warning(f"Parameter {name} in checkpoint not found in model")

        # Update model with new parameters
        self.model.update(tree_unflatten(list(model_params.items())))

        self.logger.info(f"Loaded {len(loaded.files)} LoRA parameter tensors")

    def get_model(self):
        """Get the model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.model

    def set_train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()

    def create_reference_model(self):
        """
        Create a frozen reference model for KTO training.

        This creates a deep copy of the current model state (before LoRA training)
        and freezes all parameters. The reference model is used to compute KL
        divergence in KTO loss.

        Returns:
            Frozen reference model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        self.logger.info("Creating frozen reference model for KTO...")

        # Import copy functionality
        import copy

        # Create a deep copy of the current model
        # This preserves the model architecture and current weights
        self.reference_model = copy.deepcopy(self.model)

        # Freeze all parameters in the reference model
        self._freeze_all_parameters(self.reference_model)

        # Set reference model to eval mode (permanently)
        self.reference_model.eval()

        self.logger.info("Reference model created and frozen")

        return self.reference_model

    def _freeze_all_parameters(self, model):
        """
        Freeze all parameters in a model.

        Args:
            model: Model to freeze
        """
        # In MLX, we freeze parameters by calling .freeze() on the model
        # This prevents gradients from being computed for these parameters
        model.freeze()

        self.logger.debug("All parameters frozen in reference model")

    def get_reference_model(self):
        """
        Get the reference model (for KTO training).

        Returns:
            Reference model or None if not created
        """
        return self.reference_model
