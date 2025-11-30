"""
Configuration models for the tuner package.

This module defines dataclasses that represent configuration for different workflows:
- TrainingConfig: Parameters for a training run
- CheckpointInfo: Metadata about a training checkpoint
- UploadConfig: Parameters for model upload
- EvalConfig: Parameters for evaluation

These dataclasses provide type-safe, validated configuration objects that are
passed between handlers, backends, and discovery services.

Location: /mnt/f/Code/Toolset-Training/tuner/core/config.py
Used by: Handlers, backends, discovery services
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for a training run.

    This dataclass encapsulates all parameters needed to execute training,
    extracted from YAML config files and user selections.

    Attributes:
        method: Training method identifier ('sft', 'kto', 'mlx')
        platform: Platform identifier ('rtx', 'mac')
        config_path: Path to YAML config file
        trainer_dir: Path to trainer directory (e.g., Trainers/rtx3090_sft)
        model_name: Base model name (e.g., 'unsloth/mistral-7b-v0.3-bnb-4bit')
        dataset_file: Path to dataset file (absolute or relative)
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for optimizer

    Example:
        config = TrainingConfig(
            method='sft',
            platform='rtx',
            config_path=Path('/path/to/config.yaml'),
            trainer_dir=Path('/path/to/Trainers/rtx3090_sft'),
            model_name='unsloth/mistral-7b-v0.3-bnb-4bit',
            dataset_file='../../Datasets/syngen_tools_sft_11.18.25.jsonl',
            epochs=3,
            batch_size=6,
            learning_rate=2e-4,
        )
    """
    method: str
    platform: str
    config_path: Path
    trainer_dir: Path
    model_name: str
    dataset_file: str
    epochs: int
    batch_size: int
    learning_rate: float


@dataclass
class CheckpointInfo:
    """
    Information about a training checkpoint.

    This dataclass encapsulates checkpoint metadata including path, step number,
    training metrics, and checkpoint type (intermediate vs final).

    Attributes:
        path: Path to checkpoint directory
        step: Training step number (or -1 for final_model)
        metrics: Dictionary of training metrics (loss, kl, margin, lr, etc.)
        is_final: True if this is the final_model, False for intermediate checkpoints

    Example:
        checkpoint = CheckpointInfo(
            path=Path('/path/to/checkpoint-100'),
            step=100,
            metrics={'loss': 0.5, 'kl': 0.1, 'rewards/margins': 0.3},
            is_final=False,
        )
        score = checkpoint.score('kto')  # Calculate quality score
    """
    path: Path
    step: int
    metrics: Dict[str, float]
    is_final: bool

    def score(self, training_type: str) -> float:
        """
        Calculate quality score based on training type.

        For KTO training, quality is measured by margin/KL ratio (higher is better).
        For SFT training, quality is measured by negative loss (lower loss = higher score).

        Args:
            training_type: Type of training ('kto' or 'sft')

        Returns:
            Quality score for ranking checkpoints

        Example:
            kto_checkpoint = CheckpointInfo(
                path=Path('/path/to/checkpoint-100'),
                step=100,
                metrics={'kl': 0.1, 'rewards/margins': 0.3},
                is_final=False,
            )
            score = kto_checkpoint.score('kto')  # 0.3 / 0.1 = 3.0

            sft_checkpoint = CheckpointInfo(
                path=Path('/path/to/checkpoint-200'),
                step=200,
                metrics={'loss': 0.5},
                is_final=False,
            )
            score = sft_checkpoint.score('sft')  # -0.5
        """
        if training_type == "kto":
            # KTO: Higher margin/KL ratio is better
            kl = self.metrics.get('kl', 0)
            margin = self.metrics.get('rewards/margins', 0)
            return margin / kl if kl > 0 else 0
        else:
            # SFT: Lower loss is better (negate for sorting)
            return -self.metrics.get('loss', float('inf'))


@dataclass
class UploadConfig:
    """
    Configuration for model upload.

    This dataclass encapsulates all parameters needed to upload a model to HuggingFace,
    including model path, repository ID, save method, and GGUF quantization options.

    Attributes:
        model_path: Path to model directory (final_model or checkpoint)
        repo_id: HuggingFace repository ID (e.g., 'username/model-name')
        save_method: Save method ('merged_16bit', 'merged_4bit', 'lora')
        create_gguf: Whether to create GGUF quantizations
        hf_token: HuggingFace API token (write access required)

    Example:
        upload_config = UploadConfig(
            model_path=Path('/path/to/final_model'),
            repo_id='myuser/my-awesome-model',
            save_method='merged_16bit',
            create_gguf=True,
            hf_token='hf_...',
        )
    """
    model_path: Path
    repo_id: str
    save_method: str
    create_gguf: bool
    hf_token: str


@dataclass
class EvalConfig:
    """
    Configuration for evaluation.

    This dataclass encapsulates all parameters needed to evaluate a model,
    including backend selection, model name, prompt set, and inference parameters.

    Attributes:
        backend: Evaluation backend ('ollama', 'lmstudio')
        model: Model name/identifier (backend-specific format)
        prompt_set: Path to prompt set JSON file
        prompt_count: Number of prompts in the set (for display)
        temperature: Sampling temperature (default: 0.2)
        top_p: Nucleus sampling parameter (default: 0.9)
        max_tokens: Maximum tokens to generate (default: 1024)

    Example:
        eval_config = EvalConfig(
            backend='ollama',
            model='claudesidian-mcp:latest',
            prompt_set=Path('/path/to/prompts/baseline.json'),
            prompt_count=10,
            temperature=0.2,
            top_p=0.9,
            max_tokens=1024,
        )
    """
    backend: str
    model: str
    prompt_set: Path
    prompt_count: int
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
