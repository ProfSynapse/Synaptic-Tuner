"""
File: /Users/jrosenbaum/Documents/Code/Synthetic Conversations/code/mistral_lora_mac/config/config_manager.py

Configuration Manager for MLX Fine-Tuning System

This module provides centralized configuration management with YAML loading,
validation, and type-safe configuration objects. It supports environment variable
overrides and provides default values for all settings.

Key Components:
- Configuration dataclasses (ModelConfig, LoRAConfig, TrainingConfig, etc.)
- ConfigurationManager for loading and validating configurations
- Environment variable override support
- Comprehensive validation with helpful error messages

Dependencies:
- PyYAML for configuration file parsing
- dataclasses for type-safe configuration objects
- os, pathlib for path resolution
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""
    pass


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir: str = "~/.cache/huggingface"
    dtype: str = "float16"
    max_seq_length: int = 2048
    trust_remote_code: bool = False

    def validate(self) -> List[str]:
        """Validate model configuration and return list of warnings."""
        warnings = []

        if self.dtype not in ["float16", "float32", "bfloat16"]:
            raise ConfigurationError(
                f"Invalid dtype '{self.dtype}'. Must be one of: float16, float32, bfloat16"
            )

        if self.max_seq_length < 128:
            raise ConfigurationError(
                f"max_seq_length must be >= 128, got {self.max_seq_length}"
            )

        if self.max_seq_length > 8192:
            warnings.append(
                f"max_seq_length={self.max_seq_length} is very large and may cause OOM"
            )

        return warnings


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"

    def validate(self) -> List[str]:
        """Validate LoRA configuration and return list of warnings."""
        warnings = []

        if self.rank < 1 or self.rank > 256:
            raise ConfigurationError(
                f"LoRA rank must be between 1 and 256, got {self.rank}"
            )

        if self.alpha < 1:
            raise ConfigurationError(
                f"LoRA alpha must be positive, got {self.alpha}"
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ConfigurationError(
                f"LoRA dropout must be in [0.0, 1.0), got {self.dropout}"
            )

        if self.bias not in ["none", "all", "lora_only"]:
            raise ConfigurationError(
                f"LoRA bias must be 'none', 'all', or 'lora_only', got '{self.bias}'"
            )

        if not self.target_modules:
            raise ConfigurationError("target_modules cannot be empty")

        if self.alpha != 2 * self.rank:
            warnings.append(
                f"LoRA alpha ({self.alpha}) != 2 * rank ({self.rank}). "
                "Typical setting is alpha = 2 * rank"
            )

        return warnings


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1
    seed: int = 42

    def validate(self) -> List[str]:
        """Validate training configuration and return list of warnings."""
        warnings = []

        if self.num_epochs < 1:
            raise ConfigurationError(
                f"num_epochs must be >= 1, got {self.num_epochs}"
            )

        if self.per_device_batch_size < 1:
            raise ConfigurationError(
                f"per_device_batch_size must be >= 1, got {self.per_device_batch_size}"
            )

        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )

        if self.learning_rate <= 0:
            raise ConfigurationError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.learning_rate > 1e-2:
            warnings.append(
                f"learning_rate={self.learning_rate} is very high for fine-tuning"
            )

        if self.max_grad_norm <= 0:
            raise ConfigurationError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )

        if self.weight_decay < 0:
            raise ConfigurationError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        effective_batch_size = self.per_device_batch_size * self.gradient_accumulation_steps
        if effective_batch_size < 4:
            warnings.append(
                f"Effective batch size ({effective_batch_size}) is small. "
                "Consider increasing for better stability."
            )

        return warnings


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    dataset_path: str = ""
    train_split: float = 0.8
    shuffle: bool = True
    seed: int = 42
    max_seq_length: int = 2048

    def validate(self) -> List[str]:
        """Validate data configuration and return list of warnings."""
        warnings = []

        if not self.dataset_path:
            raise ConfigurationError("dataset_path cannot be empty")

        if not (0.0 < self.train_split < 1.0):
            raise ConfigurationError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )

        if self.train_split < 0.5:
            warnings.append(
                f"train_split={self.train_split} is low. Consider using more training data."
            )

        return warnings


@dataclass
class OutputConfig:
    """Configuration for output paths and checkpoint management."""
    checkpoint_dir: str = "checkpoints"
    final_model_dir: str = "outputs/final_model"
    logs_dir: str = "logs"
    metrics_dir: str = "outputs/metrics"
    keep_last_n_checkpoints: int = 3

    def validate(self) -> List[str]:
        """Validate output configuration and return list of warnings."""
        warnings = []

        if self.keep_last_n_checkpoints < 1:
            raise ConfigurationError(
                f"keep_last_n_checkpoints must be >= 1, got {self.keep_last_n_checkpoints}"
            )

        return warnings


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    console: bool = True
    file: bool = True
    json_logs: bool = True
    log_file: str = "logs/training.log"
    json_log_file: str = "logs/training.jsonl"

    def validate(self) -> List[str]:
        """Validate logging configuration and return list of warnings."""
        warnings = []

        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ConfigurationError(
                f"log level must be one of {valid_levels}, got '{self.level}'"
            )

        return warnings


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and inference."""
    sample_prompts: List[str] = field(default_factory=list)
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    def validate(self) -> List[str]:
        """Validate evaluation configuration and return list of warnings."""
        warnings = []

        if self.max_new_tokens < 1:
            raise ConfigurationError(
                f"max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )

        if not (0.0 < self.temperature <= 2.0):
            warnings.append(
                f"temperature={self.temperature} is unusual. Typical range is (0, 2]"
            )

        if not (0.0 < self.top_p <= 1.0):
            raise ConfigurationError(
                f"top_p must be in (0, 1], got {self.top_p}"
            )

        return warnings


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    device: str = "metal"
    memory_limit_gb: int = 16
    num_workers: int = 0

    def validate(self) -> List[str]:
        """Validate system configuration and return list of warnings."""
        warnings = []

        if self.device not in ["metal", "cpu"]:
            warnings.append(
                f"device='{self.device}' may not be supported. Use 'metal' or 'cpu'"
            )

        if self.memory_limit_gb < 8:
            warnings.append(
                f"memory_limit_gb={self.memory_limit_gb} is low for 7B model fine-tuning"
            )

        return warnings


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    system: SystemConfig

    def validate(self) -> List[str]:
        """Validate all configurations and return list of warnings."""
        all_warnings = []

        # Validate each sub-config
        all_warnings.extend(self.model.validate())
        all_warnings.extend(self.lora.validate())
        all_warnings.extend(self.training.validate())
        all_warnings.extend(self.data.validate())
        all_warnings.extend(self.output.validate())
        all_warnings.extend(self.logging.validate())
        all_warnings.extend(self.evaluation.validate())
        all_warnings.extend(self.system.validate())

        # Cross-validation
        if self.data.max_seq_length != self.model.max_seq_length:
            raise ConfigurationError(
                f"data.max_seq_length ({self.data.max_seq_length}) must match "
                f"model.max_seq_length ({self.model.max_seq_length})"
            )

        return all_warnings


class ConfigurationManager:
    """Manages configuration loading, validation, and access."""

    def __init__(self, config_path: Optional[str] = None, project_root: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file. If None, uses default.
            project_root: Root directory of the project. Used for resolving relative paths.
        """
        self.config_path = config_path
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config: Optional[Config] = None

    def load(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file. Overrides constructor path if provided.

        Returns:
            Loaded and validated Config object.

        Raises:
            ConfigurationError: If config is invalid or file not found.
        """
        path = config_path or self.config_path

        if not path:
            raise ConfigurationError("No configuration path provided")

        config_file = Path(path)
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML: {e}")

        # Build configuration objects
        self.config = self._build_config(yaml_config)

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate
        warnings = self.config.validate()

        # Log warnings if any
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        return self.config

    def _build_config(self, yaml_config: Dict[str, Any]) -> Config:
        """Build Config object from YAML dictionary."""

        model_config = ModelConfig(**yaml_config.get('model', {}))
        lora_config = LoRAConfig(**yaml_config.get('lora', {}))
        training_config = TrainingConfig(**yaml_config.get('training', {}))
        data_config = DataConfig(**yaml_config.get('data', {}))
        output_config = OutputConfig(**yaml_config.get('output', {}))
        logging_config = LoggingConfig(**yaml_config.get('logging', {}))
        evaluation_config = EvaluationConfig(**yaml_config.get('evaluation', {}))
        system_config = SystemConfig(**yaml_config.get('system', {}))

        return Config(
            model=model_config,
            lora=lora_config,
            training=training_config,
            data=data_config,
            output=output_config,
            logging=logging_config,
            evaluation=evaluation_config,
            system=system_config
        )

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Support common environment variables
        if os.getenv('LEARNING_RATE'):
            self.config.training.learning_rate = float(os.getenv('LEARNING_RATE'))

        if os.getenv('BATCH_SIZE'):
            self.config.training.per_device_batch_size = int(os.getenv('BATCH_SIZE'))

        if os.getenv('NUM_EPOCHS'):
            self.config.training.num_epochs = int(os.getenv('NUM_EPOCHS'))

        if os.getenv('DATASET_PATH'):
            self.config.data.dataset_path = os.getenv('DATASET_PATH')

    def get_config(self) -> Config:
        """Get the loaded configuration."""
        if self.config is None:
            raise ConfigurationError("Configuration not loaded. Call load() first.")
        return self.config

    def save(self, path: str):
        """
        Save current configuration to YAML file.

        Args:
            path: Output path for config file.
        """
        if self.config is None:
            raise ConfigurationError("No configuration to save")

        config_dict = {
            'model': asdict(self.config.model),
            'lora': asdict(self.config.lora),
            'training': asdict(self.config.training),
            'data': asdict(self.config.data),
            'output': asdict(self.config.output),
            'logging': asdict(self.config.logging),
            'evaluation': asdict(self.config.evaluation),
            'system': asdict(self.config.system),
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
