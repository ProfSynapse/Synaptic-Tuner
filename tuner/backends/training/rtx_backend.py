"""
Location: /mnt/f/Code/Toolset-Training/tuner/backends/training/rtx_backend.py

Purpose:
    NVIDIA RTX training backend implementation for SFT, KTO, and GRPO training
    methods. Handles configuration loading from YAML files and execution of
    training scripts via subprocess.

Usage:
    from tuner.backends.training.rtx_backend import RTXBackend

    backend = RTXBackend(repo_root=Path("/path/to/repo"))
    config = backend.load_config("sft")
    exit_code = backend.execute(config, python_path="/path/to/conda/python")

Dependencies:
    - tuner.core.interfaces.ITrainingBackend
    - tuner.core.config.TrainingConfig
    - tuner.core.exceptions.ConfigurationError
    - Trainers/sft/configs/config.yaml
    - Trainers/kto/configs/config.yaml
    - Trainers/grpo/configs/env_config.yaml
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import List

from shared.utilities.paths import get_trainer_root
from .base import ITrainingBackend
from tuner.core.config import TrainingConfig
from tuner.core.exceptions import ConfigurationError


class RTXBackend(ITrainingBackend):
    """
    NVIDIA RTX training backend (SFT/KTO via Unsloth).

    Supports three training methods:
    - SFT (Supervised Fine-Tuning): Teach tool-calling from scratch
    - KTO (Preference Learning): Refine existing tool-calling behavior
    - GRPO (Environment-backed reward optimization): Multi-step env rollouts

    Both methods use configuration from YAML files in their respective
    trainer directories.
    """

    ENV_GRPO_CONFIG_NAME = "env_config.yaml"
    ENV_GRPO_SCRIPT_NAME = "train_env_grpo.py"

    @classmethod
    def _config_filename_for_method(cls, method: str) -> str:
        return cls.ENV_GRPO_CONFIG_NAME if method == "grpo" else "config.yaml"

    @classmethod
    def _script_for_config(cls, config: TrainingConfig) -> str:
        if config.method == "grpo" and config.config_path.name == cls.ENV_GRPO_CONFIG_NAME:
            return cls.ENV_GRPO_SCRIPT_NAME
        return f"train_{config.method}.py"

    @staticmethod
    def _dataset_display(dataset_config: dict) -> str:
        local_file = dataset_config.get("local_file")
        if local_file:
            return str(local_file)

        dataset_name = str(dataset_config.get("dataset_name") or "").strip()
        dataset_file = str(dataset_config.get("dataset_file") or "").strip()
        if dataset_name and dataset_file:
            return f"{dataset_name}/{dataset_file}"
        if dataset_file:
            return dataset_file
        if dataset_name:
            return dataset_name
        return "Unknown"

    @staticmethod
    def _load_raw_config(config_path: Path) -> dict:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def __init__(self, repo_root: Path):
        """
        Initialize RTX backend.

        Args:
            repo_root: Path to repository root directory
        """
        self.repo_root = Path(repo_root)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "rtx"

    def get_available_methods(self) -> List[str]:
        """
        Get available training methods for RTX backend.

        Returns:
            List of method names: ['sft', 'kto', 'grpo', 'dpo']
        """
        return ["sft", "kto", "grpo", "dpo"]

    def load_config(self, method: str) -> TrainingConfig:
        """
        Load configuration from YAML file.

        Args:
            method: Training method ('sft', 'kto', or 'grpo')

        Returns:
            Parsed training configuration

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        if method not in self.get_available_methods():
            raise ConfigurationError(
                f"Unknown method '{method}' for RTX backend. "
                f"Available: {self.get_available_methods()}"
            )

        trainer_dir = get_trainer_root(method, self.repo_root)
        config_path = trainer_dir / "configs" / self._config_filename_for_method(method)

        if not config_path.exists():
            raise ConfigurationError(f"Config not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config: {e}")

        # Extract relevant fields from nested YAML structure
        model_config = config.get('model', {})
        dataset_config = config.get('dataset', {})
        training_config = config.get('training', {})

        return TrainingConfig(
            method=method,
            platform="rtx",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get('model_name', 'Unknown'),
            dataset_file=self._dataset_display(dataset_config),
            epochs=training_config.get('num_train_epochs', 1),
            batch_size=training_config.get('per_device_train_batch_size', 4),
            learning_rate=training_config.get('learning_rate', 0.0),
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Execute training script via subprocess.

        Args:
            config: Training configuration
            python_path: Path to Python interpreter (conda environment)

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        script_name = self._script_for_config(config)
        runtime_python = python_path
        if script_name == self.ENV_GRPO_SCRIPT_NAME:
            raw_config = self._load_raw_config(config.config_path)
            env_runtime_src = str(config.trainer_dir / "src")
            if env_runtime_src not in sys.path:
                sys.path.insert(0, env_runtime_src)
            from env_runtime import ensure_local_openenv_runtime  # type: ignore

            print("Ensuring local env-GRPO runtime...")
            runtime_python = ensure_local_openenv_runtime(
                raw_config,
                repo_root=str(self.repo_root),
                bootstrap_python=python_path,
            )

        cmd = [runtime_python, script_name]
        if script_name == self.ENV_GRPO_SCRIPT_NAME:
            cmd.extend(["--config", str(config.config_path)])

        # Dashboard is now the default in train_sft.py, no flag needed
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(config.trainer_dir),
            )
            return process.wait()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            if 'process' in locals():
                process.terminate()
            return 130
        except Exception as e:
            print(f"Execution error: {e}")
            return 1

    def validate_environment(self) -> tuple[bool, str]:
        """
        Validate that CUDA is available for RTX training.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if CUDA is available
            - (False, "error description") otherwise
        """
        try:
            import torch
            if torch.cuda.is_available():
                return True, ""
            else:
                return False, "CUDA not available. Ensure NVIDIA drivers and CUDA toolkit are installed."
        except ImportError:
            return False, "PyTorch not installed. Run setup.sh to install dependencies."
