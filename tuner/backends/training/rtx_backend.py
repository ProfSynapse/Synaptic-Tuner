"""
Location: /mnt/f/Code/Toolset-Training/tuner/backends/training/rtx_backend.py

Purpose:
    NVIDIA RTX training backend implementation for SFT and KTO training methods.
    Handles configuration loading from YAML files and execution of training scripts
    via subprocess.

Usage:
    from tuner.backends.training.rtx_backend import RTXBackend

    backend = RTXBackend(repo_root=Path("/path/to/repo"))
    config = backend.load_config("sft")
    exit_code = backend.execute(config, python_path="/path/to/conda/python")

Dependencies:
    - tuner.core.interfaces.ITrainingBackend
    - tuner.core.config.TrainingConfig
    - tuner.core.exceptions.ConfigurationError
    - Trainers/rtx3090_sft/configs/config.yaml
    - Trainers/rtx3090_kto/configs/config.yaml
"""

import yaml
import subprocess
from pathlib import Path
from typing import List

from .base import ITrainingBackend
from tuner.core.config import TrainingConfig
from tuner.core.exceptions import ConfigurationError


class RTXBackend(ITrainingBackend):
    """
    NVIDIA RTX training backend (SFT/KTO via Unsloth).

    Supports two training methods:
    - SFT (Supervised Fine-Tuning): Teach tool-calling from scratch
    - KTO (Preference Learning): Refine existing tool-calling behavior

    Both methods use configuration from YAML files in their respective
    trainer directories.
    """

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
            List of method names: ['sft', 'kto']
        """
        return ["sft", "kto"]

    def load_config(self, method: str) -> TrainingConfig:
        """
        Load configuration from YAML file.

        Args:
            method: Training method ('sft' or 'kto')

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

        trainer_dir = self.repo_root / "Trainers" / f"rtx3090_{method}"
        config_path = trainer_dir / "configs" / "config.yaml"

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
            dataset_file=dataset_config.get('local_file', 'Unknown'),
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
        cmd = [python_path, f"train_{config.method}.py"]
        result = subprocess.run(cmd, cwd=str(config.trainer_dir))
        return result.returncode

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
