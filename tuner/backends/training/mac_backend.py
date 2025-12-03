"""
Location: /mnt/f/Code/Toolset-Training/tuner/backends/training/mac_backend.py

Purpose:
    Apple Silicon (M1/M2/M3) training backend implementation for MLX LoRA training.
    Handles configuration loading from YAML files and execution of training scripts
    via subprocess.

Usage:
    from tuner.backends.training.mac_backend import MacBackend

    backend = MacBackend(repo_root=Path("/path/to/repo"))
    config = backend.load_config("mlx")
    exit_code = backend.execute(config, python_path="/path/to/python")

Dependencies:
    - tuner.core.interfaces.ITrainingBackend
    - tuner.core.config.TrainingConfig
    - tuner.core.exceptions.ConfigurationError
    - Trainers/mistral_lora_mac/config/config.yaml
"""

import yaml
import subprocess
from pathlib import Path
from typing import List

from .base import ITrainingBackend
from tuner.core.config import TrainingConfig
from tuner.core.exceptions import ConfigurationError


class MacBackend(ITrainingBackend):
    """
    Apple Silicon (Mac) training backend (MLX LoRA).

    Supports one training method:
    - MLX: LoRA fine-tuning using Apple's MLX framework optimized for Metal GPU

    Uses configuration from YAML file in the Mac trainer directory.
    """

    def __init__(self, repo_root: Path):
        """
        Initialize Mac backend.

        Args:
            repo_root: Path to repository root directory
        """
        self.repo_root = Path(repo_root)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "mac"

    def get_available_methods(self) -> List[str]:
        """
        Get available training methods for Mac backend.

        Returns:
            List of method names: ['mlx']
        """
        return ["mlx"]

    def load_config(self, method: str) -> TrainingConfig:
        """
        Load configuration from YAML file.

        Args:
            method: Training method (must be 'mlx')

        Returns:
            Parsed training configuration

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        if method not in self.get_available_methods():
            raise ConfigurationError(
                f"Unknown method '{method}' for Mac backend. "
                f"Available: {self.get_available_methods()}"
            )

        trainer_dir = self.repo_root / "Trainers" / "mistral_lora_mac"
        config_path = trainer_dir / "config" / "config.yaml"

        if not config_path.exists():
            raise ConfigurationError(f"Config not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config: {e}")

        # Extract relevant fields from nested YAML structure
        # Mac config structure is different from RTX
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        lora_config = config.get('lora', {})

        return TrainingConfig(
            method=method,
            platform="mac",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get('name', 'Unknown'),
            dataset_file=data_config.get('dataset_path', 'Unknown'),
            epochs=training_config.get('num_epochs', 1),
            batch_size=training_config.get('per_device_batch_size', 2),
            learning_rate=training_config.get('learning_rate', 0.0),
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Execute training script via subprocess.

        Args:
            config: Training configuration
            python_path: Path to Python interpreter

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        import sys
        import shutil
        import threading
        import time
        from tuner.ui import console, RICH_AVAILABLE

        # Mac trainer uses main.py with --config flag
        cmd = [
            python_path,
            "main.py",
            "--config",
            str(config.config_path)
        ]
        
        if not RICH_AVAILABLE:
            result = subprocess.run(cmd, cwd=str(config.trainer_dir))
            return result.returncode

        # Interactive execution with loader
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(config.trainer_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            # Use a thread to peek at the output so we don't block the spinner
            output_started = threading.Event()
            first_char = []

            def wait_for_output():
                try:
                    char = process.stdout.read(1)
                    if char:
                        first_char.append(char)
                except Exception:
                    pass
                finally:
                    output_started.set()

            reader_thread = threading.Thread(target=wait_for_output, daemon=True)
            reader_thread.start()

            with console.status("[bold aqua]Initializing MLX & Metal Kernels...[/bold aqua]", spinner="dots12"):
                while not output_started.is_set():
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)

            if not first_char and process.poll() is not None:
                return process.returncode

            if first_char:
                sys.stdout.write(first_char[0])
                sys.stdout.flush()
            
            shutil.copyfileobj(process.stdout, sys.stdout)
            
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
        Validate that MLX is available for Mac training.

        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if MLX Metal GPU is available
            - (False, "error description") otherwise
        """
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return True, ""
            else:
                return False, "Metal GPU not available. Ensure you're on Apple Silicon (M1/M2/M3)."
        except ImportError:
            return False, "MLX not installed. Install via: pip install mlx"
