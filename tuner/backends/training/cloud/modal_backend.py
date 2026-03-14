"""
Location: tuner/backends/training/cloud/modal_backend.py

Purpose:
    Modal cloud backend implementation for the tuner CLI. Implements the
    ITrainingBackend interface to enable submitting SFT and KTO training
    jobs to Modal's serverless GPU infrastructure.

    The backend works by running `modal run Trainers/cloud/train_modal.py`
    as a subprocess. Modal's CLI handles authentication, container provisioning,
    and log streaming. The train_modal.py script defines the remote function
    that clones the repo and executes the training scripts.

Usage:
    from tuner.backends.training.cloud.modal_backend import ModalBackend

    backend = ModalBackend(repo_root=Path("/path/to/repo"))
    is_valid, error = backend.validate_environment()
    config = backend.load_config("sft")
    exit_code = backend.execute(config, python_path="python")

Dependencies:
    - tuner.core.interfaces.ITrainingBackend
    - tuner.core.config.TrainingConfig
    - tuner.core.exceptions.ConfigurationError, BackendError
    - modal (optional, checked at runtime)
    - Trainers/cloud/train_modal.py (Modal wrapper script)
"""

import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import List, Tuple

from shared.utilities.paths import get_trainer_root
from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import CloudTrainingConfig, TrainingConfig
from tuner.core.exceptions import BackendError, ConfigurationError

from .base_cloud import (
    estimate_cost,
    get_gpu_display_name,
    load_cloud_config,
    load_gpu_pricing,
    resolve_repo_source,
)

logger = logging.getLogger(__name__)

DEFAULT_GPU = "L40S"
DEFAULT_TIMEOUT_HOURS = 6


class ModalBackend(ITrainingBackend):
    """Modal cloud training backend.

    Submits training jobs to Modal's serverless GPU infrastructure via the
    Modal CLI. Training runs inside Modal containers with persistent volume
    caching for model weights, reducing cold start times on repeated runs.

    Authentication is handled by Modal's OAuth flow (modal setup) which stores
    tokens in ~/.modal.toml, or via MODAL_TOKEN_ID + MODAL_TOKEN_SECRET
    environment variables for CI/CD environments.
    """

    def __init__(self, repo_root: Path):
        """Initialize Modal backend.

        Args:
            repo_root: Path to the Toolset-Training repository root
        """
        self.repo_root = Path(repo_root)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "modal"

    def get_available_methods(self) -> List[str]:
        """Get available training methods.

        Returns:
            List of supported training methods
        """
        return ["sft", "kto"]

    def validate_environment(self) -> Tuple[bool, str]:
        """Check that Modal is installed and authenticated.

        Validates:
        1. The `modal` Python package is importable
        2. Modal authentication is configured (either ~/.modal.toml from
           `modal setup`, or MODAL_TOKEN_ID + MODAL_TOKEN_SECRET env vars)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if modal package is installed
        try:
            import modal  # noqa: F401
        except ImportError:
            return False, (
                "Modal package is not installed. "
                "Install with: pip install modal"
            )

        # Check authentication: either env vars or ~/.modal.toml
        token_id = os.environ.get("MODAL_TOKEN_ID", "")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")

        if token_id and token_secret:
            # Env var authentication configured
            return True, ""

        # Check for OAuth token file
        modal_toml = Path.home() / ".modal.toml"
        if modal_toml.exists():
            return True, ""

        # Try running `modal token show` as a final check
        try:
            result = subprocess.run(
                ["modal", "token", "show"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return False, (
            "Modal is not authenticated. Set up authentication with one of:\n"
            "  1. Run: modal setup  (opens browser for OAuth login)\n"
            "  2. Set env vars: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET\n"
            "     Get tokens from: https://modal.com/settings"
        )

    def load_config(self, method: str) -> CloudTrainingConfig:
        """Load training configuration for the specified method.

        Loads the standard training config from the trainer's config.yaml
        and merges cloud-specific settings from Trainers/cloud/cloud_config.yaml
        using the shared load_cloud_config() utility.

        Args:
            method: Training method ("sft" or "kto")

        Returns:
            CloudTrainingConfig with merged cloud settings

        Raises:
            ConfigurationError: If config files are missing or invalid
        """
        if method not in self.get_available_methods():
            raise ConfigurationError(
                f"Unknown method '{method}' for Modal backend. "
                f"Available: {self.get_available_methods()}"
            )

        # Load standard training config
        trainer_dir = get_trainer_root(method, self.repo_root)
        config_path = trainer_dir / "configs" / "config.yaml"

        if not config_path.exists():
            raise ConfigurationError(f"Training config not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse training config: {e}")

        # Load cloud config overlay using shared utility
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        cloud_config = load_cloud_config(cloud_config_path)
        modal_config = cloud_config.get("modal", {})
        artifacts_config = cloud_config.get("artifacts", {})
        repo_source = resolve_repo_source(self.repo_root)

        # Extract relevant fields
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        training_config = config.get("training", {})

        # Determine GPU and timeout from cloud config or defaults
        gpu_type = modal_config.get("gpu", DEFAULT_GPU)
        timeout_hours = modal_config.get("timeout_hours", DEFAULT_TIMEOUT_HOURS)

        return CloudTrainingConfig(
            method=method,
            platform="modal",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get("model_name", "Unknown"),
            dataset_file=dataset_config.get("local_file") or f"{dataset_config.get('dataset_name', '')}/{dataset_config.get('dataset_file', 'Unknown')}",
            epochs=training_config.get("num_train_epochs", 1),
            batch_size=training_config.get("per_device_train_batch_size", 4),
            learning_rate=training_config.get("learning_rate", 0.0),
            provider="modal",
            gpu_type=gpu_type,
            timeout_hours=timeout_hours,
            push_to_hub=cloud_config.get("push_to_hub", False),
            hub_repo=cloud_config.get("hub_repo"),
            artifact_backend=modal_config.get("artifact_backend", "modal_volume"),
            artifact_identifier=modal_config.get("output_volume_name", "toolset-training-artifacts"),
            artifact_mount_path=modal_config.get("output_mount_path", "/vol/artifacts"),
            publish_final_model=artifacts_config.get("publish_final_model", False),
            publish_target_repo=artifacts_config.get("publish_target_repo"),
            repo_url=repo_source.url,
            repo_branch=repo_source.branch,
            repo_commit=repo_source.commit,
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """Execute training by running the Modal wrapper script.

        Invokes `modal run Trainers/cloud/train_modal.py` as a subprocess.
        Modal's CLI handles:
        - Container provisioning with the specified GPU
        - Log streaming from the remote container to the local terminal
        - Automatic cleanup on completion or failure

        Args:
            config: Training configuration (method, model_name, etc.)
            python_path: Path to Python interpreter (unused for Modal,
                         included for interface compatibility)

        Returns:
            Exit code from the modal run process (0 = success)

        Raises:
            BackendError: If the Modal wrapper script is not found
        """
        wrapper_script = self.repo_root / "Trainers" / "cloud" / "train_modal.py"
        if not wrapper_script.exists():
            raise BackendError(
                f"Modal wrapper script not found: {wrapper_script}\n"
                "Ensure Trainers/cloud/train_modal.py exists."
            )

        # Resolve GPU type and timeout from cloud config using shared utility
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        cloud_config = load_cloud_config(cloud_config_path)
        modal_settings = cloud_config.get("modal", {})
        gpu_type = modal_settings.get("gpu", DEFAULT_GPU)
        timeout_hours = modal_settings.get("timeout_hours", DEFAULT_TIMEOUT_HOURS)

        if not config.repo_url or not config.repo_branch or not config.repo_commit:
            raise BackendError("Modal cloud runs require exact repo source metadata.")

        # Build the modal run command
        cmd = [
            "modal", "run",
            str(wrapper_script),
            "--trainer-type", config.method,
            "--gpu", gpu_type,
            "--timeout-hours", str(timeout_hours),
            "--repo-url", config.repo_url,
            "--repo-branch", config.repo_branch,
            "--repo-commit", config.repo_commit,
        ]
        if config.publish_final_model:
            cmd.append("--publish-final-model")
        if config.publish_target_repo:
            cmd.extend(["--publish-target-repo", config.publish_target_repo])

        # Display cost estimate before starting
        gpu_display = get_gpu_display_name("modal", gpu_type)
        cost_str = estimate_cost("modal", gpu_type, timeout_hours) or "unknown"

        logger.info(
            "Submitting Modal training job: method=%s gpu=%s timeout=%dh est_cost=%s",
            config.method, gpu_display, timeout_hours, cost_str,
        )

        try:
            env = {
                **os.environ,
                "MODAL_CACHE_VOLUME_NAME": modal_settings.get("cache_volume_name", "toolset-model-cache"),
                "MODAL_OUTPUT_VOLUME_NAME": config.artifact_identifier or "toolset-training-artifacts",
                "MODAL_OUTPUT_MOUNT_PATH": config.artifact_mount_path or "/vol/artifacts",
                "CLOUD_ARTIFACT_IDENTIFIER": config.artifact_identifier or "",
                "CLOUD_REPO_BRANCH": config.repo_branch or "",
                "CLOUD_REPO_COMMIT": config.repo_commit or "",
            }
            process = subprocess.Popen(cmd, cwd=str(self.repo_root), env=env)
            timeout_secs = int(timeout_hours * 3600)
            return process.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            process.kill()
            raise BackendError(
                f"Modal job timed out after {timeout_hours}h. "
                f"Process killed. Check Modal dashboard for job status."
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            if process is not None:
                process.terminate()
            return 130
        except FileNotFoundError:
            raise BackendError(
                "Modal CLI not found. Install with: pip install modal"
            )
        except Exception as e:
            raise BackendError(f"Modal execution failed: {e}")

    def get_gpu_options(self) -> dict:
        """Get available GPU options with pricing info.

        Returns:
            Dict mapping GPU type names to their specs (name, price)
        """
        pricing = load_gpu_pricing()
        return pricing.get("modal", {}).copy()
