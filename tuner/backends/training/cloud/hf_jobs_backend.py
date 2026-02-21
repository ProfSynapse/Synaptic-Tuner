"""
HuggingFace Jobs cloud training backend.

Location: tuner/backends/training/cloud/hf_jobs_backend.py
Purpose: Submit and monitor training jobs via HuggingFace Jobs API
Used by: CloudTrainHandler when user selects HF Jobs provider

Implements ITrainingBackend for HuggingFace Jobs. Training scripts are
submitted as UV scripts with PEP 723 inline dependency declarations.
The HF Jobs infrastructure handles provisioning GPU hardware, installing
dependencies, and running the training script.

Requirements:
- HF_TOKEN environment variable set (requires HF Pro subscription)
- huggingface_hub >= 0.27.0 installed
"""

import logging
import os
import textwrap
import time
import yaml
from pathlib import Path
from typing import List, Tuple

from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import TrainingConfig, CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError

from .base_cloud import load_cloud_config, resolve_repo_url, poll_until_done

logger = logging.getLogger(__name__)

# Default HF Jobs settings
DEFAULT_FLAVOR = "a10g-small"
DEFAULT_TIMEOUT = "4h"
DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


class HFJobsBackend(ITrainingBackend):
    """
    HuggingFace Jobs training backend.

    Submits training jobs to HuggingFace's managed GPU infrastructure using
    the huggingface_hub library. Training scripts are wrapped in UV script
    format with PEP 723 inline dependency declarations.

    The execute() flow:
    1. Build a UV script wrapper with inline deps
    2. Submit via huggingface_hub.run_job()
    3. Poll inspect_job() and stream fetch_job_logs() until completion
    4. Return exit code (0 = success, 1 = failure)

    GPU Options:
    - t4-small: T4 16GB (~$0.40/hr)
    - a10g-small: A10G 24GB (~$1.10/hr)
    - a100-large: A100 80GB (~$2.50/hr)
    """

    def __init__(self, repo_root: Path):
        """
        Initialize HF Jobs backend.

        Args:
            repo_root: Path to repository root directory
        """
        self.repo_root = Path(repo_root)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "hf_jobs"

    def get_available_methods(self) -> List[str]:
        """
        Get available training methods.

        HF Jobs supports the same methods as local RTX training since
        the scripts run unchanged in the cloud environment.

        Returns:
            List of method names: ['sft', 'kto']
        """
        return ["sft", "kto"]

    def validate_environment(self) -> Tuple[bool, str]:
        """
        Validate that HF Jobs environment is properly configured.

        Checks:
        1. HF_TOKEN environment variable is set
        2. huggingface_hub package is installed
        3. huggingface_hub version supports Jobs API

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check HF_TOKEN
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return False, (
                "HF_TOKEN not set. Required for HuggingFace Jobs. "
                "Set it in your .env file or environment."
            )

        # Validate token format without revealing full value
        if not hf_token.startswith("hf_"):
            return False, (
                "HF_TOKEN has unexpected format (should start with 'hf_'). "
                "Check your token at https://huggingface.co/settings/tokens"
            )

        # Check huggingface_hub is installed
        try:
            import huggingface_hub
        except ImportError:
            return False, (
                "huggingface_hub not installed. "
                "Install with: pip install -r requirements-cloud.txt"
            )

        # Check for run_job support (requires recent version)
        if not hasattr(huggingface_hub, "run_job"):
            version = getattr(huggingface_hub, "__version__", "unknown")
            return False, (
                f"huggingface_hub {version} does not support Jobs API. "
                "Upgrade with: pip install --upgrade huggingface_hub>=0.27.0"
            )

        return True, ""

    def load_config(self, method: str) -> CloudTrainingConfig:
        """
        Load training configuration with cloud overlay.

        Loads the standard training config from the trainer directory,
        then overlays cloud-specific settings from cloud_config.yaml.

        Args:
            method: Training method ('sft' or 'kto')

        Returns:
            CloudTrainingConfig with merged settings

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        if method not in self.get_available_methods():
            raise ConfigurationError(
                f"Unknown method '{method}' for HF Jobs backend. "
                f"Available: {self.get_available_methods()}"
            )

        # Load standard training config
        trainer_dir = self.repo_root / "Trainers" / f"rtx3090_{method}"
        config_path = trainer_dir / "configs" / "config.yaml"

        if not config_path.exists():
            raise ConfigurationError(f"Training config not found: {config_path}")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse training config: {e}")

        # Extract training parameters
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        training_config = config.get("training", {})

        # Load cloud overlay
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        cloud_config = load_cloud_config(cloud_config_path)
        hf_config = cloud_config.get("hf_jobs", {})

        flavor = hf_config.get("flavor", DEFAULT_FLAVOR)
        timeout_str = hf_config.get("timeout", DEFAULT_TIMEOUT)
        image = hf_config.get("image", DEFAULT_IMAGE)

        # Parse timeout string (e.g., "4h" -> 4.0)
        timeout_hours = _parse_timeout(timeout_str)

        return CloudTrainingConfig(
            method=method,
            platform="hf_jobs",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get("model_name", "Unknown"),
            dataset_file=dataset_config.get("local_file", "Unknown"),
            epochs=training_config.get("num_train_epochs", 1),
            batch_size=training_config.get("per_device_train_batch_size", 4),
            learning_rate=training_config.get("learning_rate", 0.0),
            provider="hf_jobs",
            gpu_type=flavor,
            timeout_hours=timeout_hours,
            cloud_image=image,
            push_to_hub=cloud_config.get("push_to_hub", True),
            hub_repo=cloud_config.get("hub_repo"),
            hf_flavor=flavor,
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Submit training job to HuggingFace Jobs and poll until completion.

        Builds a UV script wrapper with PEP 723 inline dependencies,
        submits it via huggingface_hub.run_job(), then polls for
        completion while streaming logs.

        Args:
            config: Training configuration (should be CloudTrainingConfig)
            python_path: Not used for cloud execution (kept for interface compat)

        Returns:
            Exit code (0 = success, 1 = failure)
        """
        try:
            import huggingface_hub
        except ImportError:
            raise CloudProviderError(
                "huggingface_hub not installed. "
                "Install with: pip install -r requirements-cloud.txt"
            )

        # Cast to CloudTrainingConfig if needed
        if not isinstance(config, CloudTrainingConfig):
            raise CloudProviderError(
                "HF Jobs backend requires CloudTrainingConfig, "
                f"got {type(config).__name__}"
            )

        # Build the UV script content
        script_content = self._build_uv_script(config)

        logger.info("Submitting job to HuggingFace Jobs...")
        logger.info("Flavor: %s", config.hf_flavor or config.gpu_type)
        logger.info("Timeout: %.1f hours", config.timeout_hours)

        try:
            # Submit the job
            job = huggingface_hub.run_job(
                script_content,
                flavor=config.hf_flavor or config.gpu_type,
                secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
            )
            job_id = job.job_id if hasattr(job, "job_id") else str(job)
            print(f"  Job submitted: {job_id}")
            print(f"  Polling for completion (every 30s, timeout: {config.timeout_hours}h)...")
            print()

        except Exception as e:
            error_msg = str(e)
            # Mask any token values in error messages
            if "hf_" in error_msg:
                error_msg = "Job submission failed (check credentials and subscription)"
            raise CloudProviderError(f"Failed to submit HF Jobs training: {error_msg}")

        # Poll until completion
        timeout_seconds = int(config.timeout_hours * 3600)
        last_log_offset = 0

        def check_status():
            nonlocal last_log_offset
            try:
                job_info = huggingface_hub.inspect_job(job_id)
                status = job_info.status if hasattr(job_info, "status") else str(job_info)

                # Stream logs
                try:
                    logs = huggingface_hub.fetch_job_logs(job_id)
                    if logs and len(logs) > last_log_offset:
                        new_logs = logs[last_log_offset:]
                        print(new_logs, end="", flush=True)
                        last_log_offset = len(logs)
                except Exception:
                    pass  # Log streaming is best-effort

                if status in ("completed", "COMPLETED"):
                    return "COMPLETED"
                elif status in ("error", "ERROR", "failed", "FAILED"):
                    return f"ERROR: Job failed with status: {status}"
                elif status in ("cancelled", "CANCELLED"):
                    return "ERROR: Job was cancelled"
                else:
                    return None  # Still running

            except Exception as e:
                logger.warning("Status check failed: %s", e)
                return None  # Retry on transient errors

        result = poll_until_done(
            check_status,
            interval=30,
            timeout_seconds=timeout_seconds,
        )

        print()  # Newline after log streaming

        if result == "COMPLETED":
            print(f"  Job {job_id} completed successfully.")
            return 0
        else:
            error_detail = result.replace("ERROR: ", "") if result.startswith("ERROR: ") else result
            print(f"  Job {job_id} failed: {error_detail}")
            return 1

    def _build_uv_script(self, config: CloudTrainingConfig) -> str:
        """
        Build a UV script with PEP 723 inline dependency declarations.

        The script clones the repo and runs the training script within
        the cloud environment. Dependencies are declared inline using
        the PEP 723 format so that uv can install them automatically.

        Args:
            config: Cloud training configuration

        Returns:
            Complete UV script content as a string
        """
        try:
            repo_url = resolve_repo_url()
        except CloudProviderError:
            repo_url = "https://github.com/USER/Toolset-Training.git"
            logger.warning(
                "Could not resolve repo URL. Using placeholder: %s. "
                "Set CLOUD_REPO_URL environment variable.",
                repo_url,
            )

        script = textwrap.dedent(f"""\
            # /// script
            # dependencies = [
            #   "unsloth",
            #   "trl>=0.15",
            #   "transformers",
            #   "datasets",
            #   "peft",
            #   "torch",
            #   "pyyaml",
            #   "wandb",
            #   "hf_transfer",
            # ]
            # ///

            import subprocess
            import sys
            import os

            # Enable fast HF transfers
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            # Clone repo and run training
            repo_url = "{repo_url}"
            method = "{config.method}"

            print(f"Cloning {{repo_url}}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, "/workspace/repo"],
                check=True,
            )

            trainer_dir = f"/workspace/repo/Trainers/rtx3090_{{method}}"
            script_name = f"train_{{method}}.py"

            print(f"Starting {{method.upper()}} training...")
            result = subprocess.run(
                [sys.executable, script_name],
                cwd=trainer_dir,
            )

            sys.exit(result.returncode)
        """)
        return script


def _parse_timeout(timeout_str: str) -> float:
    """
    Parse a timeout string like '4h' or '2.5h' into hours as float.

    Args:
        timeout_str: Timeout string (e.g., '4h', '2.5h', '90m')

    Returns:
        Timeout in hours as a float

    Example:
        _parse_timeout('4h')   -> 4.0
        _parse_timeout('90m')  -> 1.5
        _parse_timeout('2.5h') -> 2.5
    """
    timeout_str = str(timeout_str).strip().lower()

    if timeout_str.endswith("h"):
        try:
            return float(timeout_str[:-1])
        except ValueError:
            return 4.0
    elif timeout_str.endswith("m"):
        try:
            return float(timeout_str[:-1]) / 60.0
        except ValueError:
            return 4.0
    else:
        try:
            return float(timeout_str)
        except ValueError:
            return 4.0
