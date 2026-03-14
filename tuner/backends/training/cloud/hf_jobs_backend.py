"""
HuggingFace Jobs cloud training backend.

Location: tuner/backends/training/cloud/hf_jobs_backend.py
Purpose: Submit and monitor training jobs via HuggingFace Jobs API
Used by: CloudTrainHandler when user selects HF Jobs provider

Implements ITrainingBackend for HuggingFace Jobs. Training jobs are
submitted via huggingface_hub.run_job(image, command, flavor) which
provisions GPU hardware and runs a Docker container with the specified
command. The command clones the repo and runs the training script.

Requirements:
- HF_TOKEN environment variable set (requires HF Pro subscription)
- huggingface_hub >= 0.27.0 installed
"""

import logging
import os
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from shared.utilities.paths import get_canonical_trainer_dir_name, get_trainer_root
from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import TrainingConfig, CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError

from .base_cloud import load_cloud_config, load_project_deps, poll_until_done, resolve_repo_source

logger = logging.getLogger(__name__)

# Default HF Jobs settings
DEFAULT_FLAVOR = "a10g-small"
DEFAULT_TIMEOUT = "4h"
DEFAULT_IMAGE = "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"


class HFJobsBackend(ITrainingBackend):
    """
    HuggingFace Jobs training backend.

    Submits training jobs to HuggingFace's managed GPU infrastructure using
    huggingface_hub.run_job(image, command, flavor). The job runs a Docker
    container that clones the repo and executes the training script.

    The execute() flow:
    1. Build a shell command that installs deps, clones repo, runs training
    2. Submit via huggingface_hub.run_job(image=..., command=["bash", "-c", ...])
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
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if not hf_token:
            return False, (
                "HF_TOKEN not set. Required for HuggingFace Jobs. "
                "Set HF_TOKEN (or HF_API_KEY) in your .env file or environment."
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
        trainer_dir = get_trainer_root(method, self.repo_root)
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
        artifacts_config = cloud_config.get("artifacts", {})
        repo_source = resolve_repo_source(self.repo_root)

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
            dataset_file=dataset_config.get("local_file") or f"{dataset_config.get('dataset_name', '')}/{dataset_config.get('dataset_file', 'Unknown')}",
            epochs=training_config.get("num_train_epochs", 1),
            batch_size=training_config.get("per_device_train_batch_size", 4),
            learning_rate=training_config.get("learning_rate", 0.0),
            provider="hf_jobs",
            gpu_type=flavor,
            timeout_hours=timeout_hours,
            cloud_image=image,
            push_to_hub=cloud_config.get("push_to_hub", False),
            hub_repo=cloud_config.get("hub_repo"),
            hf_flavor=flavor,
            artifact_backend=hf_config.get("artifact_backend", "hf_bucket"),
            artifact_identifier=hf_config.get("artifact_identifier"),
            artifact_mount_path=hf_config.get("output_root", "/workspace/outputs"),
            publish_final_model=artifacts_config.get("publish_final_model", False),
            publish_target_repo=artifacts_config.get("publish_target_repo"),
            repo_url=repo_source.url,
            repo_branch=repo_source.branch,
            repo_commit=repo_source.commit,
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Submit training job to HuggingFace Jobs and poll until completion.

        Uses huggingface_hub.run_job(image, command, flavor) to submit a
        Docker-based job that clones the repo and runs the training script.
        Polls inspect_job() for status and streams fetch_job_logs() output.

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

        if not isinstance(config, CloudTrainingConfig):
            raise CloudProviderError(
                "HF Jobs backend requires CloudTrainingConfig, "
                f"got {type(config).__name__}"
            )

        # Build the training command
        training_command = self._build_training_command(config)
        image = config.cloud_image or DEFAULT_IMAGE
        flavor = config.hf_flavor or config.gpu_type

        logger.info("Submitting job to HuggingFace Jobs...")
        logger.info("Image: %s", image)
        logger.info("Flavor: %s", flavor)
        logger.info("Timeout: %.1f hours", config.timeout_hours)

        try:
            job = huggingface_hub.run_job(
                image=image,
                command=["bash", "-c", training_command],
                flavor=flavor,
            )
            # JobInfo has .id and .url attributes
            job_id = job.id if hasattr(job, "id") else str(job)
            job_url = getattr(job, "url", None)
            print(f"  Job submitted: {job_id}")
            if job_url:
                print(f"  Monitor at: {job_url}")
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
                job_info = huggingface_hub.inspect_job(job_id=job_id)
                # JobInfo.status is a JobStatus with .stage attribute
                status_obj = getattr(job_info, "status", None)
                if status_obj and hasattr(status_obj, "stage"):
                    status = status_obj.stage
                else:
                    status = str(status_obj) if status_obj else "UNKNOWN"

                # Stream logs (best-effort)
                try:
                    logs = huggingface_hub.fetch_job_logs(job_id=job_id)
                    if logs and len(logs) > last_log_offset:
                        new_logs = logs[last_log_offset:]
                        print(new_logs, end="", flush=True)
                        last_log_offset = len(logs)
                except Exception:
                    pass

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

    def _build_training_command(self, config: CloudTrainingConfig) -> str:
        """
        Build a shell command string for the HF Jobs container.

        The command installs training dependencies, clones the repo,
        and runs the training script. HF_TOKEN is available in the
        container environment automatically (HF Jobs injects it).

        Args:
            config: Cloud training configuration

        Returns:
            Shell command string to pass as ["bash", "-c", command]
        """
        if not config.repo_url or not config.repo_branch or not config.repo_commit:
            raise CloudProviderError("HF Jobs requires exact repo source metadata.")
        if not config.artifact_identifier:
            raise CloudProviderError("HF Jobs requires an artifact bucket identifier.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_slug = f"{timestamp}-{config.repo_commit[:8]}"
        artifact_prefix = f"runs/{config.provider}/{config.method}/{run_slug}"

        # Read project-specific deps from cloud_config.yaml (single source of truth)
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        project_deps = load_project_deps(cloud_config_path)

        parts = [
            # Install project-specific deps only; unsloth, trl, transformers,
            # datasets, peft, and PyTorch are pre-installed in the Docker image
            f"pip install {' '.join(project_deps)}",
            # Enable fast HF transfers
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            # Clone repo
            f"git clone --branch {config.repo_branch} --depth 1 {config.repo_url} /workspace/repo",
            f"cd /workspace/repo && git checkout {config.repo_commit}",
            # Run training
            f"cd /workspace/repo/Trainers/{get_canonical_trainer_dir_name(config.method)}",
            "python "
            f"train_{config.method}.py "
            f"--run-timestamp {timestamp} "
            f"--output-root {config.artifact_mount_path} "
            f"--cloud-provider {config.provider} "
            f"--artifact-backend {config.artifact_backend} "
            f"--artifact-bucket {config.artifact_identifier} "
            f"--artifact-prefix {artifact_prefix} "
            f"{'--publish-final-model' if config.publish_final_model else ''} "
            f"{f'--publish-target-repo {config.publish_target_repo}' if config.publish_target_repo else ''}",
        ]

        return " && ".join(parts)


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
