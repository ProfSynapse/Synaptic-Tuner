"""
HuggingFace Jobs cloud training backend.

Location: tuner/backends/training/cloud/hf_jobs_backend.py
Purpose: Submit and monitor training jobs via HuggingFace Jobs API
Used by: CloudTrainHandler when user selects HF Jobs provider

Implements ITrainingBackend for HuggingFace Jobs. Training jobs are
submitted via huggingface_hub.run_job(image, command, flavor) which
provisions GPU hardware and runs a Docker container with the specified
command. The command clones the repo and runs the training script.

This module serves as a facade: HFJobsBackend composes four mixins that
encapsulate distinct concerns (command building, job watching, bucket
operations, post-training actions). The external API is unchanged.

Requirements:
- HF_TOKEN environment variable set (requires HF Pro subscription)
- huggingface_hub >= 0.27.0 installed
"""

import logging
import os
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

from shared.cloud_artifacts import normalize_hf_bucket_id
from shared.utilities.paths import get_trainer_root
from tuner.cloud import (
    CloudJobSpec,
    HFJobExecutor,
    build_bash_command,
    build_hf_job_secrets,
    load_huggingface_hub,
    resolve_hf_bucket_id,
)
from tuner.ui import print_config
from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import TrainingConfig, CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError
from tuner.core.interfaces import ExecuteResult

from .base_cloud import (
    load_cloud_config,
    poll_until_done,
    resolve_cloud_image,
    resolve_repo_source,
)
from ._hf_command_builder import HFCommandBuilderMixin
from ._hf_job_watcher import HFJobWatcherMixin
from ._hf_bucket_ops import HFBucketOpsMixin
from ._hf_post_training import HFPostTrainingMixin

logger = logging.getLogger(__name__)

# Default HF Jobs settings
DEFAULT_FLAVOR = "a10g-small"
DEFAULT_TIMEOUT = "4h"
DEFAULT_IMAGE = "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39"


class HFJobsBackend(
    HFCommandBuilderMixin,
    HFJobWatcherMixin,
    HFBucketOpsMixin,
    HFPostTrainingMixin,
    ITrainingBackend,
):
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

    Mixin dependencies (cross-mixin method calls):
    - HFCommandBuilderMixin  -> HFBucketOpsMixin._build_remote_run_uri
    - HFJobWatcherMixin      -> HFBucketOpsMixin._recover_completed_run_from_bucket
    - HFJobWatcherMixin      -> HFPostTrainingMixin._finalize_completed_job
    - HFPostTrainingMixin    -> HFBucketOpsMixin._build_remote_run_uri,
                                ._local_download_run_dir, ._download_completed_run
    All mixins expect self.repo_root (set by __init__).
    """

    def __init__(self, repo_root: Path):
        """
        Initialize HF Jobs backend.

        Args:
            repo_root: Path to repository root directory
        """
        self.repo_root = Path(repo_root)
        self.show_post_training_actions = True
        self.last_artifact_prefix: Optional[str] = None
        self.last_bucket_id: Optional[str] = None
        self.last_job_id: Optional[str] = None

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
            List of method names: ['sft', 'kto', 'grpo', 'dpo']
        """
        return ["sft", "kto", "grpo", "dpo"]

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
            load_huggingface_hub(require_apis=("run_job", "create_bucket"))
        except CloudProviderError as exc:
            return False, str(exc)

        return True, ""

    def load_config(self, method: str) -> CloudTrainingConfig:
        """
        Load training configuration with cloud overlay.

        Loads the standard training config from the trainer directory,
        then overlays cloud-specific settings from cloud_config.yaml.

        Args:
            method: Training method ('sft', 'kto', or 'grpo')

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
        config_path = trainer_dir / "configs" / self._config_filename_for_method(method)

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
        evolutionary_config = config.get("evolutionary", {})
        evolutionary_strategy = evolutionary_config.get("strategy", {}) or {}
        evolutionary_selection = evolutionary_config.get("selection", {}) or {}
        evolutionary_logging = evolutionary_config.get("logging", {}) or {}
        evolutionary_params = evolutionary_strategy.get("params", {}) or {}

        # Load cloud overlay
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        cloud_config = load_cloud_config(cloud_config_path)
        hf_config = cloud_config.get("hf_jobs", {})
        artifacts_config = cloud_config.get("artifacts", {})
        repo_source = resolve_repo_source(self.repo_root)

        flavor = hf_config.get("flavor", DEFAULT_FLAVOR)
        timeout_str = hf_config.get("timeout", DEFAULT_TIMEOUT)
        image, image_profile = resolve_cloud_image(
            cloud_config_path,
            explicit_image=hf_config.get("image"),
            default_profile=hf_config.get("image_profile"),
            fallback_image=DEFAULT_IMAGE,
        )

        # Parse timeout string (e.g., "4h" -> 4.0)
        timeout_hours = _parse_timeout(timeout_str)

        return CloudTrainingConfig(
            method=method,
            platform="hf_jobs",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get("model_name", "Unknown"),
            dataset_file=self._dataset_display(dataset_config),
            dataset_name=dataset_config.get("dataset_name"),
            epochs=training_config.get("num_train_epochs", 1),
            batch_size=training_config.get("per_device_train_batch_size", 4),
            learning_rate=training_config.get("learning_rate", 0.0),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps"),
            chat_template_kwargs=training_config.get("chat_template_kwargs"),
            save_steps=training_config.get("save_steps"),
            save_total_limit=training_config.get("save_total_limit"),
            max_seq_length=training_config.get("max_seq_length") or training_config.get("max_prompt_length") or model_config.get("max_seq_length"),
            load_in_4bit=model_config.get("load_in_4bit"),
            lora_r=config.get("lora", {}).get("r"),
            lora_alpha=config.get("lora", {}).get("lora_alpha"),
            lora_dropout=config.get("lora", {}).get("lora_dropout"),
            use_dora=bool(config.get("lora", {}).get("use_dora", False)),
            use_rslora=bool(config.get("lora", {}).get("use_rslora", False)),
            init_lora_weights=config.get("lora", {}).get("init_lora_weights"),
            lora_target_modules=model_config.get("target_modules") or config.get("lora", {}).get("target_modules"),
            evolutionary_enabled=bool(evolutionary_config.get("enabled", False)),
            evolutionary_candidates=evolutionary_config.get("candidates"),
            evolutionary_eval_batch_size=evolutionary_config.get("eval_batch_size"),
            evolutionary_validation_config=evolutionary_config.get("validation_config"),
            evolutionary_strategy=evolutionary_strategy.get("type"),
            evolutionary_noise_scale=evolutionary_params.get("noise_scale"),
            evolutionary_max_grad_norm=evolutionary_params.get("max_grad_norm"),
            evolutionary_scale_factors=evolutionary_params.get("scale_factors"),
            evolutionary_selection_method=evolutionary_selection.get("method"),
            evolutionary_min_improvement=evolutionary_selection.get("min_improvement"),
            evolutionary_min_relative_improvement=evolutionary_selection.get("min_relative_improvement"),
            evolutionary_noise_floor_epsilon=evolutionary_selection.get("noise_floor_epsilon"),
            evolutionary_eval_frequency=evolutionary_config.get("eval_frequency"),
            evolutionary_warmup_steps=evolutionary_config.get("warmup_steps"),
            evolutionary_cache_baseline=evolutionary_config.get("cache_baseline"),
            evolutionary_log_candidates=evolutionary_logging.get("candidates"),
            evolutionary_log_selected=evolutionary_logging.get("selected"),
            provider="hf_jobs",
            gpu_type=flavor,
            timeout_hours=timeout_hours,
            cloud_image=image,
            cloud_image_profile=image_profile,
            push_to_hub=cloud_config.get("push_to_hub", False),
            hub_repo=cloud_config.get("hub_repo"),
            hf_flavor=flavor,
            artifact_backend=hf_config.get("artifact_backend", "hf_bucket"),
            artifact_identifier=hf_config.get("artifact_identifier"),
            artifact_mount_path=hf_config.get("output_root", "/workspace/outputs"),
            pip_packages=hf_config.get("pip_packages"),
            publish_final_model=artifacts_config.get("publish_final_model", False),
            publish_target_repo=artifacts_config.get("publish_target_repo"),
            repo_url=repo_source.url,
            repo_branch=repo_source.branch,
            repo_commit=repo_source.commit,
        )

    def execute(self, config: TrainingConfig, python_path: str) -> ExecuteResult:
        """
        Submit training job to HuggingFace Jobs and poll until completion.

        Uses huggingface_hub.run_job(image, command, flavor) to submit a
        Docker-based job that clones the repo and runs the training script.
        Polls inspect_job() for status and streams fetch_job_logs() output.

        Args:
            config: Training configuration (should be CloudTrainingConfig)
            python_path: Not used for cloud execution (kept for interface compat)

        Returns:
            ExecuteResult with exit_code, artifact_prefix, bucket_id, and
            job_id.  Compares equal to ``int`` so callers that only check
            ``result == 0`` continue to work unchanged.
        """
        huggingface_hub = load_huggingface_hub(require_apis=("run_job", "create_bucket"))

        if not isinstance(config, CloudTrainingConfig):
            raise CloudProviderError(
                "HF Jobs backend requires CloudTrainingConfig, "
                f"got {type(config).__name__}"
            )

        self.last_artifact_prefix = None
        self.last_bucket_id = None
        self.last_job_id = None

        if config.artifact_backend == "hf_bucket":
            self._ensure_hf_bucket(config, huggingface_hub)

        timestamp = self._new_run_timestamp()
        artifact_prefix = self._build_artifact_prefix(config, timestamp)
        self.last_artifact_prefix = artifact_prefix
        self.last_bucket_id = config.artifact_identifier

        # Build the training command
        training_command = self._build_training_command(config, timestamp=timestamp)
        image = config.cloud_image or DEFAULT_IMAGE
        flavor = config.hf_flavor or config.gpu_type

        logger.info("Submitting job to HuggingFace Jobs...")
        logger.info("Image: %s", image)
        logger.info("Flavor: %s", flavor)
        logger.info("Timeout: %.1f hours", config.timeout_hours)

        try:
            submission = HFJobExecutor(huggingface_hub).submit(
                CloudJobSpec(
                    provider=self.name,
                    image=image,
                    command=build_bash_command([training_command]),
                    flavor=flavor,
                    timeout_hours=config.timeout_hours,
                    secrets=build_hf_job_secrets(),
                    labels={
                        "task": "training",
                        "method": config.method,
                        "provider": config.provider,
                    },
                )
            )
            job_id = submission.job_id
            self.last_job_id = job_id
            job_url = submission.job_url
            print(f"  Job submitted: {job_id}")
            if job_url:
                print(f"  Monitor at: {job_url}")
            print(f"  Polling for completion (every 30s, timeout: {config.timeout_hours}h)...")
            print()

        except CloudProviderError as exc:
            raise CloudProviderError(f"Failed to submit HF Jobs training: {exc}") from exc

        def _build_result(exit_code: int) -> ExecuteResult:
            return ExecuteResult(
                exit_code=exit_code,
                artifact_prefix=self.last_artifact_prefix,
                bucket_id=self.last_bucket_id,
                job_id=self.last_job_id,
            )

        if self._should_use_remote_dashboard(config):
            exit_code = self._watch_job_with_remote_dashboard(
                config=config,
                huggingface_hub=huggingface_hub,
                job_id=job_id,
                artifact_prefix=artifact_prefix,
            )
            return _build_result(exit_code)

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
                # Let poll_until_done handle persistent vs transient classification
                raise

        result = poll_until_done(
            check_status,
            interval=30,
            timeout_seconds=timeout_seconds,
        )

        print()  # Newline after log streaming

        if result == "COMPLETED":
            print(f"  Job {job_id} completed successfully.")
            return _build_result(self._finalize_completed_job(config=config, artifact_prefix=artifact_prefix))

        recovered = self._recover_completed_run_from_bucket(
            config=config,
            artifact_prefix=artifact_prefix,
        )
        if recovered:
            print(f"  Job {job_id} appears complete based on synced artifacts.")
            return _build_result(self._finalize_completed_job(config=config, artifact_prefix=artifact_prefix))

        error_detail = result.replace("ERROR: ", "") if result.startswith("ERROR: ") else result
        print(f"  Job {job_id} failed: {error_detail}")
        return _build_result(1)


def _parse_timeout(timeout_str: str) -> float:
    """
    Parse a timeout string like '4h' or '2.5h' into hours as float.

    Args:
        timeout_str: Timeout string (e.g., '4h', '2.5h', '90m')

    Returns:
        Timeout in hours as a float (defaults to 4.0 on invalid input)

    Example:
        _parse_timeout('4h')   -> 4.0
        _parse_timeout('90m')  -> 1.5
        _parse_timeout('2.5h') -> 2.5
    """
    raw = str(timeout_str).strip().lower()
    default = 4.0

    if raw.endswith("h"):
        try:
            return float(raw[:-1])
        except ValueError:
            logger.warning("Invalid timeout value '%s', defaulting to %.1fh", timeout_str, default)
            return default
    elif raw.endswith("m"):
        try:
            return float(raw[:-1]) / 60.0
        except ValueError:
            logger.warning("Invalid timeout value '%s', defaulting to %.1fh", timeout_str, default)
            return default
    else:
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid timeout value '%s', defaulting to %.1fh", timeout_str, default)
            return default
