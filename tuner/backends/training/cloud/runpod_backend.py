"""
Location: tuner/backends/training/cloud/runpod_backend.py

Purpose:
    RunPod cloud training backend implementing ITrainingBackend.
    Creates GPU pods, runs training scripts, streams status updates, and
    ensures pod termination after job completion to prevent billing overruns.

Usage:
    from tuner.backends.training.cloud.runpod_backend import RunPodBackend

    backend = RunPodBackend(repo_root=Path("/path/to/repo"))
    is_valid, msg = backend.validate_environment()
    config = backend.load_config("sft")
    exit_code = backend.execute(config, python_path="python")

Dependencies:
    - tuner.core.interfaces.ITrainingBackend
    - tuner.core.config.TrainingConfig
    - tuner.core.exceptions (CloudProviderError, ConfigurationError)
    - tuner.backends.training.cloud.base_cloud (shared utilities)
    - runpod (optional, checked at validate_environment)
    - Trainers/cloud/cloud_config.yaml
"""

import logging
import os
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from shared.utilities.paths import get_canonical_trainer_dir_name, get_trainer_root
from shared.utilities.unique_ids import unique_utc_timestamp
from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import CloudTrainingConfig, TrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError
from tuner.backends.training.cloud.base_cloud import (
    load_cloud_config,
    load_project_deps,
    resolve_repo_source,
    estimate_cost,
    get_gpu_display_name,
)

logger = logging.getLogger(__name__)

# Polling interval for checking pod status (seconds)
_POLL_INTERVAL = 30

# Maximum time to wait for pod startup (seconds)
_POD_STARTUP_TIMEOUT = 300

# Default maximum training duration (seconds) -- 6 hours
_DEFAULT_TRAINING_TIMEOUT = 21600


class RunPodBackend(ITrainingBackend):
    """
    RunPod cloud training backend.

    Creates a GPU pod, clones the repo, runs training, and terminates
    the pod on completion. All pod lifecycle operations are wrapped in
    try/finally to guarantee pod termination and prevent billing overruns.

    The training script runs as the pod's startup command (docker_args).
    Credentials (HF_TOKEN, WANDB_API_KEY) are passed as pod environment
    variables so they are available inside the container without being
    embedded in the startup command string.
    """

    def __init__(self, repo_root: Path):
        """
        Initialize RunPod backend.

        Args:
            repo_root: Path to repository root directory.
        """
        self.repo_root = Path(repo_root)
        self._cloud_config_cache = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "runpod"

    def get_available_methods(self) -> List[str]:
        """
        Get available training methods.

        Returns:
            List of supported training methods.
        """
        return ["sft", "kto"]

    def _get_cloud_config(self) -> dict:
        """
        Load and cache the cloud section of cloud_config.yaml.

        Uses the shared load_cloud_config from base_cloud.

        Returns:
            Cloud configuration dictionary.
        """
        if self._cloud_config_cache is not None:
            return self._cloud_config_cache

        config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        self._cloud_config_cache = load_cloud_config(config_path)
        return self._cloud_config_cache

    def _get_runpod_config(self) -> dict:
        """
        Extract RunPod-specific settings from cloud config.

        Returns:
            RunPod configuration dictionary with defaults applied.
        """
        return self._get_cloud_config().get("runpod", {})

    def load_config(self, method: str) -> CloudTrainingConfig:
        """
        Load training configuration for the specified method.

        Reads the standard training config from Trainers/{method}/
        and overlays cloud-specific settings from cloud_config.yaml.

        Args:
            method: Training method ('sft' or 'kto').

        Returns:
            CloudTrainingConfig with RunPod-specific fields populated.

        Raises:
            ConfigurationError: If config files are missing or invalid.
        """
        if method not in self.get_available_methods():
            raise ConfigurationError(
                f"Unknown method '{method}' for RunPod backend. "
                f"Available: {self.get_available_methods()}"
            )

        # Load standard training config
        trainer_dir = get_trainer_root(method, self.repo_root)
        config_path = trainer_dir / "configs" / "config.yaml"

        if not config_path.exists():
            raise ConfigurationError(f"Training config not found: {config_path}")

        try:
            with open(config_path) as f:
                training_yaml = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid training config YAML: {e}")

        model_config = training_yaml.get("model", {})
        dataset_config = training_yaml.get("dataset", {})
        training_config = training_yaml.get("training", {})

        # Load cloud-specific RunPod settings
        cloud_config = self._get_cloud_config()
        runpod_config = cloud_config.get("runpod", {})
        artifacts_config = cloud_config.get("artifacts", {})
        repo_source = resolve_repo_source(self.repo_root)

        return CloudTrainingConfig(
            method=method,
            platform="runpod",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=model_config.get("model_name", "Unknown"),
            dataset_file=dataset_config.get("local_file") or f"{dataset_config.get('dataset_name', '')}/{dataset_config.get('dataset_file', 'Unknown')}",
            epochs=training_config.get("num_train_epochs", 1),
            batch_size=training_config.get("per_device_train_batch_size", 4),
            learning_rate=training_config.get("learning_rate", 0.0),
            provider="runpod",
            gpu_type=runpod_config.get("gpu_type_id", "NVIDIA A100 SXM"),
            timeout_hours=runpod_config.get("default_timeout", 7200) / 3600,
            cloud_image=runpod_config.get("default_image", ""),
            push_to_hub=cloud_config.get("push_to_hub", False),
            hub_repo=cloud_config.get("hub_repo"),
            runpod_volume_gb=runpod_config.get("volume_in_gb", 50),
            artifact_backend=runpod_config.get("artifact_backend", "runpod_network_volume"),
            artifact_identifier=runpod_config.get("network_volume_id"),
            artifact_mount_path=runpod_config.get("output_mount_path", "/runpod-volume"),
            publish_final_model=artifacts_config.get("publish_final_model", False),
            publish_target_repo=artifacts_config.get("publish_target_repo"),
            repo_url=repo_source.url,
            repo_branch=repo_source.branch,
            repo_commit=repo_source.commit,
        )

    def validate_environment(self) -> Tuple[bool, str]:
        """
        Validate that RunPod SDK and API key are available.

        Checks:
        1. runpod package is importable
        2. RUNPOD_API_KEY environment variable is set
        3. API key has valid format (non-trivially short)

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            import runpod  # noqa: F401
        except ImportError:
            return False, (
                "runpod package is not installed. "
                "Install with: pip install runpod"
            )

        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            return False, (
                "RUNPOD_API_KEY environment variable is not set. "
                "Get your API key from https://www.runpod.io/console/user/settings "
                "and add it to your .env file."
            )

        # RunPod API keys are typically 32+ hex characters
        stripped = api_key.strip()
        if len(stripped) < 32:
            return False, (
                "RUNPOD_API_KEY appears invalid (too short, expected 32+ characters). "
                "Check your .env file for the correct key."
            )

        # API keys should contain at least some alphabetic characters
        # (pure numeric strings are likely not valid API keys)
        if not any(c.isalpha() for c in stripped):
            return False, (
                "RUNPOD_API_KEY appears invalid (no alphabetic characters). "
                "Check your .env file for the correct key."
            )

        return True, ""

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Execute training on a RunPod GPU pod.

        Creates a pod with the configured GPU type, runs the training script
        as the pod startup command, polls for completion, and terminates the
        pod. The pod is ALWAYS terminated in a finally block to prevent
        billing overruns.

        Args:
            config: Training configuration.
            python_path: Python interpreter path (unused for cloud, kept
                         for interface compatibility).

        Returns:
            Exit code (0 = success, 1 = failure).

        Raises:
            CloudProviderError: If pod creation or training fails.
        """
        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")

        runpod_config = self._get_runpod_config()
        pod_id = None

        try:
            # Build the startup command
            startup_cmd = self._build_startup_command(config, runpod_config)

            # Read pod configuration
            pod_name = self._generate_pod_name(config.method)
            gpu_type = config.gpu_type or runpod_config.get("gpu_type_id", "NVIDIA A100 SXM")
            gpu_count = runpod_config.get("gpu_count", 1)
            image = runpod_config.get(
                "default_image",
                "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
            )
            container_disk_gb = runpod_config.get("container_disk_in_gb", 50)
            cloud_type = runpod_config.get("cloud_type", "COMMUNITY")
            network_volume_id = config.artifact_identifier or runpod_config.get("network_volume_id")
            mount_path = config.artifact_mount_path or runpod_config.get("output_mount_path", "/runpod-volume")

            if not network_volume_id:
                raise CloudProviderError(
                    "RunPod cloud training requires cloud.runpod.network_volume_id in cloud_config.yaml."
                )

            # Build pod environment variables (credentials passed securely)
            pod_env = self._build_pod_env(config)

            # Display configuration
            gpu_display = get_gpu_display_name("runpod", gpu_type)
            cost_str = estimate_cost("runpod", gpu_type, 6.0) or "unknown"

            logger.info("Creating RunPod pod: %s (GPU: %s)", pod_name, gpu_type)
            print(f"\nCreating RunPod pod '{pod_name}'...")
            print(f"  GPU:    {gpu_display} x{gpu_count}")
            print(f"  Image:  {image}")
            print(f"  Network Volume: {network_volume_id}")
            print(f"  Cloud:  {cloud_type}")
            print(f"  Est.:   {cost_str} (6hr max)")

            pod = runpod.create_pod(
                name=pod_name,
                image_name=image,
                gpu_type_id=gpu_type,
                gpu_count=gpu_count,
                container_disk_in_gb=container_disk_gb,
                cloud_type=cloud_type,
                env=pod_env,
                docker_args=startup_cmd,
                network_volume_id=network_volume_id,
                volume_mount_path=mount_path,
            )

            pod_id = pod.get("id")
            if not pod_id:
                raise CloudProviderError(
                    f"Failed to create RunPod pod. Response: {pod}"
                )

            cost_per_hr = pod.get("costPerHr", "unknown")
            print(f"\nPod created: {pod_id}")
            print(f"  Cost: ${cost_per_hr}/hr")
            print(f"  Status: Starting...\n")

            # Wait for pod to reach RUNNING status
            self._wait_for_pod_running(runpod, pod_id)

            # Poll for training completion
            exit_code = self._poll_training(runpod, pod_id, config)
            return exit_code

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            logger.warning("Training interrupted by user (pod: %s)", pod_id)
            return 130

        except (CloudProviderError, ConfigurationError):
            raise

        except Exception as e:
            logger.error("RunPod training failed: %s", e)
            raise CloudProviderError(f"RunPod training failed: {e}")

        finally:
            # ALWAYS terminate the pod to prevent billing overruns
            if pod_id:
                self._terminate_pod(runpod, pod_id)

    def _generate_pod_name(self, method: str) -> str:
        """
        Generate a unique pod name based on method and timestamp.

        Args:
            method: Training method.

        Returns:
            Pod name string.
        """
        timestamp = unique_utc_timestamp(fmt="%Y%m%d-%H%M%S")
        return f"toolset-{method}-{timestamp}"

    def _build_pod_env(self, config: CloudTrainingConfig) -> dict:
        """
        Build environment variables for the pod.

        Passes credentials as pod env vars so they are available inside
        the container without being embedded in the startup command
        (which would appear in logs).

        Returns:
            Dictionary of environment variable key-value pairs.
        """
        env = {}

        # HF_TOKEN for pushing trained models to HuggingFace Hub
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token

        # W&B for optional metrics logging
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            env["WANDB_API_KEY"] = wandb_key

        # GH_TOKEN for private repo cloning
        gh_token = os.environ.get("GH_TOKEN")
        if gh_token:
            env["GH_TOKEN"] = gh_token

        branch = config.repo_branch
        commit = config.repo_commit
        artifact_identifier = config.artifact_identifier
        env["CLOUD_PROVIDER"] = "runpod"
        if config.gpu_type:
            env["CLOUD_GPU_TYPE"] = config.gpu_type
        if branch:
            env["CLOUD_REPO_BRANCH"] = branch
        if commit:
            env["CLOUD_REPO_COMMIT"] = commit
        if artifact_identifier:
            env["CLOUD_ARTIFACT_IDENTIFIER"] = artifact_identifier

        return env

    def _build_startup_command(
        self, config: TrainingConfig, runpod_config: dict
    ) -> str:
        """
        Build the docker startup command for the training pod.

        The unsloth Docker image provides unsloth, transformers, trl, torch
        and CUDA pre-installed. Only project-specific deps are pip-installed.
        Chains: pip install project deps -> git clone -> run training script.

        Args:
            config: Training configuration.
            runpod_config: RunPod-specific cloud config.

        Returns:
            Shell command string for docker_args.
        """
        target_dir = "/workspace/repo"
        trainer_subdir = f"Trainers/{get_canonical_trainer_dir_name(config.method)}"
        run_timestamp = unique_utc_timestamp()

        if not config.repo_url or not config.repo_branch or not config.repo_commit:
            raise CloudProviderError("RunPod requires exact repo source metadata.")

        # Build clone command. Only inject $GH_TOKEN for authenticated cloning
        # when the token is actually set (passed via _build_pod_env() as a pod
        # env var). For public repos or when GH_TOKEN is not configured,
        # clone without authentication to avoid errors from empty token expansion.
        clone_url = config.repo_url
        if config.repo_url.startswith("https://") and os.environ.get("GH_TOKEN"):
            clone_url = config.repo_url.replace("https://", "https://$GH_TOKEN@")

        # Read project-specific deps from cloud_config.yaml (single source of truth)
        cloud_config_path = self.repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        project_deps = load_project_deps(cloud_config_path)

        parts = []

        # Install only project-specific deps (unsloth/torch/trl pre-installed
        # in image). Also honour any extra_setup_commands from config.
        parts.append(f"pip install {' '.join(project_deps)}")
        extra_commands = runpod_config.get("extra_setup_commands", [])
        parts.extend(extra_commands)

        parts.append(f"git clone --branch {config.repo_branch} --depth 1 {clone_url} {target_dir}")
        parts.append(f"cd {target_dir} && git fetch --depth 1 origin {config.repo_commit} && git checkout {config.repo_commit}")
        parts.append(f"cd {target_dir}/{trainer_subdir}")
        training_cmd = (
            f"python train_{config.method}.py "
            f"--run-timestamp {run_timestamp} "
            f"--output-root {config.artifact_mount_path}/outputs "
            f"--cloud-provider {config.provider} "
            f"--artifact-backend {config.artifact_backend} "
            f"{'--publish-final-model' if config.publish_final_model else ''} "
            f"{f'--publish-target-repo {config.publish_target_repo}' if config.publish_target_repo else ''}"
        )
        parts.append(training_cmd)

        return " && ".join(parts)

    def _wait_for_pod_running(self, runpod_module, pod_id: str) -> None:
        """
        Wait for pod to reach RUNNING status.

        Args:
            runpod_module: The imported runpod module.
            pod_id: Pod identifier.

        Raises:
            CloudProviderError: If pod fails to start or enters error state.
        """
        print("Waiting for pod to start...")
        elapsed = 0

        while elapsed < _POD_STARTUP_TIMEOUT:
            try:
                pod = runpod_module.get_pod(pod_id)
            except Exception as e:
                logger.warning("Error checking pod status: %s", e)
                time.sleep(_POLL_INTERVAL)
                elapsed += _POLL_INTERVAL
                continue

            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime")

            if status == "RUNNING" and runtime is not None:
                uptime = runtime.get("uptimeInSeconds", 0)
                print(f"Pod {pod_id} is running (uptime: {uptime}s)")
                return

            if status in ("EXITED", "ERROR", "TERMINATED"):
                raise CloudProviderError(
                    f"Pod {pod_id} failed to start (status: {status}). "
                    f"Check RunPod console for details."
                )

            print(f"  Pod status: {status} (waiting... {elapsed}s)")
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

        raise CloudProviderError(
            f"Pod {pod_id} did not start within {_POD_STARTUP_TIMEOUT}s. "
            f"Check RunPod console for details."
        )

    def _poll_training(
        self, runpod_module, pod_id: str, config: TrainingConfig
    ) -> int:
        """
        Poll pod status until training completes or times out.

        Training is considered complete when the pod exits (the training
        script finishes and the pod stops). GPU utilization and memory
        usage are reported during polling.

        Args:
            runpod_module: The imported runpod module.
            pod_id: Pod identifier.
            config: Training configuration.

        Returns:
            0 on successful completion, 1 on failure.
        """
        print(f"\nTraining in progress on RunPod (method: {config.method})...")
        print("Monitor training at: https://www.runpod.io/console/pods")
        print(f"Pod ID: {pod_id}\n")

        elapsed = 0
        last_status_msg = ""

        while elapsed < _DEFAULT_TRAINING_TIMEOUT:
            try:
                pod = runpod_module.get_pod(pod_id)
            except Exception as e:
                # Network errors during polling -- retry
                logger.warning("Polling error (will retry): %s", e)
                time.sleep(_POLL_INTERVAL)
                elapsed += _POLL_INTERVAL
                continue

            status = pod.get("desiredStatus", "UNKNOWN")
            runtime = pod.get("runtime")

            # Pod exited -- training finished (success or failure)
            if status in ("EXITED", "TERMINATED", "ERROR", "FAILED"):
                print(f"\nPod {pod_id} has stopped (status: {status})")
                if status == "EXITED":
                    print("Training completed successfully.")
                    return 0
                else:
                    print(f"Pod entered terminal state: {status}")
                    return 1

            # Pod still running -- report GPU utilization
            if runtime:
                uptime = runtime.get("uptimeInSeconds", 0)
                gpus = runtime.get("gpus", [])
                gpu_info = ""
                if gpus:
                    gpu_util = gpus[0].get("gpuUtilPercent", 0)
                    mem_util = gpus[0].get("memoryUtilPercent", 0)
                    gpu_info = f" | GPU: {gpu_util}% | VRAM: {mem_util}%"

                status_msg = f"  Running: {uptime}s elapsed{gpu_info}"
                if status_msg != last_status_msg:
                    print(status_msg)
                    last_status_msg = status_msg

            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

        # Timeout -- training ran too long
        print(f"\nTraining timed out after {_DEFAULT_TRAINING_TIMEOUT}s")
        logger.error("Training timed out on pod %s", pod_id)
        return 1

    def _terminate_pod(self, runpod_module, pod_id: str) -> None:
        """
        Terminate a RunPod pod with retry. Always called in finally blocks.

        This is the most critical safety function -- it prevents billing
        overruns by ensuring pods are always cleaned up. Retries up to 3
        times with exponential backoff on transient failures.

        Args:
            runpod_module: The imported runpod module.
            pod_id: Pod identifier.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"\nTerminating pod {pod_id}...")
                runpod_module.terminate_pod(pod_id)
                print(f"Pod {pod_id} terminated successfully.")
                logger.info("Pod %s terminated", pod_id)
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Terminate attempt %d/%d failed for pod %s: %s. Retrying in %ds...",
                        attempt + 1, max_attempts, pod_id, e, wait,
                    )
                    time.sleep(wait)
                else:
                    # Final attempt failed -- log but don't raise (finally block)
                    print(
                        f"\nWARNING: Failed to terminate pod {pod_id} after "
                        f"{max_attempts} attempts: {e}\n"
                        f"IMPORTANT: Manually terminate this pod at "
                        f"https://www.runpod.io/console/pods to avoid charges!"
                    )
                    logger.error(
                        "CRITICAL: Failed to terminate pod %s after %d attempts: %s. "
                        "Manual termination required!", pod_id, max_attempts, e
                    )
