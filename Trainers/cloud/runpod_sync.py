"""
Location: Trainers/cloud/runpod_sync.py

Purpose:
    Utility module for RunPod code synchronization operations.
    Provides functions for building pod startup commands that clone the
    repository and run training scripts. Also handles result retrieval
    and pod readiness polling.

Usage:
    from Trainers.cloud.runpod_sync import (
        build_training_startup_command,
        wait_for_pod_ready,
        sync_results_from_pod,
    )

    cmd = build_training_startup_command(method="sft")
    wait_for_pod_ready(pod_id, timeout=300)

Dependencies:
    - runpod (optional, guarded import)
    - subprocess (for git remote detection)
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_runpod():
    """Import and configure the runpod SDK. Raises ImportError if unavailable."""
    try:
        import runpod
    except ImportError:
        raise ImportError(
            "runpod package is not installed. "
            "Install it with: pip install runpod"
        )
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key:
        runpod.api_key = api_key
    return runpod


def _resolve_repo_url() -> str:
    """
    Get the git remote origin URL for code sync.

    Checks CLOUD_REPO_URL env var first, then falls back to git remote.

    Returns:
        The repository URL as a string.

    Raises:
        ValueError: If no repo URL can be determined.
    """
    url = os.environ.get("CLOUD_REPO_URL")
    if url:
        return url

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    raise ValueError(
        "Cannot determine repo URL for code sync. "
        "Set CLOUD_REPO_URL or ensure a git remote 'origin' is configured."
    )


def _build_clone_command(repo_url: str, target_dir: str) -> str:
    """
    Build a git clone command, injecting GH_TOKEN for private repos if set.

    Args:
        repo_url: The repository URL to clone.
        target_dir: Where to clone on the pod.

    Returns:
        Shell command string for cloning.
    """
    gh_token = os.environ.get("GH_TOKEN")
    if gh_token and repo_url.startswith("https://"):
        authenticated_url = repo_url.replace(
            "https://", f"https://{gh_token}@"
        )
        return f"git clone --depth 1 {authenticated_url} {target_dir}"

    return f"git clone --depth 1 {repo_url} {target_dir}"


def wait_for_pod_ready(pod_id: str, timeout: int = 300, interval: int = 10) -> bool:
    """
    Poll until pod status is RUNNING.

    Args:
        pod_id: The RunPod pod identifier.
        timeout: Maximum seconds to wait (default 300s / 5 min).
        interval: Seconds between polls (default 10s).

    Returns:
        True if pod reached RUNNING status within timeout.

    Raises:
        TimeoutError: If pod does not reach RUNNING within timeout.
        RuntimeError: If pod enters an error state.
    """
    runpod = _get_runpod()
    elapsed = 0

    while elapsed < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime = pod.get("runtime")

        if status == "RUNNING" and runtime is not None:
            logger.info("Pod %s is ready (uptime: %ss)", pod_id,
                        runtime.get("uptimeInSeconds", 0))
            return True

        if status in ("EXITED", "ERROR", "TERMINATED"):
            raise RuntimeError(
                f"Pod {pod_id} entered terminal state: {status}"
            )

        logger.info("Waiting for pod %s (status: %s, elapsed: %ds)...",
                     pod_id, status, elapsed)
        time.sleep(interval)
        elapsed += interval

    raise TimeoutError(
        f"Pod {pod_id} did not reach RUNNING status within {timeout}s"
    )


def sync_results_from_pod(
    pod_id: str,
    remote_output_dir: str,
    local_output_dir: str,
) -> bool:
    """
    Verify training results are available from a RunPod pod.

    Training scripts push directly to HuggingFace Hub using HF_TOKEN
    (passed as a pod env var). This function checks whether HF_TOKEN
    was set (results should be on the Hub) or warns the user to
    download manually from the RunPod console.

    Args:
        pod_id: The RunPod pod identifier.
        remote_output_dir: Path on pod where outputs are stored.
        local_output_dir: Local path to save results.

    Returns:
        True if results should be available (via HF Hub).
    """
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info(
            "Training outputs should be pushed to HuggingFace Hub "
            "automatically (HF_TOKEN was set in pod environment)."
        )
        return True

    logger.warning(
        "HF_TOKEN not set -- training outputs remain on pod %s at %s. "
        "Use the RunPod web console to download results before "
        "terminating the pod.",
        pod_id, remote_output_dir,
    )

    Path(local_output_dir).mkdir(parents=True, exist_ok=True)
    return False


def build_training_startup_command(
    method: str,
    setup_commands: Optional[list] = None,
    repo_url: Optional[str] = None,
    target_dir: str = "/workspace/repo",
) -> str:
    """
    Build the full startup command for a RunPod training pod.

    Chains together: dependency installation, repo clone, and training
    script execution into a single shell command for docker_args.

    Args:
        method: Training method ('sft' or 'kto').
        setup_commands: List of setup commands (pip installs, etc.).
        repo_url: Optional override for the repo URL.
        target_dir: Directory on the pod to clone into.

    Returns:
        Combined shell command string.

    Raises:
        ValueError: If repo URL cannot be determined.
    """
    url = repo_url or _resolve_repo_url()
    clone_cmd = _build_clone_command(url, target_dir)

    parts = []

    if setup_commands:
        parts.extend(setup_commands)

    parts.append(clone_cmd)

    trainer_subdir = f"Trainers/rtx3090_{method}"
    parts.append(f"cd {target_dir}/{trainer_subdir}")
    parts.append(f"python train_{method}.py")

    full_command = " && ".join(parts)
    logger.info("Built startup command for %s training (%d chars)",
                method, len(full_command))
    return full_command
