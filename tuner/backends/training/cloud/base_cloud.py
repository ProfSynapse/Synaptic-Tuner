"""
Shared utilities for cloud training backends.

Location: tuner/backends/training/cloud/base_cloud.py
Purpose: Helper functions used by all cloud provider backends (HF Jobs, Modal, RunPod)
Used by: hf_jobs_backend.py, modal_backend.py, runpod_backend.py

This module provides:
- Cloud config loading from cloud_config.yaml
- Git repo URL resolution for code sync
- Generic job polling loop with timeout
- Cost estimation for cloud GPU usage
"""

import logging
import os
import subprocess
import time
import yaml
from pathlib import Path
from typing import Callable, Dict, Optional

from tuner.core.exceptions import CloudProviderError

logger = logging.getLogger(__name__)

# GPU pricing reference (approximate $/hr as of 2026-02)
GPU_PRICING = {
    "hf_jobs": {
        "t4-small": {"name": "T4 (16GB)", "price": 0.40},
        "t4-medium": {"name": "T4 x2 (32GB)", "price": 0.80},
        "a10g-small": {"name": "A10G (24GB)", "price": 1.10},
        "a10g-large": {"name": "A10G x4 (96GB)", "price": 4.40},
        "a100-large": {"name": "A100 (80GB)", "price": 2.50},
    },
    "modal": {
        "T4": {"name": "T4 (16GB)", "price": 0.59},
        "L4": {"name": "L4 (24GB)", "price": 0.73},
        "A10G": {"name": "A10G (24GB)", "price": 1.10},
        "L40S": {"name": "L40S (48GB)", "price": 1.40},
        "A100-40GB": {"name": "A100 (40GB)", "price": 2.78},
        "A100-80GB": {"name": "A100 (80GB)", "price": 3.72},
        "H100": {"name": "H100 (80GB)", "price": 4.89},
    },
    "runpod": {
        "NVIDIA RTX A6000": {"name": "RTX A6000 (48GB)", "price": 0.79},
        "NVIDIA A100 80GB PCIe": {"name": "A100 (80GB)", "price": 1.64},
        "NVIDIA H100 80GB HBM3": {"name": "H100 (80GB)", "price": 3.89},
    },
}


def load_cloud_config(cloud_config_path: Path) -> dict:
    """
    Load cloud_config.yaml and return the cloud configuration section.

    Args:
        cloud_config_path: Path to cloud_config.yaml file

    Returns:
        Dictionary of cloud configuration settings. Returns empty dict
        if file doesn't exist.

    Example:
        config = load_cloud_config(repo_root / "Trainers" / "cloud" / "cloud_config.yaml")
        hf_settings = config.get('hf_jobs', {})
    """
    if not cloud_config_path.exists():
        return {}
    try:
        with open(cloud_config_path) as f:
            config = yaml.safe_load(f)
        return config.get("cloud", {})
    except Exception as e:
        logger.warning("Failed to load cloud config %s: %s", cloud_config_path, e)
        return {}


def resolve_repo_url() -> str:
    """
    Get the repository URL for code sync to cloud jobs.

    Checks CLOUD_REPO_URL environment variable first, then falls back
    to the git remote origin URL.

    Returns:
        Repository URL string

    Raises:
        CloudProviderError: If no repo URL can be determined
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

    raise CloudProviderError(
        "Cannot determine repo URL for cloud code sync. "
        "Set CLOUD_REPO_URL environment variable or ensure git remote 'origin' is configured."
    )


def poll_until_done(
    check_fn: Callable[[], Optional[str]],
    interval: int = 30,
    timeout_seconds: int = 14400,
) -> str:
    """
    Poll check_fn until it returns a terminal status.

    Calls check_fn repeatedly at the given interval until it returns
    a non-None value (terminal status) or the timeout is exceeded.

    Args:
        check_fn: Callable that returns:
            - None if the job is still running (keep polling)
            - "COMPLETED" on successful completion
            - "ERROR: <message>" on failure
        interval: Seconds between polls (default: 30)
        timeout_seconds: Maximum total polling time in seconds (default: 14400 = 4 hours)

    Returns:
        Terminal status string from check_fn, or "ERROR: Timeout exceeded"

    Example:
        def check_job():
            status = api.get_job_status(job_id)
            if status == "running":
                return None
            elif status == "completed":
                return "COMPLETED"
            else:
                return f"ERROR: {status}"

        result = poll_until_done(check_job, interval=30, timeout_seconds=7200)
    """
    elapsed = 0
    consecutive_errors = 0
    max_consecutive = 3

    while elapsed < timeout_seconds:
        try:
            status = check_fn()
            consecutive_errors = 0  # Reset on success
            if status is not None:
                return status
        except Exception as e:
            consecutive_errors += 1
            # Persistent errors should not be retried
            error_lower = str(e).lower()
            if any(term in error_lower for term in ("unauthorized", "not found", "forbidden", "invalid")):
                raise
            # Too many consecutive transient errors -- give up
            if consecutive_errors >= max_consecutive:
                raise
            logger.warning(
                "Poll check failed (%d/%d, will retry): %s",
                consecutive_errors, max_consecutive, e,
            )

        time.sleep(interval)
        elapsed += interval

        # Log progress every 5 minutes
        if elapsed % 300 == 0:
            minutes = elapsed // 60
            logger.info("Job polling: %d minutes elapsed", minutes)

    return "ERROR: Timeout exceeded"


def estimate_cost(provider: str, gpu_type: str, timeout_hours: float) -> Optional[str]:
    """
    Estimate cloud training cost based on provider, GPU, and timeout.

    Args:
        provider: Cloud provider identifier ('hf_jobs', 'modal', 'runpod')
        gpu_type: Provider-specific GPU identifier
        timeout_hours: Maximum job duration in hours

    Returns:
        Formatted cost estimate string (e.g., "~$4.40"), or None if
        pricing info is not available for the given GPU type.

    Example:
        cost = estimate_cost("hf_jobs", "a10g-small", 4.0)
        # Returns "~$4.40"
    """
    provider_pricing = GPU_PRICING.get(provider, {})
    gpu_info = provider_pricing.get(gpu_type)
    if not gpu_info:
        return None
    estimated = gpu_info["price"] * timeout_hours
    return f"~${estimated:.2f}"


def get_gpu_display_name(provider: str, gpu_type: str) -> str:
    """
    Get human-readable GPU name for display.

    Args:
        provider: Cloud provider identifier
        gpu_type: Provider-specific GPU identifier

    Returns:
        Display name (e.g., "A10G (24GB)") or the raw gpu_type if not found.
    """
    provider_pricing = GPU_PRICING.get(provider, {})
    gpu_info = provider_pricing.get(gpu_type)
    if gpu_info:
        return gpu_info["name"]
    return gpu_type
