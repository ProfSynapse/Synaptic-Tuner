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
from typing import Callable, Optional

from tuner.core.exceptions import CloudProviderError

logger = logging.getLogger(__name__)

# Module-level pricing cache (loaded from cloud_config.yaml on first access)
_GPU_PRICING_CACHE: Optional[dict] = None


def _find_cloud_config() -> Path:
    """
    Locate cloud_config.yaml relative to this module.

    Walks up from base_cloud.py to the repo root and looks for
    Trainers/cloud/cloud_config.yaml.

    Returns:
        Path to cloud_config.yaml (may not exist).
    """
    # base_cloud.py is at tuner/backends/training/cloud/base_cloud.py
    # repo root is 4 levels up
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "Trainers" / "cloud" / "cloud_config.yaml"


def load_gpu_pricing(cloud_config_path: Optional[Path] = None) -> dict:
    """
    Load GPU pricing data from cloud_config.yaml.

    Reads the 'pricing' section from the cloud config file. Results are
    cached at module level so the file is only read once per process.

    Args:
        cloud_config_path: Optional explicit path to cloud_config.yaml.
            If None, auto-detected relative to this module.

    Returns:
        Dictionary keyed by provider with nested GPU pricing dicts.
        Each GPU entry has 'name' (str) and 'price' (float).
        Returns empty dict if file not found or no pricing section.

    Example:
        pricing = load_gpu_pricing()
        modal_h100 = pricing["modal"]["H100"]
        # {"name": "H100 (80GB)", "price": 4.89}
    """
    global _GPU_PRICING_CACHE
    if _GPU_PRICING_CACHE is not None:
        return _GPU_PRICING_CACHE

    config_path = cloud_config_path or _find_cloud_config()
    if not config_path.exists():
        _GPU_PRICING_CACHE = {}
        return _GPU_PRICING_CACHE

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _GPU_PRICING_CACHE = config.get("pricing", {}) if config else {}
    except Exception as e:
        logger.warning("Failed to load GPU pricing from %s: %s", config_path, e)
        _GPU_PRICING_CACHE = {}

    return _GPU_PRICING_CACHE


# Supported training methods across all cloud backends
SUPPORTED_METHODS = ("sft", "kto")


def validate_training_method(method: str, backend_name: str) -> None:
    """
    Validate that a training method is supported.

    Args:
        method: Training method to validate
        backend_name: Name of the backend (for error messages)

    Raises:
        CloudProviderError: If the method is not supported
    """
    if method not in SUPPORTED_METHODS:
        raise CloudProviderError(
            f"Unknown training method '{method}' for {backend_name} backend. "
            f"Supported methods: {', '.join(SUPPORTED_METHODS)}"
        )


def load_gpu_tiers(cloud_config_path: Path) -> dict:
    """
    Load GPU tier definitions from cloud_config.yaml.

    GPU tiers map logical tier names (budget, standard, performance) to
    provider-specific GPU identifiers, enabling provider-agnostic GPU
    selection in the CLI.

    Args:
        cloud_config_path: Path to cloud_config.yaml file

    Returns:
        Dictionary of tier definitions. Returns empty dict if file
        doesn't exist or has no gpu_tiers section.

    Example:
        tiers = load_gpu_tiers(repo_root / "Trainers" / "cloud" / "cloud_config.yaml")
        modal_gpu = tiers["standard"]["modal"]  # "L40S"
    """
    if not cloud_config_path.exists():
        return {}
    try:
        with open(cloud_config_path) as f:
            config = yaml.safe_load(f)
        return config.get("gpu_tiers", {})
    except Exception as e:
        logger.warning("Failed to load GPU tiers from %s: %s", cloud_config_path, e)
        return {}


def resolve_gpu_for_tier(
    cloud_config_path: Path, provider: str, tier: str
) -> Optional[str]:
    """
    Resolve a GPU tier name to a provider-specific GPU identifier.

    Looks up the gpu_tiers section in cloud_config.yaml and returns the
    GPU identifier for the given provider and tier.

    Args:
        cloud_config_path: Path to cloud_config.yaml file
        provider: Provider name ('hf_jobs', 'modal', 'runpod')
        tier: Tier name ('budget', 'standard', 'performance')

    Returns:
        Provider-specific GPU identifier string, or None if tier/provider
        not found.

    Example:
        gpu = resolve_gpu_for_tier(config_path, "modal", "standard")
        # Returns "L40S"
    """
    tiers = load_gpu_tiers(cloud_config_path)
    tier_config = tiers.get(tier)
    if not tier_config:
        return None
    return tier_config.get(provider)


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

    Reads pricing data from cloud_config.yaml (cached after first load).

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
    pricing = load_gpu_pricing()
    provider_pricing = pricing.get(provider, {})
    gpu_info = provider_pricing.get(gpu_type)
    if not gpu_info:
        return None
    estimated = gpu_info["price"] * timeout_hours
    return f"~${estimated:.2f}"


def get_gpu_display_name(provider: str, gpu_type: str) -> str:
    """
    Get human-readable GPU name for display.

    Reads pricing data from cloud_config.yaml (cached after first load).

    Args:
        provider: Cloud provider identifier
        gpu_type: Provider-specific GPU identifier

    Returns:
        Display name (e.g., "A10G (24GB)") or the raw gpu_type if not found.
    """
    pricing = load_gpu_pricing()
    provider_pricing = pricing.get(provider, {})
    gpu_info = provider_pricing.get(gpu_type)
    if gpu_info:
        return gpu_info["name"]
    return gpu_type
