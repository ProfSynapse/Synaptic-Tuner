"""
Shared fixtures for cloud backend tests.

Provides:
- Temporary repo root with config files (including pricing)
- Cloud config YAML fixtures
- Training config YAML fixtures
- Environment variable isolation
- GPU pricing cache reset between tests
"""

import os
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import tuner.backends.training.cloud.base_cloud as base_cloud_module


@pytest.fixture(autouse=True)
def reset_pricing_cache():
    """Reset the GPU pricing cache before each test.

    The pricing cache is module-level in base_cloud. Without resetting it,
    test fixtures that create custom cloud_config.yaml files would read
    stale pricing data from a previous test's config file.
    """
    base_cloud_module._GPU_PRICING_CACHE = None
    yield
    base_cloud_module._GPU_PRICING_CACHE = None


@pytest.fixture
def repo_root(tmp_path):
    """Create a temporary repo root with standard directory structure."""
    # Create trainer directories
    for method in ("sft", "kto"):
        trainer_dir = tmp_path / "Trainers" / method / "configs"
        trainer_dir.mkdir(parents=True)

        config = {
            "model": {"model_name": f"test-org/test-model-{method}"},
            "dataset": {"local_file": f"../../Datasets/test_{method}.jsonl"},
            "training": {
                "num_train_epochs": 2,
                "per_device_train_batch_size": 4,
                "learning_rate": 2e-4,
            },
        }
        with open(trainer_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    # Create cloud config directory and file
    cloud_dir = tmp_path / "Trainers" / "cloud"
    cloud_dir.mkdir(parents=True)

    cloud_config = {
        "dependencies": {
            "docker_image": "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
            "project_pip_deps": [
                "pyyaml",
                "wandb",
                "hf_transfer",
                "python-dotenv",
                "rich",
            ],
            "extra_setup_commands": [],
        },
        "pricing": {
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
                "A100": {"name": "A100 (40GB)", "price": 2.78},
                "A100-80GB": {"name": "A100 (80GB)", "price": 3.72},
                "H100": {"name": "H100 (80GB)", "price": 4.89},
            },
            "runpod": {
                "NVIDIA RTX A6000": {"name": "RTX A6000 (48GB)", "price": 0.79},
                "NVIDIA A100 80GB PCIe": {"name": "A100 (80GB)", "price": 1.64},
                "NVIDIA H100 80GB HBM3": {"name": "H100 (80GB)", "price": 3.89},
            },
        },
        "cloud": {
            "default_provider": "hf_jobs",
            "artifacts": {
                "publish_final_model": False,
                "publish_target_repo": None,
            },
            "hf_jobs": {
                "flavor": "a10g-small",
                "timeout": "4h",
                "image": "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
                "artifact_backend": "hf_bucket",
                "artifact_identifier": "toolset-training-artifacts",
                "output_root": "/workspace/outputs",
            },
            "modal": {
                "gpu": "L40S",
                "timeout_hours": 6,
                "artifact_backend": "modal_volume",
                "cache_volume_name": "toolset-model-cache",
                "output_volume_name": "toolset-training-artifacts",
                "output_mount_path": "/vol/artifacts",
            },
            "runpod": {
                "gpu_type_id": "NVIDIA A100 SXM",
                "gpu_count": 1,
                "volume_in_gb": 50,
                "container_disk_in_gb": 50,
                "cloud_type": "SECURE",
                "default_image": "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
                "default_timeout": 7200,
                "artifact_backend": "runpod_network_volume",
                "network_volume_id": "runpod-vol-123",
                "output_mount_path": "/runpod-volume",
                "result_path": "/runpod-volume/outputs",
            },
            "push_to_hub": False,
            "hub_repo": None,
        },
    }
    with open(cloud_dir / "cloud_config.yaml", "w") as f:
        yaml.dump(cloud_config, f)

    # Create Modal wrapper script (empty file, just needs to exist)
    (cloud_dir / "train_modal.py").touch()

    origin_repo = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin_repo)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin_repo)], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial test repo"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path


@pytest.fixture
def cloud_config_path(repo_root):
    """Path to the cloud_config.yaml fixture."""
    return repo_root / "Trainers" / "cloud" / "cloud_config.yaml"


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all cloud-related environment variables for isolation."""
    env_vars = [
        "HF_TOKEN",
        "HF_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "RUNPOD_API_KEY",
        "CLOUD_REPO_URL",
        "GH_TOKEN",
        "WANDB_API_KEY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def mock_runpod():
    """Create a mock runpod module with standard API surface."""
    mock = MagicMock()
    mock.create_pod.return_value = {"id": "pod-test-123", "costPerHr": "1.64"}
    mock.get_pod.return_value = {
        "desiredStatus": "RUNNING",
        "runtime": {"uptimeInSeconds": 60, "gpus": []},
    }
    mock.terminate_pod.return_value = None
    return mock
