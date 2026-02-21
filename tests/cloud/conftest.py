"""
Shared fixtures for cloud backend tests.

Provides:
- Temporary repo root with config files
- Cloud config YAML fixtures
- Training config YAML fixtures
- Environment variable isolation
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def repo_root(tmp_path):
    """Create a temporary repo root with standard directory structure."""
    # Create trainer directories
    for method in ("sft", "kto"):
        trainer_dir = tmp_path / "Trainers" / f"rtx3090_{method}" / "configs"
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
        "cloud": {
            "default_provider": "hf_jobs",
            "hf_jobs": {
                "flavor": "a10g-small",
                "timeout": "4h",
                "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            },
            "modal": {
                "gpu": "L40S",
                "timeout_hours": 6,
            },
            "runpod": {
                "gpu_type_id": "NVIDIA A100 SXM",
                "gpu_count": 1,
                "volume_in_gb": 50,
                "container_disk_in_gb": 50,
                "cloud_type": "SECURE",
                "default_image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                "default_timeout": 7200,
            },
            "push_to_hub": True,
            "hub_repo": None,
        }
    }
    with open(cloud_dir / "cloud_config.yaml", "w") as f:
        yaml.dump(cloud_config, f)

    # Create Modal wrapper script (empty file, just needs to exist)
    (cloud_dir / "train_modal.py").touch()

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
