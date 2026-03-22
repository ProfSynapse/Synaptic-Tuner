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

    grpo_dir = tmp_path / "Trainers" / "grpo" / "configs"
    grpo_dir.mkdir(parents=True)
    grpo_config = {
        "model": {
            "model_name": "professorsynapse/Nexus-Quark-L2.5.28",
            "max_seq_length": 8192,
        },
        "dataset": {
            "dataset_name": "professorsynapse/nexus-synthetic-data",
            "dataset_file": "environment_rollouts/canonical/vault_shared_seed_dynamic_roles_aggregate_20260316.jsonl",
        },
        "training": {
            "output_dir": "./env_grpo_output",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 4,
            "max_prompt_length": 4096,
            "max_completion_length": 1024,
            "learning_rate": 5e-6,
            "num_train_epochs": 1,
            "logging_steps": 1,
            "save_steps": 25,
            "save_total_limit": 2,
            "report_to": "none",
        },
        "env_training": {
            "runtime": {
                "isolated_venv_dir": "/workspace/.venvs/grpo-openenv",
                "project_pip_deps": ["pyyaml", "python-dotenv", "rich", "datasets"],
                "python_packages": [
                    "huggingface_hub<1.0",
                    "trl>=0.24.0",
                    "transformers>=4.56.0",
                    "accelerate>=1.6.0",
                    "peft>=0.15.0",
                    "git+https://github.com/meta-pytorch/OpenEnv.git",
                ],
            }
        },
        "rewards": {
            "success_reward": 1.0,
            "failure_penalty": -1.0,
        },
    }
    with open(grpo_dir / "env_config.yaml", "w") as f:
        yaml.dump(grpo_config, f)

    # Create cloud config directory and file
    cloud_dir = tmp_path / "Trainers" / "cloud"
    cloud_dir.mkdir(parents=True)

    cloud_config = {
        "dependencies": {
            "docker_image_profiles": {
                "stable": "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
                "next": "unsloth/unsloth:2026.2.1-pt2.9.0-cu12.8-fixed-numba-numpy-error",
            },
            "eval_image_profiles": {
                "stable_unsloth": "unsloth/unsloth:2026.1.2-pt2.9.0-cu12.8-update@sha256:5266c57be21059bfb407d80dc2f448868a5c2e2dbe7b2aa27780f48b48cbec39",
                "latest_unsloth": "unsloth/unsloth:latest",
                "fast_vllm": "vllm/vllm-openai:v0.17.1",
            },
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
                "evaluation": {
                    "runtime": "unsloth",
                    "image_profile": "stable_unsloth",
                },
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
