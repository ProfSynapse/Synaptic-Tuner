from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tuner.backends.training.rtx_backend import RTXBackend


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    for method in ("sft", "kto"):
        config_dir = tmp_path / "Trainers" / method / "configs"
        config_dir.mkdir(parents=True)
        with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "model": {"model_name": f"test-org/{method}-model"},
                    "dataset": {"local_file": f"../../Datasets/{method}.jsonl"},
                    "training": {
                        "num_train_epochs": 2,
                        "per_device_train_batch_size": 4,
                        "learning_rate": 2e-4,
                    },
                },
                f,
            )

    grpo_config_dir = tmp_path / "Trainers" / "grpo" / "configs"
    grpo_config_dir.mkdir(parents=True)
    grpo_src_dir = tmp_path / "Trainers" / "grpo" / "src"
    grpo_src_dir.mkdir(parents=True)
    with open(grpo_config_dir / "env_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "model": {"model_name": "test-org/env-grpo-model"},
                "dataset": {
                    "dataset_name": "test-org/test-dataset",
                    "dataset_file": "environment_rollouts/canonical/test.jsonl",
                    "local_file": None,
                },
                "training": {
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "learning_rate": 5e-6,
                },
            },
            f,
        )
    with open(grpo_src_dir / "env_runtime.py", "w", encoding="utf-8") as f:
        f.write(
            "def ensure_local_openenv_runtime(config, *, repo_root, bootstrap_python):\n"
            "    return '/fake/runtime/python'\n"
        )

    (tmp_path / "Trainers" / "sft" / "train_sft.py").write_text("", encoding="utf-8")
    (tmp_path / "Trainers" / "kto" / "train_kto.py").write_text("", encoding="utf-8")
    (tmp_path / "Trainers" / "grpo" / "train_env_grpo.py").write_text("", encoding="utf-8")
    return tmp_path


def test_load_config_for_grpo_uses_env_config(repo_root: Path):
    backend = RTXBackend(repo_root)

    config = backend.load_config("grpo")

    assert config.method == "grpo"
    assert config.config_path.name == "env_config.yaml"
    assert config.trainer_dir == repo_root / "Trainers" / "grpo"
    assert config.model_name == "test-org/env-grpo-model"
    assert config.dataset_file == "test-org/test-dataset/environment_rollouts/canonical/test.jsonl"
    assert config.epochs == 1
    assert config.batch_size == 1
    assert config.learning_rate == 5e-6


def test_execute_for_grpo_uses_env_grpo_entrypoint(repo_root: Path):
    backend = RTXBackend(repo_root)
    config = backend.load_config("grpo")
    process = MagicMock()
    process.wait.return_value = 0
    sys.modules.pop("env_runtime", None)

    with patch("subprocess.Popen", return_value=process) as popen:
        exit_code = backend.execute(config, python_path="/fake/python")

    assert exit_code == 0
    popen.assert_called_once()
    cmd = popen.call_args.kwargs["args"] if "args" in popen.call_args.kwargs else popen.call_args.args[0]
    assert cmd == [
        "/fake/runtime/python",
        "train_env_grpo.py",
        "--config",
        str(config.config_path),
    ]
    assert popen.call_args.kwargs["cwd"] == str(config.trainer_dir)
