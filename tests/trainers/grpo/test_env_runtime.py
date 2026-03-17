import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_runtime import build_cloud_bootstrap_commands


def test_build_cloud_bootstrap_commands_uses_isolated_venv():
    config = {
        "env_training": {
            "runtime": {
                "repo_root_in_container": "/workspace/repo",
                "isolated_venv_dir": "/workspace/.venvs/test-env",
                "project_pip_deps": ["pyyaml"],
                "python_packages": ["trl>=0.24.0", "transformers>=4.56.0"],
            }
        }
    }

    commands = build_cloud_bootstrap_commands(
        config,
        repo_root="/workspace/repo",
        config_path="/workspace/repo/Trainers/grpo/configs/env_config.yaml",
    )

    assert commands[0] == "mkdir -p /tmp/grpo-openenv-bootstrap"
    assert commands[1] == "python -m pip install --upgrade --target /tmp/grpo-openenv-bootstrap virtualenv"
    assert commands[2] == "export PYTHONPATH=/tmp/grpo-openenv-bootstrap:$PYTHONPATH"
    assert commands[3] == "python -m virtualenv --no-download '/workspace/.venvs/test-env'"
    assert commands[4] == ". '/workspace/.venvs/test-env'/bin/activate"
    assert "trl>=0.24.0" in commands[5]
    assert commands[6] == "cd '/workspace/repo/Trainers/grpo'"
    assert commands[-1].endswith(
        "python train_env_grpo.py --config '/workspace/repo/Trainers/grpo/configs/env_config.yaml'"
    )
