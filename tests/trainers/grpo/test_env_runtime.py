import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "grpo" / "src"))

from env_runtime import build_cloud_bootstrap_commands, ensure_local_openenv_runtime
env_runtime = importlib.import_module("env_runtime")


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


def test_ensure_local_openenv_runtime_bootstraps_when_support_missing(tmp_path):
    config = {
        "env_training": {
            "runtime": {
                "local_venv_dir": str(tmp_path / ".venvs" / "grpo-openenv"),
                "project_pip_deps": ["pyyaml"],
                "python_packages": ["trl>=0.24.0", "transformers>=4.56.0"],
            }
        }
    }
    venv_python = tmp_path / ".venvs" / "grpo-openenv" / "bin" / "python"

    def fake_run(cmd, capture_output=False, text=False, check=False):
        result = MagicMock()
        result.returncode = 0
        if cmd[:3] == ["python3", "-m", "venv"]:
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            result.stdout = ""
            result.stderr = ""
            return result
        if cmd[:3] == [str(venv_python), "-m", "pip"]:
            result.stdout = ""
            result.stderr = ""
            return result
        if cmd[:2] == [str(venv_python), "-c"]:
            if not hasattr(fake_run, "probed"):
                fake_run.probed = 1
                result.stdout = (
                    '{"trl_version": null, "transformers_version": null, '
                    '"openenv_version": null, "has_rollout_func": false, '
                    '"has_environment_factory": false, '
                    '"has_generate_rollout_completions": false, "errors": []}'
                )
            else:
                result.stdout = (
                    '{"trl_version": "0.29.0", "transformers_version": "4.57.6", '
                    '"openenv_version": "0.1.0", "has_rollout_func": true, '
                    '"has_environment_factory": true, '
                    '"has_generate_rollout_completions": true, "errors": []}'
                )
            result.stderr = ""
            return result
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch.object(env_runtime.subprocess, "run", side_effect=fake_run):
        runtime_python = ensure_local_openenv_runtime(
            config,
            repo_root=str(tmp_path),
            bootstrap_python="python3",
        )

    assert runtime_python == str(venv_python)
