"""
Tests for tuner/backends/training/cloud/modal_backend.py

Covers:
- ModalBackend.name, get_available_methods
- validate_environment: modal not installed, env var auth, toml file, modal CLI
- load_config: valid sft/kto, unknown method, missing config file
- execute: wrapper script missing, subprocess timeout, keyboard interrupt, FileNotFoundError
- _resolve_repo_url: env var, git remote, failure
- get_gpu_options
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

from tuner.backends.training.cloud.base_cloud import load_gpu_pricing
from tuner.backends.training.cloud.modal_backend import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_HOURS,
    ModalBackend,
)
from tuner.core.config import CloudTrainingConfig, TrainingConfig
from tuner.core.exceptions import BackendError, ConfigurationError


class TestModalBackendProperties:
    def test_name(self, repo_root):
        backend = ModalBackend(repo_root)
        assert backend.name == "modal"

    def test_available_methods(self, repo_root):
        backend = ModalBackend(repo_root)
        methods = backend.get_available_methods()
        assert "sft" in methods
        assert "kto" in methods

    def test_get_gpu_options_returns_copy(self, repo_root):
        backend = ModalBackend(repo_root)
        options = backend.get_gpu_options()
        pricing = load_gpu_pricing()
        assert options == pricing["modal"]
        # Must be a copy, not the original
        options["NEW_GPU"] = {"vram_gb": 999}
        assert "NEW_GPU" not in backend.get_gpu_options()


class TestModalValidateEnvironment:
    def test_fails_when_modal_not_installed(self, repo_root, clean_env):
        backend = ModalBackend(repo_root)
        with patch.dict("sys.modules", {"modal": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'modal'")):
                is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "not installed" in error.lower()

    def test_succeeds_with_env_var_auth(self, repo_root, clean_env):
        clean_env.setenv("MODAL_TOKEN_ID", "test-token-id")
        clean_env.setenv("MODAL_TOKEN_SECRET", "test-token-secret")
        backend = ModalBackend(repo_root)
        with patch("tuner.backends.training.cloud.modal_backend.importlib", create=True):
            # Mock the import modal inside validate_environment
            mock_modal = MagicMock()
            with patch.dict("sys.modules", {"modal": mock_modal}):
                is_valid, error = backend.validate_environment()
        assert is_valid
        assert error == ""

    def test_succeeds_with_toml_file(self, repo_root, clean_env, tmp_path):
        backend = ModalBackend(repo_root)
        mock_modal = MagicMock()
        toml_path = tmp_path / ".modal.toml"
        toml_path.touch()
        with patch.dict("sys.modules", {"modal": mock_modal}):
            with patch("tuner.backends.training.cloud.modal_backend.Path.home",
                        return_value=tmp_path):
                is_valid, error = backend.validate_environment()
        assert is_valid

    def test_fails_when_no_auth_configured(self, repo_root, clean_env, tmp_path):
        backend = ModalBackend(repo_root)
        mock_modal = MagicMock()
        # No env vars, no toml, and modal CLI fails
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch.dict("sys.modules", {"modal": mock_modal}):
            with patch("tuner.backends.training.cloud.modal_backend.Path.home",
                        return_value=tmp_path):
                with patch("tuner.backends.training.cloud.modal_backend.subprocess.run",
                            return_value=mock_result):
                    is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "not authenticated" in error.lower()


class TestModalLoadConfig:
    def test_loads_sft_config(self, repo_root):
        backend = ModalBackend(repo_root)
        config = backend.load_config("sft")
        assert isinstance(config, CloudTrainingConfig)
        assert config.method == "sft"
        assert config.platform == "modal"
        assert config.provider == "modal"
        assert config.model_name == "test-org/test-model-sft"
        assert config.epochs == 2
        assert config.batch_size == 4
        assert config.gpu_type == "L40S"
        assert config.timeout_hours == 6

    def test_loads_kto_config(self, repo_root):
        backend = ModalBackend(repo_root)
        config = backend.load_config("kto")
        assert config.method == "kto"
        assert config.model_name == "test-org/test-model-kto"

    def test_raises_on_unknown_method(self, repo_root):
        backend = ModalBackend(repo_root)
        with pytest.raises(ConfigurationError, match="Unknown method 'grpo'"):
            backend.load_config("grpo")

    def test_raises_on_missing_config_file(self, tmp_path):
        # tmp_path with no config files
        backend = ModalBackend(tmp_path)
        with pytest.raises(ConfigurationError, match="Training config not found"):
            backend.load_config("sft")

    def test_raises_on_malformed_yaml(self, repo_root):
        config_path = repo_root / "Trainers" / "rtx3090_sft" / "configs" / "config.yaml"
        config_path.write_text(": : : invalid yaml [[[")
        backend = ModalBackend(repo_root)
        with pytest.raises(ConfigurationError, match="Failed to parse"):
            backend.load_config("sft")

    def test_uses_defaults_when_cloud_config_missing(self, repo_root):
        # Remove cloud config
        cloud_config = repo_root / "Trainers" / "cloud" / "cloud_config.yaml"
        cloud_config.unlink()
        backend = ModalBackend(repo_root)
        config = backend.load_config("sft")
        assert config.gpu_type == DEFAULT_GPU
        assert config.timeout_hours == DEFAULT_TIMEOUT_HOURS


class TestModalExecute:
    def test_raises_when_wrapper_script_missing(self, repo_root):
        # Remove the wrapper script
        wrapper = repo_root / "Trainers" / "cloud" / "train_modal.py"
        wrapper.unlink()
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="modal",
            gpu_type="L40S", timeout_hours=6,
        )
        with pytest.raises(BackendError, match="wrapper script not found"):
            backend.execute(config, python_path="python")

    def test_subprocess_timeout_kills_process(self, repo_root, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="modal",
            gpu_type="L40S", timeout_hours=0.001,  # Very short timeout
        )
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired(
            cmd="modal run", timeout=3
        )
        with patch("tuner.backends.training.cloud.modal_backend.subprocess.Popen",
                    return_value=mock_process):
            with pytest.raises(BackendError, match="timed out"):
                backend.execute(config, python_path="python")
        mock_process.kill.assert_called_once()

    def test_file_not_found_raises_backend_error(self, repo_root, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="modal",
            gpu_type="L40S", timeout_hours=6,
        )
        with patch("tuner.backends.training.cloud.modal_backend.subprocess.Popen",
                    side_effect=FileNotFoundError):
            with pytest.raises(BackendError, match="Modal CLI not found"):
                backend.execute(config, python_path="python")

    def test_keyboard_interrupt_returns_130(self, repo_root, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="modal",
            gpu_type="L40S", timeout_hours=6,
        )
        mock_process = MagicMock()
        mock_process.wait.side_effect = KeyboardInterrupt
        with patch("tuner.backends.training.cloud.modal_backend.subprocess.Popen",
                    return_value=mock_process):
            exit_code = backend.execute(config, python_path="python")
        assert exit_code == 130
        mock_process.terminate.assert_called_once()

    def test_successful_execution_returns_exit_code(self, repo_root, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="modal",
            gpu_type="L40S", timeout_hours=6,
        )
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        with patch("tuner.backends.training.cloud.modal_backend.subprocess.Popen",
                    return_value=mock_process):
            exit_code = backend.execute(config, python_path="python")
        assert exit_code == 0


class TestModalResolveRepoUrl:
    """Test that ModalBackend.execute() calls resolve_repo_url from base_cloud.

    The resolve_repo_url function itself is comprehensively tested in
    test_base_cloud.py::TestResolveRepoUrl. Here we verify the integration:
    execute() calls it and raises BackendError on failure.
    """

    def test_execute_raises_on_resolve_failure(self, repo_root, clean_env):
        from tuner.core.config import CloudTrainingConfig
        backend = ModalBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="modal", config_path=repo_root / "config.yaml",
            trainer_dir=repo_root / "Trainers" / "rtx3090_sft",
            model_name="test", dataset_file="test.jsonl",
            epochs=1, batch_size=4, learning_rate=2e-4,
            provider="modal", gpu_type="L40S", timeout_hours=6,
        )
        # Ensure wrapper script exists so we reach the resolve_repo_url call
        wrapper = repo_root / "Trainers" / "cloud" / "train_modal.py"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.touch()
        with patch("tuner.backends.training.cloud.modal_backend.resolve_repo_url",
                    side_effect=Exception("no url")):
            with pytest.raises(BackendError, match="Cannot determine repo URL"):
                backend.execute(config, python_path="python")
