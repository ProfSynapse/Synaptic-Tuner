"""
Tests for tuner/backends/training/cloud/runpod_backend.py

CRITICAL RISK: This backend manages real GPU pod lifecycle with billing
implications. Tests focus on:
- Pod termination ALWAYS happens (finally block)
- _terminate_pod retry with exponential backoff
- _poll_training detects ERROR/FAILED states (billing safety)
- _wait_for_pod_running handles terminal states
- Credential isolation (_build_pod_env)
- _build_startup_command does not leak tokens
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from tuner.backends.training.cloud.runpod_backend import (
    RunPodBackend,
    _DEFAULT_TRAINING_TIMEOUT,
    _POLL_INTERVAL,
    _POD_STARTUP_TIMEOUT,
)
from tuner.core.config import CloudTrainingConfig, TrainingConfig
from tuner.core.exceptions import CloudProviderError, ConfigurationError


class TestRunPodBackendProperties:
    def test_name(self, repo_root):
        backend = RunPodBackend(repo_root)
        assert backend.name == "runpod"

    def test_available_methods(self, repo_root):
        backend = RunPodBackend(repo_root)
        assert backend.get_available_methods() == ["sft", "kto"]


class TestRunPodValidateEnvironment:
    def test_fails_when_runpod_not_installed(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        with patch.dict("sys.modules", {"runpod": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "not installed" in error.lower()

    def test_fails_when_no_api_key(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        mock_runpod = MagicMock()
        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "RUNPOD_API_KEY" in error

    def test_fails_when_api_key_too_short(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "short")
        backend = RunPodBackend(repo_root)
        mock_runpod = MagicMock()
        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            is_valid, error = backend.validate_environment()
        assert not is_valid
        assert "invalid" in error.lower()

    def test_succeeds_with_valid_setup(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "rptestkey12345678901234")
        backend = RunPodBackend(repo_root)
        mock_runpod = MagicMock()
        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            is_valid, error = backend.validate_environment()
        assert is_valid
        assert error == ""


class TestRunPodLoadConfig:
    def test_loads_sft_config(self, repo_root):
        backend = RunPodBackend(repo_root)
        config = backend.load_config("sft")
        assert isinstance(config, CloudTrainingConfig)
        assert config.method == "sft"
        assert config.platform == "runpod"
        assert config.provider == "runpod"
        assert config.gpu_type == "NVIDIA A100 SXM"
        assert config.runpod_volume_gb == 50

    def test_raises_on_unknown_method(self, repo_root):
        backend = RunPodBackend(repo_root)
        with pytest.raises(ConfigurationError, match="Unknown method"):
            backend.load_config("grpo")

    def test_raises_on_missing_config(self, tmp_path):
        backend = RunPodBackend(tmp_path)
        with pytest.raises(ConfigurationError, match="Training config not found"):
            backend.load_config("sft")

    def test_cloud_config_caching(self, repo_root):
        backend = RunPodBackend(repo_root)
        config1 = backend._get_cloud_config()
        config2 = backend._get_cloud_config()
        assert config1 is config2  # Same object (cached)


# ---------------------------------------------------------------------------
# BILLING SAFETY: Pod termination
# ---------------------------------------------------------------------------


class TestTerminatePod:
    def test_terminates_successfully(self, repo_root, mock_runpod):
        backend = RunPodBackend(repo_root)
        backend._terminate_pod(mock_runpod, "pod-123")
        mock_runpod.terminate_pod.assert_called_once_with("pod-123")

    def test_retries_on_transient_failure(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.terminate_pod.side_effect = [
            Exception("Network error"),
            None,  # Succeeds on retry
        ]
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            backend._terminate_pod(mock_rp, "pod-123")
        assert mock_rp.terminate_pod.call_count == 2

    def test_retries_three_times_with_exponential_backoff(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.terminate_pod.side_effect = Exception("Persistent error")
        sleep_calls = []
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep",
                    side_effect=lambda s: sleep_calls.append(s)):
            backend._terminate_pod(mock_rp, "pod-123")
        assert mock_rp.terminate_pod.call_count == 3
        # Exponential backoff: 2^0=1, 2^1=2
        assert sleep_calls == [1, 2]

    def test_prints_manual_termination_warning_on_failure(self, repo_root, capsys):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.terminate_pod.side_effect = Exception("API error")
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            backend._terminate_pod(mock_rp, "pod-123")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Manually terminate" in captured.out
        assert "runpod.io/console/pods" in captured.out

    def test_does_not_raise_on_failure(self, repo_root):
        """_terminate_pod runs in finally blocks and must never raise."""
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.terminate_pod.side_effect = Exception("API down")
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            # Should NOT raise
            backend._terminate_pod(mock_rp, "pod-123")


# ---------------------------------------------------------------------------
# BILLING SAFETY: Poll training detects ERROR/FAILED states
# ---------------------------------------------------------------------------


class TestPollTraining:
    def _make_config(self):
        return CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )

    def test_returns_zero_on_exited(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "EXITED",
            "runtime": None,
        }
        config = self._make_config()
        exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 0

    def test_returns_one_on_error(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "ERROR",
            "runtime": None,
        }
        config = self._make_config()
        exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 1

    def test_returns_one_on_failed(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "FAILED",
            "runtime": None,
        }
        config = self._make_config()
        exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 1

    def test_returns_one_on_terminated(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "TERMINATED",
            "runtime": None,
        }
        config = self._make_config()
        exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 1

    def test_polls_until_exit(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 60, "gpus": []}},
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 120, "gpus": []}},
            {"desiredStatus": "EXITED", "runtime": None},
        ]
        config = self._make_config()
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 0
        assert mock_rp.get_pod.call_count == 3

    def test_retries_on_poll_network_error(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.side_effect = [
            Exception("Network timeout"),
            {"desiredStatus": "EXITED", "runtime": None},
        ]
        config = self._make_config()
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 0

    def test_returns_one_on_timeout(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "RUNNING",
            "runtime": {"uptimeInSeconds": 100, "gpus": []},
        }
        config = self._make_config()
        # Patch the timeout constant to something small
        with patch("tuner.backends.training.cloud.runpod_backend._DEFAULT_TRAINING_TIMEOUT", 50):
            with patch("tuner.backends.training.cloud.runpod_backend._POLL_INTERVAL", 30):
                with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
                    exit_code = backend._poll_training(mock_rp, "pod-123", config)
        assert exit_code == 1


# ---------------------------------------------------------------------------
# Wait for pod running
# ---------------------------------------------------------------------------


class TestWaitForPodRunning:
    def test_returns_when_running(self, repo_root, mock_runpod):
        backend = RunPodBackend(repo_root)
        backend._wait_for_pod_running(mock_runpod, "pod-123")
        mock_runpod.get_pod.assert_called_once_with("pod-123")

    def test_raises_on_error_state(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "ERROR",
            "runtime": None,
        }
        with pytest.raises(CloudProviderError, match="failed to start"):
            backend._wait_for_pod_running(mock_rp, "pod-123")

    def test_raises_on_exited_during_startup(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "EXITED",
            "runtime": None,
        }
        with pytest.raises(CloudProviderError, match="failed to start"):
            backend._wait_for_pod_running(mock_rp, "pod-123")

    def test_raises_on_startup_timeout(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.return_value = {
            "desiredStatus": "PENDING",
            "runtime": None,
        }
        with patch("tuner.backends.training.cloud.runpod_backend._POD_STARTUP_TIMEOUT", 20):
            with patch("tuner.backends.training.cloud.runpod_backend._POLL_INTERVAL", 15):
                with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
                    with pytest.raises(CloudProviderError, match="did not start"):
                        backend._wait_for_pod_running(mock_rp, "pod-123")

    def test_retries_on_api_error_during_startup(self, repo_root):
        backend = RunPodBackend(repo_root)
        mock_rp = MagicMock()
        mock_rp.get_pod.side_effect = [
            Exception("API error"),
            {
                "desiredStatus": "RUNNING",
                "runtime": {"uptimeInSeconds": 5},
            },
        ]
        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            # Should not raise
            backend._wait_for_pod_running(mock_rp, "pod-123")


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------


class TestBuildPodEnv:
    def test_passes_hf_token(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test_123")
        backend = RunPodBackend(repo_root)
        env = backend._build_pod_env()
        assert env["HF_TOKEN"] == "hf_test_123"

    def test_passes_wandb_key(self, repo_root, clean_env):
        clean_env.setenv("WANDB_API_KEY", "wandb_test_key")
        backend = RunPodBackend(repo_root)
        env = backend._build_pod_env()
        assert env["WANDB_API_KEY"] == "wandb_test_key"

    def test_passes_gh_token(self, repo_root, clean_env):
        clean_env.setenv("GH_TOKEN", "ghp_test_token")
        backend = RunPodBackend(repo_root)
        env = backend._build_pod_env()
        assert env["GH_TOKEN"] == "ghp_test_token"

    def test_empty_when_no_env_vars(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        env = backend._build_pod_env()
        assert env == {}

    def test_only_includes_set_vars(self, repo_root, clean_env):
        clean_env.setenv("HF_TOKEN", "hf_test")
        backend = RunPodBackend(repo_root)
        env = backend._build_pod_env()
        assert "HF_TOKEN" in env
        assert "WANDB_API_KEY" not in env
        assert "GH_TOKEN" not in env


# ---------------------------------------------------------------------------
# Startup command security
# ---------------------------------------------------------------------------


class TestBuildStartupCommand:
    def test_command_structure(self, repo_root, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )
        runpod_config = {"setup_commands": ["pip install torch"]}
        cmd = backend._build_startup_command(config, runpod_config)
        assert "pip install torch" in cmd
        assert "git clone" in cmd
        assert "Trainers/rtx3090_sft" in cmd
        assert "python train_sft.py" in cmd

    def test_uses_shell_variable_for_gh_token(self, repo_root, clean_env):
        """GH_TOKEN must be referenced as $GH_TOKEN shell variable, not embedded."""
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        clean_env.setenv("GH_TOKEN", "ghp_secret_value_do_not_leak")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )
        runpod_config = {}
        cmd = backend._build_startup_command(config, runpod_config)
        # Must use shell variable expansion, NOT the actual token value
        assert "ghp_secret_value_do_not_leak" not in cmd
        assert "$GH_TOKEN@" in cmd

    def test_fallback_when_no_repo_url(self, repo_root, clean_env):
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )
        runpod_config = {"setup_commands": ["pip install torch"]}
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("tuner.backends.training.cloud.base_cloud.subprocess.run",
                    return_value=mock_result):
            cmd = backend._build_startup_command(config, runpod_config)
        assert "ERROR" in cmd


# ---------------------------------------------------------------------------
# Execute lifecycle -- pod termination in finally
# ---------------------------------------------------------------------------


class TestExecuteLifecycle:
    def test_terminates_pod_on_success(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "rptestkey12345678901234")
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )

        mock_rp = MagicMock()
        mock_rp.create_pod.return_value = {"id": "pod-abc", "costPerHr": "1.64"}
        # Startup: RUNNING immediately
        mock_rp.get_pod.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}},
            {"desiredStatus": "EXITED", "runtime": None},  # Training done
        ]

        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            with patch.dict("sys.modules", {"runpod": mock_rp}):
                exit_code = backend.execute(config, python_path="")

        assert exit_code == 0
        mock_rp.terminate_pod.assert_called_once_with("pod-abc")

    def test_terminates_pod_on_failure(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "rptestkey12345678901234")
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )

        mock_rp = MagicMock()
        mock_rp.create_pod.return_value = {"id": "pod-def", "costPerHr": "1.64"}
        mock_rp.get_pod.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}},
            {"desiredStatus": "ERROR", "runtime": None},
        ]

        with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
            with patch.dict("sys.modules", {"runpod": mock_rp}):
                exit_code = backend.execute(config, python_path="")

        assert exit_code == 1
        mock_rp.terminate_pod.assert_called_once_with("pod-def")

    def test_terminates_pod_on_keyboard_interrupt(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "rptestkey12345678901234")
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )

        mock_rp = MagicMock()
        mock_rp.create_pod.return_value = {"id": "pod-int", "costPerHr": "1.64"}
        mock_rp.get_pod.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}},
        ]
        # Simulate KeyboardInterrupt during polling
        with patch.object(backend, "_poll_training", side_effect=KeyboardInterrupt):
            with patch("tuner.backends.training.cloud.runpod_backend.time.sleep"):
                with patch.dict("sys.modules", {"runpod": mock_rp}):
                    exit_code = backend.execute(config, python_path="")

        assert exit_code == 130
        mock_rp.terminate_pod.assert_called_once_with("pod-int")

    def test_raises_when_no_pod_id_returned(self, repo_root, clean_env):
        clean_env.setenv("RUNPOD_API_KEY", "rptestkey12345678901234")
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        backend = RunPodBackend(repo_root)
        config = CloudTrainingConfig(
            method="sft", platform="runpod", config_path=Path("/fake"),
            trainer_dir=Path("/fake"), model_name="test", dataset_file="test",
            epochs=1, batch_size=4, learning_rate=2e-4, provider="runpod",
            gpu_type="A100", timeout_hours=6,
        )

        mock_rp = MagicMock()
        mock_rp.create_pod.return_value = {}  # No 'id' key

        with patch.dict("sys.modules", {"runpod": mock_rp}):
            with pytest.raises(CloudProviderError, match="Failed to create"):
                backend.execute(config, python_path="")


class TestGeneratePodName:
    def test_includes_method(self, repo_root):
        backend = RunPodBackend(repo_root)
        name = backend._generate_pod_name("sft")
        assert "toolset-sft-" in name

    def test_includes_timestamp(self, repo_root):
        backend = RunPodBackend(repo_root)
        name = backend._generate_pod_name("kto")
        # Timestamp format: YYYYMMDD-HHMMSS
        parts = name.split("-")
        assert len(parts) >= 3
