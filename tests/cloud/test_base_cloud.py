"""
Tests for tuner/backends/training/cloud/base_cloud.py

Covers:
- load_cloud_config: file missing, valid, malformed YAML
- resolve_repo_url: env var, git remote, failure
- poll_until_done: completion, timeout, persistent errors, consecutive transient errors
- estimate_cost: valid combos, unknown GPU, unknown provider
- get_gpu_display_name: known/unknown
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tuner.backends.training.cloud.base_cloud import (
    estimate_cost,
    get_gpu_display_name,
    load_cloud_config,
    load_gpu_pricing,
    poll_until_done,
    resolve_repo_url,
)
from tuner.core.exceptions import CloudProviderError


# ---------------------------------------------------------------------------
# load_cloud_config
# ---------------------------------------------------------------------------


class TestLoadCloudConfig:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        result = load_cloud_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_returns_cloud_section(self, cloud_config_path):
        result = load_cloud_config(cloud_config_path)
        assert "hf_jobs" in result
        assert "modal" in result
        assert "runpod" in result

    def test_returns_empty_dict_on_malformed_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": : : invalid yaml [[[")
        result = load_cloud_config(bad_file)
        # Should log warning and return empty dict, not raise
        assert result == {}

    def test_returns_empty_dict_when_no_cloud_key(self, tmp_path):
        config_file = tmp_path / "no_cloud.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"other": {"key": "value"}}, f)
        result = load_cloud_config(config_file)
        assert result == {}

    def test_returns_empty_dict_when_yaml_is_none(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = load_cloud_config(config_file)
        assert result == {}


# ---------------------------------------------------------------------------
# resolve_repo_url
# ---------------------------------------------------------------------------


class TestResolveRepoUrl:
    def test_returns_env_var_when_set(self, clean_env):
        clean_env.setenv("CLOUD_REPO_URL", "https://github.com/test/repo.git")
        assert resolve_repo_url() == "https://github.com/test/repo.git"

    def test_falls_back_to_git_remote(self, clean_env):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/fallback/repo.git\n"

        with patch("tuner.backends.training.cloud.base_cloud.subprocess.run",
                    return_value=mock_result):
            assert resolve_repo_url() == "https://github.com/fallback/repo.git"

    def test_raises_when_no_url_available(self, clean_env):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("tuner.backends.training.cloud.base_cloud.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(CloudProviderError, match="Cannot determine repo URL"):
                resolve_repo_url()

    def test_handles_git_not_found(self, clean_env):
        with patch("tuner.backends.training.cloud.base_cloud.subprocess.run",
                    side_effect=FileNotFoundError):
            with pytest.raises(CloudProviderError):
                resolve_repo_url()

    def test_handles_git_timeout(self, clean_env):
        import subprocess
        with patch("tuner.backends.training.cloud.base_cloud.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10)):
            with pytest.raises(CloudProviderError):
                resolve_repo_url()


# ---------------------------------------------------------------------------
# poll_until_done
# ---------------------------------------------------------------------------


class TestPollUntilDone:
    def test_returns_immediately_on_completed(self):
        check_fn = MagicMock(return_value="COMPLETED")
        result = poll_until_done(check_fn, interval=0, timeout_seconds=10)
        assert result == "COMPLETED"
        check_fn.assert_called_once()

    def test_returns_error_status(self):
        check_fn = MagicMock(return_value="ERROR: Job failed")
        result = poll_until_done(check_fn, interval=0, timeout_seconds=10)
        assert result == "ERROR: Job failed"

    def test_polls_until_terminal(self):
        call_count = 0

        def check():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # Still running
            return "COMPLETED"

        result = poll_until_done(check, interval=0, timeout_seconds=100)
        assert result == "COMPLETED"
        assert call_count == 3

    def test_returns_timeout_error_when_exceeded(self):
        check_fn = MagicMock(return_value=None)  # Never completes
        result = poll_until_done(check_fn, interval=1, timeout_seconds=2)
        assert result == "ERROR: Timeout exceeded"

    def test_raises_on_persistent_unauthorized_error(self):
        check_fn = MagicMock(side_effect=Exception("Request unauthorized"))
        with pytest.raises(Exception, match="unauthorized"):
            poll_until_done(check_fn, interval=0, timeout_seconds=10)

    def test_raises_on_persistent_not_found_error(self):
        check_fn = MagicMock(side_effect=Exception("Resource not found"))
        with pytest.raises(Exception, match="not found"):
            poll_until_done(check_fn, interval=0, timeout_seconds=10)

    def test_raises_on_persistent_forbidden_error(self):
        check_fn = MagicMock(side_effect=Exception("Access forbidden"))
        with pytest.raises(Exception, match="forbidden"):
            poll_until_done(check_fn, interval=0, timeout_seconds=10)

    def test_raises_on_persistent_invalid_error(self):
        check_fn = MagicMock(side_effect=Exception("Invalid token"))
        with pytest.raises(Exception, match="Invalid"):
            poll_until_done(check_fn, interval=0, timeout_seconds=10)

    def test_retries_transient_errors_then_succeeds(self):
        call_count = 0

        def check():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network timeout")
            return "COMPLETED"

        result = poll_until_done(check, interval=0, timeout_seconds=100)
        assert result == "COMPLETED"
        assert call_count == 2

    def test_raises_after_max_consecutive_transient_errors(self):
        check_fn = MagicMock(side_effect=Exception("Network timeout"))
        with pytest.raises(Exception, match="Network timeout"):
            poll_until_done(check_fn, interval=0, timeout_seconds=100)
        # Should be called max_consecutive (3) times
        assert check_fn.call_count == 3

    def test_resets_consecutive_errors_on_success(self):
        """Transient error count resets after a successful poll."""
        call_count = 0

        def check():
            nonlocal call_count
            call_count += 1
            if call_count in (1, 2):
                raise Exception("Network timeout")
            if call_count in (3, 4):
                return None  # Resets counter
            if call_count == 5:
                raise Exception("Network timeout")
            if call_count == 6:
                return None  # Resets counter again
            return "COMPLETED"

        result = poll_until_done(check, interval=0, timeout_seconds=100)
        assert result == "COMPLETED"


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_known_provider_and_gpu(self):
        result = estimate_cost("hf_jobs", "a10g-small", 4.0)
        assert result == "~$4.40"

    def test_unknown_gpu_returns_none(self):
        result = estimate_cost("hf_jobs", "nonexistent-gpu", 4.0)
        assert result is None

    def test_unknown_provider_returns_none(self):
        result = estimate_cost("nonexistent_provider", "a10g-small", 4.0)
        assert result is None

    def test_zero_hours_returns_zero_cost(self):
        result = estimate_cost("modal", "T4", 0.0)
        assert result == "~$0.00"

    def test_modal_gpu_pricing(self):
        result = estimate_cost("modal", "H100", 2.0)
        pricing = load_gpu_pricing()
        expected = pricing["modal"]["H100"]["price"] * 2.0
        assert result == f"~${expected:.2f}"

    def test_runpod_gpu_pricing(self):
        result = estimate_cost("runpod", "NVIDIA RTX A6000", 1.0)
        assert result is not None
        assert result.startswith("~$")


# ---------------------------------------------------------------------------
# get_gpu_display_name
# ---------------------------------------------------------------------------


class TestGetGpuDisplayName:
    def test_known_gpu_returns_display_name(self):
        name = get_gpu_display_name("hf_jobs", "a10g-small")
        assert name == "A10G (24GB)"

    def test_unknown_gpu_returns_raw_type(self):
        name = get_gpu_display_name("hf_jobs", "unknown-gpu")
        assert name == "unknown-gpu"

    def test_unknown_provider_returns_raw_type(self):
        name = get_gpu_display_name("nonexistent", "a10g-small")
        assert name == "a10g-small"

    def test_modal_h100(self):
        name = get_gpu_display_name("modal", "H100")
        assert name == "H100 (80GB)"
