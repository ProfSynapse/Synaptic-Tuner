"""
Tests for shared/env_bootstrap.py

Verifies init_trainer_env() with various flag combinations,
and the individual helper functions it delegates to.
"""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

import pytest

from shared.env_bootstrap import (
    init_trainer_env,
    suppress_transformers_logging,
)


class TestInitTrainerEnv:
    """Test init_trainer_env() with different flag combinations."""

    _TORCH_VARS = ("TORCH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTORCH_JIT")

    @pytest.fixture(autouse=True)
    def _clean_torch_env(self, monkeypatch):
        """Remove torch compile env vars before each test; monkeypatch restores on teardown."""
        for var in self._TORCH_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_default_flags_set_env_vars(self):
        """Default call should set torch compile env vars."""
        init_trainer_env(
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=False,
        )

        assert os.environ.get("TORCH_COMPILE_DISABLE") == "1"
        assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
        assert os.environ.get("PYTORCH_JIT") == "0"

    def test_disable_torch_compile_false_skips_env_vars(self):
        """When disable_torch_compile=False, env vars should NOT be set."""
        init_trainer_env(
            disable_torch_compile=False,
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=False,
        )

        assert "TORCH_COMPILE_DISABLE" not in os.environ
        assert "TORCHDYNAMO_DISABLE" not in os.environ
        assert "PYTORCH_JIT" not in os.environ

    def test_setdefault_does_not_overwrite(self, monkeypatch):
        """setdefault should preserve existing env var values."""
        monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")

        init_trainer_env(
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=False,
        )

        # Should keep the existing "0", not overwrite to "1"
        assert os.environ["TORCH_COMPILE_DISABLE"] == "0"

    def test_all_flags_false_is_noop(self):
        """All flags False should do nothing."""
        init_trainer_env(
            disable_torch_compile=False,
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=False,
        )

        # Vars were removed by autouse fixture; should still be absent
        for var in self._TORCH_VARS:
            assert var not in os.environ

    def test_load_dotenv_without_dotenv_installed(self):
        """load_dotenv=True should silently pass if dotenv not installed."""
        with patch.dict("sys.modules", {"dotenv": None}):
            # Should not raise
            init_trainer_env(
                disable_torch_compile=False,
                apply_windows_patches=False,
                load_dotenv=True,
                suppress_transformers=False,
                utf8_output=False,
            )

    def test_suppress_transformers_sets_logger_levels(self):
        """suppress_transformers=True should set transformers loggers to WARNING."""
        init_trainer_env(
            disable_torch_compile=False,
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=True,
            utf8_output=False,
        )

        logger = logging.getLogger("transformers")
        assert logger.level == logging.WARNING
        trainer_logger = logging.getLogger("transformers.trainer")
        assert trainer_logger.level == logging.WARNING

    @pytest.mark.skipif(sys.platform == "win32", reason="Tests non-Windows path")
    def test_windows_patches_skipped_on_non_windows(self):
        """apply_windows_patches=True should skip on non-Windows."""
        # Should not raise or change anything on non-Windows
        init_trainer_env(
            disable_torch_compile=False,
            apply_windows_patches=True,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=False,
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="Tests non-Windows path")
    def test_utf8_output_skipped_on_non_windows(self):
        """utf8_output=True should skip on non-Windows."""
        original_stdout = sys.stdout
        init_trainer_env(
            disable_torch_compile=False,
            apply_windows_patches=False,
            load_dotenv=False,
            suppress_transformers=False,
            utf8_output=True,
        )
        # stdout should be unchanged on non-Windows
        assert sys.stdout is original_stdout


class TestSuppressTransformersLogging:
    """Test the post-import suppress function."""

    def test_suppress_transformers_logging_no_import(self):
        """Should handle missing transformers gracefully."""
        with patch.dict("sys.modules", {"transformers": None}):
            # Should not raise
            suppress_transformers_logging()

    def test_suppress_transformers_logging_with_mock(self):
        """Should call set_verbosity_warning when transformers available."""
        from unittest.mock import MagicMock

        mock_tf = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_tf}):
            suppress_transformers_logging()
            mock_tf.logging.set_verbosity_warning.assert_called_once()
