"""Tests for shared.eval_backend — EvalBackend protocol and implementations."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.eval_backend import (
    CloudEvalBackend,
    CloudProvider,
    EvalBackend,
    EvalResult,
    LocalEvalBackend,
    create_eval_backend,
)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    """EvalResult dataclass fields."""

    def test_default_values(self):
        r = EvalResult(eval_score=0.85)
        assert r.eval_score == 0.85
        assert r.results_path is None
        assert r.raw_results is None

    def test_all_fields(self):
        r = EvalResult(
            eval_score=0.92,
            results_path="/tmp/results.json",
            raw_results={"pass_rate": 0.92},
        )
        assert r.eval_score == 0.92
        assert r.results_path == "/tmp/results.json"
        assert r.raw_results == {"pass_rate": 0.92}


# ---------------------------------------------------------------------------
# LocalEvalBackend.check_hardware
# ---------------------------------------------------------------------------


class TestLocalEvalBackendHardware:
    """LocalEvalBackend GPU VRAM checks via mocked nvidia-smi."""

    @patch("shared.eval_backend.subprocess.run")
    def test_sufficient_vram(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="24576\n")
        backend = LocalEvalBackend(min_vram_gb=8)
        assert backend.check_hardware() is True

    @patch("shared.eval_backend.subprocess.run")
    def test_insufficient_vram(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="4096\n")
        backend = LocalEvalBackend(min_vram_gb=8)
        assert backend.check_hardware() is False

    @patch("shared.eval_backend.subprocess.run")
    def test_exact_threshold(self, mock_run):
        # 8192 MB = 8 GB exactly
        mock_run.return_value = MagicMock(returncode=0, stdout="8192\n")
        backend = LocalEvalBackend(min_vram_gb=8)
        assert backend.check_hardware() is True

    @patch("shared.eval_backend.subprocess.run")
    def test_nvidia_smi_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        backend = LocalEvalBackend()
        assert backend.check_hardware() is False

    @patch("shared.eval_backend.subprocess.run")
    def test_nvidia_smi_nonzero_return(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        backend = LocalEvalBackend()
        assert backend.check_hardware() is False

    @patch("shared.eval_backend.subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
        backend = LocalEvalBackend()
        assert backend.check_hardware() is False

    @patch("shared.eval_backend.subprocess.run")
    def test_nvidia_smi_garbage_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not a number\n")
        backend = LocalEvalBackend()
        assert backend.check_hardware() is False

    @patch("shared.eval_backend.subprocess.run")
    def test_multi_gpu_uses_first(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="24576\n12288\n")
        backend = LocalEvalBackend(min_vram_gb=16)
        # 24576 MB = 24 GB >= 16 GB
        assert backend.check_hardware() is True


# ---------------------------------------------------------------------------
# LocalEvalBackend.run_eval
# ---------------------------------------------------------------------------


class TestLocalEvalBackendRunEval:
    """LocalEvalBackend.run_eval with mocked subprocess and hardware check."""

    @pytest.mark.asyncio
    async def test_raises_on_insufficient_hardware(self):
        backend = LocalEvalBackend(min_vram_gb=8)
        with patch.object(backend, "check_hardware", return_value=False):
            with pytest.raises(RuntimeError, match="Insufficient GPU VRAM"):
                await backend.run_eval("/path/to/adapter", "tool_prompts.yaml")

    @pytest.mark.asyncio
    @patch("shared.eval_backend.subprocess.run")
    async def test_successful_eval_json_output(self, mock_run):
        import json

        output = json.dumps(
            {"results_summary": {"overall_pass_rate": 0.88}, "details": []}
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")

        backend = LocalEvalBackend()
        with patch.object(backend, "check_hardware", return_value=True):
            result = await backend.run_eval("/path/to/adapter", "tool_prompts.yaml")

        assert result.eval_score == 0.88
        assert result.raw_results is not None
        assert result.raw_results["results_summary"]["overall_pass_rate"] == 0.88

    @pytest.mark.asyncio
    @patch("shared.eval_backend.subprocess.run")
    async def test_non_json_output_returns_zero(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Some non-JSON output\n", stderr=""
        )

        backend = LocalEvalBackend()
        with patch.object(backend, "check_hardware", return_value=True):
            result = await backend.run_eval("/path/to/adapter", "scenario.yaml")

        assert result.eval_score == 0.0

    @pytest.mark.asyncio
    @patch("shared.eval_backend.subprocess.run")
    async def test_eval_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: model not found"
        )

        backend = LocalEvalBackend()
        with patch.object(backend, "check_hardware", return_value=True):
            with pytest.raises(RuntimeError, match="Evaluation failed"):
                await backend.run_eval("/path/to/adapter", "scenario.yaml")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """EvalBackend protocol compliance for concrete implementations."""

    def test_local_backend_is_eval_backend(self):
        backend = LocalEvalBackend()
        assert isinstance(backend, EvalBackend)

    def test_cloud_backend_is_eval_backend(self):
        mock_provider = AsyncMock(spec=CloudProvider)
        backend = CloudEvalBackend(provider=mock_provider)
        assert isinstance(backend, EvalBackend)


# ---------------------------------------------------------------------------
# CloudEvalBackend
# ---------------------------------------------------------------------------


class TestCloudEvalBackend:
    """CloudEvalBackend delegates to injected CloudProvider."""

    @pytest.mark.asyncio
    async def test_delegates_to_provider(self):
        mock_provider = AsyncMock()
        mock_provider.upload_adapter.return_value = "s3://bucket/adapter"
        mock_provider.submit_eval_job.return_value = "job-123"
        mock_provider.wait_for_result.return_value = EvalResult(eval_score=0.95)

        backend = CloudEvalBackend(provider=mock_provider)
        result = await backend.run_eval("/local/adapter", "scenario.yaml")

        mock_provider.upload_adapter.assert_awaited_once_with("/local/adapter")
        mock_provider.submit_eval_job.assert_awaited_once_with(
            "s3://bucket/adapter", "scenario.yaml"
        )
        mock_provider.wait_for_result.assert_awaited_once_with("job-123")
        assert result.eval_score == 0.95

    @pytest.mark.asyncio
    async def test_propagates_provider_error(self):
        mock_provider = AsyncMock()
        mock_provider.upload_adapter.side_effect = ConnectionError("upload failed")

        backend = CloudEvalBackend(provider=mock_provider)
        with pytest.raises(ConnectionError, match="upload failed"):
            await backend.run_eval("/local/adapter", "scenario.yaml")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateEvalBackend:
    """create_eval_backend factory function."""

    def test_local_default(self):
        backend = create_eval_backend()
        assert isinstance(backend, LocalEvalBackend)

    def test_local_explicit(self):
        backend = create_eval_backend(backend_type="local", min_vram_gb=16)
        assert isinstance(backend, LocalEvalBackend)
        assert backend.min_vram_gb == 16

    def test_cloud_with_provider(self):
        mock_provider = AsyncMock()
        backend = create_eval_backend(
            backend_type="cloud", cloud_provider=mock_provider
        )
        assert isinstance(backend, CloudEvalBackend)

    def test_cloud_without_provider_raises(self):
        with pytest.raises(ValueError, match="cloud_provider required"):
            create_eval_backend(backend_type="cloud")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown eval backend"):
            create_eval_backend(backend_type="tpu")
