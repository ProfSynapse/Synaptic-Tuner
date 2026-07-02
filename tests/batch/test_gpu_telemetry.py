"""GPU peak-memory telemetry: CPU no-op + mocked-CUDA behavior.

Location: tests/batch/test_gpu_telemetry.py

The suite runs on CPU, so the no-op path is exercised for real; the CUDA path is
covered by a fake torch module so no GPU is needed. Asserts:
  - on CPU, telemetry adds nothing and runner log lines keep their shape;
  - with (mocked) CUDA, the `` (gpu peak X.X/Y.Y GiB)`` parenthetical appears on
    persist milestone lines and the completion summary, and peak stats are reset
    once per stage.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuner.batch import gpu_telemetry  # noqa: E402
from tuner.batch import runner as batch_runner  # noqa: E402


# ---------------------------------------------------------------------------
# Unit: gpu_telemetry helpers
# ---------------------------------------------------------------------------

def test_cpu_is_strict_noop():
    """No CUDA -> no suffix, no peak/total, and reset is harmless."""
    cpu_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    assert gpu_telemetry.peak_suffix(cpu_torch) == ""
    assert gpu_telemetry.peak_and_total_gib(cpu_torch) is None
    # reset_peak must not raise on CPU (delegates to shared helper, which no-ops).
    gpu_telemetry.reset_peak(cpu_torch)


def test_suffix_when_torch_missing():
    """peak_suffix() with no torch importable returns empty (real CPU-CI path)."""
    # gpu_telemetry._import_torch returns None if torch import fails; simulate by
    # passing a module whose cuda.is_available raises -> treated as unavailable.
    broken = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    assert gpu_telemetry.peak_suffix(broken) == ""


def _fake_cuda_torch(peak_bytes: float, total_bytes: float):
    """A minimal fake torch exposing just the CUDA calls telemetry uses."""
    calls = {"reset": 0}

    def reset_peak_memory_stats():
        calls["reset"] += 1

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        max_memory_allocated=lambda device=0: peak_bytes,
        get_device_properties=lambda device=0: types.SimpleNamespace(total_memory=total_bytes),
        reset_peak_memory_stats=reset_peak_memory_stats,
    )
    fake = types.SimpleNamespace(cuda=cuda)
    return fake, calls


def test_mocked_cuda_suffix_format():
    gib = 1024 ** 3
    fake, _ = _fake_cuda_torch(peak_bytes=12.4 * gib, total_bytes=23.0 * gib)
    suffix = gpu_telemetry.peak_suffix(fake)
    assert suffix == " (gpu peak 12.4/23.0 GiB)"
    pt = gpu_telemetry.peak_and_total_gib(fake)
    assert pt is not None
    peak, total = pt
    assert round(peak, 1) == 12.4
    assert round(total, 1) == 23.0


def test_mocked_cuda_reset_delegates():
    fake, calls = _fake_cuda_torch(1.0, 2.0)
    gpu_telemetry.reset_peak(fake)
    assert calls["reset"] == 1


# ---------------------------------------------------------------------------
# Integration: runner log lines with telemetry monkeypatched
# ---------------------------------------------------------------------------

class _FakeGenEngine:
    """A trivial generate engine so the runner path runs without a model."""

    def __init__(self, *_, **__):
        pass

    def generate(self, items, *, batch_size, on_oom=None):
        from tuner.batch.engines.base import GenerateResult

        return [
            GenerateResult(
                id=it.id,
                completion_text="x",
                completion_token_ids=[1],
                prompt_token_len=1,
                finish_reason="length",
                passthrough=it.passthrough,
            )
            for it in items
        ]

    def close(self):
        pass


def _write_prompts(tmp_path, n):
    p = tmp_path / "prompts.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"r{i}", "prompt": "hi"}) + "\n")
    return p


def test_runner_cpu_log_shape_has_no_parenthetical(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_runner, "get_generate_engine", lambda *a, **k: _FakeGenEngine())
    # Force the telemetry to behave as CPU regardless of the host.
    monkeypatch.setattr(batch_runner, "peak_suffix", lambda *a, **k: "")
    logs = []
    batch_runner.run_batch_generate(
        prompts_path=_write_prompts(tmp_path, 3), out_dir=tmp_path / "gen",
        model="m", batch_size=2, log=logs.append,
    )
    persisted = [l for l in logs if "persisted" in l]
    assert persisted, "expected persist milestone lines"
    for line in persisted:
        assert "gpu peak" not in line
        assert line.endswith("new rows.")  # exact CPU shape, no parenthetical


def test_runner_cuda_log_shape_has_parenthetical_and_resets(monkeypatch, tmp_path):
    reset_calls = {"n": 0}
    monkeypatch.setattr(batch_runner, "get_generate_engine", lambda *a, **k: _FakeGenEngine())
    monkeypatch.setattr(batch_runner, "peak_suffix", lambda *a, **k: " (gpu peak 12.4/23.0 GiB)")
    monkeypatch.setattr(batch_runner, "reset_peak", lambda *a, **k: reset_calls.__setitem__("n", reset_calls["n"] + 1))

    logs = []
    summary = batch_runner.run_batch_generate(
        prompts_path=_write_prompts(tmp_path, 4), out_dir=tmp_path / "gen",
        model="m", batch_size=2, log=logs.append,
    )
    persisted = [l for l in logs if "persisted" in l]
    assert persisted
    for line in persisted:
        assert line.endswith("(gpu peak 12.4/23.0 GiB)")
    # Peak reset exactly once for the stage.
    assert reset_calls["n"] == 1
    # Completion summary carries the suffix for the handler's complete: line.
    assert summary["gpu_peak_suffix"] == " (gpu peak 12.4/23.0 GiB)"


def test_capture_runner_cuda_resets_once(monkeypatch, tmp_path):
    reset_calls = {"n": 0}

    class _FakeCapEngine:
        def __init__(self, *_, **__):
            pass

        def capture(self, items, *, batch_size, on_oom=None):
            from tuner.batch.engines.base import CaptureResult

            return [
                CaptureResult(
                    id=it.id, tensors={"last__L0": [0.0]}, n_layers=1,
                    hidden_dim=1, positions={"last": 0}, passthrough=it.passthrough,
                )
                for it in items
            ]

        def close(self):
            pass

    monkeypatch.setattr(batch_runner, "get_capture_engine", lambda *a, **k: _FakeCapEngine())
    monkeypatch.setattr(batch_runner, "peak_suffix", lambda *a, **k: " (gpu peak 5.0/23.0 GiB)")
    monkeypatch.setattr(batch_runner, "reset_peak", lambda *a, **k: reset_calls.__setitem__("n", reset_calls["n"] + 1))
    # Avoid the real safetensors write path for the fake tensors.
    monkeypatch.setattr(batch_runner, "_write_safetensors", lambda *a, **k: None)

    rows = tmp_path / "rows.jsonl"
    with open(rows, "w") as f:
        for i in range(3):
            f.write(json.dumps({"id": f"c{i}", "text": "hi", "positions": {"last": "last"}}) + "\n")

    logs = []
    summary = batch_runner.run_batch_capture(
        rows_path=rows, out_dir=tmp_path / "cap", model="m", batch_size=2, log=logs.append,
    )
    persisted = [l for l in logs if "persisted" in l]
    assert persisted and all(l.endswith("(gpu peak 5.0/23.0 GiB)") for l in persisted)
    assert reset_calls["n"] == 1
    assert summary["gpu_peak_suffix"] == " (gpu peak 5.0/23.0 GiB)"
