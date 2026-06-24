"""Shared fixtures + markers for the ACE-STEP v1.5 pipeline test suite.

Registers the ``gpu`` marker (P1 tests that need a real CUDA device + the real
ACE-STEP model — skipped automatically when no CUDA is present) and provides a
deterministic WAV-synthesis helper so the P0 verifier tests can build tiny audio
files with no heavy audio dependency (stdlib ``wave`` + ``numpy`` only).
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``gpu`` marker locally (P1 real-pipeline tests)."""
    config.addinivalue_line(
        "markers",
        "gpu: requires a real CUDA device + the ACE-STEP model (skipped without CUDA)",
    )


def _has_cuda() -> bool:
    """True iff a CUDA device is available (torch may be absent → False)."""
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch absent / broken → no GPU
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.gpu`` tests when no CUDA device is present.

    This is the runtime arm of the test plan's 3-layer GPU-gating split: the P0
    layer runs everywhere; the P1 ``gpu``-marked layer (real preprocess / train /
    generate) is skipped on CPU-only CI and runs only on a local/cloud GPU box.
    """
    if _has_cuda():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available (P1 GPU-gated test)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


def write_wav(
    path: str | Path,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
    freq_hz: float = 220.0,
    silent: bool = False,
) -> str:
    """Synthesize a deterministic 16-bit PCM WAV and return its path.

    Used by the P0 verifier tests to build tiny structural fixtures (right/wrong
    sample-rate, mono/stereo, silent, short/long) with ZERO heavy dependencies —
    only the Python stdlib ``wave`` module. Deterministic (a fixed sine or pure
    silence), so tests are repeatable and assert only on structural properties.

    Args:
        path: Output file path.
        sample_rate: Frames per second.
        channels: Channel count (1=mono, 2=stereo).
        duration_s: Length in seconds.
        amplitude: Peak amplitude in ``[0, 1]`` (ignored when ``silent``).
        freq_hz: Sine frequency (ignored when ``silent``).
        silent: If True, write pure digital silence (all-zero samples).

    Returns:
        The output path as a string.
    """
    path = str(path)
    n_frames = int(round(sample_rate * duration_s))
    peak = int(amplitude * 32767)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        frames = bytearray()
        for n in range(n_frames):
            if silent:
                sample = 0
            else:
                sample = int(peak * math.sin(2.0 * math.pi * freq_hz * (n / sample_rate)))
            for _ in range(channels):
                frames += struct.pack("<h", sample)
        wf.writeframes(bytes(frames))
    return path


@pytest.fixture
def wav_factory(tmp_path):
    """Return a factory that writes named deterministic WAVs into ``tmp_path``.

    Usage:
        p = wav_factory("good.wav", sample_rate=48000, channels=2, duration_s=1.0)
    """

    def _make(name: str, **kwargs) -> str:
        return write_wav(tmp_path / name, **kwargs)

    return _make
