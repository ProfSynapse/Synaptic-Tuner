"""
shared/verifiers/builtins/audio_verifier.py

Corpus-level audio verifier. Scores a set of rendered audio files against a
**structural** pass/warn/fail ladder: each file must be loadable, non-silent
(RMS above a floor), at the required sample rate (48 kHz), with the required
channel count (stereo), and a duration inside ``[min_duration_s, max_duration_s]``.

This is the Phase-1 "valid-audio smoke" verifier for the ACE-STEP v1.5 music
pipeline: it answers "did training/generation produce a loadable, non-silent
audio file of the expected shape?" — NOT "is the music good." Diffusion
generation is non-deterministic, so this verifier asserts ONLY structural
properties and NEVER bit-exact waveform equality. FAD / CLAP (distributional
realism, text-audio alignment) are Phase-2 drop-ins on the same ``metrics``
field of :class:`AudioConfig`.

Registration vs invocation (mirrors retrieval_verifier.py's R9 split):
- This module registers a factory under the registry ``type`` key ``"audio"``
  via ``@register("audio")`` so the verifier is discoverable/buildable through
  ``shared.verifiers.registry.build_verifier`` like every other builtin. The
  registry decorates a FACTORY FUNCTION (``_build_audio_verifier``) that returns
  an :class:`AudioVerifier` — same shape as ``_build_retrieval_verifier`` at
  ``retrieval_verifier.py:145`` (the registry stores ``type -> factory``, not a
  bare class).
- BUT audio is corpus-level, not per-completion. The existing
  ``Verifier.verify(VerifierInput) -> VerifierOutput`` contract maps one
  completion to a scalar; audio scores a whole set of files. So the verifier is
  invoked through a DEDICATED corpus-level entry point,
  :meth:`AudioVerifier.evaluate_audio`, which consumes an :class:`AudioConfig`
  (NOT a ``VerifierInput``). ``verify()`` is implemented only to satisfy the
  ``Verifier`` protocol and is intentionally unused for audio — calling it raises
  ``NotImplementedError`` pointing at the corpus-level entry point.

shared/ purity (NON-NEGOTIABLE): this module imports nothing from ``Evaluator/``
or ``Trainers/``. Audio decoding is done lazily inside
:meth:`AudioVerifier.evaluate_audio` so importing this module for registration
stays cheap and dependency-free.

Decoder strategy (Phase-1 = WAV smoke, no heavy deps in CI):
- The locked generation output format is **WAV** (contract §7, ``audio_format="wav"``),
  and WAV decodes with the Python **stdlib** ``wave`` module + ``numpy`` — both
  already available wherever the eval suite runs. So the Phase-1 valid-audio smoke
  needs NO new heavy dependency, and the P0 verifier tests run in plain CI.
- For non-WAV inputs (mp3/flac/ogg — e.g. a v1 corpus or a future format), the
  loader falls back to ``soundfile`` IF it is importable (it ships in the
  ACE-STEP training image alongside ``torchaudio``), and otherwise records an
  honest "unloadable: <fmt> needs soundfile" structural failure rather than
  crashing. This keeps ``shared/`` import-pure and CI-light while still scoring
  the real generated WAVs.

Used by:
- ``shared/verifiers/builtins/__init__.py`` imports this module for its
  ``@register`` side-effect.
- ``Evaluator/runner.py`` resolves a scenario's ``audio_config`` into an
  :class:`AudioConfig` and calls :meth:`evaluate_audio` once per audio scenario
  (a sibling to the per-completion loop, mirroring the retrieval branch).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from shared.verifiers.contract import VerifierInput, VerifierOutput
from shared.verifiers.registry import register

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + result dataclasses (mirror RetrievalThresholds / RetrievalConfig /
# RetrievalValidationResult at retrieval_verifier.py:57-137)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioThresholds:
    """Structural threshold block controlling the pass/warn/fail ladder.

    A file FAILS if any enabled check is violated. A check is "enabled" when its
    threshold is non-trivial (e.g. ``require_sr > 0``); a threshold left at its
    disabling sentinel (``0``) is reported but never gates, mirroring how
    :class:`RetrievalThresholds` treats metrics without a ``min`` entry.

    Attributes:
        min_duration_s: Minimum audio duration in seconds. ``0`` disables the
            lower bound. This is OUR render-quality policy, NOT an ACE-STEP
            constraint (ACE-STEP has no min-duration validation, only a 240 s
            cap). A real render needs ≥ ~11.88 s to yield a full 128-frame DCAE
            latent, so a music smoke typically sets this to ~12.
        max_duration_s: Maximum audio duration in seconds. ``0`` disables the
            upper bound (no cap). ACE-STEP's own ceiling is 240 s.
        require_sr: Required sample rate in Hz (ACE-STEP v1.5 = 48000). ``0``
            disables the sample-rate check.
        require_channels: Required channel count (2 = stereo). ``0`` disables the
            channel check.
        min_rms: Non-silence floor: a file's root-mean-square amplitude must be
            ``>= min_rms``. ``0`` disables the non-silence check (any file,
            including pure digital silence, passes the RMS gate).
    """

    min_duration_s: float = 0.0
    max_duration_s: float = 0.0  # 0 = no cap
    require_sr: int = 48000
    require_channels: int = 2  # stereo
    min_rms: float = 0.0  # non-silence floor


@dataclass(frozen=True)
class AudioConfig:
    """Inputs for a corpus-level audio (valid-audio smoke) evaluation.

    Phase-1 uses only ``audio_paths`` + ``thresholds`` (structural smoke). The
    ``reference_set`` / ``captions`` / ``metrics`` fields are Phase-2 drop-ins for
    FAD / CLAP on the SAME seam — empty/None in Phase-1.

    Attributes:
        audio_paths: Rendered audio files to score (required). These are
            PRE-RENDERED — the generation step (ACE-STEP ``cli.py``, contract §7)
            runs separately and hands paths in; this verifier never generates.
        thresholds: The structural threshold block.
        reference_set: Reference-distribution path for FAD (Phase-2).
        captions: Per-file captions for CLAP text-audio alignment (Phase-2).
        metrics: Phase-2 metric names to compute (e.g. ``("fad", "clap")``).
            Empty in Phase-1 (structural smoke only).
    """

    audio_paths: Sequence[str]
    thresholds: AudioThresholds = field(default_factory=AudioThresholds)
    reference_set: str | None = None
    captions: Sequence[str] | None = None
    metrics: Sequence[str] = ()


@dataclass(frozen=True)
class AudioValidationResult:
    """Outcome of a corpus-level audio evaluation (mirrors RetrievalValidationResult).

    Attributes:
        metrics: Aggregate structural metrics across the file set, e.g.
            ``{"num_files", "num_loadable", "num_passed", "pass_rate"}``.
        passed: True iff EVERY file passed all enabled structural checks.
        warned: True iff ``passed`` and at least one advisory condition fired
            (Phase-1 reserves this for forward-compat; currently never set when
            passed, since structural checks are hard pass/fail).
        primary_metric_name: Spec of the headline metric (``"pass_rate"``).
        primary_metric: Value of the headline metric (fraction of files passing).
        detail: Per-file diagnostics: ``{path: {loadable, sr, channels,
            duration_s, rms, ok, reasons:[...]}}`` plus a ``failures`` summary.
    """

    metrics: dict[str, float]
    passed: bool
    warned: bool
    primary_metric_name: str
    primary_metric: float
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registration + verifier (mirror retrieval_verifier.py:144-169)
# ---------------------------------------------------------------------------

@register("audio")
def _build_audio_verifier(spec: Mapping) -> "AudioVerifier":
    """Factory for the ``audio`` verifier type.

    The spec mapping is accepted for registry-uniformity; the audio verifier is
    stateless and configured per-call via :class:`AudioConfig`, so no spec fields
    are consumed here today (mirrors ``_build_retrieval_verifier``).
    """
    return AudioVerifier()


class AudioVerifier:
    """Corpus-level audio verifier (see module docstring for the registration/
    invocation split)."""

    name = "audio"

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        """Per-completion entry point — NOT used for audio.

        Audio is corpus-level; use :meth:`evaluate_audio`. This method exists
        only to satisfy the ``Verifier`` protocol (mirrors
        ``RetrievalVerifier.verify``).
        """
        raise NotImplementedError(
            "AudioVerifier is corpus-level; call evaluate_audio(AudioConfig), "
            "not the per-completion verify(VerifierInput) entry point."
        )

    def evaluate_audio(self, cfg: AudioConfig) -> AudioValidationResult:
        """Score each audio file structurally, then aggregate to a corpus verdict.

        Phase-1 structural checks per file: loadable, non-silent (RMS >=
        ``min_rms``), sample-rate == ``require_sr``, channels == ``require_channels``,
        duration in ``[min_duration_s, max_duration_s]``. NEVER bit-exact —
        diffusion is non-deterministic.

        Heavy audio deps (``soundfile``, ``numpy``) are imported lazily here so
        module import (for registration) stays cheap and ``shared/``-pure.

        Args:
            cfg: The audio configuration (paths + structural thresholds).

        Returns:
            An :class:`AudioValidationResult`: ``passed`` iff every file passed
            all enabled checks; ``detail`` carries per-file diagnostics.

        Raises:
            ValueError: If ``cfg.audio_paths`` is empty.
        """
        if not cfg.audio_paths:
            raise ValueError("AudioConfig.audio_paths is empty; nothing to evaluate.")

        # Lazy import (shared/-purity + cheap registration import). numpy is the
        # only hard dependency; the actual decode is delegated to _decode_audio,
        # which uses stdlib `wave` for WAV and an optional `soundfile` fallback.
        import numpy as np

        thr = cfg.thresholds
        per_file: dict[str, dict[str, Any]] = {}
        num_loadable = 0
        num_passed = 0

        for path in cfg.audio_paths:
            reasons: list[str] = []
            entry: dict[str, Any] = {
                "loadable": False,
                "sr": None,
                "channels": None,
                "duration_s": None,
                "rms": None,
                "ok": False,
                "reasons": reasons,
            }

            # ---- loadable? ----
            try:
                data, sr = _decode_audio(path, np)  # data: (frames, channels) float
            except Exception as exc:  # noqa: BLE001 - any decode failure is a structural fail
                reasons.append(f"unloadable: {exc}")
                per_file[path] = entry
                continue

            num_loadable += 1
            entry["loadable"] = True

            frames = int(data.shape[0])
            channels = int(data.shape[1])
            sr = int(sr)
            duration_s = (frames / sr) if sr > 0 else 0.0
            # RMS across all samples/channels; 0.0 for an empty buffer.
            rms = float(np.sqrt(np.mean(np.square(data.astype(np.float64))))) if frames > 0 else 0.0

            entry["sr"] = sr
            entry["channels"] = channels
            entry["duration_s"] = duration_s
            entry["rms"] = rms

            # ---- structural checks (each gated only when its threshold is enabled) ----
            if thr.require_sr and sr != thr.require_sr:
                reasons.append(f"sample_rate {sr} != required {thr.require_sr}")
            if thr.require_channels and channels != thr.require_channels:
                reasons.append(f"channels {channels} != required {thr.require_channels}")
            if thr.min_rms and rms < thr.min_rms:
                reasons.append(f"silent: rms {rms:.6g} < min_rms {thr.min_rms:.6g}")
            if thr.min_duration_s and duration_s < thr.min_duration_s:
                reasons.append(
                    f"too_short: duration {duration_s:.4g}s < min {thr.min_duration_s:.4g}s"
                )
            if thr.max_duration_s and duration_s > thr.max_duration_s:
                reasons.append(
                    f"too_long: duration {duration_s:.4g}s > max {thr.max_duration_s:.4g}s"
                )

            entry["ok"] = not reasons
            if entry["ok"]:
                num_passed += 1
            per_file[path] = entry

        num_files = len(cfg.audio_paths)
        pass_rate = (num_passed / num_files) if num_files else 0.0
        passed = num_passed == num_files
        failures = {p: e["reasons"] for p, e in per_file.items() if not e["ok"]}

        return AudioValidationResult(
            metrics={
                "num_files": float(num_files),
                "num_loadable": float(num_loadable),
                "num_passed": float(num_passed),
                "pass_rate": pass_rate,
            },
            passed=passed,
            warned=False,
            primary_metric_name="pass_rate",
            primary_metric=pass_rate,
            detail={
                "per_file": per_file,
                "failures": failures,
                "thresholds": {
                    "min_duration_s": thr.min_duration_s,
                    "max_duration_s": thr.max_duration_s,
                    "require_sr": thr.require_sr,
                    "require_channels": thr.require_channels,
                    "min_rms": thr.min_rms,
                },
            },
        )


# ---------------------------------------------------------------------------
# Decoding helper (stdlib WAV + optional soundfile fallback)
# ---------------------------------------------------------------------------

def _decode_audio(path: str, np: Any) -> tuple[Any, int]:
    """Decode an audio file to ``(samples, sample_rate)``.

    Returns ``samples`` as a 2D float array shaped ``(frames, channels)`` with
    values normalized to roughly ``[-1.0, 1.0]`` (so the RMS floor is amplitude-
    scale-independent), plus the integer sample rate.

    WAV is decoded with the Python **stdlib** ``wave`` module (no heavy dep, so
    the Phase-1 smoke + its tests run in plain CI). Any non-WAV extension is
    delegated to ``soundfile`` when importable; if ``soundfile`` is absent, a
    ``RuntimeError`` is raised (the caller records it as a structural
    "unloadable" failure rather than crashing the run).

    Args:
        path: Filesystem path to the audio file.
        np: The (lazily-imported) numpy module, passed in by the caller.

    Returns:
        ``(samples_2d_float, sample_rate)``.

    Raises:
        RuntimeError: For a non-WAV file when ``soundfile`` is not installed.
        wave.Error / OSError / ValueError: On a malformed/unreadable file.
    """
    lower = path.lower()
    if lower.endswith(".wav"):
        import wave

        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Map PCM sample width -> numpy dtype + full-scale for normalization.
        if sampwidth == 1:
            # 8-bit PCM is unsigned, centered at 128.
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

        if n_channels > 1:
            arr = arr.reshape(-1, n_channels)
        else:
            arr = arr.reshape(-1, 1)
        return arr, int(sr)

    # Non-WAV: optional soundfile fallback (ships in the ACE-STEP image).
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"{path}: non-WAV audio needs 'soundfile' (not installed in this env)"
        ) from exc

    data, sr = sf.read(path, always_2d=True)  # already float, (frames, channels)
    return np.asarray(data, dtype=np.float64), int(sr)
