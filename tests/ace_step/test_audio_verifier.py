"""P0 (CPU/CI) tests for the ACE-STEP audio verifier — the valid-audio smoke.

Risk tier: HIGH (net-new audio-modality verifier; the eval gate for the whole
music pipeline). These tests assert ONLY structural properties (loadable / SR /
channels / duration / RMS non-silence) — NEVER bit-exact waveform equality,
because diffusion generation is non-deterministic. Every structural branch is
exercised independently (one signal at a time) plus the all-pass and mixed-corpus
aggregate cases.

No GPU and no real ACE-STEP model are needed: fixtures are tiny deterministic
WAVs synthesized via the stdlib ``wave`` module (see conftest.write_wav).
"""
from __future__ import annotations

import pytest

from shared.verifiers.builtins.audio_verifier import (
    AudioConfig,
    AudioThresholds,
    AudioValidationResult,
    AudioVerifier,
    _build_audio_verifier,
)
from shared.verifiers.registry import VERIFIER_FACTORIES, build_verifier


# ---------------------------------------------------------------------------
# Registration + protocol contract
# ---------------------------------------------------------------------------

def test_audio_type_is_registered():
    """The module's @register('audio') side-effect populates the registry."""
    import shared.verifiers.builtins  # noqa: F401 - triggers registration imports

    assert "audio" in VERIFIER_FACTORIES


def test_build_verifier_constructs_audio_verifier():
    """build_verifier({'type':'audio'}) returns an AudioVerifier (factory pattern)."""
    import shared.verifiers.builtins  # noqa: F401

    v = build_verifier({"type": "audio"})
    assert isinstance(v, AudioVerifier)
    assert v.name == "audio"


def test_factory_returns_audio_verifier_directly():
    """The registered factory mirrors _build_retrieval_verifier (function, not class)."""
    assert isinstance(_build_audio_verifier({"type": "audio"}), AudioVerifier)


def test_verify_raises_not_implemented():
    """Per-completion verify() is protocol-only and must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        AudioVerifier().verify(None)  # type: ignore[arg-type]


def test_empty_audio_paths_raises_value_error():
    """An empty corpus is a config error, not a silent pass."""
    with pytest.raises(ValueError):
        AudioVerifier().evaluate_audio(AudioConfig(audio_paths=[]))


# ---------------------------------------------------------------------------
# Happy path: a valid 48 kHz / stereo / non-silent file passes every check
# ---------------------------------------------------------------------------

def test_valid_audio_passes(wav_factory):
    good = wav_factory("good.wav", sample_rate=48000, channels=2, duration_s=1.0)
    cfg = AudioConfig(
        audio_paths=[good],
        thresholds=AudioThresholds(
            min_duration_s=0.5, max_duration_s=600.0, require_sr=48000,
            require_channels=2, min_rms=1e-4,
        ),
    )
    result = AudioVerifier().evaluate_audio(cfg)

    assert isinstance(result, AudioValidationResult)
    assert result.passed is True
    assert result.primary_metric_name == "pass_rate"
    assert result.primary_metric == 1.0
    assert result.metrics["num_files"] == 1.0
    assert result.metrics["num_loadable"] == 1.0
    assert result.metrics["num_passed"] == 1.0
    assert result.detail["failures"] == {}
    entry = result.detail["per_file"][good]
    assert entry["ok"] is True
    assert entry["sr"] == 48000
    assert entry["channels"] == 2
    assert entry["rms"] > 0.0
    assert 0.99 <= entry["duration_s"] <= 1.01  # structural, not exact


# ---------------------------------------------------------------------------
# Each structural failure mode — exercised INDEPENDENTLY (one signal at a time)
# ---------------------------------------------------------------------------

def test_wrong_sample_rate_fails(wav_factory):
    """44.1 kHz file fails the 48 kHz requirement."""
    f = wav_factory("sr.wav", sample_rate=44100, channels=2, duration_s=1.0)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(require_sr=48000, require_channels=2))
    )
    assert result.passed is False
    reasons = result.detail["per_file"][f]["reasons"]
    assert any("sample_rate" in r for r in reasons)


def test_mono_fails_stereo_requirement(wav_factory):
    """A mono file fails the stereo (2-channel) requirement."""
    f = wav_factory("mono.wav", sample_rate=48000, channels=1, duration_s=1.0)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(require_sr=48000, require_channels=2))
    )
    assert result.passed is False
    assert any("channels" in r for r in result.detail["per_file"][f]["reasons"])


def test_silent_fails_rms_floor(wav_factory):
    """Pure digital silence fails a non-zero RMS floor."""
    f = wav_factory("silent.wav", sample_rate=48000, channels=2, duration_s=1.0, silent=True)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(require_sr=48000, require_channels=2, min_rms=1e-3))
    )
    assert result.passed is False
    assert any("silent" in r for r in result.detail["per_file"][f]["reasons"])
    # And the RMS really is ~0 for silence.
    assert result.detail["per_file"][f]["rms"] == pytest.approx(0.0, abs=1e-9)


def test_too_short_fails(wav_factory):
    f = wav_factory("short.wav", sample_rate=48000, channels=2, duration_s=0.2)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(min_duration_s=1.0, require_sr=48000, require_channels=2))
    )
    assert result.passed is False
    assert any("too_short" in r for r in result.detail["per_file"][f]["reasons"])


def test_too_long_fails(wav_factory):
    f = wav_factory("long.wav", sample_rate=48000, channels=2, duration_s=2.0)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(max_duration_s=1.0, require_sr=48000, require_channels=2))
    )
    assert result.passed is False
    assert any("too_long" in r for r in result.detail["per_file"][f]["reasons"])


def test_unloadable_path_fails_gracefully(tmp_path):
    """A missing / non-audio file is a structural fail, not a crash."""
    missing = str(tmp_path / "does_not_exist.wav")
    notaudio = tmp_path / "garbage.wav"
    notaudio.write_bytes(b"this is not a wav file")
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[missing, str(notaudio)], thresholds=AudioThresholds())
    )
    assert result.passed is False
    assert result.metrics["num_loadable"] == 0.0
    for p in (missing, str(notaudio)):
        entry = result.detail["per_file"][p]
        assert entry["loadable"] is False
        assert any("unloadable" in r for r in entry["reasons"])


# ---------------------------------------------------------------------------
# Disabled-threshold semantics (sentinel 0 = don't gate)
# ---------------------------------------------------------------------------

def test_disabled_thresholds_do_not_gate(wav_factory):
    """With all sentinels at 0, even a 44.1 kHz mono silent file is loadable→pass."""
    f = wav_factory("anything.wav", sample_rate=44100, channels=1, duration_s=0.1, silent=True)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(
            audio_paths=[f],
            thresholds=AudioThresholds(
                min_duration_s=0.0, max_duration_s=0.0, require_sr=0, require_channels=0, min_rms=0.0
            ),
        )
    )
    # Loadable + nothing gated → passes. This pins the "0 = disabled" contract:
    # a regression that treated 0 as an active floor would flip this to fail.
    assert result.passed is True
    assert result.detail["per_file"][f]["reasons"] == []


def test_silence_passes_when_rms_floor_disabled(wav_factory):
    """min_rms=0 must NOT gate silence (sentinel semantics, counter to the floor test)."""
    f = wav_factory("silent2.wav", sample_rate=48000, channels=2, duration_s=1.0, silent=True)
    result = AudioVerifier().evaluate_audio(
        AudioConfig(audio_paths=[f], thresholds=AudioThresholds(require_sr=48000, require_channels=2, min_rms=0.0))
    )
    assert result.passed is True


# ---------------------------------------------------------------------------
# Corpus aggregation: pass_rate + all-must-pass semantics
# ---------------------------------------------------------------------------

def test_mixed_corpus_aggregates_and_fails_overall(wav_factory):
    """One bad file in a 3-file corpus → overall fail, pass_rate = 2/3."""
    good1 = wav_factory("g1.wav", sample_rate=48000, channels=2, duration_s=1.0)
    good2 = wav_factory("g2.wav", sample_rate=48000, channels=2, duration_s=1.5)
    bad = wav_factory("b.wav", sample_rate=44100, channels=2, duration_s=1.0)
    thr = AudioThresholds(require_sr=48000, require_channels=2, min_rms=1e-4)
    result = AudioVerifier().evaluate_audio(AudioConfig(audio_paths=[good1, good2, bad], thresholds=thr))

    assert result.passed is False
    assert result.metrics["num_files"] == 3.0
    assert result.metrics["num_passed"] == 2.0
    assert result.primary_metric == pytest.approx(2.0 / 3.0)
    assert set(result.detail["failures"]) == {bad}


def test_all_pass_corpus(wav_factory):
    files = [
        wav_factory(f"ok{i}.wav", sample_rate=48000, channels=2, duration_s=1.0)
        for i in range(3)
    ]
    thr = AudioThresholds(require_sr=48000, require_channels=2, min_rms=1e-4)
    result = AudioVerifier().evaluate_audio(AudioConfig(audio_paths=files, thresholds=thr))
    assert result.passed is True
    assert result.primary_metric == 1.0


def test_default_thresholds_are_48k_stereo():
    """The dataclass defaults encode the ACE-STEP v1.5 contract (48 kHz, stereo)."""
    thr = AudioThresholds()
    assert thr.require_sr == 48000
    assert thr.require_channels == 2
    assert thr.max_duration_s == 0.0  # no cap by default
    assert thr.min_rms == 0.0  # silence allowed unless a floor is set
