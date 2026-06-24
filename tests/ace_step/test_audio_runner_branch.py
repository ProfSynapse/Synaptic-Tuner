"""P0 (CPU/CI) tests for the Evaluator audio_config runner branch.

Verifies that a scenario carrying ``metadata['audio_config']`` is routed to the
corpus-level audio path (a SIBLING to the per-completion loop, mirroring the
retrieval branch) and that the result lands on ``EvaluationRecord.audio`` with
the correct pass/warn/fail status ladder. No backend / chat() turn is involved.
"""
from __future__ import annotations

import pytest

from Evaluator.prompt_sets import PromptCase
from Evaluator.runner import (
    EvaluationRecord,
    _build_audio_config,
    _evaluate_audio_case,
    _evaluate_single_case,
)
from shared.verifiers.builtins.audio_verifier import AudioConfig, AudioThresholds


def _case(audio_config: dict) -> PromptCase:
    return PromptCase(case_id="audio-1", question="", metadata={"audio_config": audio_config})


# ---------------------------------------------------------------------------
# _build_audio_config — required-key validation + threshold defaults
# ---------------------------------------------------------------------------

def test_build_audio_config_requires_audio_paths():
    with pytest.raises(ValueError, match="audio_paths"):
        _build_audio_config({"thresholds": {"require_sr": 48000}})


def test_build_audio_config_rejects_empty_paths():
    with pytest.raises(ValueError, match="non-empty"):
        _build_audio_config({"audio_paths": []})


def test_build_audio_config_applies_defaults_and_overrides():
    cfg = _build_audio_config(
        {"audio_paths": ["a.wav", "b.wav"], "thresholds": {"min_duration_s": 2.0, "min_rms": 0.01}}
    )
    assert isinstance(cfg, AudioConfig)
    assert list(cfg.audio_paths) == ["a.wav", "b.wav"]
    # overridden
    assert cfg.thresholds.min_duration_s == 2.0
    assert cfg.thresholds.min_rms == 0.01
    # defaulted (ACE-STEP v1.5 contract)
    assert cfg.thresholds.require_sr == 48000
    assert cfg.thresholds.require_channels == 2


def test_build_audio_config_passes_through_phase2_fields():
    cfg = _build_audio_config(
        {"audio_paths": ["a.wav"], "reference_set": "ref/", "captions": ["jazz"], "metrics": ["fad"]}
    )
    assert cfg.reference_set == "ref/"
    assert list(cfg.captions) == ["jazz"]
    assert tuple(cfg.metrics) == ("fad",)


# ---------------------------------------------------------------------------
# _evaluate_audio_case — result on EvaluationRecord.audio + status ladder
# ---------------------------------------------------------------------------

def test_audio_case_pass(wav_factory):
    good = wav_factory("good.wav", sample_rate=48000, channels=2, duration_s=1.0)
    case = _case({"audio_paths": [good], "thresholds": {"require_sr": 48000, "require_channels": 2, "min_rms": 1e-4}})

    record = _evaluate_audio_case(case)
    assert isinstance(record, EvaluationRecord)
    assert record.audio is not None
    assert record.error is None
    assert record.audio.passed is True
    assert record.status == "pass"
    assert record.passed is True


def test_audio_case_fail(wav_factory):
    bad = wav_factory("bad.wav", sample_rate=44100, channels=2, duration_s=1.0)
    case = _case({"audio_paths": [bad], "thresholds": {"require_sr": 48000, "require_channels": 2}})

    record = _evaluate_audio_case(case)
    assert record.audio is not None
    assert record.audio.passed is False
    assert record.status == "fail"
    assert record.failed is True


def test_audio_case_missing_key_is_error_not_crash():
    """A malformed audio_config surfaces as EvaluationRecord.error → fail status."""
    case = _case({"thresholds": {"require_sr": 48000}})  # no audio_paths
    record = _evaluate_audio_case(case)
    assert record.audio is None
    assert record.error is not None
    assert "Audio evaluation error" in record.error
    assert record.status == "fail"


# ---------------------------------------------------------------------------
# evaluate_case dispatch — the audio_config branch is reached as a sibling
# ---------------------------------------------------------------------------

def test_single_case_routes_audio_config_without_backend(wav_factory):
    """_evaluate_single_case() with audio_config must NOT call the backend client.

    A None client would crash the per-completion path; reaching a pass proves
    the audio branch short-circuits BEFORE any chat() turn (the sibling fence).
    """
    good = wav_factory("g.wav", sample_rate=48000, channels=2, duration_s=1.0)
    case = _case({"audio_paths": [good], "thresholds": {"require_sr": 48000, "require_channels": 2, "min_rms": 1e-4}})

    record = _evaluate_single_case(case, client=None, dry_run=False)  # client untouched
    assert record.audio is not None
    assert record.status == "pass"


def test_single_case_audio_takes_priority_over_correctness(wav_factory):
    """An audio scenario is scored ONLY on its structural ladder, never additionally
    subjected to per-completion correctness assertions (mirrors retrieval fence)."""
    bad = wav_factory("b.wav", sample_rate=44100, channels=2, duration_s=1.0)
    case = _case({"audio_paths": [bad], "thresholds": {"require_sr": 48000}})
    # Even if a correctness block were present, the audio branch returns first.
    case.metadata["expected"] = {"some": "assertion"}

    record = _evaluate_single_case(case, client=None, dry_run=False)
    assert record.audio is not None
    assert record.correctness is None
    assert record.status == "fail"
