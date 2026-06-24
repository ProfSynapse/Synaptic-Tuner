"""P0 (CPU/CI) E2E test: the audio verifier against the golden fixtures on disk.

Unlike test_audio_verifier.py (which synthesizes WAVs into tmp_path), this test
scores real files under scratch/fixtures/ace_step/ through the runner's
audio_config branch — proving the verifier reads actual on-disk artifacts (no
mock of the decode seam), the same path a real ACE-STEP render feeds.

scratch/ is GITIGNORED (per the repo's "test outputs → scratch/, never /tmp"
rule), so the golden WAVs are NOT committed. This module SELF-PROVISIONS them
deterministically on first use (via the conftest write_wav helper) so the E2E
test always RUNS in a fresh checkout / CI rather than skipping for absent
fixtures. The committed scratch/fixtures/ace_step/make_fixtures.py is the same
generator for manual/standalone use.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from Evaluator.prompt_sets import PromptCase
from Evaluator.runner import _evaluate_single_case

from .conftest import write_wav

FIXTURES = Path(__file__).resolve().parents[2] / "scratch" / "fixtures" / "ace_step"

# Policy: a valid ACE-STEP render must be at least this long. This is OUR check
# (AudioThresholds.min_duration_s), NOT an ACE-STEP constraint — ACE-STEP has no
# min-duration validation (only a 240 s cap). 12 s is the team-lead/architect
# guidance: a real clip needs ≥ ~11.88 s to yield a full 128-frame DCAE latent,
# so the golden "valid" fixture is sized like a real render.
MIN_DURATION_S = 12.0

# (name -> synthesis kwargs) for the golden fixtures this E2E exercises. The
# "valid" clip is ≥ MIN_DURATION_S (real-render length); the too-short negative
# is deliberately below it to exercise the duration policy. The SR/channel/
# silence negatives stay short — they fail on their own signal, so length is
# irrelevant and tiny keeps the gitignored scratch dir light.
_GOLDEN = {
    "valid_48k_stereo.wav": dict(sample_rate=48000, channels=2, duration_s=MIN_DURATION_S),
    "too_short_48k_stereo.wav": dict(sample_rate=48000, channels=2, duration_s=1.0),
    "wrong_sr_44k_stereo.wav": dict(sample_rate=44100, channels=2, duration_s=0.5),
    "mono_48k.wav": dict(sample_rate=48000, channels=1, duration_s=0.5),
    "silent_48k_stereo.wav": dict(sample_rate=48000, channels=2, duration_s=0.5, silent=True),
}


@pytest.fixture(scope="module", autouse=True)
def _provision_golden_fixtures():
    """Deterministically (re)generate the golden WAVs into the gitignored scratch dir."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, kwargs in _GOLDEN.items():
        write_wav(FIXTURES / name, **kwargs)


def _require(name: str) -> str:
    return str(FIXTURES / name)


def _audio_case(paths, thresholds) -> PromptCase:
    return PromptCase(
        case_id="e2e",
        question="",
        metadata={"audio_config": {"audio_paths": paths, "thresholds": thresholds}},
    )


def test_committed_valid_fixture_passes_via_runner():
    good = _require("valid_48k_stereo.wav")
    case = _audio_case(
        [good],
        {"require_sr": 48000, "require_channels": 2, "min_rms": 1e-4, "min_duration_s": MIN_DURATION_S},
    )
    record = _evaluate_single_case(case, client=None, dry_run=False)
    assert record.audio is not None
    assert record.status == "pass"
    entry = record.audio.detail["per_file"][good]
    assert entry["sr"] == 48000
    assert entry["channels"] == 2
    # The valid fixture clears the ≥12 s render-length policy.
    assert entry["duration_s"] >= MIN_DURATION_S


def test_committed_too_short_fixture_fails_duration_policy():
    """A 1 s clip fails OUR ≥12 s render-length policy (min_duration_s)."""
    short = _require("too_short_48k_stereo.wav")
    case = _audio_case(
        [short],
        {"require_sr": 48000, "require_channels": 2, "min_rms": 1e-4, "min_duration_s": MIN_DURATION_S},
    )
    record = _evaluate_single_case(case, client=None, dry_run=False)
    assert record.audio is not None
    assert record.status == "fail"
    assert any("too_short" in r for r in record.audio.detail["failures"][short])


def test_committed_negative_fixtures_fail_via_runner():
    paths = [
        _require("wrong_sr_44k_stereo.wav"),
        _require("mono_48k.wav"),
        _require("silent_48k_stereo.wav"),
    ]
    case = _audio_case(paths, {"require_sr": 48000, "require_channels": 2, "min_rms": 1e-3})
    record = _evaluate_single_case(case, client=None, dry_run=False)
    assert record.audio is not None
    assert record.status == "fail"
    # Each negative fixture fails for its OWN reason (independent signals).
    failures = record.audio.detail["failures"]
    assert any("sample_rate" in r for r in failures[_require("wrong_sr_44k_stereo.wav")])
    assert any("channels" in r for r in failures[_require("mono_48k.wav")])
    assert any("silent" in r for r in failures[_require("silent_48k_stereo.wav")])
