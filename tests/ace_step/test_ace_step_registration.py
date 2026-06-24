"""P0 (CPU/CI) tests: ace_step method registration + SSOT-derived dispatch.

The cheapest, highest-yield integration guard (test plan P0 #1): a half-wired
method (in the SSOT tuple but not resolvable through the backends, or vice-versa)
is the most common breakage when adding a training method. After the §8.2 dedup,
both backends derive their method list from the single TRAINING_METHODS SSOT, so
these tests pin BOTH that ace_step is in the SSOT AND that it resolves through the
backend dispatch — and that the two can't drift apart.
"""
from __future__ import annotations

from pathlib import Path

from shared.utilities.paths import TRAINING_METHODS

# get_available_methods() derives purely from the TRAINING_METHODS SSOT and does
# not touch the filesystem, so any repo_root suffices for these dispatch checks.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ace_step_in_training_methods_ssot():
    """ace_step is registered in the single source of truth."""
    assert "ace_step" in TRAINING_METHODS


def test_rtx_backend_resolves_ace_step_from_ssot():
    """The local (rtx) backend exposes ace_step, derived from TRAINING_METHODS."""
    from tuner.backends.training.rtx_backend import RTXBackend

    methods = RTXBackend(_REPO_ROOT).get_available_methods()
    assert "ace_step" in methods
    # Derived-from-SSOT: the backend list must EQUAL the SSOT (no hand-maintained
    # literal that could drift). This is the dedup invariant (§8.2).
    assert set(methods) == set(TRAINING_METHODS)


def test_cloud_backend_resolves_ace_step_from_ssot():
    """The cloud (HF Jobs) backend exposes ace_step, derived from TRAINING_METHODS."""
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend

    methods = HFJobsBackend(_REPO_ROOT).get_available_methods()
    assert "ace_step" in methods
    assert set(methods) == set(TRAINING_METHODS)


def test_backends_agree_on_method_list():
    """Local and cloud backends expose the SAME method set (both SSOT-derived)."""
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.backends.training.rtx_backend import RTXBackend

    assert set(RTXBackend(_REPO_ROOT).get_available_methods()) == set(
        HFJobsBackend(_REPO_ROOT).get_available_methods()
    )
