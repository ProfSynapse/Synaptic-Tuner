"""P1 (GPU-gated) real-pipeline tests for the ACE-STEP v1.5 music pipeline.

These are the @pytest.mark.gpu layer of the test plan's 3-layer GPU-gating split:
they exercise the REAL preprocess / train / generate steps end-to-end and are
auto-skipped when no CUDA device is present (see conftest.pytest_collection_
modifyitems). They run on a local 3090/4090 or an HF-Jobs GPU box, NOT in CPU CI.

Status: these are SCAFFOLDS. They depend on artifacts other coders own and are
landing concurrently:
  - backend-coder-2: Trainers/ace_step/ (preprocess + train_ace_step.py wrapper)
  - contract §7: the cli.py generation flag spellings (pending byte-confirm)
Each test SKIPS with an explicit reason until its dependency is importable/runnable,
so the file is green-collectable today and "lights up" as the pipeline lands —
rather than hard-failing on a not-yet-built seam. The structural assertions
(NEVER bit-exact) match the audio verifier's Phase-1 contract.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "scratch" / "fixtures" / "ace_step"


def _module_available(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def test_preprocess_produces_pt_cache(tmp_path):
    """Real DCAE preprocess of a tiny corpus → self-describing .pt tensors.

    ⚠️ Confirm the minimum clip duration with backend-coder-2: the DCAE f8c8
    codec needs ≥ ~11.88 s for a valid 128-frame latent (contract §4 / plan
    fixture caveat); the 0.5 s smoke fixtures are NOT long enough for a real
    preprocess and must be replaced with ≥12 s clips for this P1 test.
    """
    if not _module_available("Trainers.ace_step"):
        pytest.skip("Trainers/ace_step/ not importable yet (backend-coder-2)")
    pytest.skip(
        "preprocess P1 scaffold: needs ≥12 s real audio fixtures + the preprocess "
        "entry pinned by backend-coder-2 (contract §4). Wire once landed."
    )


def test_train_produces_loadable_adapter(tmp_path):
    """Real LoKr train (~5 min on GPU) → a loadable adapter on disk."""
    if not _module_available("Trainers.ace_step"):
        pytest.skip("Trainers/ace_step/ not importable yet (backend-coder-2)")
    pytest.skip(
        "train P1 scaffold: needs the train_ace_step.py wrapper + a preprocessed "
        ".pt cache from the preprocess step. Wire once both land."
    )


def test_generate_then_verify_audio_smoke(tmp_path):
    """Real generate (fixed seed) → rendered WAV → audio verifier passes.

    This is the full generate→eval E2E: ACE-STEP cli.py renders a clip from a
    fixed-seed prompt, and the SAME structural verifier the P0 tests use scores
    it (loadable / 48 kHz / stereo / non-silent / duration). Structural-only —
    never bit-exact (diffusion is non-deterministic).
    """
    if not _module_available("Trainers.ace_step"):
        pytest.skip("Trainers/ace_step/ not importable yet (backend-coder-2)")
    pytest.skip(
        "generate P1 scaffold: needs the cli.py generation entry + byte-confirmed "
        "flag spellings (contract §7, pending preparer-acestep Q5). Wire once pinned."
    )
