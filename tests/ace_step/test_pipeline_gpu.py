"""P1 (GPU-gated) real-pipeline tests for the ACE-STEP v1.5 music pipeline.

These are the @pytest.mark.gpu layer of the test plan's 3-layer GPU-gating split:
they exercise the REAL preprocess / train / generate steps end-to-end and are
auto-skipped when no CUDA device is present (see conftest.pytest_collection_
modifyitems). They run on a local 3090/4090 or an HF-Jobs GPU box, NOT in CPU CI.

Status: these are SCAFFOLDS, and they split into TWO honest deferral classes —
not one. The dependency code (Trainers/ace_step/ preprocess + train_ace_step.py
wrapper) has LANDED (#28); what these tests still lack differs by test:

  - preprocess / train: GPU-EXECUTION-deferred. The wrapper + argv builders exist
    and are CPU-tested (see test_argv_contract.py); what's missing is a real CUDA
    device + the ACE-STEP model + ≥12 s real audio fixtures — none of which exist
    in CPU CI. These "light up" on a local 3090/4090 or HF-Jobs GPU box.
  - generate: ENTRY-not-built. ACE-STEP cli.py generation is genuinely deferred
    (no generate entry in the wrapper yet, contract §7) — this skips because the
    code does not exist, NOT merely because there's no GPU.

Each test SKIPS with the reason that actually applies, so the file is green-
collectable today and "lights up" as each seam lands. The structural assertions
(NEVER bit-exact) match the audio verifier's Phase-1 contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "scratch" / "fixtures" / "ace_step"


def test_preprocess_produces_pt_cache(tmp_path):
    """Real DCAE preprocess of a tiny corpus → self-describing .pt tensors.

    ⚠️ Confirm the minimum clip duration with backend-coder-2: the DCAE f8c8
    codec needs ≥ ~11.88 s for a valid 128-frame latent (contract §4 / plan
    fixture caveat); the 0.5 s smoke fixtures are NOT long enough for a real
    preprocess and must be replaced with ≥12 s clips for this P1 test.
    """
    pytest.skip(
        "GPU-execution-deferred: the preprocess wrapper + argv builder exist (#28, "
        "argv pinned by test_argv_contract.py); this needs a real CUDA device + the "
        "ACE-STEP model + ≥12 s real audio fixtures (absent in CPU CI). Wire the real "
        "preprocess run here on a GPU box."
    )


def test_train_produces_loadable_adapter(tmp_path):
    """Real LoKr train (~5 min on GPU) → a loadable adapter on disk."""
    pytest.skip(
        "GPU-execution-deferred: the train_ace_step.py wrapper + argv builder exist "
        "(#28, argv pinned by test_argv_contract.py); this needs a real CUDA device + "
        "a preprocessed .pt cache from the preprocess step (absent in CPU CI). Wire the "
        "real LoKr train here on a GPU box."
    )


def test_generate_then_verify_audio_smoke(tmp_path):
    """Real generate (fixed seed) → rendered WAV → audio verifier passes.

    This is the full generate→eval E2E: ACE-STEP cli.py renders a clip from a
    fixed-seed prompt, and the SAME structural verifier the P0 tests use scores
    it (loadable / 48 kHz / stereo / non-silent / duration). Structural-only —
    never bit-exact (diffusion is non-deterministic).
    """
    pytest.skip(
        "entry-not-built (§7): ACE-STEP cli.py generation is deferred — there is no "
        "generate entry in the wrapper yet (this is a missing-code deferral, NOT a "
        "GPU-only one). Wire once the generation handler + byte-confirmed cli.py flag "
        "spellings land."
    )
