"""Registration-sweep sibling test for the ``embedding`` method (mirror of
``tests/trainers/dpo/test_dpo_method_registration.py``).

CONTEXT (CONTRACTS §5): the embedding method-wiring was a deliberate SPLIT —
``embedding`` is ADDED at the 5 train-time gates but EXCLUDED from the 3
eval-backend serving-discovery tuples (CONTRACTS §5.3), because embedding models
are scored via the retrieval verifier and never quantized/served through the
llamacpp/mlc/unsloth backends. This test makes that R3 exclusion DURABLE: it
asserts embedding is registered where it must be, AND explicitly that it is NOT
present in the 3 eval-backend tuples — so a future "consistency fix" that adds
embedding there fails loudly with this test pointing at the design rationale.

The two halves are the mechanism that locks the asymmetry:
- train-time gates: presence assertions (embedding must be selectable/dispatchable)
- eval-backend tuples: ABSENCE assertions on the iterated literal (the exclusion
  is intentional, comment-documented in each backend at CONTRACTS §5.3)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---- Central registration: paths.py TRAINING_METHODS (the auto-derive anchor) ----

def test_paths_training_methods_includes_embedding():
    sys.path.insert(0, str(REPO_ROOT))
    from shared.utilities import paths

    assert "embedding" in paths.TRAINING_METHODS
    # Auto-derived maps resolve embedding -> Trainers/embedding and embedding_output.
    assert paths.CANONICAL_TRAINER_DIRS["embedding"] == "embedding"
    assert paths.CANONICAL_OUTPUT_DIRS["embedding"] == "embedding_output"
    assert paths.get_canonical_trainer_dir_name("embedding") == "embedding"


# ---- Cloud / CLI gate sites: embedding ADDED (the 5 train-time gates) ----

def test_base_cloud_supported_methods_includes_embedding():
    sys.path.insert(0, str(REPO_ROOT))
    from tuner.backends.training.cloud import base_cloud

    assert "embedding" in base_cloud.SUPPORTED_METHODS


def test_named_gate_sites_source_contains_embedding():
    """The 3 named train-time gates carry the 5-element literal including embedding.

    These are the same needles the dpo sweep updated same-commit (CONTRACTS
    §5.1.1); asserting the 5-element form keeps the sweep loud if a future edit
    drops embedding from any named gate.
    """
    sites = {
        "tuner/cli/parser.py": '"sft", "kto", "grpo", "dpo", "embedding"',
        "tuner/backends/training/cloud/hf_jobs_backend.py": '["sft", "kto", "grpo", "dpo", "embedding"]',
        "tuner/backends/training/rtx_backend.py": '["sft", "kto", "grpo", "dpo", "embedding"]',
    }
    for rel, needle in sites.items():
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert needle in source, f"{rel} missing embedding registration ({needle!r})"


def test_experiment_spec_allowlist_includes_embedding():
    """The functional method-allowlist (CONTRACTS §5.1.2, 6th gate, WU-B) must
    accept embedding, else ExperimentSpec.validate() rejects embedding runs."""
    source = (REPO_ROOT / "shared/experiment_tracking/experiment_spec.py").read_text(
        encoding="utf-8"
    )
    # The allowlist set literal must contain "embedding".
    assert re.search(
        r'\{\s*"sft",\s*"kto",\s*"grpo",\s*"dpo",\s*"embedding"\s*\}', source
    ), "experiment_spec allowlist missing 'embedding'"


# ---- Eval-backend serving tuples: embedding EXCLUDED (the intentional asymmetry) ----

def test_eval_backend_tuples_exclude_embedding():
    """The 3 eval-backend discovery tuples iterate ("sft","kto","grpo","dpo")
    WITHOUT embedding (CONTRACTS §5.3). Lock the exclusion as intentional.

    We assert the exact 4-method iterated literal is present AND that embedding
    is not appended to it. The literal-level check is robust to the file holding
    the word "embedding" elsewhere (e.g. in the explanatory NOTE comment).
    """
    eval_backends = [
        "tuner/backends/evaluation/unsloth_backend.py",
        "tuner/backends/evaluation/mlc_backend.py",
        "tuner/backends/evaluation/llamacpp_backend.py",
    ]
    four_method = 'for method in ("sft", "kto", "grpo", "dpo"):'
    five_method = 'for method in ("sft", "kto", "grpo", "dpo", "embedding"):'
    for rel in eval_backends:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert four_method in source, (
            f"{rel} missing the 4-method serving-discovery loop (CONTRACTS §5.3)"
        )
        assert five_method not in source, (
            f"{rel} ADDED embedding to the serving-discovery loop — this breaks "
            f"the intentional R3 exclusion (CONTRACTS §5.3). Embedding models are "
            f"scored via the retrieval verifier, never served through this backend."
        )


def test_eval_backend_exclusion_is_documented():
    """Each eval backend carries the CONTRACTS §5.3 NOTE so a future reader sees
    the exclusion is deliberate, not an omission."""
    eval_backends = [
        "tuner/backends/evaluation/unsloth_backend.py",
        "tuner/backends/evaluation/mlc_backend.py",
        "tuner/backends/evaluation/llamacpp_backend.py",
    ]
    for rel in eval_backends:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "embedding" in source and "EXCLUDED" in source, (
            f"{rel} missing the intentional-exclusion NOTE (CONTRACTS §5.3)"
        )
