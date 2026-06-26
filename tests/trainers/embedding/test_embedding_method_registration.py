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

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _validate_method_issues(experiment_spec_cls, method: str) -> list[str]:
    """Build a minimal-but-otherwise-valid ExperimentSpec with ``method`` and
    return validate()'s issues. Every other required field is satisfied, so the
    ONLY issue that can mention the method is the SSOT allowlist gate — letting
    the caller assert presence/absence of 'unsupported method' cleanly."""
    from shared.experiment_tracking.experiment_spec import (
        DatasetSpec,
        TrainingStageSpec,
    )

    spec = experiment_spec_cls(
        name="t",
        provider="local",
        method=method,
        dataset=DatasetSpec(source="hf", file="d.jsonl"),
        training=TrainingStageSpec(model_name="m"),
    )
    return spec.validate()


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


def test_named_gate_sites_derive_embedding_from_ssot():
    """The named train-time gates now DERIVE from TRAINING_METHODS (SSOT), so
    embedding resolves through each without a hardcoded literal.

    SSOT-DERIVED SHAPE (CONTRACTS §5.1.1, backend-coder's #27 dedup): the old
    5-element string literals were removed in favour of ``list(TRAINING_METHODS)``
    / ``SUPPORTED_METHODS = TRAINING_METHODS``. We assert FUNCTIONALLY that each
    gate's method set equals the SSOT (so embedding — and every method — resolves)
    rather than grepping a frozen literal. This is regression-loud the SSOT way:
    drop embedding from TRAINING_METHODS and every gate fails at once; add a 7th
    method and NOTHING here needs editing.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from shared.utilities.paths import TRAINING_METHODS
    from tuner.backends.training.cloud import base_cloud
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.backends.training.rtx_backend import RTXBackend

    ssot = set(TRAINING_METHODS)
    assert "embedding" in ssot

    # rtx + cloud backend method lists are SSOT-derived.
    assert set(RTXBackend(REPO_ROOT).get_available_methods()) == ssot
    assert set(HFJobsBackend(REPO_ROOT).get_available_methods()) == ssot
    # base_cloud.SUPPORTED_METHODS is the SSOT tuple itself.
    assert set(base_cloud.SUPPORTED_METHODS) == ssot
    # The CLI --method choices derive from the SSOT (parse a known method).
    from tuner.cli.parser import create_parser

    args = create_parser().parse_args(["train", "--method", "embedding"])
    assert args.method == "embedding"


def test_experiment_spec_allowlist_derives_embedding_from_ssot():
    """The functional method-allowlist (CONTRACTS §5.1.2) is SSOT-derived: it
    validates against ``set(TRAINING_METHODS)``, so embedding is accepted and a
    bogus method is rejected. Asserts BEHAVIOUR (validate()), not a source regex."""
    sys.path.insert(0, str(REPO_ROOT))
    from shared.experiment_tracking.experiment_spec import ExperimentSpec
    from shared.utilities.paths import TRAINING_METHODS

    assert "embedding" in set(TRAINING_METHODS)
    # A valid method produces no "unsupported method" issue; a bogus one does.
    ok = _validate_method_issues(ExperimentSpec, "embedding")
    bad = _validate_method_issues(ExperimentSpec, "not_a_method")
    assert not any("unsupported method" in i for i in ok), ok
    assert any("unsupported method" in i for i in bad), bad


# ---- Eval-backend serving tuples: embedding EXCLUDED (the intentional asymmetry) ----

EVAL_BACKENDS = [
    "tuner/backends/evaluation/unsloth_backend.py",
    "tuner/backends/evaluation/mlc_backend.py",
    "tuner/backends/evaluation/llamacpp_backend.py",
]


def test_eval_backend_tuples_exclude_embedding_and_ace_step():
    """The 3 eval-backend discovery tuples iterate ("sft","kto","grpo","dpo")
    WITHOUT embedding OR ace_step (CONTRACTS §5.3). Lock BOTH exclusions.

    These serving-discovery tuples are deliberately NOT SSOT-derived: they enumerate
    only the causal-LM methods that quantize/serve through unsloth/mlc/llamacpp.
    embedding (retrieval verifier) and ace_step (diffusion DiT audio) are scored by
    their OWN verifiers and never served here, so they must stay absent. We assert
    the 4-method literal is present AND that neither method was appended — robust to
    the word appearing elsewhere (e.g. the explanatory NOTE comment).
    """
    four_method = 'for method in ("sft", "kto", "grpo", "dpo"):'
    forbidden_loops = [
        'for method in ("sft", "kto", "grpo", "dpo", "embedding"):',
        'for method in ("sft", "kto", "grpo", "dpo", "ace_step"):',
        'for method in ("sft", "kto", "grpo", "dpo", "embedding", "ace_step"):',
    ]
    for rel in EVAL_BACKENDS:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert four_method in source, (
            f"{rel} missing the 4-method serving-discovery loop (CONTRACTS §5.3)"
        )
        for bad in forbidden_loops:
            assert bad not in source, (
                f"{rel} ADDED a non-servable method to the serving-discovery loop "
                f"({bad!r}) — breaks the intentional R3 exclusion (CONTRACTS §5.3). "
                f"embedding (retrieval) and ace_step (diffusion audio) are scored by "
                f"their own verifiers, never served through this causal-LM backend."
            )


def test_eval_backend_exclusion_is_documented():
    """Each eval backend carries the CONTRACTS §5.3 NOTE so a future reader sees
    the exclusion is deliberate, not an omission."""
    for rel in EVAL_BACKENDS:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "embedding" in source and "EXCLUDED" in source, (
            f"{rel} missing the intentional-exclusion NOTE (CONTRACTS §5.3)"
        )


# ---- ace_step: ADDED at train-time gates, EXCLUDED from serving (same asymmetry) ----

def test_ace_step_resolves_through_train_time_gates():
    """ace_step (the 6th method, ACE-STEP v1.5 music) resolves through every
    SSOT-derived train-time gate — proving the SSOT-derivation gives a new method
    full train-time registration with ZERO per-gate edits (the dedup's payoff)."""
    sys.path.insert(0, str(REPO_ROOT))
    from shared.experiment_tracking.experiment_spec import ExperimentSpec
    from shared.utilities.paths import TRAINING_METHODS
    from tuner.backends.training.cloud import base_cloud
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.backends.training.rtx_backend import RTXBackend
    from tuner.cli.parser import create_parser

    assert "ace_step" in set(TRAINING_METHODS)
    assert "ace_step" in RTXBackend(REPO_ROOT).get_available_methods()
    assert "ace_step" in HFJobsBackend(REPO_ROOT).get_available_methods()
    assert "ace_step" in base_cloud.SUPPORTED_METHODS
    assert create_parser().parse_args(["train", "--method", "ace_step"]).method == "ace_step"
    # experiment_spec allowlist accepts ace_step (SSOT-derived).
    assert not any(
        "unsupported method" in i for i in _validate_method_issues(ExperimentSpec, "ace_step")
    )
