"""Registration-sweep test: 'dpo' must be registered at every method-enumeration
site discovered in the WS-3 grep sweep.

The design doc named 4 sites; the actual surface is ~11, anchored on
shared/utilities/paths.py:TRAINING_METHODS (which auto-derives the trainer-dir
and output-dir maps that the cloud backends use to dispatch dpo -> Trainers/dpo).
This test pins the full set so a future edit that drops 'dpo' from any site
fails loudly. Source-scanning sites import-light; logic sites import the module.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---- Central registration: paths.py TRAINING_METHODS (the auto-derive anchor) ----

def test_paths_training_methods_includes_dpo():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from shared.utilities import paths

    assert "dpo" in paths.TRAINING_METHODS
    # Auto-derived maps must resolve dpo -> Trainers/dpo and dpo_output.
    assert paths.CANONICAL_TRAINER_DIRS["dpo"] == "dpo"
    assert paths.CANONICAL_OUTPUT_DIRS["dpo"] == "dpo_output"
    assert paths.get_canonical_trainer_dir_name("dpo") == "dpo"


# ---- Cloud / CLI gate sites (the doc's named 4) ----

def test_base_cloud_supported_methods_includes_dpo():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from tuner.backends.training.cloud import base_cloud

    assert "dpo" in base_cloud.SUPPORTED_METHODS


def test_named_gate_sites_derive_dpo_from_ssot():
    """The named train-time gates DERIVE from TRAINING_METHODS (SSOT) after
    backend-coder's #27 dedup — the hardcoded method-list literals were removed in
    favour of ``list(TRAINING_METHODS)`` / ``SUPPORTED_METHODS = TRAINING_METHODS``.

    So we assert FUNCTIONALLY that dpo resolves through each gate (its method set
    equals the SSOT) rather than grepping a now-deleted literal. Regression-loud
    the SSOT way: drop dpo from TRAINING_METHODS and every gate fails at once; add
    a new method and NOTHING here needs editing.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from shared.utilities.paths import TRAINING_METHODS
    from tuner.backends.training.cloud import base_cloud
    from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
    from tuner.backends.training.rtx_backend import RTXBackend
    from tuner.cli.parser import create_parser

    ssot = set(TRAINING_METHODS)
    assert "dpo" in ssot
    assert set(RTXBackend(REPO_ROOT).get_available_methods()) == ssot
    assert set(HFJobsBackend(REPO_ROOT).get_available_methods()) == ssot
    assert set(base_cloud.SUPPORTED_METHODS) == ssot
    assert create_parser().parse_args(["train", "--method", "dpo"]).method == "dpo"


# ---- Lifecycle-parity iteration sites (eval discovery, model discovery, handlers) ----

def test_lifecycle_iteration_sites_include_dpo():
    """Lifecycle-iteration sites must still resolve dpo.

    These sites split into two classes as the SSOT-derive dedup rolls out:

    - LITERAL sites still enumerate methods explicitly (the eval-backend serving
      tuples intentionally lock the 4-method causal-LM set per CONTRACTS §5.3;
      merge_handler.py is pending its own dedup), so they keep the literal-grep.
    - SSOT-DERIVED sites iterate ``TRAINING_METHODS`` and carry NO "dpo" literal —
      asserting the literal would be a false regression. train_handler.py (#27)
      led; base_models.py + doctor_handler.py joined it under F-1 (backend-coder
      #47, which made ace_step/embedding discoverable). For these we assert the
      SSOT-derivation marker IS present AND the "dpo" literal is ABSENT (a
      reintroduced literal signals a regression away from the dedup), while dpo
      still resolves because TRAINING_METHODS contains it.
    """
    literal_sites = [
        "tuner/backends/evaluation/unsloth_backend.py",
        "tuner/backends/evaluation/mlc_backend.py",
        "tuner/backends/evaluation/llamacpp_backend.py",
        "tuner/handlers/merge_handler.py",
    ]
    for rel in literal_sites:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert '"dpo"' in source or "'dpo'" in source, f"{rel} missing 'dpo' in its method enumeration"

    # Sites converted to SSOT-derive their enumeration from TRAINING_METHODS: they
    # carry no "dpo" literal but resolve dpo by iterating the SSOT. Assert the
    # derivation, not the literal.
    ssot_derived_sites = [
        "tuner/handlers/train_handler.py",      # #27
        "tuner/discovery/base_models.py",       # F-1 (#47)
        "tuner/handlers/doctor_handler.py",     # F-1 (#47)
    ]
    for rel in ssot_derived_sites:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "TRAINING_METHODS" in source, (
            f"{rel} should SSOT-derive its method enumeration from TRAINING_METHODS"
        )
        assert '"dpo"' not in source and "'dpo'" not in source, (
            f"{rel} is SSOT-derived; a reintroduced 'dpo' literal signals a regression "
            f"away from the dedup (CONTRACTS §5.1.1)"
        )

    # dpo functionally resolves through every SSOT-derived site: it is in the SSOT.
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from shared.utilities.paths import TRAINING_METHODS

    assert "dpo" in TRAINING_METHODS


# ---- No method-tuple enumeration site left without dpo ----

def test_no_three_method_tuple_left_unregistered():
    """Fail if any (sft, kto, grpo) tuple WITHOUT dpo remains under tuner/ or shared/.

    Catches a missed enumeration site. Skips comments/docstrings by checking the
    canonical literal forms only.
    """
    stale_forms = [
        '("sft", "kto", "grpo")',
        '("sft","kto","grpo")',
        '["sft", "kto", "grpo"]',
        "['sft', 'kto', 'grpo']",
    ]
    offenders = []
    for root in ("tuner", "shared"):
        for py in (REPO_ROOT / root).rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for form in stale_forms:
                if form in text:
                    offenders.append(f"{py.relative_to(REPO_ROOT)}: {form}")
    assert not offenders, "Stale 3-method tuples (missing dpo):\n" + "\n".join(offenders)
