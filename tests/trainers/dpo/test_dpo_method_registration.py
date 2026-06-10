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


def test_named_gate_sites_source_contains_dpo():
    sites = {
        "tuner/cli/parser.py": '"sft", "kto", "grpo", "dpo"',
        "tuner/backends/training/cloud/hf_jobs_backend.py": '["sft", "kto", "grpo", "dpo"]',
        "tuner/backends/training/rtx_backend.py": '["sft", "kto", "grpo", "dpo"]',
    }
    for rel, needle in sites.items():
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert needle in source, f"{rel} missing dpo registration ({needle!r})"


# ---- Lifecycle-parity iteration sites (eval discovery, model discovery, handlers) ----

def test_lifecycle_iteration_sites_include_dpo():
    sites = [
        "tuner/backends/evaluation/unsloth_backend.py",
        "tuner/backends/evaluation/mlc_backend.py",
        "tuner/backends/evaluation/llamacpp_backend.py",
        "tuner/discovery/base_models.py",
        "tuner/handlers/merge_handler.py",
        "tuner/handlers/doctor_handler.py",
        "tuner/handlers/train_handler.py",
    ]
    for rel in sites:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert '"dpo"' in source or "'dpo'" in source, f"{rel} missing 'dpo' in its method enumeration"


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
