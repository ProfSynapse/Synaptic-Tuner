"""Source-level smoke tests for train_dpo.py.

These assert on the trainer source text (no import), mirroring
tests/trainers/sft/test_train_sft_source.py, so they verify the DPO API choices
without loading trl/unsloth/torch.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DPO = REPO_ROOT / "Trainers" / "dpo" / "train_dpo.py"


def _source() -> str:
    return TRAIN_DPO.read_text(encoding="utf-8")


def test_uses_trl_dpo_api_not_kto():
    source = _source()
    assert "from trl import DPOConfig, DPOTrainer" in source
    assert "DPOConfig(" in source
    assert "DPOTrainer(" in source
    # No KTO API is actually invoked (the docstring may *mention* KTO to explain
    # the deltas, so check for instantiation/import forms, not the bare word).
    assert "KTOConfig(" not in source
    assert "KTOTrainer(" not in source
    assert "KTOSTrainer(" not in source
    assert "from trl import KTOConfig" not in source


def test_dpo_config_drops_kto_only_fields():
    source = _source()
    assert "loss_type=config.training.loss_type" in source
    assert "desirable_weight" not in source
    assert "undesirable_weight" not in source
    assert "use_kto_s" not in source


def test_reuses_reference_model_seam():
    source = _source()
    assert "create_reference_model" in source
    assert "USE_EXPLICIT_REF_MODEL" in source


def test_no_interleaving_for_paired_dpo():
    source = _source()
    # KTO's interleave step must NOT be present (DPO is paired, not binary).
    assert "interleave_dataset" not in source
    assert "validate_dpo_dataset" in source


def test_dry_run_exits_before_model_load():
    source = _source()
    # The dry-run guard must precede the heavy trl/unsloth imports + model load.
    dry_idx = source.find("if args.dry_run:")
    model_load_idx = source.find("load_model_and_tokenizer(")
    trl_import_idx = source.find("from trl import DPOConfig, DPOTrainer")
    assert dry_idx != -1 and model_load_idx != -1 and trl_import_idx != -1
    assert dry_idx < trl_import_idx
    assert dry_idx < model_load_idx


def test_uses_modern_trl_processing_class_kwarg():
    source = _source()
    # Cloud recipes pin modern trl (>=0.22) where DPOTrainer takes processing_class.
    assert "processing_class=tokenizer" in source
