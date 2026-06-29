"""Wire-in + trainer-override tests for the aux_head feature.

Location: tests/trainers/sft/test_aux_head_trainer_source.py

train_sft.py imports unsloth at module load, so its wiring is asserted at the
source level (the repo's established pattern, cf. test_train_sft_source.py).
AuxHeadTrainer itself is unsloth-free, so its overrides are asserted by import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = ROOT / "Trainers" / "sft"


# ---------------------------------------------------------------------------
# Source-level: train_sft.py wiring (byte-identical-off + two-hop + sidecar)
# ---------------------------------------------------------------------------

def _train_sft_source() -> str:
    return (SFT_DIR / "train_sft.py").read_text(encoding="utf-8")


def test_collator_stacks_aux_target_only_when_present():
    source = _train_sft_source()
    # Guarded so the feature-off batch is byte-identical.
    assert 'if "aux_target" in features[0]:' in source
    assert '"aux_target"' in source


def test_dataset_call_threads_aux_target_field():
    source = _train_sft_source()
    assert "aux_target_field=aux_target_field" in source
    # Only set when enabled ⇒ None on the off-path.
    assert "aux_target_field = aux_head_cfg.target_field if aux_head_enabled else None" in source


def test_trainer_swap_is_gated_on_enabled():
    source = _train_sft_source()
    assert "if aux_head_enabled:" in source
    assert "AuxHeadTrainer(" in source
    # Off-path still constructs the stock Trainer unchanged.
    assert "trainer = Trainer(**trainer_kwargs)" in source


def test_sidecar_save_runs_only_when_head_present():
    source = _train_sft_source()
    assert "if aux_head_module is not None:" in source
    assert "save_aux_head(" in source


def test_hidden_size_resolver_has_peft_fallback():
    source = _train_sft_source()
    assert "def _resolve_hidden_size(model)" in source
    assert "base_model" in source


# ---------------------------------------------------------------------------
# Import-level: AuxHeadTrainer overrides (transformers, no unsloth)
# ---------------------------------------------------------------------------

def test_aux_head_trainer_overrides_present():
    pytest.importorskip("transformers")
    sys.path.insert(0, str(SFT_DIR / "src"))
    import aux_head_trainer  # noqa: E402
    from transformers import Trainer

    cls = aux_head_trainer.AuxHeadTrainer
    assert issubclass(cls, Trainer)
    # Real overrides (defined on the subclass, not merely inherited).
    assert "compute_loss" in cls.__dict__
    assert "create_optimizer" in cls.__dict__
    assert "_freeze_base_keep_head" in cls.__dict__


def test_aux_head_trainer_source_marks_phase_b_seam_and_no_lm_term():
    source = (SFT_DIR / "src" / "aux_head_trainer.py").read_text(encoding="utf-8")
    # Phase A: head loss is the entire loss.
    assert "loss = head_loss" in source
    # Phase B seam left as a one-line comment, NOT enabled.
    assert "lm_loss_weight" in source
    assert "output_hidden_states=True" in source
