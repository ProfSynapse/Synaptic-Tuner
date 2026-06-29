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


def test_hidden_size_resolver_imported_from_aux_head():
    # The resolver moved into the unsloth-free aux_head module so its fallback
    # branches are unit-testable; train_sft.py now imports + calls it.
    source = _train_sft_source()
    assert "from src.aux_head import AuxHead, resolve_hidden_size" in source
    assert "input_dim = resolve_hidden_size(model)" in source
    # The inline definition is gone (logic lives in aux_head.py now).
    assert "def _resolve_hidden_size(model)" not in source


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
    # Resume guard (Phase A does not support resume_from_checkpoint).
    assert "train" in cls.__dict__


def test_aux_head_trainer_source_wires_live_joint_loss_seam():
    source = (SFT_DIR / "src" / "aux_head_trainer.py").read_text(encoding="utf-8")
    # Phase B: the joint-loss seam is now LIVE (it is no longer a disabled
    # comment). The discriminator is the exact weighted-sum line, which exists
    # ONLY when the seam is wired — guarding against the green-by-omission trap
    # where a bare ``lm_loss_weight`` token would pass even as a comment.
    assert "loss = outputs.loss + lm_loss_weight * head_loss" in source
    # Gated on the weight so the lm_loss_weight==0 path stays byte-identical.
    assert "if lm_loss_weight > 0:" in source
    # Phase A (lm_loss_weight == 0): the head loss is still the entire loss.
    assert "loss = head_loss" in source
    # Hidden states are enabled inside compute_loss (no global flag).
    assert "output_hidden_states=True" in source
