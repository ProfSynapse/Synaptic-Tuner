"""Real ``Trainer.train()`` integration smoke for the aux_head feature.

Location: tests/trainers/sft/test_aux_head_integration.py

The other aux_head tests cover the module, the loss, the config, the
preprocessing hop, and assert the ``train_sft.py`` wiring at the source level
(that file imports unsloth at module load, so it cannot be imported here). This
file closes the remaining gap: it drives the **actual** ``transformers.Trainer``
training loop through ``AuxHeadTrainer`` on a tiny real ``LlamaForCausalLM``,
on CPU, with no unsloth — exercising the live ``compute_loss`` override,
``create_optimizer`` (head-only param group), base freezing, and the
save/reload/inference roundtrip end to end.

transformers-5.5 removed the ``no_cuda`` TrainingArguments kwarg; CPU training is
selected with ``use_cpu=True`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "configs"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors.torch")

from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments  # noqa: E402

from aux_head import AuxHead, infer_aux_scalar, load_aux_head, save_aux_head  # noqa: E402
from aux_head_trainer import AuxHeadTrainer  # noqa: E402
from config_loader import AuxHeadConfig  # noqa: E402


HIDDEN = 16
N_LAYERS = 3
VOCAB = 32
SEQ = 6
READ_LAYER = 2  # a mid hidden_states index (0 = embeddings, so 1..N_LAYERS are blocks)


def _tiny_causal_lm() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        max_position_embeddings=SEQ + 2,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


class _RowDataset(torch.utils.data.Dataset):
    """A handful of right-padded rows carrying a per-row ``aux_target``."""

    def __init__(self, n: int = 8):
        torch.manual_seed(1)
        self.rows = []
        for i in range(n):
            real = 3 + (i % 3)  # 3..5 real tokens, then right-pad to SEQ
            ids = [int(torch.randint(1, VOCAB, (1,))) for _ in range(real)] + [0] * (SEQ - real)
            mask = [1] * real + [0] * (SEQ - real)
            self.rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "labels": ids,
                    # Separable-ish soft target in [0, 1].
                    "aux_target": 1.0 if i % 2 == 0 else 0.0,
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _collate(features):
    """Minimal collator: stack the pre-padded rows into tensors.

    The production collator ``collate_prepared_sft_batch`` (which also pads) is
    asserted separately at the source level; importing it here would pull unsloth.
    The contract under test is the trainer, not the padding.
    """
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        "aux_target": torch.tensor([float(f["aux_target"]) for f in features], dtype=torch.float32),
    }


def _make_trainer(tmp_path, *, freeze_base=True, loss="bce", token_position="last", remove_unused_columns=False):
    model = _tiny_causal_lm()
    head = AuxHead(input_dim=HIDDEN, head_type="linear")
    cfg = AuxHeadConfig(
        enabled=True,
        layer=READ_LAYER,
        token_position=token_position,
        target_field="aux_target",
        loss=loss,
        head_type="linear",
        freeze_base=freeze_base,
        lm_loss_weight=0.0,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        use_cpu=True,  # transformers-5.5: replaces the removed no_cuda kwarg
        remove_unused_columns=remove_unused_columns,  # keep aux_target on the rows (mirrors train_sft)
        per_device_train_batch_size=4,
        max_steps=5,
        learning_rate=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = AuxHeadTrainer(
        model=model,
        args=args,
        data_collator=_collate,
        train_dataset=_RowDataset(),
        aux_head=head,
        aux_head_config=cfg,
    )
    return trainer, model, head


def test_construction_freezes_base_and_leaves_only_head_trainable(tmp_path):
    trainer, model, head = _make_trainer(tmp_path)
    base_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert base_trainable == 0  # Phase A: base fully frozen
    assert head_trainable == (HIDDEN + 1)  # linear head: weight + bias


def test_optimizer_covers_head_params_only(tmp_path):
    trainer, model, head = _make_trainer(tmp_path)
    optimizer = trainer.create_optimizer()
    opt_param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    assert opt_param_count == sum(p.numel() for p in head.parameters())


def test_real_trainer_train_runs_and_updates_only_the_head(tmp_path):
    trainer, model, head = _make_trainer(tmp_path)
    head_before = [p.detach().clone() for p in head.parameters()]
    base_before = [p.detach().clone() for p in model.parameters()]

    result = trainer.train()

    assert result.training_loss is not None
    assert result.training_loss == pytest.approx(result.training_loss)  # finite (not NaN)

    # The head moved (it trained); the frozen base did not.
    head_moved = any(not torch.equal(b, a) for b, a in zip(head_before, head.parameters()))
    base_moved = any(not torch.equal(b, a) for b, a in zip(base_before, model.parameters()))
    assert head_moved
    assert not base_moved


def test_brier_loss_path_also_trains(tmp_path):
    # The other proper-scoring loss must drive a real training step too.
    trainer, model, head = _make_trainer(tmp_path, loss="brier")
    head_before = [p.detach().clone() for p in head.parameters()]
    trainer.train()
    assert any(not torch.equal(b, a) for b, a in zip(head_before, head.parameters()))


def test_trained_head_saves_reloads_and_infers_in_unit_interval(tmp_path):
    trainer, model, head = _make_trainer(tmp_path)
    trainer.train()

    sidecar = tmp_path / "sidecar"
    save_aux_head(head, sidecar, layer=READ_LAYER, token_position="last", loss="bce")
    reloaded = load_aux_head(sidecar, base_model=model)

    input_ids = torch.randint(1, VOCAB, (3, SEQ))
    attention_mask = torch.ones(3, SEQ, dtype=torch.long)
    scores = infer_aux_scalar(
        model, reloaded, input_ids=input_ids, attention_mask=attention_mask,
        layer=READ_LAYER, token_position="last",
    )
    assert scores.shape == (3,)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_missing_aux_target_in_batch_fails_loud(tmp_path):
    # If the collator/plumbing ever drops aux_target, compute_loss must shout,
    # never silently train on a default.
    trainer, model, head = _make_trainer(tmp_path)
    batch = _collate([_RowDataset()[0], _RowDataset()[1]])
    batch.pop("aux_target")
    with pytest.raises(ValueError, match="aux_target"):
        trainer.compute_loss(model, batch)


def test_construction_rejects_remove_unused_columns_true(tmp_path):
    # The per-row aux_target column survives ONLY because remove_unused_columns is
    # False. Flipping it must fail loud at construction (naming the coupling), not
    # surface as an opaque "missing aux_target" error mid-training.
    with pytest.raises(ValueError, match="remove_unused_columns"):
        _make_trainer(tmp_path, remove_unused_columns=True)


def test_train_refuses_resume_from_checkpoint(tmp_path):
    # The head is sidecar-saved post-train and is NOT in HF per-step checkpoints,
    # so resume would reinitialize it while reloading stale optimizer state.
    # The guard must refuse loudly rather than train a corrupted head.
    trainer, model, head = _make_trainer(tmp_path)
    with pytest.raises(RuntimeError, match="resume_from_checkpoint"):
        trainer.train(resume_from_checkpoint=str(tmp_path / "out" / "checkpoint-1"))
    with pytest.raises(RuntimeError, match="resume_from_checkpoint"):
        trainer.train(resume_from_checkpoint=True)
