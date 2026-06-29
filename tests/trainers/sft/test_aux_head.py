"""Unit + component smoke tests for the auxiliary scalar readout head.

Location: tests/trainers/sft/test_aux_head.py

Covers AuxHead shapes/dtype/range for linear & mlp, the token-position reduction,
the proper-scoring loss, the sidecar save/load roundtrip, the inference hook, and
a component-level training smoke (only the head's params move; loss decreases).
The full Trainer.train() integration (and the AUROC-vs-oracle bar) is TEST-phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

import aux_head as aux_head_mod  # noqa: E402
from aux_head import (  # noqa: E402
    AuxHead,
    compute_aux_head_loss,
    infer_aux_scalar,
    load_aux_head,
    reduce_hidden_states,
    save_aux_head,
)


# ---------------------------------------------------------------------------
# AuxHead module: shapes, dtype, [0,1] range, linear & mlp
# ---------------------------------------------------------------------------

def test_linear_head_emits_batch_scalar_in_unit_interval():
    head = AuxHead(input_dim=16, head_type="linear")
    out = head(torch.randn(5, 16))
    assert out.shape == (5,)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_mlp_head_emits_batch_scalar_in_unit_interval():
    head = AuxHead(input_dim=16, head_type="mlp", hidden_dims=(8, 4))
    out = head(torch.randn(3, 16))
    assert out.shape == (3,)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_head_casts_pulled_hidden_state_to_head_dtype():
    head = AuxHead(input_dim=8, head_type="linear")  # fp32 params by default
    out = head(torch.randn(4, 8, dtype=torch.float16))
    assert out.dtype == torch.float32
    assert out.shape == (4,)


def test_identity_activation_returns_raw_logits():
    head = AuxHead(input_dim=8, head_type="linear", out_activation="identity")
    out = head(torch.randn(4, 8) * 100.0)
    # Without sigmoid, values are not constrained to [0, 1].
    assert out.shape == (4,)
    assert (out.min() < 0.0) or (out.max() > 1.0)


def test_invalid_head_type_and_activation_raise():
    with pytest.raises(ValueError):
        AuxHead(input_dim=8, head_type="quadratic")
    with pytest.raises(ValueError):
        AuxHead(input_dim=8, out_activation="softmax")
    with pytest.raises(ValueError):
        AuxHead(input_dim=0)


# ---------------------------------------------------------------------------
# reduce_hidden_states: last / mean / int, right-pad aware
# ---------------------------------------------------------------------------

def test_reduce_last_picks_last_non_pad_token_under_right_padding():
    # batch=2, seq=4, hidden=3. Row 0 has 2 real tokens, row 1 has 4.
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    reduced = reduce_hidden_states(hidden, attention_mask, "last")
    assert reduced.shape == (2, 3)
    # Row 0: last real token is index 1; Row 1: index 3.
    assert torch.equal(reduced[0], hidden[0, 1])
    assert torch.equal(reduced[1], hidden[1, 3])


def test_reduce_mean_is_masked_mean_over_real_tokens():
    hidden = torch.ones(1, 4, 2)
    hidden[0, 2:] = 5.0  # padded positions carry junk
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    reduced = reduce_hidden_states(hidden, attention_mask, "mean")
    # Only the two real (value=1.0) tokens count.
    assert torch.allclose(reduced, torch.ones(1, 2))


def test_reduce_int_index_selects_that_position():
    hidden = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    attention_mask = torch.ones(1, 4)
    reduced = reduce_hidden_states(hidden, attention_mask, 0)
    assert torch.equal(reduced[0], hidden[0, 0])


def test_reduce_rejects_bad_inputs():
    with pytest.raises(ValueError):
        reduce_hidden_states(torch.randn(2, 3), torch.ones(2, 3), "last")  # not 3D
    with pytest.raises(ValueError):
        reduce_hidden_states(torch.randn(1, 4, 2), torch.ones(1, 4), "median")
    with pytest.raises(ValueError):
        reduce_hidden_states(torch.randn(1, 4, 2), torch.ones(1, 4), 99)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_bce_and_brier_losses_are_nonnegative_and_match_known_values():
    pred = torch.tensor([0.5, 0.5])
    target = torch.tensor([1.0, 0.0])
    brier = compute_aux_head_loss(pred, target, "brier")
    assert torch.allclose(brier, torch.tensor(0.25))
    bce = compute_aux_head_loss(pred, target, "bce")
    assert bce.item() > 0.0


def test_loss_rejects_unknown_type():
    with pytest.raises(ValueError):
        compute_aux_head_loss(torch.tensor([0.5]), torch.tensor([1.0]), "hinge")


# ---------------------------------------------------------------------------
# Save / load sidecar roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_preserves_weights_and_config(tmp_path):
    head = AuxHead(input_dim=12, head_type="mlp", hidden_dims=(6,))
    save_aux_head(head, tmp_path, layer=35, token_position="last", loss="bce")

    assert (tmp_path / "aux_head.safetensors").exists()
    assert (tmp_path / "aux_head_config.json").exists()

    reloaded = load_aux_head(tmp_path)
    assert reloaded.input_dim == 12
    assert reloaded.head_type == "mlp"
    assert reloaded.hidden_dims == (6,)

    x = torch.randn(4, 12)
    assert torch.allclose(head(x), reloaded(x), atol=1e-6)

    resolved = aux_head_mod.read_aux_head_resolved_config(tmp_path)
    assert resolved["layer"] == 35
    assert resolved["token_position"] == "last"
    assert resolved["loss"] == "bce"


def test_load_missing_sidecar_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_aux_head(tmp_path)


# ---------------------------------------------------------------------------
# Inference hook
# ---------------------------------------------------------------------------

class _FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeBase(torch.nn.Module):
    """Minimal stand-in: emits per-layer hidden states from a learned embedding."""

    def __init__(self, vocab=20, hidden=8, layers=3):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.num_layers = layers

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.embed(input_ids)
        hidden_states = tuple(h * (i + 1) for i in range(self.num_layers))
        return _FakeOutputs(hidden_states)


def test_infer_aux_scalar_returns_per_row_scalar_in_unit_interval():
    base = _FakeBase(hidden=8)
    head = AuxHead(input_dim=8, head_type="linear")
    input_ids = torch.randint(0, 20, (3, 5))
    attention_mask = torch.ones(3, 5, dtype=torch.long)
    scores = infer_aux_scalar(
        base, head, input_ids=input_ids, attention_mask=attention_mask, layer=2, token_position="last"
    )
    assert scores.shape == (3,)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


# ---------------------------------------------------------------------------
# Component-level training smoke: only the head moves; loss decreases
# ---------------------------------------------------------------------------

def test_only_head_trains_and_loss_decreases():
    torch.manual_seed(0)
    # A "frozen base" stand-in: parameters that must NOT receive gradient.
    frozen_base = torch.nn.Linear(8, 8)
    for p in frozen_base.parameters():
        p.requires_grad = False

    head = AuxHead(input_dim=8, head_type="linear")
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=0.1)

    # Fixed separable batch.
    hidden = torch.randn(32, 8)
    target = (hidden.sum(dim=1) > 0).float()

    first_loss = None
    last_loss = None
    for _ in range(50):
        optimizer.zero_grad()
        with torch.no_grad():
            features = frozen_base(hidden)  # frozen base produces features
        pred = head(features)
        loss = compute_aux_head_loss(pred, target, "bce")
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss  # the head learned
    # The frozen base never accumulated gradient.
    assert all(p.grad is None for p in frozen_base.parameters())
    assert all(not p.requires_grad for p in frozen_base.parameters())
