"""Tests for AuxHeadConfig loading + the Config field wiring.

Location: tests/trainers/sft/test_aux_head_config.py

Guards the silent-drop gotcha: the aux_head block is only honored because
AuxHeadConfig is a real dataclass field on Config (dict_to_dataclass drops
unknown keys). Also verifies absent ⇒ off and the loud layer-required check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "sft" / "configs"))

import config_loader  # noqa: E402
from config_loader import (  # noqa: E402
    AuxHeadConfig,
    Config,
    load_aux_head_config,
    validate_aux_head_coherence,
)


def test_absent_block_yields_disabled_default():
    cfg = load_aux_head_config({})
    assert isinstance(cfg, AuxHeadConfig)
    assert cfg.enabled is False
    assert cfg.layer is None
    assert cfg.target_field == "target"
    assert cfg.loss == "bce"
    assert cfg.head_type == "linear"
    assert cfg.freeze_base is True
    assert cfg.lm_loss_weight == 0.0
    assert cfg.input_norm == "none"


def test_enabled_block_is_parsed_fieldwise():
    # NOTE: out_activation is "sigmoid" here (not "identity") because the loader
    # now rejects the identity + probability-loss (brier/bce) combination as
    # incoherent — see test_rejects_identity_out_activation_with_probability_loss.
    # Identity is still a valid dataclass value for callers that consume raw
    # logits; only the loader refuses to pair it with a probability-scoring loss.
    cfg = load_aux_head_config(
        {
            "enabled": True,
            "layer": 35,
            "token_position": "mean",
            "target_field": "score",
            "loss": "brier",
            "head_type": "mlp",
            "hidden_dims": [64, 16],
            "out_activation": "sigmoid",
            "freeze_base": True,
            "lm_loss_weight": 0.0,
            "head_lr": 1e-3,
        }
    )
    assert cfg.enabled is True
    assert cfg.layer == 35
    assert cfg.token_position == "mean"
    assert cfg.target_field == "score"
    assert cfg.loss == "brier"
    assert cfg.head_type == "mlp"
    assert cfg.hidden_dims == [64, 16]
    assert cfg.out_activation == "sigmoid"
    assert cfg.head_lr == 1e-3


def test_phase_b_block_is_parsed_fieldwise():
    cfg = load_aux_head_config(
        {
            "enabled": True,
            "layer": 35,
            "token_position": "end_of_prompt",
            "input_norm": "layernorm",
            "freeze_base": False,
            "lm_loss_weight": 1.0,
        }
    )
    assert cfg.token_position == "end_of_prompt"
    assert cfg.input_norm == "layernorm"
    assert cfg.freeze_base is False
    assert cfg.lm_loss_weight == 1.0


def test_enabled_without_layer_fails_loud():
    with pytest.raises(ValueError, match="requires aux_head.layer"):
        load_aux_head_config({"enabled": True})


@pytest.mark.parametrize(
    "freeze_base, lm_loss_weight",
    [
        (False, 0.0),  # co-train base on head loss alone, no LM anchor
        (True, 1.0),   # weighted LM term with no gradient (base frozen)
    ],
)
def test_rejects_incoherent_freeze_base_lm_loss_weight(freeze_base, lm_loss_weight):
    with pytest.raises(ValueError, match="must agree on the phase"):
        load_aux_head_config(
            {
                "enabled": True,
                "layer": 35,
                "freeze_base": freeze_base,
                "lm_loss_weight": lm_loss_weight,
            }
        )


@pytest.mark.parametrize("loss", ["bce", "brier"])
def test_rejects_identity_out_activation_with_probability_loss(loss):
    with pytest.raises(ValueError, match="out_activation='identity' is incompatible"):
        load_aux_head_config(
            {
                "enabled": True,
                "layer": 35,
                "loss": loss,
                "out_activation": "identity",
            }
        )


def test_coherent_phase_configs_still_load():
    # Both valid phase combinations must continue to load (byte-identical valid
    # path): frozen-base head-only and unfrozen-base joint co-training.
    phase_a = load_aux_head_config(
        {"enabled": True, "layer": 35, "freeze_base": True, "lm_loss_weight": 0.0}
    )
    assert phase_a.freeze_base is True and phase_a.lm_loss_weight == 0.0
    phase_b = load_aux_head_config(
        {"enabled": True, "layer": 35, "freeze_base": False, "lm_loss_weight": 1.0}
    )
    assert phase_b.freeze_base is False and phase_b.lm_loss_weight == 1.0


# --- validate_aux_head_coherence: direct unit tests -------------------------
# The same guard is reused by both the YAML-load path (load_aux_head_config,
# exercised above) and the CLI-override path (train_sft.run). These tests pin
# the shared function directly so the CLI lane is covered without importing
# train_sft (which imports unsloth and cannot load in-process).


@pytest.mark.parametrize(
    "freeze_base, lm_loss_weight",
    [
        (False, 0.0),  # co-train base on head loss alone, no LM anchor
        (True, 1.0),   # weighted LM term with no gradient (base frozen)
    ],
)
def test_validate_coherence_rejects_incoherent_phase(freeze_base, lm_loss_weight):
    with pytest.raises(ValueError, match="must agree on the phase"):
        validate_aux_head_coherence(
            enabled=True,
            freeze_base=freeze_base,
            lm_loss_weight=lm_loss_weight,
            out_activation="sigmoid",
            loss="bce",
        )


@pytest.mark.parametrize(
    "freeze_base, lm_loss_weight",
    [
        (True, 0.0),   # frozen base, head-only — Phase A
        (False, 1.0),  # unfrozen base, weighted LM term — Phase B
    ],
)
def test_validate_coherence_accepts_coherent_phase(freeze_base, lm_loss_weight):
    # No raise on either valid phase combination.
    validate_aux_head_coherence(
        enabled=True,
        freeze_base=freeze_base,
        lm_loss_weight=lm_loss_weight,
        out_activation="sigmoid",
        loss="bce",
    )


@pytest.mark.parametrize("loss", ["bce", "brier"])
def test_validate_coherence_rejects_identity_with_probability_loss(loss):
    with pytest.raises(ValueError, match="out_activation='identity' is incompatible"):
        validate_aux_head_coherence(
            enabled=True,
            freeze_base=True,
            lm_loss_weight=0.0,
            out_activation="identity",
            loss=loss,
        )


def test_validate_coherence_is_noop_when_disabled():
    # A disabled head has inert fields — even an otherwise-incoherent combination
    # must not raise (mirrors the enabled-gated YAML-load behavior).
    validate_aux_head_coherence(
        enabled=False,
        freeze_base=False,
        lm_loss_weight=0.0,
        out_activation="identity",
        loss="bce",
    )


def test_config_has_real_aux_head_field_so_block_is_not_silently_dropped():
    # The whole point: aux_head must be a declared field on Config, otherwise
    # dict_to_dataclass-style loading would silently ignore the YAML block.
    field_names = {f.name for f in Config.__dataclass_fields__.values()}
    assert "aux_head" in field_names
    assert Config.__dataclass_fields__["aux_head"].type is AuxHeadConfig


def test_default_constructed_config_field_is_disabled():
    # default_factory must give every existing config an inert, disabled head.
    default = AuxHeadConfig()
    assert default.enabled is False
