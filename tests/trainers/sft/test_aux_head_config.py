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
from config_loader import AuxHeadConfig, Config, load_aux_head_config  # noqa: E402


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
    cfg = load_aux_head_config(
        {
            "enabled": True,
            "layer": 35,
            "token_position": "mean",
            "target_field": "score",
            "loss": "brier",
            "head_type": "mlp",
            "hidden_dims": [64, 16],
            "out_activation": "identity",
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
    assert cfg.out_activation == "identity"
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
