"""Smoke tests for DPO model-preset resolution (no ML stack)."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "dpo"))

from configs import model_presets  # noqa: E402


def _args(**flags):
    # All MODEL_MAP flags default False; override the ones under test.
    defaults = {flag: False for flag in model_presets.MODEL_MAP}
    defaults.update(flags)
    return SimpleNamespace(**defaults)


def test_qwen3_4b_resolves_to_pilot_pin():
    size, repo = model_presets.resolve_model_flag(_args(qwen3_4b=True))
    assert size == "3b"
    assert repo == "unsloth/Qwen3-4B-bnb-4bit"


def test_qwen3_8b_resolves_to_confirm_pin():
    size, repo = model_presets.resolve_model_flag(_args(qwen3_8b=True))
    assert size == "7b"
    assert repo == "unsloth/Qwen3-8B-bnb-4bit"


def test_no_flag_returns_none():
    assert model_presets.resolve_model_flag(_args()) is None


def test_first_flag_in_map_order_wins():
    # qwen_3b precedes qwen3_8b in MODEL_MAP; both set -> first wins.
    size, repo = model_presets.resolve_model_flag(_args(qwen_3b=True, qwen3_8b=True))
    assert repo == "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
