"""Smoke tests for the DPO config loader (DPOTrainingConfig).

Imports only yaml/dataclasses via config_loader; no ML stack.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Trainers" / "dpo"))

from configs import config_loader  # noqa: E402


def test_default_config_yaml_loads_into_dpo_dataclasses():
    config = config_loader.load_config()  # defaults to Trainers/dpo/configs/config.yaml

    # DPO-specific fields present, KTO-only fields absent.
    assert hasattr(config.training, "beta")
    assert hasattr(config.training, "loss_type")
    assert config.training.loss_type == "sigmoid"
    assert not hasattr(config.training, "desirable_weight")
    assert not hasattr(config.training, "undesirable_weight")
    assert not hasattr(config.training, "use_kto_s")


def test_default_lora_budget_matches_4b_pilot():
    config = config_loader.load_config()
    assert config.lora.r == 32
    assert config.lora.lora_alpha == 64
    assert config.lora.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]


def test_default_model_is_qwen3_4b_thinking_off_pin():
    config = config_loader.load_config()
    assert "Qwen3-4B" in config.model.model_name


def test_string_numeric_fields_are_coerced(tmp_path):
    # mirrors KTO loader's dict_to_dataclass numeric coercion
    import yaml

    cfg = config_loader.load_yaml_config()
    cfg["training"]["learning_rate"] = "5e-6"  # string -> float
    cfg["training"]["save_steps"] = "25"       # string -> int
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    config = config_loader.load_config(str(path))
    assert isinstance(config.training.learning_rate, float)
    assert isinstance(config.training.save_steps, int)
