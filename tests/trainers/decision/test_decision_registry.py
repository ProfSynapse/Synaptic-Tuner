"""Decision torso registry + run-config resolution (torch-free).

Every checked-in config and recipe must resolve to a valid registry entry, and
the resolved model config must carry everything a checkpoint needs to reload.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DECISION_DIR = REPO_ROOT / "Trainers" / "decision"
for p in (REPO_ROOT, DECISION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from decision_core.config import load_run_config  # noqa: E402
from decision_core.registry import get_spec, list_models, load_registry  # noqa: E402

CONFIGS = sorted(p.name for p in (DECISION_DIR / "configs").glob("*.yaml")
                 if p.name not in {"model_registry.yaml", "corpus_strands_v5.yaml"})


def test_registry_entries_validate_and_pin_revisions():
    registry = load_registry()
    assert "qwen35-2b-base" in registry
    for name, spec in registry.items():
        assert spec.revision and len(spec.revision) == 40, f"{name} must pin a full commit sha"
        if spec.loader == "text_tower":
            assert spec.causal_lm_class
    spec = get_spec("qwen35-2b-base")
    assert spec.hf_id == "Qwen/Qwen3.5-2B-Base"
    assert {"in_proj_qkv", "out_proj"} <= set(spec.lora_target_modules)


@pytest.mark.parametrize("body,match", [
    ({"hf_id": "x", "loader": "causal_lm", "torch_dtype": "bfloat16", "max_length": 10,
      "lora_target_modules": ["q"], "typo": 1}, "unknown key"),
    ({"hf_id": "x", "loader": "text_tower", "torch_dtype": "bfloat16", "max_length": 10,
      "lora_target_modules": ["q"]}, "causal_lm_class"),
    ({"hf_id": "x", "loader": "magic", "torch_dtype": "bfloat16", "max_length": 10,
      "lora_target_modules": ["q"]}, "loader"),
    ({"hf_id": "x", "loader": "causal_lm", "torch_dtype": "bfloat16", "max_length": 10}, "missing"),
])
def test_registry_rejects_bad_entries(tmp_path, body, match):
    path = tmp_path / "reg.yaml"
    path.write_text(json.dumps({"models": {"bad": body}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_registry(path)
    with pytest.raises(ValueError):
        get_spec("nope")


@pytest.mark.parametrize("name", CONFIGS)
def test_every_config_resolves(name):
    import train_decision

    cfg = load_run_config(DECISION_DIR / "configs" / name)
    mcfg, spec = train_decision.resolve_model_config(cfg)
    assert spec.name in list_models()
    assert mcfg.hf_id == spec.hf_id and mcfg.revision == spec.revision
    assert mcfg.lora_targets == list(spec.lora_target_modules)
    assert mcfg.max_length == spec.max_length and mcfg.torch_dtype == spec.torch_dtype
    assert mcfg.loader == spec.loader and mcfg.causal_lm_class == spec.causal_lm_class
    assert set(mcfg.headers) == {"noul", "choice", "score"}
    assert mcfg.marker_style == cfg.prompt.marker_style


def test_run_config_overrides_registry_values():
    import train_decision

    cfg = load_run_config(DECISION_DIR / "configs" / "config.yaml")
    cfg.model.registry_name = "qwen3-1.7b-base"
    cfg.model.max_length = 2048
    cfg.lora.target_modules = ["q_proj", "v_proj"]
    mcfg, _ = train_decision.resolve_model_config(cfg)
    assert mcfg.hf_id == "Qwen/Qwen3-1.7B-Base" and mcfg.loader == "causal_lm"
    assert mcfg.max_length == 2048 and mcfg.lora_targets == ["q_proj", "v_proj"]


def test_smoke_configs_match_full_configs_except_size():
    """A smoke config may only shrink the run; model/prompt/loss must match its full config."""
    pairs = {"smoke_pointer.yaml": "config.yaml", "smoke_letter_logits.yaml": "letter_logits.yaml"}
    for smoke, full in pairs.items():
        a = yaml.safe_load((DECISION_DIR / "configs" / smoke).read_text(encoding="utf-8"))
        b = yaml.safe_load((DECISION_DIR / "configs" / full).read_text(encoding="utf-8"))
        for section in ("model", "prompt", "lora", "augmentation", "loss"):
            assert a[section] == b[section], (smoke, section)


@pytest.mark.parametrize("recipe", sorted((REPO_ROOT / "Trainers" / "recipes").glob("decision_qwen35_*.yaml")))
def test_recipes_point_at_configs_without_inline_overrides(recipe):
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    command = data["run"]["command"]
    config = command[command.index("--config") + 1]
    assert (REPO_ROOT / config).exists()
    for flag in ("--max-steps", "--max-train-examples", "--max-eval-examples", "--model", "--readout"):
        assert flag not in command, f"{recipe.name} overrides {flag}; put it in the config"
