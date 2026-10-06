"""
Decision torso registry: load and validate configs/model_registry.yaml.

Location: Trainers/decision/decision_core/registry.py
Used by:  train_decision.py (resolves model.registry_name), tests.

Mirrors Trainers/embedding/src/registry.py: the YAML is the single source of
truth for how each base model is loaded and adapted, every entry is validated
at load time, and a typo fails loudly naming the offending key. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

VALID_LOADERS = frozenset({"causal_lm", "text_tower"})
VALID_DTYPES = frozenset({"bfloat16", "float16", "float32"})

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_registry.yaml"


@dataclass(frozen=True)
class TorsoSpec:
    name: str
    hf_id: str
    revision: str | None
    loader: str
    causal_lm_class: str | None
    torch_dtype: str
    max_length: int
    attn_implementation: str | None
    lora_target_modules: tuple[str, ...]
    notes: str = ""


_BODY_FIELDS = frozenset(f.name for f in fields(TorsoSpec)) - {"name"}
_REQUIRED = frozenset({"hf_id", "loader", "torch_dtype", "max_length", "lora_target_modules"})


def _spec(name: str, body: Any) -> TorsoSpec:
    if not isinstance(body, dict):
        raise ValueError(f"model_registry[{name!r}] must be a mapping")
    unknown = sorted(set(body) - _BODY_FIELDS)
    if unknown:
        raise ValueError(f"model_registry[{name!r}] has unknown key(s) {unknown}")
    missing = sorted(k for k in _REQUIRED if body.get(k) in (None, "", []))
    if missing:
        raise ValueError(f"model_registry[{name!r}] is missing {missing}")
    if body["loader"] not in VALID_LOADERS:
        raise ValueError(f"model_registry[{name!r}].loader must be one of {sorted(VALID_LOADERS)}")
    if body["loader"] == "text_tower" and not body.get("causal_lm_class"):
        raise ValueError(f"model_registry[{name!r}] uses loader text_tower but sets no causal_lm_class")
    if body["torch_dtype"] not in VALID_DTYPES:
        raise ValueError(f"model_registry[{name!r}].torch_dtype must be one of {sorted(VALID_DTYPES)}")
    if int(body["max_length"]) <= 0:
        raise ValueError(f"model_registry[{name!r}].max_length must be positive")
    return TorsoSpec(
        name=name,
        hf_id=str(body["hf_id"]),
        revision=body.get("revision"),
        loader=str(body["loader"]),
        causal_lm_class=body.get("causal_lm_class"),
        torch_dtype=str(body["torch_dtype"]),
        max_length=int(body["max_length"]),
        attn_implementation=body.get("attn_implementation"),
        lora_target_modules=tuple(str(m) for m in body["lora_target_modules"]),
        notes=str(body.get("notes") or ""),
    )


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, TorsoSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path}: expected a non-empty top-level 'models' mapping")
    return {str(name): _spec(str(name), body) for name, body in models.items()}


def get_spec(name: str, path: str | Path = DEFAULT_REGISTRY_PATH) -> TorsoSpec:
    registry = load_registry(path)
    if name not in registry:
        raise ValueError(f"unknown decision torso {name!r}; registered: {sorted(registry)}")
    return registry[name]


def list_models(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    return sorted(load_registry(path))
