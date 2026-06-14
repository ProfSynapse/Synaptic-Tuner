"""
Embedding model registry — EmbeddingModelSpec loader + validation.

Location: Trainers/embedding/src/registry.py
Purpose:  Load and validate the embedding model registry
          (Trainers/embedding/configs/model_registry.yaml) into frozen
          EmbeddingModelSpec dataclasses. The registry is the SSOT for how each
          base model is loaded, prompted, and adapted — no model-specific
          behavior lives in Python (the config-driven rule).
Used by:  Trainers/embedding/src/model_loader.py (resolves a spec -> a loaded
          model), Trainers/embedding/train_embedding.py (entry point selects a
          spec by registry name), and eval-time embedding (a spec is passed in
          via the retrieval verifier's RetrievalConfig by the Evaluator caller).

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §1.2.
The module-level surface (the only thing callers import) is:
    load_registry(path) -> dict[str, EmbeddingModelSpec]
    get_spec(name, path) -> EmbeddingModelSpec
    list_models(path)   -> list[str]

The loader validates every entry and raises ValueError (naming the offending
registry key) on any schema violation, so a typo fails loudly at load time
rather than silently mis-loading a model downstream.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

# ---------------------------------------------------------------------------
# Allowed enum values (validation domains). Kept as module constants so both the
# dataclass docstring and the validator reference one source.
# ---------------------------------------------------------------------------
VALID_FAMILIES = frozenset({"bert", "xlm-roberta", "decoder"})
VALID_POOLINGS = frozenset({"mean", "cls", "last_token", "weighted_mean"})
VALID_EMBEDDING_TYPES = frozenset({"bi_encoder", "cross_encoder"})

# Default registry path (this file lives in Trainers/embedding/src/).
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_registry.yaml"


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Fully describes how to load, prompt, and adapt one embedding base model.

    Frozen so a loaded spec cannot be mutated after validation. Every field maps
    1:1 to a key in a model_registry.yaml model block (§1.1).
    """

    name: str                                  # registry key (e.g. "bge-base-en")
    hf_id: str
    family: str                                # "bert" | "xlm-roberta" | "decoder"
    embedding_type: str = "bi_encoder"         # "bi_encoder" | "cross_encoder"
    pooling: str = "mean"                       # "mean" | "cls" | "last_token" | "weighted_mean"
    normalize: bool = True
    max_seq_length: int = 512
    default_dim: int | None = None
    matryoshka_dims: tuple[int, ...] = ()
    query_prompt: str = ""
    passage_prompt: str = ""
    prompt_required: bool = False
    lora_target_modules: tuple[str, ...] = ()
    lora_task_type: str | None = None          # "FEATURE_EXTRACTION" for decoder; None otherwise
    trust_remote_code: bool = False
    fast_path_hf_id: str | None = None         # optional unsloth/-prefixed mirror id

    def resolved_fast_path_id(self) -> str:
        """hf_id to use on the Unsloth fast path (mirror override or canonical)."""
        return self.fast_path_hf_id or self.hf_id


# Set of dataclass field names that may legitimately appear in a YAML block.
# `name` is supplied from the block key, not the block body, so it is excluded
# from the body's allowed-key set.
_SPEC_FIELD_NAMES = frozenset(f.name for f in fields(EmbeddingModelSpec))
_BODY_ALLOWED_KEYS = _SPEC_FIELD_NAMES - {"name"}


def _coerce_and_validate(name: str, block: Mapping[str, Any]) -> EmbeddingModelSpec:
    """Validate one registry block and build a frozen EmbeddingModelSpec.

    Raises ValueError (naming the offending registry key) on any violation.
    """
    if not isinstance(block, Mapping):
        raise ValueError(f"[{name}] registry entry must be a mapping, got {type(block).__name__}")

    # Unknown keys -> fail (catch typos; config-driven discipline). §1.2.
    unknown = set(block.keys()) - _BODY_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"[{name}] unknown key(s) in registry block: {sorted(unknown)}. "
            f"Allowed keys: {sorted(_BODY_ALLOWED_KEYS)}"
        )

    hf_id = block.get("hf_id")
    if not hf_id or not isinstance(hf_id, str):
        raise ValueError(f"[{name}] 'hf_id' is required and must be a non-empty string")

    family = block.get("family")
    if family not in VALID_FAMILIES:
        raise ValueError(
            f"[{name}] invalid family {family!r}; must be one of {sorted(VALID_FAMILIES)}"
        )

    pooling = block.get("pooling", "mean")
    if pooling not in VALID_POOLINGS:
        raise ValueError(
            f"[{name}] invalid pooling {pooling!r}; must be one of {sorted(VALID_POOLINGS)}"
        )

    embedding_type = block.get("embedding_type", "bi_encoder")
    if embedding_type not in VALID_EMBEDDING_TYPES:
        raise ValueError(
            f"[{name}] invalid embedding_type {embedding_type!r}; "
            f"must be one of {sorted(VALID_EMBEDDING_TYPES)}"
        )

    default_dim = block.get("default_dim")
    if default_dim is not None and (not isinstance(default_dim, int) or default_dim <= 0):
        raise ValueError(f"[{name}] default_dim must be a positive int or null, got {default_dim!r}")

    # matryoshka_dims: all <= default_dim, sorted descending. §1.2.
    raw_matryoshka = block.get("matryoshka_dims", []) or []
    if not isinstance(raw_matryoshka, (list, tuple)):
        raise ValueError(f"[{name}] matryoshka_dims must be a list, got {type(raw_matryoshka).__name__}")
    matryoshka_dims = tuple(int(d) for d in raw_matryoshka)
    if matryoshka_dims:
        if default_dim is not None and any(d > default_dim for d in matryoshka_dims):
            raise ValueError(
                f"[{name}] matryoshka_dims {matryoshka_dims} contains a value greater than "
                f"default_dim ({default_dim})"
            )
        if list(matryoshka_dims) != sorted(matryoshka_dims, reverse=True):
            raise ValueError(
                f"[{name}] matryoshka_dims {matryoshka_dims} must be sorted descending"
            )

    prompt_required = bool(block.get("prompt_required", False))
    query_prompt = block.get("query_prompt", "") or ""
    passage_prompt = block.get("passage_prompt", "") or ""
    if prompt_required and (not query_prompt or not passage_prompt):
        raise ValueError(
            f"[{name}] prompt_required is true but query_prompt/passage_prompt are not both "
            f"non-empty (query_prompt={query_prompt!r}, passage_prompt={passage_prompt!r})"
        )

    raw_targets = block.get("lora_target_modules", []) or []
    if not isinstance(raw_targets, (list, tuple)):
        raise ValueError(
            f"[{name}] lora_target_modules must be a list, got {type(raw_targets).__name__}"
        )
    lora_target_modules = tuple(str(m) for m in raw_targets)

    lora_task_type = block.get("lora_task_type")
    # decoder family -> lora_task_type must be FEATURE_EXTRACTION (warn-and-set if omitted). §1.2.
    if family == "decoder":
        if lora_task_type is None:
            warnings.warn(
                f"[{name}] decoder family requires lora_task_type=FEATURE_EXTRACTION; "
                f"it was omitted — defaulting it on.",
                stacklevel=2,
            )
            lora_task_type = "FEATURE_EXTRACTION"
        elif lora_task_type != "FEATURE_EXTRACTION":
            raise ValueError(
                f"[{name}] decoder family requires lora_task_type=FEATURE_EXTRACTION, "
                f"got {lora_task_type!r}"
            )

    return EmbeddingModelSpec(
        name=name,
        hf_id=hf_id,
        family=family,
        embedding_type=embedding_type,
        pooling=pooling,
        normalize=bool(block.get("normalize", True)),
        max_seq_length=int(block.get("max_seq_length", 512)),
        default_dim=default_dim,
        matryoshka_dims=matryoshka_dims,
        query_prompt=query_prompt,
        passage_prompt=passage_prompt,
        prompt_required=prompt_required,
        lora_target_modules=lora_target_modules,
        lora_task_type=lora_task_type,
        trust_remote_code=bool(block.get("trust_remote_code", False)),
        fast_path_hf_id=block.get("fast_path_hf_id"),
    )


def load_registry(path: Path | None = None) -> dict[str, EmbeddingModelSpec]:
    """Load + validate the full registry into {name: EmbeddingModelSpec}.

    Args:
        path: Optional override for the registry YAML. Defaults to
              Trainers/embedding/configs/model_registry.yaml.

    Returns:
        Mapping of registry key -> validated EmbeddingModelSpec.

    Raises:
        FileNotFoundError: registry file does not exist.
        ValueError: any block fails validation (the offending key is named).
    """
    registry_path = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Embedding model registry not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    models = raw.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError(
            f"Registry {registry_path} must contain a non-empty top-level 'models' mapping"
        )

    specs: dict[str, EmbeddingModelSpec] = {}
    for name, block in models.items():
        specs[str(name)] = _coerce_and_validate(str(name), block)
    return specs


def get_spec(name: str, path: Path | None = None) -> EmbeddingModelSpec:
    """Return the validated spec for one registry key.

    Raises:
        KeyError: name is not in the registry (lists the available keys).
    """
    registry = load_registry(path)
    if name not in registry:
        raise KeyError(
            f"Unknown embedding model {name!r}. Available: {sorted(registry.keys())}"
        )
    return registry[name]


def list_models(path: Path | None = None) -> list[str]:
    """Return the sorted list of registry keys."""
    return sorted(load_registry(path).keys())
