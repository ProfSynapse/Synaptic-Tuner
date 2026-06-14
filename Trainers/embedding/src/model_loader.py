"""
Dual embedding-model loader — fallback-primary with an optional Unsloth fast path.

Location: Trainers/embedding/src/model_loader.py
Purpose:  Load an embedding base model (per an EmbeddingModelSpec) into a uniform
          object the trainer consumes loader-agnostically. The plain
          SentenceTransformer is the CORRECTNESS BASELINE; the Unsloth
          FastSentenceTransformer is an OPTIONAL accelerator selected by a
          capability probe that NEVER raises (R1).
Used by:  Trainers/embedding/train_embedding.py (the training entry point) and
          any code path that needs a trained/base embedding model in-process.

Contract: docs/architecture/embedding-reranker-phase1/01_CONTRACTS.md §2.

Design stance (§2.1): the trainer code is identical regardless of which loader
returns — both yield a LoadedEmbeddingModel whose `.model` is a
SentenceTransformer-compatible object. The fast path is a non-blocking
optimization: any probe failure (no CUDA, unsloth not importable, opt-in
disabled) degrades silently to the fallback.

Adapter-mode axis (§2.4): exactly {full, lora, frozen_head}. There is NO qlora
mode in v1 (R8) — a "qlora" value raises ValueError with a deferral message.

Adapter application (§2.4 — the loader is the single application authority): this
module does not merely *load* a base model, it returns a model already shaped for
the requested adapter_mode so the trainer is loader-agnostic and trains whatever
it receives:
  - full        -> all base weights trainable (fast: full_finetuning=True).
  - lora        -> a PEFT/Unsloth LoRA adapter is applied (target_modules /
                   task_type from the spec; r/alpha/dropout from lora_config), so
                   training emits a small adapter (the §2.4 "reuses merge/upload"
                   promise) instead of silently full-tuning.
  - frozen_head -> the base is frozen and only an appended Dense head is trained.
The per-path mechanism differs (Unsloth get_peft_model on the fast model vs PEFT
add_adapter on the fallback SentenceTransformer), but both honor the same spec
fields — the WHAT is invariant, only the HOW diverges (§2 dual-loader contract).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

# NOTE: torch is imported lazily inside probe_capabilities so that importing this
# module for its dataclasses/typing (e.g. in a no-torch context) does not force a
# torch import. The trainer always has torch available before it calls the loader.

from registry import EmbeddingModelSpec  # noqa: E402  (src is on sys.path via the trainer bootstrap)

# Adapter modes are a closed set (§2.4). qlora is explicitly deferred (R8).
VALID_ADAPTER_MODES = frozenset({"full", "lora", "frozen_head"})
_DEFERRED_ADAPTER_MODES = frozenset({"qlora"})

# Default LoRA hyperparameters when config supplies no `lora:` block. Mirrors
# config.yaml's documented defaults so an omitted block is not a silent r=0.
_DEFAULT_LORA_R = 16
_DEFAULT_LORA_ALPHA = 32
_DEFAULT_LORA_DROPOUT = 0.05

# Name of the appended trainable head for frozen_head mode. A bge/e5/gte base
# ships Transformer + Pooling (+ Normalize) with no trainable projection head, so
# frozen_head must append one to have something to train (§2.4).
_FROZEN_HEAD_MODULE_NAME = "frozen_head_dense"


@dataclass(frozen=True)
class LoaderCapabilities:
    """Result of the capability probe — why fast vs fallback was chosen."""

    fast_path_available: bool      # unsloth importable AND CUDA AND opt-in not disabled
    cuda: bool
    reason: str                    # human-readable explanation


@dataclass
class LoadedEmbeddingModel:
    """Uniform return object so downstream training code is loader-agnostic."""

    model: Any                     # SentenceTransformer-compatible (fast or fallback)
    spec: EmbeddingModelSpec
    loader_path: str               # "fast" | "fallback"
    capabilities: LoaderCapabilities


def probe_capabilities(*, allow_fast_path: bool = True) -> LoaderCapabilities:
    """Decide fast vs fallback. NEVER raises — any failure degrades to fallback.

    Order (§2.2):
      1. allow_fast_path is False               -> fallback (reason="disabled by config")
      2. torch.cuda.is_available() is False      -> fallback (reason="no CUDA (Mac/MPS/CPU)")
      3. import unsloth; FastSentenceTransformer -> fast    (reason="unsloth available")
         ImportError / any error                 -> fallback (reason="unsloth not importable")

    The probe is import-guarded and TOTAL: every failure mode resolves to a
    fallback capability rather than propagating an exception (R1).
    """
    if not allow_fast_path:
        return LoaderCapabilities(fast_path_available=False, cuda=False, reason="disabled by config")

    # Step 2: CUDA. Guard the torch import itself so a broken/absent torch -> fallback.
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - environment-dependent
        return LoaderCapabilities(
            fast_path_available=False, cuda=False, reason=f"torch unavailable: {exc}"
        )

    if not cuda:
        return LoaderCapabilities(
            fast_path_available=False, cuda=False, reason="no CUDA (Mac/MPS/CPU)"
        )

    # Step 3: unsloth import. Total guard — ImportError or any runtime error -> fallback.
    try:
        import unsloth  # noqa: F401
        from unsloth import FastSentenceTransformer  # noqa: F401
    except Exception as exc:
        return LoaderCapabilities(
            fast_path_available=False, cuda=cuda, reason=f"unsloth not importable: {exc}"
        )

    return LoaderCapabilities(fast_path_available=True, cuda=cuda, reason="unsloth available")


def _validate_adapter_mode(adapter_mode: str) -> None:
    """Reject qlora (deferred, R8) and any value outside the closed set (§2.4)."""
    if adapter_mode in _DEFERRED_ADAPTER_MODES:
        raise ValueError(
            f"adapter_mode={adapter_mode!r} (QLoRA) is deferred to a later phase. "
            f"v1 supports only {sorted(VALID_ADAPTER_MODES)}."
        )
    if adapter_mode not in VALID_ADAPTER_MODES:
        raise ValueError(
            f"Unknown adapter_mode={adapter_mode!r}; must be one of {sorted(VALID_ADAPTER_MODES)}"
        )


def _build_prompts(spec: EmbeddingModelSpec) -> dict[str, str] | None:
    """Map the spec's query/passage prompts into ST's prompt dict, or None.

    SentenceTransformer accepts a `prompts` dict keyed by a prompt name; the
    trainer/data loader reference these by the canonical keys "query"/"passage".
    Returns None when neither prompt is set (so ST defaults apply).
    """
    prompts: dict[str, str] = {}
    if spec.query_prompt:
        prompts["query"] = spec.query_prompt
    if spec.passage_prompt:
        prompts["passage"] = spec.passage_prompt
    return prompts or None


def _lora_hyperparams(lora_config: Mapping[str, Any] | None) -> tuple[int, int, float]:
    """Resolve (r, alpha, dropout) from a config `lora:` block, applying defaults.

    A missing block or missing key falls back to the documented config.yaml
    defaults (r=16, alpha=32, dropout=0.05) rather than a silent zero.
    """
    cfg = lora_config or {}
    r = int(cfg.get("r", _DEFAULT_LORA_R))
    alpha = int(cfg.get("alpha", _DEFAULT_LORA_ALPHA))
    dropout = float(cfg.get("dropout", _DEFAULT_LORA_DROPOUT))
    return r, alpha, dropout


def _apply_lora_fallback(model, spec: EmbeddingModelSpec, lora_config: Mapping[str, Any] | None):
    """Attach a PEFT LoRA adapter to a plain SentenceTransformer (fallback path).

    Uses the modern ST surface `model.add_adapter(LoraConfig(...))` (sentence-
    transformers 3.x+), honoring spec.lora_target_modules and — only when the spec
    sets it (decoder family) — spec.lora_task_type; for encoders the spec leaves it
    null so PEFT infers it. r/alpha/dropout come from the config `lora:` block.

    enable_input_require_grads() is called after attaching so gradients flow when a
    cached in-batch-negative loss (CachedMultipleNegativesRankingLoss) is selected
    — without it the cached path raises "None of the inputs have requires_grad".
    """
    from peft import LoraConfig

    r, alpha, dropout = _lora_hyperparams(lora_config)
    config_kwargs: dict[str, Any] = {
        "inference_mode": False,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
    }
    if spec.lora_target_modules:
        config_kwargs["target_modules"] = list(spec.lora_target_modules)
    if spec.lora_task_type:
        config_kwargs["task_type"] = spec.lora_task_type

    model.add_adapter(LoraConfig(**config_kwargs))

    # Cached-loss gradient flow through the frozen base (ST troubleshooting note).
    transformers_model = getattr(model, "transformers_model", None)
    if transformers_model is not None and hasattr(transformers_model, "enable_input_require_grads"):
        transformers_model.enable_input_require_grads()
    return model


def _apply_lora_fast(model, spec: EmbeddingModelSpec, lora_config: Mapping[str, Any] | None):
    """Attach an Unsloth LoRA adapter to a FastSentenceTransformer (fast path).

    Mirrors FastLanguageModel.get_peft_model — Unsloth infers the task type from
    the model, so only r/alpha/dropout + target_modules are threaded. The fast
    path is CUDA-only and never load-bearing (R1); the fallback above is the
    correctness baseline.
    """
    from unsloth import FastSentenceTransformer

    r, alpha, dropout = _lora_hyperparams(lora_config)
    return FastSentenceTransformer.get_peft_model(
        model,
        r=r,
        target_modules=list(spec.lora_target_modules),
        lora_alpha=alpha,
        lora_dropout=dropout,
    )


def _apply_frozen_head(model, spec: EmbeddingModelSpec):
    """Freeze the base and train only an appended Dense head (frozen_head, §2.4).

    Encoder bases (bge/e5/gte) ship Transformer + Pooling (+ Normalize) with no
    trainable projection head, so a Dense head is appended when absent — otherwise
    there is nothing to train. All pre-existing parameters are frozen
    (requires_grad=False); only the appended head stays trainable.

    Returns the model with exactly the head trainable.
    """
    from sentence_transformers import models as st_models

    # Freeze everything currently in the model.
    for param in model.parameters():
        param.requires_grad = False

    # Append a Dense head if the model has none (idempotent on the module name).
    has_named_head = any(name == _FROZEN_HEAD_MODULE_NAME for name, _ in model.named_children()) or (
        any(isinstance(mod, st_models.Dense) for mod in model.modules())
    )
    if not has_named_head:
        embedding_dim = model.get_sentence_embedding_dimension()
        head = st_models.Dense(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=True,
        )
        # ST containers behave like nn.Sequential keyed by string indices; append
        # under a stable name so the head is discoverable and saved with the model.
        model.add_module(_FROZEN_HEAD_MODULE_NAME, head)

    # Unfreeze only the appended head's parameters.
    for name, module in model.named_modules():
        if isinstance(module, st_models.Dense):
            for param in module.parameters():
                param.requires_grad = True
    return model


def _load_fallback(spec: EmbeddingModelSpec):
    """Load the plain SentenceTransformer baseline (CPU/MPS/CUDA-capable).

    Honors spec.pooling / spec.normalize via the model's own modules.json (the
    base model ships its pooling+normalize configuration); spec.prompts are
    attached so the data loader can prefix by prompt name. Adapter application
    (LoRA / frozen head) is layered on by load_embedding_model after this load.
    """
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {}
    prompts = _build_prompts(spec)
    if prompts is not None:
        model_kwargs["prompts"] = prompts

    return SentenceTransformer(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        **model_kwargs,
    )


def _load_fast(spec: EmbeddingModelSpec, adapter_mode: str):
    """Load via Unsloth FastSentenceTransformer (fast path).

    full_finetuning=True for adapter_mode "full"; otherwise a base load that
    load_embedding_model adapts with LoRA / a frozen head. The fast path resolves
    the mirror id (spec.resolved_fast_path_id()).
    """
    from unsloth import FastSentenceTransformer

    return FastSentenceTransformer.from_pretrained(
        spec.resolved_fast_path_id(),
        max_seq_length=spec.max_seq_length,
        full_finetuning=(adapter_mode == "full"),
    )


def load_embedding_model(
    spec: EmbeddingModelSpec,
    adapter_mode: str,             # "full" | "lora" | "frozen_head"
    *,
    lora_config: Mapping[str, Any] | None = None,
    allow_fast_path: bool = True,
) -> LoadedEmbeddingModel:
    """Load `spec` into a uniform LoadedEmbeddingModel.

    The fast path (Unsloth FastSentenceTransformer) is used only when the
    capability probe reports it available AND it loads without error; ANY failure
    falls back to the plain SentenceTransformer baseline. The returned object is
    loader-agnostic so the trainer is identical regardless of path.

    The loader is the single adapter-application authority (§2.4): the returned
    model is already shaped for adapter_mode (LoRA-wrapped, or base-frozen with a
    trainable head), so the trainer trains whatever it receives — config.yaml's
    default `lora` therefore actually emits a small adapter, not a silent
    full-tune.

    Args:
        spec:           validated EmbeddingModelSpec (the SSOT for load/prompt).
        adapter_mode:   one of {full, lora, frozen_head}. qlora -> ValueError.
        lora_config:    optional LoRA hyperparameters (r/alpha/dropout). CONSUMED
                        here for adapter_mode=="lora" — combined with
                        spec.lora_target_modules / spec.lora_task_type to build the
                        adapter. Missing keys fall back to the documented defaults.
        allow_fast_path: opt-out switch for the Unsloth accelerator.

    Returns:
        LoadedEmbeddingModel with .loader_path == "fast" | "fallback", whose
        .model already has the requested adapter_mode applied.

    Raises:
        ValueError: adapter_mode is qlora (deferred) or outside the closed set.
    """
    _validate_adapter_mode(adapter_mode)

    capabilities = probe_capabilities(allow_fast_path=allow_fast_path)

    if capabilities.fast_path_available:
        try:
            model = _load_fast(spec, adapter_mode)
            model = _apply_adapter_mode_fast(model, spec, adapter_mode, lora_config)
            return LoadedEmbeddingModel(
                model=model,
                spec=spec,
                loader_path="fast",
                capabilities=capabilities,
            )
        except Exception as exc:
            # Fast path probed available but failed to load/adapt -> degrade.
            warnings.warn(
                f"Fast path load failed for {spec.name} ({exc}); falling back to "
                f"plain SentenceTransformer.",
                stacklevel=2,
            )

    model = _load_fallback(spec)
    model = _apply_adapter_mode_fallback(model, spec, adapter_mode, lora_config)
    return LoadedEmbeddingModel(
        model=model,
        spec=spec,
        loader_path="fallback",
        capabilities=capabilities,
    )


def _apply_adapter_mode_fast(
    model, spec: EmbeddingModelSpec, adapter_mode: str, lora_config: Mapping[str, Any] | None
):
    """Dispatch adapter application on the fast (Unsloth) model.

    full -> no-op (full_finetuning was set at load); lora -> Unsloth LoRA;
    frozen_head -> freeze base + train an appended head.
    """
    if adapter_mode == "lora":
        return _apply_lora_fast(model, spec, lora_config)
    if adapter_mode == "frozen_head":
        return _apply_frozen_head(model, spec)
    return model


def _apply_adapter_mode_fallback(
    model, spec: EmbeddingModelSpec, adapter_mode: str, lora_config: Mapping[str, Any] | None
):
    """Dispatch adapter application on the fallback (plain ST) model.

    full -> no-op (all base weights already trainable); lora -> PEFT adapter;
    frozen_head -> freeze base + train an appended head.
    """
    if adapter_mode == "lora":
        return _apply_lora_fallback(model, spec, lora_config)
    if adapter_mode == "frozen_head":
        return _apply_frozen_head(model, spec)
    return model
