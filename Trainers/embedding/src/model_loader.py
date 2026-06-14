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


def _load_fallback(spec: EmbeddingModelSpec):
    """Load the plain SentenceTransformer baseline (CPU/MPS/CUDA-capable).

    Honors spec.pooling / spec.normalize via the model's own modules.json (the
    base model ships its pooling+normalize configuration); spec.prompts are
    attached so the data loader can prefix by prompt name.
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

    full_finetuning=True for adapter_mode "full"; otherwise a base load that the
    caller adapts with LoRA / a frozen head. The fast path resolves the mirror
    id (spec.resolved_fast_path_id()).
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

    Args:
        spec:           validated EmbeddingModelSpec (the SSOT for load/prompt).
        adapter_mode:   one of {full, lora, frozen_head}. qlora -> ValueError.
        lora_config:    optional LoRA hyperparameters (r/alpha/dropout); the
                        trainer applies them with spec.lora_target_modules /
                        spec.lora_task_type. Carried through for the caller; not
                        consumed here.
        allow_fast_path: opt-out switch for the Unsloth accelerator.

    Returns:
        LoadedEmbeddingModel with .loader_path == "fast" | "fallback".

    Raises:
        ValueError: adapter_mode is qlora (deferred) or outside the closed set.
    """
    _validate_adapter_mode(adapter_mode)

    capabilities = probe_capabilities(allow_fast_path=allow_fast_path)

    if capabilities.fast_path_available:
        try:
            model = _load_fast(spec, adapter_mode)
            return LoadedEmbeddingModel(
                model=model,
                spec=spec,
                loader_path="fast",
                capabilities=capabilities,
            )
        except Exception as exc:
            # Fast path probed available but failed to load -> degrade, do not crash.
            warnings.warn(
                f"Fast path load failed for {spec.name} ({exc}); falling back to "
                f"plain SentenceTransformer.",
                stacklevel=2,
            )

    model = _load_fallback(spec)
    return LoadedEmbeddingModel(
        model=model,
        spec=spec,
        loader_path="fallback",
        capabilities=capabilities,
    )
