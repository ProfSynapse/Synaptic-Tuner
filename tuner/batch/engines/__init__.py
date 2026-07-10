"""Pluggable batch engines behind a small common interface.

Location: tuner/batch/engines/__init__.py

An engine implements one or both of the batch verbs:

  - ``GenerateEngine.generate(items, ...)`` -> completions
  - ``CaptureEngine.capture(items, ...)``   -> per-layer hidden states

Engines are chosen by name (``hf-batched``, ``vllm``). ``vllm`` is a soft,
optional dependency: constructing that engine raises a clear error if vLLM is
not installed or too old, and it is never imported at module load, so the tuner
has no hard vLLM dependency.
"""

from tuner.batch.engines.base import (
    GenerateEngine,
    CaptureEngine,
    GenerateItem,
    GenerateResult,
    CaptureItem,
    CaptureResult,
    OutOfMemoryError,
)


def get_generate_engine(name: str, **kwargs) -> GenerateEngine:
    """Construct a generation engine by name."""
    if name == "hf-batched":
        from tuner.batch.engines.hf_batched import HFBatchedGenerateEngine

        return HFBatchedGenerateEngine(**kwargs)
    if name == "vllm":
        from tuner.batch.engines.vllm_engine import VLLMGenerateEngine

        return VLLMGenerateEngine(**kwargs)
    raise ValueError(f"Unknown generate engine: {name!r} (expected hf-batched or vllm)")


def get_capture_engine(name: str, **kwargs) -> CaptureEngine:
    """Construct a capture engine by name."""
    if name == "hf-batched":
        from tuner.batch.engines.hf_batched import HFBatchedCaptureEngine

        return HFBatchedCaptureEngine(**kwargs)
    if name == "vllm":
        from tuner.batch.engines.vllm_engine import VLLMCaptureEngine

        return VLLMCaptureEngine(**kwargs)
    raise ValueError(f"Unknown capture engine: {name!r} (expected hf-batched or vllm)")


__all__ = [
    "GenerateEngine",
    "CaptureEngine",
    "GenerateItem",
    "GenerateResult",
    "CaptureItem",
    "CaptureResult",
    "OutOfMemoryError",
    "get_generate_engine",
    "get_capture_engine",
]
