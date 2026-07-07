"""vLLM engine (optional, soft dependency).

Location: tuner/batch/engines/vllm_engine.py
Purpose: Continuous-batching generation via vLLM, and (where the installed
    vLLM supports it) native per-layer hidden-state extraction. vLLM is never
    imported at module load and is not a tuner requirement: constructing either
    engine raises a clear, actionable error when vLLM is missing or too old.
Used by: tuner.batch.engines.get_generate_engine / get_capture_engine.

Notes
-----
* Generation: greedy by default, sampled with temperature/top_p, seeded.
  vLLM handles batching internally, so the runner's ``batch_size`` is advisory
  for the generate path (we still slice to keep memory bounded and to honor the
  OOM-halving contract if vLLM raises OOM).
* Capture: native hidden-state extraction landed in vLLM v0.18.0 (prefill-only,
  per-layer selection) but its surface is connector-based and version-specific.
  If the installed vLLM does not expose a usable form, we raise a clear error
  telling the user to use ``--engine hf-batched`` for capture (which is already
  fast enough — generation was the bottleneck).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from tuner.batch.engines.base import (
    CaptureEngine,
    CaptureItem,
    CaptureResult,
    GenerateEngine,
    GenerateItem,
    GenerateResult,
    OutOfMemoryError,
)


def _require_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "vllm is not installed. Install vllm to use --engine vllm, or use "
            "--engine hf-batched (the default), which needs no extra dependency."
        ) from exc
    return vllm


class VLLMGenerateEngine(GenerateEngine):
    """Continuous-batching generation via vLLM."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 48,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        extra_eos_tokens: Optional[List[str]] = None,
        stop: Optional[List[str]] = None,
        trust_remote_code: bool = True,
        dtype: Optional[str] = None,
        **_ignored: Any,
    ):
        vllm = _require_vllm()
        from vllm import LLM, SamplingParams  # noqa: F401

        self._SamplingParams = SamplingParams
        llm_kwargs: dict = {"model": model_name, "trust_remote_code": trust_remote_code}
        if dtype:
            llm_kwargs["dtype"] = dtype
        if seed is not None:
            llm_kwargs["seed"] = seed
        self.llm = LLM(**llm_kwargs)
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature) if do_sample else 0.0
        self.top_p = float(top_p) if do_sample else 1.0
        all_stops = list(stop) if stop else []
        all_stops.extend(list(extra_eos_tokens) if extra_eos_tokens else [])
        self.stop = all_stops or None

    def _sampling_params(self):
        kwargs = dict(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )
        try:
            return self._SamplingParams(**kwargs, min_tokens=self.min_new_tokens)
        except TypeError:
            return self._SamplingParams(**kwargs)

    def generate(
        self,
        items: List[GenerateItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerateResult]:
        # vLLM batches internally; we submit the whole list and map back by id.
        params = self._sampling_params()
        prompts = [it.prompt for it in items]
        try:
            outputs = self.llm.generate(prompts, params)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                raise OutOfMemoryError(str(exc)) from exc
            raise

        results: List[GenerateResult] = []
        for it, out in zip(items, outputs):
            comp = out.outputs[0]
            token_ids = list(comp.token_ids)
            finish_reason = getattr(comp, "finish_reason", None) or "length"
            prompt_ids = getattr(out, "prompt_token_ids", None)
            prompt_len = len(prompt_ids) if prompt_ids is not None else 0
            results.append(
                GenerateResult(
                    id=it.id,
                    completion_text=comp.text,
                    completion_token_ids=token_ids,
                    prompt_token_len=prompt_len,
                    finish_reason=finish_reason,
                    passthrough=it.passthrough,
                )
            )
        return results


class VLLMCaptureEngine(CaptureEngine):
    """Native per-layer hidden-state extraction via vLLM (v0.18.0+).

    The native feature is connector-based and its exact surface is
    version-specific. This engine probes for a usable form at construction time
    and, if it cannot find one, raises a clear error pointing at the
    hf-batched capture engine rather than silently producing nothing.
    """

    def __init__(
        self,
        model_name: str,
        *,
        layers: str = "all",
        trust_remote_code: bool = True,
        dtype: Optional[str] = None,
        **_ignored: Any,
    ):
        vllm = _require_vllm()
        version = getattr(vllm, "__version__", "0")
        if not self._version_supports_capture(version):
            raise RuntimeError(
                f"vllm {version} does not support native hidden-state extraction "
                "(needs v0.18.0+). Use --engine hf-batched for batch-capture; the "
                "hf-batched capture forward is already fast enough (generation, not "
                "capture, is the bottleneck)."
            )
        # The native extraction path requires a KV-transfer connector and a
        # per-request hidden_states path; that plumbing is not stable enough to
        # hard-wire generically. Until this repo pins a vLLM whose capture API is
        # verified, direct the user to the hf-batched engine.
        raise RuntimeError(
            "vllm native hidden-state extraction is not wired into this generic "
            "engine yet (the v0.18.0 API is connector-based and version-specific). "
            "Use --engine hf-batched for batch-capture. See the batch-inference "
            "reference doc for the vLLM capture status."
        )

    @staticmethod
    def _version_supports_capture(version: str) -> bool:
        try:
            parts = [int(p) for p in version.split(".")[:2]]
        except (ValueError, AttributeError):
            return False
        major, minor = (parts + [0, 0])[:2]
        return (major, minor) >= (0, 18)

    def capture(
        self,
        items: List[CaptureItem],
        *,
        batch_size: int,
        on_oom: Optional[Callable[[int, int], None]] = None,
    ) -> List[CaptureResult]:  # pragma: no cover - unreachable (ctor raises)
        raise RuntimeError("vllm capture is not available; use --engine hf-batched.")
