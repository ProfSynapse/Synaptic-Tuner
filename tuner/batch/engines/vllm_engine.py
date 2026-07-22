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

import os
import subprocess
from typing import Any, Callable, List, Optional

from tuner.batch.engines.base import (
    CaptureEngine,
    CaptureItem,
    CaptureResult,
    GenerateEngine,
    GenerateItem,
    GenerateResult,
    OutOfMemoryError,
    hash_token_ids,
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


def _parse_version_pair(value: str, *, label: str) -> tuple[int, int]:
    try:
        major_text, minor_text = value.split(".", 2)[:2]
        return int(major_text), int(minor_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must begin with MAJOR.MINOR") from exc


def _documented_batch_invariance_floor(vllm_version: str) -> tuple[int, int]:
    """Return the official architecture floor for supported vLLM releases."""
    version = _parse_version_pair(vllm_version, label="expected_vllm_version")
    if version >= (0, 23):
        return (8, 0)
    if version >= (0, 18):
        return (9, 0)
    raise ValueError(
        "This batch-invariance engine supports vLLM 0.18.0 or newer only."
    )


def _require_batch_invariant_hardware(
    minimum: tuple[int, int], tensor_parallel_size: int
) -> dict:
    """Fail before model construction unless every participating GPU qualifies."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "vLLM batch invariance requires CUDA hardware, but CUDA is unavailable."
        )
    visible = int(torch.cuda.device_count())
    if visible < tensor_parallel_size:
        raise RuntimeError(
            f"tensor_parallel_size={tensor_parallel_size} requires that many visible "
            f"GPUs, but only {visible} are available."
        )
    devices = []
    for index in range(tensor_parallel_size):
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
        if capability < minimum:
            found = f"{capability[0]}.{capability[1]}"
            required_text = f"{minimum[0]}.{minimum[1]}"
            raise RuntimeError(
                f"GPU {index} has compute capability {found}, below the pinned "
                f"batch-invariance minimum {required_text}."
            )
        devices.append(
            {
                "index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "compute_capability": f"{capability[0]}.{capability[1]}",
            }
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver_versions = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
    except (OSError, subprocess.SubprocessError):
        driver_versions = []
    return {
        "devices": devices,
        "nvidia_driver_versions": driver_versions,
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_version": getattr(torch, "__version__", None),
    }


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
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        trust_remote_code: bool = False,
        dtype: Optional[str] = None,
        json_schema: Optional[dict] = None,
        structured_output_backend: str = "auto",
        structured_output_disable_any_whitespace: bool = False,
        expected_vllm_version: Optional[str] = None,
        min_compute_capability: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
        limit_mm_per_prompt: Optional[dict] = None,
        gpu_memory_utilization: Optional[float] = None,
        **_ignored: Any,
    ):
        if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
            raise RuntimeError(
                "--engine vllm requires VLLM_BATCH_INVARIANT=1 to be set before "
                "engine construction. Refusing a batch-order-sensitive run."
            )
        if not expected_vllm_version:
            raise ValueError(
                "--engine vllm requires --expected-vllm-version so runtime "
                "provenance cannot drift silently."
            )
        if not min_compute_capability:
            raise ValueError(
                "--engine vllm requires --min-compute-capability so the batch-"
                "invariance hardware requirement is checked before model load."
            )
        vllm = _require_vllm()
        installed_version = getattr(vllm, "__version__", None)
        if installed_version != expected_vllm_version:
            raise RuntimeError(
                "vLLM version mismatch: expected "
                f"{expected_vllm_version}, found {installed_version or 'unknown'}."
            )
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        requested_floor = _parse_version_pair(
            min_compute_capability, label="min_compute_capability"
        )
        documented_floor = _documented_batch_invariance_floor(installed_version)
        effective_floor = max(requested_floor, documented_floor)
        self._documented_compute_capability_floor = (
            f"{documented_floor[0]}.{documented_floor[1]}"
        )
        self._effective_compute_capability_floor = (
            f"{effective_floor[0]}.{effective_floor[1]}"
        )
        self._hardware = _require_batch_invariant_hardware(
            effective_floor, tensor_parallel_size
        )
        from vllm import LLM, SamplingParams
        try:
            from vllm.config import StructuredOutputsConfig
        except ImportError as exc:
            raise RuntimeError(
                f"vLLM {installed_version} does not expose "
                "StructuredOutputsConfig; use a verified vLLM release."
            ) from exc

        self._SamplingParams = SamplingParams
        self._vllm_version = installed_version
        self._json_schema = json_schema
        self._structured_output_backend = structured_output_backend
        self._structured_output_disable_any_whitespace = bool(
            structured_output_disable_any_whitespace
        )
        self._structured_outputs = None
        if json_schema is not None:
            try:
                from vllm.sampling_params import StructuredOutputsParams
            except ImportError as exc:
                raise RuntimeError(
                    f"vLLM {installed_version} does not expose "
                    "StructuredOutputsParams; use a verified vLLM release."
                ) from exc
            self._structured_outputs = StructuredOutputsParams(json=json_schema)

        llm_kwargs: dict = {"model": model_name, "trust_remote_code": trust_remote_code}
        if structured_output_backend not in {"auto", "xgrammar"}:
            raise ValueError(
                "structured_output_backend must be one of: auto, xgrammar"
            )
        llm_kwargs["structured_outputs_config"] = StructuredOutputsConfig(
            backend=structured_output_backend,
            disable_any_whitespace=self._structured_output_disable_any_whitespace,
        )
        if revision:
            llm_kwargs["revision"] = revision
        if tokenizer_revision:
            llm_kwargs["tokenizer_revision"] = tokenizer_revision
        if dtype:
            llm_kwargs["dtype"] = dtype
        if seed is not None:
            llm_kwargs["seed"] = seed
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
        if max_num_seqs is not None:
            if max_num_seqs < 1:
                raise ValueError("max_num_seqs must be at least 1")
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_num_batched_tokens is not None:
            if max_num_batched_tokens < 1:
                raise ValueError("max_num_batched_tokens must be at least 1")
            llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if max_model_len is not None:
            if max_model_len < 1:
                raise ValueError("max_model_len must be at least 1")
            llm_kwargs["max_model_len"] = int(max_model_len)
        if limit_mm_per_prompt is not None:
            if not isinstance(limit_mm_per_prompt, dict) or any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in limit_mm_per_prompt.items()
            ):
                raise ValueError(
                    "limit_mm_per_prompt must map modality names to "
                    "non-negative integers"
                )
            llm_kwargs["limit_mm_per_prompt"] = dict(limit_mm_per_prompt)
        if gpu_memory_utilization is not None:
            if not 0.0 < gpu_memory_utilization <= 1.0:
                raise ValueError(
                    "gpu_memory_utilization must be in the interval (0, 1]"
                )
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        self.llm = LLM(**llm_kwargs)
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature) if do_sample else 0.0
        self.top_p = float(top_p) if do_sample else 1.0
        self.seed = seed
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
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self._structured_outputs is not None:
            kwargs["structured_outputs"] = self._structured_outputs
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
        results: List[GenerateResult] = []
        index = 0
        current_batch_size = max(1, int(batch_size))
        while index < len(items):
            chunk = items[index : index + current_batch_size]
            try:
                results.extend(self._generate_chunk(chunk))
                index += len(chunk)
            except Exception as exc:  # noqa: BLE001
                if "out of memory" not in str(exc).lower():
                    raise
                if current_batch_size <= 1:
                    raise OutOfMemoryError(str(exc)) from exc
                new_batch_size = max(1, current_batch_size // 2)
                if on_oom is not None:
                    on_oom(current_batch_size, new_batch_size)
                current_batch_size = new_batch_size
        return results

    def _generate_chunk(self, items: List[GenerateItem]) -> List[GenerateResult]:
        params = self._sampling_params()
        prompts = [it.prompt for it in items]
        outputs = self.llm.generate(prompts, params)

        results: List[GenerateResult] = []
        for it, out in zip(items, outputs):
            comp = out.outputs[0]
            token_ids = list(comp.token_ids)
            finish_reason = getattr(comp, "finish_reason", None) or "length"
            prompt_ids = getattr(out, "prompt_token_ids", None)
            if prompt_ids is None:
                raise RuntimeError(
                    "vLLM did not return prompt_token_ids; exact prompt-token "
                    "evidence is required for batch generation."
                )
            prompt_ids = list(prompt_ids)
            prompt_len = len(prompt_ids)
            results.append(
                GenerateResult(
                    id=it.id,
                    completion_text=comp.text,
                    completion_token_ids=token_ids,
                    prompt_token_ids_sha256=hash_token_ids(prompt_ids),
                    prompt_token_len=prompt_len,
                    finish_reason=finish_reason,
                    passthrough=it.passthrough,
                )
            )
        return results

    def provenance(self) -> dict:
        return {
            "vllm_version": self._vllm_version,
            "vllm_batch_invariant": True,
            "structured_outputs": self._json_schema is not None,
            "structured_output_backend": self._structured_output_backend,
            "structured_output_disable_any_whitespace": (
                self._structured_output_disable_any_whitespace
            ),
            "hardware": self._hardware,
            "documented_compute_capability_floor": (
                self._documented_compute_capability_floor
            ),
            "effective_compute_capability_floor": (
                self._effective_compute_capability_floor
            ),
        }


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
        revision: Optional[str] = None,
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
