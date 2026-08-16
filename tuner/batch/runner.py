"""Batch run orchestration: engines + persistence + sync, both verbs.

Location: tuner/batch/runner.py
Purpose: The generic driver shared by the two batch handlers. Reads the input
    JSONL, filters out ids already done (resume), processes the remainder in
    micro-batches, and after EACH batch flushes artifacts + updates the
    checkpoint + fires the sync hook. A killed run resumed with the same
    out-dir produces the identical artifact set as an uninterrupted run.
Used by: tuner.handlers.batch_generate_handler / batch_capture_handler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from tuner.batch.engines import (
    CaptureItem,
    GenerateItem,
    get_capture_engine,
    get_generate_engine,
)
from tuner.batch.persistence import (
    ConfigMismatchError,
    JsonlAppender,
    RunCheckpoint,
    atomic_write_bytes,
    read_jsonl_ids,
    sanitize_id,
)
from tuner.batch.gpu_telemetry import peak_suffix, reset_peak
from tuner.batch.sync_hook import SyncHook
from tuner.project import ProjectContext


COMPLETIONS_FILENAME = "completions.jsonl"
PROVENANCE_FILENAME = "provenance.json"
CAPTURE_INDEX_FILENAME = "capture.jsonl"
_RESERVED_ROW_FIELDS = {"id", "prompt", "text", "token_ids", "positions"}


def _context_identity(context: ProjectContext | None) -> Dict[str, Any] | None:
    if context is None:
        return None
    manifest_sha256 = None
    if context.manifest_path and context.manifest_path.is_file():
        manifest_sha256 = hashlib.sha256(context.manifest_path.read_bytes()).hexdigest()
    return {
        "mode": context.mode,
        "path_mode": context.path_mode,
        "engine_root": str(context.engine_root),
        "project_root": str(context.project_root),
        "manifest_sha256": manifest_sha256,
    }


def _runtime_dirs(
    out_dir: Path, context: ProjectContext | None
) -> tuple[Path, Path | None]:
    resolved = out_dir.resolve(strict=False)
    if context is None or context.mode == "standalone":
        return resolved, None
    artifact_root = context.artifact_root.resolve(strict=False)
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(
            f"Batch output directory must be below the project artifact root: {artifact_root}"
        )
    identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:20]
    return resolved, context.state_root / "batch" / identity


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _passthrough(row: Dict[str, Any]) -> Dict[str, Any]:
    """Every input field that is not a reserved schema field, passed untouched."""
    return {k: v for k, v in row.items() if k not in _RESERVED_ROW_FIELDS}


def _chunks(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_batch_generate(
    *,
    prompts_path: Path,
    out_dir: Path,
    model: str,
    model_revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    engine: str = "hf-batched",
    max_new_tokens: int = 48,
    min_new_tokens: int = 0,
    batch_size: int = 16,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    extra_eos_tokens: Optional[List[str]] = None,
    stop: Optional[List[str]] = None,
    suppress_tokens: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    structured_output_backend: str = "auto",
    structured_output_disable_any_whitespace: bool = False,
    expected_vllm_version: Optional[str] = None,
    vllm_model_runner: Optional[str] = None,
    min_compute_capability: Optional[str] = None,
    tensor_parallel_size: int = 1,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    max_model_len: Optional[int] = None,
    limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    gpu_memory_utilization: Optional[float] = None,
    resume: bool = False,
    sync_every: int = 0,
    sync_cmd: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    context: ProjectContext | None = None,
) -> Dict[str, Any]:
    """Run ``batch-generate`` with incremental persistence + resume.

    Returns a small summary dict (counts + artifact paths).
    """
    log = log or (lambda m: print(m))
    out_dir, state_dir = _runtime_dirs(Path(out_dir), context)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(prompts_path))
    prompts_sha256 = hashlib.sha256(Path(prompts_path).read_bytes()).hexdigest()
    prompt_by_id: Dict[str, str] = {}
    for r in rows:
        if "id" not in r or "prompt" not in r:
            raise ValueError("Each prompts row must have 'id' and 'prompt' fields.")
        row_id = str(r["id"])
        if row_id in prompt_by_id:
            raise ValueError(f"Duplicate prompt id: {row_id!r}")
        if not isinstance(r["prompt"], str):
            raise ValueError(f"Prompt for id {row_id!r} must be a string.")
        prompt_by_id[row_id] = r["prompt"]
    if json_schema is not None and not isinstance(json_schema, dict):
        raise ValueError("json_schema must be a JSON object")
    if json_schema is not None and engine != "vllm":
        raise ValueError("json_schema structured outputs require engine='vllm'")
    if engine != "vllm" and structured_output_backend != "auto":
        raise ValueError("structured_output_backend requires engine='vllm'")
    if suppress_tokens and engine != "vllm":
        raise ValueError("suppress_tokens require engine='vllm'")
    if engine != "vllm" and max_model_len is not None:
        raise ValueError("max_model_len requires engine='vllm'")
    if engine != "vllm" and limit_mm_per_prompt is not None:
        raise ValueError("limit_mm_per_prompt requires engine='vllm'")
    if structured_output_backend not in {"auto", "xgrammar"}:
        raise ValueError(
            "structured_output_backend must be one of: auto, xgrammar"
        )
    if max_model_len is not None and max_model_len < 1:
        raise ValueError("max_model_len must be at least 1")
    if limit_mm_per_prompt is not None:
        if not isinstance(limit_mm_per_prompt, dict):
            raise ValueError("limit_mm_per_prompt must be a JSON object")
        if any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in limit_mm_per_prompt.items()
        ):
            raise ValueError(
                "limit_mm_per_prompt must map modality names to non-negative integers"
            )
    if gpu_memory_utilization is not None and not (
        0.0 < gpu_memory_utilization <= 1.0
    ):
        raise ValueError("gpu_memory_utilization must be in the interval (0, 1]")

    schema_hash = None
    if json_schema is not None:
        schema_bytes = json.dumps(
            json_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        schema_hash = hashlib.sha256(schema_bytes).hexdigest()

    # Config hash: everything that changes WHAT is produced. Not out-dir, resume,
    # or sync (those don't affect output content), matching the resume contract.
    config = {
        "verb": "batch-generate",
        "prompts_sha256": prompts_sha256,
        "model": model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "engine": engine,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "seed": seed,
        "extra_eos_tokens": list(extra_eos_tokens) if extra_eos_tokens else None,
        "stop": list(stop) if stop else None,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "json_schema_sha256": schema_hash,
        "structured_output_backend": (
            structured_output_backend if engine == "vllm" else None
        ),
        "structured_output_disable_any_whitespace": (
            structured_output_disable_any_whitespace if engine == "vllm" else None
        ),
        "expected_vllm_version": expected_vllm_version,
        "vllm_model_runner": vllm_model_runner if engine == "vllm" else None,
        "min_compute_capability": (
            min_compute_capability if engine == "vllm" else None
        ),
        "vllm_batch_invariant": engine == "vllm",
        "tensor_parallel_size": tensor_parallel_size if engine == "vllm" else None,
        "max_num_seqs": max_num_seqs if engine == "vllm" else None,
        "max_num_batched_tokens": max_num_batched_tokens if engine == "vllm" else None,
        "max_model_len": max_model_len if engine == "vllm" else None,
        "limit_mm_per_prompt": limit_mm_per_prompt if engine == "vllm" else None,
        "gpu_memory_utilization": (
            gpu_memory_utilization if engine == "vllm" else None
        ),
        "project_context": _context_identity(context),
    }
    if suppress_tokens:
        config["suppress_tokens"] = list(suppress_tokens)

    completions_path = out_dir / COMPLETIONS_FILENAME
    index_ids = read_jsonl_ids(completions_path, id_field="id")
    checkpoint = RunCheckpoint.load_or_create(
        out_dir, config, resume=resume, index_ids=index_ids, state_dir=state_dir
    )
    appender = JsonlAppender(completions_path)
    sync = SyncHook(out_dir, sync_cmd, sync_every, state_dir=state_dir, warn=log)

    todo = [r for r in rows if not checkpoint.is_done(r["id"])]
    log(
        f"[batch-generate] {len(rows)} rows total, {len(rows) - len(todo)} already "
        f"done, {len(todo)} to process (engine={engine}, batch_size={batch_size})."
    )
    if not todo:
        provenance_path = out_dir / PROVENANCE_FILENAME
        if index_ids and not provenance_path.exists():
            raise ConfigMismatchError(
                "Cannot --resume: completed rows exist without provenance.json."
            )
        if provenance_path.exists():
            existing_provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            if (
                existing_provenance.get("config_hash") != checkpoint.config_hash
                or existing_provenance.get("config") != config
            ):
                raise ConfigMismatchError(
                    "Cannot --resume: static provenance differs from the current run."
                )
        sync.final()
        summary = _summary(out_dir, completions_path, len(rows), 0, engine)
        summary["runtime_provenance_verified"] = False
        return summary

    engine_kwargs = {
        "model_name": model,
        "revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "extra_eos_tokens": extra_eos_tokens,
        "stop": stop,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
    }
    if engine == "vllm":
        engine_kwargs.update(
            json_schema=json_schema,
            structured_output_backend=structured_output_backend,
            structured_output_disable_any_whitespace=(
                structured_output_disable_any_whitespace
            ),
            expected_vllm_version=expected_vllm_version,
            vllm_model_runner=vllm_model_runner,
            min_compute_capability=min_compute_capability,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            limit_mm_per_prompt=limit_mm_per_prompt,
            gpu_memory_utilization=gpu_memory_utilization,
            suppress_tokens=suppress_tokens,
        )
    gen_engine = get_generate_engine(engine, **engine_kwargs)
    provenance = {
        "version": 1,
        "config_hash": checkpoint.config_hash,
        "config": config,
        "runtime": gen_engine.provenance(),
    }
    provenance_path = out_dir / PROVENANCE_FILENAME
    if provenance_path.exists():
        existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing_provenance != provenance:
            gen_engine.close()
            raise ConfigMismatchError(
                "Cannot --resume: runtime provenance differs from the existing "
                "run. vLLM resume requires the same version and hardware."
            )
    else:
        if index_ids:
            gen_engine.close()
            raise ConfigMismatchError(
                "Cannot --resume: completed rows exist without provenance.json."
            )
        atomic_write_bytes(
            provenance_path,
            (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def _on_oom(old: int, new: int) -> None:
        log(f"[batch-generate] CUDA OOM at batch_size={old}; halving to {new}.")

    # Reset CUDA peak-memory stats so the reported peak reflects THIS stage only.
    # No-op on CPU.
    reset_peak()

    processed = 0
    try:
        for chunk in _chunks(todo, max(1, batch_size)):
            items = [
                GenerateItem(id=str(r["id"]), prompt=r["prompt"], passthrough=_passthrough(r))
                for r in chunk
            ]
            results = gen_engine.generate(items, batch_size=batch_size, on_oom=_on_oom)
            out_rows = []
            for res in results:
                row = {
                    "id": res.id,
                    "completion_text": res.completion_text,
                    "completion_token_ids": res.completion_token_ids,
                    "prompt_token_ids_sha256": res.prompt_token_ids_sha256,
                    "prompt_token_len": res.prompt_token_len,
                    "prompt_sha256": hashlib.sha256(
                        prompt_by_id[res.id].encode("utf-8")
                    ).hexdigest(),
                    "finish_reason": res.finish_reason,
                }
                row.update(res.passthrough)
                out_rows.append(row)
            appender.append_many(out_rows)
            checkpoint.mark_done(res.id for res in results)
            processed += len(results)
            sync.note_rows(len(results))
            log(f"[batch-generate] persisted {processed}/{len(todo)} new rows.{peak_suffix()}")
    finally:
        gen_engine.close()

    sync.final()
    summary = _summary(
        out_dir, completions_path, len(rows), processed, engine, peak_suffix()
    )
    summary["runtime_provenance_verified"] = True
    return summary


def run_batch_capture(
    *,
    rows_path: Path,
    out_dir: Path,
    model: str,
    model_revision: Optional[str] = None,
    engine: str = "hf-batched",
    layers: str = "all",
    batch_size: int = 16,
    persist_dtype: str = "float32",
    resume: bool = False,
    sync_every: int = 0,
    sync_cmd: Optional[str] = None,
    trust_remote_code: bool = True,
    dtype: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    engine_overrides: Optional[Dict[str, Any]] = None,
    context: ProjectContext | None = None,
) -> Dict[str, Any]:
    """Run ``batch-capture`` with incremental per-row safetensors + resume."""
    log = log or (lambda m: print(m))
    out_dir, state_dir = _runtime_dirs(Path(out_dir), context)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    if persist_dtype not in ("float32", "bfloat16"):
        raise ValueError("persist_dtype must be float32 or bfloat16")

    rows = _read_jsonl(Path(rows_path))
    for r in rows:
        if "id" not in r:
            raise ValueError("Each capture row must have an 'id' field.")
        if "text" not in r and "token_ids" not in r:
            raise ValueError("Each capture row must have 'text' or 'token_ids'.")
        if "positions" not in r or not isinstance(r["positions"], dict):
            raise ValueError("Each capture row must have a 'positions' object.")

    config = {
        "verb": "batch-capture",
        "model": model,
        "model_revision": model_revision,
        "engine": engine,
        "layers": layers,
        "persist_dtype": persist_dtype,
        "dtype": dtype,
        "project_context": _context_identity(context),
    }

    index_path = out_dir / CAPTURE_INDEX_FILENAME
    index_ids = read_jsonl_ids(index_path, id_field="id")
    checkpoint = RunCheckpoint.load_or_create(
        out_dir, config, resume=resume, index_ids=index_ids, state_dir=state_dir
    )
    appender = JsonlAppender(index_path)
    sync = SyncHook(out_dir, sync_cmd, sync_every, state_dir=state_dir, warn=log)

    todo = [r for r in rows if not checkpoint.is_done(r["id"])]
    log(
        f"[batch-capture] {len(rows)} rows total, {len(rows) - len(todo)} already "
        f"done, {len(todo)} to process (engine={engine}, batch_size={batch_size})."
    )
    if not todo:
        sync.final()
        return _summary(out_dir, index_path, len(rows), 0, engine)

    cap_engine = get_capture_engine(
        engine,
        model_name=model,
        revision=model_revision,
        layers=layers,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        **(engine_overrides or {}),
    )

    def _on_oom(old: int, new: int) -> None:
        log(f"[batch-capture] CUDA OOM at batch_size={old}; halving to {new}.")

    # Reset CUDA peak-memory stats so the reported peak reflects THIS stage only.
    # No-op on CPU.
    reset_peak()

    processed = 0
    try:
        for chunk in _chunks(todo, max(1, batch_size)):
            items = [
                CaptureItem(
                    id=str(r["id"]),
                    positions=r["positions"],
                    text=r.get("text"),
                    token_ids=r.get("token_ids"),
                    passthrough=_passthrough(r),
                )
                for r in chunk
            ]
            results = cap_engine.capture(items, batch_size=batch_size, on_oom=_on_oom)
            index_rows = []
            for res in results:
                stem = sanitize_id(res.id)
                filename = f"{stem}.safetensors"
                _write_safetensors(
                    tensors_dir / filename, res.tensors, persist_dtype
                )
                index_row = {
                    "id": res.id,
                    "file": f"tensors/{filename}",
                    "n_layers": res.n_layers,
                    "hidden_dim": res.hidden_dim,
                    "positions": res.positions,
                }
                index_row.update(res.passthrough)
                index_rows.append(index_row)
            appender.append_many(index_rows)
            checkpoint.mark_done(res.id for res in results)
            processed += len(results)
            sync.note_rows(len(results))
            log(f"[batch-capture] persisted {processed}/{len(todo)} new rows.{peak_suffix()}")
    finally:
        cap_engine.close()

    sync.final()
    return _summary(out_dir, index_path, len(rows), processed, engine, peak_suffix())


def _write_safetensors(path: Path, tensors: Dict[str, Any], persist_dtype: str) -> None:
    """Persist a row's named tensors to a single safetensors file, atomically."""
    import torch
    from safetensors.torch import save

    dtype = torch.float32 if persist_dtype == "float32" else torch.bfloat16
    typed = {}
    for key, vec in tensors.items():
        t = vec if isinstance(vec, torch.Tensor) else torch.as_tensor(vec)
        typed[key] = t.to(dtype).contiguous()
    atomic_write_bytes(path, save(typed))


def _summary(
    out_dir: Path,
    artifact: Path,
    total: int,
    processed: int,
    engine: str,
    gpu_peak_suffix: str = "",
) -> Dict[str, Any]:
    return {
        "out_dir": str(out_dir),
        "artifact": str(artifact),
        "total_rows": total,
        "newly_processed": processed,
        "engine": engine,
        # A `` (gpu peak X.X/Y.Y GiB)`` string for the completion log line, or ""
        # on CPU / when no stage ran. Handlers append it verbatim.
        "gpu_peak_suffix": gpu_peak_suffix,
    }
