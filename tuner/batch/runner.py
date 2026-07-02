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
    JsonlAppender,
    RunCheckpoint,
    atomic_write_bytes,
    read_jsonl_ids,
    sanitize_id,
)
from tuner.batch.sync_hook import SyncHook


COMPLETIONS_FILENAME = "completions.jsonl"
CAPTURE_INDEX_FILENAME = "capture.jsonl"
_RESERVED_ROW_FIELDS = {"id", "prompt", "text", "token_ids", "positions"}


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
    engine: str = "hf-batched",
    max_new_tokens: int = 48,
    batch_size: int = 16,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    resume: bool = False,
    sync_every: int = 0,
    sync_cmd: Optional[str] = None,
    trust_remote_code: bool = True,
    dtype: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    engine_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run ``batch-generate`` with incremental persistence + resume.

    Returns a small summary dict (counts + artifact paths).
    """
    log = log or (lambda m: print(m))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(prompts_path))
    for r in rows:
        if "id" not in r or "prompt" not in r:
            raise ValueError("Each prompts row must have 'id' and 'prompt' fields.")

    # Config hash: everything that changes WHAT is produced. Not out-dir, resume,
    # or sync (those don't affect output content), matching the resume contract.
    config = {
        "verb": "batch-generate",
        "model": model,
        "engine": engine,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "seed": seed,
        "stop": list(stop) if stop else None,
        "dtype": dtype,
    }

    completions_path = out_dir / COMPLETIONS_FILENAME
    index_ids = read_jsonl_ids(completions_path, id_field="id")
    checkpoint = RunCheckpoint.load_or_create(
        out_dir, config, resume=resume, index_ids=index_ids
    )
    appender = JsonlAppender(completions_path)
    sync = SyncHook(out_dir, sync_cmd, sync_every, warn=log)

    todo = [r for r in rows if not checkpoint.is_done(r["id"])]
    log(
        f"[batch-generate] {len(rows)} rows total, {len(rows) - len(todo)} already "
        f"done, {len(todo)} to process (engine={engine}, batch_size={batch_size})."
    )
    if not todo:
        sync.final()
        return _summary(out_dir, completions_path, len(rows), 0, engine)

    gen_engine = get_generate_engine(
        engine,
        model_name=model,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=stop,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        **(engine_overrides or {}),
    )

    def _on_oom(old: int, new: int) -> None:
        log(f"[batch-generate] CUDA OOM at batch_size={old}; halving to {new}.")

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
                    "prompt_token_len": res.prompt_token_len,
                    "finish_reason": res.finish_reason,
                }
                row.update(res.passthrough)
                out_rows.append(row)
            appender.append_many(out_rows)
            checkpoint.mark_done(res.id for res in results)
            processed += len(results)
            sync.note_rows(len(results))
            log(f"[batch-generate] persisted {processed}/{len(todo)} new rows.")
    finally:
        gen_engine.close()

    sync.final()
    return _summary(out_dir, completions_path, len(rows), processed, engine)


def run_batch_capture(
    *,
    rows_path: Path,
    out_dir: Path,
    model: str,
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
) -> Dict[str, Any]:
    """Run ``batch-capture`` with incremental per-row safetensors + resume."""
    log = log or (lambda m: print(m))
    out_dir = Path(out_dir)
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
        "engine": engine,
        "layers": layers,
        "persist_dtype": persist_dtype,
        "dtype": dtype,
    }

    index_path = out_dir / CAPTURE_INDEX_FILENAME
    index_ids = read_jsonl_ids(index_path, id_field="id")
    checkpoint = RunCheckpoint.load_or_create(
        out_dir, config, resume=resume, index_ids=index_ids
    )
    appender = JsonlAppender(index_path)
    sync = SyncHook(out_dir, sync_cmd, sync_every, warn=log)

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
        layers=layers,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        **(engine_overrides or {}),
    )

    def _on_oom(old: int, new: int) -> None:
        log(f"[batch-capture] CUDA OOM at batch_size={old}; halving to {new}.")

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
            log(f"[batch-capture] persisted {processed}/{len(todo)} new rows.")
    finally:
        cap_engine.close()

    sync.final()
    return _summary(out_dir, index_path, len(rows), processed, engine)


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
    out_dir: Path, artifact: Path, total: int, processed: int, engine: str
) -> Dict[str, Any]:
    return {
        "out_dir": str(out_dir),
        "artifact": str(artifact),
        "total_rows": total,
        "newly_processed": processed,
        "engine": engine,
    }
