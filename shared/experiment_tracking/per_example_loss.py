"""
shared/experiment_tracking/per_example_loss.py

Computes per-example cross entropy losses for JSONL chat datasets.

The module keeps the original simple `compute_per_example_losses()` entrypoint,
but now executes in token-budgeted batches and can optionally persist shard
artifacts incrementally so long-running cloud jobs do not lose all loss data if
they crash mid-run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .runtime_autotune import AdaptiveTokenBudget, cuda_headroom_bytes
from .schema import LossResult

logger = logging.getLogger(__name__)


@dataclass
class PreparedLossExample:
    index: int
    jsonl_hash: str
    input_ids: list[int]
    labels: list[int]

    @property
    def num_completion_tokens(self) -> int:
        return sum(1 for label in self.labels if label != -100)

    @property
    def num_total_tokens(self) -> int:
        return len(self.input_ids)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _hash_jsonl_line(line: str) -> str:
    """Returns the first 8 characters of the SHA-256 hash of a stripped JSONL line."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()[:8]


def _render_tool_call_content(tool_calls: list[dict[str, Any]]) -> str:
    """Render OpenAI-style tool_calls into text compatible with chat templates."""
    rendered_parts: list[str] = []
    for tool_call in tool_calls:
        function_payload = tool_call.get("function") or {}
        name = function_payload.get("name") or tool_call.get("name") or "unknown"
        arguments = function_payload.get("arguments", tool_call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments_obj = json.loads(arguments)
            except json.JSONDecodeError:
                arguments_obj = arguments
        else:
            arguments_obj = arguments
        rendered_parts.append(
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": arguments_obj}, ensure_ascii=False, indent=2)
            + "\n</tool_call>"
        )
    return "\n".join(rendered_parts)


def _sanitize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize nullable/tool-call messages into plain text for tokenizer templates."""
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        tool_calls = normalized.get("tool_calls") or []
        if tool_calls:
            tool_content = _render_tool_call_content(tool_calls)
            content = f"{content}\n\n{tool_content}".strip() if content else tool_content

        normalized["content"] = content
        normalized.pop("tool_calls", None)
        sanitized.append(normalized)
    return sanitized


def _prepare_loss_example(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    line_hash: str,
    index: int,
    max_seq_length: int,
    completion_only: bool,
) -> Optional[PreparedLossExample]:
    conv = record.get("conversations", [])
    if not conv:
        conv = record.get("messages", [])
    if not conv:
        return None

    conv = _sanitize_messages_for_chat_template(conv)

    if completion_only and len(conv) > 0 and conv[-1].get("role") == "assistant":
        prompt_messages = conv[:-1]
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)

        full_str = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        full_tokens = tokenizer.encode(full_str, add_special_tokens=False)

        input_ids = list(full_tokens[:max_seq_length])
        labels = list(input_ids)

        mask_len = min(len(prompt_tokens), len(labels))
        for j in range(mask_len):
            if labels[j] == prompt_tokens[j]:
                labels[j] = -100
            else:
                break
    else:
        full_str = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer.encode(full_str, add_special_tokens=False)[:max_seq_length]
        labels = list(input_ids)

    if not input_ids:
        return None

    return PreparedLossExample(
        index=index,
        jsonl_hash=line_hash,
        input_ids=list(input_ids),
        labels=labels,
    )


def _iter_prepared_examples(
    *,
    tokenizer: Any,
    dataset_path: Path,
    max_seq_length: int,
    completion_only: bool,
    start_index: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Iterator[PreparedLossExample]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start_index:
                continue
            if shard_count > 1 and index % shard_count != shard_index:
                continue
            line_hash = _hash_jsonl_line(line)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            prepared = _prepare_loss_example(
                tokenizer=tokenizer,
                record=record,
                line_hash=line_hash,
                index=index,
                max_seq_length=max_seq_length,
                completion_only=completion_only,
            )
            if prepared is not None:
                yield prepared


def _projected_batch_tokens(current: list[PreparedLossExample], candidate: PreparedLossExample) -> int:
    lengths = [len(item.input_ids) for item in current]
    max_len = max(lengths + [len(candidate.input_ids)])
    return max_len * (len(current) + 1)


def _iter_batches(
    examples: Iterator[PreparedLossExample],
    *,
    max_batch_tokens: int,
    max_batch_size: int | None = None,
) -> Iterator[list[PreparedLossExample]]:
    batch: list[PreparedLossExample] = []
    for example in examples:
        if batch:
            exceeds_tokens = _projected_batch_tokens(batch, example) > max_batch_tokens
            exceeds_batch_size = max_batch_size is not None and len(batch) >= max_batch_size
            if exceeds_tokens or exceeds_batch_size:
                yield batch
                batch = []
        batch.append(example)
    if batch:
        yield batch


def _pad_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    return int(token_id) if token_id is not None else 0


def _compute_batch_losses(
    *,
    model: Any,
    batch: list[PreparedLossExample],
    device: str | torch.device,
    pad_token_id: int,
) -> list[LossResult]:
    seq_len = max(len(item.input_ids) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for row_idx, item in enumerate(batch):
        item_len = len(item.input_ids)
        input_ids[row_idx, :item_len] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row_idx, :item_len] = torch.tensor(item.labels, dtype=torch.long)
        attention_mask[row_idx, :item_len] = 1

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model outputs do not contain logits for per-example loss computation.")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(batch_size, -1)

        valid_mask = shift_labels.ne(-100)
        valid_counts = valid_mask.sum(dim=1)
        summed_losses = token_losses.sum(dim=1)
        per_example = torch.where(
            valid_counts > 0,
            summed_losses / valid_counts.clamp(min=1),
            torch.zeros_like(summed_losses),
        )

    results: list[LossResult] = []
    for item, loss_value in zip(batch, per_example.tolist(), strict=False):
        results.append(
            LossResult(
                index=item.index,
                loss=float(loss_value),
                num_completion_tokens=item.num_completion_tokens,
                num_total_tokens=item.num_total_tokens,
                jsonl_hash=item.jsonl_hash,
            )
        )
    return results


class IncrementalLossWriter:
    """Persist loss shards and partial summaries as batches complete."""

    def __init__(self, output_root: Path | str, *, top_k: int = 25, state_stride: int = 1) -> None:
        self.output_root = Path(output_root)
        self.shards_dir = self.output_root / "shards"
        self.partial_dir = self.output_root / "partial"
        self.manifests_dir = self.output_root / "manifests"
        self.final_losses_path = self.output_root / "per_example_losses.jsonl"
        self.summary_path = self.output_root / "loss_summary.json"
        self.partial_summary_path = self.partial_dir / "loss_summary.partial.json"
        self.partial_high_loss_path = self.partial_dir / "high_loss_examples.partial.jsonl"
        self.manifest_path = self.manifests_dir / "loss_state.json"
        self.top_k = top_k
        self.state_stride = max(int(state_stride), 1)
        self.state = self._load_manifest()
        self._top_loss_rows = self._load_top_loss_rows()

    @property
    def next_index(self) -> int:
        return int(self.state.get("next_index", 0))

    @property
    def is_complete(self) -> bool:
        return bool(self.state.get("completed", False))

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "completed": False,
            "next_index": 0,
            "rows_written": 0,
            "batch_count": 0,
            "loss_sum": 0.0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_loss": None,
            "shards": [],
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._default_state()
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load loss manifest at %s; starting fresh.", self.manifest_path)
            return self._default_state()

    def _load_top_loss_rows(self) -> list[dict[str, Any]]:
        if not self.partial_high_loss_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.partial_high_loss_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    def _persist_manifest(self) -> None:
        _atomic_write_text(self.manifest_path, json.dumps(self.state, indent=2, ensure_ascii=False) + "\n")

    def _persist_partials(self) -> None:
        rows_written = int(self.state.get("rows_written", 0))
        summary = {
            "rows_written": rows_written,
            "batch_count": int(self.state.get("batch_count", 0)),
            "next_index": int(self.state.get("next_index", 0)),
            "completed": bool(self.state.get("completed", False)),
            "mean_loss": (float(self.state.get("loss_sum", 0.0)) / rows_written) if rows_written else 0.0,
            "completion_tokens": int(self.state.get("completion_tokens", 0)),
            "total_tokens": int(self.state.get("total_tokens", 0)),
            "max_loss": self.state.get("max_loss"),
        }
        _atomic_write_text(self.partial_summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        _write_jsonl_rows(self.partial_high_loss_path, self._top_loss_rows)

    def write_batch(self, losses: list[LossResult]) -> None:
        if not losses:
            return
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        start_index = losses[0].index
        end_index = losses[-1].index
        shard_path = self.shards_dir / f"losses-{start_index:08d}-{end_index:08d}.jsonl"
        save_losses(losses, shard_path)

        rows = [asdict(item) for item in losses]
        self._top_loss_rows.extend(rows)
        self._top_loss_rows = sorted(self._top_loss_rows, key=lambda row: row["loss"], reverse=True)[: self.top_k]

        self.state["rows_written"] = int(self.state.get("rows_written", 0)) + len(losses)
        self.state["batch_count"] = int(self.state.get("batch_count", 0)) + 1
        self.state["next_index"] = end_index + self.state_stride
        self.state["loss_sum"] = float(self.state.get("loss_sum", 0.0)) + sum(item.loss for item in losses)
        self.state["completion_tokens"] = int(self.state.get("completion_tokens", 0)) + sum(
            item.num_completion_tokens for item in losses
        )
        self.state["total_tokens"] = int(self.state.get("total_tokens", 0)) + sum(item.num_total_tokens for item in losses)
        batch_max_loss = max(item.loss for item in losses)
        existing_max_loss = self.state.get("max_loss")
        self.state["max_loss"] = batch_max_loss if existing_max_loss is None else max(existing_max_loss, batch_max_loss)
        self.state.setdefault("shards", []).append(
            {
                "path": str(shard_path.relative_to(self.output_root)),
                "start_index": start_index,
                "end_index": end_index,
                "rows": len(losses),
            }
        )
        self._persist_manifest()
        self._persist_partials()

    def finalize(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        shard_entries = self.state.get("shards", [])
        shard_paths = [
            self.output_root / entry["path"]
            for entry in sorted(shard_entries, key=lambda item: int(item["start_index"]))
        ]
        with self.final_losses_path.open("w", encoding="utf-8") as handle:
            for shard_path in shard_paths:
                if not shard_path.exists():
                    continue
                handle.write(shard_path.read_text(encoding="utf-8"))

        self.state["completed"] = True
        self._persist_manifest()
        self._persist_partials()
        summary_payload = json.loads(self.partial_summary_path.read_text(encoding="utf-8"))
        _atomic_write_text(self.summary_path, json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_per_example_losses(
    model: Any,
    tokenizer: Any,
    dataset_path: Path | str,
    max_seq_length: int = 2048,
    completion_only: bool = True,
    device: str | torch.device | None = None,
    batch_max_tokens: int = 8192,
    max_batch_size: int | None = None,
    writer: IncrementalLossWriter | None = None,
    on_batch: Callable[[list[LossResult], IncrementalLossWriter | None], None] | None = None,
    adaptive_batching: bool = True,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[LossResult]:
    """Compute per-example loss for each record in a JSONL dataset.

    Args:
        model: Hugging Face-compatible model in evaluation mode.
        tokenizer: Tokenizer with chat template.
        dataset_path: Path to JSONL dataset.
        max_seq_length: Truncation threshold.
        completion_only: If True, mask everything before the assistant completion.
        device: Device to place tensors on. Defaults to model device.
        batch_max_tokens: Approximate maximum padded tokens per microbatch.
        max_batch_size: Optional hard cap on examples per microbatch.
        writer: Optional incremental shard writer.
        on_batch: Optional callback invoked after each completed batch write.
        adaptive_batching: Dynamically adjust token budget using runtime telemetry.

    Returns:
        List of LossResult objects in dataset order.
    """
    if device is None:
        device = next(model.parameters()).device

    dataset_path = Path(dataset_path)
    start_index = writer.next_index if writer is not None else 0
    prepared_examples = _iter_prepared_examples(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        max_seq_length=max_seq_length,
        completion_only=completion_only,
        start_index=start_index,
        shard_index=shard_index,
        shard_count=shard_count,
    )

    pad_token_id = _pad_token_id(tokenizer)
    controller = AdaptiveTokenBudget(initial_tokens=batch_max_tokens) if adaptive_batching else None
    results: list[LossResult] = []
    pending_batch: list[PreparedLossExample] = []

    def _flush_batch(batch: list[PreparedLossExample]) -> None:
        if not batch:
            return
        nonlocal results
        start_time = time.perf_counter()
        batch_losses = _compute_batch_losses(
            model=model,
            batch=batch,
            device=device,
            pad_token_id=pad_token_id,
        )
        elapsed = time.perf_counter() - start_time
        if writer is not None:
            writer.write_batch(batch_losses)
        if on_batch is not None:
            on_batch(batch_losses, writer)
        results.extend(batch_losses)
        if controller is not None:
            controller.observe_success(
                padded_batch_tokens=max(len(item.input_ids) for item in batch) * len(batch),
                elapsed_seconds=elapsed,
                headroom_bytes=cuda_headroom_bytes(device),
            )

    for example in tqdm(prepared_examples, desc="Preparing losses"):
        active_limit = controller.current_tokens if controller is not None else batch_max_tokens
        if pending_batch:
            exceeds_tokens = _projected_batch_tokens(pending_batch, example) > active_limit
            exceeds_batch_size = max_batch_size is not None and len(pending_batch) >= max_batch_size
            if exceeds_tokens or exceeds_batch_size:
                _flush_batch(pending_batch)
                pending_batch = []
        pending_batch.append(example)

    if pending_batch:
        _flush_batch(pending_batch)

    if writer is not None:
        writer.finalize()
    return results


def _worker_output_root(output_root: Path, worker_index: int) -> Path:
    return output_root / "_workers" / f"worker-{worker_index:02d}"


def _load_loss_rows_from_path(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_loss_worker_outputs(
    output_root: Path | str,
    worker_roots: list[Path | str],
    *,
    top_k: int = 25,
    finalize: bool = False,
) -> list[LossResult]:
    root = Path(output_root)
    worker_paths = [Path(path) for path in worker_roots]
    manifests: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []

    for worker_root in worker_paths:
        manifest_path = worker_root / "manifests" / "loss_state.json"
        if not manifest_path.exists():
            continue
        try:
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Failed to load worker loss manifest at %s", manifest_path)
            continue
        top_rows.extend(_load_loss_rows_from_path(worker_root / "partial" / "high_loss_examples.partial.jsonl"))
        if finalize:
            final_path = worker_root / "per_example_losses.jsonl"
            if final_path.exists():
                merged_rows.extend(_load_loss_rows_from_path(final_path))
            else:
                for shard_path in sorted((worker_root / "shards").glob("*.jsonl")):
                    merged_rows.extend(_load_loss_rows_from_path(shard_path))

    rows_written = sum(int(manifest.get("rows_written", 0)) for manifest in manifests)
    batch_count = sum(int(manifest.get("batch_count", 0)) for manifest in manifests)
    loss_sum = sum(float(manifest.get("loss_sum", 0.0)) for manifest in manifests)
    completion_tokens = sum(int(manifest.get("completion_tokens", 0)) for manifest in manifests)
    total_tokens = sum(int(manifest.get("total_tokens", 0)) for manifest in manifests)
    max_loss_values = [manifest.get("max_loss") for manifest in manifests if manifest.get("max_loss") is not None]
    top_rows = sorted(top_rows, key=lambda row: row["loss"], reverse=True)[:top_k]

    partial_dir = root / "partial"
    manifests_dir = root / "manifests"
    partial_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows_written": rows_written,
        "batch_count": batch_count,
        "worker_count": len(worker_paths),
        "completed": finalize and bool(manifests) and all(bool(manifest.get("completed", False)) for manifest in manifests),
        "mean_loss": (loss_sum / rows_written) if rows_written else 0.0,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "max_loss": max(max_loss_values) if max_loss_values else None,
    }
    _atomic_write_text(partial_dir / "loss_summary.partial.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_jsonl_rows(partial_dir / "high_loss_examples.partial.jsonl", top_rows)

    manifest_payload = {
        "version": 2,
        "completed": summary["completed"],
        "rows_written": rows_written,
        "batch_count": batch_count,
        "loss_sum": loss_sum,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "max_loss": summary["max_loss"],
        "workers": [str(path.relative_to(root)) if path.is_relative_to(root) else str(path) for path in worker_paths],
    }
    _atomic_write_text(manifests_dir / "loss_state.json", json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

    if not finalize:
        return []

    merged_rows = sorted(merged_rows, key=lambda row: row["index"])
    losses = [LossResult(**row) for row in merged_rows]
    save_losses(losses, root / "per_example_losses.jsonl")
    _atomic_write_text(root / "loss_summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return losses


def _loss_worker_entry(
    *,
    worker_index: int,
    worker_count: int,
    model_dir: str,
    dataset_path: str,
    output_root: str,
    max_seq_length: int,
    completion_only: bool,
    batch_max_tokens: int,
    max_batch_size: int | None,
    adaptive_batching: bool,
) -> None:
    try:
        from .transformers_loss_loader import load_transformers_loss_model

        worker_root = _worker_output_root(Path(output_root), worker_index)
        device = (
            torch.device(f"cuda:{worker_index}")
            if torch.cuda.is_available() and torch.cuda.device_count() > worker_index
            else torch.device("cpu")
        )
        model, tokenizer = load_transformers_loss_model(model_dir, device=device)
        writer = IncrementalLossWriter(worker_root, state_stride=worker_count)
        compute_per_example_losses(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            max_seq_length=max_seq_length,
            completion_only=completion_only,
            device=device,
            batch_max_tokens=batch_max_tokens,
            max_batch_size=max_batch_size,
            writer=writer,
            adaptive_batching=adaptive_batching,
            shard_index=worker_index,
            shard_count=worker_count,
        )
    except Exception as exc:  # pragma: no cover - exercised through parent orchestration
        worker_root = _worker_output_root(Path(output_root), worker_index)
        worker_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            worker_root / "error.json",
            json.dumps({"worker_index": worker_index, "error": repr(exc)}, indent=2, ensure_ascii=False) + "\n",
        )
        raise


def compute_per_example_losses_parallel(
    *,
    model_dir: Path | str,
    dataset_path: Path | str,
    output_root: Path | str,
    max_seq_length: int = 2048,
    completion_only: bool = True,
    batch_max_tokens: int = 8192,
    max_batch_size: int | None = None,
    adaptive_batching: bool = True,
    num_workers: int | None = None,
    on_aggregate: Callable[[Path], None] | None = None,
    aggregate_interval_seconds: float = 5.0,
) -> list[LossResult]:
    """Compute exact loss using one worker per visible GPU when beneficial."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    worker_count = max(int(num_workers or 0), 0) or max(available_gpus, 1)
    if worker_count <= 1:
        from .transformers_loss_loader import load_transformers_loss_model

        model, tokenizer = load_transformers_loss_model(model_dir)
        writer = IncrementalLossWriter(root)
        losses = compute_per_example_losses(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            max_seq_length=max_seq_length,
            completion_only=completion_only,
            batch_max_tokens=batch_max_tokens,
            max_batch_size=max_batch_size,
            writer=writer,
            on_batch=(lambda _losses, _writer: on_aggregate(root) if on_aggregate is not None else None),
            adaptive_batching=adaptive_batching,
        )
        if on_aggregate is not None:
            on_aggregate(root)
        return losses

    worker_roots = [_worker_output_root(root, worker_index) for worker_index in range(worker_count)]
    ctx = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []
    for worker_index in range(worker_count):
        process = ctx.Process(
            target=_loss_worker_entry,
            kwargs={
                "worker_index": worker_index,
                "worker_count": worker_count,
                "model_dir": str(model_dir),
                "dataset_path": str(dataset_path),
                "output_root": str(root),
                "max_seq_length": max_seq_length,
                "completion_only": completion_only,
                "batch_max_tokens": batch_max_tokens,
                "max_batch_size": max_batch_size,
                "adaptive_batching": adaptive_batching,
            },
        )
        process.start()
        processes.append(process)

    try:
        while any(process.is_alive() for process in processes):
            aggregate_loss_worker_outputs(root, worker_roots, finalize=False)
            if on_aggregate is not None:
                on_aggregate(root)
            time.sleep(max(aggregate_interval_seconds, 1.0))
        for process in processes:
            process.join()
        aggregate_loss_worker_outputs(root, worker_roots, finalize=False)
        if on_aggregate is not None:
            on_aggregate(root)
        failed_workers = [process.pid for process in processes if process.exitcode not in (0, None)]
        if failed_workers:
            errors: list[str] = []
            for worker_root in worker_roots:
                error_path = worker_root / "error.json"
                if error_path.exists():
                    errors.append(error_path.read_text(encoding="utf-8").strip())
            detail = "; ".join(errors) if errors else f"worker pids={failed_workers}"
            raise RuntimeError(f"Parallel loss computation failed: {detail}")
        losses = aggregate_loss_worker_outputs(root, worker_roots, finalize=True)
        if on_aggregate is not None:
            on_aggregate(root)
        return losses
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def save_losses(losses: list[LossResult], out_path: Path | str) -> None:
    """Save a list of LossResult to a JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for loss in losses:
            handle.write(json.dumps(asdict(loss), ensure_ascii=False) + "\n")


def load_losses(in_path: Path | str) -> list[LossResult]:
    """Load a list of LossResult from a JSONL file."""
    results = []
    with Path(in_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            results.append(LossResult(**data))
    return results
