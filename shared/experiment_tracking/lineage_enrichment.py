from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional, Sequence

from tuner.cloud.hardware_planner import load_live_hf_hardware_rows

from .schema import LossResult


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def compute_duration_seconds(started_at: str | None, finished_at: str | None) -> Optional[float]:
    started = parse_iso_datetime(started_at)
    finished = parse_iso_datetime(finished_at)
    if not started or not finished:
        return None
    return max((finished - started).total_seconds(), 0.0)


def dataset_variant_from_source(dataset_source: str | None) -> str:
    if not dataset_source:
        return ""
    candidate = str(dataset_source).strip().rstrip("/")
    name = Path(candidate).name
    if name.endswith(".jsonl"):
        return name[:-6]
    return name


def resolve_stage_hardware_flavor(
    *,
    explicit_flavor: str | None = None,
    hardware_info: dict[str, Any] | None = None,
    args: Any | None = None,
) -> str | None:
    hardware = hardware_info or {}
    if explicit_flavor:
        return str(explicit_flavor)
    if args is not None:
        for attr in ("hf_flavor", "gpu", "gpu_type"):
            value = getattr(args, attr, None)
            if value:
                return str(value)
    for key in ("cloud_gpu_type", "hf_flavor", "gpu_flavor", "gpu_type"):
        value = hardware.get(key)
        if value:
            return str(value)
    return None


def live_price_lookup(flavor: str | None) -> tuple[str, Optional[float]]:
    if not flavor:
        return ("unavailable", None)
    try:
        for row in load_live_hf_hardware_rows():
            if row.flavor == flavor:
                return ("hf_jobs_live", float(row.price_hr))
    except Exception:
        return ("unavailable", None)
    return ("unavailable", None)


def pricing_payload(flavor: str | None, runtime_seconds: float | None) -> dict[str, Any]:
    source, hourly_price = live_price_lookup(flavor)
    cost = None
    if hourly_price is not None and runtime_seconds is not None:
        cost = hourly_price * (float(runtime_seconds) / 3600.0)
    return {
        "price_source": source,
        "price_hour_usd": hourly_price,
        "estimated_cost_usd": cost,
    }


def loss_distribution(loss_results: Sequence[LossResult]) -> dict[str, Any]:
    if not loss_results:
        return {
            "row_count": 0,
            "mean_loss": 0.0,
            "median_loss": 0.0,
            "p95_loss": 0.0,
            "max_loss": None,
        }
    values = sorted(float(item.loss) for item in loss_results)
    p95_index = max(int(math.ceil(len(values) * 0.95)) - 1, 0)
    return {
        "row_count": len(values),
        "mean_loss": sum(values) / len(values),
        "median_loss": float(median(values)),
        "p95_loss": float(values[min(p95_index, len(values) - 1)]),
        "max_loss": float(values[-1]),
    }


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def enrich_training_lineage(
    lineage: dict[str, Any],
    *,
    args: Any | None = None,
    explicit_flavor: str | None = None,
    runtime_backend: str = "trainer",
) -> dict[str, Any]:
    payload = dict(lineage)
    hardware = payload.get("hardware") if isinstance(payload.get("hardware"), dict) else {}
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}

    finished_at = payload.get("timestamp") or iso_utc_now()
    runtime_seconds = results.get("training_time_seconds")
    flavor = resolve_stage_hardware_flavor(explicit_flavor=explicit_flavor, hardware_info=hardware, args=args)

    dataset = dict(dataset)
    dataset.setdefault("variant", dataset_variant_from_source(dataset.get("source")))
    payload["dataset"] = dataset
    payload["stage"] = "training"
    payload["runtime"] = {
        "backend": runtime_backend,
        "status": "completed",
        "finished_at": finished_at,
        "duration_seconds": runtime_seconds,
    }
    payload["pricing"] = pricing_payload(flavor, runtime_seconds)
    payload["planner"] = {
        "auto_hardware": bool(getattr(args, "auto_hardware", False)) if args is not None else False,
        "optimization_objective": getattr(args, "optimize_for", None) if args is not None else None,
        "resolved_hardware_flavor": flavor,
        "effective_batch_size": training.get("effective_batch_size"),
    }
    return payload


def enrich_evaluation_lineage(
    lineage: dict[str, Any],
    *,
    backend: str,
    hardware_flavor: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    tensor_parallel_size: int | None = None,
    worker_count: int | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    payload = dict(lineage)
    duration_seconds = compute_duration_seconds(started_at, finished_at)
    payload["stage"] = "evaluation"
    payload["runtime"] = {
        "backend": backend,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
    }
    payload["pricing"] = pricing_payload(hardware_flavor, duration_seconds)
    payload["execution"] = {
        "hardware_flavor": hardware_flavor,
        "tensor_parallel_size": tensor_parallel_size,
        "worker_count": worker_count,
        "fallback_reason": fallback_reason,
    }
    return payload


def build_loss_lineage(
    *,
    dataset_path: str | Path,
    output_root: str | Path,
    loss_results: Sequence[LossResult],
    completion_only: bool,
    max_seq_length: int,
    batch_max_tokens: int | None = None,
    max_batch_size: int | None = None,
    adaptive_batching: bool | None = None,
    runtime_backend: str = "transformers",
    hardware_flavor: str | None = None,
    worker_count: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    status: str = "completed",
    experiment_id: str | None = None,
    training_run_id: str | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / "loss_summary.json"
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = {}

    distribution = loss_distribution(loss_results)
    duration_seconds = compute_duration_seconds(started_at, finished_at)
    dataset_source = str(dataset_path)
    lineage = {
        "stage": "loss",
        "status": status,
        "generated_at": iso_utc_now(),
        "experiment_id": experiment_id,
        "training_run_id": training_run_id,
        "dataset": {
            "source": dataset_source,
            "variant": dataset_variant_from_source(dataset_source),
        },
        "runtime": {
            "backend": runtime_backend,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        },
        "pricing": pricing_payload(hardware_flavor, duration_seconds),
        "execution": {
            "hardware_flavor": hardware_flavor,
            "worker_count": worker_count or summary_payload.get("worker_count") or 1,
            "batch_count": summary_payload.get("batch_count"),
            "batch_max_tokens": batch_max_tokens,
            "max_batch_size": max_batch_size,
            "adaptive_batching": adaptive_batching,
            "completion_only": completion_only,
            "max_seq_length": max_seq_length,
        },
        "results": {
            **distribution,
            "rows_written": summary_payload.get("rows_written", distribution["row_count"]),
            "completion_tokens": summary_payload.get("completion_tokens"),
            "total_tokens": summary_payload.get("total_tokens"),
        },
        "artifacts": {
            "per_example_losses": str(root / "per_example_losses.jsonl"),
            "loss_summary": str(summary_path),
            "high_loss_examples": str(root / "failure_slices" / "high_loss_examples.jsonl"),
            "partial_summary": str(root / "partial" / "loss_summary.partial.json"),
            "manifest": str(root / "manifests" / "loss_state.json"),
        },
    }
    return lineage
