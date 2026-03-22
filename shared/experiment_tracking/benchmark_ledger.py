from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tuner.cloud.hardware_planner import load_live_hf_hardware_rows

from .experiment import Experiment
from .schema import LossResult, RunRecord


LEDGER_CSV_RELATIVE_PATH = Path("docs/benchmarks/model_hardware_benchmark_ledger.csv")
LEDGER_MD_RELATIVE_PATH = Path("docs/benchmarks/model_hardware_benchmark_ledger.md")

_LEDGER_HEADERS = [
    "recorded_at",
    "experiment_id",
    "experiment_name",
    "benchmark_group",
    "base_model",
    "model_size_bucket",
    "dataset_variant",
    "method",
    "epochs",
    "max_seq_length",
    "train_flavor",
    "eval_flavor",
    "loss_flavor",
    "train_batch_size",
    "train_grad_accum",
    "train_effective_batch",
    "train_time_seconds",
    "train_cost_usd_est",
    "eval_time_seconds",
    "eval_cost_usd_est",
    "loss_time_seconds",
    "loss_cost_usd_est",
    "total_time_seconds",
    "total_cost_usd_est",
    "training_final_loss",
    "eval_passed",
    "eval_failed",
    "eval_warned",
    "eval_total",
    "eval_pass_rate",
    "eval_schema_pass_rate",
    "loss_examples",
    "loss_mean",
    "loss_status",
    "notes",
]


def _read_json(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    if path.startswith("hf://"):
        from shared.utilities.bucket_artifacts import read_artifact

        return json.loads(read_artifact(path))
    raise FileNotFoundError(path)


def _append_artifact(path: str | None, suffix: str) -> str | None:
    if not path:
        return None
    normalized = str(path).rstrip("/")
    return f"{normalized}/{suffix}"


def _load_artifact_payload(run: RunRecord | None, suffix: str) -> dict[str, Any]:
    if run is None:
        return {}
    artifact_path = _append_artifact(run.artifact_root or run.output_dir, suffix)
    if not artifact_path:
        return {}
    try:
        return _read_json(artifact_path)
    except Exception:
        return {}


def _iso_to_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_seconds(stage_details: dict[str, Any]) -> Optional[float]:
    started = _iso_to_dt(stage_details.get("started_at"))
    finished = _iso_to_dt(stage_details.get("finished_at"))
    if not started or not finished:
        return None
    return max((finished - started).total_seconds(), 0.0)


def _payload_runtime_seconds(payload: dict[str, Any]) -> Optional[float]:
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    value = runtime.get("duration_seconds")
    if value is not None:
        return _coerce_float(value)
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    performance = payload.get("performance") if isinstance(payload.get("performance"), dict) else {}
    return (
        _coerce_float(results.get("training_time_seconds"))
        or _coerce_float(performance.get("total_time_s"))
    )


def _payload_cost(payload: dict[str, Any]) -> Optional[float]:
    pricing = payload.get("pricing") if isinstance(payload.get("pricing"), dict) else {}
    return _coerce_float(pricing.get("estimated_cost_usd"))


def _payload_hardware(payload: dict[str, Any]) -> str:
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    planner = payload.get("planner") if isinstance(payload.get("planner"), dict) else {}
    hardware = payload.get("hardware") if isinstance(payload.get("hardware"), dict) else {}
    return str(
        execution.get("hardware_flavor")
        or planner.get("resolved_hardware_flavor")
        or hardware.get("cloud_gpu_type")
        or ""
    )


def _price_lookup() -> dict[str, float]:
    try:
        return {row.flavor: float(row.price_hr) for row in load_live_hf_hardware_rows()}
    except Exception:
        return {}


def _estimate_cost(flavor: str | None, seconds: Optional[float], prices: dict[str, float]) -> Optional[float]:
    if not flavor or seconds is None:
        return None
    hourly = prices.get(str(flavor))
    if hourly is None:
        return None
    return hourly * (seconds / 3600.0)


def _model_size_bucket(model_name: str) -> str:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*([bm])", model_name, re.IGNORECASE)
    if not match:
        return ""
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        value /= 1000.0
    return f"{value:g}b"


def _dataset_variant(dataset_path: str) -> str:
    name = Path(dataset_path).name
    if name.endswith(".jsonl"):
        name = name[:-6]
    return name


def _experiment_group(experiment: Experiment) -> str:
    objective = str(experiment.objective or "").strip()
    if objective:
        return objective
    return f"{experiment.base_model_name}-{experiment.method}".replace("/", "_")


def _run_by_stage(runs: list[RunRecord], stage: str) -> Optional[RunRecord]:
    for run in runs:
        if (run.stage or "") == stage:
            return run
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_benchmark_ledger_row(
    *,
    experiment: Experiment,
    runs: list[RunRecord],
    eval_payload: Optional[dict[str, Any]] = None,
    loss_results: Optional[list[LossResult]] = None,
) -> dict[str, Any]:
    training_run = _run_by_stage(runs, "training")
    evaluation_run = _run_by_stage(runs, "evaluation")
    loss_run = _run_by_stage(runs, "loss")

    training_payload = _load_artifact_payload(training_run, "training_lineage.json")
    evaluation_lineage = _load_artifact_payload(evaluation_run, "evaluation_lineage.json")
    loss_lineage = _load_artifact_payload(loss_run, "loss_lineage.json")
    training_cfg = training_payload.get("training", {})
    training_results = training_payload.get("results", {})
    eval_summary = (eval_payload or {}).get("summary", {})
    if not eval_summary and evaluation_lineage:
        eval_summary = evaluation_lineage.get("results_summary", {})

    stage_training = experiment.stage_details.get("training", {})
    stage_evaluation = experiment.stage_details.get("evaluation", {})
    stage_loss = experiment.stage_details.get("loss", {})

    train_seconds = _payload_runtime_seconds(training_payload) or _duration_seconds(stage_training)
    eval_seconds = _payload_runtime_seconds(evaluation_lineage) or _duration_seconds(stage_evaluation)
    loss_seconds = _payload_runtime_seconds(loss_lineage) or _duration_seconds(stage_loss)

    train_flavor = _payload_hardware(training_payload) or (training_run.hardware if training_run else None) or stage_training.get("hardware") or ""
    eval_flavor = _payload_hardware(evaluation_lineage) or (evaluation_run.hardware if evaluation_run else None) or stage_evaluation.get("hardware") or ""
    loss_flavor = _payload_hardware(loss_lineage) or (loss_run.hardware if loss_run else None) or stage_loss.get("hardware") or ""

    prices = _price_lookup()
    train_cost = _payload_cost(training_payload) or _estimate_cost(train_flavor, train_seconds, prices)
    eval_cost = _payload_cost(evaluation_lineage) or _estimate_cost(eval_flavor, eval_seconds, prices)
    loss_cost = _payload_cost(loss_lineage) or _estimate_cost(loss_flavor, loss_seconds, prices)

    total_seconds = sum(
        value for value in (train_seconds, eval_seconds, loss_seconds) if isinstance(value, (int, float))
    ) or None
    total_cost = sum(
        value for value in (train_cost, eval_cost, loss_cost) if isinstance(value, (int, float))
    ) or None

    loss_mean = None
    loss_examples = len(loss_results or [])
    if loss_results:
        loss_mean = sum(item.loss for item in loss_results) / max(len(loss_results), 1)
    elif loss_lineage:
        loss_metrics = loss_lineage.get("results", {})
        loss_mean = _coerce_float(loss_metrics.get("mean_loss"))
        loss_examples = int(loss_metrics.get("row_count", 0) or 0)

    notes: list[str] = []
    if stage_loss.get("status") and stage_loss.get("status") != "completed":
        error_message = stage_loss.get("error_message") or ""
        if error_message:
            notes.append(str(error_message))
    if stage_evaluation.get("status") and stage_evaluation.get("status") != "completed":
        error_message = stage_evaluation.get("error_message") or ""
        if error_message:
            notes.append(str(error_message))

    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.name,
        "benchmark_group": _experiment_group(experiment),
        "base_model": experiment.base_model_name,
        "model_size_bucket": _model_size_bucket(experiment.base_model_name),
        "dataset_variant": _dataset_variant(experiment.dataset_path),
        "method": experiment.method,
        "epochs": training_cfg.get("num_epochs", ""),
        "max_seq_length": training_cfg.get("max_seq_length", ""),
        "train_flavor": train_flavor,
        "eval_flavor": eval_flavor,
        "loss_flavor": loss_flavor,
        "train_batch_size": training_cfg.get("batch_size", ""),
        "train_grad_accum": training_cfg.get("gradient_accumulation_steps", ""),
        "train_effective_batch": training_cfg.get("effective_batch_size", ""),
        "train_time_seconds": train_seconds,
        "train_cost_usd_est": train_cost,
        "eval_time_seconds": eval_seconds,
        "eval_cost_usd_est": eval_cost,
        "loss_time_seconds": loss_seconds,
        "loss_cost_usd_est": loss_cost,
        "total_time_seconds": total_seconds,
        "total_cost_usd_est": total_cost,
        "training_final_loss": training_results.get("final_loss", ""),
        "eval_passed": eval_summary.get("passed", ""),
        "eval_failed": eval_summary.get("failed", ""),
        "eval_warned": eval_summary.get("warned", ""),
        "eval_total": eval_summary.get("total", ""),
        "eval_pass_rate": eval_summary.get("pass_rate", eval_summary.get("overall_pass_rate", "")),
        "eval_schema_pass_rate": eval_summary.get("schema_pass_rate", eval_summary.get("schema_pass_rate", "")),
        "loss_examples": loss_examples,
        "loss_mean": loss_mean,
        "loss_status": loss_lineage.get("status", "") or stage_loss.get("status", ""),
        "notes": " | ".join(notes),
    }
    return row


def upsert_benchmark_ledger(
    *,
    repo_root: str | Path,
    experiment: Experiment,
    runs: list[RunRecord],
    eval_payload: Optional[dict[str, Any]] = None,
    loss_results: Optional[list[LossResult]] = None,
) -> str:
    repo_root = Path(repo_root)
    ledger_path = repo_root / LEDGER_CSV_RELATIVE_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = build_benchmark_ledger_row(
        experiment=experiment,
        runs=runs,
        eval_payload=eval_payload,
        loss_results=loss_results,
    )

    existing: list[dict[str, Any]] = []
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing.extend(reader)

    replaced = False
    for index, candidate in enumerate(existing):
        if candidate.get("experiment_id") == experiment.experiment_id:
            existing[index] = {header: row.get(header, "") for header in _LEDGER_HEADERS}
            replaced = True
            break
    if not replaced:
        existing.append({header: row.get(header, "") for header in _LEDGER_HEADERS})

    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LEDGER_HEADERS)
        writer.writeheader()
        writer.writerows(existing)

    return str(ledger_path)
