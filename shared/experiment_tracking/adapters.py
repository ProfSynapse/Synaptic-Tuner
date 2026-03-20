"""
shared/experiment_tracking/adapters.py

Adapter functions that bridge existing lineage/manifest formats into RunRecord.
Each adapter takes the native output of one system and returns a RunRecord
ready for registry insertion.

Used by: SFT/KTO/GRPO post-training hooks, cloud manifest hooks, Evaluator hooks.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import RunRecord

_adapters_logger = logging.getLogger(__name__)


def _training_lineage_to_run_record(
    lineage: dict[str, Any],
    run_dir: str,
    method: str,
    *,
    run_id: str | None = None,
    cloud: bool = False,
) -> RunRecord:
    """Shared helper for SFT/KTO lineage-to-RunRecord conversion.

    Args:
        lineage: Parsed content of training_lineage.json.
        run_dir: Absolute path to the run output directory.
        method: Training method ("sft" or "kto").
        run_id: Optional pre-generated run ID. If None, a new UUID4 is created.
        cloud: If True, prefixes run_type with "cloud_".

    Returns:
        A RunRecord populated from the training lineage.
    """
    results = lineage.get("results", {})
    model_info = lineage.get("model", {})
    dataset_info = lineage.get("dataset", {})
    hardware_info = lineage.get("hardware", {})

    final_loss = results.get("final_loss")
    metric_name = "final_loss" if final_loss is not None else None

    gpu_name = hardware_info.get("gpu_name", "")
    hw_str = gpu_name if gpu_name else None

    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type=f"cloud_{method}" if cloud else method,
        name=f"{method.upper()} run {lineage.get('timestamp', '')}".strip(),
        timestamp=lineage.get("timestamp", datetime.now(timezone.utc).isoformat()),
        status="completed",
        output_dir=run_dir,
        tags={"method": method, "provider": "cloud" if cloud else "local"},
        model_name=model_info.get("base_model"),
        dataset_source=dataset_info.get("source"),
        primary_metric=final_loss,
        primary_metric_name=metric_name,
        hardware=hw_str,
    )


def sft_lineage_to_run_record(
    lineage: dict[str, Any],
    run_dir: str,
    *,
    run_id: str | None = None,
    cloud: bool = False,
) -> RunRecord:
    """Convert an SFT training_lineage.json dict to a RunRecord."""
    return _training_lineage_to_run_record(
        lineage, run_dir, "sft", run_id=run_id, cloud=cloud,
    )


def kto_lineage_to_run_record(
    lineage: dict[str, Any],
    run_dir: str,
    *,
    run_id: str | None = None,
    cloud: bool = False,
) -> RunRecord:
    """Convert a KTO training_lineage.json dict to a RunRecord."""
    return _training_lineage_to_run_record(
        lineage, run_dir, "kto", run_id=run_id, cloud=cloud,
    )


def ml_tracking_to_run_record(
    tracking_data: dict[str, Any],
    run_dir: str,
    *,
    run_id: str | None = None,
) -> RunRecord:
    """Convert LocalTracker tracking.json to a RunRecord.

    Args:
        tracking_data: Parsed content of tracking.json.
        run_dir: Absolute path to the run output directory.
        run_id: Optional pre-generated run ID.

    Returns:
        A RunRecord populated from the ML tracker output.
    """
    params = tracking_data.get("params", {})
    metrics = tracking_data.get("metrics", {})

    # Find the "best" metric to use as primary
    primary_metric = None
    primary_metric_name = None
    for candidate in ("accuracy", "f1", "r2", "rmse", "mse"):
        if candidate in metrics:
            primary_metric = metrics[candidate]
            primary_metric_name = candidate
            break

    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type="ml",
        name=tracking_data.get("run_name", "ML run"),
        timestamp=tracking_data.get("started_at", datetime.now(timezone.utc).isoformat()),
        status="completed",
        output_dir=run_dir,
        tags={
            "method": "ml",
            "algorithm": params.get("algorithm", "unknown"),
            "task_type": params.get("task_type", "unknown"),
        },
        model_name=params.get("algorithm"),
        primary_metric=primary_metric,
        primary_metric_name=primary_metric_name,
    )


def manifest_to_run_record(
    manifest: dict[str, Any],
    *,
    run_id: str | None = None,
) -> RunRecord:
    """Convert a cloud training manifest.json to a RunRecord.

    Args:
        manifest: Parsed content of manifest.json (from build_manifest()).
        run_id: Optional pre-generated run ID.

    Returns:
        A RunRecord populated from the cloud manifest.
    """
    method = manifest.get("method", "sft")
    run_type = f"cloud_{method}"
    paths = manifest.get("paths", {})

    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type=run_type,
        name=f"Cloud {method.upper()} {manifest.get('generated_at', '')}".strip(),
        timestamp=manifest.get("generated_at", datetime.now(timezone.utc).isoformat()),
        status=manifest.get("status", "completed"),
        output_dir=paths.get("run_dir", ""),
        tags={
            "method": method,
            "provider": manifest.get("provider", "unknown"),
            "artifact_backend": manifest.get("artifact_backend", ""),
        },
    )


def grpo_log_to_run_record(
    log_entries: list[dict[str, Any]],
    run_dir: str,
    *,
    run_id: str | None = None,
    model_name: str | None = None,
    dataset_source: str | None = None,
    cloud: bool = False,
) -> RunRecord:
    """Convert GRPO training log JSONL entries to a RunRecord.

    The GRPO trainer writes per-step metrics to a JSONL log file.  The final
    entry (with ``event == "train_end"``) carries aggregate statistics such as
    ``total_steps``, ``total_epochs``, and ``train_runtime``.

    Args:
        log_entries: Parsed list of JSONL dicts from ``training_*.jsonl``.
        run_dir: Absolute path to the run output directory.
        run_id: Optional pre-generated run ID. If None, a new UUID4 is created.
        model_name: Model identifier (from YAML config or CLI).
        dataset_source: Dataset path or HuggingFace identifier.
        cloud: If True, sets run_type to ``"cloud_grpo"`` instead of ``"grpo"``.

    Returns:
        A RunRecord populated from the GRPO training log.
    """
    # Find the final summary entry (event == "train_end") — fall back to last entry
    summary: dict[str, Any] = {}
    last_step_entry: dict[str, Any] = {}
    for entry in log_entries:
        if entry.get("event") == "train_end":
            summary = entry
        elif "step" in entry:
            last_step_entry = entry

    # Extract the best available primary metric
    # GRPO tracks reward; use the last logged reward as primary metric
    primary_metric = None
    primary_metric_name = None
    for key in ("reward", "rewards", "rewards/mean", "mean_reward"):
        val = last_step_entry.get(key)
        if val is not None:
            primary_metric = float(val)
            primary_metric_name = "reward"
            break

    # Fall back to final loss if no reward found
    if primary_metric is None:
        loss = last_step_entry.get("loss")
        if loss is not None:
            primary_metric = float(loss)
            primary_metric_name = "loss"

    # Hardware from capacity snapshot
    gpu = last_step_entry.get("gpu_name") or summary.get("gpu_name")
    hw_str = str(gpu) if gpu else None

    # Timestamp from summary or last entry
    ts = (
        summary.get("timestamp")
        or last_step_entry.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )

    total_steps = summary.get("total_steps") or last_step_entry.get("step")
    name_suffix = f"{total_steps} steps" if total_steps else ""

    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type="cloud_grpo" if cloud else "grpo",
        name=f"GRPO run {name_suffix}".strip(),
        timestamp=ts,
        status="completed",
        output_dir=run_dir,
        tags={"method": "grpo", "provider": "cloud" if cloud else "local"},
        model_name=model_name,
        dataset_source=dataset_source,
        primary_metric=primary_metric,
        primary_metric_name=primary_metric_name,
        hardware=hw_str,
    )


def eval_to_run_record(
    lineage: dict[str, Any],
    output_dir: str,
    *,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> RunRecord:
    """Convert evaluation lineage to a RunRecord.

    Args:
        lineage: Parsed content of evaluation_lineage.json.
        output_dir: Path to the evaluation output directory.
        run_id: Optional pre-generated run ID.
        parent_run_id: Optional training run ID that this evaluation targets.

    Returns:
        A RunRecord populated from the evaluation lineage.
    """
    results = lineage.get("results_summary", {})
    perf = lineage.get("performance", {})

    pass_rate = results.get("overall_pass_rate")

    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        run_type="evaluation",
        name=f"Eval of {lineage.get('model_evaluated', 'unknown')}",
        timestamp=lineage.get("evaluation_timestamp", datetime.now(timezone.utc).isoformat()),
        status="completed",
        output_dir=output_dir,
        parent_run_id=parent_run_id,
        tags={
            "suite": ",".join(lineage.get("test_config", {}).get("test_suites", [])),
        },
        model_name=lineage.get("model_evaluated"),
        primary_metric=pass_rate,
        primary_metric_name="pass_rate",
    )


def register_grpo_run(
    log_dir: Path,
    run_dir: str,
    *,
    model_name: str | None = None,
    dataset_source: str | None = None,
    cloud: bool = False,
) -> str | None:
    """Load GRPO training logs from disk and register in the unified registry.

    Best-effort helper used by both train_grpo.py and train_env_grpo.py to
    avoid duplicating the JSONL-read + register logic.

    Args:
        log_dir: Directory containing ``training_*.jsonl`` log files.
        run_dir: Absolute path to the run output directory.
        model_name: Model identifier (from YAML config or CLI).
        dataset_source: Dataset path or HuggingFace identifier.
        cloud: If True, sets run_type to ``"cloud_grpo"``.

    Returns:
        The registered run_id, or None if no log files were found.
    """
    from .registry import RunRegistry

    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("training_*.jsonl")) if log_dir.exists() else []
    if not log_files:
        return None

    entries: list[dict[str, Any]] = []
    with open(log_files[-1], "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if raw_line:
                entries.append(json.loads(raw_line))

    record = grpo_log_to_run_record(
        entries,
        run_dir,
        model_name=model_name,
        dataset_source=dataset_source,
        cloud=cloud,
    )
    run_id = RunRegistry().register_run(record)
    _adapters_logger.info("GRPO run registered in unified tracking: %s", run_id)
    return run_id
