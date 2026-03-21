from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .experiment import Experiment
from .schema import LossResult, RunRecord


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _run_row(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "run_type": record.run_type,
        "stage": record.stage or "",
        "status": record.status,
        "provider": record.provider or record.tags.get("provider", ""),
        "model_name": record.model_name or "",
        "primary_metric_name": record.primary_metric_name or "",
        "primary_metric": record.primary_metric,
        "artifact_root": record.artifact_root or record.output_dir,
        "job_ref": record.job_ref or "",
        "timestamp": record.timestamp,
    }


def _next_run_candidates(experiment: Experiment, runs: list[RunRecord], eval_payload: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    training_run = next((run for run in runs if run.run_id == experiment.training_run_id), None)
    total = int((eval_payload or {}).get("summary", {}).get("total", 0))
    failed = int((eval_payload or {}).get("summary", {}).get("failed", 0))
    passed = int((eval_payload or {}).get("summary", {}).get("passed", 0))

    if training_run and training_run.primary_metric_name == "final_loss" and (training_run.primary_metric or 0) > 1.0:
        candidates.append(
            {
                "rank": 1,
                "hypothesis": "Training likely underfit or learning rate is too aggressive.",
                "suggested_changes": {
                    "train_learning_rate": 0.5,
                    "operator": "multiply",
                },
                "why": f"Observed final_loss={training_run.primary_metric}",
            }
        )

    if total > 0 and failed >= passed:
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "hypothesis": "Evaluation failures dominate. The run may need more optimization steps before comparison.",
                "suggested_changes": {
                    "train_max_steps": 2.0,
                    "operator": "multiply",
                },
                "why": f"Observed {failed} failed vs {passed} passed evaluation cases.",
            }
        )

    if not candidates:
        candidates.append(
            {
                "rank": 1,
                "hypothesis": "Collect another comparable run before making an aggressive change.",
                "suggested_changes": {"experiment": "rerun_with_small_lr_or_steps_delta"},
                "why": "Current experiment does not yet have enough signal for a stronger hypothesis.",
            }
        )

    return candidates


def write_analysis_bundle(
    *,
    experiment: Experiment,
    runs: list[RunRecord],
    base_dir: str | Path = ".tracking",
    eval_payload: Optional[dict[str, Any]] = None,
    loss_results: Optional[list[LossResult]] = None,
) -> dict[str, str]:
    base_dir = Path(base_dir)
    analysis_dir = base_dir / "experiments" / experiment.experiment_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "status": experiment.status,
        "provider": experiment.provider,
        "method": experiment.method,
        "objective": experiment.objective,
        "run_count": len(runs),
        "training_run_id": experiment.training_run_id,
        "evaluation_run_id": experiment.evaluation_run_id,
        "loss_run_id": experiment.loss_run_id,
        "stage_statuses": experiment.stage_statuses,
        "artifact_roots": experiment.artifact_roots,
        "derived_outputs": experiment.derived_outputs,
        "eval_summary": (eval_payload or {}).get("summary", {}),
    }
    summary_json = analysis_dir / "experiment_summary.json"
    _write_json(summary_json, summary_payload)

    summary_md = analysis_dir / "experiment_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                f"# Experiment {experiment.experiment_id}",
                "",
                f"- Name: {experiment.name}",
                f"- Status: {experiment.status}",
                f"- Provider: {experiment.provider or '-'}",
                f"- Method: {experiment.method or '-'}",
                f"- Runs: {len(runs)}",
            ]
        ),
        encoding="utf-8",
    )

    run_rows = [_run_row(run) for run in runs]
    run_matrix = analysis_dir / "run_matrix.csv"
    _write_csv(run_matrix, run_rows)

    outputs = {
        "experiment_summary_json": str(summary_json),
        "experiment_summary_md": str(summary_md),
        "run_matrix_csv": str(run_matrix),
    }

    if eval_payload:
        failed_records = [record for record in (eval_payload.get("records") or []) if not record.get("passed", False)]
        failure_path = analysis_dir / "failure_slices" / "eval_failures.jsonl"
        _write_jsonl(failure_path, failed_records)
        outputs["eval_failures_jsonl"] = str(failure_path)

    if loss_results:
        loss_rows = [
            {
                "experiment_id": experiment.experiment_id,
                "training_run_id": experiment.training_run_id or "",
                "index": item.index,
                "loss": item.loss,
                "num_completion_tokens": item.num_completion_tokens,
                "num_total_tokens": item.num_total_tokens,
                "jsonl_hash": item.jsonl_hash,
            }
            for item in loss_results
        ]
        feature_jsonl = analysis_dir / "feature_dataset.jsonl"
        feature_csv = analysis_dir / "feature_dataset.csv"
        _write_jsonl(feature_jsonl, loss_rows)
        _write_csv(feature_csv, loss_rows)
        outputs["feature_dataset_jsonl"] = str(feature_jsonl)
        outputs["feature_dataset_csv"] = str(feature_csv)

        high_loss = sorted(loss_rows, key=lambda row: row["loss"], reverse=True)[:25]
        high_loss_path = analysis_dir / "failure_slices" / "high_loss_examples.jsonl"
        _write_jsonl(high_loss_path, high_loss)
        outputs["high_loss_examples_jsonl"] = str(high_loss_path)

    candidates = _next_run_candidates(experiment, runs, eval_payload)
    next_run_candidates = analysis_dir / "next_run_candidates.json"
    _write_json(next_run_candidates, {"candidates": candidates})
    outputs["next_run_candidates_json"] = str(next_run_candidates)

    hypothesis_context = analysis_dir / "hypothesis_context.json"
    _write_json(
        hypothesis_context,
        {
            "experiment": summary_payload,
            "runs": run_rows,
            "top_candidates": candidates,
        },
    )
    outputs["hypothesis_context_json"] = str(hypothesis_context)

    return outputs
