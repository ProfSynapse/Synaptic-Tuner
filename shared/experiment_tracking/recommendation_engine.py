from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Any, Optional

import yaml

from .experiment import Experiment
from .experiment_spec import EXPERIMENT_STAGES
from .schema import LossResult, RunRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dataset_identifier(identifier: str) -> dict[str, str]:
    if "/" not in identifier:
        return {"source": identifier, "file": ""}
    source, file_name = identifier.rsplit("/", 1)
    if not source:
        source = identifier
        file_name = ""
    return {"source": source, "file": file_name}


def _training_run(experiment: Experiment, runs: list[RunRecord]) -> RunRecord | None:
    if experiment.training_run_id:
        match = next((run for run in runs if run.run_id == experiment.training_run_id), None)
        if match is not None:
            return match
    return next((run for run in runs if run.stage == "training"), None)


def _eval_summary(eval_payload: Optional[dict[str, Any]]) -> dict[str, int]:
    summary = (eval_payload or {}).get("summary") or {}
    return {
        "passed": int(summary.get("passed", 0) or 0),
        "failed": int(summary.get("failed", 0) or 0),
        "warned": int(summary.get("warned", 0) or 0),
        "total": int(summary.get("total", 0) or 0),
    }


def _round_factor(value: float) -> float:
    return round(value, 3)


def _loss_snapshot(loss_results: list[LossResult]) -> dict[str, Any] | None:
    if not loss_results:
        return None
    ordered = sorted(loss_results, key=lambda item: item.loss, reverse=True)
    top = ordered[:5]
    losses = [item.loss for item in ordered]
    return {
        "top_loss": top[0].loss,
        "median_loss": median(losses),
        "mean_loss": round(sum(losses) / len(losses), 4),
        "high_loss_examples": [
            {
                "index": item.index,
                "loss": item.loss,
                "jsonl_hash": item.jsonl_hash,
                "num_completion_tokens": item.num_completion_tokens,
                "num_total_tokens": item.num_total_tokens,
            }
            for item in top
        ],
    }


def _candidate(
    *,
    rank: int,
    hypothesis: str,
    why: str,
    confidence: float,
    suggested_changes: dict[str, Any],
    evidence: list[str],
    signal: str,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "hypothesis": hypothesis,
        "why": why,
        "confidence": _round_factor(confidence),
        "signal": signal,
        "suggested_changes": suggested_changes,
        "evidence": evidence,
    }


def build_recommendation_bundle(
    *,
    experiment: Experiment,
    runs: list[RunRecord],
    eval_payload: Optional[dict[str, Any]] = None,
    loss_results: Optional[list[LossResult]] = None,
) -> dict[str, Any]:
    loss_results = loss_results or []
    training_run = _training_run(experiment, runs)
    eval_summary = _eval_summary(eval_payload)
    loss_snapshot = _loss_snapshot(loss_results)

    signals: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    if training_run and training_run.primary_metric_name == "final_loss" and training_run.primary_metric is not None:
        final_loss = float(training_run.primary_metric)
        signals.append(
            {
                "name": "training_final_loss",
                "severity": "high" if final_loss >= 1.0 else "medium" if final_loss >= 0.7 else "low",
                "value": final_loss,
                "evidence": f"final_loss={final_loss}",
            }
        )
        if final_loss >= 1.0:
            candidates.append(
                _candidate(
                    rank=len(candidates) + 1,
                    hypothesis="The run looks underfit or too aggressive on optimization.",
                    why=f"Observed final_loss={final_loss} on the training run.",
                    confidence=0.88,
                    suggested_changes={
                        "train_learning_rate": {"operator": "multiply", "value": 0.5},
                        "train_max_steps": {"operator": "multiply", "value": 1.25},
                    },
                    evidence=[f"training_run_id={training_run.run_id}", f"final_loss={final_loss}"],
                    signal="training_final_loss",
                )
            )
        elif final_loss >= 0.7:
            candidates.append(
                _candidate(
                    rank=len(candidates) + 1,
                    hypothesis="The run is close, but a smaller learning rate should stabilize the next attempt.",
                    why=f"Observed final_loss={final_loss} without a clear evaluation collapse.",
                    confidence=0.68,
                    suggested_changes={
                        "train_learning_rate": {"operator": "multiply", "value": 0.75},
                    },
                    evidence=[f"training_run_id={training_run.run_id}", f"final_loss={final_loss}"],
                    signal="training_final_loss",
                )
            )

    if eval_summary["total"] > 0:
        failure_rate = eval_summary["failed"] / eval_summary["total"]
        severity = "high" if eval_summary["failed"] >= eval_summary["passed"] else "medium"
        signals.append(
            {
                "name": "evaluation_failure_rate",
                "severity": severity if failure_rate >= 0.34 else "low",
                "value": round(failure_rate, 4),
                "evidence": f"failed={eval_summary['failed']} passed={eval_summary['passed']} total={eval_summary['total']}",
            }
        )
        if eval_summary["failed"] > 0:
            candidates.append(
                _candidate(
                    rank=len(candidates) + 1,
                    hypothesis="Evaluation failures dominate, so the next run should favor longer optimization before comparison.",
                    why=f"Observed {eval_summary['failed']} failed cases out of {eval_summary['total']} total.",
                    confidence=0.74 if eval_summary["failed"] >= eval_summary["passed"] else 0.58,
                    suggested_changes={
                        "train_max_steps": {"operator": "multiply", "value": 1.5},
                    },
                    evidence=[
                        f"failed={eval_summary['failed']}",
                        f"passed={eval_summary['passed']}",
                        f"total={eval_summary['total']}",
                    ],
                    signal="evaluation_failure_rate",
                )
            )

    if loss_snapshot is not None:
        top_loss = float(loss_snapshot["top_loss"])
        median_loss = float(loss_snapshot["median_loss"])
        spread = top_loss / median_loss if median_loss else top_loss
        signals.append(
            {
                "name": "loss_spread",
                "severity": "high" if spread >= 1.75 else "medium" if spread >= 1.25 else "low",
                "value": round(spread, 4),
                "evidence": f"top_loss={top_loss} median_loss={median_loss}",
            }
        )
        if spread >= 1.25:
            candidates.append(
                _candidate(
                    rank=len(candidates) + 1,
                    hypothesis="A small set of examples is much harder than the rest; target those examples before the next sweep.",
                    why=f"Top loss {top_loss} is {spread:.2f}x the median loss {median_loss}.",
                    confidence=0.7,
                    suggested_changes={
                        "dataset_review": {
                            "action": "prioritize_high_loss_examples",
                            "jsonl_hashes": [item["jsonl_hash"] for item in loss_snapshot["high_loss_examples"]],
                        }
                    },
                    evidence=[f"top_loss={top_loss}", f"median_loss={median_loss}"],
                    signal="loss_spread",
                )
            )

    if not candidates:
        candidates.append(
            _candidate(
                rank=1,
                hypothesis="Collect one more comparable run before making a larger change.",
                why="The finished experiment does not yet expose a strong enough signal for a heavier recommendation.",
                confidence=0.34,
                suggested_changes={"experiment": "rerun_with_small_delta"},
                evidence=["insufficient_signal"],
                signal="fallback",
            )
        )
        signals.append(
            {
                "name": "fallback",
                "severity": "low",
                "value": 1,
                "evidence": "insufficient_signal",
            }
        )

    dataset = _parse_dataset_identifier(experiment.dataset_path)
    selected_candidate = candidates[0]
    draft_spec = {
        "experiment": {
            "name": f"{experiment.name}-next",
            "provider": experiment.provider,
            "method": experiment.method,
            "objective": experiment.objective or "next_run",
            "dataset": dataset,
            "training": {
                "model_name": experiment.base_model_name,
            },
            "execution": {
                "stages": list(EXPERIMENT_STAGES),
            },
            "recommendation": {
                "source_experiment_id": experiment.experiment_id,
                "generated_at": _now_iso(),
                "selected_candidate_rank": selected_candidate["rank"],
                "selected_hypothesis": selected_candidate["hypothesis"],
                "candidates": candidates,
                "signals": signals,
                "summary": {
                    "run_count": len(runs),
                    "has_training_signal": training_run is not None,
                    "eval_summary": eval_summary,
                    "loss_snapshot": loss_snapshot,
                },
            },
        }
    }

    return {
        "summary": {
            "experiment_id": experiment.experiment_id,
            "run_count": len(runs),
            "training_run_id": experiment.training_run_id,
            "evaluation_run_id": experiment.evaluation_run_id,
            "loss_run_id": experiment.loss_run_id,
            "eval_summary": eval_summary,
        },
        "signals": signals,
        "candidates": candidates,
        "draft_next_spec": draft_spec,
    }


def render_draft_next_spec(bundle: dict[str, Any]) -> str:
    return yaml.safe_dump(bundle, sort_keys=False, allow_unicode=False)
