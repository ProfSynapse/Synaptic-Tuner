from __future__ import annotations

import json
from pathlib import Path

from shared.experiment_tracking import ExperimentOrchestrator, ExperimentSpec, StageResult, TrackingService
from shared.experiment_tracking.experiment_spec import DatasetSpec, EvaluationStageSpec, FeaturesStageSpec, LossStageSpec, TrainingStageSpec
from shared.experiment_tracking.schema import LossResult, RunRecord


class _StaticRunner:
    def __init__(self, result: StageResult):
        self.result = result

    def run(self, spec, experiment, previous=None):
        return self.result


class _FailIfCalledRunner:
    def run(self, spec, experiment, previous=None):
        raise AssertionError("runner should not have been called")


def _record(*, run_id: str, run_type: str, stage: str, status: str = "completed") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        run_type=run_type,
        name=f"{stage} run",
        timestamp="2026-03-21T18:00:00+00:00",
        status=status,
        output_dir=f"/tmp/{run_id}",
        provider="hf_jobs",
        artifact_root=f"/tmp/{run_id}",
        stage=stage,
    )


def test_experiment_orchestrator_runs_full_lifecycle(tmp_path: Path):
    spec = ExperimentSpec(
        name="smoke",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )

    training_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-training", run_type="sft", stage="training"),
            artifact_root="/tmp/train-artifacts",
        )
    )
    eval_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-eval", run_type="evaluation", stage="evaluation"),
            eval_payload={
                "summary": {"passed": 1, "failed": 0, "warned": 0, "total": 1},
                "records": [{"case_id": "ok", "passed": True}],
            },
            artifact_root="/tmp/eval-artifacts",
        )
    )
    loss_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-loss", run_type="loss", stage="loss"),
            loss_results=[LossResult(index=0, loss=0.4, num_completion_tokens=10, num_total_tokens=20, jsonl_hash="aaaa1111")],
            artifact_root="/tmp/loss-artifacts",
        )
    )

    orchestrator = ExperimentOrchestrator(
        tracking_service=TrackingService(tmp_path),
        training_runner=training_runner,
        eval_runner=eval_runner,
        loss_runner=loss_runner,
        base_dir=tmp_path,
    )

    experiment = orchestrator.run(spec, spec_path="/tmp/spec.yaml")

    assert experiment.status == "completed"
    assert experiment.training_run_id == "exp-training"
    assert experiment.evaluation_run_id == "exp-eval"
    assert experiment.loss_run_id == "exp-loss"
    assert experiment.stage_statuses == {
        "training": "completed",
        "evaluation": "completed",
        "loss": "completed",
    }
    assert experiment.derived_outputs["feature_dataset_csv"].endswith("feature_dataset.csv")

    summary_path = Path(experiment.derived_outputs["experiment_summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["run_count"] == 3


def test_experiment_orchestrator_resumes_from_completed_training_stage(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="resume-smoke",
        dataset_path="repo/dataset/sample.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
    )
    service.update_stage_details(
        experiment,
        "training",
        status="completed",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/20260321_191536-deadbeef",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
            "image": "unsloth/unsloth:latest",
        },
    )

    spec = ExperimentSpec(
        name="resume-smoke",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )

    eval_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-eval", run_type="evaluation", stage="evaluation"),
            eval_payload={
                "summary": {"passed": 1, "failed": 0, "warned": 0, "total": 1},
                "records": [{"case_id": "ok", "passed": True}],
            },
            artifact_root="/tmp/eval-artifacts",
        )
    )
    loss_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-loss", run_type="loss", stage="loss"),
            loss_results=[LossResult(index=0, loss=0.4, num_completion_tokens=10, num_total_tokens=20, jsonl_hash="aaaa1111")],
            artifact_root="/tmp/loss-artifacts",
        )
    )

    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_FailIfCalledRunner(),
        eval_runner=eval_runner,
        loss_runner=loss_runner,
        base_dir=tmp_path,
    )

    resumed = orchestrator.run(spec, spec_path="/tmp/spec.yaml", experiment=experiment)

    assert resumed.status == "completed"
    assert resumed.training_run_id == f"{experiment.experiment_id}-training"
    assert resumed.evaluation_run_id == "exp-eval"
    assert resumed.loss_run_id == "exp-loss"
    assert resumed.stage_statuses["training"] == "completed"
    assert resumed.stage_details["training"]["artifact_root"].startswith("hf://buckets/test/")
