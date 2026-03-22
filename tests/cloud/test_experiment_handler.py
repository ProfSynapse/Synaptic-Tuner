from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from shared.experiment_tracking import Experiment, TrackingService
from shared.experiment_tracking.experiment_spec import DatasetSpec, EvaluationStageSpec, FeaturesStageSpec, LossStageSpec, TrainingStageSpec
from shared.experiment_tracking.schema import LossResult, RunRecord
from shared.experiment_tracking.per_example_loss import save_losses
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.experiment_handler import HFEvalStageRunner, HFLossStageRunner, HFTrainingStageRunner, StageResult


def _experiment() -> Experiment:
    return Experiment(
        experiment_id="exp_20260321_191536",
        name="resume-smoke",
        created_at="2026-03-21T19:15:36.480744+00:00",
        dataset_path="repo/dataset/sample.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
    )


def test_training_stage_runner_recovers_completed_training_without_resubmitting(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "training",
        status="running",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        bucket_id="test/toolset-training-artifacts",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
            "image": "unsloth/unsloth:latest",
        },
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    with patch.object(runner, "_bucket_has_path", return_value=True):
        with patch("tuner.handlers.experiment_handler.TrainingBackendRegistry.get") as mock_get:
            result = runner.run(spec=None, experiment=experiment)

    assert result.status == "completed"
    assert result.run_record is not None
    assert result.run_record.artifact_root == experiment.stage_details["training"]["artifact_root"]
    mock_get.assert_not_called()


def test_training_stage_runner_refuses_duplicate_submit_while_training_is_still_running(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "training",
        status="running",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        bucket_id="test/toolset-training-artifacts",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    with patch.object(runner, "_bucket_has_path", return_value=False):
        with pytest.raises(CloudProviderError, match="refusing to submit a duplicate training job"):
            runner.run(spec=None, experiment=experiment)


def test_loss_stage_runner_recovers_saved_losses_without_resubmitting(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "loss",
        status="running",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef/analysis/loss",
        bucket_id="test/toolset-training-artifacts",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    losses_dir = tmp_path / "loss-results"
    losses_dir.mkdir()
    save_losses(
        [LossResult(index=0, loss=0.25, num_completion_tokens=10, num_total_tokens=20, jsonl_hash="abcd1234")],
        losses_dir / "per_example_losses.jsonl",
    )

    runner = HFLossStageRunner(repo_root=repo_root, tracking_service=service)
    previous = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="exp-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-21T19:15:36+00:00",
            status="completed",
            output_dir="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            source_commit="deadbeefcafebabe",
            stage="training",
            tags={
                "provider": "hf_jobs",
                "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
                "bucket_id": "test/toolset-training-artifacts",
                "image": "unsloth/unsloth:latest",
            },
        ),
    )

    with patch.object(runner, "_download_results", return_value=losses_dir):
        with patch("tuner.handlers.experiment_handler.HFJobExecutor.submit") as mock_submit:
            result = runner.run(spec=None, experiment=experiment, previous=previous)

    assert result.status == "completed"
    assert len(result.loss_results) == 1
    mock_submit.assert_not_called()


def test_loss_stage_runner_allows_retry_when_running_stage_job_already_failed(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "loss",
        status="running",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef/analysis/loss",
        bucket_id="test/toolset-training-artifacts",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        job_ref="failed-loss-job",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    runner = HFLossStageRunner(repo_root=repo_root, tracking_service=service)

    with patch.object(runner, "_download_results", return_value=None):
        with patch.object(runner, "_inspect_job_stage", return_value="error"):
            result = runner._recover_existing_loss(experiment=experiment)

    assert result is None
    assert experiment.stage_details["loss"]["status"] == "failed"


def test_eval_stage_runner_requests_same_job_loss_when_spec_enables_loss(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    runner = HFEvalStageRunner(repo_root=repo_root, tracking_service=service)
    previous = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="exp-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-21T19:15:36+00:00",
            status="completed",
            output_dir="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            source_commit="deadbeefcafebabe",
            stage="training",
            tags={
                "provider": "hf_jobs",
                "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
                "bucket_id": "test/toolset-training-artifacts",
                "image": "unsloth/unsloth:latest",
            },
        ),
    )

    spec = type(
        "Spec",
        (),
        {
            "method": "sft",
            "provider": "hf_jobs",
            "dataset": type("Dataset", (), {"source": "professorsynapse/claudesidian-synthetic-dataset", "file": "train.jsonl", "identifier": "professorsynapse/claudesidian-synthetic-dataset/train.jsonl"})(),
            "training": type("Training", (), {"model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "max_seq_length": 2048})(),
            "evaluation": type("Evaluation", (), {"preset": "full", "scenarios": [], "tags": None, "gpu": None, "timeout_hours": None})(),
            "loss": type("Loss", (), {"enabled": True, "max_seq_length": 2048, "completion_only": True})(),
            "name": "resume-smoke",
        },
    )()

    captured = {}

    class _FakeCloudEvalHandler:
        def __init__(self, args):
            captured["args"] = args
            self.last_results_uri = "hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef/evaluations/vllm/20260321_200000"
            self.last_job_id = "eval-job-123"
            self.last_eval_payload = {"summary": {"passed": 1, "failed": 0, "warned": 0, "total": 1}}

        def handle(self):
            return 0

    with patch("tuner.handlers.experiment_handler.CloudEvalHandler", _FakeCloudEvalHandler):
        result = runner.run(spec=spec, experiment=experiment, previous=previous)

    assert result.status == "completed"
    assert captured["args"].with_loss is True
    assert captured["args"].loss_dataset_name == "professorsynapse/claudesidian-synthetic-dataset"
    assert captured["args"].loss_dataset_file == "train.jsonl"


def test_loss_stage_runner_recovers_embedded_eval_losses_without_resubmitting(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "evaluation",
        status="completed",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef/evaluations/vllm/20260321_200000",
        bucket_id="test/toolset-training-artifacts",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    losses_dir = tmp_path / "eval-analysis-results"
    losses_dir.mkdir()
    save_losses(
        [LossResult(index=0, loss=0.15, num_completion_tokens=12, num_total_tokens=22, jsonl_hash="wxyz5678")],
        losses_dir / "per_example_losses.jsonl",
    )

    runner = HFLossStageRunner(repo_root=repo_root, tracking_service=service)
    previous = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="exp-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-21T19:15:36+00:00",
            status="completed",
            output_dir="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
            source_commit="deadbeefcafebabe",
            stage="training",
            tags={
                "provider": "hf_jobs",
                "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
                "bucket_id": "test/toolset-training-artifacts",
                "image": "unsloth/unsloth:latest",
            },
        ),
    )

    with patch.object(runner, "_download_results", return_value=losses_dir):
        with patch("tuner.handlers.experiment_handler.HFJobExecutor.submit") as mock_submit:
            result = runner.run(spec=None, experiment=experiment, previous=previous)

    assert result.status == "completed"


def test_experiment_handler_applies_stage_overrides_to_spec(tmp_path: Path):
    from argparse import Namespace

    from shared.experiment_tracking import ExperimentSpec
    from tuner.handlers.experiment_handler import ExperimentHandler

    handler = ExperimentHandler(
        Namespace(
            json=False,
            only_stage="evaluation",
            from_stage=None,
            skip_stage=["loss", "analysis"],
        )
    )
    spec = ExperimentSpec(
        name="stage-selection",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )

    updated = handler._apply_stage_overrides(spec)

    assert updated.execution.only_stage == "evaluation"
    assert updated.execution.from_stage is None
    assert updated.execution.skip_stages == ["loss", "analysis"]
    assert updated.execution.selected_stages() == ["evaluation"]
