"""
Behavioral tests for stage runner .run() methods.

Verifies:
- HFTrainingStageRunner.run() happy path & failure
- HFTrainingStageRunner._recover_existing_training() state machine
- HFEvalStageRunner.run() happy path & failure
- HFLossStageRunner.run() happy path & failure
- HFLossStageRunner._recover_existing_loss() state machine
- HFLossStageRunner._recover_loss_from_evaluation()
- Error/edge cases: missing artifacts, already-running experiments, job failures
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shared.experiment_tracking import (
    Experiment,
    ExperimentSpec,
    StageResult,
    TrackingService,
)
from shared.experiment_tracking.experiment_spec import (
    DatasetSpec,
    EvaluationStageSpec,
    LossStageSpec,
    TrainingStageSpec,
)
from shared.experiment_tracking.schema import RunRecord
from tuner.core.exceptions import CloudProviderError
from tuner.handlers.stages import (
    HFEvalStageRunner,
    HFLossStageRunner,
    HFTrainingStageRunner,
)


# =========================================================
# Helpers
# =========================================================

def _make_experiment(
    experiment_id: str = "exp_20260401_000000_abc",
    method: str = "sft",
    stage_details: dict | None = None,
) -> Experiment:
    """Build a minimal Experiment for testing."""
    return Experiment(
        experiment_id=experiment_id,
        name="test-experiment",
        created_at="2026-04-01T00:00:00+00:00",
        dataset_path="test-org/test-data",
        dataset_hash="abc123",
        base_model_name="test-org/test-model",
        provider="hf_jobs",
        method=method,
        stage_details=stage_details or {},
    )


def _make_spec(method: str = "sft") -> ExperimentSpec:
    """Build a minimal ExperimentSpec for testing."""
    return ExperimentSpec(
        name="test-experiment",
        provider="hf_jobs",
        method=method,
        dataset=DatasetSpec(source="test-org/test-data", file="train.jsonl"),
        training=TrainingStageSpec(
            model_name="test-org/test-model",
            gpu="a10g-small",
            timeout_hours=4.0,
        ),
        evaluation=EvaluationStageSpec(
            preset="quick",
            scenarios=["basic"],
        ),
        loss=LossStageSpec(enabled=True),
    )


def _make_training_result(
    status: str = "completed",
    artifact_prefix: str = "runs/hf_jobs/sft/20260401_000000-abc12345",
    bucket_id: str = "user/bucket",
) -> StageResult:
    """Build a StageResult representing a completed training run."""
    return StageResult(
        status=status,
        run_record=RunRecord(
            run_id="exp_20260401_000000_abc-training",
            run_type="sft",
            name="test-experiment training",
            timestamp="2026-04-01T00:00:00+00:00",
            status=status,
            output_dir=f"hf://buckets/{bucket_id}/{artifact_prefix}",
            model_name="test-org/test-model",
            dataset_source="test-org/test-data/train.jsonl",
            provider="hf_jobs",
            artifact_backend="hf_bucket",
            artifact_root=f"hf://buckets/{bucket_id}/{artifact_prefix}",
            source_commit="abc12345def67890",
            stage="training",
            tags={
                "provider": "hf_jobs",
                "artifact_prefix": artifact_prefix,
                "bucket_id": bucket_id,
                "image": "unsloth/unsloth:latest",
            },
        ),
        artifact_root=f"hf://buckets/{bucket_id}/{artifact_prefix}",
    )


def _mock_backend(exit_code: int = 0):
    """Build a mock training backend that returns a given exit code."""
    backend = MagicMock()
    backend.validate_environment.return_value = (True, None)
    mock_config = MagicMock()
    mock_config.artifact_identifier = "user/bucket"
    mock_config.repo_commit = "abc12345def67890"
    mock_config.provider = "hf_jobs"
    mock_config.method = "sft"
    mock_config.hf_flavor = "a10g-small"
    mock_config.gpu_type = "a10g-small"
    mock_config.cloud_image = "unsloth/unsloth:latest"
    mock_config.artifact_mount_path = "/workspace/outputs"
    backend.load_config.return_value = mock_config
    backend.execute.return_value = exit_code
    backend.last_artifact_prefix = "runs/hf_jobs/sft/20260401_000000-abc12345"
    backend.last_bucket_id = "user/bucket"
    backend.last_job_id = "job-123"
    return backend


# =========================================================
# HFTrainingStageRunner.run() — Happy Path
# =========================================================

class TestTrainingRunnerHappyPath:
    @patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry")
    def test_run_completes_successfully(self, mock_registry, tmp_path):
        """run() should execute backend, record details, return completed StageResult."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment()
        spec = _make_spec()
        backend = _mock_backend(exit_code=0)
        mock_registry.get.return_value = backend

        # Mock _resolve_bucket_id to pass through
        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")

        result = runner.run(spec, experiment)

        assert result.status == "completed"
        assert result.run_record is not None
        assert result.run_record.run_type == "sft"
        assert result.run_record.stage == "training"
        assert result.artifact_root is not None
        assert "hf://buckets/" in result.artifact_root
        backend.execute.assert_called_once()

    @patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry")
    def test_run_maps_seed_and_beta_onto_config(self, mock_registry, tmp_path):
        """spec.training.seed/beta must be mapped onto the loaded cloud config."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment()
        spec = _make_spec(method="dpo")
        spec.training.seed = 7
        spec.training.beta = 0.5
        backend = _mock_backend(exit_code=0)
        backend.load_config.return_value.method = "dpo"
        mock_registry.get.return_value = backend
        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")

        runner.run(spec, experiment)

        config = backend.load_config.return_value
        assert config.seed == 7
        assert config.beta == 0.5

    @patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry")
    def test_run_does_not_map_beta_for_sft(self, mock_registry, tmp_path):
        """beta is DPO/KTO-only; an SFT run must not have beta mapped onto config."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment()
        spec = _make_spec(method="sft")
        spec.training.beta = 0.5
        backend = _mock_backend(exit_code=0)
        backend.load_config.return_value.method = "sft"
        mock_registry.get.return_value = backend
        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")

        runner.run(spec, experiment)

        config = backend.load_config.return_value
        assert "beta" not in config.__dict__

    @patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry")
    def test_run_records_failure_on_nonzero_exit(self, mock_registry, tmp_path):
        """run() should return failed StageResult when backend returns exit_code=1."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment()
        spec = _make_spec()
        backend = _mock_backend(exit_code=1)
        mock_registry.get.return_value = backend

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")

        result = runner.run(spec, experiment)

        assert result.status == "failed"
        assert result.run_record.status == "failed"

    @patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry")
    def test_run_raises_on_invalid_environment(self, mock_registry, tmp_path):
        """run() should raise CloudProviderError when environment validation fails."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment()
        spec = _make_spec()
        backend = MagicMock()
        backend.validate_environment.return_value = (False, "Missing HF_TOKEN")
        mock_registry.get.return_value = backend

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")

        with pytest.raises(CloudProviderError, match="Missing HF_TOKEN"):
            runner.run(spec, experiment)


# =========================================================
# HFTrainingStageRunner._recover_existing_training()
# =========================================================

class TestTrainingRecoveryStateMachine:
    def test_returns_none_when_no_training_details(self, tmp_path):
        """No stage_details['training'] means nothing to recover."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={})

        result = runner._recover_existing_training(experiment=experiment)
        assert result is None

    def test_returns_none_when_status_not_recoverable(self, tmp_path):
        """Only 'running' and 'completed' are recoverable statuses."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={
            "training": {
                "status": "failed",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401",
            }
        })

        result = runner._recover_existing_training(experiment=experiment)
        assert result is None

    def test_returns_none_when_missing_bucket_id(self, tmp_path):
        """Recovery requires bucket_id."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={
            "training": {
                "status": "completed",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401",
            }
        })

        result = runner._recover_existing_training(experiment=experiment)
        assert result is None

    def test_recovers_completed_training_with_artifacts(self, tmp_path):
        """Should return StageResult when training artifacts exist in bucket."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "training": {
                "status": "completed",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401",
            }
        })

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")
        runner._bucket_has_path = MagicMock(return_value=True)

        result = runner._recover_existing_training(experiment=experiment)

        assert result is not None
        assert result.status == "completed"
        assert result.run_record.stage == "training"
        assert result.artifact_root == "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401"

    def test_raises_when_already_running_with_job_ref(self, tmp_path):
        """Should raise when training is running (has job_ref) and artifacts incomplete."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "training": {
                "status": "running",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401",
                "job_ref": "job-existing-123",
            }
        })

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")
        runner._bucket_has_path = MagicMock(return_value=False)

        with pytest.raises(CloudProviderError, match="already has a running training"):
            runner._recover_existing_training(experiment=experiment)

    def test_marks_failed_when_running_without_job_ref(self, tmp_path):
        """Running without job_ref means orphaned state — mark failed, return None."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "training": {
                "status": "running",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/sft/20260401",
                # no job_ref
            }
        })

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")
        runner._bucket_has_path = MagicMock(return_value=False)

        result = runner._recover_existing_training(experiment=experiment)
        assert result is None
        # Check that stage was marked failed
        assert experiment.stage_details["training"]["status"] == "failed"

    def test_recovery_updates_bucket_id_when_resolved_differs(self, tmp_path):
        """When _resolve_bucket_id returns a different ID, artifact_root is updated."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "training": {
                "status": "completed",
                "bucket_id": "short-name",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "artifact_root": "hf://buckets/short-name/runs/hf_jobs/sft/20260401",
            }
        })

        runner._resolve_bucket_id = MagicMock(return_value="user/resolved-name")
        runner._bucket_has_path = MagicMock(return_value=True)

        result = runner._recover_existing_training(experiment=experiment)

        assert result is not None
        assert "user/resolved-name" in result.artifact_root

    def test_recovery_checks_grpo_completion_suffixes(self, tmp_path):
        """GRPO method should check for logs/training_latest.jsonl, not training_lineage.json."""
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(method="grpo", stage_details={
            "training": {
                "status": "completed",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/grpo/20260401",
                "artifact_root": "hf://buckets/user/bucket/runs/hf_jobs/grpo/20260401",
            }
        })

        runner._resolve_bucket_id = MagicMock(return_value="user/bucket")
        calls = []

        def track_bucket_has_path(*, bucket_id, prefix, suffix):
            calls.append(suffix)
            return suffix == "logs/training_latest.jsonl"

        runner._bucket_has_path = track_bucket_has_path

        result = runner._recover_existing_training(experiment=experiment)
        assert result is not None
        # Should check GRPO-specific suffixes
        grpo_suffixes = runner._training_completion_suffixes("grpo")
        assert "logs/training_latest.jsonl" in grpo_suffixes


# =========================================================
# HFEvalStageRunner.run() — Happy Path & Errors
# =========================================================

class TestEvalRunnerBehavior:
    @patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler")
    def test_run_completes_successfully(self, mock_handler_cls, tmp_path):
        """run() should invoke CloudEvalHandler.handle() and return completed result."""
        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = _make_training_result()

        mock_handler = MagicMock()
        mock_handler.handle.return_value = 0
        mock_handler.last_results_uri = "hf://buckets/user/bucket/eval-results"
        mock_handler.last_job_id = "eval-job-123"
        mock_handler.last_eval_payload = {"accuracy": 0.95}
        mock_handler_cls.return_value = mock_handler

        result = runner.run(spec, experiment, previous=previous)

        assert result.status == "completed"
        assert result.run_record.run_type == "evaluation"
        assert result.run_record.stage == "evaluation"
        assert result.eval_payload == {"accuracy": 0.95}
        mock_handler.handle.assert_called_once()

    @patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler")
    def test_run_returns_failed_on_nonzero_exit(self, mock_handler_cls, tmp_path):
        """run() should return failed StageResult when eval handler returns exit_code=1."""
        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = _make_training_result()

        mock_handler = MagicMock()
        mock_handler.handle.return_value = 1
        mock_handler.last_results_uri = None
        mock_handler.last_job_id = "eval-job-fail"
        mock_handler.last_eval_payload = None
        mock_handler_cls.return_value = mock_handler

        result = runner.run(spec, experiment, previous=previous)

        assert result.status == "failed"
        assert result.run_record.status == "failed"

    def test_run_raises_without_previous_result(self, tmp_path):
        """run() should raise CloudProviderError when no previous training result."""
        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()

        with pytest.raises(CloudProviderError, match="completed training run"):
            runner.run(spec, experiment, previous=None)

    def test_run_raises_when_training_missing_tags(self, tmp_path):
        """run() should raise when training result lacks artifact_prefix or bucket_id."""
        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = StageResult(
            status="completed",
            run_record=RunRecord(
                run_id="training-run",
                run_type="sft",
                name="training",
                timestamp="2026-04-01T00:00:00+00:00",
                status="completed",
                output_dir="",
                tags={},  # Missing artifact_prefix and bucket_id
            ),
        )

        with pytest.raises(CloudProviderError, match="artifact prefix and bucket"):
            runner.run(spec, experiment, previous=previous)

    def test_use_same_job_loss_returns_true_for_same_job_mode(self):
        """_use_same_job_loss should return True when mode=same_job and loss enabled."""
        spec = _make_spec()
        spec.post_training.mode = "same_job"
        spec.loss.enabled = True
        assert HFEvalStageRunner._use_same_job_loss(spec) is True

    def test_use_same_job_loss_returns_false_for_parallel_mode(self):
        """_use_same_job_loss should return False for default parallel mode."""
        spec = _make_spec()
        assert HFEvalStageRunner._use_same_job_loss(spec) is False


# =========================================================
# HFLossStageRunner._recover_existing_loss()
# =========================================================

class TestLossRecoveryStateMachine:
    def test_returns_none_when_no_loss_details(self, tmp_path):
        """No stage_details['loss'] means nothing to recover."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={})

        result = runner._recover_existing_loss(experiment=experiment)
        assert result is None

    def test_returns_none_when_status_not_recoverable(self, tmp_path):
        """Only 'running' and 'completed' are recoverable statuses."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "failed",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
            }
        })

        result = runner._recover_existing_loss(experiment=experiment)
        assert result is None

    def test_recovers_completed_loss_with_downloaded_results(self, tmp_path):
        """Should return StageResult when per_example_losses.jsonl exists."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "completed",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
            }
        })

        # Create fake downloaded results
        results_dir = tmp_path / "fake-results"
        results_dir.mkdir()
        losses_file = results_dir / "per_example_losses.jsonl"
        losses_file.write_text(
            '{"index":0,"loss":0.5,"num_completion_tokens":10,"num_total_tokens":20,"jsonl_hash":"abcd1234"}\n'
        )

        runner._download_results = MagicMock(return_value=results_dir)

        result = runner._recover_existing_loss(experiment=experiment)

        assert result is not None
        assert result.status == "completed"
        assert len(result.loss_results) == 1
        assert result.loss_results[0].loss == 0.5

    def test_returns_none_when_download_finds_no_losses(self, tmp_path):
        """Recovery returns None when download succeeds but no losses file found."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "completed",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
            }
        })

        # Download succeeds but directory is empty
        results_dir = tmp_path / "empty-results"
        results_dir.mkdir()
        runner._download_results = MagicMock(return_value=results_dir)

        result = runner._recover_existing_loss(experiment=experiment)
        assert result is None

    def test_raises_when_loss_running_with_active_job(self, tmp_path):
        """Running loss job with no complete results should raise to prevent duplicates."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "running",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "job_ref": "loss-job-456",
            }
        })

        runner._download_results = MagicMock(return_value=None)
        runner._inspect_job_stage = MagicMock(return_value="running")

        with pytest.raises(CloudProviderError, match="already has a running loss"):
            runner._recover_existing_loss(experiment=experiment)

    def test_marks_failed_when_job_errored(self, tmp_path):
        """When inspected job shows error/failed, mark status failed and return None."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "running",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "job_ref": "loss-job-456",
            }
        })

        runner._download_results = MagicMock(return_value=None)
        runner._inspect_job_stage = MagicMock(return_value="failed")

        result = runner._recover_existing_loss(experiment=experiment)
        assert result is None
        assert experiment.stage_details["loss"]["status"] == "failed"

    def test_marks_failed_when_job_completed_but_no_results(self, tmp_path):
        """Job completed but no loss results downloaded means loss extraction failed."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "loss": {
                "status": "running",
                "artifact_root": "hf://buckets/user/bucket/analysis/loss",
                "bucket_id": "user/bucket",
                "artifact_prefix": "runs/hf_jobs/sft/20260401",
                "job_ref": "loss-job-456",
            }
        })

        runner._download_results = MagicMock(return_value=None)
        runner._inspect_job_stage = MagicMock(return_value="completed")

        result = runner._recover_existing_loss(experiment=experiment)
        assert result is None
        assert experiment.stage_details["loss"]["status"] == "failed"


# =========================================================
# HFLossStageRunner._recover_loss_from_evaluation()
# =========================================================

class TestLossRecoveryFromEvaluation:
    def test_returns_none_when_no_eval_details(self, tmp_path):
        """No evaluation stage details means nothing to recover from."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={})

        result = runner._recover_loss_from_evaluation(experiment=experiment)
        assert result is None

    def test_returns_none_when_eval_not_completed(self, tmp_path):
        """Eval must be completed to try recovering embedded loss."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        experiment = _make_experiment(stage_details={
            "evaluation": {
                "status": "running",
                "artifact_root": "hf://buckets/user/bucket/eval-results",
                "bucket_id": "user/bucket",
            }
        })

        result = runner._recover_loss_from_evaluation(experiment=experiment)
        assert result is None

    def test_recovers_embedded_loss_from_eval_artifacts(self, tmp_path):
        """Should find per_example_losses.jsonl under eval_artifact_root/analysis."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "evaluation": {
                "status": "completed",
                "artifact_root": "hf://buckets/user/bucket/eval-results",
                "bucket_id": "user/bucket",
            }
        })

        # Create fake analysis dir with losses
        results_dir = tmp_path / "analysis-results"
        results_dir.mkdir()
        (results_dir / "per_example_losses.jsonl").write_text(
            '{"index":0,"loss":0.3,"num_completion_tokens":5,"num_total_tokens":15,"jsonl_hash":"efgh5678"}\n'
        )

        runner._download_results = MagicMock(return_value=results_dir)

        result = runner._recover_loss_from_evaluation(experiment=experiment)

        assert result is not None
        assert result.status == "completed"
        assert len(result.loss_results) == 1
        assert result.loss_results[0].loss == 0.3
        assert "analysis" in result.artifact_root

    def test_returns_none_when_no_losses_in_eval(self, tmp_path):
        """No per_example_losses.jsonl under eval analysis means no recovery."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        experiment = _make_experiment(stage_details={
            "evaluation": {
                "status": "completed",
                "artifact_root": "hf://buckets/user/bucket/eval-results",
                "bucket_id": "user/bucket",
            }
        })

        runner._download_results = MagicMock(return_value=None)

        result = runner._recover_loss_from_evaluation(experiment=experiment)
        assert result is None


# =========================================================
# HFLossStageRunner.run() — Happy Path & Errors
# =========================================================

class TestLossRunnerRun:
    @patch("tuner.handlers.stages.hf_loss_stage.load_huggingface_hub")
    @patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor")
    @patch("tuner.handlers.stages.hf_loss_stage.TrainingBackendRegistry")
    def test_run_raises_without_previous_result(self, mock_registry, mock_executor, mock_load_hub, tmp_path):
        """run() should raise when no previous training result provided."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()

        with pytest.raises(CloudProviderError, match="completed training run"):
            runner.run(spec, experiment, previous=None)

    @patch("tuner.handlers.stages.hf_loss_stage.load_huggingface_hub")
    @patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor")
    @patch("tuner.handlers.stages.hf_loss_stage.TrainingBackendRegistry")
    def test_run_raises_when_training_missing_tags(self, mock_registry, mock_executor, mock_load_hub, tmp_path):
        """run() should raise when training result lacks bucket tags."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = StageResult(
            status="completed",
            run_record=RunRecord(
                run_id="training-run",
                run_type="sft",
                name="training",
                timestamp="2026-04-01T00:00:00+00:00",
                status="completed",
                output_dir="",
                tags={},
            ),
        )

        with pytest.raises(CloudProviderError, match="artifact prefix and bucket"):
            runner.run(spec, experiment, previous=previous)

    @patch("tuner.handlers.stages.hf_loss_stage.build_hf_job_secrets")
    @patch("tuner.handlers.stages.hf_loss_stage.build_bash_command")
    @patch("tuner.handlers.stages.hf_loss_stage.load_huggingface_hub")
    @patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor")
    @patch("tuner.handlers.stages.hf_loss_stage.TrainingBackendRegistry")
    def test_run_submits_job_and_returns_result(
        self, mock_registry, mock_executor_cls, mock_load_hub, mock_build_cmd, mock_secrets, tmp_path
    ):
        """run() should submit HF job, poll it, and return result."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = _make_training_result()

        # Mock the build_command to avoid filesystem dependencies
        runner._build_command = MagicMock(return_value="echo test")
        runner._poll_job = MagicMock(return_value=0)
        runner._download_results = MagicMock(return_value=None)

        mock_hub = MagicMock()
        mock_load_hub.return_value = mock_hub

        mock_executor = MagicMock()
        mock_submission = MagicMock()
        mock_submission.job_id = "loss-job-789"
        mock_executor.submit.return_value = mock_submission
        mock_executor_cls.return_value = mock_executor

        mock_build_cmd.return_value = "bash -c 'echo test'"
        mock_secrets.return_value = {}

        # Mock backend for image resolution fallback
        mock_backend = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.cloud_image = "unsloth/unsloth:latest"
        mock_backend.load_config.return_value = mock_cfg
        mock_registry.get.return_value = mock_backend

        result = runner.run(spec, experiment, previous=previous)

        assert result.status == "completed"
        assert result.run_record.run_type == "loss"
        assert result.run_record.stage == "loss"
        assert result.run_record.job_ref == "loss-job-789"
        runner._poll_job.assert_called_once()

    @patch("tuner.handlers.stages.hf_loss_stage.build_hf_job_secrets")
    @patch("tuner.handlers.stages.hf_loss_stage.build_bash_command")
    @patch("tuner.handlers.stages.hf_loss_stage.load_huggingface_hub")
    @patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor")
    @patch("tuner.handlers.stages.hf_loss_stage.TrainingBackendRegistry")
    def test_run_returns_failed_when_poll_fails(
        self, mock_registry, mock_executor_cls, mock_load_hub, mock_build_cmd, mock_secrets, tmp_path
    ):
        """run() should return failed status when job poll returns nonzero."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        spec = _make_spec()
        experiment = _make_experiment()
        previous = _make_training_result()

        runner._build_command = MagicMock(return_value="echo test")
        runner._poll_job = MagicMock(return_value=1)

        mock_hub = MagicMock()
        mock_load_hub.return_value = mock_hub

        mock_executor = MagicMock()
        mock_submission = MagicMock()
        mock_submission.job_id = "loss-job-fail"
        mock_executor.submit.return_value = mock_submission
        mock_executor_cls.return_value = mock_executor

        mock_build_cmd.return_value = "bash -c 'echo test'"
        mock_secrets.return_value = {}

        mock_backend = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.cloud_image = "unsloth/unsloth:latest"
        mock_backend.load_config.return_value = mock_cfg
        mock_registry.get.return_value = mock_backend

        result = runner.run(spec, experiment, previous=previous)

        assert result.status == "failed"
        assert result.run_record.status == "failed"


# =========================================================
# HFLossStageRunner._poll_job()
# =========================================================

class TestLossRunnerPollJob:
    def test_poll_returns_zero_on_completed(self, tmp_path):
        """_poll_job returns 0 when job status transitions to completed."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        mock_hub = MagicMock()
        mock_job_info = MagicMock()
        mock_job_info.status = MagicMock(stage="completed")
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = ""

        with patch("time.sleep"):
            result = runner._poll_job(mock_hub, job_id="job-1", timeout_hours=0.001)

        assert result == 0

    def test_poll_returns_one_on_error(self, tmp_path):
        """_poll_job returns 1 when job status shows error."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        mock_hub = MagicMock()
        mock_job_info = MagicMock()
        mock_job_info.status = MagicMock(stage="error")
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = ""

        with patch("time.sleep"):
            result = runner._poll_job(mock_hub, job_id="job-1", timeout_hours=0.001)

        assert result == 1

    def test_poll_returns_one_on_timeout(self, tmp_path):
        """_poll_job returns 1 when timeout expires while job still running."""
        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)

        mock_hub = MagicMock()
        mock_job_info = MagicMock()
        mock_job_info.status = MagicMock(stage="running")
        mock_hub.inspect_job.return_value = mock_job_info
        mock_hub.fetch_job_logs.return_value = ""

        with patch("time.sleep"):
            # Very small timeout so poll exits immediately
            result = runner._poll_job(mock_hub, job_id="job-1", timeout_hours=0.0001)

        assert result == 1


# =========================================================
# HFTrainingStageRunner._training_completion_suffixes()
# =========================================================

class TestTrainingCompletionSuffixes:
    def test_sft_method(self, tmp_path):
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        suffixes = runner._training_completion_suffixes("sft")
        assert "training_lineage.json" in suffixes

    def test_kto_method(self, tmp_path):
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        suffixes = runner._training_completion_suffixes("kto")
        assert "training_lineage.json" in suffixes

    def test_grpo_method(self, tmp_path):
        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        suffixes = runner._training_completion_suffixes("grpo")
        assert "logs/training_latest.jsonl" in suffixes
        assert "final_model/adapter_config.json" in suffixes
        assert "training_lineage.json" not in suffixes
