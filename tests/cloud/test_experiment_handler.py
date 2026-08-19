from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _emit_stage_event_process(log_dir: str, index: int, start, results) -> None:
    try:
        from shared.cloud_stage_logging import CloudStageLogger

        start.wait(timeout=20)
        CloudStageLogger(Path(log_dir), stage="evaluation").emit(
            f"process-{index}"
        )
        results.put((0, index, ""))
    except BaseException as exc:
        results.put((1, index, repr(exc)))

from shared.experiment_tracking import Experiment, ExperimentSpec, TrackingService
from shared.experiment_tracking.experiment_spec import DatasetSpec, EvaluationStageSpec, FeaturesStageSpec, LossStageSpec, TrainingStageSpec
from shared.experiment_tracking.schema import LossResult, RunRecord
from shared.experiment_tracking.per_example_loss import save_losses
from tuner.backends.training.cloud.base_cloud import RepoSource
from tuner.core.config import CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError
from tuner.project.source_bundle import GitSource, RepositoryLocation
from tuner.project import ProjectContext
from tuner.project.source_bundle import SourceLock
from tuner.cloud.hf_provisioning import (
    EVIDENCE_SCHEMA_VERSION,
    canonical_json_bytes,
    consume_hf_source_transport,
    prepare_hf_source_transport,
)
from tuner.cloud.runtime_layout import build_runtime_layout
from shared.experiment_tracking import StageResult
from tuner.handlers.stages import HFEvalStageRunner, HFLossStageRunner, HFTrainingStageRunner
from tuner.handlers.stages._util import (
    HFSourcePreparation,
    hf_source_preparation_from_consumable,
)


_FIXTURE_HEAD = "0123456789abcdef0123456789abcdef01234567"
_ENGINE_REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_hf_source_transport(
    *,
    service: TrackingService,
    experiment: Experiment,
    acknowledged: bool,
    record_prepared: bool = True,
) -> HFSourcePreparation:
    """Create real immutable descriptor/evidence fixtures without provider calls."""

    repo_root = _ENGINE_REPO_ROOT
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = GitSource(
        location=RepositoryLocation.parse("https://github.com/test/repo.git"),
        branch="main",
        commit=commit,
        dirty=False,
        pushed=True,
    )
    source_lock = SourceLock(
        run_id=experiment.experiment_id,
        mode="standalone",
        project_source=source,
        engine_source=source,
        project={},
        configuration={},
    )
    context = ProjectContext.standalone(engine_root=repo_root)
    service.persist_source_lock(experiment, source_lock)
    descriptor_path = (
        service.base_dir
        / "experiments"
        / experiment.experiment_id
        / "cloud"
        / "hf"
        / "source-transport"
        / "descriptor.json"
    )
    descriptor_uri = service.tracking_uri(descriptor_path)
    prepared = prepare_hf_source_transport(
        context,
        source_lock=source_lock,
        source_lock_uri=experiment.source_lock_uri,
        descriptor_uri=descriptor_uri,
        transport_root=descriptor_path.parent.resolve(),
        volume_source="professorsynapse/toolset-training-bootstrap",
        path_prefix="synaptic/source-transport",
    )
    if record_prepared:
        service.record_source_transport_prepared(
            experiment,
            uri=prepared.descriptor_uri,
            sha256=prepared.descriptor_sha256,
        )
    if not acknowledged:
        return HFSourcePreparation(
            source_lock=prepared.source_lock,
            source_lock_sha256=str(prepared.descriptor["source_lock"]["sha256"]),
            source_lock_uri=experiment.source_lock_uri,
            volume_spec=None,
            runtime_layout=build_runtime_layout(context),
            staging_root=prepared.root,
            descriptor_uri=prepared.descriptor_uri,
            descriptor_sha256=prepared.descriptor_sha256,
            source_transport_state="PREPARED",
        )

    if not record_prepared:
        raise AssertionError("Acknowledged fixtures must first record PREPARED state")
    descriptor = prepared.descriptor
    volume = descriptor["volume"]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "descriptor": {
            "uri": prepared.descriptor_uri,
            "sha256": prepared.descriptor_sha256,
        },
        "run_id": experiment.experiment_id,
        "provider": "hf_jobs",
        "profile": "C",
        "volume": {
            "source": volume["source"],
            "path": volume["path"],
            "type": "bucket",
            "read_only": True,
        },
        "bundle_sha256": descriptor["bundle"]["content_sha256"],
        "capsule_manifest_sha256": descriptor["capsule"]["manifest"]["sha256"],
        "source_lock_sha256": descriptor["source_lock"]["sha256"],
        "checkout_policy_sha256": descriptor["checkout_policy"]["sha256"],
        "status": "provisioned",
        "authority": "protected_workflow",
        "actor": "fixture-workflow",
        "asserted_at": "2026-08-19T12:00:00Z",
        "provider_receipt_id": "fixture-receipt",
    }
    evidence_path = descriptor_path.with_name("provisioning-evidence.json")
    evidence_path.write_bytes(canonical_json_bytes(evidence))
    evidence_uri = service.tracking_uri(evidence_path)
    evidence_sha256 = hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()
    service.record_provisioning_acknowledged(
        experiment,
        uri=evidence_uri,
        sha256=evidence_sha256,
    )
    consumed = consume_hf_source_transport(
        context,
        transport_root=prepared.root,
        descriptor_uri=prepared.descriptor_uri,
        source_lock_uri=experiment.source_lock_uri,
        evidence=evidence,
    )
    return hf_source_preparation_from_consumable(
        consumed,
        runtime_layout=build_runtime_layout(context),
        provisioning_evidence_uri=evidence_uri,
    )


def _canonical_repo_source() -> RepoSource:
    source = GitSource(
        location=RepositoryLocation.parse("https://github.com/test/repo.git"),
        branch="main",
        commit=_FIXTURE_HEAD,
        dirty=False,
        pushed=True,
    )
    return RepoSource(
        url=source.location.canonical_url,
        branch=source.branch,
        commit=source.commit,
        canonical_source=source,
    )


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
        with patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry.get") as mock_get:
            result = runner.run(spec=None, experiment=experiment)

    assert result.status == "completed"
    assert result.run_record is not None
    assert result.run_record.artifact_root == experiment.stage_details["training"]["artifact_root"]
    mock_get.assert_not_called()


def test_training_stage_runner_resolves_bare_bucket_ids_during_recovery(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "training",
        status="running",
        artifact_root="hf://buckets/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        bucket_id="toolset-training-artifacts",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "toolset-training-artifacts",
            "image": "unsloth/unsloth:latest",
        },
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
        with patch.object(runner, "_bucket_has_path", return_value=True):
            result = runner.run(spec=None, experiment=experiment)

    assert result.status == "completed"
    assert experiment.stage_details["training"]["bucket_id"] == "professorsynapse/toolset-training-artifacts"
    assert experiment.stage_details["training"]["artifact_root"].startswith(
        "hf://buckets/professorsynapse/toolset-training-artifacts/"
    )


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
        job_ref="live-training-job",
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


def test_training_stage_runner_allows_retry_when_running_stage_has_no_job_ref(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "training",
        status="running",
        artifact_root="hf://buckets/toolset-training-artifacts/runs/hf_jobs/sft/20260321_191536-deadbeef",
        artifact_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef",
        bucket_id="toolset-training-artifacts",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_191536-deadbeef",
            "bucket_id": "toolset-training-artifacts",
        },
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
        with patch.object(runner, "_bucket_has_path", return_value=False):
            result = runner._recover_existing_training(experiment=experiment)

    assert result is None
    assert experiment.stage_details["training"]["status"] == "failed"


def test_training_stage_runner_recovers_grpo_training_from_final_model_artifact(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    experiment.method = "grpo"
    service.save_experiment(experiment)
    service.update_stage_details(
        experiment,
        "training",
        status="running",
        artifact_root="hf://buckets/test/toolset-training-artifacts/runs/hf_jobs/grpo/20260321_191536-deadbeef",
        artifact_prefix="runs/hf_jobs/grpo/20260321_191536-deadbeef",
        bucket_id="test/toolset-training-artifacts",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/grpo/20260321_191536-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
            "image": "unsloth/unsloth:latest",
        },
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    def _fake_bucket_has_path(*, bucket_id: str, prefix: str, suffix: str) -> bool:
        return suffix == "final_model/adapter_config.json"

    with patch.object(runner, "_bucket_has_path", side_effect=_fake_bucket_has_path):
        result = runner.run(spec=None, experiment=experiment)

    assert result.status == "completed"
    assert result.run_record is not None
    assert result.run_record.artifact_root == experiment.stage_details["training"]["artifact_root"]


def test_training_stage_runner_forwards_lora_variant_fields_to_cloud_config(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)

    spec = ExperimentSpec(
        name="lora-variant-smoke",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(
            model_name="Qwen/Qwen3-4B",
            max_steps=20,
            lora_r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            use_dora=True,
            use_rslora=True,
            init_lora_weights="loftq",
            lora_target_modules="all-linear",
        ),
        evaluation=EvaluationStageSpec(enabled=False),
        loss=LossStageSpec(enabled=False),
        features=FeaturesStageSpec(enabled=False),
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.execute.return_value = 0
    backend.load_config.return_value = CloudTrainingConfig(
        method="sft",
        platform="hf_jobs",
        config_path=Path("/fake/config.yaml"),
        trainer_dir=Path("/fake/trainer"),
        model_name="base",
        dataset_file="dataset.jsonl",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        gpu_type="a100-large",
        timeout_hours=4.0,
        cloud_image="unsloth/unsloth:latest",
        hf_flavor="a100-large",
        artifact_backend="hf_bucket",
        artifact_identifier="professorsynapse/toolset-training-artifacts",
        artifact_mount_path="/workspace/outputs",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="deadbeefcafebabe",
    )

    backend.prepare_source.return_value = _install_hf_source_transport(
        service=service,
        experiment=experiment,
        acknowledged=False,
        record_prepared=False,
    )
    with patch.object(runner, "_recover_existing_training", return_value=None):
        with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
            with patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry.get", return_value=backend):
                with pytest.raises(CloudProviderError, match="awaits separately approved external provisioning"):
                    runner.run(spec=spec, experiment=experiment)

    config = backend.prepare_source.call_args.args[0]
    backend.execute.assert_not_called()
    assert experiment.source_transport_state == "PREPARED"
    assert config.lora_r == 128
    assert config.lora_alpha == 256
    assert config.lora_dropout == 0.05
    assert config.use_dora is True
    assert config.use_rslora is True
    assert config.init_lora_weights == "loftq"
    assert config.lora_target_modules == "all-linear"


def test_training_stage_runner_forwards_stage_pip_packages_to_cloud_config(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)

    spec = ExperimentSpec(
        name="stage-pip-packages-smoke",
        provider="hf_jobs",
        method="sft",
        objective="train_only",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(
            model_name="Qwen/Qwen3-4B",
            max_steps=20,
            pip_packages=["unsloth==2026.4.2", "transformers==5.3.0"],
        ),
        evaluation=EvaluationStageSpec(enabled=False),
        loss=LossStageSpec(enabled=False),
        features=FeaturesStageSpec(enabled=False),
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.execute.return_value = 0
    backend.load_config.return_value = CloudTrainingConfig(
        method="sft",
        platform="hf_jobs",
        config_path=Path("/fake/config.yaml"),
        trainer_dir=Path("/fake/trainer"),
        model_name="base",
        dataset_file="dataset.jsonl",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        gpu_type="a100-large",
        timeout_hours=4.0,
        cloud_image="unsloth/unsloth:latest",
        hf_flavor="a100-large",
        artifact_backend="hf_bucket",
        artifact_identifier="professorsynapse/toolset-training-artifacts",
        artifact_mount_path="/workspace/outputs",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="deadbeefcafebabe",
    )

    backend.prepare_source.return_value = _install_hf_source_transport(
        service=service,
        experiment=experiment,
        acknowledged=False,
        record_prepared=False,
    )
    with patch.object(runner, "_recover_existing_training", return_value=None):
        with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
            with patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry.get", return_value=backend):
                with pytest.raises(CloudProviderError, match="awaits separately approved external provisioning"):
                    runner.run(spec=spec, experiment=experiment)

    config = backend.prepare_source.call_args.args[0]
    backend.execute.assert_not_called()
    assert experiment.source_transport_state == "PREPARED"
    assert config.pip_packages == ["unsloth==2026.4.2", "transformers==5.3.0"]


def test_training_stage_runner_does_not_forward_evolutionary_defaults_when_disabled(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)

    spec = ExperimentSpec(
        name="non-evolutionary-smoke",
        provider="hf_jobs",
        method="sft",
        objective="train_only",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(
            model_name="Qwen/Qwen3-4B",
            max_steps=20,
        ),
        evaluation=EvaluationStageSpec(enabled=False),
        loss=LossStageSpec(enabled=False),
        features=FeaturesStageSpec(enabled=False),
    )

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.execute.return_value = 0
    backend.load_config.return_value = CloudTrainingConfig(
        method="sft",
        platform="hf_jobs",
        config_path=Path("/fake/config.yaml"),
        trainer_dir=Path("/fake/trainer"),
        model_name="base",
        dataset_file="dataset.jsonl",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        gpu_type="a100-large",
        timeout_hours=4.0,
        cloud_image="unsloth/unsloth:latest",
        hf_flavor="a100-large",
        artifact_backend="hf_bucket",
        artifact_identifier="professorsynapse/toolset-training-artifacts",
        artifact_mount_path="/workspace/outputs",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="deadbeefcafebabe",
    )

    backend.prepare_source.return_value = _install_hf_source_transport(
        service=service,
        experiment=experiment,
        acknowledged=False,
        record_prepared=False,
    )
    with patch.object(runner, "_recover_existing_training", return_value=None):
        with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
            with patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry.get", return_value=backend):
                with pytest.raises(CloudProviderError, match="awaits separately approved external provisioning"):
                    runner.run(spec=spec, experiment=experiment)

    config = backend.prepare_source.call_args.args[0]
    backend.execute.assert_not_called()
    assert experiment.source_transport_state == "PREPARED"
    assert config.evolutionary_enabled is False
    assert config.evolutionary_candidates is None
    assert config.evolutionary_eval_batch_size is None
    assert config.evolutionary_validation_config is None
    assert config.evolutionary_strategy is None
    assert config.evolutionary_noise_scale is None
    assert config.evolutionary_max_grad_norm is None
    assert config.evolutionary_scale_factors is None
    assert config.evolutionary_selection_method is None
    assert config.evolutionary_min_improvement is None
    assert config.evolutionary_min_relative_improvement is None
    assert config.evolutionary_noise_floor_epsilon is None
    assert config.evolutionary_eval_frequency is None
    assert config.evolutionary_warmup_steps is None
    assert config.evolutionary_cache_baseline is None
    assert config.evolutionary_log_candidates is None
    assert config.evolutionary_log_selected is None


def test_training_stage_runner_forwards_evolutionary_fields_to_cloud_config(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)

    spec = ExperimentSpec(
        name="evolutionary-smoke",
        provider="hf_jobs",
        method="sft",
        objective="evolutionary_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(
            model_name="Qwen/Qwen3-4B",
            max_steps=20,
        ),
        evaluation=EvaluationStageSpec(enabled=False),
        loss=LossStageSpec(enabled=False),
        features=FeaturesStageSpec(enabled=False),
    )
    spec.training.evolutionary.enabled = True
    spec.training.evolutionary.candidates = 4
    spec.training.evolutionary.eval_batch_size = 2
    spec.training.evolutionary.validation_config = "configs/fitness/tool_calling.yaml"
    spec.training.evolutionary.strategy.type = "antithetic_noise"
    spec.training.evolutionary.strategy.params = {
        "noise_scale": 0.03,
        "max_grad_norm": 1.0,
        "scale_factors": [0.5, 1.0, 1.5],
    }
    spec.training.evolutionary.selection.method = "best"
    spec.training.evolutionary.selection.min_improvement = 0.01
    spec.training.evolutionary.selection.min_relative_improvement = 0.0001
    spec.training.evolutionary.selection.noise_floor_epsilon = 0.000001
    spec.training.evolutionary.eval_frequency = 5
    spec.training.evolutionary.warmup_steps = 200
    spec.training.evolutionary.cache_baseline = True
    spec.training.evolutionary.logging.candidates = False
    spec.training.evolutionary.logging.selected = True

    runner = HFTrainingStageRunner(repo_root=repo_root, tracking_service=service)

    backend = MagicMock()
    backend.validate_environment.return_value = (True, "")
    backend.execute.return_value = 0
    backend.load_config.return_value = CloudTrainingConfig(
        method="sft",
        platform="hf_jobs",
        config_path=Path("/fake/config.yaml"),
        trainer_dir=Path("/fake/trainer"),
        model_name="base",
        dataset_file="dataset.jsonl",
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        provider="hf_jobs",
        gpu_type="a100-large",
        timeout_hours=4.0,
        cloud_image="unsloth/unsloth:latest",
        hf_flavor="a100-large",
        artifact_backend="hf_bucket",
        artifact_identifier="professorsynapse/toolset-training-artifacts",
        artifact_mount_path="/workspace/outputs",
        repo_url="https://github.com/test/repo.git",
        repo_branch="main",
        repo_commit="deadbeefcafebabe",
    )

    backend.prepare_source.return_value = _install_hf_source_transport(
        service=service,
        experiment=experiment,
        acknowledged=False,
        record_prepared=False,
    )
    with patch.object(runner, "_recover_existing_training", return_value=None):
        with patch.object(runner, "_resolve_bucket_id", return_value="professorsynapse/toolset-training-artifacts"):
            with patch("tuner.handlers.stages.hf_training_stage.TrainingBackendRegistry.get", return_value=backend):
                with pytest.raises(CloudProviderError, match="awaits separately approved external provisioning"):
                    runner.run(spec=spec, experiment=experiment)

    config = backend.prepare_source.call_args.args[0]
    backend.execute.assert_not_called()
    assert experiment.source_transport_state == "PREPARED"
    assert config.evolutionary_enabled is True
    assert config.evolutionary_candidates == 4
    assert config.evolutionary_eval_batch_size == 2
    assert config.evolutionary_validation_config == "configs/fitness/tool_calling.yaml"
    assert config.evolutionary_strategy == "antithetic_noise"
    assert config.evolutionary_noise_scale == 0.03
    assert config.evolutionary_max_grad_norm == 1.0
    assert config.evolutionary_scale_factors == [0.5, 1.0, 1.5]
    assert config.evolutionary_selection_method == "best"
    assert config.evolutionary_min_improvement == 0.01
    assert config.evolutionary_min_relative_improvement == 0.0001
    assert config.evolutionary_noise_floor_epsilon == 0.000001
    assert config.evolutionary_eval_frequency == 5
    assert config.evolutionary_warmup_steps == 200
    assert config.evolutionary_cache_baseline is True
    assert config.evolutionary_log_candidates is False
    assert config.evolutionary_log_selected is True


def test_eval_stage_runner_defaults_to_parallel_loss_mode(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    source_preparation = _install_hf_source_transport(
        service=service, experiment=experiment, acknowledged=True
    )
    spec = ExperimentSpec(
        name="parallel-post-training",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="Qwen/Qwen3-4B", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=False),
    )
    runner = HFEvalStageRunner(repo_root=_ENGINE_REPO_ROOT, tracking_service=service)
    training = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="train-run",
            run_type="sft",
            name="train",
            timestamp="2026-03-23T00:00:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/runs/hf_jobs/sft/abc",
            model_name="Qwen/Qwen3-4B",
            dataset_source="repo/dataset/sample.jsonl",
            provider="hf_jobs",
            artifact_backend="hf_bucket",
            artifact_root="hf://buckets/test/runs/hf_jobs/sft/abc",
            source_commit="deadbeef",
            stage="training",
            tags={"bucket_id": "test/toolset-training-artifacts", "artifact_prefix": "runs/hf_jobs/sft/abc"},
        ),
    )

    args = runner.build_evaluation_plan_args(
        spec=spec,
        artifact_prefix=training.run_record.tags["artifact_prefix"],
        bucket_id=training.run_record.tags["bucket_id"],
        source_preparation=source_preparation,
    )
    assert args.with_loss is False
    assert args.loss_dataset_name is None
    assert args.loss_dataset_file is None
    assert args.eval_pip_packages == []
    assert args._source_preparation.source_transport_state == "CONSUMABLE"
    with patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler") as mock_handler_cls:
        with pytest.raises(CloudProviderError, match="separately authorized exact-run approval"):
            runner.run(spec=spec, experiment=experiment, previous=training)
        mock_handler_cls.assert_not_called()
    assert experiment.source_transport_state == "CONSUMABLE"


def test_eval_stage_runner_can_use_same_job_loss_mode(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    source_preparation = _install_hf_source_transport(
        service=service, experiment=experiment, acknowledged=True
    )
    spec = ExperimentSpec(
        name="same-job-post-training",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="Qwen/Qwen3-4B", max_steps=20, max_seq_length=2048),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True, completion_only=False),
        features=FeaturesStageSpec(enabled=False),
    )
    spec.post_training.mode = "same_job"
    runner = HFEvalStageRunner(repo_root=_ENGINE_REPO_ROOT, tracking_service=service)
    training = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="train-run",
            run_type="sft",
            name="train",
            timestamp="2026-03-23T00:00:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/runs/hf_jobs/sft/abc",
            model_name="Qwen/Qwen3-4B",
            dataset_source="repo/dataset/sample.jsonl",
            provider="hf_jobs",
            artifact_backend="hf_bucket",
            artifact_root="hf://buckets/test/runs/hf_jobs/sft/abc",
            source_commit="deadbeef",
            stage="training",
            tags={"bucket_id": "test/toolset-training-artifacts", "artifact_prefix": "runs/hf_jobs/sft/abc"},
        ),
    )

    args = runner.build_evaluation_plan_args(
        spec=spec,
        artifact_prefix=training.run_record.tags["artifact_prefix"],
        bucket_id=training.run_record.tags["bucket_id"],
        source_preparation=source_preparation,
    )
    assert args.with_loss is True
    assert args.loss_dataset_name == "repo/dataset"
    assert args.loss_dataset_file == "sample.jsonl"
    assert args.loss_no_completion_only is True
    with patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler") as mock_handler_cls:
        with pytest.raises(CloudProviderError, match="no approval contract is implemented"):
            runner.run(spec=spec, experiment=experiment, previous=training)
        mock_handler_cls.assert_not_called()


def test_eval_stage_runner_forwards_stage_pip_packages(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    source_preparation = _install_hf_source_transport(
        service=service, experiment=experiment, acknowledged=True
    )
    spec = ExperimentSpec(
        name="eval-pip-packages",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="Qwen/Qwen3-4B", max_steps=20),
        evaluation=EvaluationStageSpec(
            enabled=True,
            preset="quick",
            pip_packages=["vllm==0.12.0", "transformers==5.3.0"],
        ),
        loss=LossStageSpec(enabled=False),
        features=FeaturesStageSpec(enabled=False),
    )
    runner = HFEvalStageRunner(repo_root=_ENGINE_REPO_ROOT, tracking_service=service)
    training = StageResult(
        status="completed",
        run_record=RunRecord(
            run_id="train-run",
            run_type="sft",
            name="train",
            timestamp="2026-03-23T00:00:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/runs/hf_jobs/sft/abc",
            model_name="Qwen/Qwen3-4B",
            dataset_source="repo/dataset/sample.jsonl",
            provider="hf_jobs",
            artifact_backend="hf_bucket",
            artifact_root="hf://buckets/test/runs/hf_jobs/sft/abc",
            source_commit="deadbeef",
            stage="training",
            tags={"bucket_id": "test/toolset-training-artifacts", "artifact_prefix": "runs/hf_jobs/sft/abc"},
        ),
    )

    args = runner.build_evaluation_plan_args(
        spec=spec,
        artifact_prefix=training.run_record.tags["artifact_prefix"],
        bucket_id=training.run_record.tags["bucket_id"],
        source_preparation=source_preparation,
    )
    assert args.eval_pip_packages == ["vllm==0.12.0", "transformers==5.3.0"]
    with patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler") as mock_handler_cls:
        with pytest.raises(CloudProviderError, match="separately authorized exact-run approval"):
            runner.run(spec=spec, experiment=experiment, previous=training)
        mock_handler_cls.assert_not_called()


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
        with patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor.submit") as mock_submit:
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


def test_eval_stage_runner_requests_same_job_loss_when_post_training_mode_is_same_job(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    experiment = _experiment()
    service.save_experiment(experiment)
    source_preparation = _install_hf_source_transport(
        service=service, experiment=experiment, acknowledged=True
    )
    runner = HFEvalStageRunner(repo_root=_ENGINE_REPO_ROOT, tracking_service=service)
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
            "evaluation": type(
                "Evaluation",
                (),
                {
                    "preset": "full",
                    "scenarios": [],
                    "tags": None,
                    "runtime": "vllm",
                    "image_profile": "fast_vllm",
                    "cloud_image": None,
                    "gpu": None,
                    "timeout_hours": None,
                },
            )(),
            "loss": type("Loss", (), {"enabled": True, "max_seq_length": 2048, "completion_only": True})(),
            "post_training": type("PostTraining", (), {"mode": "same_job"})(),
            "name": "resume-smoke",
        },
    )()

    args = runner.build_evaluation_plan_args(
        spec=spec,
        artifact_prefix=previous.run_record.tags["artifact_prefix"],
        bucket_id=previous.run_record.tags["bucket_id"],
        source_preparation=source_preparation,
    )
    assert args.with_loss is True
    assert args.eval_runtime == "vllm"
    assert args._source_preparation.source_transport_state == "CONSUMABLE"
    assert args.eval_image_profile == "fast_vllm"
    assert args.loss_dataset_name == "professorsynapse/claudesidian-synthetic-dataset"
    assert args.loss_dataset_file == "train.jsonl"
    with patch("tuner.handlers.stages.hf_eval_stage.CloudEvalHandler") as mock_handler_cls:
        with pytest.raises(CloudProviderError, match="no approval contract is implemented"):
            runner.run(spec=spec, experiment=experiment, previous=previous)
        mock_handler_cls.assert_not_called()


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
        with patch("tuner.handlers.stages.hf_loss_stage.HFJobExecutor.submit") as mock_submit:
            result = runner.run(spec=None, experiment=experiment, previous=previous)

    assert result.status == "completed"


def test_loss_stage_runner_build_command_uses_python3_and_dataset_file(tmp_path: Path, repo_root):
    service = TrackingService(tmp_path)
    runner = HFLossStageRunner(repo_root=repo_root, tracking_service=service)
    experiment = _experiment()
    service.save_experiment(experiment)
    source_preparation = _install_hf_source_transport(
        service=service, experiment=experiment, acknowledged=True
    )
    training_run = RunRecord(
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
    )
    spec = type(
        "Spec",
        (),
        {
            "dataset": type(
                "Dataset",
                (),
                {
                    "source": "professorsynapse/claudesidian-synthetic-dataset",
                    "file": "train.jsonl",
                },
            )(),
            "loss": type(
                "Loss",
                (),
                {
                    "max_seq_length": 2048,
                    "completion_only": True,
                    "pip_packages": ["transformers==5.3.0"],
                },
            )(),
            "training": type("Training", (), {"max_seq_length": 2048})(),
        },
    )()

    command = runner._build_command(
        spec=spec,
        training_run=training_run,
        results_prefix="runs/hf_jobs/sft/20260321_191536-deadbeef/analysis/loss",
        source_preparation=source_preparation,
    )

    assert "$(command -v python3 || command -v python)" in command
    assert "python3 -m shared.experiment_tracking.cloud_loss_job" in command
    assert "--dataset-name professorsynapse/claudesidian-synthetic-dataset" in command
    assert "--dataset-file train.jsonl" in command
    assert "pip install --upgrade --target /workspace/cache/hf-bucket-sync-site 'huggingface_hub>=1.5.0' hf_transfer hf_xet" in command
    assert command.index("_verify-identities") < command.index("cloud_loss_job")
    assert "pip install --upgrade transformers==5.3.0" in command


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


def test_experiment_handler_loads_host_spec_with_injected_context(tmp_path: Path):
    from argparse import Namespace

    from tuner.handlers.experiment_handler import ExperimentHandler
    from tuner.project import ProjectContext

    host = tmp_path / "host"
    engine = host / "dependencies" / "synaptic-tuner"
    spec_path = host / "experiments" / "smoke.yaml"
    engine.mkdir(parents=True)
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        """experiment:
  name: host-smoke
  provider: hf_jobs
  method: sft
  dataset: {source: org/data, file: train.jsonl}
  training: {model_name: org/model}
""",
        encoding="utf-8",
    )
    context = ProjectContext.host(
        engine_root=engine,
        project_root=host,
        invocation_cwd=host,
    )
    handler = ExperimentHandler(
        Namespace(json=False, experiment_spec="experiments/smoke.yaml"),
        context=context,
    )

    spec, resolved_path = handler._load_spec()

    assert resolved_path == spec_path
    assert spec.resolved_config is not None
    assert spec.resolved_config.sources[0]["uri"] == "project://experiments/smoke.yaml"
    assert not (engine / ".tracking").exists()


def test_cloud_manifest_atomic_failure_preserves_previous_bytes(tmp_path: Path):
    from unittest.mock import patch

    from shared.cloud_artifacts import write_manifest

    path = tmp_path / "manifest.json"
    path.write_bytes(b"previous\n")
    with patch(
        "shared.experiment_tracking.experiment.os.replace",
        side_effect=PermissionError("injected manifest replace failure"),
    ):
        with pytest.raises(PermissionError, match="injected manifest"):
            write_manifest(path, {"status": "new"})

    assert path.read_bytes() == b"previous\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_cloud_stage_event_appends_remain_parseable_under_threads(tmp_path: Path):
    import threading

    from shared.cloud_stage_logging import CloudStageLogger

    logger = CloudStageLogger(tmp_path / "logs", stage="evaluation")
    threads = [
        threading.Thread(target=logger.emit, args=(f"event-{index}",))
        for index in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [
        json.loads(line)
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 12
    assert {row["event"] for row in rows} == {f"event-{index}" for index in range(12)}


def test_cloud_stage_event_appends_are_serialized_across_processes(tmp_path: Path):
    from shared.cloud_stage_logging import STAGE_EVENTS_FILENAME, STAGE_SUMMARY_FILENAME

    log_dir = tmp_path / "logs"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    process_count = 16
    processes = [
        context.Process(
            target=_emit_stage_event_process,
            args=(str(log_dir), index, start, results),
        )
        for index in range(process_count)
    ]
    for process in processes:
        process.start()
    start.set()
    received = []
    try:
        received = [results.get(timeout=40) for _ in range(process_count)]
    finally:
        deadline = time.monotonic() + 40
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert all(code == 0 for code, _index, _error in received), received
    rows = [
        json.loads(line)
        for line in (log_dir / STAGE_EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == process_count
    assert {row["event"] for row in rows} == {
        f"process-{index}" for index in range(process_count)
    }
    summary = json.loads((log_dir / STAGE_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["event_count"] == process_count


def test_cloud_stage_event_retries_short_writes(tmp_path: Path, monkeypatch):
    from shared.cloud_stage_logging import CloudStageLogger

    logger = CloudStageLogger(tmp_path / "logs", stage="evaluation")
    logger.emit("prior")
    real_write = logger._write_event_bytes
    calls = 0

    def short_once(fd: int, data: memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[: max(1, len(data) // 2)])
        return real_write(fd, data)

    monkeypatch.setattr(logger, "_write_event_bytes", short_once)
    logger.emit("short-success")

    rows = [json.loads(line) for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["prior", "short-success"]
    assert calls >= 2


@pytest.mark.parametrize("failure", ["error", "zero", "fsync", "summary"])
def test_cloud_stage_event_failure_rolls_back_exact_complete_bytes(
    tmp_path: Path, monkeypatch, failure: str
):
    from shared.cloud_stage_logging import CloudStageLogger

    logger = CloudStageLogger(tmp_path / "logs", stage="evaluation")
    logger.emit("prior")
    prior_events = logger.events_path.read_bytes()
    prior_summary = logger.summary_path.read_bytes()

    if failure in {"error", "zero"}:
        real_write = logger._write_event_bytes
        calls = 0

        def failed_write(fd: int, data: memoryview) -> int:
            nonlocal calls
            calls += 1
            if calls == 1 and failure == "error":
                real_write(fd, data[: max(1, len(data) // 2)])
                return max(1, len(data) // 2)
            if (calls == 2 and failure == "error") or failure == "zero":
                if failure == "zero":
                    return 0
                raise OSError("injected event write failure")
            return real_write(fd, data)

        monkeypatch.setattr(logger, "_write_event_bytes", failed_write)
    elif failure == "fsync":
        real_fsync = logger._fsync_event_file
        calls = 0

        def failed_fsync(fd: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected event fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(logger, "_fsync_event_file", failed_fsync)
    else:
        monkeypatch.setattr(
            logger,
            "_write_summary",
            lambda _payload: (_ for _ in ()).throw(OSError("injected summary failure")),
        )

    with pytest.raises(OSError):
        logger.emit("must-rollback")

    assert logger.events_path.read_bytes() == prior_events
    assert logger.summary_path.read_bytes() == prior_summary
    assert [
        json.loads(line)["event"]
        for line in logger.events_path.read_text(encoding="utf-8").splitlines()
    ] == ["prior"]


def test_cloud_stage_event_discards_incomplete_legacy_tail(tmp_path: Path):
    from shared.cloud_stage_logging import CloudStageLogger

    logger = CloudStageLogger(tmp_path / "logs", stage="evaluation")
    logger.emit("prior")
    with logger.events_path.open("ab") as handle:
        handle.write(b'{"event":"partial"')
    logger.emit("after-partial")

    rows = [json.loads(line) for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["prior", "after-partial"]
