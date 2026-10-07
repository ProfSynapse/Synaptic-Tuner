from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path

import pytest

from shared.experiment_tracking import ExperimentOrchestrator, ExperimentSpec, StageResult, TrackingService
from shared.experiment_tracking.experiment_spec import DatasetSpec, EvaluationStageSpec, ExecutionStageSpec, FeaturesStageSpec, LossStageSpec, TrainingStageSpec
from shared.experiment_tracking.schema import LossResult, RunRecord
from shared.experiment_tracking.service import ProvenanceIntegrityError
from tests.shared.experiment_tracking.test_tracking_service import (
    _descriptor,
    _source_lock,
    _write_canonical,
)
from tuner.project import (
    ConfigDocument,
    GitSource,
    ProjectContext,
    RepositoryLocation,
    SourceLock,
    resolve_config_layers,
)


class _StaticRunner:
    def __init__(self, result: StageResult):
        self.result = result

    def run(self, spec, experiment, previous=None):
        return self.result


class _FailIfCalledRunner:
    def run(self, spec, experiment, previous=None):
        raise AssertionError("runner should not have been called")


class _BarrierRunner:
    def __init__(self, result: StageResult, barrier: threading.Barrier, events: list[tuple[str, float]], label: str):
        self.result = result
        self.barrier = barrier
        self.events = events
        self.label = label

    def run(self, spec, experiment, previous=None):
        self.events.append((f"{self.label}_start", time.perf_counter()))
        self.barrier.wait(timeout=2.0)
        time.sleep(0.05)
        self.events.append((f"{self.label}_end", time.perf_counter()))
        return self.result


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
    assert experiment.derived_outputs["draft_next_spec_yaml"].endswith("draft_next_spec.yaml")

    summary_path = Path(experiment.derived_outputs["experiment_summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["run_count"] == 3


def test_orchestrator_reuses_one_resolved_config_across_all_stage_records(tmp_path: Path):
    host = tmp_path / "host"
    engine = host / "vendor" / "engine"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=host)
    spec = ExperimentSpec(
        name="portable",
        provider="hf_jobs",
        method="sft",
        dataset=DatasetSpec(source="org/data", file="train.jsonl"),
        training=TrainingStageSpec(model_name="org/model"),
        evaluation=EvaluationStageSpec(enabled=True),
        loss=LossStageSpec(enabled=True),
        execution=ExecutionStageSpec(stages=["training", "evaluation", "loss"]),
    )
    spec.resolved_config = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://experiments/portable.yaml",
                data={"experiment": {"name": "portable"}},
                precedence=0,
            )
        ]
    )
    service = TrackingService(project_context=context)
    experiment = service.create_experiment(
        name=spec.name,
        dataset_path=spec.dataset.identifier,
        dataset_hash=spec.dataset.hash,
        base_model_name=spec.training.model_name,
        provider=spec.provider,
        method=spec.method,
    )
    source_lock = SourceLock(
        run_id=experiment.experiment_id,
        mode="superproject",
        project_source=GitSource(
            location=RepositoryLocation.parse("https://github.com/org/host.git"),
            commit="1" * 40,
            pushed=True,
        ),
        engine_source=GitSource(
            location=RepositoryLocation.parse("https://github.com/org/engine.git"),
            commit="2" * 40,
            pushed=True,
            submodule_path="vendor/engine",
            gitlink_commit="2" * 40,
        ),
        project={"id": "host"},
        configuration={"resolved_uri": "tracking://resolved-config.json"},
    )
    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_StaticRunner(StageResult("completed", _record(run_id="train", run_type="sft", stage="training"))),
        eval_runner=_StaticRunner(StageResult("completed", _record(run_id="eval", run_type="evaluation", stage="evaluation"))),
        loss_runner=_StaticRunner(StageResult("completed", _record(run_id="loss", run_type="loss", stage="loss"))),
        project_context=context,
        source_lock=source_lock,
    )

    experiment = orchestrator.run(
        spec,
        spec_path="project://experiments/portable.yaml",
        experiment=experiment,
    )

    assert experiment.resolved_config_uri == (
        f"tracking://experiments/{experiment.experiment_id}/resolved-config.json"
    )
    assert experiment.spec_path is None
    resolved_path = service.resolve_uri(experiment.resolved_config_uri or "")
    assert experiment.resolved_config_sha256 == hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    assert experiment.source_lock_uri == (
        f"tracking://experiments/{experiment.experiment_id}/source-lock.json"
    )
    assert len(experiment.source_lock_sha256 or "") == 64
    records = service.registry.find_runs()
    assert len(records) == 3
    assert {record.resolved_config_uri for record in records} == {experiment.resolved_config_uri}
    assert {record.resolved_config_sha256 for record in records} == {
        experiment.resolved_config_sha256
    }
    assert {record.source_lock_uri for record in records} == {experiment.source_lock_uri}
    assert {record.source_lock_sha256 for record in records} == {
        experiment.source_lock_sha256
    }
    assert not (engine / ".tracking").exists()

    source_path = service.resolve_uri(experiment.source_lock_uri or "")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_payload["project"]["id"] = "tampered"
    source_path.write_bytes(
        (json.dumps(source_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    with pytest.raises(ProvenanceIntegrityError, match="SHA-256"):
        service.verify_experiment_provenance(experiment)


def test_resume_rejects_tampered_provenance_before_any_stage(tmp_path: Path):
    service = TrackingService(tmp_path)
    spec = ExperimentSpec(
        name="resume-integrity",
        provider="hf_jobs",
        method="sft",
        dataset=DatasetSpec(source="org/data", file="train.jsonl"),
        training=TrainingStageSpec(model_name="org/model"),
        execution=ExecutionStageSpec(stages=["training"]),
    )
    spec.resolved_config = resolve_config_layers(
        [ConfigDocument.from_mapping(uri="project://spec.yaml", data={"value": 1}, precedence=0)]
    )
    experiment = service.create_experiment(
        name=spec.name,
        dataset_path=spec.dataset.identifier,
        dataset_hash="",
        base_model_name=spec.training.model_name,
        provider=spec.provider,
        method=spec.method,
    )
    service.persist_resolved_config(experiment, spec.resolved_config)
    path = service.resolve_uri(experiment.resolved_config_uri or "")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["value"] = 2
    path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_FailIfCalledRunner(),
    )

    with pytest.raises(ProvenanceIntegrityError, match="SHA-256|does not match|canonically"):
        orchestrator.run(spec, experiment=experiment)

    assert experiment.stage_statuses == {}


def test_resume_rejects_tampered_source_transport_before_runner(tmp_path: Path):
    service = TrackingService(tmp_path)
    spec = ExperimentSpec(
        name="resume-transport-integrity",
        provider="hf_jobs",
        method="sft",
        dataset=DatasetSpec(source="org/data", file="train.jsonl"),
        training=TrainingStageSpec(model_name="org/model"),
        execution=ExecutionStageSpec(stages=["training"]),
    )
    experiment = service.create_experiment(
        name=spec.name,
        dataset_path=spec.dataset.identifier,
        dataset_hash="",
        base_model_name=spec.training.model_name,
        provider=spec.provider,
        method=spec.method,
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "descriptor.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    descriptor_path = service.resolve_uri(descriptor_uri)
    descriptor_path.write_bytes(
        descriptor_path.read_bytes().replace(b'"profile":"C"', b'"profile":"B"')
    )
    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_FailIfCalledRunner(),
    )

    with pytest.raises(ProvenanceIntegrityError, match="SHA-256|canonically"):
        orchestrator.run(spec, experiment=experiment)

    assert experiment.stage_statuses == {}


def test_new_host_experiment_persists_portable_spec_uri(tmp_path: Path):
    host = tmp_path / "host"
    engine = host / "vendor" / "engine"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=host)
    service = TrackingService(project_context=context)
    spec = ExperimentSpec(
        name="portable-spec",
        provider="hf_jobs",
        method="sft",
        dataset=DatasetSpec(source="org/data", file="train.jsonl"),
        training=TrainingStageSpec(model_name="org/model"),
        execution=ExecutionStageSpec(stages=["training"]),
    )
    spec.resolved_config = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://experiments/portable.yaml",
                data={"experiment": {"name": "portable-spec"}},
                precedence=0,
            )
        ]
    )
    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_StaticRunner(
            StageResult("completed", _record(run_id="train-portable", run_type="sft", stage="training"))
        ),
        project_context=context,
    )

    experiment = orchestrator.run(spec, spec_path=str(host / "experiments" / "portable.yaml"))

    assert experiment.spec_path == "project://experiments/portable.yaml"


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


def test_experiment_orchestrator_reruns_failed_loss_stage_on_resume(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="resume-loss",
        dataset_path="repo/dataset/sample.jsonl",
        dataset_hash="abc123",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
    )
    service.update_stage_details(
        experiment,
        "training",
        status="completed",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/20260321_221651-deadbeef",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_221651-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )
    service.update_stage_details(
        experiment,
        "evaluation",
        status="completed",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/20260321_221651-deadbeef/evaluations/vllm/20260321_230000",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_221651-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )
    service.update_stage_details(
        experiment,
        "loss",
        status="failed",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/20260321_221651-deadbeef/analysis/loss",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_221651-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    spec = ExperimentSpec(
        name="resume-loss",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="Qwen/Qwen3-4B", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )

    loss_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-loss-rerun", run_type="loss", stage="loss"),
            loss_results=[LossResult(index=0, loss=0.2, num_completion_tokens=10, num_total_tokens=20, jsonl_hash="bbbb2222")],
            artifact_root="/tmp/loss-rerun-artifacts",
        )
    )

    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_FailIfCalledRunner(),
        eval_runner=_FailIfCalledRunner(),
        loss_runner=loss_runner,
        base_dir=tmp_path,
    )

    resumed = orchestrator.run(spec, spec_path="/tmp/spec.yaml", experiment=experiment)

    assert resumed.status == "completed"
    assert resumed.training_run_id == f"{experiment.experiment_id}-training"
    assert resumed.evaluation_run_id == f"{experiment.experiment_id}-evaluation"
    assert resumed.loss_run_id == "exp-loss-rerun"
    assert resumed.stage_statuses["loss"] == "completed"


def test_experiment_orchestrator_only_stage_evaluation_reuses_training_and_skips_loss(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="eval-only",
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
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/20260321_221651-deadbeef",
        source_commit="deadbeefcafebabe",
        tags={
            "provider": "hf_jobs",
            "artifact_prefix": "runs/hf_jobs/sft/20260321_221651-deadbeef",
            "bucket_id": "test/toolset-training-artifacts",
        },
    )

    spec = ExperimentSpec(
        name="eval-only",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )
    spec.execution.only_stage = "evaluation"

    eval_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-eval-only", run_type="evaluation", stage="evaluation"),
            eval_payload={
                "summary": {"passed": 1, "failed": 0, "warned": 0, "total": 1},
                "records": [{"case_id": "ok", "passed": True}],
            },
            artifact_root="/tmp/eval-only-artifacts",
        )
    )

    orchestrator = ExperimentOrchestrator(
        tracking_service=service,
        training_runner=_FailIfCalledRunner(),
        eval_runner=eval_runner,
        loss_runner=_FailIfCalledRunner(),
        base_dir=tmp_path,
    )

    resumed = orchestrator.run(spec, spec_path="/tmp/spec.yaml", experiment=experiment)

    assert resumed.status == "completed"
    assert resumed.training_run_id == f"{experiment.experiment_id}-training"
    assert resumed.evaluation_run_id == "exp-eval-only"
    assert resumed.loss_run_id is None
    assert resumed.derived_outputs == {}


def test_experiment_orchestrator_runs_eval_and_loss_in_parallel_mode(tmp_path: Path):
    spec = ExperimentSpec(
        name="parallel-post-training",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        dataset=DatasetSpec(source="repo/dataset", file="sample.jsonl", hash="abc123"),
        training=TrainingStageSpec(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_steps=20),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick"),
        loss=LossStageSpec(enabled=True),
        features=FeaturesStageSpec(enabled=True),
    )
    spec.post_training.mode = "parallel"

    training_runner = _StaticRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-training", run_type="sft", stage="training"),
            artifact_root="/tmp/train-artifacts",
        )
    )
    barrier = threading.Barrier(2)
    events: list[tuple[str, float]] = []
    eval_runner = _BarrierRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-eval", run_type="evaluation", stage="evaluation"),
            eval_payload={"summary": {"passed": 1, "failed": 0, "warned": 0, "total": 1}},
            artifact_root="/tmp/eval-artifacts",
        ),
        barrier=barrier,
        events=events,
        label="eval",
    )
    loss_runner = _BarrierRunner(
        StageResult(
            status="completed",
            run_record=_record(run_id="exp-loss", run_type="loss", stage="loss"),
            loss_results=[LossResult(index=0, loss=0.4, num_completion_tokens=10, num_total_tokens=20, jsonl_hash="aaaa1111")],
            artifact_root="/tmp/loss-artifacts",
        ),
        barrier=barrier,
        events=events,
        label="loss",
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
    event_times = dict(events)
    assert "eval_start" in event_times
    assert "loss_start" in event_times
    assert abs(event_times["eval_start"] - event_times["loss_start"]) < 0.1
