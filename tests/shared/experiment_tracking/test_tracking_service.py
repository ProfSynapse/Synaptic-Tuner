from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.experiment_tracking import TrackingService
from shared.experiment_tracking.experiment import (
    Experiment,
    _atomic_write_text,
    save_experiment,
)
from shared.experiment_tracking.service import ProvenanceIntegrityError
from shared.experiment_tracking.schema import RunRecord
from tuner.project import ConfigDocument, ProjectContext, resolve_config_layers


def test_tracking_service_creates_and_updates_experiment(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="smoke",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
    )

    record = RunRecord(
        run_id="exp-training",
        run_type="sft",
        name="training",
        timestamp="2026-03-21T18:00:00+00:00",
        status="completed",
        output_dir="hf://buckets/test/runs/hf_jobs/sft/123",
        provider="hf_jobs",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/123",
        stage="training",
    )

    run_id = service.attach_run(experiment, record, role="training")
    service.mark_stage(experiment, "training", "completed")
    service.set_artifact_root(experiment, "training", record.artifact_root or "")
    service.set_derived_output(experiment, "feature_dataset_csv", "/tmp/features.csv")
    service.set_derived_output(experiment, "hypothesis_context_json", "/tmp/hypothesis.json")

    reloaded = service.load_experiment(experiment.experiment_id)
    assert run_id == "exp-training"
    assert reloaded.training_run_id == "exp-training"
    assert reloaded.stage_statuses["training"] == "completed"
    assert reloaded.artifact_roots["training"] == "hf://buckets/test/runs/hf_jobs/sft/123"
    assert reloaded.features_csv_path == "/tmp/features.csv"
    assert reloaded.hypothesis_context_path == "/tmp/hypothesis.json"

    registry_lines = (tmp_path / "registry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(registry_lines) == 1
    assert json.loads(registry_lines[0])["experiment_id"] == experiment.experiment_id
    assert experiment.experiment_id.startswith("exp_")
    prefix, nonce = experiment.experiment_id.rsplit("_", 1)
    assert prefix.count("_") == 2
    assert len(nonce) == 4
    int(nonce, 16)


def test_tracking_service_finds_latest_recoverable_experiment(tmp_path: Path):
    service = TrackingService(tmp_path)

    older = Experiment(
        experiment_id="exp_older",
        name="smoke",
        created_at="2026-03-21T18:00:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
        status="partial",
    )
    save_experiment(older, tmp_path)

    newer = Experiment(
        experiment_id="exp_newer",
        name="smoke",
        created_at="2026-03-21T18:05:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
        status="partial",
    )
    save_experiment(newer, tmp_path)

    completed = Experiment(
        experiment_id="exp_done",
        name="done",
        created_at="2026-03-21T18:10:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/other.yaml",
        status="completed",
    )
    save_experiment(completed, tmp_path)

    recovered = service.find_recoverable_experiment(
        spec_path="/tmp/spec.yaml",
        provider="hf_jobs",
        method="sft",
    )

    assert recovered is not None
    assert recovered.experiment_id == newer.experiment_id


def test_host_tracking_uses_context_root_and_stamps_portable_provenance(tmp_path: Path):
    host = tmp_path / "host"
    engine = host / "vendor" / "engine"
    engine.mkdir(parents=True)
    context = ProjectContext.host(engine_root=engine, project_root=host)
    service = TrackingService(project_context=context)
    experiment = service.create_experiment(
        name="host-smoke",
        dataset_path="project://data/train.jsonl",
        dataset_hash="abc123",
        base_model_name="model",
        source_lock_uri="tracking://experiments/host/source-lock.json",
        source_lock_sha256="a" * 64,
    )
    resolved = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://experiments/smoke.yaml",
                data={"experiment": {"name": "host-smoke"}},
                precedence=0,
            )
        ]
    )
    service.persist_resolved_config(experiment, resolved)
    service.attach_run(
        experiment,
        RunRecord(
            run_id="run-1",
            run_type="sft",
            name="run",
            timestamp="2026-08-16T00:00:00+00:00",
            status="completed",
            output_dir="artifact://runs/one",
        ),
    )

    assert service.base_dir == context.tracking_root
    assert experiment.resolved_config_uri.startswith("tracking://experiments/")
    record = service.registry.get_run("run-1")
    assert record is not None
    assert record.source_lock_sha256 == "a" * 64
    resolved_path = service.resolve_uri(experiment.resolved_config_uri or "")
    assert record.resolved_config_sha256 == hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    assert not (engine / ".tracking").exists()


def test_loading_historical_experiment_does_not_rewrite_bytes(tmp_path: Path):
    experiment_dir = tmp_path / "experiments" / "legacy"
    experiment_dir.mkdir(parents=True)
    path = experiment_dir / "experiment.json"
    original = b'{"experiment_id":"legacy","name":"old","created_at":"2024-01-01T00:00:00Z","dataset_path":"data.jsonl","dataset_hash":"x","base_model_name":"model"}\n'
    path.write_bytes(original)

    loaded = TrackingService(tmp_path).load_experiment("legacy")

    assert loaded.source_lock_uri is None
    assert loaded.resolved_config_uri is None
    assert path.read_bytes() == original


def test_provenance_verification_rejects_missing_and_tampered_config(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="integrity",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    resolved = resolve_config_layers(
        [ConfigDocument.from_mapping(uri="project://spec.yaml", data={"value": 1}, precedence=0)]
    )
    service.persist_resolved_config(experiment, resolved)
    path = service.resolve_uri(experiment.resolved_config_uri or "")
    path.write_bytes(b'{"schema_version":"synaptic-resolved-config/v1","value":2}\n')

    with pytest.raises(ProvenanceIntegrityError, match="SHA-256|canonically"):
        service.verify_experiment_provenance(experiment)

    path.unlink()
    with pytest.raises(ProvenanceIntegrityError, match="missing"):
        service.verify_experiment_provenance(experiment)


def test_provenance_verification_rejects_symlink_and_incomplete_reference(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="integrity",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        resolved_config_uri="tracking://experiments/integrity/resolved-config.json",
    )
    with pytest.raises(ProvenanceIntegrityError, match="both URI and SHA-256"):
        service.verify_experiment_provenance(experiment)

    experiment.resolved_config_uri = "tracking://../outside.json"
    experiment.resolved_config_sha256 = "a" * 64
    with pytest.raises(ProvenanceIntegrityError, match="escapes"):
        service.verify_experiment_provenance(experiment)

    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "experiments" / "integrity" / "resolved-config.json"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are unavailable in this environment")
    experiment.resolved_config_sha256 = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ProvenanceIntegrityError, match="symlinks"):
        service.verify_experiment_provenance(experiment)


def test_historical_absolute_spec_matches_portable_uri_without_rewrite(tmp_path: Path):
    host = tmp_path / "host"
    engine = host / "vendor" / "engine"
    spec_path = host / "experiments" / "smoke.yaml"
    engine.mkdir(parents=True)
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("experiment: {}\n", encoding="utf-8")
    context = ProjectContext.host(engine_root=engine, project_root=host)
    service = TrackingService(project_context=context)
    experiment = service.create_experiment(
        name="legacy",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
        method="sft",
        spec_path=str(spec_path),
    )
    record_path = service.base_dir / "experiments" / experiment.experiment_id / "experiment.json"
    original = record_path.read_bytes()

    recovered = service.find_recoverable_experiment(
        spec_path="project://experiments/smoke.yaml",
        provider="hf_jobs",
        method="sft",
    )

    assert recovered is not None
    assert recovered.experiment_id == experiment.experiment_id
    assert recovered.spec_path == str(spec_path)
    assert record_path.read_bytes() == original

    other = tmp_path / "other" / "experiments" / "smoke.yaml"
    other.parent.mkdir(parents=True)
    other.write_text("experiment: {}\n", encoding="utf-8")
    assert service.find_recoverable_experiment(
        spec_path=str(other),
        provider="hf_jobs",
        method="sft",
    ) is None


def test_atomic_writer_preserves_previous_file_and_cleans_own_temp(tmp_path: Path):
    path = tmp_path / "canonical.json"
    path.write_text("old", encoding="utf-8")
    with patch(
        "shared.experiment_tracking.experiment.os.replace",
        side_effect=PermissionError("injected replace failure"),
    ):
        with pytest.raises(PermissionError, match="injected"):
            _atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob("*.tmp")) == []
