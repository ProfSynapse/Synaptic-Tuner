from __future__ import annotations

import json
import hashlib
import multiprocessing
import queue
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

import shared.experiment_tracking as tracking_package
import shared.experiment_tracking.experiment as experiment_module
from shared.experiment_tracking import TrackingService
from shared.experiment_tracking.experiment import (
    Experiment,
    _atomic_write_text,
)
from shared.experiment_tracking.service import ProvenanceIntegrityError
from shared.experiment_tracking.root_identity import ensure_tracking_root_identity
from shared.experiment_tracking.schema import RunRecord
from tuner.project import (
    ConfigDocument,
    GitSource,
    ProjectContext,
    RepositoryLocation,
    SourceLock,
    resolve_config_layers,
)
from tuner.cloud.hf_run_approval import (
    HFRunApproval,
    build_hf_ambiguous_event,
    build_hf_run_approval,
    build_hf_submitted_event,
    build_hf_submitting_event,
    canonical_json_bytes as canonical_approval_bytes,
)
from tuner.cloud.hf_provisioning_claim import (
    build_hf_provisioning_ambiguous_event,
    build_hf_provisioning_claim,
    build_hf_provisioning_succeeded_event,
    canonical_json_bytes as canonical_provisioning_bytes,
)
from tuner.cloud.hf_training_smoke_contract import (
    ARTIFACT_SLOT_INPUT_SCHEMA,
    canonical_json_bytes as canonical_training_bytes,
    derive_hf_training_artifact_prefix,
    derive_hf_training_artifact_slot,
    document_sha256,
    seal_training_document,
)


def _source_lock(run_id: str) -> SourceLock:
    return SourceLock(
        run_id=run_id,
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


def _write_canonical(service: TrackingService, relative: str, payload: dict) -> tuple[str, str]:
    path = service.base_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(data)
    return service.tracking_uri(path), hashlib.sha256(data).hexdigest()


def _descriptor(experiment, *, run_id: str | None = None) -> dict:
    return {
        "schema_version": "synaptic-hf-source-transport/v1",
        "run_id": run_id or experiment.experiment_id,
        "profile": "C",
        "provider": "hf_jobs",
        "source_lock": {
            "uri": experiment.source_lock_uri,
            "sha256": experiment.source_lock_sha256,
            "path": "source-lock.json",
        },
        "capsule": {
            "engine_commit": "2" * 40,
            "uri": "tracking://bundle/capsule",
            "root": "capsule",
            "manifest": {"path": "capsule/synaptic-bootstrap-capsule.json", "sha256": "3" * 64},
        },
        "checkout_policy": {
            "uri": "tracking://bundle/checkout-policy.json",
            "path": "checkout-policy.json",
            "sha256": "4" * 64,
        },
        "bundle": {"uri": "tracking://bundle", "content_sha256": "5" * 64},
        "volume": {
            "type": "bucket",
            "source": "org/bucket",
            "path": f"bootstrap/{experiment.experiment_id}/{'5' * 64}",
            "mount_path": "/workspace/synaptic-bootstrap-input",
            "read_only": True,
        },
    }


def _evidence(experiment, descriptor_uri: str, descriptor_sha256: str) -> dict:
    return {
        "schema_version": "synaptic-hf-provisioning-evidence/v1",
        "descriptor": {"uri": descriptor_uri, "sha256": descriptor_sha256},
        "run_id": experiment.experiment_id,
        "provider": "hf_jobs",
        "profile": "C",
        "volume": {
            "source": "org/bucket",
            "path": f"bootstrap/{experiment.experiment_id}/{'5' * 64}",
            "type": "bucket",
            "read_only": True,
        },
        "bundle_sha256": "5" * 64,
        "capsule_manifest_sha256": "3" * 64,
        "source_lock_sha256": experiment.source_lock_sha256,
        "checkout_policy_sha256": "4" * 64,
        "status": "provisioned",
        "authority": "operator",
        "actor": "test-operator",
        "asserted_at": "2026-08-19T12:00:00Z",
        "provider_receipt_id": "receipt-1",
    }


def _acknowledge_transport_process(
    base_dir: str,
    experiment_id: str,
    evidence_uri: str,
    evidence_sha256: str,
    start,
    results,
) -> None:
    service = TrackingService(Path(base_dir))
    experiment = service.load_experiment(experiment_id)
    if not start.wait(timeout=10):
        results.put(("timeout", "start"))
        return
    try:
        service.record_provisioning_acknowledged(
            experiment, uri=evidence_uri, sha256=evidence_sha256
        )
        results.put(("ok", experiment.provisioning_evidence_sha256))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def _set_derived_output_process(
    base_dir: str,
    experiment_id: str,
    key: str,
    value: str,
    start,
    results,
) -> None:
    service = TrackingService(Path(base_dir))
    experiment = service.load_experiment(experiment_id)
    if not start.wait(timeout=10):
        results.put(("timeout", key))
        return
    try:
        service.set_derived_output(experiment, key, value)
        results.put(("ok", key, dict(experiment.derived_outputs)))
    except Exception as exc:
        results.put(("error", key, type(exc).__name__))


def _claim_submission_process(
    base_dir: str,
    experiment_id: str,
    event: dict,
    start,
    results,
) -> None:
    service = TrackingService(Path(base_dir))
    experiment = service.load_experiment(experiment_id)
    if not start.wait(timeout=10):
        results.put(("timeout", "start"))
        return
    try:
        service.claim_hf_submission(experiment, event)
        results.put(("ok", experiment.hf_submission_event_sha256))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def _claim_cancellation_process(
    base_dir: str,
    experiment_id: str,
    event: dict,
    start,
    results,
) -> None:
    service = TrackingService(Path(base_dir))
    experiment = service.load_experiment(experiment_id)
    if not start.wait(timeout=10):
        results.put(("timeout", "start"))
        return
    try:
        outcome = service.claim_hf_cancellation(experiment, event)
        results.put(("ok", outcome.provider_attempt_authorized, outcome.event_sha256))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def _claim_provisioning_process(
    base_dir: str,
    experiment_id: str,
    claim: dict,
    start,
    results,
) -> None:
    service = TrackingService(Path(base_dir))
    experiment = service.load_experiment(experiment_id)
    if not start.wait(timeout=10):
        results.put(("timeout", "start"))
        return
    try:
        with service.hf_provisioning_execution_lock(experiment_id):
            outcome = service.claim_hf_provisioning(experiment, claim)
        results.put(
            (
                "ok",
                outcome.provider_attempt_authorized,
                outcome.event_sha256,
                outcome.state,
            )
        )
    except Exception as exc:
        results.put(("error", type(exc).__name__))


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
    service.save_experiment(older)

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
    service.save_experiment(newer)

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
    service.save_experiment(completed)

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
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
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
    assert record.source_lock_sha256 == experiment.source_lock_sha256
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
    assert loaded.hf_run_approval_uri is None
    assert loaded.hf_authorization_id is None
    assert loaded.hf_submission_state is None
    assert loaded.hf_cancellation_event_uri is None
    assert loaded.hf_cancellation_state is None
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
    )
    experiment.resolved_config_uri = (
        "tracking://experiments/integrity/resolved-config.json"
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


def test_unsafe_raw_experiment_writer_is_intentionally_not_public():
    assert "save_experiment" not in tracking_package.__all__
    assert not hasattr(tracking_package, "save_experiment")
    assert not hasattr(experiment_module, "save_experiment")
    with pytest.raises(ImportError):
        exec(
            "from shared.experiment_tracking.experiment import save_experiment",
            {},
        )
    private_writer = getattr(
        experiment_module, "_save_experiment_unlocked_after_validation"
    )
    assert "caller must already hold the path lock" in (private_writer.__doc__ or "")


def test_hf_transport_lifecycle_is_monotonic_verified_and_propagated(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="transport",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha256 = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/descriptor.json",
        _descriptor(experiment),
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha256
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha256
    )
    assert experiment.source_transport_state == "PREPARED"

    evidence_uri, evidence_sha256 = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/evidence.json",
        _evidence(experiment, descriptor_uri, descriptor_sha256),
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha256
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha256
    )
    service.mark_source_transport_consumable(experiment)
    service.mark_source_transport_consumable(experiment)
    with pytest.raises(ProvenanceIntegrityError, match="SUCCEEDED provisioning"):
        service.require_consumable_hf_transport(experiment)

    with pytest.raises(ValueError, match="source_transport_uri"):
        service.attach_run(
            experiment,
            RunRecord(
                run_id="run-mismatch",
                run_type="sft",
                name="mismatch",
                timestamp="2026-08-19T12:00:00Z",
                status="completed",
                output_dir="artifact://runs/mismatch",
                source_transport_uri="tracking://other-descriptor.json",
                source_transport_sha256=descriptor_sha256,
            ),
        )

    service.attach_run(
        experiment,
        RunRecord(
            run_id="run-transport",
            run_type="sft",
            name="transport run",
            timestamp="2026-08-19T12:00:00Z",
            status="completed",
            output_dir="artifact://runs/transport",
        ),
        role="training",
    )
    stored = service.registry.get_run("run-transport")
    assert stored is not None
    assert stored.source_transport_uri == descriptor_uri
    assert stored.source_transport_sha256 == descriptor_sha256
    assert stored.provisioning_evidence_uri == evidence_uri
    assert stored.provisioning_evidence_sha256 == evidence_sha256
    assert stored.source_transport_state == "CONSUMABLE"

    record_path = (
        service.base_dir / "experiments" / experiment.experiment_id / "experiment.json"
    )
    before = record_path.read_bytes()
    assert not hasattr(service, "mark_source_transport_submitted")
    with pytest.raises(ProvenanceIntegrityError, match="separately approved"):
        service._transition_source_transport(experiment, state="SUBMITTED")
    assert experiment.source_transport_state == "CONSUMABLE"
    assert service.load_experiment(experiment.experiment_id).source_transport_state == "CONSUMABLE"
    assert record_path.read_bytes() == before

    future_payload = json.loads(before)
    future_payload["source_transport_state"] = "SUBMITTED"
    record_path.write_text(json.dumps(future_payload, indent=2), encoding="utf-8")
    future_record = service.load_experiment(experiment.experiment_id)
    assert future_record.source_transport_state == "SUBMITTED"
    service.verify_source_transport_provenance(future_record)
    with pytest.raises(ProvenanceIntegrityError, match="CONSUMABLE"):
        service.require_consumable_hf_transport(future_record)


def test_hf_transport_rejects_partial_wrong_order_replay_tamper_and_extensions(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="transport",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    with pytest.raises(ValueError, match="both URI and SHA-256"):
        Experiment(
            experiment_id="partial",
            name="partial",
            created_at="2026-08-19T12:00:00Z",
            dataset_path="data.jsonl",
            dataset_hash="abc",
            base_model_name="model",
            source_transport_uri="tracking://descriptor.json",
        )
    with pytest.raises(ProvenanceIntegrityError, match="cannot transition"):
        service.mark_source_transport_consumable(experiment)

    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    replay_uri, replay_sha = _write_canonical(
        service, "replayed.json", _descriptor(experiment, run_id="another-run")
    )
    with pytest.raises(ProvenanceIntegrityError, match="another run"):
        service.record_source_transport_prepared(experiment, uri=replay_uri, sha256=replay_sha)

    extended = _descriptor(experiment)
    extended["extension"] = "forbidden"
    extension_uri, extension_sha = _write_canonical(service, "extended.json", extended)
    with pytest.raises(ProvenanceIntegrityError, match="exact schema"):
        service.record_source_transport_prepared(
            experiment, uri=extension_uri, sha256=extension_sha
        )

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
    with pytest.raises(ProvenanceIntegrityError, match="SHA-256|canonically"):
        service.verify_source_transport_provenance(experiment)


def test_historical_hf_record_remains_readable_unchanged_but_not_consumable(tmp_path: Path):
    experiment_dir = tmp_path / "experiments" / "legacy-hf"
    experiment_dir.mkdir(parents=True)
    path = experiment_dir / "experiment.json"
    original = b'{"experiment_id":"legacy-hf","name":"old","created_at":"2024-01-01T00:00:00Z","dataset_path":"data.jsonl","dataset_hash":"x","base_model_name":"model","provider":"hf_jobs"}\n'
    path.write_bytes(original)
    service = TrackingService(tmp_path)

    experiment = service.load_experiment("legacy-hf")

    assert path.read_bytes() == original
    assert experiment.source_transport_uri is None
    assert experiment.hf_provisioning_event_uri is None
    assert experiment.hf_provisioning_event_sha256 is None
    assert experiment.hf_provisioning_state is None
    assert experiment.hf_run_approval_uri is None
    assert experiment.hf_submission_event_uri is None
    assert experiment.hf_submission_state is None
    assert experiment.hf_cancellation_event_uri is None
    assert experiment.hf_cancellation_state is None
    with pytest.raises(ProvenanceIntegrityError, match="not verified as CONSUMABLE"):
        service.require_consumable_hf_transport(experiment)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "injected",
    [
        {"source_transport_state": "ACKNOWLEDGED"},
        {"source_transport_state": "CONSUMABLE"},
        {"source_transport_state": "SUBMITTED"},
        {
            "source_transport_uri": "tracking://descriptor.json",
            "source_transport_sha256": "1" * 64,
        },
        {
            "provisioning_evidence_uri": "tracking://evidence.json",
            "provisioning_evidence_sha256": "2" * 64,
        },
    ],
)
def test_create_experiment_rejects_direct_transport_projection_without_writes(
    tmp_path: Path, injected: dict[str, str]
):
    service = TrackingService(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        service.create_experiment(
            name="bypass",
            dataset_path="data.jsonl",
            dataset_hash="abc",
            base_model_name="model",
            provider="hf_jobs",
            **injected,
        )

    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not (tmp_path / "experiments").exists()


def test_hf_acknowledgement_requires_exact_reference_and_consumption_requires_full_binding(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="binding",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "descriptor-binding.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )

    wrong_reference = _evidence(experiment, descriptor_uri, "9" * 64)
    wrong_uri, wrong_sha = _write_canonical(service, "wrong-reference.json", wrong_reference)
    with pytest.raises(ProvenanceIntegrityError, match="does not reference"):
        service.record_provisioning_acknowledged(
            experiment, uri=wrong_uri, sha256=wrong_sha
        )

    mismatched_binding = _evidence(experiment, descriptor_uri, descriptor_sha)
    mismatched_binding["bundle_sha256"] = "8" * 64
    evidence_uri, evidence_sha = _write_canonical(
        service, "mismatched-binding.json", mismatched_binding
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha
    )
    assert experiment.source_transport_state == "ACKNOWLEDGED"
    with pytest.raises(ProvenanceIntegrityError, match="does not bind"):
        service.mark_source_transport_consumable(experiment)
    assert experiment.source_transport_state == "ACKNOWLEDGED"


def test_stale_prepared_object_cannot_roll_back_durable_consumable_bytes(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="stale",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "stale-descriptor.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    stale = TrackingService(tmp_path).load_experiment(experiment.experiment_id)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        "stale-evidence.json",
        _evidence(experiment, descriptor_uri, descriptor_sha),
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha
    )
    service.mark_source_transport_consumable(experiment)
    record_path = tmp_path / "experiments" / experiment.experiment_id / "experiment.json"
    durable_bytes = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="cannot transition"):
        TrackingService(tmp_path).record_provisioning_acknowledged(
            stale, uri=evidence_uri, sha256=evidence_sha
        )

    assert stale.source_transport_state == "PREPARED"
    assert record_path.read_bytes() == durable_bytes
    assert service.load_experiment(experiment.experiment_id).source_transport_state == "CONSUMABLE"


def test_competing_thread_acknowledgements_use_durable_compare_and_swap(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="threads",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "thread-descriptor.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    evidence_a = _evidence(experiment, descriptor_uri, descriptor_sha)
    evidence_b = dict(evidence_a)
    evidence_b["provider_receipt_id"] = "receipt-2"
    uri_a, sha_a = _write_canonical(service, "thread-evidence-a.json", evidence_a)
    uri_b, sha_b = _write_canonical(service, "thread-evidence-b.json", evidence_b)
    callers = [
        TrackingService(tmp_path).load_experiment(experiment.experiment_id),
        TrackingService(tmp_path).load_experiment(experiment.experiment_id),
    ]
    barrier = threading.Barrier(3)
    results: list[str] = []

    def acknowledge(caller: Experiment, uri: str, digest: str) -> None:
        barrier.wait(timeout=5)
        try:
            TrackingService(tmp_path).record_provisioning_acknowledged(
                caller, uri=uri, sha256=digest
            )
            results.append("ok")
        except ProvenanceIntegrityError:
            results.append("error")

    threads = [
        threading.Thread(target=acknowledge, args=(callers[0], uri_a, sha_a)),
        threading.Thread(target=acknowledge, args=(callers[1], uri_b, sha_b)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(results) == ["error", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.source_transport_state == "ACKNOWLEDGED"
    assert durable.provisioning_evidence_sha256 in {sha_a, sha_b}


def test_spawned_competing_acknowledgements_use_interprocess_cas(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="processes",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "process-descriptor.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    evidence_a = _evidence(experiment, descriptor_uri, descriptor_sha)
    evidence_b = dict(evidence_a)
    evidence_b["provider_receipt_id"] = "receipt-process-2"
    uri_a, sha_a = _write_canonical(service, "process-evidence-a.json", evidence_a)
    uri_b, sha_b = _write_canonical(service, "process-evidence-b.json", evidence_b)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_acknowledge_transport_process,
            args=(
                str(tmp_path),
                experiment.experiment_id,
                uri,
                digest,
                start,
                results,
            ),
        )
        for uri, digest in ((uri_a, sha_a), (uri_b, sha_b))
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=15) for _ in processes]
    except queue.Empty:
        pytest.fail("spawned acknowledgement process did not report within timeout")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcome[0] for outcome in outcomes) == ["error", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.source_transport_state == "ACKNOWLEDGED"
    assert durable.provisioning_evidence_sha256 in {sha_a, sha_b}


def _make_consumable_transport(
    service: TrackingService,
) -> tuple[Experiment, Experiment]:
    experiment = service.create_experiment(
        name="guarded",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor_uri, descriptor_sha = _write_canonical(
        service, "guarded-descriptor.json", _descriptor(experiment)
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    stale = TrackingService(service.base_dir).load_experiment(experiment.experiment_id)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        "guarded-evidence.json",
        _evidence(experiment, descriptor_uri, descriptor_sha),
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha
    )
    service.mark_source_transport_consumable(experiment)
    return experiment, stale


def _make_prepared_provisioning_claim(
    service: TrackingService,
) -> tuple[Experiment, dict[str, object]]:
    experiment = service.create_experiment(
        name="provisioning-claim",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        provider="hf_jobs",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    descriptor = _descriptor(experiment)
    descriptor_uri, descriptor_sha = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/descriptor.json",
        descriptor,
    )
    service.record_source_transport_prepared(
        experiment, uri=descriptor_uri, sha256=descriptor_sha
    )
    claim = build_hf_provisioning_claim(
        experiment_id=experiment.experiment_id,
        descriptor_uri=descriptor_uri,
        descriptor_sha256=descriptor_sha,
        descriptor=descriptor,
        actor="test-operator",
        authority="operator",
        occurred_at="2026-08-20T12:00:00Z",
    )
    return experiment, claim


def test_hf_provisioning_claim_requires_execution_lock_and_authorizes_only_first(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    with pytest.raises(ProvenanceIntegrityError, match="execution lock"):
        service.claim_hf_provisioning(experiment, claim)

    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        first = service.claim_hf_provisioning(experiment, claim)
    assert first.provider_attempt_authorized is True
    assert first.state == "CLAIMED"
    assert experiment.hf_provisioning_state == "CLAIMED"

    resumed = TrackingService(tmp_path).load_experiment(experiment.experiment_id)
    resume_service = TrackingService(tmp_path)
    with resume_service.hf_provisioning_execution_lock(experiment.experiment_id):
        second = resume_service.claim_hf_provisioning(resumed, claim)
    assert second.provider_attempt_authorized is False
    assert second.event_sha256 == first.event_sha256

    replacement = dict(claim)
    replacement["actor"] = "another-operator"
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        with pytest.raises(ProvenanceIntegrityError):
            service.claim_hf_provisioning(experiment, replacement)


def test_hf_provisioning_success_atomically_binds_evidence_and_acknowledges(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/evidence.json",
        _evidence(experiment, experiment.source_transport_uri or "", experiment.source_transport_sha256 or ""),
    )
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        terminal = build_hf_provisioning_succeeded_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
            occurred_at="2026-08-20T12:01:00Z",
        )
        service.record_hf_provisioning_succeeded(
            experiment,
            terminal,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
        )
    assert experiment.hf_provisioning_state == "SUCCEEDED"
    assert experiment.source_transport_state == "ACKNOWLEDGED"
    assert experiment.provisioning_evidence_sha256 == evidence_sha
    service.verify_experiment_provenance(experiment)

    recovery = TrackingService(tmp_path)
    stale = Experiment.from_dict({
        **experiment.to_dict(),
        "hf_provisioning_event_uri": None,
        "hf_provisioning_event_sha256": None,
        "hf_provisioning_state": None,
        "provisioning_evidence_uri": None,
        "provisioning_evidence_sha256": None,
        "source_transport_state": "PREPARED",
    })
    with recovery.hf_provisioning_execution_lock(experiment.experiment_id):
        recovered = recovery.claim_hf_provisioning(stale, claim)
    assert recovered.provider_attempt_authorized is False
    assert recovered.state == "SUCCEEDED"
    assert stale.provisioning_evidence_sha256 == evidence_sha


def test_threaded_provisioning_claim_authorizes_exactly_one_provider_attempt(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    barrier = threading.Barrier(3)
    outcomes: list[bool] = []

    def run_claim() -> None:
        claimant = TrackingService(tmp_path)
        caller = claimant.load_experiment(experiment.experiment_id)
        barrier.wait(timeout=5)
        with claimant.hf_provisioning_execution_lock(experiment.experiment_id):
            outcomes.append(
                claimant.claim_hf_provisioning(
                    caller, claim
                ).provider_attempt_authorized
            )

    threads = [threading.Thread(target=run_claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == [False, True]


def test_hf_provisioning_ambiguity_is_terminal_without_evidence(tmp_path: Path) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        terminal = build_hf_provisioning_ambiguous_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
            occurred_at="2026-08-20T12:01:00Z",
        )
        service.record_hf_provisioning_ambiguous(experiment, terminal)
    assert experiment.hf_provisioning_state == "AMBIGUOUS"
    assert experiment.source_transport_state == "PREPARED"
    assert experiment.provisioning_evidence_uri is None
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        recovered = service.claim_hf_provisioning(experiment, claim)
    assert recovered.provider_attempt_authorized is False
    assert recovered.state == "AMBIGUOUS"


def test_source_lock_identical_orphan_is_adopted_but_mismatch_is_never_overwritten(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="source-lock-orphan",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    source_lock = _source_lock(experiment.experiment_id)
    path = service._experiment_path(experiment.experiment_id).with_name("source-lock.json")
    exact = (json.dumps(source_lock.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(exact)

    adopted = service.persist_source_lock(experiment, source_lock)
    assert adopted == source_lock
    assert service.load_source_lock(experiment) == source_lock
    assert path.read_bytes() == exact

    other = service.create_experiment(
        name="source-lock-conflict",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    other_path = service._experiment_path(other.experiment_id).with_name("source-lock.json")
    hostile = b'{"hostile":true}\n'
    other_path.write_bytes(hostile)
    record_before = service._experiment_path(other.experiment_id).read_bytes()
    with pytest.raises(ProvenanceIntegrityError, match="not byte-identical"):
        service.persist_source_lock(other, _source_lock(other.experiment_id))
    assert other_path.read_bytes() == hostile
    assert service._experiment_path(other.experiment_id).read_bytes() == record_before
    assert service.load_experiment(other.experiment_id).source_lock_uri is None


def test_source_preparation_execution_lock_is_thread_exclusive_and_nonreentrant(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment_id = "exp-preparation-lock"
    with service.hf_source_preparation_execution_lock(experiment_id):
        with pytest.raises(ProvenanceIntegrityError, match="not reentrant"):
            with service.hf_source_preparation_execution_lock(experiment_id):
                pass
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with service.hf_source_preparation_execution_lock(experiment_id):
            entered.set()
            release.wait(timeout=5)

    def second() -> None:
        entered.wait(timeout=5)
        with TrackingService(tmp_path).hf_source_preparation_execution_lock(experiment_id):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert entered.wait(timeout=5)
    second_thread.start()
    assert not second_entered.wait(timeout=0.2)
    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert second_entered.is_set()


def test_concurrent_identical_source_lock_persistence_is_create_or_adopt(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="source-lock-concurrent",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    source_lock = _source_lock(experiment.experiment_id)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def persist(caller: Experiment) -> None:
        barrier.wait(timeout=5)
        try:
            loaded = TrackingService(tmp_path).persist_source_lock(caller, source_lock)
            outcomes.append("ok" if loaded == source_lock else "wrong")
        except Exception as exc:
            outcomes.append(type(exc).__name__)

    threads = [threading.Thread(target=persist, args=(caller,)) for caller in callers]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert outcomes == ["ok", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert service.load_source_lock(durable) == source_lock


def test_bounded_orphan_terminal_discovery_and_atomic_adoption(tmp_path: Path) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/evidence.json",
        _evidence(
            experiment,
            experiment.source_transport_uri or "",
            experiment.source_transport_sha256 or "",
        ),
    )
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        assert service.find_hf_provisioning_terminal(experiment) is None
        terminal = build_hf_provisioning_succeeded_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
            occurred_at="2026-08-20T12:01:00Z",
        )
        terminal_path = service._hf_provisioning_event_path(
            experiment.experiment_id, str(terminal["event_id"])
        )
        terminal_path.write_bytes(canonical_provisioning_bytes(terminal))
        recovered = service.find_hf_provisioning_terminal(experiment)
        assert recovered is not None
        assert recovered.state == "SUCCEEDED"
        assert recovered.provider_attempt_authorized is False
        service.record_hf_provisioning_succeeded(
            experiment,
            recovered.document,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
        )
    assert experiment.hf_provisioning_state == "SUCCEEDED"
    assert experiment.source_transport_state == "ACKNOWLEDGED"


def test_orphan_terminal_discovery_rejects_unknown_and_conflicting_artifacts(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        events_dir = service._hf_provisioning_event_path(
            experiment.experiment_id, claimed.event_id
        ).parent
        unknown = events_dir / "unknown.tmp"
        unknown.write_bytes(b"x")
        with pytest.raises(ProvenanceIntegrityError, match="unknown artifact"):
            service.find_hf_provisioning_terminal(experiment)
        unknown.unlink()
        first = build_hf_provisioning_ambiguous_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            reason_code="LOCAL_POSTCLAIM_FAILURE",
            occurred_at="2026-08-20T12:01:00Z",
        )
        second = build_hf_provisioning_ambiguous_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            reason_code="PROVIDER_OUTCOME_AMBIGUOUS",
            occurred_at="2026-08-20T12:01:01Z",
        )
        for event in (first, second):
            service._hf_provisioning_event_path(
                experiment.experiment_id, str(event["event_id"])
            ).write_bytes(canonical_provisioning_bytes(event))
        with pytest.raises(ProvenanceIntegrityError, match="multiple terminal"):
            service.find_hf_provisioning_terminal(experiment)


def test_generic_mutation_cannot_erase_or_replace_provisioning_projection(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    stale = service.load_experiment(experiment.experiment_id)
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        service.claim_hf_provisioning(experiment, claim)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    stale.name = "hostile stale mutation"
    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        service.save_experiment(stale)
    assert record_path.read_bytes() == durable_before


def test_spawned_provisioning_claim_is_kernel_exclusive_and_crash_durable(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_provisioning_process,
            args=(str(tmp_path), experiment.experiment_id, claim, start, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=20) for _ in processes]
    except queue.Empty:
        pytest.fail("spawned provisioning claimant did not report within timeout")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()
    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcome[1] for outcome in outcomes) == [False, True]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_provisioning_state == "CLAIMED"
    assert durable.provisioning_evidence_uri is None


def test_legacy_consumable_record_is_readable_but_cannot_claim_provider_authority(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    experiment, claim = _make_prepared_provisioning_claim(service)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        "legacy-evidence.json",
        _evidence(experiment, experiment.source_transport_uri or "", experiment.source_transport_sha256 or ""),
    )
    service.record_provisioning_acknowledged(
        experiment, uri=evidence_uri, sha256=evidence_sha
    )
    service.mark_source_transport_consumable(experiment)
    legacy = service.load_experiment(experiment.experiment_id)
    assert legacy.hf_provisioning_state is None
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        with pytest.raises(ProvenanceIntegrityError, match="requires PREPARED"):
            service.claim_hf_provisioning(legacy, claim)


def test_legacy_consumable_cannot_create_approval_or_submission_artifacts(
    tmp_path: Path,
) -> None:
    service = TrackingService(tmp_path)
    legacy, _ = _make_consumable_transport(service)
    assert legacy.source_transport_state == "CONSUMABLE"
    assert legacy.hf_provisioning_state is None
    approval = _approval(legacy)
    submitting = build_hf_submitting_event(
        approval,
        approval_uri="tracking://hostile-approval.json",
        occurred_at="2026-08-20T12:01:00Z",
    )
    record_path = service._experiment_path(legacy.experiment_id)
    before_bytes = record_path.read_bytes()
    before_files = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ProvenanceIntegrityError, match="SUCCEEDED provisioning"):
        service.record_hf_run_approval(legacy, approval)
    with pytest.raises(ProvenanceIntegrityError, match="SUCCEEDED provisioning"):
        service.claim_hf_submission(legacy, submitting)

    assert record_path.read_bytes() == before_bytes
    assert service.load_experiment(legacy.experiment_id).hf_submission_state is None
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before_files


def _approval(experiment: Experiment, *, run_id: str = "hf-smoke-1"):
    return build_hf_run_approval(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        descriptor_uri=experiment.source_transport_uri or "",
        descriptor_sha256=experiment.source_transport_sha256 or "",
        provisioning_evidence_uri=experiment.provisioning_evidence_uri or "",
        provisioning_evidence_sha256=experiment.provisioning_evidence_sha256 or "",
        source_lock_uri=experiment.source_lock_uri or "",
        source_lock_sha256=experiment.source_lock_sha256 or "",
        bundle_sha256="5" * 64,
        capsule_manifest_sha256="3" * 64,
        checkout_policy_sha256="4" * 64,
        hardware_flavor="cpu-basic",
        user_authorization_reference="conversation-2026-08-20-one-hf-smoke",
        issued_at="2026-08-20T12:00:00Z",
        expires_at="2026-08-20T13:00:00Z",
        hourly_price_usd="0.01",
        projected_cost_usd="0.01",
        quoted_at="2026-08-20T11:59:00Z",
    )


def _make_approved_submission(
    service: TrackingService,
) -> tuple[Experiment, HFRunApproval]:
    experiment, claim = _make_prepared_provisioning_claim(service)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/evidence.json",
        _evidence(
            experiment,
            experiment.source_transport_uri or "",
            experiment.source_transport_sha256 or "",
        ),
    )
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        terminal = build_hf_provisioning_succeeded_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
            occurred_at="2026-08-20T12:00:30Z",
        )
        service.record_hf_provisioning_succeeded(
            experiment,
            terminal,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
        )
    service.mark_source_transport_consumable(experiment)
    approval = _approval(experiment)
    service.record_hf_run_approval(experiment, approval)
    return experiment, approval


def _make_submitted_submission(
    service: TrackingService,
) -> tuple[Experiment, HFRunApproval, object]:
    experiment, approval = _make_approved_submission(service)
    submitting = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, submitting)
    submitted = build_hf_submitted_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        previous_event=submitting,
        previous_event_uri=experiment.hf_submission_event_uri or "",
        occurred_at="2026-08-20T12:02:00Z",
        provider_namespace="professorsynapse",
        provider_job_id="job-smoke-1",
    )
    service.record_hf_submission_terminal(experiment, submitted)
    return experiment, approval, submitted


def test_hf_cancellation_claim_is_durable_exclusive_and_identity_equal_on_resume(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    before_claim = service.load_experiment(experiment.experiment_id)
    event = service.build_hf_cancellation_attempt_event(
        before_claim,
        occurred_at="2026-08-20T12:12:00Z",
    )

    first = service.claim_hf_cancellation(before_claim, event)
    assert first.provider_attempt_authorized is True
    assert first.document == event
    detached = first.document
    detached["occurred_at"] = "2099-01-01T00:00:00Z"
    assert first.document == event
    with pytest.raises(FrozenInstanceError):
        first.provider_attempt_authorized = False
    assert before_claim.hf_cancellation_state == "CLAIMED"
    assert service.resolve_uri(first.event_uri).read_bytes().endswith(b"\n")

    stale_resume = service.load_experiment(experiment.experiment_id)
    second = TrackingService(tmp_path).claim_hf_cancellation(stale_resume, event)
    third = TrackingService(tmp_path).claim_hf_cancellation(before_claim, event)

    assert second.provider_attempt_authorized is False
    assert third.provider_attempt_authorized is False
    assert second.document == first.document == third.document
    assert second.event_uri == first.event_uri == third.event_uri
    assert second.event_sha256 == first.event_sha256 == third.event_sha256
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_submission_state == "SUBMITTED"
    assert durable.source_transport_state == "CONSUMABLE"
    service.verify_experiment_provenance(durable)


def test_hf_cancellation_rejects_non_submitted_replacement_and_caller_provider_identity(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    approved, _ = _make_approved_submission(service)
    with pytest.raises(ProvenanceIntegrityError, match="SUBMITTED"):
        service.build_hf_cancellation_attempt_event(
            approved,
            occurred_at="2026-08-20T12:12:00Z",
        )

    experiment, _, _ = _make_submitted_submission(service)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    first = service.claim_hf_cancellation(experiment, event)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()

    replacement = dict(event)
    replacement["occurred_at"] = "2026-08-20T12:13:00Z"
    replacement["event_id"] = service._hf_cancellation_event_id(replacement)
    with pytest.raises(ProvenanceIntegrityError, match="replayed or replaced"):
        service.claim_hf_cancellation(experiment, replacement)

    hostile = dict(event)
    hostile["provider_job"] = {
        "namespace": "attacker",
        "job_id": "different-job",
    }
    hostile["event_id"] = service._hf_cancellation_event_id(hostile)
    with pytest.raises(ProvenanceIntegrityError, match="provider_job"):
        service.claim_hf_cancellation(experiment, hostile)

    assert first.provider_attempt_authorized is True
    assert record_path.read_bytes() == durable_before


def test_competing_thread_hf_cancellation_claims_grant_one_provider_attempt(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    outcomes: list[bool] = []
    errors: list[str] = []

    def claim(caller: Experiment) -> None:
        barrier.wait(timeout=5)
        try:
            result = TrackingService(tmp_path).claim_hf_cancellation(caller, event)
            outcomes.append(result.provider_attempt_authorized)
        except Exception as exc:
            errors.append(type(exc).__name__)

    threads = [threading.Thread(target=claim, args=(caller,)) for caller in callers]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert sorted(outcomes) == [False, True]
    assert service.load_experiment(experiment.experiment_id).hf_cancellation_state == "CLAIMED"


def test_spawned_hf_cancellation_claims_grant_one_provider_attempt(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_cancellation_process,
            args=(
                str(tmp_path),
                experiment.experiment_id,
                event,
                start,
                results,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=15) for _ in processes]
    except queue.Empty:
        pytest.fail("spawned HF cancellation process did not report within timeout")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()

    assert all(process.exitcode == 0 for process in processes)
    assert all(outcome[0] == "ok" for outcome in outcomes)
    assert sorted(outcome[1] for outcome in outcomes) == [False, True]
    assert len({outcome[2] for outcome in outcomes}) == 1
    assert service.load_experiment(experiment.experiment_id).hf_cancellation_state == "CLAIMED"


@pytest.mark.parametrize(
    "mutation",
    [
        "save",
        "mark_stage",
        "update_stage_details",
        "set_artifact_root",
        "set_derived_output",
        "attach_run",
    ],
)
def test_generic_mutations_cannot_erase_hf_cancellation_claim(
    tmp_path: Path, mutation: str
):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    stale = service.load_experiment(experiment.experiment_id)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    service.claim_hf_cancellation(experiment, event)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    registry_before = service.registry.path.read_bytes() if service.registry.path.exists() else None

    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        if mutation == "save":
            stale.name = "stale cancellation overwrite"
            service.save_experiment(stale)
        elif mutation == "mark_stage":
            service.mark_stage(stale, "training", "running")
        elif mutation == "update_stage_details":
            service.update_stage_details(stale, "training", job_ref="job-stale")
        elif mutation == "set_artifact_root":
            service.set_artifact_root(stale, "training", "artifact://stale")
        elif mutation == "set_derived_output":
            service.set_derived_output(stale, "features_csv", "artifact://stale.csv")
        else:
            service.attach_run(
                stale,
                RunRecord(
                    run_id="stale-cancellation-run",
                    run_type="cloud_sft",
                    name="stale",
                    timestamp="2026-08-20T12:13:00Z",
                    status="running",
                    output_dir="artifact://runs/stale",
                ),
            )

    assert record_path.read_bytes() == durable_before
    assert (
        service.registry.path.read_bytes() if service.registry.path.exists() else None
    ) == registry_before


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("hf_cancellation_event_uri", "tracking://replacement-cancellation.json"),
        ("hf_cancellation_event_sha256", "9" * 64),
        ("hf_cancellation_state", None),
    ],
)
def test_public_save_protects_every_hf_cancellation_projection_field(
    tmp_path: Path, field_name: str, value: str | None
):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    service.claim_hf_cancellation(experiment, event)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    setattr(experiment, field_name, value)

    with pytest.raises((ProvenanceIntegrityError, ValueError)):
        service.save_experiment(experiment)

    assert record_path.read_bytes() == durable_before


def test_attach_run_propagates_hf_cancellation_claim(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, _, _ = _make_submitted_submission(service)
    event = service.build_hf_cancellation_attempt_event(
        experiment,
        occurred_at="2026-08-20T12:12:00Z",
    )
    claim = service.claim_hf_cancellation(experiment, event)
    service.attach_run(
        experiment,
        RunRecord(
            run_id="cancellation-claimed-run",
            run_type="cloud_sft",
            name="claimed",
            timestamp="2026-08-20T12:13:00Z",
            status="running",
            output_dir="artifact://runs/claimed",
        ),
    )

    stored = service.registry.get_run("cancellation-claimed-run")
    assert stored is not None
    assert stored.hf_cancellation_event_uri == claim.event_uri
    assert stored.hf_cancellation_event_sha256 == claim.event_sha256
    assert stored.hf_cancellation_state == "CLAIMED"


@pytest.mark.parametrize("terminal_state", ["SUBMITTED", "AMBIGUOUS"])
def test_hf_submission_claim_is_separate_durable_and_terminal(
    tmp_path: Path, terminal_state: str
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)

    assert experiment.source_transport_state == "CONSUMABLE"
    assert experiment.hf_submission_state == "APPROVED"
    assert experiment.hf_authorization_id == approval.authorization_id
    approval_path = service.resolve_uri(experiment.hf_run_approval_uri or "")
    assert approval_path.read_bytes() == canonical_approval_bytes(approval.to_dict())

    submitting = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, submitting)
    submitting_uri = experiment.hf_submission_event_uri
    submitting_sha256 = experiment.hf_submission_event_sha256
    assert experiment.hf_submission_state == "SUBMITTING"
    assert experiment.source_transport_state == "CONSUMABLE"

    if terminal_state == "SUBMITTED":
        terminal = build_hf_submitted_event(
            approval,
            approval_uri=experiment.hf_run_approval_uri or "",
            previous_event=submitting,
            previous_event_uri=submitting_uri or "",
            occurred_at="2026-08-20T12:02:00Z",
            provider_namespace="professorsynapse",
            provider_job_id="job-smoke-1",
        )
    else:
        terminal = build_hf_ambiguous_event(
            approval,
            approval_uri=experiment.hf_run_approval_uri or "",
            previous_event=submitting,
            previous_event_uri=submitting_uri or "",
            occurred_at="2026-08-20T12:02:00Z",
            reason_code="SUBMISSION_RESPONSE_LOST",
        )
    service.record_hf_submission_terminal(experiment, terminal)

    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_submission_state == terminal_state
    assert durable.source_transport_state == "CONSUMABLE"
    assert durable.hf_submission_event_uri != submitting_uri
    assert service.resolve_uri(submitting_uri or "").is_file()
    assert hashlib.sha256(service.resolve_uri(submitting_uri or "").read_bytes()).hexdigest() == (
        submitting_sha256
    )
    service.verify_experiment_provenance(durable)


def test_hf_approval_and_submission_replay_replacement_and_backward_transitions_fail(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    record_path = service._experiment_path(experiment.experiment_id)
    approved_bytes = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="replayed or replaced"):
        service.record_hf_run_approval(experiment, approval)
    assert record_path.read_bytes() == approved_bytes

    submitting = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, submitting)
    claimed_bytes = record_path.read_bytes()
    with pytest.raises(ProvenanceIntegrityError, match="already claimed"):
        service.claim_hf_submission(experiment, submitting)
    with pytest.raises(ProvenanceIntegrityError, match="terminal submission event is invalid"):
        service.record_hf_submission_terminal(
            service.load_experiment(experiment.experiment_id),
            submitting,
        )
    assert record_path.read_bytes() == claimed_bytes

    ambiguous = build_hf_ambiguous_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        previous_event=submitting,
        previous_event_uri=experiment.hf_submission_event_uri or "",
        occurred_at="2026-08-20T12:02:00Z",
        reason_code="PROVIDER_STATUS_UNKNOWN",
    )
    service.record_hf_submission_terminal(experiment, ambiguous)
    terminal_bytes = record_path.read_bytes()
    with pytest.raises(ProvenanceIntegrityError, match="active SUBMITTING"):
        service.record_hf_submission_terminal(experiment, ambiguous)
    with pytest.raises(ProvenanceIntegrityError, match="already claimed"):
        service.claim_hf_submission(experiment, submitting)
    assert record_path.read_bytes() == terminal_bytes


def test_hf_terminal_event_must_bind_exact_durable_submitting_head(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    submitting = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, submitting)
    record_path = service._experiment_path(experiment.experiment_id)
    before = record_path.read_bytes()

    terminal = build_hf_ambiguous_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        previous_event=submitting,
        previous_event_uri="tracking://wrong-submitting-event.json",
        occurred_at="2026-08-20T12:02:00Z",
        reason_code="PROVIDER_STATUS_UNKNOWN",
    )
    with pytest.raises(ProvenanceIntegrityError, match="durable SUBMITTING"):
        service.record_hf_submission_terminal(experiment, terminal)

    assert record_path.read_bytes() == before
    assert service.load_experiment(experiment.experiment_id).hf_submission_state == "SUBMITTING"


def test_competing_thread_hf_claims_are_exclusive_by_authorization_id(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    event = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    ).to_dict()
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def claim(caller: Experiment) -> None:
        barrier.wait(timeout=5)
        try:
            TrackingService(tmp_path).claim_hf_submission(caller, event)
            outcomes.append("ok")
        except ProvenanceIntegrityError:
            outcomes.append("error")

    threads = [threading.Thread(target=claim, args=(caller,)) for caller in callers]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["error", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_submission_state == "SUBMITTING"
    assert durable.hf_authorization_id == approval.authorization_id


def test_competing_terminal_events_use_the_durable_submission_head(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    submitting = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, submitting)
    previous_uri = experiment.hf_submission_event_uri or ""
    submitted = build_hf_submitted_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        previous_event=submitting,
        previous_event_uri=previous_uri,
        occurred_at="2026-08-20T12:02:00Z",
        provider_namespace="professorsynapse",
        provider_job_id="job-smoke-1",
    )
    ambiguous = build_hf_ambiguous_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        previous_event=submitting,
        previous_event_uri=previous_uri,
        occurred_at="2026-08-20T12:02:00Z",
        reason_code="PROVIDER_STATUS_UNKNOWN",
    )
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def finish(caller: Experiment, event) -> None:
        barrier.wait(timeout=5)
        try:
            TrackingService(tmp_path).record_hf_submission_terminal(caller, event)
            outcomes.append("ok")
        except ProvenanceIntegrityError:
            outcomes.append("error")

    threads = [
        threading.Thread(target=finish, args=(callers[0], submitted)),
        threading.Thread(target=finish, args=(callers[1], ambiguous)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["error", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_submission_state in {"SUBMITTED", "AMBIGUOUS"}
    service.verify_experiment_provenance(durable)


def test_spawned_hf_claim_is_exclusive_and_crash_after_claim_stays_consumed(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    event = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    ).to_dict()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_submission_process,
            args=(
                str(tmp_path),
                experiment.experiment_id,
                event,
                start,
                results,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=15) for _ in processes]
    except queue.Empty:
        pytest.fail("spawned HF claim process did not report within timeout")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcome[0] for outcome in outcomes) == ["error", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.hf_submission_state == "SUBMITTING"
    assert service.resolve_uri(durable.hf_submission_event_uri or "").is_file()

    # A worker disappearing after the durable claim cannot return the approval
    # to APPROVED or permit a paid retry.
    with pytest.raises(ProvenanceIntegrityError, match="already claimed"):
        TrackingService(tmp_path).claim_hf_submission(durable, event)
    assert service.load_experiment(experiment.experiment_id).hf_submission_state == "SUBMITTING"


@pytest.mark.parametrize(
    "mutation",
    [
        "save",
        "mark_stage",
        "update_stage_details",
        "set_artifact_root",
        "set_derived_output",
        "attach_run",
    ],
)
def test_generic_mutations_cannot_erase_or_roll_back_hf_claim(
    tmp_path: Path, mutation: str
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    stale = service.load_experiment(experiment.experiment_id)
    event = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, event)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    registry_before = service.registry.path.read_bytes() if service.registry.path.exists() else None

    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        if mutation == "save":
            stale.name = "stale approval overwrite"
            service.save_experiment(stale)
        elif mutation == "mark_stage":
            service.mark_stage(stale, "training", "running")
        elif mutation == "update_stage_details":
            service.update_stage_details(stale, "training", job_ref="job-stale")
        elif mutation == "set_artifact_root":
            service.set_artifact_root(stale, "training", "artifact://stale")
        elif mutation == "set_derived_output":
            service.set_derived_output(stale, "features_csv", "artifact://stale.csv")
        else:
            service.attach_run(
                stale,
                RunRecord(
                    run_id="stale-hf-claim-run",
                    run_type="sft",
                    name="stale",
                    timestamp="2026-08-20T12:02:00Z",
                    status="running",
                    output_dir="artifact://runs/stale",
                ),
            )

    assert record_path.read_bytes() == durable_before
    assert (
        service.registry.path.read_bytes() if service.registry.path.exists() else None
    ) == registry_before


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("hf_run_approval_uri", "tracking://replacement-approval.json"),
        ("hf_run_approval_sha256", "9" * 64),
        ("hf_authorization_id", "8" * 64),
        ("hf_submission_event_uri", "tracking://replacement-event.json"),
        ("hf_submission_event_sha256", "7" * 64),
        ("hf_submission_state", "APPROVED"),
    ],
)
def test_public_save_protects_every_hf_submission_projection_field(
    tmp_path: Path, field_name: str, value: str
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    event = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, event)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    setattr(experiment, field_name, value)

    with pytest.raises((ProvenanceIntegrityError, ValueError)):
        service.save_experiment(experiment)

    assert record_path.read_bytes() == durable_before


def test_attach_run_propagates_exact_hf_submission_projection_and_rejects_replacement(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment, approval = _make_approved_submission(service)
    event = build_hf_submitting_event(
        approval,
        approval_uri=experiment.hf_run_approval_uri or "",
        occurred_at="2026-08-20T12:01:00Z",
    )
    service.claim_hf_submission(experiment, event)

    with pytest.raises(ProvenanceIntegrityError, match="hf_authorization_id"):
        service.attach_run(
            experiment,
            RunRecord(
                run_id="wrong-authorization-run",
                run_type="cloud_sft",
                name="wrong",
                timestamp="2026-08-20T12:02:00Z",
                status="running",
                output_dir="artifact://runs/wrong",
                source_transport_uri=experiment.source_transport_uri,
                source_transport_sha256=experiment.source_transport_sha256,
                provisioning_evidence_uri=experiment.provisioning_evidence_uri,
                provisioning_evidence_sha256=experiment.provisioning_evidence_sha256,
                source_transport_state="CONSUMABLE",
                hf_provisioning_event_uri=experiment.hf_provisioning_event_uri,
                hf_provisioning_event_sha256=experiment.hf_provisioning_event_sha256,
                hf_provisioning_state="SUCCEEDED",
                hf_run_approval_uri=experiment.hf_run_approval_uri,
                hf_run_approval_sha256=experiment.hf_run_approval_sha256,
                hf_authorization_id="9" * 64,
                hf_submission_event_uri=experiment.hf_submission_event_uri,
                hf_submission_event_sha256=experiment.hf_submission_event_sha256,
                hf_submission_state="SUBMITTING",
            ),
        )
    assert service.registry.get_run("wrong-authorization-run") is None

    service.attach_run(
        experiment,
        RunRecord(
            run_id="claimed-run",
            run_type="cloud_sft",
            name="claimed",
            timestamp="2026-08-20T12:02:00Z",
            status="running",
            output_dir="artifact://runs/claimed",
        ),
    )
    stored = service.registry.get_run("claimed-run")
    assert stored is not None
    assert stored.hf_run_approval_uri == experiment.hf_run_approval_uri
    assert stored.hf_run_approval_sha256 == experiment.hf_run_approval_sha256
    assert stored.hf_authorization_id == approval.authorization_id
    assert stored.hf_submission_event_uri == experiment.hf_submission_event_uri
    assert stored.hf_submission_event_sha256 == experiment.hf_submission_event_sha256
    assert stored.hf_submission_state == "SUBMITTING"


@pytest.mark.parametrize(
    "mutation",
    [
        "save",
        "mark_stage",
        "update_stage_details",
        "set_artifact_root",
        "set_derived_output",
        "attach_run",
    ],
)
def test_generic_mutations_reject_stale_transport_and_preserve_durable_bytes(
    tmp_path: Path, mutation: str
):
    service = TrackingService(tmp_path)
    experiment, stale = _make_consumable_transport(service)
    record_path = service._experiment_path(experiment.experiment_id)
    registry_before = service.registry.path.read_bytes() if service.registry.path.exists() else None
    durable_before = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="conflicts with durable"):
        if mutation == "save":
            stale.name = "stale overwrite"
            service.save_experiment(stale)
        elif mutation == "mark_stage":
            service.mark_stage(stale, "training", "running")
        elif mutation == "update_stage_details":
            service.update_stage_details(stale, "training", job_ref="job-stale")
        elif mutation == "set_artifact_root":
            service.set_artifact_root(stale, "training", "artifact://stale")
        elif mutation == "set_derived_output":
            service.set_derived_output(stale, "features_csv", "artifact://stale.csv")
        else:
            service.attach_run(
                stale,
                RunRecord(
                    run_id="stale-run",
                    run_type="sft",
                    name="stale",
                    timestamp="2026-08-19T12:00:00Z",
                    status="running",
                    output_dir="artifact://runs/stale",
                ),
            )

    assert record_path.read_bytes() == durable_before
    assert (
        service.registry.path.read_bytes() if service.registry.path.exists() else None
    ) == registry_before


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_transport_uri", "tracking://swapped-descriptor.json"),
        ("provisioning_evidence_sha256", "9" * 64),
        ("source_transport_state", "SUBMITTED"),
    ],
)
def test_public_save_rejects_transport_ref_swap_and_submitted_projection(
    tmp_path: Path, field_name: str, value: str
):
    service = TrackingService(tmp_path)
    experiment, _ = _make_consumable_transport(service)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    setattr(experiment, field_name, value)

    with pytest.raises(ProvenanceIntegrityError, match="conflicts with durable"):
        service.save_experiment(experiment)

    assert record_path.read_bytes() == durable_before


def test_public_save_rejects_non_neutral_new_transport_without_writes(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = Experiment(
        experiment_id="injected",
        name="injected",
        created_at="2026-08-19T12:00:00Z",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
        source_transport_uri="tracking://descriptor.json",
        source_transport_sha256="1" * 64,
        source_transport_state="PREPARED",
    )

    with pytest.raises(ProvenanceIntegrityError, match="neutral"):
        service.save_experiment(experiment)

    assert not (tmp_path / "experiments").exists()


def test_public_save_merges_ordinary_fields_and_idempotent_mutations_preserve_bytes(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="ordinary",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    experiment.objective = "updated"
    service.save_experiment(experiment)
    assert service.load_experiment(experiment.experiment_id).objective == "updated"

    service.set_derived_output(experiment, "features_csv", "artifact://features.csv")
    record_path = service._experiment_path(experiment.experiment_id)
    derived_bytes = record_path.read_bytes()
    service.set_derived_output(experiment, "features_csv", "artifact://features.csv")
    assert record_path.read_bytes() == derived_bytes

    service.mark_stage(experiment, "training", "running")
    stage_bytes = record_path.read_bytes()
    service.mark_stage(experiment, "training", "running")
    assert record_path.read_bytes() == stage_bytes


def test_public_save_three_way_merges_disjoint_edits_and_rejects_conflicts(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="baseline",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    writer_a = service.load_experiment(experiment.experiment_id)
    writer_b = service.load_experiment(experiment.experiment_id)
    writer_a.objective = "objective-a"
    service.save_experiment(writer_a)
    writer_b.name = "name-b"
    service.save_experiment(writer_b)

    merged = service.load_experiment(experiment.experiment_id)
    assert merged.objective == "objective-a"
    assert merged.name == "name-b"

    conflicting_a = service.load_experiment(experiment.experiment_id)
    conflicting_b = service.load_experiment(experiment.experiment_id)
    conflicting_a.objective = "objective-c"
    service.save_experiment(conflicting_a)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    conflicting_b.objective = "objective-d"

    with pytest.raises(ProvenanceIntegrityError, match="conflicts on objective"):
        service.save_experiment(conflicting_b)

    assert record_path.read_bytes() == durable_before
    assert service.load_experiment(experiment.experiment_id).objective == "objective-c"


def _make_complete_experiment_provenance(
    service: TrackingService,
) -> Experiment:
    experiment = service.create_experiment(
        name="complete-provenance",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    service.persist_source_lock(experiment, _source_lock(experiment.experiment_id))
    resolved = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://spec.yaml",
                data={"experiment": {"name": "complete-provenance"}},
                precedence=0,
            )
        ]
    )
    service.persist_resolved_config(experiment, resolved)
    return experiment


@pytest.mark.parametrize(
    "mutation",
    [
        "source_complete_replace",
        "source_complete_remove",
        "source_partial_uri",
        "source_partial_sha256",
        "resolved_complete_replace",
        "resolved_complete_remove",
        "resolved_partial_uri",
        "resolved_partial_sha256",
    ],
)
def test_public_save_rejects_complete_and_partial_core_provenance_changes(
    tmp_path: Path, mutation: str
):
    service = TrackingService(tmp_path)
    experiment = _make_complete_experiment_provenance(service)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    changes = {
        "source_complete_replace": {
            "source_lock_uri": "tracking://replacement-source-lock.json",
            "source_lock_sha256": "8" * 64,
        },
        "source_complete_remove": {
            "source_lock_uri": None,
            "source_lock_sha256": None,
        },
        "source_partial_uri": {"source_lock_uri": None},
        "source_partial_sha256": {"source_lock_sha256": None},
        "resolved_complete_replace": {
            "resolved_config_uri": "tracking://replacement-resolved-config.json",
            "resolved_config_sha256": "7" * 64,
        },
        "resolved_complete_remove": {
            "resolved_config_uri": None,
            "resolved_config_sha256": None,
        },
        "resolved_partial_uri": {"resolved_config_uri": None},
        "resolved_partial_sha256": {"resolved_config_sha256": None},
    }[mutation]
    for field_name, value in changes.items():
        setattr(experiment, field_name, value)

    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        service.save_experiment(experiment)

    assert record_path.read_bytes() == durable_before


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_lock_uri", "tracking://introduced-source-lock.json"),
        ("source_lock_sha256", "6" * 64),
        ("resolved_config_uri", "tracking://introduced-resolved-config.json"),
        ("resolved_config_sha256", "5" * 64),
    ],
)
def test_generic_mutator_rejects_core_provenance_introduction_or_partial_pair(
    tmp_path: Path, field_name: str, value: str
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="neutral-provenance",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()
    setattr(experiment, field_name, value)

    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        service.mark_stage(experiment, "training", "running")

    assert record_path.read_bytes() == durable_before


@pytest.mark.parametrize(
    "record_provenance",
    [
        {
            "source_lock_uri": "tracking://introduced-source-lock.json",
            "source_lock_sha256": "4" * 64,
        },
        {"source_lock_uri": "tracking://partial-source-lock.json"},
        {
            "resolved_config_uri": "tracking://introduced-resolved-config.json",
            "resolved_config_sha256": "3" * 64,
        },
        {"resolved_config_sha256": "2" * 64},
    ],
)
def test_attach_run_cannot_introduce_complete_or_partial_core_provenance(
    tmp_path: Path, record_provenance: dict[str, str]
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="attach-neutral",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="protected provenance"):
        service.attach_run(
            experiment,
            RunRecord(
                run_id="hostile-run",
                run_type="sft",
                name="hostile",
                timestamp="2026-08-19T12:00:00Z",
                status="running",
                output_dir="artifact://runs/hostile",
                **record_provenance,
            ),
        )

    assert record_path.read_bytes() == durable_before
    assert not service.registry.path.exists()


def test_attach_run_rejects_core_provenance_replacement_without_registry_write(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment = _make_complete_experiment_provenance(service)
    record_path = service._experiment_path(experiment.experiment_id)
    durable_before = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="source_lock_sha256"):
        service.attach_run(
            experiment,
            RunRecord(
                run_id="replacement-run",
                run_type="sft",
                name="replacement",
                timestamp="2026-08-19T12:00:00Z",
                status="running",
                output_dir="artifact://runs/replacement",
                source_lock_uri=experiment.source_lock_uri,
                source_lock_sha256="1" * 64,
                resolved_config_uri=experiment.resolved_config_uri,
                resolved_config_sha256=experiment.resolved_config_sha256,
            ),
        )

    assert record_path.read_bytes() == durable_before
    assert not service.registry.path.exists()


def test_dedicated_provenance_first_write_is_idempotent_and_replacement_safe(
    tmp_path: Path,
):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="provenance-cas",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    stale_source = TrackingService(tmp_path).load_experiment(experiment.experiment_id)
    source_lock = _source_lock(experiment.experiment_id)
    service.persist_source_lock(experiment, source_lock)
    record_path = service._experiment_path(experiment.experiment_id)
    source_bytes = record_path.read_bytes()

    TrackingService(tmp_path).persist_source_lock(stale_source, source_lock)

    assert record_path.read_bytes() == source_bytes
    assert stale_source.source_lock_uri == experiment.source_lock_uri
    assert stale_source.source_lock_sha256 == experiment.source_lock_sha256

    stale_config = TrackingService(tmp_path).load_experiment(experiment.experiment_id)
    config_a = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://spec-a.yaml",
                data={"value": "a"},
                precedence=0,
            )
        ]
    )
    config_b = resolve_config_layers(
        [
            ConfigDocument.from_mapping(
                uri="project://spec-b.yaml",
                data={"value": "b"},
                precedence=0,
            )
        ]
    )
    service.persist_resolved_config(experiment, config_a)
    config_bytes = record_path.read_bytes()

    with pytest.raises(ProvenanceIntegrityError, match="does not match stored"):
        TrackingService(tmp_path).persist_resolved_config(stale_config, config_b)

    assert record_path.read_bytes() == config_bytes


def test_threaded_generic_mutations_merge_against_fresh_durable_state(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="thread-merge",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    errors: list[str] = []

    def update(caller: Experiment, key: str) -> None:
        barrier.wait(timeout=5)
        try:
            TrackingService(tmp_path).set_derived_output(caller, key, f"artifact://{key}")
        except Exception as exc:
            errors.append(type(exc).__name__)

    threads = [
        threading.Thread(target=update, args=(callers[0], "features_csv")),
        threading.Thread(target=update, args=(callers[1], "judge_scores")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.derived_outputs == {
        "features_csv": "artifact://features_csv",
        "judge_scores": "artifact://judge_scores",
    }
    assert any(len(caller.derived_outputs) == 2 for caller in callers)


def _training_seal(document: dict[str, object], field: str) -> dict[str, object]:
    document[field] = "0" * 64
    document[field] = document_sha256({key: value for key, value in document.items() if key != field})
    return document


def _training_runtime_lock() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[3]
        / "Trainers"
        / "cloud"
        / "runtime-locks"
        / "hf_training_smoke_unsloth_2026_1_2.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _make_training_ready(service: TrackingService) -> Experiment:
    experiment, claim = _make_prepared_provisioning_claim(service)
    evidence_uri, evidence_sha = _write_canonical(
        service,
        f"experiments/{experiment.experiment_id}/cloud/hf/source-transport/evidence.json",
        _evidence(experiment, experiment.source_transport_uri or "", experiment.source_transport_sha256 or ""),
    )
    with service.hf_provisioning_execution_lock(experiment.experiment_id):
        claimed = service.claim_hf_provisioning(experiment, claim)
        terminal = build_hf_provisioning_succeeded_event(
            claimed.document,
            claim_uri=claimed.event_uri,
            claim_sha256=claimed.event_sha256,
            evidence_uri=evidence_uri,
            evidence_sha256=evidence_sha,
            occurred_at="2026-08-20T12:00:30Z",
        )
        service.record_hf_provisioning_succeeded(
            experiment, terminal, evidence_uri=evidence_uri, evidence_sha256=evidence_sha
        )
    service.mark_source_transport_consumable(experiment)
    return experiment


def _training_preflight(
    experiment: Experiment, root_id: str, runtime_lock_ref: dict[str, str]
) -> dict[str, object]:
    slot_input = {
        "schema_version": ARTIFACT_SLOT_INPUT_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "run_id": "training-smoke-1",
        "tracking_root_id": root_id,
        "source_lock_sha256": experiment.source_lock_sha256,
        "workload_digest": "b" * 64,
        "runtime_lock_sha256": runtime_lock_ref["sha256"],
        "artifact_bucket_id": "owner/bucket",
        "artifact_base_prefix": "training/artifacts",
    }
    slot_id = derive_hf_training_artifact_slot(slot_input)
    return _training_seal(
        {
            "schema_version": "synaptic-hf-training-preflight/v1",
            "experiment_id": experiment.experiment_id, "run_id": "training-smoke-1", "tracking_root_id": root_id,
            "occurred_at": "2026-08-20T12:01:00Z", "status": "PASS",
            "source": {
                "descriptor": {"uri": experiment.source_transport_uri, "sha256": experiment.source_transport_sha256},
                "source_lock": {"uri": experiment.source_lock_uri, "sha256": experiment.source_lock_sha256},
                "provisioning_evidence": {"uri": experiment.provisioning_evidence_uri, "sha256": experiment.provisioning_evidence_sha256},
                "bundle_sha256": "5" * 64, "capsule_manifest_sha256": "3" * 64,
                "checkout_policy_sha256": "4" * 64, "project_commit": "1" * 40, "engine_commit": "2" * 40,
            },
            "runtime_lock": runtime_lock_ref,
            "workload_digest": "b" * 64,
            "model": {"repository": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "c" * 40},
            "dataset": {"path": "Datasets/smoke.jsonl", "sha256": "d" * 64, "git_blob": "e" * 40, "bytes": 10, "row_count": 1, "row_sha256": "f" * 64},
            "image": {
                "registry_repository": "docker.io/unsloth/unsloth",
                "provider_repository": "unsloth/unsloth",
                "requested_digest": f"sha256:{'1' * 64}",
                "requested_media_type": "application/vnd.docker.distribution.manifest.v2+json",
                "requested_kind": "manifest",
                "index_digest": None,
                "index_media_type": None,
                "child_digest": f"sha256:{'1' * 64}",
                "child_media_type": "application/vnd.docker.distribution.manifest.v2+json",
                "config_digest": f"sha256:{'3' * 64}",
                "config_media_type": "application/vnd.docker.container.image.v1+json",
                "config_size": 123,
                "platform": "linux/amd64",
                "layers": [{
                    "media_type": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    "digest": f"sha256:{'4' * 64}",
                    "size": 456,
                }],
                "provider_reference": f"unsloth/unsloth@sha256:{'1' * 64}",
            },
            "hardware": {"endpoint": "https://huggingface.co", "flavor": "a10g-small", "unit_cost_micro_usd": 16000, "unit_label": "minute", "hourly_cost_micro_usd": 960000, "timeout_cost_micro_usd": 480000, "fetched_at": "2026-08-20T12:00:00Z"},
            "artifact_slot_input": slot_input, "artifact_slot_id": slot_id,
            "volumes": [
                {"bucket_id": "owner/bucket", "prefix": "source/capsule", "mount_path": "/workspace/synaptic-bootstrap-input", "read_only": True},
                {"bucket_id": "owner/bucket", "prefix": derive_hf_training_artifact_prefix("training/artifacts", slot_id), "mount_path": "/workspace/artifacts", "read_only": False},
            ],
            "command": {"remote_argv_sha256": "4" * 64, "provider_command_sha256": "5" * 64},
            "launcher_auth": {"mode": "explicit_file", "expected_namespace": "owner"}, "job_secrets": [],
        },
        "preflight_id",
    )


def _training_approval(experiment: Experiment) -> dict[str, object]:
    runtime_lock_sha256 = document_sha256(_training_runtime_lock())
    slot_input = {
        "schema_version": ARTIFACT_SLOT_INPUT_SCHEMA,
        "experiment_id": experiment.experiment_id,
        "run_id": experiment.hf_training_run_id,
        "tracking_root_id": experiment.hf_training_root_id,
        "source_lock_sha256": experiment.source_lock_sha256,
        "workload_digest": "b" * 64,
        "runtime_lock_sha256": runtime_lock_sha256,
        "artifact_bucket_id": "owner/bucket",
        "artifact_base_prefix": "training/artifacts",
    }
    slot_id = derive_hf_training_artifact_slot(slot_input)
    return _training_seal(
        {
            "schema_version": "synaptic-hf-training-approval/v1", "kind": "hf.training-smoke",
            "experiment_id": experiment.experiment_id, "run_id": experiment.hf_training_run_id, "tracking_root_id": experiment.hf_training_root_id,
            "preflight": {"uri": experiment.hf_training_preflight_uri, "sha256": experiment.hf_training_preflight_sha256},
            "user_authorization_reference": "conversation-2026-08-20-training-smoke",
            "issued_at": "2026-08-20T12:02:00Z", "expires_at": "2026-08-20T13:02:00Z",
            "hardware": "a10g-small", "hardware_quote": {"preflight_sha256": experiment.hf_training_preflight_sha256, "unit_cost_micro_usd": 16000, "hourly_cost_micro_usd": 960000, "timeout_cost_micro_usd": 480000, "fetched_at": "2026-08-20T12:00:00Z"},
            "provider_timeout_seconds": 1800, "cancel_after_seconds": 1500, "observe_until_seconds": 2100,
            "maximum_submissions": 1, "maximum_retries": 0, "publication": False, "ssh": False, "ports": False, "wandb": False,
            "launcher_auth": {"mode": "explicit_file", "expected_namespace": "owner"}, "job_secrets": [],
            "bindings": {
                "source_lock_sha256": experiment.source_lock_sha256, "workload_digest": "b" * 64,
                "runtime_lock_sha256": runtime_lock_sha256, "model_revision": "c" * 40, "dataset_sha256": "d" * 64,
                "image_child_digest": f"sha256:{'1' * 64}", "remote_argv_sha256": "4" * 64,
                "provider_command_sha256": "5" * 64, "source_bucket_id": "owner/bucket",
                "source_prefix": "source/capsule", "artifact_bucket_id": "owner/bucket",
                "artifact_base_prefix": "training/artifacts",
                "artifact_prefix": derive_hf_training_artifact_prefix("training/artifacts", slot_id),
                "artifact_slot_id": slot_id,
            },
        },
        "authorization_id",
    )


def _make_training_approved(service: TrackingService) -> tuple[Experiment, dict[str, object]]:
    experiment = _make_training_ready(service)
    root_id = str(ensure_tracking_root_identity(service.base_dir)["root_id"])
    runtime_lock_ref = service.snapshot_hf_training_runtime_lock(
        experiment, _training_runtime_lock()
    )
    service.record_hf_training_preflight(
        experiment, _training_preflight(experiment, root_id, runtime_lock_ref)
    )
    approval = _training_approval(experiment)
    service.record_hf_training_approval(experiment, approval)
    return experiment, approval


def test_hf_training_runtime_lock_snapshot_is_exact_idempotent_and_preflight_gated(tmp_path: Path):
    service = TrackingService(tmp_path)
    unready = service.create_experiment(
        name="unready-runtime-lock", dataset_path="data.jsonl", dataset_hash="abc", base_model_name="model"
    )
    with pytest.raises(ProvenanceIntegrityError):
        service.snapshot_hf_training_runtime_lock(unready, _training_runtime_lock())

    experiment = _make_training_ready(service)
    runtime_lock = _training_runtime_lock()
    orphan = (
        service.base_dir
        / "experiments"
        / experiment.experiment_id
        / "cloud"
        / "hf"
        / "training-smoke"
        / "runtime-locks"
        / f"{runtime_lock['lock_id']}.json"
    )
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(canonical_training_bytes(runtime_lock))
    first = service.snapshot_hf_training_runtime_lock(experiment, runtime_lock)
    second = service.snapshot_hf_training_runtime_lock(experiment, runtime_lock)
    assert first == second
    snapshot = service._strict_tracking_file(first["uri"], kind="test runtime lock")
    assert snapshot.read_bytes() == canonical_training_bytes(runtime_lock)
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == first["sha256"]

    root_id = str(ensure_tracking_root_identity(service.base_dir)["root_id"])
    service.record_hf_training_preflight(
        experiment, _training_preflight(experiment, root_id, first)
    )
    with pytest.raises(ProvenanceIntegrityError, match="before preflight"):
        service.snapshot_hf_training_runtime_lock(experiment, runtime_lock)


def test_hf_training_runtime_lock_snapshot_rejects_conflicting_orphan(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = _make_training_ready(service)
    runtime_lock = _training_runtime_lock()
    orphan = (
        service.base_dir
        / "experiments"
        / experiment.experiment_id
        / "cloud"
        / "hf"
        / "training-smoke"
        / "runtime-locks"
        / f"{runtime_lock['lock_id']}.json"
    )
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"{}\n")
    with pytest.raises(ProvenanceIntegrityError, match="cannot be replaced"):
        service.snapshot_hf_training_runtime_lock(experiment, runtime_lock)


@pytest.mark.parametrize("case", ["wrong_uri", "wrong_digest", "wrong_schema", "noncanonical", "drift"])
def test_hf_training_preflight_reauthenticates_runtime_lock_snapshot(tmp_path: Path, case: str):
    service = TrackingService(tmp_path)
    experiment = _make_training_ready(service)
    runtime_lock = _training_runtime_lock()
    reference = service.snapshot_hf_training_runtime_lock(experiment, runtime_lock)
    snapshot = service._strict_tracking_file(reference["uri"], kind="test runtime lock")

    if case == "wrong_uri":
        wrong = snapshot.parent / "alternate.json"
        wrong.write_bytes(snapshot.read_bytes())
        reference = {"uri": service.tracking_uri(wrong), "sha256": reference["sha256"]}
    elif case == "wrong_digest":
        reference = {**reference, "sha256": "f" * 64}
    elif case == "wrong_schema":
        hostile = {**runtime_lock, "schema_version": "synaptic-hf-training-result/v1"}
        snapshot.write_bytes(canonical_training_bytes(hostile))
        reference = {**reference, "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    elif case == "noncanonical":
        snapshot.write_text(json.dumps(runtime_lock, indent=2), encoding="utf-8")
        reference = {**reference, "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    else:
        hostile = json.loads(canonical_training_bytes(runtime_lock))
        hostile["image"]["provider_reference"] = f"unsloth/unsloth@sha256:{'f' * 64}"
        hostile = seal_training_document(
            {key: value for key, value in hostile.items() if key != "lock_id"}
        )
        hostile_path = snapshot.parent / f"{hostile['lock_id']}.json"
        hostile_path.write_bytes(canonical_training_bytes(hostile))
        reference = {
            "uri": service.tracking_uri(hostile_path),
            "sha256": hashlib.sha256(hostile_path.read_bytes()).hexdigest(),
        }

    root_id = str(ensure_tracking_root_identity(service.base_dir)["root_id"])
    preflight = _training_preflight(experiment, root_id, reference)
    with pytest.raises(ProvenanceIntegrityError):
        service.record_hf_training_preflight(experiment, preflight)


def _training_submission(experiment: Experiment, state: str, previous: tuple[str, str] | None = None) -> dict[str, object]:
    terminal = state != "SUBMITTING"
    return _training_seal(
        {
            "schema_version": "synaptic-hf-training-submission-event/v1",
            "authorization_id": experiment.hf_training_authorization_id,
            "approval": {"uri": experiment.hf_training_approval_uri, "sha256": experiment.hf_training_approval_sha256},
            "experiment_id": experiment.experiment_id, "run_id": experiment.hf_training_run_id, "tracking_root_id": experiment.hf_training_root_id,
            "state": state, "sequence": 2 if terminal else 1, "occurred_at": "2026-08-20T12:04:00Z" if terminal else "2026-08-20T12:03:00Z",
            "previous_event": ({"uri": previous[0], "sha256": previous[1]} if previous else None),
            "provider_job": ({"namespace": "owner", "job_id": "job-1", "created_at": "2026-08-20T12:03:30Z"} if state == "SUBMITTED" else None),
            "reason_code": ("INTERRUPTED_AFTER_CLAIM" if state == "AMBIGUOUS" else "PREFIX_NOT_EMPTY" if state == "NOT_SUBMITTED" else None),
            "provider_effect_possible": state != "NOT_SUBMITTED",
        },
        "event_id",
    )


def _claim_training_process(base_dir: str, experiment_id: str, event: dict, results) -> None:
    service = TrackingService(base_dir)
    experiment = service.load_experiment(experiment_id)
    try:
        outcome = service.claim_hf_training_submission(experiment, event)
        results.put(("ok", outcome.provider_attempt_authorized))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def _cancel_training_process(base_dir: str, experiment_id: str, event: dict, results) -> None:
    service = TrackingService(base_dir)
    experiment = service.load_experiment(experiment_id)
    try:
        outcome = service.claim_hf_training_cancellation(experiment, event)
        results.put(("ok", outcome.provider_attempt_authorized))
    except Exception as exc:
        results.put(("error", type(exc).__name__))


def test_hf_training_projection_is_separate_immutable_and_first_claim_only(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, approval = _make_training_approved(service)
    assert experiment.hf_submission_state is None
    assert experiment.hf_training_submission_state == "APPROVED"
    before = service._experiment_path(experiment.experiment_id).read_bytes()
    hostile = _training_seal({**approval, "publication": True}, "authorization_id")
    with pytest.raises(Exception):
        service.record_hf_training_approval(experiment, hostile)
    assert service._experiment_path(experiment.experiment_id).read_bytes() == before

    event = _training_submission(experiment, "SUBMITTING")
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    outcomes: list[bool] = []

    def claim(caller: Experiment) -> None:
        barrier.wait(timeout=5)
        outcomes.append(TrackingService(tmp_path).claim_hf_training_submission(caller, event).provider_attempt_authorized)

    threads = [threading.Thread(target=claim, args=(caller,)) for caller in callers]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(outcomes) == [False, True]


def test_spawned_hf_training_claim_and_terminal_stale_copy_denial(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, _ = _make_training_approved(service)
    event = _training_submission(experiment, "SUBMITTING")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=_claim_training_process, args=(str(tmp_path), experiment.experiment_id, event, results)) for _ in range(2)]
    for process in processes:
        process.start()
    outcomes = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
    assert sorted(outcomes) == [("ok", False), ("ok", True)]

    durable = service.load_experiment(experiment.experiment_id)
    terminal = _training_submission(durable, "SUBMITTED", (durable.hf_training_submission_event_uri or "", durable.hf_training_submission_event_sha256 or ""))
    service.record_hf_training_submission_terminal(durable, terminal)
    stale = experiment
    with pytest.raises(ProvenanceIntegrityError):
        service.record_hf_training_submission_terminal(stale, terminal)


def test_hf_training_ambiguous_submission_has_one_confirmed_submitted_recovery(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment, _ = _make_training_approved(service)
    service.claim_hf_training_submission(experiment, _training_submission(experiment, "SUBMITTING"))
    stale_submitting = service.load_experiment(experiment.experiment_id)
    ambiguous = _training_submission(
        experiment,
        "AMBIGUOUS",
        (
            experiment.hf_training_submission_event_uri or "",
            experiment.hf_training_submission_event_sha256 or "",
        ),
    )
    service.record_hf_training_submission_terminal(experiment, ambiguous)
    previous_uri = experiment.hf_training_submission_event_uri or ""
    previous_sha = experiment.hf_training_submission_event_sha256 or ""
    recovered = _training_seal(
        {
            "schema_version": "synaptic-hf-training-submission-event/v1",
            "authorization_id": experiment.hf_training_authorization_id,
            "approval": {"uri": experiment.hf_training_approval_uri, "sha256": experiment.hf_training_approval_sha256},
            "experiment_id": experiment.experiment_id,
            "run_id": experiment.hf_training_run_id,
            "tracking_root_id": experiment.hf_training_root_id,
            "state": "SUBMITTED",
            "sequence": 3,
            "occurred_at": "2026-08-20T12:05:00Z",
            "previous_event": {"uri": previous_uri, "sha256": previous_sha},
            "provider_job": {
                "namespace": "owner",
                "job_id": "job-1",
                "created_at": "2026-08-20T12:03:30Z",
            },
            "reason_code": "RECOVERY_CONFIRMED_SUBMITTED",
            "provider_effect_possible": True,
        },
        "event_id",
    )
    with pytest.raises(ProvenanceIntegrityError, match="Stale HF training projection"):
        service.recover_hf_training_submission(stale_submitting, recovered)
    outcome = service.recover_hf_training_submission(experiment, recovered)
    assert outcome.state == "SUBMITTED"
    assert outcome.provider_attempt_authorized is False
    assert experiment.hf_training_submission_state == "SUBMITTED"
    with pytest.raises(ProvenanceIntegrityError, match="AMBIGUOUS"):
        service.recover_hf_training_submission(experiment, recovered)


def _make_training_submitted(service: TrackingService) -> Experiment:
    experiment, _ = _make_training_approved(service)
    service.claim_hf_training_submission(experiment, _training_submission(experiment, "SUBMITTING"))
    terminal = _training_submission(
        experiment,
        "SUBMITTED",
        (experiment.hf_training_submission_event_uri or "", experiment.hf_training_submission_event_sha256 or ""),
    )
    service.record_hf_training_submission_terminal(experiment, terminal)
    return experiment


def _training_common(experiment: Experiment, schema: str) -> dict[str, object]:
    return {
        "schema_version": schema,
        "authorization_id": experiment.hf_training_authorization_id,
        "approval": {"uri": experiment.hf_training_approval_uri, "sha256": experiment.hf_training_approval_sha256},
        "submission": {"uri": experiment.hf_training_submission_event_uri, "sha256": experiment.hf_training_submission_event_sha256},
        "provider_job": {"namespace": "owner", "job_id": "job-1", "created_at": "2026-08-20T12:03:30Z"},
        "experiment_id": experiment.experiment_id,
        "run_id": experiment.hf_training_run_id,
        "tracking_root_id": experiment.hf_training_root_id,
    }


def test_hf_training_cancellation_observation_and_result_recovery(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = _make_training_submitted(service)
    cancellation = _training_seal(
        {
            **_training_common(experiment, "synaptic-hf-training-cancellation-event/v1"),
            "state": "CLAIMED", "sequence": 1, "occurred_at": "2026-08-20T12:30:00Z",
            "previous_event": None, "reason_code": None, "provider_effect_possible": True,
        },
        "event_id",
    )
    callers = [service.load_experiment(experiment.experiment_id) for _ in range(2)]
    barrier = threading.Barrier(3)
    authorities: list[bool] = []

    def cancel(caller: Experiment) -> None:
        barrier.wait(timeout=5)
        authorities.append(
            TrackingService(tmp_path)
            .claim_hf_training_cancellation(caller, cancellation)
            .provider_attempt_authorized
        )

    threads = [threading.Thread(target=cancel, args=(caller,)) for caller in callers]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(authorities) == [False, True]
    experiment = service.load_experiment(experiment.experiment_id)
    ambiguous = _training_seal(
        {
            **cancellation,
            "state": "AMBIGUOUS", "sequence": 2,
            "previous_event": {"uri": experiment.hf_training_cancellation_event_uri, "sha256": experiment.hf_training_cancellation_event_sha256},
            "reason_code": "INTERRUPTED_AFTER_CLAIM",
        },
        "event_id",
    )
    service.record_hf_training_cancellation_terminal(experiment, ambiguous)

    observation_common = {
        **_training_common(experiment, "synaptic-hf-training-observation-event/v1"),
        "status_intervals": [{"status": "RUNNING", "started_at": "2026-08-20T12:03:30Z", "ended_at": "2026-08-20T12:38:30Z"}],
        "hourly_price_usd": "1.00", "estimated_cost_usd": "0.583333",
    }
    stopped = _training_seal(
        {**observation_common, "state": "STOPPED", "terminal": False, "occurred_at": "2026-08-20T12:38:30Z", "previous_event": None, "cost_bounded_completion": False},
        "event_id",
    )
    service.record_hf_training_observation(experiment, stopped)
    wrong_observation_uri = _training_seal(
        {
            **observation_common,
            "state": "COMPLETED", "terminal": True,
            "occurred_at": "2026-08-20T12:40:00Z",
            "previous_event": {
                "uri": "tracking://wrong-observation.json",
                "sha256": experiment.hf_training_observation_event_sha256,
            },
            "cost_bounded_completion": True,
        },
        "event_id",
    )
    with pytest.raises(ProvenanceIntegrityError, match="predecessor reference changed"):
        service.record_hf_training_observation(experiment, wrong_observation_uri)
    completed = _training_seal(
        {**observation_common, "state": "COMPLETED", "terminal": True, "occurred_at": "2026-08-20T12:40:00Z", "previous_event": {"uri": experiment.hf_training_observation_event_uri, "sha256": experiment.hf_training_observation_event_sha256}, "cost_bounded_completion": True},
        "event_id",
    )
    service.record_hf_training_observation(experiment, completed)
    assert experiment.hf_training_observation_state == "COMPLETED"

    result_common = {
        **_training_common(experiment, "synaptic-hf-training-result/v1"),
        "observation": {"uri": experiment.hf_training_observation_event_uri, "sha256": experiment.hf_training_observation_event_sha256},
        "occurred_at": "2026-08-20T12:41:00Z",
        "artifact_prefix": {"bucket_id": "owner/bucket", "base_prefix": "training/artifacts", "slot_id": _training_approval(experiment)["bindings"]["artifact_slot_id"], "prefix": _training_approval(experiment)["bindings"]["artifact_prefix"], "pre_download_inventory_sha256": None, "post_download_inventory_sha256": None, "verified_inventory_sha256": None},
        "inventory": [], "publication": False, "ssh": False, "ports": False, "wandb": False, "job_secrets": [],
    }
    verifying = _training_seal({**result_common, "state": "VERIFYING", "previous_result": None, "optimizer_proof": None, "reason_code": None}, "result_id")
    service.claim_hf_training_verification(experiment, verifying)
    inconclusive = _training_seal(
        {**result_common, "state": "INCONCLUSIVE", "previous_result": {"uri": experiment.hf_training_result_uri, "sha256": experiment.hf_training_result_sha256}, "optimizer_proof": None, "reason_code": "READBACK_INTERRUPTED"},
        "result_id",
    )
    service.record_hf_training_result(experiment, inconclusive)
    wrong_result_uri = _training_seal(
        {
            **result_common,
            "state": "VERIFYING",
            "previous_result": {
                "uri": "tracking://wrong-result.json",
                "sha256": experiment.hf_training_result_sha256,
            },
            "optimizer_proof": None,
            "reason_code": None,
        },
        "result_id",
    )
    with pytest.raises(ProvenanceIntegrityError, match="predecessor reference changed"):
        service.claim_hf_training_verification(experiment, wrong_result_uri)
    reclaim = _training_seal(
        {**result_common, "state": "VERIFYING", "previous_result": {"uri": experiment.hf_training_result_uri, "sha256": experiment.hf_training_result_sha256}, "optimizer_proof": None, "reason_code": None},
        "result_id",
    )
    outcome = service.claim_hf_training_verification(experiment, reclaim)
    assert outcome.state == "VERIFYING"


def test_spawned_hf_training_cancellation_claim_authorizes_only_first(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = _make_training_submitted(service)
    cancellation = _training_seal(
        {
            **_training_common(experiment, "synaptic-hf-training-cancellation-event/v1"),
            "state": "CLAIMED", "sequence": 1, "occurred_at": "2026-08-20T12:30:00Z",
            "previous_event": None, "reason_code": None, "provider_effect_possible": True,
        },
        "event_id",
    )
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_cancel_training_process,
            args=(str(tmp_path), experiment.experiment_id, cancellation, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    outcomes = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
    assert sorted(outcomes) == [("ok", False), ("ok", True)]


def test_spawned_generic_mutations_merge_against_fresh_durable_state(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="process-merge",
        dataset_path="data.jsonl",
        dataset_hash="abc",
        base_model_name="model",
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_set_derived_output_process,
            args=(
                str(tmp_path),
                experiment.experiment_id,
                key,
                f"artifact://{key}",
                start,
                results,
            ),
        )
        for key in ("features_csv", "judge_scores")
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        outcomes = [results.get(timeout=15) for _ in processes]
    except queue.Empty:
        pytest.fail("spawned generic mutation did not report within timeout")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(outcome[0] for outcome in outcomes) == ["ok", "ok"]
    durable = service.load_experiment(experiment.experiment_id)
    assert durable.derived_outputs == {
        "features_csv": "artifact://features_csv",
        "judge_scores": "artifact://judge_scores",
    }
