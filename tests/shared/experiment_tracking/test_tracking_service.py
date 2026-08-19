from __future__ import annotations

import json
import hashlib
import multiprocessing
import queue
import threading
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
from shared.experiment_tracking.schema import RunRecord
from tuner.project import (
    ConfigDocument,
    GitSource,
    ProjectContext,
    RepositoryLocation,
    SourceLock,
    resolve_config_layers,
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
