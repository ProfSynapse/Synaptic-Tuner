"""CPU-only transport qualification: exact SDK reads, evidence and default compatibility."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

import tuner.execution.providers.modal.runtime_release_qualification as qualification
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalCPUTransportProbeV1, ModalRuntimeQualificationHmacAuthenticator,
    build_modal_runtime_release_qualification_dispatch,
    parse_modal_runtime_release_qualification_dispatch,
    transport_probe_root, transport_probe_content, transport_probe_envelope,
    verify_transport_probe_envelope, TRANSPORT_PROBE_SCHEMA,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import (
    ModalCPUTransportCaptureV1, ModalCPUTransportVerificationV1,
    ModalRuntimeReleaseQualificationReader, _transport_evidence_bytes,
)
from tests.execution.providers.test_modal_runtime_release_qualification import Facts, Observer, _case
from tests.execution.providers.test_modal_runtime_release_qualification_io import _facade
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume
from tests.test_modal_volume_root_binding import layout as _volume_layout, _bind


@pytest.fixture
def binding_layout(monkeypatch):
    yield from _volume_layout.__wrapped__(monkeypatch)


@pytest.fixture(autouse=True)
def facts_type(monkeypatch):
    monkeypatch.setattr(qualification, "_qualification_facts_type", lambda: Facts)


def setup_probe(monkeypatch):
    default, facts = _case()
    auth = ModalRuntimeQualificationHmacAuthenticator(b"k" * 32)
    probe = ModalCPUTransportProbeV1("probe-" + "a" * 32 + ".bin", hashlib.sha256(b"m" * 32).hexdigest())
    dispatch = replace(default, transport_probe=probe)
    facade = _facade(facts)
    monkeypatch.setattr(type(facade), "observe_known_call_pending", lambda self, _: ModalFunctionCallState.PENDING)
    reader = ModalRuntimeReleaseQualificationReader(facade=facade, deployment_observer=Observer(facts), verifier=auth)
    monkeypatch.setattr(reader, "observe", lambda *args, **kwargs: SimpleNamespace(receipt=SimpleNamespace(canonical_bytes=b"core-receipt")))
    return default, dispatch, facts, auth, facade, reader


def snapshot(dispatch, auth):
    root = transport_probe_root(dispatch)
    bodies = {root + f"/final/{i:012d}.json": transport_probe_envelope(
        transport_probe_content(dispatch, "fc-probe", i), auth, dispatch.key_ref) for i in range(9)}
    stage = root + "/staging/" + "b" * 32 + ".json"
    body = bodies[root + "/final/000000000000.json"]
    manifest = {**transport_probe_content(dispatch, "fc-probe", 0), "stage_path": stage,
        "final_path": root + "/final/000000000000.json", "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest()}
    staged = {stage: body, root + "/manifests/000000000000.json": transport_probe_envelope(manifest, auth, dispatch.key_ref)}
    return staged, bodies


def terminal(dispatch, auth, capture):
    content = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
        "effect_id": dispatch.effect_id, "dispatch_digest": dispatch.dispatch_digest,
        "runtime_release_digest": dispatch.runtime_release.manifest_digest,
        "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
        "fixture_sha256": dispatch.fixture.sha256, "provider_call_id": "fc-probe",
        "probe": dispatch.transport_probe.to_dict(), "evidence": {
            "schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
            "root": transport_probe_root(dispatch), "records": capture.to_dict()["records"],
            "manual_commit_used": False, "training_executed": False, "gpu_qualified": False},
        "training_executed": False, "gpu_qualified": False}
    path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    FakeVolume.registry["artifact-name"].files[path] = transport_probe_envelope(content, auth, dispatch.key_ref)


def test_optional_signed_dispatch_roundtrip_preserves_absent_bytes(monkeypatch):
    default, dispatch, _, auth, _, _ = setup_probe(monkeypatch)
    legacy = build_modal_runtime_release_qualification_dispatch(default, auth)
    assert "transport_probe" not in default.unsigned_dict()
    assert build_modal_runtime_release_qualification_dispatch(replace(default, transport_probe=None), auth) == legacy
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    assert parse_modal_runtime_release_qualification_dispatch(raw, auth) == dispatch
    assert dispatch.policy.gpu is False and dispatch.policy.timeout_seconds == 120


@pytest.mark.parametrize("field,bad", [("metric_records", True), ("metric_records", 9),
    ("max_leaf_bytes", 8193), ("max_polls", 181), ("max_host_seconds", 181),
    ("marker_name", "probe-" + "A" * 32 + ".bin"), ("marker_sha256", "f" * 63)])
def test_probe_rejects_unreviewed_bounds_and_marker(field, bad):
    probe = ModalCPUTransportProbeV1("probe-" + "a" * 32 + ".bin", "d" * 64)
    with pytest.raises(ValueError):
        replace(probe, **{field: bad})


def test_stage_then_final_pending_proof_requires_durable_exact_capture(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    staged, finals = snapshot(dispatch, auth)
    volume = FakeVolume.registry["control-name"]
    volume.files.update(staged)
    clock = [0]
    def sleep(_):
        clock[0] += 1
        volume.files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert type(capture) is ModalCPUTransportCaptureV1
    assert capture.to_dict()["verdict"] == "observed_pending", capture.to_dict()
    assert capture.to_dict()["polls"][1]["elapsed_ms"] == 1000
    terminal(dispatch, auth, capture)
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=capture._raw)
    assert type(proof) is ModalCPUTransportVerificationV1
    assert proof.to_dict()["status"] == "passed" and proof.to_dict()["durability_acknowledged"] is True
    assert proof.to_dict()["core_receipt_sha256"] == hashlib.sha256(b"core-receipt").hexdigest()
    assert proof.to_dict()["transport_qualified"] is False
    with pytest.raises(ValueError):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture.to_dict())
    with pytest.raises(ValueError):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=b"changed")
    forged = dict(capture.to_dict(), verdict="inconclusive")
    raw = _transport_evidence_bytes(forged)
    with pytest.raises(ValueError):
        replace(capture, _raw=raw, _sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        replace(proof, _raw=b"{}", _sha256=hashlib.sha256(b"{}").hexdigest())


def test_final_seen_before_sealed_stage_is_inconclusive(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    staged, finals = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update({**staged, **finals})
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["staged_final_absent"] is False
    assert capture.to_dict()["verdict"] == "inconclusive"
    terminal(dispatch, auth, capture)
    assert reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture).to_dict()["status"] == "inconclusive"


def test_unknown_after_read_cannot_accept_pending_points(monkeypatch):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    staged, _ = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update(staged)
    states = iter((ModalFunctionCallState.PENDING, ModalFunctionCallState.UNKNOWN))
    monkeypatch.setattr(type(facade), "observe_known_call_pending", lambda self, _: next(states))
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["records"] == [] and capture.to_dict()["staged_final_absent"] is False
    assert capture.to_dict()["polls"][0]["accepted"] is False


def test_large_ledger_uses_evidence_budget_not_foundation_16k(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    _, finals = snapshot(dispatch, auth)
    finals.pop(transport_probe_root(dispatch) + "/final/000000000000.json")
    FakeVolume.registry["control-name"].files.update(finals)
    clock = [0]
    def sleep(_): clock[0] += 1
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert 16384 < len(capture._raw) < 1048576
    terminal(dispatch, auth, capture)
    assert reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture).to_dict()["status"] == "inconclusive"
    with pytest.raises(ValueError):
        _transport_evidence_bytes({"oversized": "x" * 1048576})


def test_strict_tag_and_host_deadline(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    raw = canonical_bytes({"content": {}, "tag_hex": "AA"})
    with pytest.raises(ValueError):
        verify_transport_probe_envelope(raw, auth, dispatch.key_ref)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: 10, host_deadline=9)
    assert capture.to_dict()["polls"] == [] and capture.to_dict()["verdict"] == "inconclusive"
    with pytest.raises(ValueError):
        reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", host_deadline=float("inf"))


def test_complete_ledger_over_16k_remains_admitted(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    staged, finals = snapshot(dispatch, auth)
    clock = [0]
    volume = FakeVolume.registry["control-name"]
    def sleep(_):
        clock[0] += 1
        if clock[0] == 130: volume.files.update(staged)
        if clock[0] == 131: volume.files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert len(capture._raw) > 16384 and capture.to_dict()["verdict"] == "observed_pending"
    terminal(dispatch, auth, capture)
    assert reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture,
        retained_capture_bytes=capture._raw).to_dict()["status"] == "passed"


def test_transient_sdk_read_retry_preserves_window_and_same_call(monkeypatch):
    from tuner.execution.providers.modal.facade import ModalFacadeError
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    staged, finals = snapshot(dispatch, auth)
    volume = FakeVolume.registry["control-name"]
    volume.files.update(staged)
    original = type(facade).read_complete
    first = [True]
    def read(self, *args, **kwargs):
        if first[0]:
            first[0] = False
            raise ModalFacadeError("modal_volume_read_failed")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(facade), "read_complete", read)
    clock = [0]
    def sleep(_):
        clock[0] += 1
        if clock[0] == 2: volume.files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert capture.to_dict()["verdict"] == "observed_pending"
    assert capture.to_dict()["polls"][0]["status"] == "read_unavailable"
    assert capture.to_dict()["polls"][0]["accepted"] is False


def test_duplicate_or_cross_namespace_inventory_is_invalid(monkeypatch):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    staged, _ = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update(staged)
    original = type(facade).list_prefix
    def duplicate(self, volume, prefix, **kwargs):
        result = original(self, volume, prefix, **kwargs)
        return result + result if prefix.endswith("/manifests") else result
    monkeypatch.setattr(type(facade), "list_prefix", duplicate)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["verdict"] == "evidence_invalid"


def test_signed_receipt_wrong_scope_or_missing_core_cannot_prove_transport(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: 10, host_deadline=9)
    terminal(dispatch, auth, capture)
    path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    volume = FakeVolume.registry["artifact-name"]
    content = verify_transport_probe_envelope(volume.files[path], auth, dispatch.key_ref)
    content["provider_call_id"] = "fc-other"
    volume.files[path] = transport_probe_envelope(content, auth, dispatch.key_ref)
    with pytest.raises(ValueError):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)
    terminal(dispatch, auth, capture)
    def missing(*args, **kwargs): raise ValueError("CPU receipt unavailable")
    monkeypatch.setattr(reader, "observe", missing)
    with pytest.raises(ValueError):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)


def test_marker_stage_is_create_only_exact_and_collision_preserving(monkeypatch):
    from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOperator
    from tests.execution.providers.test_modal_runtime_release_qualification_io import Catalog
    _, dispatch, facts, auth, facade, _ = setup_probe(monkeypatch)
    operator = ModalRuntimeQualificationOperator(facade=facade, deployment_observer=Observer(facts),
        verifier=auth, call_catalog=Catalog())
    operator.stage_transport_marker(probe=dispatch.transport_probe, marker_bytes=b"m" * 32, deployment_facts=facts)
    assert FakeVolume.registry["control-name"].files[dispatch.transport_probe.marker_name] == b"m" * 32
    with pytest.raises(ValueError):
        operator.stage_transport_marker(probe=dispatch.transport_probe, marker_bytes=b"m" * 32, deployment_facts=facts)
    with pytest.raises(ValueError):
        operator.stage_transport_marker(probe=dispatch.transport_probe, marker_bytes=b"z" * 32, deployment_facts=facts)


@pytest.mark.parametrize("optional_failure", ["probe", "atomic", "receipt"])
def test_original_cpu_proof_survives_optional_probe_failure(monkeypatch, tmp_path, optional_failure):
    from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA, local_cpu_result
    from tuner.execution.providers.modal.runtime_release_qualification import (
        ModalRuntimeReleaseQualificationRoots, ModalRuntimeReleaseQualificationWorker,
    )
    from tuner.execution.providers.modal.volume_root_binding import VolumeRootBinding
    _, dispatch, facts, auth, _, _ = setup_probe(monkeypatch)
    roots = ModalRuntimeReleaseQualificationRoots(tmp_path / "control", tmp_path / "artifacts")
    roots.control.mkdir(); roots.artifacts.mkdir()
    fixture = roots.artifacts / dispatch.fixture.path
    fixture.parent.mkdir(parents=True); fixture.write_bytes(LOCAL_CPU_DATA)
    monkeypatch.setattr("tuner.runtime.packaged_sft_execution._inspect_parent_release", lambda _: None)
    monkeypatch.setattr("tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda _: local_cpu_result(dispatch.runtime_release, "a" * 64))
    monkeypatch.setattr(VolumeRootBinding, "bind", lambda **kwargs: SimpleNamespace(close=lambda: None))
    def failure(*args, **kwargs):
        if optional_failure == "atomic":
            from tuner.execution.providers.modal.volume_root_binding import VolumeRootBindingError
            raise VolumeRootBindingError("PUBLISH_RENAME_UNSUPPORTED")
        raise RuntimeError("private failure not projected")
    signed_evidence = []
    if optional_failure in {"probe", "atomic"}:
        monkeypatch.setattr(qualification, "_run_transport_probe", failure)
        monkeypatch.setattr(qualification, "_publish_transport_receipt",
            lambda *args, **kwargs: signed_evidence.append(kwargs["evidence"]))
    else:
        monkeypatch.setattr(qualification, "_run_transport_probe", lambda *args, **kwargs: {})
        monkeypatch.setattr(qualification, "_publish_transport_receipt", failure)
    worker = ModalRuntimeReleaseQualificationWorker(expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-probe", roots=roots)
    result = worker(build_modal_runtime_release_qualification_dispatch(dispatch, auth),
        commit_artifacts=lambda: None, commit_control=lambda: None)
    assert result["status_code"] == "completed"
    assert (roots.control / operation_path(dispatch.effect_id, "runtime-release-qualification", "receipt", "receipt.json")).is_file()
    assert (roots.artifacts / operation_path(dispatch.effect_id, "runtime-release-qualification", "output", "evidence.json")).is_file()
    assert "private failure" not in str(result)
    if signed_evidence:
        assert signed_evidence == [{"status": "inconclusive", "failure_code":
            "PUBLISH_RENAME_UNSUPPORTED" if optional_failure == "atomic" else "TRANSPORT_PROBE_UNAVAILABLE"}]


@pytest.mark.parametrize("code", ["PUBLISH_RENAME_UNSUPPORTED", "PUBLISH_RENAME_COLLISION",
    "TRANSPORT_PROBE_UNAVAILABLE", "untrusted exception text"])
def test_signed_optional_failure_keeps_only_closed_diagnostic(monkeypatch, code):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    staged, finals = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update({**staged, **finals})
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    terminal(dispatch, auth, capture)
    path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    volume = FakeVolume.registry["artifact-name"]
    content = verify_transport_probe_envelope(volume.files[path], auth, dispatch.key_ref)
    content["evidence"] = {"status": "inconclusive", "failure_code": code}
    volume.files[path] = transport_probe_envelope(content, auth, dispatch.key_ref)
    if code == "untrusted exception text":
        with pytest.raises(ValueError):
            reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)
    else:
        proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)
        assert proof.to_dict()["status"] == "inconclusive"
        assert proof.to_dict()["failure_code"] == code


def test_real_atomic_probe_uses_sealed_stage_and_distinct_receipt_volume(monkeypatch, binding_layout, tmp_path):
    _, dispatch, _, auth, _, _ = setup_probe(monkeypatch)
    _, _, target, _ = binding_layout
    events = []
    def sleep(seconds):
        events.append(seconds)
        if len(events) == 1 and seconds == dispatch.transport_probe.hold_stage_seconds:
            root = target / transport_probe_root(dispatch)
            stage_files = list((root / "staging").iterdir())
            assert len(stage_files) == 1
            assert not (root / "final/000000000000.json").exists()
            assert verify_transport_probe_envelope(stage_files[0].read_bytes(), auth, dispatch.key_ref) == transport_probe_content(dispatch, "fc-probe", 0)
    monkeypatch.setattr(qualification.time, "sleep", sleep)
    with _bind(binding_layout) as binding:
        # Mandatory core CPU receipt already creates these parents in production.
        binding.claim_directory("operations")
        binding.claim_directory("operations/" + dispatch.effect_id)
        evidence = qualification._run_transport_probe(dispatch, binding=binding, signer=auth,
            call_id="fc-probe", deadline=qualification.time.monotonic() + 100)
    root = target / transport_probe_root(dispatch)
    assert len(list((root / "final").iterdir())) == 9
    assert len(list((root / "manifests").iterdir())) == 9
    assert not list((root / "staging").iterdir())
    assert evidence["manual_commit_used"] is False
    assert events == [20, *([2] * 9), 20]
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    qualification._publish_transport_receipt(dispatch,
        roots=SimpleNamespace(control=binding_layout[0], artifacts=artifact_root),
        signer=auth, call_id="fc-probe", evidence=evidence)
    receipt_path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    assert (artifact_root / receipt_path).is_file()
    assert not (target / receipt_path).exists()


def test_short_effective_host_deadline_rejects_late_pending_read(monkeypatch):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    staged, _ = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update(staged)
    clock, calls = [0], [0]
    def observe(self, call):
        calls[0] += 1
        if calls[0] == 2: clock[0] = 2
        return ModalFunctionCallState.PENDING
    monkeypatch.setattr(type(facade), "observe_known_call_pending", observe)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe",
        monotonic=lambda: clock[0], host_deadline=1, sleep=lambda _: None)
    assert capture.to_dict()["staged_final_absent"] is False
    assert capture.to_dict()["polls"][0]["status"] == "deadline"
    assert capture.to_dict()["polls"][0]["accepted"] is False


def test_primary_marker_admission_failure_never_publishes_core_proof(monkeypatch, tmp_path):
    from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA
    from tuner.execution.providers.modal.runtime_release_qualification import (
        ModalRuntimeReleaseQualificationRoots, ModalRuntimeReleaseQualificationWorker,
    )
    from tuner.execution.providers.modal.volume_root_binding import VolumeRootBinding
    _, dispatch, facts, auth, _, _ = setup_probe(monkeypatch)
    roots = ModalRuntimeReleaseQualificationRoots(tmp_path / "control", tmp_path / "artifacts")
    roots.control.mkdir(); roots.artifacts.mkdir()
    fixture = roots.artifacts / dispatch.fixture.path
    fixture.parent.mkdir(parents=True); fixture.write_bytes(LOCAL_CPU_DATA)
    events = []
    def reject(**kwargs): raise ValueError("root marker mismatch")
    monkeypatch.setattr(VolumeRootBinding, "bind", reject)
    monkeypatch.setattr("tuner.runtime.packaged_training_worker.qualify_installed_child", lambda _: events.append("child"))
    worker = ModalRuntimeReleaseQualificationWorker(expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-probe", roots=roots)
    result = worker(build_modal_runtime_release_qualification_dispatch(dispatch, auth),
        commit_artifacts=lambda: events.append("commit"), commit_control=lambda: events.append("commit"))
    assert result["status_code"] == "failed" and events == []
    assert not (roots.control / operation_path(dispatch.effect_id, "runtime-release-qualification", "receipt", "receipt.json")).exists()


def test_verifier_rejects_negative_first_elapsed_from_faulty_internal_issuer(monkeypatch):
    import tuner.execution.providers.modal.runtime_release_qualification_reader as reader_module
    _, dispatch, _, _, _, reader = setup_probe(monkeypatch)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: 10, host_deadline=9)
    document = capture.to_dict()
    document["polls"] = [{"poll": 0, "elapsed_ms": -1, "before": "pending", "after": "pending",
                          "status": "empty", "accepted": True, "observations": []}]
    raw = _transport_evidence_bytes(document)
    digest = hashlib.sha256(raw).hexdigest()
    # Explicit trusted-issuer fault fixture, not a public proof reconstruction route.
    issued = ModalCPUTransportCaptureV1(raw, digest,
        reader_module._TransportWitness(raw, digest, reader_module._CAPTURE_AUTHORITY))
    with pytest.raises(ValueError, match="retained poll invalid"):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=issued)
