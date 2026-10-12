"""CPU SDK publication proof: complete pending reads, collision and terminal inventory."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

import tuner.execution.providers.modal.runtime_release_qualification as qualification
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal.facade import ModalFunctionCallState, ModalFacadeError
from tuner.execution.providers.modal.packaged_progress import ModalSDKProgressPublication, SDK_PUBLICATION_DIAGNOSTICS
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalCPUTransportProbeV1, ModalRuntimeQualificationHmacAuthenticator,
    build_modal_runtime_release_qualification_dispatch, parse_modal_runtime_release_qualification_dispatch,
    transport_probe_root, transport_probe_content, transport_probe_envelope,
    TRANSPORT_PROBE_SCHEMA, QUALIFICATION_RESULT_SCHEMA,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import (
    ModalCPUTransportCaptureV1, ModalCPUTransportVerificationV1,
    ModalRuntimeReleaseQualificationReader, _transport_evidence_bytes,
)
from tests.execution.providers.test_modal_runtime_release_qualification import Facts, Observer, _case
from tests.execution.providers.test_modal_runtime_release_qualification_io import _facade
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, FakeCall, FakeFunctionCall
from tests.execution.providers.test_modal_sdk_progress_publication import bound as _bound_fixture, MemoryVolume


@pytest.fixture
def bound(monkeypatch):
    yield from _bound_fixture.__wrapped__(monkeypatch)


@pytest.fixture(autouse=True)
def facts_type(monkeypatch):
    monkeypatch.setattr(qualification, "_qualification_facts_type", lambda: Facts)
    monkeypatch.setattr(FakeCall, "result", {"schema_version": QUALIFICATION_RESULT_SCHEMA, "status_code": "completed"})
    for name, value in (("hydrate_error", None), ("hydrated_id", None),
                        ("hydrate_result_other", False), ("remain_unhydrated", False)):
        monkeypatch.setattr(FakeFunctionCall, name, value)


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
    return {root + f"/final/{i:012d}.json": transport_probe_envelope(
        transport_probe_content(dispatch, "fc-probe", i), auth, dispatch.key_ref) for i in range(9)}


def evidence(dispatch, finals):
    root = transport_probe_root(dispatch)
    records = [{"sequence": i, "path": path, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
               for i, (path, raw) in enumerate(finals.items())]
    return {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
        "publication_strategy": dispatch.transport_probe.publication_strategy, "root": root, "records": records,
        "collision": {"attempts": 1, "rejected": True, "path": root + "/final/000000000000.json",
                      "original_sha256": records[0]["sha256"]}, "training_executed": False, "gpu_qualified": False}


def terminal(dispatch, auth, finals, outcome=None):
    content = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
        "effect_id": dispatch.effect_id, "dispatch_digest": dispatch.dispatch_digest,
        "runtime_release_digest": dispatch.runtime_release.manifest_digest,
        "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
        "fixture_sha256": dispatch.fixture.sha256, "provider_call_id": "fc-probe",
        "probe": dispatch.transport_probe.to_dict(), "evidence": evidence(dispatch, finals) if outcome is None else outcome,
        "training_executed": False, "gpu_qualified": False}
    path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    FakeVolume.registry["control-name"].files[path] = transport_probe_envelope(content, auth, dispatch.key_ref)
    return path


def complete_capture(monkeypatch):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    finals = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["verdict"] == "observed_pending"
    terminal(dispatch, auth, finals)
    return dispatch, auth, facade, reader, finals, capture


def test_optional_sdk_strategy_dispatch_preserves_absent_bytes(monkeypatch):
    default, dispatch, _, auth, _, _ = setup_probe(monkeypatch)
    legacy = build_modal_runtime_release_qualification_dispatch(default, auth)
    assert "transport_probe" not in default.unsigned_dict()
    assert build_modal_runtime_release_qualification_dispatch(replace(default, transport_probe=None), auth) == legacy
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    assert parse_modal_runtime_release_qualification_dispatch(raw, auth) == dispatch
    assert dispatch.transport_probe.publication_strategy == "modal_sdk_create_only/v1"
    assert dispatch.transport_probe.collision_attempts == 1
    assert dispatch.policy.gpu is False and dispatch.policy.timeout_seconds == 120


@pytest.mark.parametrize("field,bad", [("metric_records", True), ("metric_records", 9), ("max_leaf_bytes", 8193),
    ("max_polls", 181), ("max_host_seconds", 181), ("collision_attempts", 2),
    ("publication_strategy", "rename_fallback"), ("terminal_receipt_volume_role", "artifacts"),
    ("marker_name", "probe-" + "A" * 32 + ".bin")])
def test_probe_rejects_unreviewed_strategy_bounds_or_marker(field, bad):
    probe = ModalCPUTransportProbeV1("probe-" + "a" * 32 + ".bin", "d" * 64)
    with pytest.raises(ValueError): replace(probe, **{field: bad})


def test_complete_pending_records_terminal_inventory_and_durable_sealed_proof(monkeypatch):
    dispatch, _, _, reader, _, capture = complete_capture(monkeypatch)
    assert type(capture) is ModalCPUTransportCaptureV1
    assert all(row["terminal_absent_before"] and row["terminal_absent_after"] for row in capture.to_dict()["polls"])
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=capture._raw)
    assert type(proof) is ModalCPUTransportVerificationV1
    document = proof.to_dict()
    assert document["status"] == "passed" and document["durability_acknowledged"] is True
    assert document["call_completed"] and document["final_inventory_verified"] and document["collision_verified"]
    assert document["transport_qualified"] is False
    with pytest.raises(ValueError): reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture.to_dict())
    with pytest.raises(ValueError): reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=b"changed")
    raw = _transport_evidence_bytes(dict(capture.to_dict(), verdict="inconclusive"))
    with pytest.raises(ValueError): replace(capture, _raw=raw, _sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError): replace(proof, _raw=b"{}", _sha256=hashlib.sha256(b"{}").hexdigest())


@pytest.mark.parametrize("when", ["before", "during"])
def test_terminal_receipt_visibility_rejects_pending_page(monkeypatch, when):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    finals = snapshot(dispatch, auth)
    FakeVolume.registry["control-name"].files.update(finals)
    if when == "before": terminal(dispatch, auth, finals)
    else:
        original = type(facade).read_complete
        def read(self, volume, path, **kwargs):
            raw = original(self, volume, path, **kwargs)
            if path.endswith("/final/000000000008.json"): terminal(dispatch, auth, finals)
            return raw
        monkeypatch.setattr(type(facade), "read_complete", read)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["records"] == [] and capture.to_dict()["verdict"] == "inconclusive"
    assert capture.to_dict()["polls"][0]["accepted"] is False
    assert reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture).to_dict()["status"] == "inconclusive"


@pytest.mark.parametrize("fault", ["pending", "other_call", "missing", "extra", "changed", "collision_unverified"])
def test_terminal_call_and_exact_inventory_fail_closed(monkeypatch, fault):
    dispatch, auth, _, reader, finals, capture = complete_capture(monkeypatch)
    control = FakeVolume.registry["control-name"]
    if fault == "pending": monkeypatch.setattr(FakeCall, "result", TimeoutError())
    elif fault == "other_call": monkeypatch.setattr(FakeFunctionCall, "hydrated_id", "fc-other")
    elif fault == "missing": control.files.pop(next(iter(finals)))
    elif fault == "extra": control.files[transport_probe_root(dispatch) + "/final/extra.json"] = b"{}"
    elif fault == "changed": control.files[next(iter(finals))] = transport_probe_envelope(
        {**transport_probe_content(dispatch, "fc-probe", 0), "value": 99}, auth, dispatch.key_ref)
    else:
        altered = evidence(dispatch, finals)
        altered["collision"]["rejected"] = False
        terminal(dispatch, auth, finals, altered)
    with pytest.raises((ValueError, TimeoutError)):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)


def test_complete_evidence_over16k_admitted_and_1mib_rejected(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    finals, clock = snapshot(dispatch, auth), [0]
    def sleep(_):
        clock[0] += 1
        if clock[0] == 130: FakeVolume.registry["control-name"].files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert len(capture._raw) > 16384 and capture.to_dict()["verdict"] == "observed_pending"
    terminal(dispatch, auth, finals)
    assert reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture).to_dict()["status"] == "passed"
    with pytest.raises(ValueError): _transport_evidence_bytes({"oversized": "x" * 1048576})


def test_transient_read_unknown_after_and_effective_deadline(monkeypatch):
    _, dispatch, _, auth, facade, reader = setup_probe(monkeypatch)
    finals, clock = snapshot(dispatch, auth), [0]
    FakeVolume.registry["control-name"].files.update(finals)
    original, first = type(facade).read_complete, [True]
    def read(self, *args, **kwargs):
        if first[0]:
            first[0] = False
            raise ModalFacadeError("modal_volume_read_failed")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(facade), "read_complete", read)
    def sleep(_): clock[0] += 1
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    assert capture.to_dict()["verdict"] == "observed_pending"
    assert capture.to_dict()["polls"][0]["status"] == "read_unavailable"
    states = iter((ModalFunctionCallState.PENDING, ModalFunctionCallState.UNKNOWN))
    monkeypatch.setattr(type(facade), "observe_known_call_pending", lambda self, _: next(states))
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["records"] == []
    def late(self, call):
        clock[0] += 1
        return ModalFunctionCallState.PENDING
    monkeypatch.setattr(type(facade), "observe_known_call_pending", late)
    clock[0] = 0
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], host_deadline=1.5)
    assert capture.to_dict()["polls"][0]["status"] == "deadline" and capture.to_dict()["records"] == []


def test_actual_shared_sdk_adapter_one_collision_and_original_preserved(monkeypatch, bound):
    _, dispatch, _, auth, _, _ = setup_probe(monkeypatch)
    binding, _ = bound
    client, factories = object(), []
    volume = MemoryVolume(binding, client)
    def factory(guard, prefix, deadline):
        publisher = ModalSDKProgressPublication(binding=guard, volume=volume, client=client, prefix=prefix, deadline=deadline)
        factories.append(publisher)
        return publisher
    def no_remote_read(*args, **kwargs):
        raise AssertionError("CPU publication must not perform SDK readback")
    monkeypatch.setattr(ModalSDKProgressPublication, "read_final", no_remote_read, raising=False)
    monkeypatch.setattr(qualification.time, "sleep", lambda _: None)
    result = qualification._run_transport_probe(dispatch, binding=binding, signer=auth,
        call_id="fc-probe", deadline=qualification.time.monotonic() + 100, publication_factory=factory)
    assert len(factories) == 2 and len(volume.calls) == 10 and len(volume.files) == 9
    assert result["collision"]["attempts"] == 1 and result["collision"]["rejected"] is True
    assert volume.files[result["collision"]["path"]] == snapshot(dispatch, auth)[result["collision"]["path"]]
    assert set(volume.modes.values()) == {0o600}


@pytest.mark.parametrize("code", sorted(SDK_PUBLICATION_DIAGNOSTICS))
def test_only_closed_signed_sdk_failure_diagnostics_are_retained(monkeypatch, code):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: 10, host_deadline=9)
    terminal(dispatch, auth, snapshot(dispatch, auth), {"status": "inconclusive", "failure_code": code})
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)
    assert proof.to_dict()["status"] == "inconclusive" and proof.to_dict()["failure_code"] == code


@pytest.mark.parametrize("elapsed", [-1, -2])
def test_negative_and_regressing_elapsed_reject(monkeypatch, elapsed):
    dispatch, _, _, reader, _, capture = complete_capture(monkeypatch)
    document = capture.to_dict()
    document["polls"][0]["elapsed_ms"] = elapsed
    # A changed capture cannot retain the reader's immutable witness.
    raw = _transport_evidence_bytes(document)
    with pytest.raises(ValueError):
        replace(capture, _raw=raw, _sha256=hashlib.sha256(raw).hexdigest())


def test_reader_issued_regressing_elapsed_rejected_by_verifier(monkeypatch):
    _, dispatch, _, auth, _, reader = setup_probe(monkeypatch)
    finals, clock, sleeps = snapshot(dispatch, auth), [0], [0]
    def sleep(_):
        sleeps[0] += 1
        clock[0] = 10 if sleeps[0] == 1 else 5
        if sleeps[0] == 2:
            FakeVolume.registry["control-name"].files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe",
        monotonic=lambda: clock[0], sleep=sleep)
    assert [row["elapsed_ms"] for row in capture.to_dict()["polls"]] == [0, 10000, 5000]
    terminal(dispatch, auth, finals)
    with pytest.raises(ValueError):
        reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture,
            retained_capture_bytes=capture._raw)
