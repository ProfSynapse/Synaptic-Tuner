"""Independent exact-proof gate and private retention regressions; no provider effects."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade, ModalFunctionCallState
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalCPUTransportProbeV1, ModalRuntimeQualificationHmacAuthenticator,
    ModalRuntimeReleaseFixtureReceiptV1, ModalRuntimeReleaseQualificationDispatchV1,
    ModalRuntimeReleaseQualificationPolicyV1, ModalRuntimeReleaseQualificationRoots,
    ModalRuntimeReleaseQualificationWorker, QUALIFICATION_HMAC_KEY_REF,
    TRANSPORT_PROBE_SCHEMA, build_modal_runtime_release_qualification_dispatch,
    transport_probe_content, transport_probe_envelope, transport_probe_root,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import ModalRuntimeReleaseQualificationReader
from tuner.execution.providers.modal.volume_root_binding import VolumeRootBinding
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA, local_cpu_result
from tuner.runtime.releases import ProviderRuntimeBindingV1
from tuner.training.modal_local_preparation import ModalLocalPreparationUnavailable
from tuner.training.modal_host_composition import _transport_logs_authorized
from tuner.training.modal_host_effects import ModalPackagedHostEffectsV1
from tuner.training.modal_host_qualification import ModalHostCPUQualificationV1, _retain_transport_capture
from tests.execution.providers.test_modal_sdk154_adapter import FakeVolume, FakeCall, FakeFunctionCall, SDK
from tests.training.test_modal_host_effects import _Storage
from tests.training.test_modal_host_transport_qualification import bound_runtime


@pytest.fixture
def genuine_proof(monkeypatch, tmp_path):
    """Real V2 facts plus worker-signed receipts and reader-issued sealed proof."""
    runtime, _ = bound_runtime(monkeypatch)
    facts, release, binding = runtime.deployment_facts, runtime.release, runtime.facts.client_binding
    auth = ModalRuntimeQualificationHmacAuthenticator(b"q" * 32)
    probe = ModalCPUTransportProbeV1("probe-" + "a" * 32 + ".bin", hashlib.sha256(b"m" * 32).hexdigest())
    volumes = {item.spec.role: item for item in facts.volumes}
    fixture = ModalRuntimeReleaseFixtureReceiptV1.create(effect_id="cpu-proof-gate",
        artifact_volume_id=volumes["artifacts"].volume_id)
    provider = ProviderRuntimeBindingV1.build(provider_ref="modal", runtime_release=release,
        provider_facts_schema=facts.schema_version, provider_facts_digest=facts.facts_digest)
    dispatch = ModalRuntimeReleaseQualificationDispatchV1("cpu-proof-gate", release, provider, facts,
        fixture, ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF, probe)
    observer = SimpleNamespace(observe=lambda expected: expected)
    root = transport_probe_root(dispatch)
    finals = {root + f"/final/{i:012d}.json": transport_probe_envelope(
        transport_probe_content(dispatch, "fc-probe", i), auth, dispatch.key_ref) for i in range(9)}
    evidence = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
        "publication_strategy": probe.publication_strategy, "root": root,
        "records": [{"sequence": i, "path": path, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for i, (path, raw) in enumerate(finals.items())],
        "collision": {"attempts": 1, "rejected": True, "path": root + "/final/000000000000.json",
            "original_sha256": hashlib.sha256(next(iter(finals.values()))).hexdigest()},
        "training_executed": False, "gpu_qualified": False}
    # ML/runtime admission and the transport primitive have their own tests.
    # These provider-free collaborators let the real worker produce its two signed receipts.
    monkeypatch.setattr("tuner.runtime.packaged_sft_execution._inspect_parent_release", lambda _: None)
    monkeypatch.setattr("tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda _: local_cpu_result(release, "a" * 64))
    class Guard:
        def close(self): pass
        def claim_directory(self, path): (roots.control / path).mkdir(mode=0o700)
        def write_exclusive(self, path, raw, maximum):
            assert len(raw) <= maximum
            with (roots.control / path).open("xb") as stream: stream.write(raw)
    monkeypatch.setattr(VolumeRootBinding, "bind", lambda **kwargs: Guard())
    monkeypatch.setattr("tuner.execution.providers.modal.runtime_release_qualification._run_transport_probe",
        lambda *args, **kwargs: evidence)
    roots = ModalRuntimeReleaseQualificationRoots(tmp_path / "control", tmp_path / "artifacts")
    roots.control.mkdir(); roots.artifacts.mkdir()
    source = roots.artifacts / fixture.path
    source.parent.mkdir(parents=True); source.write_bytes(LOCAL_CPU_DATA)
    worker = ModalRuntimeReleaseQualificationWorker(expected_facts=facts, verifier=auth, signer=auth,
        observer=observer, call_id_provider=lambda: "fc-probe", roots=roots)
    commits = []
    def commit_control():
        commits.append(tuple(path.relative_to(roots.control).as_posix()
            for path in roots.control.rglob("*") if path.is_file()))
    assert worker(build_modal_runtime_release_qualification_dispatch(dispatch, auth),
        commit_artifacts=lambda: None, commit_control=commit_control)["status_code"] == "completed"
    terminal_name = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    assert len(commits) == 2
    assert terminal_name not in commits[0] and terminal_name in commits[1]
    assert not tuple(roots.artifacts.rglob("*transport*"))
    FakeVolume.registry = {item.spec.name: FakeVolume(item.volume_id) for item in facts.volumes}
    for role, local_root in (("control", roots.control), ("artifacts", roots.artifacts)):
        for file in local_root.rglob("*"):
            if file.is_file():
                FakeVolume.registry[volumes[role].spec.name].files[file.relative_to(local_root).as_posix()] = file.read_bytes()
    facade = ExplicitModal154ReadFacade(binding, sdk=SDK, client=object(),
        scope_observer=lambda _: (binding.account_ref, binding.workspace_ref, binding.environment_ref, binding.client_ref),
        deployment_observer=lambda **kwargs: facts,
        volume_names=dict(runtime.volume_names_by_id))
    monkeypatch.setattr(type(facade), "observe_known_call_pending", lambda self, _: ModalFunctionCallState.PENDING)
    reader = ModalRuntimeReleaseQualificationReader(facade=facade, deployment_observer=observer, verifier=auth)
    control = FakeVolume.registry[volumes["control"].spec.name]
    terminal_path = operation_path(dispatch.effect_id, "runtime-release-transport-evidence", "receipt.json")
    terminal_raw = control.files.pop(terminal_path)
    from tuner.execution.providers.modal.runtime_release_qualification import QUALIFICATION_RESULT_SCHEMA
    monkeypatch.setattr(FakeCall, "result", {"schema_version": QUALIFICATION_RESULT_SCHEMA, "status_code": "completed"})
    for name, value in (("hydrate_error", None), ("hydrated_id", None), ("hydrate_result_other", False), ("remain_unhydrated", False)):
        monkeypatch.setattr(FakeFunctionCall, name, value)
    clock = [0]
    def sleep(_):
        clock[0] += 1
        control.files.update(finals)
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", monotonic=lambda: clock[0], sleep=sleep)
    control.files[terminal_path] = terminal_raw
    path, digest, raw = _retain_transport_capture(tmp_path / "good-retention", dispatch.effect_id, "fc-probe", capture)
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=raw)
    core = reader.observe(dispatch, provider_call_id="fc-probe")
    qualification = ModalHostCPUQualificationV1(core.receipt, core.receipt.output_sha256, "fc-probe",
        release.manifest_digest, facts.facts_digest, deployment_facts=facts, transport_probe=probe,
        transport_verification=proof, transport_capture_path=path, transport_capture_sha256=digest, transport_status="passed")
    effects = ModalPackagedHostEffectsV1(storage=_Storage(), runtime_release=release, provider_binding=provider,
        provider_facts=runtime.facts, execution_binding=object(), retained_source=object(), workload_bytes=b"fixture",
        artifact_policy=object(), signer=auth, key_ref=dispatch.key_ref, maximum_cost_minor_units=500)
    components = SimpleNamespace(execution_source=SimpleNamespace(runtime_release_digest=release.manifest_digest))
    return qualification, effects, components, reader, dispatch, capture


def test_exact_reader_issued_v2_proof_passes_gate_and_retention_is_private(genuine_proof):
    qualification, effects, components, _, _, capture = genuine_proof
    assert _transport_logs_authorized(qualification, effects, components) is True
    path = Path(qualification.transport_capture_path)
    assert path.read_bytes() == capture._raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert qualification.transport_verification.to_dict()["durability_acknowledged"] is True


@pytest.mark.parametrize("fault", ["dict", "bool", "call", "core", "capture", "release", "facts", "source", "probe"])
def test_gate_rejects_unsealed_or_mismatched_proof_binding(genuine_proof, fault):
    qualification, effects, components, _, _, _ = genuine_proof
    if fault == "dict": qualification = replace(qualification, transport_verification=qualification.transport_verification.to_dict())
    elif fault == "bool": qualification = replace(qualification, transport_verification=True)
    elif fault == "call": qualification = replace(qualification, provider_call_id="fc-other")
    elif fault == "core": qualification = replace(qualification, receipt=replace(qualification.receipt, output_sha256="f" * 64))
    elif fault == "capture": qualification = replace(qualification, transport_capture_sha256="f" * 64)
    elif fault == "release": qualification = replace(qualification, runtime_release_digest="f" * 64)
    elif fault == "facts": qualification = replace(qualification, deployment_facts_digest="f" * 64)
    elif fault == "source": components = SimpleNamespace(execution_source=SimpleNamespace(runtime_release_digest="f" * 64))
    elif fault == "probe": qualification = replace(qualification, transport_probe=replace(qualification.transport_probe, marker_sha256="f" * 64))
    with pytest.raises(ValueError):
        _transport_logs_authorized(qualification, effects, components)


def test_gate_rejects_replaced_proof_and_unacknowledged_real_proof(genuine_proof):
    qualification, effects, components, reader, dispatch, capture = genuine_proof
    changed = dict(qualification.transport_verification.to_dict(), provider_call_id="fc-other")
    raw = canonical_bytes(changed)
    with pytest.raises(ValueError):
        replace(qualification.transport_verification, _raw=raw, _sha256=hashlib.sha256(raw).hexdigest())
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture)
    assert proof.to_dict()["status"] == "passed" and proof.to_dict()["durability_acknowledged"] is False
    assert _transport_logs_authorized(replace(qualification, transport_verification=proof), effects, components) is False


@pytest.mark.parametrize("fault", ["fsync", "ancestor_symlink", "named_directory"])
def test_retention_fault_never_returns_durable_capture(genuine_proof, monkeypatch, tmp_path, fault):
    _, _, _, _, dispatch, capture = genuine_proof
    root = tmp_path / "fault-retention"
    root.mkdir(mode=0o700)
    original = os.fsync
    triggered = [False]
    def fsync(descriptor):
        if fault == "fsync": raise OSError("fixture fsync failure")
        original(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not triggered[0]:
            triggered[0] = True
            if fault == "ancestor_symlink":
                moved = tmp_path / "displaced-retention"
                root.rename(moved)
                root.symlink_to(moved, target_is_directory=True)
            else:
                directory = root / "cpu-transport-evidence"
                directory.rename(root / "displaced-evidence")
                directory.mkdir(mode=0o700)
    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises((ValueError, OSError, ModalLocalPreparationUnavailable)) as error:
        _retain_transport_capture(root, dispatch.effect_id, "fc-probe", capture)
    if fault == "ancestor_symlink":
        assert error.value.code == "STATE_ROOT_INVALID"
    if fault == "fsync":
        assert len(list((root / "cpu-transport-evidence").glob("*.json"))) == 1
        with pytest.raises(FileExistsError):
            _retain_transport_capture(root, dispatch.effect_id, "fc-probe", capture)


def test_genuine_inconclusive_visibility_proof_cannot_enable_logs(genuine_proof, tmp_path):
    qualification, effects, components, reader, dispatch, _ = genuine_proof
    # A signed terminal receipt already exists; complete final bytes read afterward
    # cannot establish the admitted pre-terminal pending-delivery window.
    capture = reader.capture_transport_probe(dispatch, provider_call_id="fc-probe", sleep=lambda _: None)
    assert capture.to_dict()["verdict"] == "inconclusive"
    path, digest, raw = _retain_transport_capture(tmp_path / "inconclusive-retention", dispatch.effect_id, "fc-probe", capture)
    proof = reader.verify_transport_probe(dispatch, provider_call_id="fc-probe", capture=capture, retained_capture_bytes=raw)
    assert proof.to_dict()["status"] == "inconclusive" and proof.to_dict()["durability_acknowledged"] is True
    assert _transport_logs_authorized(replace(qualification, transport_verification=proof,
        transport_capture_path=path, transport_capture_sha256=digest), effects, components) is False


def test_absent_or_original_cpu_only_qualification_keeps_logs_false(genuine_proof):
    qualification, effects, components, _, _, _ = genuine_proof
    original = ModalHostCPUQualificationV1(qualification.receipt, qualification.output_sha256,
        qualification.provider_call_id, qualification.runtime_release_digest, qualification.deployment_facts_digest)
    assert _transport_logs_authorized(None, effects, components) is False
    assert _transport_logs_authorized(original, effects, components) is False
