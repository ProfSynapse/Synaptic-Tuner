"""Public facade contract test with a substituted ML child, not hardware evidence."""

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from types import SimpleNamespace

import pytest

from tests.runtime import test_native_mlx_publication as publication_fixtures
from tests.runtime.test_native_mlx_child import inputs
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest
from synaptic_tuner.api.v1.reference import (
    ReferenceHostPortsV1, ReferenceRequestPortsV1, compose_reference_authority, compose_reference_host,
)
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunLogsRequest, RunOperationError
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.runtime.native_mlx_material import canonical, native_runtime_spec, strict_json
from tuner.runtime.native_process import NativeInvocation, NativeOwnedProcess, NativeProcessLimits
from tuner.training.coordinator_material import _domain, _runtime
from tuner.training.modal_host_authority import HMACAuthenticator
from tuner.training.native_mlx_authority import NativeEvidenceAuthority, NativeMaterialAuthority
from tuner.training.native_mlx_compilation import compile_native_sft
from tuner.training.native_mlx_family import NativeCommandMaterialV1, NativeCommandCatalogV1, native_mlx_family
from tuner.training.native_mlx_planning import NativePlanningV1, NativeQualificationV1


publication = publication_fixtures.publication


class Clock:
    def now(self):
        return "2026-10-20T12:00:00Z"

    def now_epoch(self):
        return int(datetime.fromisoformat(self.now().replace("Z", "+00:00")).timestamp())


def prepare(publication):
    root = publication["output"].parent
    dataset, lock = b"fixture-prepared-rows", b"fixture-hash-lock"
    executable = str(root / "python")
    worker = str(root / "engine/tuner/runtime/native_mlx_child.py")
    guardian = str(root / "engine/tuner/runtime/native_process_guardian.py")
    runtime = canonical({
        "schema": "synaptic-native-mlx-capture/v1", "status": "OBSERVED_NOT_ADMITTED",
        "platform": {"system": "Darwin", "machine": "arm64", "macos": "15.0", "kernel_build": "fixture"},
        "config": {"stdlib": str(root / "stdlib"), "packages": str(root / "packages"),
                   "sources": [str(root / "engine/tuner")], "files": [executable],
                   "system_libraries": ["/usr/lib/libSystem.B.dylib"], "aliases": []},
        "runtime": {"python_version": "3.11.14", "executable_sha256": "1" * 64,
                    "dependency_lock_sha256": hashlib.sha256(lock).hexdigest(),
                    "distributions": {"mlx": "0.32.2", "mlx-lm": "0.32.2"},
                    "status": "CANDIDATE_MATCH_ONLY"},
        "executable": executable,
        "trees": [{"root": str(root / "engine/tuner"), "members": [
            {"path": "runtime/native_mlx_child.py", "size": 1, "sha256": "2" * 64},
            {"path": "runtime/native_process_guardian.py", "size": 1, "sha256": "3" * 64},
        ]}], "files": [], "native_dependencies": {}, "aliases": [],
        "trust_boundary": "exact-os-build-shared-cache-and-exclusive-host-ownership",
    })
    training, source, profile = inputs()
    source = replace(source, content_digest=hashlib.sha256(dataset).hexdigest(), size_bytes=len(dataset))
    profile = replace(profile, runtime_digest=hashlib.sha256(runtime).hexdigest())
    projection = compile_native_sft(training, source, profile)
    # Rebind fixture-produced artifacts to the exact public test plan.
    binding_path, result_path = publication["output"] / "input_binding.json", publication["output"] / "native_result.json"
    binding, result = strict_json(binding_path.read_bytes()), strict_json(result_path.read_bytes())
    binding.update(source=source.to_dict(), workload_digest=projection.workload.fingerprint,
                   runtime_digest=profile.runtime_digest)
    binding_path.write_bytes(canonical(binding))
    result.update(input_binding_digest=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
                  workload_digest=projection.workload.fingerprint, runtime_digest=profile.runtime_digest)
    result_path.write_bytes(canonical(result))
    (publication["output"] / "workload.json").write_bytes(projection.workload.canonical_bytes)
    for name, data in (("capture.json", runtime), ("dependency.lock", lock), ("dataset.jsonl", dataset)):
        (root / name).write_bytes(data)
    request = canonical({
        "schema": "synaptic-native-child-request/v1", "workload": projection.workload.document,
        "runtime_capture": str(root / "capture.json"), "dependency_lock": str(root / "dependency.lock"),
        "dataset": str(root / "dataset.jsonl"), "output": str(publication["output"]),
        "model_snapshot": asdict(publication["model"]), "tokenizer_snapshot": asdict(publication["tokenizer"]),
    })
    (root / "request.json").write_bytes(request)
    resolved = ResolvedTrainingRequest(
        "synaptic-resolved-training-request/v1", "request", "project", profile.source_digest,
        hashlib.sha256(training.canonical_json().encode()).hexdigest(),
        projection.workload.fingerprint,
        _domain("synaptic-coordinator-runtime/v1", _runtime(native_runtime_spec(runtime))),
        _domain("synaptic-coordinator-artifact-policy/v1", training.to_dict()["artifacts"]),
    )
    kwargs = dict(
        resolved=resolved, run=TrainingRunRef("run", "project"), workload=projection.workload.canonical_bytes,
        runtime_capture=runtime, limits=NativeProcessLimits(5, 1, 4096, 1024 * 1024),
        scope=ExecutionScopeV1("owner", "machine"), clock=Clock(),
    )
    return SimpleNamespace(root=root, executable=executable, worker=worker, guardian=guardian,
                           request=request, training=training, resolved=resolved, kwargs=kwargs)


class TestQualificationAuthority:
    __test__ = False

    def authenticate(self, proof):
        # Only a test trust boundary. No fixture is a real qualification receipt.
        return proof.tag == "f" * 64


def qualified_planning(state):
    candidate = NativePlanningV1(**state.kwargs)
    proof = NativeQualificationV1(
        candidate.profile_digest, state.resolved.source_digest, state.resolved.runtime_digest,
        "4" * 64, ("5" * 64, "6" * 64), "test-only-authority", "test-key", "f" * 64,
    )
    return NativePlanningV1(**state.kwargs, qualification=proof,
                            qualification_authority=TestQualificationAuthority())


def compose(state, publication, monkeypatch, *, fail=False):
    planning = qualified_planning(state)
    records, streams = InMemoryDurableRecordStoreV1(), InMemoryDurableStreamStoreV1()
    material_authority = NativeMaterialAuthority(
        "test-material", "test-key", HMACAuthenticator(
            {"test-key": b"test-material-key-" + b"x" * 32},
            allowed_purposes=frozenset({"native-command-material/v1"}),
        ),
    )
    catalog = NativeCommandCatalogV1(records, material_authority)
    calls = []

    class Grants:
        def authorize(self, requirements):
            return ExecutionGrant("test-native-grant")

        def bind(self, grant, *, operation, requirements):
            invocation = NativeInvocation(
                operation.digest, operation.digest, state.executable, "1" * 64,
                state.guardian, "3" * 64, state.worker, "2" * 64,
                str(state.root / "request.json"), hashlib.sha256(state.request).hexdigest(),
                str(state.root), planning.limits,
            )
            material = NativeCommandMaterialV1(
                operation.canonical_bytes, invocation, publication["model"], publication["tokenizer"],
                publication["tokenizer_members"], str(publication["output"]), publication["budget"],
                "test-material", "test-key", "0" * 64,
            )
            catalog.publish_once(material_authority.issue(material))
            return object()

    class Secrets:
        def resolve(self, ref):
            return "test-native-reference-authority-" + "s" * 32

    class Requests:
        def load(self, raw):
            assert raw == state.training.canonical_json()
            return TrainingRequest("request", "project", raw)

        def resolve(self, request):
            return state.resolved

    def child(self, invocation):
        calls.append(invocation.digest)
        self._retain(invocation, "claim", {"automatic_replay": False})
        if fail:
            raise RuntimeError("simulated uncertain child boundary")
        outcome = {
            "schema": "synaptic-native-process-outcome/v1", "reason": "worker_exited",
            "exit_code": 0, "termination_confirmed": True, "stdout_bytes": 0, "stderr_bytes": 0,
        }
        self._retain(invocation, "outcome", outcome)
        return outcome
    monkeypatch.setattr(NativeOwnedProcess, "run_once", child)
    ports = ReferenceHostPortsV1(records, streams, Clock(), Grants(), Secrets())
    authority = compose_reference_authority(
        clock=ports.clock, secrets=ports.secrets, authority_secret=SecretRef("env", "TEST_SECRET"),
    )
    evidence = NativeEvidenceAuthority("native-test-reader", "native-test-key", HMACAuthenticator(
        {"native-test-key": b"native-test-key-" + b"k" * 32},
        allowed_purposes=frozenset({
            "provider-run-observation/v1", "provider-log-page/v1", "native-artifact-verification/v1",
            "native-process-outcome/v1",
        }),
    ))
    family = native_mlx_family(
        planning=planning, catalog=catalog, material_authority=material_authority,
        records=records, evidence_authority=evidence,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority, clock=Clock(),
    )
    def recompose():
        return compose_reference_host(
            family=family, ports=ports, requests=ReferenceRequestPortsV1(Requests(), Requests(), planning),
            authority=authority,
        )
    return recompose(), recompose, planning, calls, catalog


def test_unqualified_native_candidate_cannot_advertise_or_preflight_execution(publication):
    state = prepare(publication)
    candidate = NativePlanningV1(**state.kwargs)
    assert not candidate.preflight(candidate.plan).ready
    assert not candidate.descriptor.capabilities.observe
    assert not candidate.descriptor.capabilities.artifact_streaming
    with pytest.raises(AttributeError, match="immutable"):
        candidate.limits = NativeProcessLimits(60, 1, 4096, 1024 * 1024)
    with pytest.raises(ValueError, match="qualification"):
        candidate.require_qualified()


def test_public_training_runs_verification_streaming_and_recomposition(publication, monkeypatch):
    state = prepare(publication)
    composition, recompose, planning, calls, catalog = compose(state, publication, monkeypatch)
    api = composition.api()
    loaded = api.training.load(state.training.canonical_json())
    plan = api.training.plan(api.training.resolve(loaded), planning.provider)
    ready = api.training.preflight(plan)
    started = api.training.start(plan, ready)
    assert started.accepted and len(calls) == 1
    assert api.training.start(plan, ready) == started and len(calls) == 1
    assert api.runs.outcome(started.run).state.value == "succeeded"
    assert api.runs.verify(started.run).verified
    stream = api.runs.artifacts(RunArtifactRequest(started.run, "final_model", 1024 * 1024))
    assert len(b"".join(stream.iter_bytes())) > 0
    assert recompose().api().runs.reverify(started.run).verified
    assert len(calls) == 1
    with pytest.raises(RunOperationError):
        api.runs.logs(RunLogsRequest(started.run, limit=1, maximum_bytes=4096))


def test_consumed_uncertain_native_submission_is_not_replayed(publication, monkeypatch):
    state = prepare(publication)
    composition, recompose, planning, calls, catalog = compose(state, publication, monkeypatch, fail=True)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())),
                             planning.provider)
    ready = api.training.preflight(plan)
    start = api.training.start(plan, ready)
    assert api.runs.show(start.run).state.value == "reconcile_required"
    recompose().api().training.start(plan, ready)
    assert len(calls) == 1


def test_native_catalog_rejects_forged_envelope_and_authenticated_collision(publication, monkeypatch):
    state = prepare(publication)
    composition, _, planning, _, catalog = compose(state, publication, monkeypatch)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())),
                             planning.provider)
    started = api.training.start(plan, api.training.preflight(plan))
    workflow = composition.stores.workflow_store.get(started.run)
    material = catalog.resolve(workflow.submit.command_digest)
    assert NativeCommandMaterialV1.parse(material.canonical_bytes) == material
    with pytest.raises(ValueError, match="unauthenticated"):
        catalog.publish_once(replace(material, tag="0" * 64))
    changed = catalog.authority.issue(replace(material, tokenizer_members=("different.json",)))
    with pytest.raises(ValueError, match="collision"):
        catalog.publish_once(changed)


def test_public_native_reverification_detects_changed_saved_adapter(publication, monkeypatch):
    state = prepare(publication)
    composition, _, planning, calls, _ = compose(state, publication, monkeypatch)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())),
                             planning.provider)
    started = api.training.start(plan, api.training.preflight(plan))
    api.runs.outcome(started.run)
    assert api.runs.verify(started.run).verified
    (publication["output"] / "adapters.safetensors").write_bytes(b"corrupt")
    with pytest.raises(RunOperationError):
        api.runs.reverify(started.run)
    assert len(calls) == 1


def test_native_qualification_is_bound_and_authority_failure_closes_preflight(publication):
    state = prepare(publication)
    qualified = qualified_planning(state)
    proof = replace(qualified._qualification, runtime_digest="9" * 64)
    rejected = NativePlanningV1(
        **state.kwargs, qualification=proof, qualification_authority=TestQualificationAuthority(),
    )
    assert not rejected.preflight(rejected.plan).ready
    assert not rejected.descriptor.capabilities.artifact_streaming


def test_unsigned_process_record_tampering_cannot_change_authenticated_terminal_outcome(publication, monkeypatch):
    state = prepare(publication)
    composition, _, planning, calls, catalog = compose(state, publication, monkeypatch)
    api = composition.api()
    plan = api.training.plan(api.training.resolve(api.training.load(state.training.canonical_json())),
                             planning.provider)
    started = api.training.start(plan, api.training.preflight(plan))
    workflow = composition.stores.workflow_store.get(started.run)
    material = catalog.resolve(workflow.submit.command_digest)
    key = f"native-mlx/{material.invocation.attempt}/outcome"
    retained = catalog.records.read(partition="effects", key=key)
    changed = strict_json(retained.canonical)
    changed["value"]["exit_code"] = 9
    assert catalog.records.compare_and_swap(
        partition="effects", key=key, expected_revision=retained.revision, canonical=canonical(changed),
    )
    with pytest.raises(RunOperationError):
        api.runs.outcome(started.run)
    assert len(calls) == 1
