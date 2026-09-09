from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import build_submit_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.providers.modal.contracts import operation_path
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_bundle import ModalCoordinatorBundle
from tuner.execution.providers.modal.coordinator_dispatch import build_modal_worker_dispatch
from tuner.execution.providers.modal.coordinator_wire import ModalWorkerLaunchExpectation
from tuner.execution.providers.modal.coordinator_worker import (
    ModalWorkerInvocation, ModalWorkerStaticExpectation, MountedModalCoordinatorWorker,
    _execute_modal_worker, admit_modal_worker,
)
from tuner.execution.providers.modal.worker_ports import ModalProcessResult
from tests.execution.providers.test_modal_coordinator_adapter import composed
from tests.execution.providers.test_modal_coordinator_bundle import _fixture
from tests.execution.providers.modal_coordinator_fixtures import real_launch_bundle_case


D = tuple(str(index) * 64 for index in range(1, 10))


class Auth:
    def __init__(self): self.calls = []
    def sign(self, purpose, payload, key_ref):
        return hmac.new(b"k" * 32, purpose.encode() + b"\0" + key_ref.encode() + b"\0" + payload, hashlib.sha256).digest()
    def verify(self, purpose, payload, tag, key_ref):
        self.calls.append(purpose)
        return hmac.compare_digest(self.sign(purpose, payload, key_ref), tag)


def case(*, roots=("/workspace/control", "/workspace/run", "/workspace/worker-control")):
    stage_binding, material, recipes, policy, closure = _fixture()
    bundle = ModalCoordinatorBundle.build(
        stage_binding, material, recipes, log_terminal_policy=policy,
        worker_closure_manifest=closure,
    )
    stage = __import__("tuner.execution.foundation_v2.commands", fromlist=["parse_exact_command"]).parse_exact_command(stage_binding.command_bytes)
    prep = stage.preparation
    predecessor = StagePredecessorV2(
        prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref,
        prep.scope.namespace_ref, prep.project_ref, prep.run_id, prep.plan_fingerprint,
        prep.preparation_digest, prep.workload_digest, stage.operation.effect.effect_id,
        D[0], D[1],
    )
    adapter, _, plan, execution = composed(resolved=material.planning_request)
    submit = build_submit_command(
        prep, "nonce-submit", adapter.payload(prep, EffectKind.SUBMIT),
        execution.executor_descriptor, predecessor,
    )
    submit_binding = ModalCommandBinding(
        submit.canonical_bytes, stage_binding.preparation_snapshot, stage_binding.deployment_bytes,
    )
    selection = submit_binding.deployment.selection
    key_ref = "stage-key"
    stage_claim = canonical_bytes({
        "schema_version": "synaptic.modal-stage-claim/v2", "command": stage.to_dict(),
        "command_digest": stage.digest, "binding_digest": stage_binding.authenticated_binding_digest,
        "provider_id": prep.provider.provider_id, "profile_ref": prep.provider.profile_ref,
        "account_ref": prep.scope.account_ref, "namespace_ref": prep.scope.namespace_ref,
        "project_ref": prep.project_ref, "run_id": prep.run_id,
        "effect_id": stage.operation.effect.effect_id,
        "invocation_nonce": stage.operation.invocation_nonce,
        "plan_fingerprint": prep.plan_fingerprint,
        "preparation_digest": prep.preparation_digest,
        "control_volume_id": "control-id", "artifact_volume_id": "artifact-id",
        "key_ref": key_ref, "bundle_sha256": hashlib.sha256(bundle.transport_bytes).hexdigest(),
        "bundle_size": len(bundle.transport_bytes),
    })
    auth = Auth()
    stage_tag = auth.sign("modal-stage-claim/v2", stage_claim, key_ref)
    bound_doc = {
        "reference": {"provider_id": prep.provider.provider_id, "profile_ref": prep.provider.profile_ref,
                      "account_ref": prep.scope.account_ref, "namespace_ref": prep.scope.namespace_ref,
                      "stage_ref": "modal-stage-claim:" + hashlib.sha256(stage_claim).hexdigest()},
        "effect_id": stage.operation.effect.effect_id, "command_digest": stage.digest,
        "command_bytes_digest": domain_digest("synaptic-foundation-command-bytes/v1", stage.canonical_bytes),
        "preparation_digest": prep.preparation_digest, "foundation_binding_digest": D[2],
        "foundation_outcome_digest": D[3], "authenticated_receipt_digest": D[0],
    }
    snapshot = json.loads(stage_binding.preparation_snapshot)
    volumes = snapshot["configuration"]["profile"]["volumes"]
    launch = {
        "schema_version": "synaptic.modal-launch-claim/v1",
        "submit_binding": json.loads(submit_binding.canonical_bytes),
        "submit_binding_digest": submit_binding.authenticated_binding_digest,
        "submit_command": submit.to_dict(), "submit_command_digest": submit.digest,
        "submit_effect_id": submit.operation.effect.effect_id,
        "submit_invocation_nonce": submit.operation.invocation_nonce,
        "stage_binding_digest": stage_binding.authenticated_binding_digest,
        "stage_command_digest": stage.digest, "stage_record_digest": D[1],
        "stage_assessment_digest": D[4], "stage_foundation_binding_digest": D[2],
        "stage_outcome_digest": D[3],
        "stage_bound_reference_digest": domain_digest("synaptic-stage-evidence-binding/v1", canonical_bytes(bound_doc)),
        "stage_predecessor": predecessor.to_dict(),
        "stage_claim_sha256": hashlib.sha256(stage_claim).hexdigest(),
        "stage_bundle_sha256": hashlib.sha256(bundle.transport_bytes).hexdigest(),
        "stage_bundle_size": len(bundle.transport_bytes),
        "control_volume_id": "control-id", "artifact_volume_id": "artifact-id",
        "configured_control_volume_ref": volumes["control_ref"],
        "configured_artifact_volume_ref": volumes["artifact_ref"], "key_ref": key_ref,
    }
    claim = canonical_bytes(launch)
    claim_tag = auth.sign("modal-launch-claim/v1", claim, key_ref)
    expectation = ModalWorkerLaunchExpectation(
        submit.canonical_bytes, submit_binding.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref,
        prep.scope.namespace_ref, selection.app_name, selection.function_name,
        "control-id", "artifact-id", volumes["control_ref"], volumes["artifact_ref"], key_ref,
        hashlib.sha256(stage_claim).hexdigest(), hashlib.sha256(bundle.transport_bytes).hexdigest(),
        len(bundle.transport_bytes), submit.executor.executor_id, submit.executor.implementation_version,
    )
    dispatch = build_modal_worker_dispatch(expectation, claim, claim_tag)
    static = ModalWorkerStaticExpectation(
        canonical_bytes(selection.to_dict()), prep.provider.provider_id, prep.provider.profile_ref,
        submit.executor.executor_id, submit.executor.implementation_version,
        "control-id", "artifact-id", volumes["control_ref"], volumes["artifact_ref"], key_ref,
        *roots,
    )
    return dispatch, stage_claim, stage_tag, bundle, auth, recipes, static


def admit(values):
    dispatch, claim, tag, bundle, auth, recipes, static = values
    return admit_modal_worker(
        dispatch.canonical_bytes, stage_claim=claim, stage_claim_tag=tag,
        bundle_transport=bundle.transport_bytes, verifier=auth, recipes=recipes, static=static,
    )


def real_case(monkeypatch):
    shared = real_launch_bundle_case(monkeypatch)
    envelope, staged, bundle = shared["envelope"], shared["material"], shared["bundle"]
    submit = __import__("tuner.execution.foundation_v2.commands", fromlist=["parse_exact_command"]).parse_exact_command(envelope.submit_binding.command_bytes)
    prep, selection = submit.preparation, envelope.submit_binding.deployment.selection
    snapshot = json.loads(envelope.submit_binding.preparation_snapshot)
    volumes = snapshot["configuration"]["profile"]["volumes"]
    expectation = ModalWorkerLaunchExpectation(
        submit.canonical_bytes, envelope.submit_binding.deployment_bytes,
        prep.provider.provider_id, prep.provider.profile_ref, prep.scope.account_ref,
        prep.scope.namespace_ref, selection.app_name, selection.function_name,
        staged.control_volume_id, staged.artifact_volume_id,
        volumes["control_ref"], volumes["artifact_ref"], staged.key_ref,
        hashlib.sha256(staged.claim).hexdigest(), hashlib.sha256(staged.bundle).hexdigest(),
        len(staged.bundle), submit.executor.executor_id, submit.executor.implementation_version,
    )
    dispatch = build_modal_worker_dispatch(expectation, envelope.claim, envelope.claim_tag)
    static = ModalWorkerStaticExpectation(
        canonical_bytes(selection.to_dict()), prep.provider.provider_id, prep.provider.profile_ref,
        submit.executor.executor_id, submit.executor.implementation_version,
        staged.control_volume_id, staged.artifact_volume_id,
        volumes["control_ref"], volumes["artifact_ref"], staged.key_ref,
        "/workspace/control", "/workspace/run", "/workspace/worker-control",
    )
    return dispatch, staged.claim, staged.claim_tag, bundle, shared["authenticator"], shared["recipes"], static


def test_pure_admission_reconstructs_foundation_native_invocation(monkeypatch):
    values = real_case(monkeypatch)
    invocation = admit(values)
    assert invocation.submit_command.stage_predecessor.stage_effect_id == invocation.stage_command.operation.effect.effect_id
    assert invocation.source.canonical_bytes == invocation.execution_source_bytes
    assert invocation.argv[1].endswith("/Trainers/sft/runtime_v1.py")
    assert invocation.closure_manifest_runtime_path.startswith(
        "/workspace/worker-control/operations/" + invocation.submit_command.operation.effect.effect_id
    )
    assert invocation.environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] == invocation.submit_command.preparation.workload_digest
    first = invocation.environment
    first["POISON"] = "x"
    assert "POISON" not in invocation.environment
    policy = invocation.log_policy
    policy["generation"] = 999
    assert invocation.log_policy["generation"] == 1
    assert invocation.stage_claim_sha256 == hashlib.sha256(values[1]).hexdigest()


def test_static_selection_mismatch_fails_before_wire_verification():
    values = list(case())
    values[-1] = replace(values[-1], provider_id="other")
    with pytest.raises(ValueError, match="static worker"):
        admit(tuple(values))
    assert values[4].calls == []


def test_static_selection_compares_every_canonical_field():
    values = list(case())
    selection = json.loads(values[-1].deployment_selection_bytes)
    selection["timeout_seconds"] += 1
    values[-1] = replace(
        values[-1], deployment_selection_bytes=canonical_bytes(selection),
    )
    with pytest.raises(ValueError, match="static worker selection"):
        admit(tuple(values))
    assert values[4].calls == []


def test_stage_authentication_and_bundle_are_both_required():
    values = case()
    dispatch, claim, _, bundle, auth, recipes, static = values
    with pytest.raises(ValueError, match="authentication"):
        admit_modal_worker(
            dispatch.canonical_bytes, stage_claim=claim, stage_claim_tag=b"bad",
            bundle_transport=bundle.transport_bytes, verifier=auth, recipes=recipes, static=static,
        )


def test_forged_or_mutated_invocation_stops_before_source_io():
    with pytest.raises(TypeError, match="admission-minted"):
        ModalWorkerInvocation()
    invocation = admit(case())
    object.__setattr__(invocation, "workload", b"{}")
    calls = []
    class Sources:
        def prepare_and_verify(self, *args): calls.append("source")
    class Processes:
        def run(self, *args, **kwargs): calls.append("process")
    with pytest.raises(ValueError):
        _execute_modal_worker(
            invocation, sources=Sources(), processes=Processes(),
            commit_prepared=lambda: None,
        )
    assert calls == []


def _execute_rejects_before_source(invocation, match, monkeypatch=None):
    calls = []
    class Sources:
        def prepare_and_verify(self, *args): calls.append("source")
    class Processes:
        def run(self, *args, **kwargs): calls.append("process")
    with pytest.raises(ValueError, match=match):
        _execute_modal_worker(
            invocation, sources=Sources(), processes=Processes(),
            commit_prepared=lambda: calls.append("commit"),
        )
    assert calls == []


def test_changed_self_consistent_workload_rejected_by_preparation_digest():
    invocation = admit(case())
    workload = json.loads(invocation.workload)
    workload["configuration"]["document"]["sft"]["max_steps"] = 2
    changed = canonical_bytes(workload)
    environment = invocation.environment
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + changed
    ).hexdigest()
    object.__setattr__(invocation, "workload", changed)
    object.__setattr__(invocation, "environment_items", tuple(sorted(environment.items())))
    _execute_rejects_before_source(invocation, "workload digest")


def test_changed_self_consistent_source_rejected_by_preparation_digest():
    invocation = admit(case())
    source = json.loads(invocation.execution_source_bytes)
    source["created_at"] = "2026-08-25T12:00:01Z"
    source_bytes = canonical_bytes(source)
    workload = json.loads(invocation.workload)
    workload["execution_source"] = source
    workload_bytes = canonical_bytes(workload)
    environment = invocation.environment
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + workload_bytes
    ).hexdigest()
    object.__setattr__(invocation, "execution_source_bytes", source_bytes)
    object.__setattr__(invocation, "workload", workload_bytes)
    object.__setattr__(invocation, "environment_items", tuple(sorted(environment.items())))
    _execute_rejects_before_source(invocation, "source digest")


def test_log_policy_requires_closed_fields_and_strict_bounds():
    invocation = admit(case())
    policy = invocation.log_policy | {"unknown": True}
    object.__setattr__(invocation, "log_policy_bytes", canonical_bytes(policy))
    _execute_rejects_before_source(invocation, "log policy")
    invocation = admit(case())
    policy = invocation.log_policy
    policy["max_chunk_bytes"] = True
    object.__setattr__(invocation, "log_policy_bytes", canonical_bytes(policy))
    _execute_rejects_before_source(invocation, "max_chunk_bytes")


def test_closure_path_rejects_noncanonical_raw_spelling():
    invocation = admit(case())
    changed = invocation.closure_manifest_runtime_path.replace("/workspace/", "/workspace//")
    environment = invocation.environment
    environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = changed
    object.__setattr__(invocation, "closure_manifest_runtime_path", changed)
    object.__setattr__(invocation, "environment_items", tuple(sorted(environment.items())))
    _execute_rejects_before_source(invocation, "closure path")


def test_bad_launch_tag_stops_before_mounted_reads_or_other_effects(tmp_path):
    values = case(roots=tuple((tmp_path / name).as_posix() for name in ("control", "artifact", "worker")))
    dispatch, _, _, _, auth, _, static = values
    poisoned = build_modal_worker_dispatch(
        dispatch.expectation, dispatch.launch_claim, b"bad",
    )
    calls = []
    class Sources:
        def prepare_and_verify(self, *args): calls.append("source")
    class Processes:
        def run(self, *args, **kwargs): calls.append("process")
    class Completion:
        def finalize(self, *args, **kwargs): calls.append("completion")
    worker = MountedModalCoordinatorWorker(
        verifier=auth, sources=Sources(), processes=Processes(),
        completion=Completion(), static=static,
    )
    with pytest.raises(ValueError, match="launch authentication failed"):
        worker(poisoned.canonical_bytes, "job-a", lambda: calls.append("commit"))
    assert calls == []


def test_cloned_closure_mismatch_stops_before_write_stage_or_process(monkeypatch):
    invocation = admit(case())
    events = []
    class Sources:
        def prepare_and_verify(self, *args): events.append("source")
    class Processes:
        def run(self, *args, **kwargs): events.append("process")
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.read_locked_closure_manifest",
        lambda source: events.append("closure") or b"different",
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.write_runtime_closure_manifest",
        lambda *args: events.append("write"),
    )
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.stage_runtime_worker",
        lambda *args: events.append("stage"),
    )
    with pytest.raises(Exception, match="locked_source_mismatch"):
        _execute_modal_worker(
            invocation, sources=Sources(), processes=Processes(),
            commit_prepared=lambda: events.append("commit"),
        )
    assert events == ["source", "closure"]


def test_mounted_wrapper_reads_stage_effect_then_runs_and_completes(tmp_path, monkeypatch):
    control, artifact, worker_control = (tmp_path / "control", tmp_path / "artifact", tmp_path / "worker")
    values = case(roots=tuple(path.as_posix() for path in (control, artifact, worker_control)))
    dispatch, claim, tag, bundle, auth, _, static = values
    stage_id = admit(values).stage_command.operation.effect.effect_id
    control_path = control / operation_path(stage_id, "control")
    artifact_path = artifact / operation_path(stage_id, "input")
    control_path.mkdir(parents=True); artifact_path.mkdir(parents=True)
    (control_path / "stage-claim.v2.json").write_bytes(claim)
    (control_path / "stage-claim.v2.mac").write_bytes(tag)
    (artifact_path / "bundle.bin").write_bytes(bundle.transport_bytes)
    events = []
    class Sources:
        def prepare_and_verify(self, source, deployment): events.append("source")
    class Processes:
        def run(self, argv, **kwargs):
            events.append("process"); kwargs["commit_prepared"](); return ModalProcessResult(0)
    class Completion:
        def finalize(self, invocation, result, *, job_ref):
            events.append("completion")
            self.invocation = invocation
            return type("Done", (), {"status_code": "completed"})()
    sources, processes, completion, commits = Sources(), Processes(), Completion(), []
    monkeypatch.setattr(
        "tuner.execution.providers.modal.coordinator_worker.worker_source.read_locked_closure_manifest",
        lambda source: completion.invocation.closure_manifest if hasattr(completion, "invocation") else bundle.members[-2].content,
    )
    # Avoid local checkout mutation; source mechanics have their own focused tests.
    monkeypatch.setattr("tuner.execution.providers.modal.coordinator_worker.worker_source.write_runtime_closure_manifest", lambda *a: events.append("write"))
    monkeypatch.setattr("tuner.execution.providers.modal.coordinator_worker.worker_source.stage_runtime_worker", lambda *a: events.append("stage"))
    closure = next(member.content for member in bundle.members if member.name == "worker-closure-manifest.json")
    monkeypatch.setattr("tuner.execution.providers.modal.coordinator_worker.worker_source.read_locked_closure_manifest", lambda source: events.append("closure") or closure)
    worker = MountedModalCoordinatorWorker(
        verifier=auth, sources=sources, processes=processes, completion=completion, static=static,
    )
    result = worker(dispatch.canonical_bytes, "job-a", lambda: (events.append("commit"), commits.append(True)))
    assert result["status_code"] == "completed" and commits == [True]
    assert events == ["source", "closure", "write", "stage", "process", "commit", "completion"]
