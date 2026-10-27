"""Public facade lifecycle controls with a substituted CPU worker, never ML proof."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

from tests.runtime import test_native_mlx_family as fixtures
from tuner.runtime import native_process_guardian, native_lifecycle_protocol
from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
from tuner.runtime.native_process_session import NativeProcessSession
from tuner.runtime.native_mlx_material import canonical, native_runtime_spec, strict_json
from tuner.training.native_mlx_compilation import compile_native_sft, parse_native_workload
from tuner.training.coordinator_material import _domain, _runtime
from tuner.training.native_mlx_family import NativeCancellationMaterialV1
from tuner.training.native_mlx_planning import NativeLifecycleGrantV1
from synaptic_tuner.api.v1.native_mlx import (
    NativeCommandCatalogV1, NativeCommandMaterialV1, NativeMaterialAuthority, NativeEvidenceAuthority,
    NativeInvocation, NativePlanningV1, NativeQualificationV1, NativeRestrictedAdmissionV1, native_mlx_family,
)
from synaptic_tuner.api.v1.reference import (
    ReferenceHostPortsV1, ReferenceRequestPortsV1, compose_reference_authority, compose_reference_host,
)
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import TrainingRequest
from synaptic_tuner.api.v1.runs_facade import RunLogsRequest, RunOperationError
from tuner.training.modal_host_authority import HMACAuthenticator

publication = fixtures.publication

SCRIPT = """import json,os,sys,time
from pathlib import Path
fd=int(sys.argv[sys.argv.index('--progress-fd')+1])
for step in (1,2):
    value={'kind':'progress','phase':'training','step':step,'metrics':{'loss':2.0/step}}
    os.write(fd,json.dumps(value,sort_keys=True,separators=(',',':')).encode()+b'\\n')
while not Path('release').exists(): time.sleep(.01)
"""


def setup(publication, monkeypatch, script=SCRIPT):
    state = fixtures.prepare(publication)
    root = state.root
    executable, worker, guardian = map(Path, (state.executable, state.worker, state.guardian))
    executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    executable.chmod(0o700)
    worker.parent.mkdir(parents=True, mode=0o700)
    worker.write_text(script)
    guardian.write_bytes(Path(native_process_guardian.__file__).read_bytes())
    codec = worker.with_name("native_lifecycle_protocol.py")
    codec.write_bytes(Path(native_lifecycle_protocol.__file__).read_bytes())
    for path in (worker.parent.parent, worker.parent, worker, guardian, codec):
        path.chmod(0o700 if path.is_dir() else 0o600)
    launch = NativeLifecycleLaunchV1(native_lifecycle_protocol.NativeLifecyclePolicyV1(32, 65536),
                                    hashlib.sha256(codec.read_bytes()).hexdigest())
    capture = strict_json(state.kwargs["runtime_capture"])
    capture["runtime"]["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    capture["trees"][0]["members"] = [
        {"path": "runtime/" + path.name, "size": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (worker, guardian, codec)]
    runtime = canonical(capture)
    old = parse_native_workload(state.kwargs["workload"])
    projection = compile_native_sft(state.training, old.source,
                                    replace(old.profile, runtime_digest=hashlib.sha256(runtime).hexdigest()))
    resolved = replace(state.resolved, workload_digest=projection.workload.fingerprint,
        runtime_digest=_domain("synaptic-coordinator-runtime/v1", _runtime(native_runtime_spec(runtime))))
    request = strict_json(state.request)
    request.update(schema="synaptic-native-child-request/v2", lifecycle=launch.to_dict(),
                   workload=projection.workload.document)
    raw = canonical(request)
    (root / "request.json").write_bytes(raw)
    (root / "capture.json").write_bytes(runtime)
    # Rebind synthetic fixture publication, not real model/run evidence.
    output = publication["output"]
    binding = strict_json((output / "input_binding.json").read_bytes())
    binding.update(workload_digest=projection.workload.fingerprint, runtime_digest=projection.profile.runtime_digest)
    (output / "input_binding.json").write_bytes(canonical(binding))
    result = strict_json((output / "native_result.json").read_bytes())
    result.update(workload_digest=projection.workload.fingerprint, runtime_digest=projection.profile.runtime_digest,
                  input_binding_digest=hashlib.sha256(canonical(binding)).hexdigest())
    (output / "native_result.json").write_bytes(canonical(result))
    (output / "workload.json").write_bytes(projection.workload.canonical_bytes)
    kwargs = dict(state.kwargs, resolved=resolved, workload=projection.workload.canonical_bytes,
                  runtime_capture=runtime, lifecycle=launch)
    candidate = NativePlanningV1(**kwargs)
    proof = NativeQualificationV1(candidate.profile_digest, resolved.source_digest, resolved.runtime_digest,
        "4" * 64, ("5" * 64, "6" * 64), "fixture-only", "fixture-key", "f" * 64)
    grant = NativeLifecycleGrantV1(candidate.profile_digest, resolved.source_digest, resolved.runtime_digest,
        hashlib.sha256(canonical(launch.to_dict())).hexdigest(), "4" * 64, "qualified",
        ("7" * 64, "8" * 64, "9" * 64), None, None, None, "fixture-only", "fixture-key", "f" * 64)
    planning = NativePlanningV1(**kwargs, qualification=proof, qualification_authority=fixtures.TestQualificationAuthority(),
        lifecycle_grant=grant, lifecycle_authority=fixtures.TestQualificationAuthority())
    monkeypatch.setattr("tuner.training.native_mlx_family.verify_native_material", lambda *a, **k: None)
    return SimpleNamespace(**{**vars(state), "request": raw, "resolved": resolved, "kwargs": kwargs},
                           planning=planning, launch=launch)


def compose(state, publication, *, records=None, streams=None):
    records = records if records is not None else InMemoryDurableRecordStoreV1()
    streams = streams if streams is not None else InMemoryDurableStreamStoreV1()
    planning = state.planning
    material_authority = NativeMaterialAuthority("fixture-material", "fixture-key", HMACAuthenticator(
        {"fixture-key": b"material-fixture-" + b"x" * 32},
        allowed_purposes=frozenset({"native-command-material/v1", "native-cancellation-material/v1"})))
    catalog = NativeCommandCatalogV1(records, material_authority)
    submit = []
    class Grants:
        def authorize(self, requirements): return ExecutionGrant("fixture-native-grant")
        def bind(self, grant, *, operation, requirements):
            if operation.operation.effect.kind.value == "cancel":
                reference = operation.to_dict()["cancellation"]["provider_job_ref"]
                assert reference.startswith("native-run-")
                original = catalog.resolve_submission(reference[len("native-run-"):])
                material = NativeCancellationMaterialV1(operation.canonical_bytes, original.command_digest,
                    original.invocation.digest, "fixture-material", "fixture-key", "0" * 64)
            else:
                attempt = (planning._restricted_admission.attempt if planning._restricted_admission is not None
                           else operation.digest)
                invocation = NativeInvocation(attempt, operation.digest,
                    state.executable, hashlib.sha256(Path(state.executable).read_bytes()).hexdigest(),
                    state.guardian, hashlib.sha256(Path(state.guardian).read_bytes()).hexdigest(),
                    state.worker, hashlib.sha256(Path(state.worker).read_bytes()).hexdigest(),
                    str(state.root / "request.json"), hashlib.sha256(state.request).hexdigest(),
                    str(state.root), planning.limits)
                material = NativeCommandMaterialV1(operation.canonical_bytes, invocation, publication["model"],
                    publication["tokenizer"], publication["tokenizer_members"], str(publication["output"]),
                    publication["budget"], "fixture-material", "fixture-key", "0" * 64)
                if operation.operation.effect.kind.value == "submit": submit.append(operation.digest)
            catalog.publish_once(material_authority.issue(material))
            return object()
    class Secrets:
        def resolve(self, ref): return "fixture-native-reference-" + "s" * 32
    class Requests:
        def load(self, raw): return TrainingRequest("request", "project", raw)
        def resolve(self, request): return state.resolved
    clock = planning._clock
    ports = ReferenceHostPortsV1(records, streams, clock, Grants(), Secrets())
    authority = compose_reference_authority(clock=clock, secrets=ports.secrets, authority_secret=SecretRef("env", "FIXTURE"))
    evidence = NativeEvidenceAuthority("fixture-reader", "fixture-key", HMACAuthenticator(
        {"fixture-key": b"reader-fixture-" + b"r" * 32}, allowed_purposes=frozenset({
            "provider-run-observation/v1", "provider-log-page/v1", "native-artifact-verification/v1",
            "native-process-outcome/v1", "native-lifecycle-capability/v1", "native-cancellation-ack/v1",
            "native-progress-metadata/v1", "native-stage-ack/v1", "native-quiescence/v1"})))
    session = NativeProcessSession(records, lifecycle=state.launch, lifecycle_authority=evidence, streams=streams)
    family = native_mlx_family(planning=planning, catalog=catalog, material_authority=material_authority,
        records=records, evidence_authority=evidence, foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority, clock=clock, lifecycle_session=session)
    host = compose_reference_host(family=family, ports=ports,
        requests=ReferenceRequestPortsV1(Requests(), Requests(), planning), authority=authority,
        allow_current_cancel_authorization=True)
    return SimpleNamespace(api=host.api(), host=host, session=session, family=family, records=records,
                           streams=streams, catalog=catalog, submit=submit)


def start(state, bundle):
    training = bundle.api.training
    plan = training.plan(training.resolve(training.load(state.training.canonical_json())), state.planning.provider)
    ready = training.preflight(plan)
    assert ready.ready
    return training.start(plan, ready)


def until(action, predicate):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        value = action()
        if predicate(value): return value
        time.sleep(.01)
    pytest.fail("bounded public native lifecycle fixture did not complete")


def test_public_progress_completion_verification_and_cold_reads(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    bundle = compose(state, publication)
    try:
        started = start(state, bundle)
        assert started.accepted
        assert bundle.api.runs.outcome(started.run).state.value == "running"
        page = until(lambda: bundle.api.runs.logs(RunLogsRequest(started.run, limit=1, maximum_bytes=4096)),
                     lambda page: bool(page.entries) and page.truncated)
        assert page.entries[0].sequence == 0 and json.loads(page.entries[0].message)["phase"] == "training"
        assert page.next_cursor is not None
        next_page = bundle.api.runs.logs(RunLogsRequest(started.run, cursor=page.next_cursor, limit=1, maximum_bytes=4096))
        assert next_page.entries[0].sequence == 1
        (state.root / "release").write_text("complete")
        done = until(lambda: bundle.api.runs.outcome(started.run), lambda out: out.state.value == "succeeded")
        assert bundle.api.runs.verify(done.run).verified
        cold = compose(state, publication, records=bundle.records, streams=bundle.streams)
        try:
            assert cold.api.runs.reverify(done.run).verified
            repeated = cold.api.runs.logs(RunLogsRequest(done.run, limit=1, maximum_bytes=4096))
            assert repeated.entries == page.entries
        finally: cold.session.close()
    finally: bundle.session.close()


def test_public_cold_recovery_after_owner_close_uses_no_source_or_dispatch(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    owner = compose(state, publication)
    started = start(state, owner)
    until(lambda: owner.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
    owner.session.close()
    for name in ("request.json",): (state.root / name).unlink()
    Path(state.worker).unlink()
    Path(state.guardian).unlink()
    cold = compose(state, publication, records=owner.records, streams=owner.streams)
    try:
        def forbidden(*args, **kwargs): raise AssertionError("cold read must not dispatch")
        monkeypatch.setattr(cold.session, "start_once", forbidden)
        cold.api.runs.reconcile(started.run)
        done = cold.api.runs.outcome(started.run)
        assert done.state.value == "failed" and done.diagnostic_code == "native_worker_failed"
        assert cold.submit == []
    finally: cold.session.close()


def test_public_lost_submit_ack_recovers_original_run_without_retraining(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    owner = compose(state, publication)
    original = owner.family.reader.execute_once
    calls = []
    def lost(payload, request):
        result = original(payload, request)
        if request.effect_kind == "submit":
            calls.append(request.command_digest)
            raise RuntimeError("fixture caller lost submission acknowledgement")
        return result
    monkeypatch.setattr(owner.family.reader, "execute_once", lost)
    try:
        from tuner.execution.coordinator_v1.coordinator import CoordinatorErrorV1
        with pytest.raises(CoordinatorErrorV1): start(state, owner)
        run = state.planning.run
        assert len(calls) == 1
    finally: owner.session.close()
    cold = compose(state, publication, records=owner.records, streams=owner.streams)
    try:
        def forbidden(*args, **kwargs): raise AssertionError("reconciliation must not dispatch")
        monkeypatch.setattr(cold.session, "start_once", forbidden)
        recovered = until(lambda: cold.api.runs.reconcile(run), lambda value: value.state.value == "queued")
        assert recovered.state.value == "queued"
        assert cold.api.runs.outcome(run).state.value == "failed"
        assert cold.submit == [] and len(calls) == 1
    finally: cold.session.close()


@pytest.mark.parametrize("fault", ["absent", "wrong_profile", "wrong_launch", "wrong_hardware", "wrong_tag"])
def test_candidate_selection_or_mismatched_grant_cannot_enable_lifecycle(publication, monkeypatch, fault):
    state = setup(publication, monkeypatch)
    proof, grant = state.planning._qualification, state.planning._lifecycle_grant
    if fault == "absent": grant = None
    elif fault == "wrong_profile": grant = replace(grant, profile_digest="0" * 64)
    elif fault == "wrong_launch": grant = replace(grant, launch_digest="0" * 64)
    elif fault == "wrong_hardware": grant = replace(grant, hardware_digest="0" * 64)
    elif fault == "wrong_tag": grant = replace(grant, tag="0" * 64)
    plan = NativePlanningV1(**state.kwargs, qualification=proof,
        qualification_authority=fixtures.TestQualificationAuthority(), lifecycle_grant=grant,
        lifecycle_authority=fixtures.TestQualificationAuthority())
    assert plan.qualified() is True
    assert plan.descriptor.capabilities.logs is False
    assert plan.descriptor.capabilities.cancel is False
    assert plan.descriptor.capabilities.reconcile is False
    assert plan.preflight(plan.plan).ready is False
    with pytest.raises(ValueError, match="lifecycle_authority"):
        plan.require_lifecycle()


def test_public_owned_cancel_does_not_require_mutable_source_files(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    bundle = compose(state, publication)
    try:
        started = start(state, bundle)
        until(lambda: bundle.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
        for path in (state.root / "request.json", Path(state.worker), Path(state.guardian)):
            path.unlink()
        assert bundle.api.runs.cancel(started.run, "stop retained owner").state.value == "cancel_requested"
        assert until(lambda: bundle.api.runs.outcome(started.run), lambda out: out.state.value == "cancelled")
    finally: bundle.session.close()


def test_restricted_start_expiry_does_not_renew_or_block_owned_stop_and_reads(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    class Clock(fixtures.Clock):
        value = "2026-10-20T12:00:00Z"
        def now(self): return self.value
    clock = Clock()
    kwargs = dict(state.kwargs, clock=clock)
    candidate = NativePlanningV1(**kwargs)
    projection = parse_native_workload(kwargs["workload"])
    admission = NativeRestrictedAdmissionV1(candidate.profile_digest, state.resolved.source_digest,
        state.resolved.runtime_digest, "4" * 64, projection.source.content_digest,
        publication["model"].digest, publication["tokenizer"].digest, state.resolved.workload_digest,
        "c" * 64, candidate.run.run_id, "restricted-fixture", projection.controls.iterations,
        "2026-10-20T12:30:00Z", "fixture-only", "fixture-key", "f" * 64)
    grant = NativeLifecycleGrantV1(candidate.profile_digest, state.resolved.source_digest, state.resolved.runtime_digest,
        hashlib.sha256(canonical(state.launch.to_dict())).hexdigest(), "4" * 64, "restricted", (),
        "c" * 64, candidate.run.run_id, "restricted-fixture", "fixture-only", "fixture-key", "f" * 64)
    bound = dict(restricted_admission=admission, restricted_authority=fixtures.TestQualificationAuthority(),
        selection_digest="c" * 64, lifecycle_grant=grant, lifecycle_authority=fixtures.TestQualificationAuthority())
    state.planning = NativePlanningV1(**kwargs, **bound)
    assert not state.planning.qualified() and state.planning.restricted()
    bundle = compose(state, publication)
    try:
        started = start(state, bundle)
        until(lambda: bundle.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
        clock.value = "2026-10-20T12:31:00Z"
        assert not state.planning.restricted() and state.planning.lifecycle_allowed()
        with pytest.raises(ValueError, match="admission_required"):
            state.planning.require_admitted(attempt=admission.attempt, model_digest=admission.model_digest,
                                           tokenizer_digest=admission.tokenizer_digest)
        assert bundle.api.runs.cancel(started.run, "stop after initial start expiry").state.value == "cancel_requested"
        until(lambda: bundle.api.runs.outcome(started.run), lambda out: out.state.value == "cancelled")
        state.planning = NativePlanningV1(**kwargs, **bound, retained_read=True)
        cold = compose(state, publication, records=bundle.records, streams=bundle.streams)
        try:
            assert cold.api.runs.logs(RunLogsRequest(started.run)).entries
            assert not state.planning.preflight(state.planning.plan).ready
            assert cold.submit == []
        finally: cold.session.close()
    finally: bundle.session.close()


def test_lost_cancel_ack_reconciles_without_resending_stop(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    owner = compose(state, publication)
    try:
        started = start(state, owner)
        until(lambda: owner.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
        original = owner.family.reader.execute_once
        calls = []
        def lost(payload, request):
            result = original(payload, request)
            if request.effect_kind == "cancel":
                calls.append(request.command_digest)
                raise RuntimeError("fixture cancellation acknowledgement lost")
            return result
        monkeypatch.setattr(owner.family.reader, "execute_once", lost)
        with pytest.raises(RunOperationError): owner.api.runs.cancel(started.run, "fixture lost ack")
    finally: owner.session.close()
    cold = compose(state, publication, records=owner.records, streams=owner.streams)
    try:
        def forbidden(*args, **kwargs): raise AssertionError("reconcile cannot resend stop or start")
        monkeypatch.setattr(cold.session, "start_once", forbidden)
        monkeypatch.setattr(cold.session, "cancel", forbidden)
        until(lambda: cold.api.runs.reconcile(started.run), lambda value: value.state.value == "cancel_requested")
        assert cold.api.runs.outcome(started.run).state.value == "cancelled"
        assert len(calls) == 1 and cold.submit == []
    finally: cold.session.close()


@pytest.mark.parametrize("complete", [True, False])
def test_exit_zero_during_stop_needs_complete_publication_to_resolve_success(publication, monkeypatch, complete):
    script = SCRIPT.replace("fd=int", "import signal\nsignal.signal(signal.SIGTERM,lambda *args:sys.exit(0))\nfd=int")
    state = setup(publication, monkeypatch, script)
    owner = compose(state, publication)
    try:
        started = start(state, owner)
        until(lambda: owner.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
        if not complete: (publication["output"] / "native_result.json").unlink()
        owner.api.runs.cancel(started.run, "completion race")
        target = "succeeded" if complete else "cancelled"
        done = until(lambda: owner.api.runs.outcome(started.run), lambda value: value.state.value == target)
        if complete: assert owner.api.runs.verify(done.run).verified
    finally: owner.session.close()


@pytest.mark.parametrize("change", ["tag", "future", "witness", "record"])
def test_quiescence_proof_cannot_change_scope_time_record_or_terminal_witness(publication, monkeypatch, change):
    from tuner.execution.coordinator_v1.coordinator import CoordinatorErrorV1
    state = setup(publication, monkeypatch)
    owner = compose(state, publication)
    original = owner.family.reader.execute_once
    def lost(payload, request):
        result = original(payload, request)
        if request.effect_kind == "submit": raise RuntimeError("fixture lost caller")
        return result
    monkeypatch.setattr(owner.family.reader, "execute_once", lost)
    try:
        with pytest.raises(CoordinatorErrorV1): start(state, owner)
    finally: owner.session.close()
    command = parse_native_command(owner.catalog.resolve(owner.submit[-1]).command_bytes)
    record = owner.host.foundation.get(command.operation.effect.effect_id)
    recovery = owner.family.recovery_verifier
    now = state.planning._clock.now_epoch()
    proof = recovery.obtain(recovery._request(record), now_epoch=now)
    assert recovery.verify_quiescence(proof, record, now_epoch=now)
    if change == "record":
        record = replace(record, dispatch_epoch=record.dispatch_epoch + 1)
    else:
        envelope = strict_json(proof.canonical_bytes)
        if change == "tag": envelope["tag"] = "0" * 64
        else:
            content = envelope["content"]
            if change == "future": content["issued_epoch"] = now + 1
            else: content["witness"]["terminal_digest"] = "0" * 64
            envelope = strict_json(owner.family.evidence_authority.recovery_evidence(content))
        proof = replace(proof, canonical_bytes=canonical(envelope))
    assert recovery.verify_quiescence(proof, record, now_epoch=now) is False
    assert recovery.verify_finality(proof, record, None, now_epoch=now) is False


def parse_native_command(raw):
    from tuner.execution.foundation_v2.commands import parse_exact_command
    return parse_exact_command(raw)


def test_public_cancel_records_request_then_confirmed_cancellation(publication, monkeypatch):
    state = setup(publication, monkeypatch)
    bundle = compose(state, publication)
    try:
        started = start(state, bundle)
        until(lambda: bundle.api.runs.logs(RunLogsRequest(started.run)), lambda page: bool(page.entries))
        requested = bundle.api.runs.cancel(started.run, "fixture stop")
        assert requested.state.value == "cancel_requested"
        done = until(lambda: bundle.api.runs.outcome(started.run), lambda out: out.state.value == "cancelled")
        assert done.state.value == "cancelled"
    finally: bundle.session.close()
