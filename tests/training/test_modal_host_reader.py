from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from threading import Event
import time

import pytest

from tuner.training.modal_host_authority import HMACAuthenticator, ReaderEvidenceAuthority, UTCClock
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.model import (
    EffectIntentV1, ProviderReadPurposeV1, ProviderRunReadRequestV1,
    WorkflowPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.repository import EffectState
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1
from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.execution.providers.modal.packaged_reader import (
    ModalPackagedArtifactMember, ModalPackagedCompletionObservation,
)
from tuner.training.modal_host_reader import (
    ModalPackagedCoordinatorReaderV1, ModalPackagedReadUnavailable,
)
import tuner.training.modal_host_reader as host_reader_module

from tests.execution.providers.test_modal_packaged_dispatch import _case
from tests.execution.coordinator_v1.test_state_machine import (
    Auth as FoundationAuth, AssessmentAuth, assessment, record,
)


class _FakeFacade:
    def __init__(self, state, effect_id, completion_digest, result=None):
        self.state = state
        self.result = result
        self.client = object()
        facade = self

        class Call:
            is_hydrated = True
            object_id = "fc-1"

            def hydrate(self, client):
                assert client is facade.client
                return self

            def get(self, *, timeout):
                assert timeout == 0
                if facade.state is ModalFunctionCallState.PENDING:
                    raise TimeoutError()
                if facade.result is not None:
                    return facade.result
                return {
                    "schema_version": "synaptic-modal-packaged-worker-result/v1",
                    "effect_id": effect_id,
                    "status_code": ("failed" if facade.state == "failed" else "completed"),
                    "completion_sha256": completion_digest,
                }

        class FunctionCall:
            @staticmethod
            def from_id(ref, *, client):
                assert ref == "fc-1" and client is facade.client
                return Call()

        self.sdk = SimpleNamespace(FunctionCall=FunctionCall)


class _FakePackagedReader:
    def __init__(self, completion):
        self.completion = completion

    def observe_completion(self, binding, *, provider_job_ref):
        assert binding.command_digest == self.completion.command_digest
        assert provider_job_ref == self.completion.provider_job_ref
        return self.completion

    def iter_artifact(self, _binding, _observation, *, role, maximum_bytes):
        assert maximum_bytes >= 4 and role == "final_model"
        yield b"data"


class _ProjectedReader(ModalPackagedCoordinatorReaderV1):
    def _binding(self, _request, _purpose):
        return self.binding
    def _source_workflow(self, request):
        return SimpleNamespace(phase=WorkflowPhaseV1.QUEUED)


def _reader(state):
    binding, _, _, _ = _case()
    ref = ScopedProviderRunRefV1(
        "modal", binding.command.preparation.provider.profile_ref,
        binding.provider_facts.account_ref,
        binding.command.preparation.scope.namespace_ref, "fc-1",
    )
    roles = sorted((
        "workload_record", "training_lineage", "training_metrics",
        "final_model", "tokenizer",
    ))
    members = tuple(ModalPackagedArtifactMember(
        role, f"operations/x/output/{role}", 4, sha256(b"data").hexdigest(),
        f"entry-{role}",
    ) for role in roles)
    completion = ModalPackagedCompletionObservation(
        binding.command.operation.effect.effect_id, binding.command_digest,
        "fc-1", binding.runtime_release.manifest_digest,
        binding.provider_binding.binding_digest,
        binding.execution_binding.binding_digest, "a" * 64, members,
    )
    auth = ReaderEvidenceAuthority(
        "modal-reader", "reader-key",
        HMACAuthenticator(
            {"reader-key": b"x" * 32},
            allowed_purposes=frozenset({
                "provider-run-observation/v1", "provider-log-page/v1",
            }),
        ),
    )
    reader = _ProjectedReader(
        bindings=None, binding_authority=None, foundation_repository=None,
        foundation_authenticator=None, assessment_authenticator=None,
        packaged_reader=_FakePackagedReader(completion),
        facade=_FakeFacade(state, completion.effect_id, completion.completion_digest),
        evidence_authority=auth, clock=UTCClock(),
    )
    reader.binding = binding
    request = SimpleNamespace(
        provider_run=SimpleNamespace(reference=ref, binding_digest="b" * 64),
        request_digest="c" * 64, source_workflow_record_digest="d" * 64,
        source_revision=1, run=TrainingRunRef(binding.execution_binding.run_ref, "project"),
    )
    return reader, request, auth


def test_returned_call_requires_completion_then_projects_five_artifacts():
    reader, request, auth = _reader(ModalFunctionCallState.RETURNED)
    observation = reader.observe(request)
    assert auth.authenticate_observation(observation)
    assert observation.content.phase.value == "succeeded"
    manifest = reader.artifacts(request)
    assert len(manifest.artifacts) == 5
    assert b"".join(reader.iter_artifact_bytes(
        request, manifest, "final_model", maximum_bytes=4,
    )) == b"data"


def test_pending_call_does_not_manufacture_queued_or_running_state():
    reader, request, auth = _reader(ModalFunctionCallState.PENDING)
    observation = reader.observe(request)
    assert auth.authenticate_observation(observation)
    assert observation.content.phase.value == "queued"
    with pytest.raises(ModalPackagedReadUnavailable, match="log_query_invalid"):
        reader.logs(request, None)


def test_failed_packaged_result_stops_with_fixed_failure_diagnostic():
    reader, request, _ = _reader("failed")
    reader._facade.result = {
        "schema_version": "synaptic-modal-packaged-worker-result/v1",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
    }
    with pytest.raises(ModalPackagedReadUnavailable, match="call_failed"):
        reader.observe(request)


@pytest.mark.parametrize("stage", sorted(host_reader_module._FIXED_WORKER_FAILURE_STAGES))
def test_v2_worker_failure_reports_only_fixed_stage(stage):
    reader, request, _ = _reader("failed")
    reader._facade.result = {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": stage,
    }
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        reader.observe(request)
    assert str(error.value) == f"modal_packaged_call_failed_{stage}"


@pytest.mark.parametrize("stage", (
    "SFT_TRAINER_CHILD_EXEC_TORCH_OOM",
    "SFT_TRAINER_CHILD_EXEC_TORCH_OOM_TRAIN_CALL",
    "SFT_TRAINER_CHILD_EXEC_OTHER_TRAIN_CALL",
))
@pytest.mark.parametrize("change", (
    {"status_code": "completed"}, {"completion_sha256": "a" * 64},
    {"extra": "PRIVATE_SENTINEL"}, {"effect_id": "private-effect"},
    {"schema_version": "synaptic-modal-packaged-worker-result/v1"},
    {"failure_stage": "SFT_TRAINER_CHILD_EXEC_TORCH_OOM_TRAIN_CALL_EXTRA"},
    {"failure_stage": "SFT_TRAINER_CHILD_EXEC_OTHER_UNKNOWN"},
))
def test_new_diagnostic_near_matches_remain_unknown_through_observe(stage, change):
    from tuner.execution.providers.modal.packaged_worker import packaged_worker_failure
    reader, request, _ = _reader("failed")
    result = packaged_worker_failure(stage)
    result.update(change)
    reader._facade.result = result
    with pytest.raises(ModalPackagedReadUnavailable) as caught:
        reader.observe(request)
    assert caught.value.args == ("modal_packaged_call_unknown",)


@pytest.mark.parametrize("change", [
    {"effect_id": "other"}, {"status_code": "completed"},
    {"completion_sha256": "a" * 64}, {"schema_version": "other"},
    {"failure_stage": "SFT_OTHER"}, {"failure_stage": "sft_trainer"},
    {"failure_stage": "ENTRYPOINT_MOUNT_CONTROL_DIR_EXTRA"},
    {"failure_stage": True}, {"extra": "untrusted"},
])
def test_v2_near_miss_worker_failures_remain_unknown(change):
    reader, _request, _ = _reader("failed")
    result = {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
        "failure_stage": "SFT_TRAINER",
    }
    result.update(change)
    reader._facade.result = result
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        reader._poll_packaged_call(reader.binding, "fc-1")
    assert str(error.value) == "modal_packaged_call_unknown"


@pytest.mark.parametrize("change", [
    {"effect_id": "other"},
    {"status_code": "completed"},
    {"completion_sha256": "a" * 64},
    {"schema_version": "other"},
    {"extra": "untrusted"},
])
def test_near_miss_worker_failures_remain_unknown(change):
    reader, _request, _ = _reader("failed")
    result = {
        "schema_version": "synaptic-modal-packaged-worker-result/v1",
        "effect_id": "unavailable",
        "status_code": "failed",
        "completion_sha256": "0" * 64,
    }
    result.update(change)
    reader._facade.result = result
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        reader._poll_packaged_call(reader.binding, "fc-1")
    assert str(error.value) == "modal_packaged_call_unknown"


def test_other_failed_result_with_bound_effect_id_remains_unknown():
    reader, _request, _ = _reader("failed")
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        reader._poll_packaged_call(reader.binding, "fc-1")
    assert str(error.value) == "modal_packaged_call_unknown"


def test_blocked_sdk_hydration_has_bounded_fixed_read_failure():
    reader, _request, _ = _reader(ModalFunctionCallState.RETURNED)
    release = Event()

    class BlockedCall:
        object_id = "fc-1"
        is_hydrated = False

        def hydrate(self, _client):
            release.wait(5)
            return self

    reader._facade.sdk.FunctionCall = SimpleNamespace(
        from_id=lambda *_args, **_kwargs: BlockedCall(),
    )
    started = time.monotonic()
    try:
        with pytest.raises(ModalPackagedReadUnavailable) as error:
            reader._poll_packaged_call(
                reader.binding, "fc-1", deadline=started + 0.05,
            )
        assert time.monotonic() - started < 1
        assert str(error.value) == "modal_packaged_call_unknown"
    finally:
        release.set()


def test_artifact_stream_sanitizes_provider_iterator_exception():
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    manifest = reader.artifacts(request)

    def broken(*_args, **_kwargs):
        yield b"data"
        raise RuntimeError("private-token-and-volume-path")

    reader._reader.iter_artifact = broken
    with pytest.raises(ModalPackagedReadUnavailable) as error:
        b"".join(reader.iter_artifact_bytes(
            request, manifest, "final_model", maximum_bytes=4,
        ))
    assert str(error.value) == "modal_packaged_artifact_stream_unavailable"
    assert "private-token" not in str(error.value)


def test_blocked_artifact_stream_has_bounded_fixed_read_failure(monkeypatch):
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    manifest = reader.artifacts(request)
    release = Event()
    monkeypatch.setattr(host_reader_module, "_ARTIFACT_READ_STEP_SECONDS", 0.05)

    def blocked(*_args, **_kwargs):
        release.wait(5)
        yield b"data"

    reader._reader.iter_artifact = blocked
    started = time.monotonic()
    try:
        with pytest.raises(ModalPackagedReadUnavailable) as error:
            b"".join(reader.iter_artifact_bytes(
                request, manifest, "final_model", maximum_bytes=4,
            ))
        assert time.monotonic() - started < 1
        assert str(error.value) == "modal_packaged_artifact_stream_unavailable"
    finally:
        release.set()


def test_reader_authenticates_retained_foundation_assessment_after_clock_advance():
    binding, _, _, _ = _case()
    command = binding.command
    intent = EffectIntentV1.from_command_bytes(command.canonical_bytes)
    preparation = command.preparation
    ref = ScopedProviderRunRefV1(
        preparation.provider.provider_id, preparation.provider.profile_ref,
        preparation.scope.account_ref, preparation.scope.namespace_ref, "fc-1",
    )
    current = record(intent, EffectState.FOUND, ref)
    retained = assessment(current)
    derived_binding, outcome, provider_run = _derive_foundation(
        intent, current, retained, FoundationAuth(), AssessmentAuth(), None,
    )
    run = TrainingRunRef(preparation.run_id, preparation.project_ref)
    document = {
        "schema_version": "synaptic-provider-run-read-request/v1",
        "purpose": "observe", "source_workflow_record_digest": "d" * 64,
        "source_revision": 1, "run": run.to_dict(),
        "provider_run_binding_digest": provider_run.binding_digest,
        "submit_command_bytes_digest": domain_digest(
            "synaptic-foundation-command-bytes/v1", command.canonical_bytes,
        ),
        "foundation_record_digest": current.record_digest,
        "assessment_digest": retained.authenticated_assessment_digest,
        "foundation_binding_digest": derived_binding.binding_digest,
        "foundation_outcome_digest": outcome.outcome_digest,
        "found_receipt_digest": provider_run.authenticated_receipt_digest,
    }
    raw = canonical_bytes(document)
    request = ProviderRunReadRequestV1(
        ProviderReadPurposeV1.OBSERVE, "d" * 64, 1, run,
        provider_run, command.canonical_bytes, current, retained,
        derived_binding, outcome, provider_run.authenticated_receipt_digest,
        raw, domain_digest("synaptic-provider-run-read-request/v1", raw),
    )

    class Catalog:
        def resolve(self, digest):
            return binding if digest == binding.command_digest else None

    class Authority:
        def authenticate(self, candidate):
            return candidate == binding

    class Repository:
        def get(self, effect_id):
            return current if effect_id == intent.effect_id else None

    reader = ModalPackagedCoordinatorReaderV1(
        bindings=Catalog(), binding_authority=Authority(),
        foundation_repository=Repository(),
        foundation_authenticator=FoundationAuth(),
        assessment_authenticator=AssessmentAuth(),
        packaged_reader=None, facade=None, evidence_authority=None,
        clock=SimpleNamespace(now=lambda: "2026-10-01T00:00:00Z"),
    )
    assert reader._binding(request, ProviderReadPurposeV1.OBSERVE) == binding


@pytest.mark.parametrize('phase', [WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.RUNNING, WorkflowPhaseV1.CANCEL_REQUESTED])
def test_pending_preserves_authenticated_source_phase_through_reducer(monkeypatch, phase):
    from tests.execution.coordinator_v1 import test_state_machine as state
    from tuner.execution.coordinator_v1.state_machine import apply_provider_observation, record_cancel_intent
    from tuner.execution.foundation_v2.references import CancellationRefV1, ProviderRunRefV1
    from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
    current, foundation = state.queued_evidence()
    if phase is WorkflowPhaseV1.RUNNING:
        request, observed = state.observation(current, foundation, ProviderRunPhaseV1.RUNNING, {'phase': 'fixture-running'})
        current = apply_provider_observation(current, request, observed, state.ObservationAuth())
    elif phase is WorkflowPhaseV1.CANCEL_REQUESTED:
        raw = state.intent('cancel')
        pending = record_cancel_intent(current, raw)
        cancellation = CancellationRefV1(ProviderRunRefV1(state.PROVIDER_RUN.provider_job_ref), state.D[14])
        current = state.apply_cancel_effect_record(pending, state.record(raw, state.EffectState.FOUND, cancellation), state.Auth())
    request, _ = state.observation(current, foundation, ProviderRunPhaseV1.QUEUED, {'fixture': 'request'})
    reader, _, evidence = _reader(ModalFunctionCallState.PENDING)
    monkeypatch.setattr(reader, '_source_workflow', lambda selected: current)
    polls = []
    monkeypatch.setattr(reader, '_poll_packaged_call', lambda *args: (polls.append(1) or ModalFunctionCallState.PENDING, None))
    observed = reader.observe(request)
    assert evidence.authenticate_observation(observed)
    assert apply_provider_observation(current, request, observed, state.ObservationAuth()) is current
    from tuner.execution.foundation_v2.canonical import parse_canonical_object
    document = parse_canonical_object(observed.content.canonical_evidence, name='pending hint')
    assert document['retained_workflow_phase'] == phase.value
    assert document['phase_basis'] == 'preserved_authenticated_workflow'
    assert document['status'] == 'no_completion_observed'
    assert polls == [1]


def test_source_workflow_mismatch_or_unbound_reader_refuses_before_poll(monkeypatch):
    from dataclasses import replace
    from tests.execution.coordinator_v1 import test_state_machine as state
    from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
    current, foundation = state.queued_evidence()
    request, _ = state.observation(current, foundation, ProviderRunPhaseV1.QUEUED, {'fixture': 'request'})
    reader, _, _ = _reader(ModalFunctionCallState.PENDING)
    monkeypatch.setattr(reader, '_source_workflow', ModalPackagedCoordinatorReaderV1._source_workflow.__get__(reader))
    monkeypatch.setattr(reader, '_poll_packaged_call', lambda *args: pytest.fail('polled without exact source workflow'))
    with pytest.raises(ModalPackagedReadUnavailable, match='workflow_unavailable'):
        reader.observe(request)
    reader.bind_workflows(SimpleNamespace(get=lambda run: replace(current, revision=current.revision + 1)))
    with pytest.raises(ModalPackagedReadUnavailable, match='workflow_mismatch'):
        reader.observe(request)


def test_source_revision_change_during_pending_poll_is_closed(monkeypatch):
    from dataclasses import replace
    from tests.execution.coordinator_v1 import test_state_machine as state
    from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
    current, foundation = state.queued_evidence()
    request, _ = state.observation(current, foundation, ProviderRunPhaseV1.QUEUED, {'fixture': 'request'})
    reader, _, _ = _reader(ModalFunctionCallState.PENDING)
    selected = [current]
    reader.bind_workflows(SimpleNamespace(get=lambda run: selected[0]))
    monkeypatch.setattr(reader, '_source_workflow', ModalPackagedCoordinatorReaderV1._source_workflow.__get__(reader))
    def poll(*args):
        selected[0] = replace(current, revision=current.revision + 1)
        return ModalFunctionCallState.PENDING, None
    monkeypatch.setattr(reader, '_poll_packaged_call', poll)
    with pytest.raises(ModalPackagedReadUnavailable, match='workflow_mismatch'):
        reader.observe(request)


@pytest.mark.parametrize('returned', ['none', 'timeout_subclass'])
def test_no_result_and_timeout_subclass_do_not_mean_pending(monkeypatch, returned):
    reader, _, _ = _reader(ModalFunctionCallState.RETURNED)
    call = reader._facade.sdk.FunctionCall.from_id('fc-1', client=reader._facade.client)
    class TransportTimeout(TimeoutError):
        pass
    def get(**kwargs):
        if returned == 'timeout_subclass':
            raise TransportTimeout('controlled timeout')
        return None
    monkeypatch.setattr(call, 'get', get)
    monkeypatch.setattr(reader._facade.sdk.FunctionCall, 'from_id', lambda *args, **kwargs: call)
    with pytest.raises(ModalPackagedReadUnavailable, match='call_unknown'):
        reader._poll_packaged_call(reader.binding, 'fc-1')


@pytest.mark.parametrize("stage", (
    "GRPO_ADMISSION", "GRPO_REVALIDATION", "GRPO_PREPARATION", "GRPO_INVOCATION",
    "GRPO_TRAINER", "GRPO_CHILD_DEADLINE", "GRPO_CLEANUP_UNCERTAIN",
    "GRPO_ARTIFACT", "GRPO_UNKNOWN",
))
def test_actual_grpo_worker_failure_projects_only_closed_stage(stage):
    from tuner.execution.providers.modal.packaged_worker import packaged_worker_failure

    reader, request, _ = _reader("failed")
    reader._facade.result = packaged_worker_failure(stage)
    with pytest.raises(ModalPackagedReadUnavailable) as caught:
        reader.observe(request)
    assert caught.value.args == (f"modal_packaged_call_failed_{stage}",)


@pytest.mark.parametrize("stage", ("GRPO_TRAINER_PRIVATE_SENTINEL", "GRPO_private credential", True))
def test_unknown_grpo_stage_never_enters_public_failure(stage):
    from tuner.execution.providers.modal.packaged_worker import packaged_worker_failure

    issued = packaged_worker_failure(stage)
    assert issued["failure_stage"] == "SFT_UNKNOWN"
    assert "PRIVATE_SENTINEL" not in str(issued) and "credential" not in str(issued)
    reader, request, _ = _reader("failed")
    reader._facade.result = {
        "schema_version": "synaptic-modal-packaged-worker-result/v2",
        "effect_id": "unavailable", "status_code": "failed",
        "completion_sha256": "0" * 64, "failure_stage": stage,
    }
    with pytest.raises(ModalPackagedReadUnavailable) as caught:
        reader.observe(request)
    assert caught.value.args == ("modal_packaged_call_unknown",)
