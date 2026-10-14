"""Terminal provider statuses retain the existing authenticated reader boundary."""

from __future__ import annotations

import json
from threading import get_ident

import pytest

from tuner.execution.providers.modal.facade import ModalFunctionCallState
from tuner.training.modal_host_reader import ModalPackagedReadUnavailable
from tests.training.test_modal_host_reader import _reader


@pytest.mark.parametrize(
    "state,phase",
    [
        (ModalFunctionCallState.CANCELLED, "cancelled"),
        (ModalFunctionCallState.FAILED, "failed"),
    ],
)
def test_terminal_status_signs_only_closed_bound_evidence(state, phase, monkeypatch):
    reader, request, authority = _reader(ModalFunctionCallState.RETURNED)
    reader._facade.observe_known_call_status = lambda ref, **_: state
    monkeypatch.setattr(
        reader._facade.sdk.FunctionCall,
        "from_id",
        lambda *_a, **_k: pytest.fail("result deserialization"),
    )
    monkeypatch.setattr(
        reader, "_completion", lambda *_a, **_k: pytest.fail("native success invented")
    )
    observation = reader.observe(request)
    assert authority.authenticate_observation(observation)
    assert observation.content.phase.value == phase
    evidence = json.loads(observation.content.canonical_evidence)
    assert evidence["provider_job_ref"] == "fc-1"
    assert evidence["command_digest"] == reader.binding.command_digest
    assert (
        evidence["execution_binding_digest"]
        == reader.binding.execution_binding.binding_digest
    )
    assert evidence["basis"] == "provider_input_terminal_status"


def test_current_binding_rechecked_on_owner_thread_before_signing(monkeypatch):
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    reader._facade.observe_known_call_status = lambda *_a, **_k: (
        ModalFunctionCallState.CANCELLED
    )
    owner = get_ident()
    checks = []

    def binding(*_args):
        assert get_ident() == owner
        checks.append(1)
        if len(checks) == 2:
            raise ModalPackagedReadUnavailable("modal_packaged_binding_mismatch")
        return reader.binding

    monkeypatch.setattr(reader, "_binding", binding)
    with pytest.raises(ModalPackagedReadUnavailable):
        reader.observe(request)
    assert len(checks) == 2


def test_unknown_status_cannot_fall_back_to_worker_get(monkeypatch):
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    reader._facade.observe_known_call_status = lambda *_a, **_k: (
        ModalFunctionCallState.UNKNOWN
    )
    monkeypatch.setattr(
        reader._facade.sdk.FunctionCall,
        "from_id",
        lambda *_a, **_k: pytest.fail("fallback poll"),
    )
    with pytest.raises(
        ModalPackagedReadUnavailable, match="modal_packaged_call_unknown"
    ):
        reader.observe(request)


@pytest.mark.parametrize("state", ["cancelled", "failed", None])
def test_status_collaborator_requires_exact_internal_enum(state):
    reader, request, _ = _reader(ModalFunctionCallState.RETURNED)
    reader._facade.observe_known_call_status = lambda *_a, **_k: state
    with pytest.raises(
        ModalPackagedReadUnavailable, match="modal_packaged_call_unknown"
    ):
        reader.observe(request)


@pytest.mark.parametrize(
    "terminal", [ModalFunctionCallState.CANCELLED, ModalFunctionCallState.FAILED]
)
def test_public_cancel_ack_then_independent_terminal_observation(
    tmp_path, monkeypatch, terminal
):
    from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
    from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
    from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1
    from tests.training.test_modal_cancel_machinery import _scripted

    original_poll = ModalPackagedCoordinatorReaderV1._poll_packaged_call
    monkeypatch.setattr(
        ExplicitModal154ReadFacade,
        "observe_known_call_status",
        lambda *_a, **_k: terminal,
    )

    def action(host, run, clock, calls, events):
        assert (
            host.api.runs.cancel(run, "operator stop").state.value == "cancel_requested"
        )
        before = host.composition.stores.workflow_store.get(run)
        assert before.phase is WorkflowPhaseV1.CANCEL_REQUESTED
        original_cancel = before.cancel.to_dict()
        monkeypatch.setattr(
            ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", original_poll
        )
        result = host.api.runs.outcome(run)
        assert result.state.value == terminal.value
        after = host.composition.stores.workflow_store.get(run)
        assert after.cancel.to_dict() == original_cancel
        assert calls == ["fc-owned"]
        assert not result.artifacts

    _scripted(tmp_path, monkeypatch, action)
