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
