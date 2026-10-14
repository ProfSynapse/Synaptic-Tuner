"""Durable replay retains an unadvanced cancellation during independent reads."""

from dataclasses import replace
import pytest

from synaptic_tuner.api.v1.reference.repositories import DurableWorkflowStoreV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.runs_facade import RunsAPI, RunOperationError
from tuner.execution.coordinator_v1.coordinator import (
    RecordCancelIntentTransitionV1,
)
from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
from tuner.execution.coordinator_v1.stores import CoordinatorStoreError
from tuner.execution.coordinator_v1.state_machine import (
    record_cancel_intent,
)
from . import test_operational_cas as fixtures
from .test_operations_service import operations, Reader
from .test_state_machine import (
    Auth,
    AssessmentAuth,
    Verifier,
    planned,
    intent,
    observation,
)


from tuner.training.modal_host_authority import (
    HMACAuthenticator,
    ReaderEvidenceAuthority,
    ObservationAuthenticator,
)


def _evidence(key=b"deterministic-test-only-key-000000"):
    return ReaderEvidenceAuthority(
        "authority-a",
        "key-a",
        HMACAuthenticator(
            {"key-a": key}, allowed_purposes=frozenset({"provider-run-observation/v1"})
        ),
    )


def _repository(storage, *, evidence=None):
    return DurableWorkflowStoreV1(
        storage,
        storage,
        foundation_authenticator=Auth(),
        assessment_authenticator=AssessmentAuth(),
        observation_authenticator=ObservationAuthenticator(evidence or _evidence()),
        artifact_verifier=Verifier(),
    )


@pytest.mark.parametrize(
    "phase,state",
    [
        (ProviderRunPhaseV1.SUCCEEDED, "succeeded"),
        (ProviderRunPhaseV1.CANCELLED, "cancelled"),
        (ProviderRunPhaseV1.FAILED, "failed"),
    ],
)
def test_cold_sqlite_outcome_preserves_orphaned_intent_and_observes_terminal(
    tmp_path, monkeypatch, phase, state
):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "workflow.sqlite3"
    with SQLiteDurableStoreV1(path) as storage:
        repository = _repository(storage)
        assert repository.create(planned())
        monkeypatch.setattr(fixtures, "_store", lambda **kwargs: repository)
        store, queued, foundation = fixtures._queued_store()
        raw = intent("cancel")
        pending = record_cancel_intent(queued, raw)
        assert store.compare_and_swap(
            queued, pending, transition=RecordCancelIntentTransitionV1(raw)
        )
        uncertain = pending
    with SQLiteDurableStoreV1(path, create=False) as storage:
        repository = _repository(storage)
        source = repository.get(queued.run)
        assert source == uncertain

        class Observer(Reader):
            selected = ProviderRunPhaseV1.RUNNING
            fault = None

            def observe(self, request):
                self.observe_calls += 1
                assert request.provider_run == uncertain.provider_run_ref
                diagnostic = (
                    "provider_run_failed"
                    if self.selected is ProviderRunPhaseV1.FAILED
                    else None
                )
                envelope = observation(
                    repository.get(queued.run),
                    foundation,
                    self.selected,
                    {"phase": self.selected.value},
                    diagnostic,
                )[1]
                content = envelope.content
                if self.fault == "scope":
                    content = replace(content, provider_job_ref="other-job")
                signed = _evidence().observation(content)
                if self.fault == "key":
                    signed = _evidence(
                        b"different-test-only-key-000000000"
                    ).observation(content)
                return signed

        observer = Observer()
        service, _ = operations(repository, source, foundation, reader=observer)
        service._observation_authenticator = ObservationAuthenticator(_evidence())
        api = RunsAPI(service)
        for fault in ("key", "scope"):
            observer.fault = fault
            with pytest.raises(RunOperationError) as rejected:
                api.outcome(queued.run)
            assert rejected.value.code.value == "provider_read_invalid"
            assert repository.get(queued.run) == uncertain
        observer.fault = None
        assert api.outcome(queued.run).state.value == "cancel_requested"
        assert repository.get(queued.run) == uncertain
        observer.selected = phase
        with monkeypatch.context() as conflict:
            conflict.setattr(repository, "compare_and_swap", lambda *a, **kw: False)
            with pytest.raises(RunOperationError) as rejected:
                api.outcome(queued.run)
            assert rejected.value.code.value == "state_conflict"
            assert repository.get(queued.run) == uncertain
        assert api.outcome(queued.run).state.value == state
        terminal = repository.get(queued.run)
        assert (
            terminal.cancel == uncertain.cancel and terminal.submit == uncertain.submit
        )
        assert terminal.bound_cancellation is None
        assert (
            service._coordinator.cancel_calls
            == service._coordinator.reconcile_calls
            == 0
        )
    with SQLiteDurableStoreV1(path, create=False) as storage:
        with pytest.raises(CoordinatorStoreError):
            _repository(
                storage, evidence=_evidence(b"wrong-replay-test-only-key-000000")
            ).get(queued.run)
        assert _repository(storage).get(queued.run) == terminal

        assert not terminal.cancel.advanced
        assert (
            not terminal.cancel.foundation_bindings
            and not terminal.cancel.foundation_outcomes
        )
