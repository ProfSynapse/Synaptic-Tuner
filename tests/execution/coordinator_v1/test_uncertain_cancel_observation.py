"""Known terminal run truth does not repair an uncertain cancellation receipt."""

import pytest

from synaptic_tuner.api.v1.reference.repositories import DurableWorkflowStoreV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.execution.coordinator_v1.coordinator import (
    RecordCancelIntentTransitionV1,
    ApplyCancelEffectTransitionV1,
)
from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
from tuner.execution.coordinator_v1.state_machine import (
    record_cancel_intent,
    apply_cancel_effect_record,
)
from tuner.execution.foundation_v2.repository import EffectState
from . import test_operational_cas as fixtures
from .test_operations_service import operations, Reader
from .test_state_machine import (
    Auth,
    AssessmentAuth,
    ObservationAuth,
    Verifier,
    planned,
    intent,
    record,
    assessment,
    observation,
)


def _repository(storage):
    return DurableWorkflowStoreV1(
        storage,
        storage,
        foundation_authenticator=Auth(),
        assessment_authenticator=AssessmentAuth(),
        observation_authenticator=ObservationAuth(),
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
def test_cold_sqlite_outcome_preserves_uncertain_cancel_and_observes_terminal(
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
        cancel_record = record(raw, EffectState.INDETERMINATE)
        cancel_assessment = assessment(cancel_record)
        uncertain = apply_cancel_effect_record(
            pending, cancel_record, cancel_assessment, Auth(), AssessmentAuth()
        )
        assert store.compare_and_swap(
            pending,
            uncertain,
            transition=ApplyCancelEffectTransitionV1(cancel_record, cancel_assessment),
        )
    with SQLiteDurableStoreV1(path, create=False) as storage:
        repository = _repository(storage)
        source = repository.get(queued.run)
        assert source == uncertain

        class Observer(Reader):
            selected = ProviderRunPhaseV1.RUNNING

            def observe(self, request):
                self.observe_calls += 1
                assert request.provider_run == uncertain.provider_run_ref
                diagnostic = (
                    "provider_run_failed"
                    if self.selected is ProviderRunPhaseV1.FAILED
                    else None
                )
                return observation(
                    repository.get(queued.run),
                    foundation,
                    self.selected,
                    {"phase": self.selected.value},
                    diagnostic,
                )[1]

        observer = Observer()
        service, _ = operations(repository, source, foundation, reader=observer)
        api = RunsAPI(service)
        assert api.outcome(queued.run).state.value == "reconcile_required"
        assert repository.get(queued.run) == uncertain
        observer.selected = phase
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
        assert _repository(storage).get(queued.run) == terminal
