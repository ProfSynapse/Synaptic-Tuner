"""Scripted public cancellation; production capabilities deliberately remain closed."""
from dataclasses import replace
from datetime import datetime, timezone
import sys
from threading import Event

import pytest

from synaptic_tuner.api.v1.reference.training import (
    ReferenceAuthorizationError, ReferenceAuthorizationV1, _commitment_document,
)
from synaptic_tuner.api.v1.runs_facade import RunOperationError
from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
from tuner.training import modal_host_composition as composition
from tuner.training import modal_standalone_runner as runner
from tuner.training.modal_host_reader import ModalPackagedCoordinatorReaderV1, ModalPackagedReadUnavailable
from tuner.execution.providers.modal import packaged_transport as transport
from tests.training.test_modal_standalone_runner import _setup


def _scripted(tmp_path, monkeypatch, action, *, opt_in=True, capabilities=True, cancel=None, after=None):
    plan, context, events = _setup(tmp_path, monkeypatch)
    epoch = runner.UTCClock().now_epoch()
    class Clock:
        offset = 0
        def now_epoch(self):
            return epoch + self.offset
        def now(self):
            return datetime.fromtimestamp(self.now_epoch(), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        now_iso = now
    clock = Clock()
    monkeypatch.setattr(runner, "UTCClock", lambda: clock)
    descriptor = composition._modal_packaged_descriptor
    if capabilities:
        def scripted_descriptor(**kwargs):
            original = descriptor(**kwargs)
            return replace(original, capabilities=replace(original.capabilities, cancel=True, reconcile=True))
        monkeypatch.setattr(composition, "_modal_packaged_descriptor", scripted_descriptor)
    generic = composition.compose_reference_host
    generic_options = []
    def scripted_generic(**kwargs):
        generic_options.append(kwargs)
        return generic(**kwargs, allow_current_cancel_authorization=opt_in)
    monkeypatch.setattr(composition, "compose_reference_host", scripted_generic)
    captured = []
    original_compose = runner.compose_modal_packaged_reference_host
    def capture(**kwargs):
        host = original_compose(**kwargs)
        captured.append((host, kwargs["run"]))
        return host
    monkeypatch.setattr(runner, "compose_modal_packaged_reference_host", capture)
    sdk = sys.modules["modal"]
    calls = []
    class Call:
        object_id = "fc-owned"
        is_hydrated = True
        def __init__(self, client):
            self.client = client
        def hydrate(self, client):
            assert self.client is client
            return self
        def cancel(self, *, terminate_containers):
            assert terminate_containers is True
            calls.append(self.object_id)
            if cancel:
                cancel()
    class Calls:
        @staticmethod
        def from_id(reference, *, client):
            assert reference == "fc-owned"
            return Call(client)
    monkeypatch.setattr(sdk, "FunctionCall", Calls, raising=False)
    original_poll = ModalPackagedCoordinatorReaderV1._poll_packaged_call
    errors = []
    executed = []
    def poll(self, binding, reference, **kwargs):
        if executed:
            return original_poll(self, binding, reference, **kwargs)
        executed.append(True)
        try:
            host, run = captured[0]
            action(host, run, clock, calls, events)
        except BaseException as error:
            errors.append(error)
        raise ModalPackagedReadUnavailable("modal_packaged_call_read_unavailable")
    monkeypatch.setattr(ModalPackagedCoordinatorReaderV1, "_poll_packaged_call", poll)
    with pytest.raises(runner.ModalStandalonePhaseUnavailable) as stopped:
        runner.run_modal_standalone_job(plan=plan, context=context, modal_profile="fixture", modal_environment="main")
    assert stopped.value.phase == "RUN_CALL_OBSERVE"
    assert executed
    if errors:
        raise errors[0]
    if after:
        after(captured[0][1], generic_options[0], calls)
    assert events == ["bootstrap", "cpu", "stage", "spawn"]


def test_public_cancel_after_start_expiry_preserves_original_commitment(tmp_path, monkeypatch):
    def action(host, run_ref, clock, calls, events):
        workflow = host.composition.stores.workflow_store.get(run_ref)
        assert workflow is not None
        before = host.composition.coordinator._authorization._commitment(workflow.preflight_digest)
        original_bytes = _commitment_document(before)
        clock.offset = 901
        result = host.api.runs.cancel(run_ref, "operator requested stop")
        assert result.state.value == "cancel_requested"
        assert calls == ["fc-owned"]
        after = host.composition.coordinator._authorization._commitment(workflow.preflight_digest)
        assert after == before
        assert _commitment_document(after) == original_bytes
        current = host.composition.stores.workflow_store.get(run_ref)
        record = host.composition.foundation.get(current.cancel.effect_id)
        assert record.grant.content.policy_digest == before.policy_digest
        assert record.grant.content.requirement_digest == before.requirement_digest
        assert record.grant.content.not_before_epoch == clock.now_epoch()
        assert record.grant.content.expires_at_epoch == clock.now_epoch() + 900
        for intent in (workflow.stage, workflow.submit):
            with pytest.raises(ReferenceAuthorizationError, match="authorization_expired"):
                host.composition.coordinator._authorization.issue_effect_grant(
                    intent.canonical_command_bytes,
                    preflight_digest=workflow.preflight_digest, now_epoch=clock.now_epoch(),
                )
        def forbidden(*args, **kwargs):
            pytest.fail("acknowledged cancellation must not renew permission or binding")
        monkeypatch.setattr(host.effects, "authorize", forbidden)
        monkeypatch.setattr(host.effects, "bind", forbidden)
        assert host.api.runs.cancel(run_ref, "different reason").state.value == "cancel_requested"
        assert calls == ["fc-owned"]
    _scripted(tmp_path, monkeypatch, action)


def test_default_capabilities_remain_closed(tmp_path, monkeypatch):
    def action(host, run, clock, calls, events):
        with pytest.raises(RunOperationError):
            host.api.runs.cancel(run, "stop")
        with pytest.raises(RunOperationError):
            host.api.runs.reconcile(run)
        assert calls == []
        assert host.composition.stores.workflow_store.get(run).phase is WorkflowPhaseV1.QUEUED
    _scripted(tmp_path, monkeypatch, action, capabilities=False)


def test_no_opt_in_preserves_expired_start_authorization(tmp_path, monkeypatch):
    def action(host, run, clock, calls, events):
        clock.offset = 901
        with pytest.raises(RunOperationError):
            host.api.runs.cancel(run, "stop")
        assert calls == []
        current = host.composition.stores.workflow_store.get(run)
        assert current.phase is WorkflowPhaseV1.CANCEL_INTENT_RECORDED
        assert host.effects._storage.attempts.resolve(current.cancel.command_digest) is None
    _scripted(tmp_path, monkeypatch, action, opt_in=False)


def test_lost_cancel_reply_consumes_attempt_and_reconcile_only_looks_up(tmp_path, monkeypatch):
    def lost_reply():
        raise RuntimeError("controlled response loss")
    def action(host, run, clock, calls, events):
        clock.offset = 901
        original = host.composition.stores.workflow_store.get(run)
        result = host.api.runs.cancel(run, "stop")
        assert result.state.value == "reconcile_required"
        current = host.composition.stores.workflow_store.get(run)
        assert current.phase is WorkflowPhaseV1.CANCEL_RECONCILE_REQUIRED
        assert host.effects._storage.attempts.resolve(current.cancel.command_digest) is not None
        assert calls == ["fc-owned"]
        host.api.runs.cancel(run, "another reason")
        assert calls == ["fc-owned"]
        looked_up = host.api.runs.reconcile(run)
        assert looked_up.state.value == "reconcile_required"
        assert calls == ["fc-owned"]
        assert host.composition.stores.workflow_store.get(run).submit == original.submit
    _scripted(tmp_path, monkeypatch, action, cancel=lost_reply)


def test_reopened_workflow_cancel_does_not_renew_permission_or_repeat_rpc(tmp_path, monkeypatch):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from synaptic_tuner.api.v1.reference.composition import compose_reference_host
    from tuner.training.modal_host_storage import ModalHostStorageV1, ModalHostAttemptAlreadyClaimed
    retained = {}
    def census(storage):
        return tuple(storage._connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
                     for table in ("attempts", "catalogs"))
    def lost_reply():
        raise RuntimeError("controlled response loss")
    def action(host, run, clock, calls, events):
        assert host.api.runs.cancel(run, "stop").state.value == "reconcile_required"
        storage = host.effects._storage
        retained["path"] = storage._connection.execute("PRAGMA database_list").fetchone()[2]
        retained["namespace"] = storage.namespace_ref
        current = host.composition.stores.workflow_store.get(run)
        retained["command"] = current.cancel.command_digest
        retained["evidence"] = storage.attempts.resolve(current.cancel.command_digest)
        retained["census"] = census(storage)
    def after(run, options, calls):
        old_store = options["ports"].records
        with pytest.raises(RuntimeError, match="closed"):
            old_store.read(partition="authorization", key="missing")
        def no_permission(*args, **kwargs):
            pytest.fail("repeat cancellation must not authorize or bind")
        monkeypatch.setattr(options["ports"].grants, "authorize", no_permission)
        monkeypatch.setattr(options["ports"].grants, "bind", no_permission)
        # The same scripted provider family and original authenticators remain;
        # only the workflow DB and reference host are genuinely reopened here.
        with SQLiteDurableStoreV1(old_store._path, create=False) as store:
            ports = replace(options["ports"], records=store, streams=store)
            cold = compose_reference_host(**{**options, "ports": ports}, allow_current_cancel_authorization=True)
            assert cold.api().runs.show(run).state.value == "reconcile_required"
            assert cold.api().runs.cancel(run, "stop again").state.value == "reconcile_required"
            assert calls == ["fc-owned"]
        from pathlib import Path
        with ModalHostStorageV1(Path(retained["path"]), retained["namespace"], create=False) as journal:
            assert journal.attempts.resolve(retained["command"]) == retained["evidence"]
            with pytest.raises(ModalHostAttemptAlreadyClaimed):
                journal.attempts.claim(retained["command"], retained["evidence"])
            assert census(journal) == retained["census"]
    _scripted(tmp_path, monkeypatch, action, cancel=lost_reply, after=after)


def test_bounded_cancel_late_ack_stays_uncertain_without_replay(tmp_path, monkeypatch):
    entered, release, exited = Event(), Event(), Event()
    bounded = transport._bounded_provider_call
    def short(operation, **kwargs):
        return bounded(operation, timeout_seconds=0.05)
    def held_cancel():
        entered.set()
        assert release.wait(5)
        exited.set()
    def action(host, run, clock, calls, events):
        monkeypatch.setattr(transport, "_bounded_provider_call", short)
        try:
            result = host.api.runs.cancel(run, "stop")
            assert entered.is_set()
            assert result.state.value == "reconcile_required"
            current = host.composition.stores.workflow_store.get(run)
            assert host.effects._storage.attempts.resolve(current.cancel.command_digest) is not None
            assert host.api.runs.cancel(run, "stop again").state.value == "reconcile_required"
        finally:
            release.set()
            assert exited.wait(5)
        assert host.api.runs.reconcile(run).state.value == "reconcile_required"
        assert calls == ["fc-owned"]
    _scripted(tmp_path, monkeypatch, action, cancel=held_cancel)


@pytest.mark.parametrize("fault", ["call_catalog", "foundation", "submit_binding", "workflow_intent"])
def test_cancel_original_target_fault_rejects_before_sdk_or_claim(tmp_path, monkeypatch, fault):
    def action(host, run, clock, calls, events):
        effects = host.effects
        if fault == "call_catalog":
            monkeypatch.setattr(effects.calls, "resolve", lambda digest: "fc-other")
        elif fault == "submit_binding":
            monkeypatch.setattr(effects.bindings, "resolve", lambda digest: None)
        elif fault == "foundation":
            monkeypatch.setattr(host.composition.foundation, "get", lambda effect_id: None)
        else:
            original = effects._cancel_target
            def changed(command):
                stores = effects._cancel_stores
                class ChangedWorkflow:
                    def get(self, ref):
                        workflow = stores[0].get(ref)
                        return replace(workflow, cancel=None)
                effects._cancel_stores = (ChangedWorkflow(), *stores[1:])
                try:
                    original(command)
                finally:
                    effects._cancel_stores = stores
            monkeypatch.setattr(effects, "_cancel_target", changed)
        with pytest.raises(RunOperationError):
            host.api.runs.cancel(run, "stop")
        current = host.composition.stores.workflow_store.get(run)
        assert current.phase is WorkflowPhaseV1.CANCEL_INTENT_RECORDED
        assert calls == []
        assert effects._storage.attempts.resolve(current.cancel.command_digest) is None
    _scripted(tmp_path, monkeypatch, action)


def test_completion_racing_acknowledged_cancel_remains_successful(tmp_path, monkeypatch):
    def action(host, run, clock, calls, events):
        assert host.api.runs.cancel(run, "stop").state.value == "cancel_requested"
        assert host.api.runs.outcome(run).state.value == "succeeded"
        with pytest.raises(RunOperationError):
            host.api.runs.cancel(run, "too late")
        assert calls == ["fc-owned"]
    _scripted(tmp_path, monkeypatch, action)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_cancel_authorization_opt_in_requires_exact_bool(value):
    from synaptic_tuner.api.v1.reference import compose_reference_authority
    from synaptic_tuner.api.v1.secrets import SecretRef
    from tuner.training.modal_host_authority import UTCClock
    class Secrets:
        def resolve(self, ref):
            assert ref == SecretRef("env", "test")
            return "host-only-test-authority-value-" + "x" * 40
    authority = compose_reference_authority(clock=UTCClock(), secrets=Secrets(), authority_secret=SecretRef("env", "test"))
    with pytest.raises(TypeError, match="cancellation opt-in"):
        ReferenceAuthorizationV1(records=None, grants=None, authority=authority.grant_authority,
            clock=authority.clock, allow_current_cancel_authorization=value)
