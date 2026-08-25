"""P0 gates for the private durable job lifecycle candidate.

These tests deliberately keep both databases in a host-project-owned state
directory and model the engine as a forbidden nested checkout.  SQLite is an
implementation detail of the host control plane, never submodule-owned state.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from synaptic_tuner._next_api_v1 import _store as store_module
from synaptic_tuner._next_api_v1._authority import create_grant_registrar
from synaptic_tuner._next_api_v1._provider import (
    EffectKind,
    ExecutionScope,
    ProviderAuth,
    ProviderRunState,
)
from synaptic_tuner._next_api_v1._service import JobService, JobStoreUnconfigured
from synaptic_tuner._next_api_v1._store import (
    AccessDenied,
    CorruptStore,
    GrantBinding,
    GrantRejected,
    JobStore,
    JobStoreLocation,
    OperationConflict,
)
from synaptic_tuner._next_api_v1.execution import (
    AccessContext,
    AuthorizationRequirement,
    ExecutionGrant,
    LogCursor,
    RunRef,
    RunState,
)
from synaptic_tuner._next_api_v1.training import (
    AdapterSpec,
    ArtifactPolicy,
    DatasetSpec,
    ExecutionSpec,
    ModelSpec,
    TrainingMethod,
    TrainingParameters,
    TrainingPlan,
)

from ._fake_provider import FakeExecutionProvider, FakeFault


NOW = "2026-08-25T16:00:00Z"
LATER = "2026-08-25T17:00:00Z"
HEX_A = "a" * 64
HEX_B = "b" * 64
IMAGE_DIGEST = "sha256:" + "c" * 64


class SequentialIds:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, kind: str) -> str:
        with self._lock:
            value = self._counts.get(kind, 0) + 1
            self._counts[kind] = value
        return f"{kind}-{value:04d}"


@dataclass
class Harness:
    host_project: Path
    engine_checkout: Path
    state_root: Path
    store_path: Path
    fake_path: Path
    ids: SequentialIds
    access: AccessContext
    grant: ExecutionGrant
    plan: TrainingPlan
    binding: GrantBinding
    scope: ExecutionScope
    store: JobStore
    provider: FakeExecutionProvider
    service: JobService


def _plan() -> TrainingPlan:
    return TrainingPlan(
        method=TrainingMethod.SFT,
        model=ModelSpec("org/model", "1" * 40, "2" * 40),
        dataset=DatasetSpec("org/dataset", "3" * 40, file="train.jsonl"),
        parameters=TrainingParameters(max_steps=5),
        adapter=AdapterSpec(mode="lora", rank=16, alpha=32),
        execution=ExecutionSpec(
            provider="fake",
            accelerator="cpu",
            runtime_image="registry.example/trainer@" + IMAGE_DIGEST,
            runtime_image_digest=IMAGE_DIGEST,
            dependency_lock_digest=HEX_B,
            allowed_secret_refs=("secret://hf/token",),
        ),
        artifacts=ArtifactPolicy(),
        source_digest=HEX_A,
        workload_digest=HEX_B,
        artifact_slot_ref="artifact://runs/exclusive-slot",
        quote_ref="quote://fake/one",
        authorization=(AuthorizationRequirement("training.start", False),),
    )


def _binding(plan: TrainingPlan, scope: ExecutionScope, operation_key: str = "op-001") -> GrantBinding:
    return GrantBinding(
        operation_key=operation_key,
        scope=scope,
        plan_fingerprint=plan.fingerprint,
        source_digest=plan.source_digest,
        workload_digest=plan.workload_digest,
        artifact_slot_ref=plan.artifact_slot_ref,
        quote_digest=HEX_A,
        resource_digest=HEX_B,
        allowed_secret_refs_digest=HEX_A,
        issued_at=NOW,
        expires_at=LATER,
    )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    host = tmp_path / "main-project"
    engine = host / "vendor" / "toolset-training"
    state = host / ".synaptic" / "jobs"
    engine.mkdir(parents=True)
    state.mkdir(parents=True)
    ids = SequentialIds()
    access = AccessContext("principal://joseph", "project://alpha", "auth://session")
    grant = ExecutionGrant("grant://one")
    plan = _plan()
    scope = ExecutionScope("fake", "account://one", "namespace://alpha")
    binding = _binding(plan, scope)
    store_path = state / "jobs.sqlite3"
    fake_path = state / "fake-provider.sqlite3"
    location = JobStoreLocation(store_path, state, (engine,))
    store = JobStore(location, clock=lambda: NOW, id_factory=ids)
    provider = FakeExecutionProvider(fake_path, scope, clock=lambda: NOW, id_factory=ids)
    create_grant_registrar(store).register(access, grant, binding)
    service = JobService(store, {"fake": provider})
    return Harness(host, engine, state, store_path, fake_path, ids, access, grant,
                   plan, binding, scope, store, provider, service)


def _start(harness: Harness):
    return harness.service.start(
        harness.access, harness.plan, harness.grant, harness.binding,
        ProviderAuth(harness.scope, object()),
    )


def test_store_is_host_owned_and_creates_nothing_under_engine_checkout(harness: Harness) -> None:
    assert harness.store_path.is_relative_to(harness.host_project)
    assert not harness.store_path.is_relative_to(harness.engine_checkout)
    assert harness.store_path.exists()
    assert list(harness.engine_checkout.iterdir()) == []


def test_store_has_no_implicit_default_and_service_fails_closed_when_unconfigured(
    harness: Harness,
) -> None:
    with pytest.raises(TypeError):
        JobStore()  # type: ignore[call-arg]
    with pytest.raises(JobStoreUnconfigured, match="job_store_unconfigured"):
        JobService(None, {}).show(harness.access, RunRef("run-x", harness.access.project_ref))


def test_store_rejects_database_beneath_forbidden_engine_checkout(tmp_path: Path) -> None:
    host = tmp_path / "main-project"
    engine = host / "vendor" / "engine"
    engine.mkdir(parents=True)
    location = JobStoreLocation(engine / "jobs.sqlite3", host, (engine,))
    with pytest.raises(CorruptStore, match="outside engine code roots"):
        JobStore(location, clock=lambda: NOW, id_factory=SequentialIds())
    assert list(engine.iterdir()) == []


def test_store_refuses_corrupt_unknown_version_and_incomplete_schema(tmp_path: Path) -> None:
    host = tmp_path / "host"
    engine = host / "engine"
    state = host / "state"
    engine.mkdir(parents=True)
    state.mkdir()

    corrupt = state / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(CorruptStore):
        JobStore(JobStoreLocation(corrupt, state, (engine,)), clock=lambda: NOW,
                 id_factory=SequentialIds())
    assert not Path(str(corrupt) + "-wal").exists()
    assert not Path(str(corrupt) + "-shm").exists()

    wrong_version = state / "wrong-version.sqlite3"
    connection = sqlite3.connect(wrong_version)
    connection.execute(f"PRAGMA application_id={store_module.APPLICATION_ID}")
    connection.execute("PRAGMA user_version=1")
    connection.close()
    wrong_version_before = wrong_version.read_bytes()
    with pytest.raises(CorruptStore):
        JobStore(JobStoreLocation(wrong_version, state, (engine,)), clock=lambda: NOW,
                 id_factory=SequentialIds())
    assert wrong_version.read_bytes() == wrong_version_before
    assert not Path(str(wrong_version) + "-wal").exists()
    assert not Path(str(wrong_version) + "-shm").exists()

    incomplete = state / "incomplete.sqlite3"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE placeholder(value TEXT)")
    connection.execute(f"PRAGMA application_id={store_module.APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version={store_module.SCHEMA_VERSION}")
    connection.close()
    with pytest.raises(CorruptStore):
        JobStore(JobStoreLocation(incomplete, state, (engine,)), clock=lambda: NOW,
                 id_factory=SequentialIds())


def test_happy_path_persists_across_fresh_store_and_fake_database_is_separate(harness: Harness) -> None:
    result = _start(harness)
    assert result.state is RunState.SUBMITTED
    assert not result.replayed
    identity = harness.store.effect(harness.access, result.run, EffectKind.SUBMIT).identity
    assert harness.provider.mutation_count(identity) == 1
    assert harness.fake_path != harness.store_path

    fresh_store = JobStore(harness.store.location, clock=lambda: NOW, id_factory=harness.ids)
    fresh_service = JobService(fresh_store, {"fake": harness.provider})
    assert fresh_service.show(harness.access, result.run).state is RunState.SUBMITTED
    assert fresh_service.list(harness.access) == (result.run,)


def test_exact_replay_never_resubmits_and_conflicting_grant_is_rejected(harness: Harness) -> None:
    first = _start(harness)
    identity = harness.store.effect(harness.access, first.run, EffectKind.SUBMIT).identity
    replay = _start(harness)
    assert replay.run == first.run
    assert replay.replayed
    assert harness.provider.mutation_count(identity) == 1

    other_grant = ExecutionGrant("grant://two")
    create_grant_registrar(harness.store).register(harness.access, other_grant, harness.binding)
    with pytest.raises(OperationConflict, match="bound differently"):
        harness.service.start(
            harness.access, harness.plan, other_grant, harness.binding,
            ProviderAuth(harness.scope, object()),
        )
    assert harness.provider.mutation_count(identity) == 1


def test_grant_binding_lifetime_and_access_are_enforced(harness: Harness) -> None:
    wrong_access = AccessContext("principal://other", harness.access.project_ref, "auth://other")
    with pytest.raises(GrantRejected):
        harness.service.start(
            wrong_access, harness.plan, harness.grant, harness.binding, ProviderAuth(harness.scope, object())
        )

    expired_grant = ExecutionGrant("grant://expired")
    expired = replace(
        harness.binding, operation_key="op-expired",
        issued_at="2026-08-25T14:00:00Z", expires_at="2026-08-25T15:00:00Z",
    )
    create_grant_registrar(harness.store).register(harness.access, expired_grant, expired)
    with pytest.raises(GrantRejected, match="lifetime"):
        harness.service.start(
            harness.access, harness.plan, expired_grant, expired, ProviderAuth(harness.scope, object())
        )

    with pytest.raises(ValueError, match="does not match"):
        harness.service.start(
            harness.access, harness.plan, harness.grant,
            replace(harness.binding, plan_fingerprint="d" * 64), ProviderAuth(harness.scope, object())
        )


def test_owned_run_cannot_be_read_listed_or_logged_by_another_principal(harness: Harness) -> None:
    result = _start(harness)
    outsider = AccessContext("principal://other", harness.access.project_ref, "auth://other")
    with pytest.raises(AccessDenied):
        harness.service.show(outsider, result.run)
    with pytest.raises(AccessDenied):
        harness.service.logs(outsider, result.run)
    assert harness.service.list(outsider) == ()


def test_s1_claimed_without_started_recovers_as_definitively_absent(harness: Harness) -> None:
    claim = harness.store.claim_submission(
        harness.access, harness.grant, harness.binding,
        canonical_plan="{}",
    )
    assert claim.state is RunState.SUBMITTING
    harness.store.recover()
    assert harness.store.status(harness.access, claim.run).state is RunState.NOT_SUBMITTED
    assert harness.provider.mutation_count(claim.identity) == 0


def test_s2_started_before_provider_call_never_resubmits_and_reconciles_absent(harness: Harness) -> None:
    claim = harness.store.claim_submission(
        harness.access, harness.grant, harness.binding, canonical_plan="{}"
    )
    harness.store.mark_effect_started(harness.access, claim.run, EffectKind.SUBMIT)
    harness.store.recover()
    assert harness.store.status(harness.access, claim.run).state is RunState.RECONCILE_REQUIRED

    replay = _start(harness)
    assert replay.replayed
    assert replay.state is RunState.RECONCILE_REQUIRED
    assert harness.provider.mutation_count(claim.identity) == 0

    reconciled = harness.service.reconcile(
        harness.access, claim.run, ProviderAuth(harness.scope, object())
    )
    assert reconciled.state is RunState.NOT_SUBMITTED
    assert harness.provider.mutation_count(claim.identity) == 0


def test_s3_after_provider_effect_is_ambiguous_then_reconciles_without_resubmit(harness: Harness) -> None:
    harness.provider.arm(FakeFault.AFTER_SUBMIT)
    result = _start(harness)
    identity = harness.store.effect(harness.access, result.run, EffectKind.SUBMIT).identity
    assert result.state is RunState.SUBMISSION_AMBIGUOUS
    assert harness.provider.mutation_count(identity) == 1
    assert _start(harness).replayed
    assert harness.provider.mutation_count(identity) == 1

    reconciled = harness.service.reconcile(harness.access, result.run, ProviderAuth(harness.scope, object()))
    assert reconciled.state is RunState.SUBMITTED
    assert harness.provider.mutation_count(identity) == 1


def test_persistence_failure_after_provider_effect_recovers_by_lookup_only(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = JobStore.finish_effect
    failed_once = False

    def fail_first_finish(self, *args, **kwargs):
        nonlocal failed_once
        if not failed_once and kwargs.get("effect_status") == "confirmed":
            failed_once = True
            raise sqlite3.OperationalError("injected local persistence failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JobStore, "finish_effect", fail_first_finish)
    with pytest.raises(sqlite3.OperationalError, match="persistence failure"):
        _start(harness)
    effect = harness.store.effect(
        harness.access,
        harness.service.list(harness.access)[0],
        EffectKind.SUBMIT,
    )
    assert effect.status == "started"
    assert harness.provider.mutation_count(effect.identity) == 1

    fresh_store = JobStore(harness.store.location, clock=lambda: NOW, id_factory=harness.ids)
    fresh_store.recover()
    run = fresh_store.list_runs(harness.access)[0]
    fresh_service = JobService(fresh_store, {"fake": harness.provider})
    assert fresh_service.reconcile(harness.access, run, ProviderAuth(harness.scope, object())).state is RunState.SUBMITTED
    assert harness.provider.mutation_count(effect.identity) == 1


@pytest.mark.parametrize(
    "fault,expected",
    [
        (FakeFault.RECONCILE_ABSENT, RunState.NOT_SUBMITTED),
        (FakeFault.RECONCILE_INDETERMINATE, RunState.SUBMISSION_AMBIGUOUS),
        (FakeFault.RECONCILE_COLLISION, RunState.SUBMISSION_AMBIGUOUS),
    ],
)
def test_reconcile_distinguishes_absence_from_uncertainty_and_collision(
    harness: Harness, fault: FakeFault, expected: RunState,
) -> None:
    harness.provider.arm(FakeFault.AFTER_SUBMIT)
    result = _start(harness)
    harness.provider.arm(fault)
    status = harness.service.reconcile(harness.access, result.run, ProviderAuth(harness.scope, object()))
    assert status.state is expected
    identity = harness.store.effect(harness.access, result.run, EffectKind.SUBMIT).identity
    assert harness.provider.mutation_count(identity) == 1


def test_two_threads_have_one_submit_winner_and_one_replay(harness: Harness) -> None:
    harness.provider.arm(FakeFault.SUBMIT_BARRIER)
    entered, release = harness.provider.barrier(FakeFault.SUBMIT_BARRIER)
    results = []
    errors = []

    def invoke() -> None:
        try:
            results.append(_start(harness))
        except Exception as exc:  # pragma: no cover - asserted empty
            errors.append(exc)

    first = threading.Thread(target=invoke)
    first.start()
    assert entered.wait(timeout=5)
    second = threading.Thread(target=invoke)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert sorted(result.replayed for result in results) == [False, True]
    identity = harness.store.effect(harness.access, results[0].run, EffectKind.SUBMIT).identity
    assert harness.provider.mutation_count(identity) == 1


def test_cancel_before_effect_is_definitively_failed_and_terminal_run_is_truthful(harness: Harness) -> None:
    started = _start(harness)
    harness.provider.arm(FakeFault.BEFORE_CANCEL)
    cancelled = harness.service.cancel(harness.access, started.run, ProviderAuth(harness.scope, object()))
    assert not cancelled.accepted
    assert cancelled.state is RunState.CANCEL_FAILED
    cancel_effect = harness.store.effect(harness.access, started.run, EffectKind.CANCEL)
    assert harness.provider.mutation_count(cancel_effect.identity) == 0

    # A fresh successful run observed terminal must not produce a cancel effect.
    other_grant = ExecutionGrant("grant://terminal")
    other_binding = replace(harness.binding, operation_key="op-terminal")
    create_grant_registrar(harness.store).register(harness.access, other_grant, other_binding)
    other = harness.service.start(
        harness.access, harness.plan, other_grant, other_binding, ProviderAuth(harness.scope, object())
    )
    submit_effect = harness.store.effect(harness.access, other.run, EffectKind.SUBMIT)
    assert submit_effect.provider_job is not None
    harness.provider.set_state(submit_effect.provider_job, ProviderRunState.SUCCEEDED)
    assert harness.service.refresh(harness.access, other.run, ProviderAuth(harness.scope, object())).state is RunState.SUCCEEDED
    terminal_cancel = harness.service.cancel(harness.access, other.run, ProviderAuth(harness.scope, object()))
    assert not terminal_cancel.accepted
    assert terminal_cancel.state is RunState.CANCEL_FAILED


def test_cancel_after_effect_is_ambiguous_then_reconciles_without_second_mutation(harness: Harness) -> None:
    started = _start(harness)
    harness.provider.arm(FakeFault.AFTER_CANCEL)
    cancelled = harness.service.cancel(harness.access, started.run, ProviderAuth(harness.scope, object()))
    assert cancelled.state is RunState.CANCEL_AMBIGUOUS
    effect = harness.store.effect(harness.access, started.run, EffectKind.CANCEL)
    assert harness.provider.mutation_count(effect.identity) == 1

    status = harness.service.reconcile(
        harness.access, started.run, ProviderAuth(harness.scope, object()), kind=EffectKind.CANCEL
    )
    assert status.state is RunState.CANCELLING
    assert harness.provider.mutation_count(effect.identity) == 1
    assert harness.service.refresh(harness.access, started.run, ProviderAuth(harness.scope, object())).state is RunState.CANCELLED


def test_two_threads_have_one_cancel_mutation(harness: Harness) -> None:
    started = _start(harness)
    harness.provider.arm(FakeFault.CANCEL_BARRIER)
    entered, release = harness.provider.barrier(FakeFault.CANCEL_BARRIER)
    results = []
    errors = []

    def invoke() -> None:
        try:
            results.append(harness.service.cancel(harness.access, started.run, ProviderAuth(harness.scope, object())))
        except Exception as exc:  # pragma: no cover - asserted empty
            errors.append(exc)

    first = threading.Thread(target=invoke)
    first.start()
    assert entered.wait(timeout=5)
    second = threading.Thread(target=invoke)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(results) == 2
    effect = harness.store.effect(harness.access, started.run, EffectKind.CANCEL)
    assert harness.provider.mutation_count(effect.identity) == 1


def test_logs_are_closed_structured_codes_and_never_include_provider_exception_text(harness: Harness) -> None:
    secret = "hf_secret_that_must_not_persist"

    def explode(*_args, **_kwargs):
        raise RuntimeError(secret)

    harness.provider.submit = explode  # type: ignore[method-assign]
    result = _start(harness)
    page = harness.service.logs(harness.access, result.run)
    serialized = repr(page.to_dict())
    assert secret not in serialized
    assert page.entries
    assert {entry.level for entry in page.entries} <= {"info", "warning", "error"}
    assert all(" " not in entry.event and " " not in entry.message for entry in page.entries)


@pytest.mark.parametrize("limit", [0, 501, True])
def test_log_query_limit_fails_closed(harness: Harness, limit: int) -> None:
    run = _start(harness).run
    with pytest.raises(ValueError, match="limit"):
        harness.service.logs(harness.access, run, limit=limit)


def test_log_cursor_survives_restart_and_is_bound_to_run(harness: Harness) -> None:
    run = _start(harness).run
    first = harness.service.logs(harness.access, run, limit=1)
    assert first.next_cursor is not None

    fresh = JobStore(harness.store.location, clock=lambda: NOW, id_factory=harness.ids)
    second = fresh.logs(harness.access, run, first.next_cursor, 10)
    assert second.entries
    assert second.entries[0].sequence > first.entries[-1].sequence

    other_grant = ExecutionGrant("grant://cursor-other")
    other_binding = replace(harness.binding, operation_key="op-cursor-other")
    create_grant_registrar(fresh).register(harness.access, other_grant, other_binding)
    other = JobService(fresh, {"fake": harness.provider}).start(
        harness.access, harness.plan, other_grant, other_binding,
        ProviderAuth(harness.scope, object()),
    )
    with pytest.raises(ValueError, match="invalid log cursor"):
        fresh.logs(harness.access, other.run, first.next_cursor, 10)
    with pytest.raises(ValueError, match="invalid log cursor"):
        fresh.logs(harness.access, run, LogCursor(first.next_cursor.value + "x"), 10)


def test_log_retention_reports_truncation_for_an_old_cursor(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_RETAINED_RECORDS", 3)
    claim = harness.store.claim_submission(
        harness.access, harness.grant, harness.binding, canonical_plan="{}"
    )
    harness.store.mark_effect_started(harness.access, claim.run, EffectKind.SUBMIT)
    old_page = harness.store.logs(harness.access, claim.run, None, 1)
    assert old_page.next_cursor is not None
    harness.store.finish_effect(
        harness.access, claim.run, EffectKind.SUBMIT,
        effect_status="ambiguous", state=RunState.SUBMISSION_AMBIGUOUS,
        message=store_module.MessageCode.EFFECT_OUTCOME_UNKNOWN,
    )
    reconcile = harness.store.claim_reconcile(
        harness.access, claim.run, EffectKind.SUBMIT
    )
    harness.store.finish_effect(
        harness.access, claim.run, EffectKind.SUBMIT,
        effect_status="absent", state=RunState.NOT_SUBMITTED,
        message=store_module.MessageCode.EFFECT_DEFINITIVELY_ABSENT,
        observation_result="definitively_absent",
        reconciliation_token=reconcile.claim_token,
    )
    page = harness.store.logs(harness.access, claim.run, old_page.next_cursor, 10)
    assert page.truncated
    assert len(page.entries) <= 3


def test_fake_and_store_internals_are_not_exported_from_private_contract_root() -> None:
    import synaptic_tuner._next_api_v1 as candidate

    assert "JobStore" not in candidate.__all__
    assert "FakeExecutionProvider" not in candidate.__all__
    assert not hasattr(candidate, "JobStore")
    assert not hasattr(candidate, "FakeExecutionProvider")
