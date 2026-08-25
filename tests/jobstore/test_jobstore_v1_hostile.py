"""Hostile remediation gates layered on the green JobStore lifecycle suite."""

from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from synaptic_tuner._next_api_v1 import _store as store_module
from synaptic_tuner._next_api_v1._authority import create_grant_registrar
from synaptic_tuner._next_api_v1._provider import (
    EffectKind,
    EffectObservation,
    ExecutionScope,
    LookupResult,
    ProviderAuth,
    ProviderJobRef,
    ProviderRunState,
)
from synaptic_tuner._next_api_v1._service import JobService
from synaptic_tuner._next_api_v1._store import (
    CorruptStore,
    GrantRejected,
    JobStore,
    JobStoreLocation,
)
from synaptic_tuner._next_api_v1.execution import ExecutionGrant, RunState

from ._fake_provider import FakeFault
from .test_jobstore_v1 import (
    HEX_A,
    Harness,
    NOW,
    SequentialIds,
    _binding,
    _start,
    harness,
)


def _auth(h: Harness) -> ProviderAuth:
    return ProviderAuth(h.scope, object())


def _fresh(h: Harness) -> JobStore:
    return JobStore(h.store.location, clock=lambda: NOW, id_factory=h.ids)


def test_wrong_application_id_is_refused_without_open_side_effects(harness: Harness) -> None:
    connection = sqlite3.connect(harness.store_path)
    connection.execute("PRAGMA application_id=123")
    connection.close()
    before = harness.store_path.read_bytes()
    sidecars = [Path(str(harness.store_path) + suffix) for suffix in ("-wal", "-shm")]
    sidecar_before = {
        path: path.read_bytes() if path.exists() else None for path in sidecars
    }
    with pytest.raises(CorruptStore):
        _fresh(harness)
    assert harness.store_path.read_bytes() == before
    assert {
        path: path.read_bytes() if path.exists() else None for path in sidecars
    } == sidecar_before


@pytest.mark.parametrize(
    "mutation",
    [
        "ALTER TABLE logs ADD COLUMN hostile TEXT",
        "DROP INDEX idx_logs_run_sequence",
        "CREATE TRIGGER hostile_trigger AFTER INSERT ON logs BEGIN SELECT 1; END",
        "CREATE VIEW hostile_view AS SELECT * FROM logs",
    ],
)
def test_actual_schema_tampering_is_refused(harness: Harness, mutation: str) -> None:
    connection = sqlite3.connect(harness.store_path)
    connection.execute(mutation)
    connection.close()
    with pytest.raises(CorruptStore):
        _fresh(harness)


def test_wrong_scope_auth_never_reaches_provider_for_refresh_cancel_or_reconcile(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _start(harness)
    calls = {"observe": 0, "cancel": 0, "lookup": 0}

    def counted(name):
        def call(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError("provider must not be called")
        return call

    monkeypatch.setattr(harness.provider, "observe", counted("observe"))
    monkeypatch.setattr(harness.provider, "cancel", counted("cancel"))
    wrong = ProviderAuth(ExecutionScope("fake", "account://other", "namespace://alpha"), object())
    assert harness.service.refresh(harness.access, started.run, wrong).state is RunState.SUBMITTED
    cancelled = harness.service.cancel(harness.access, started.run, wrong)
    assert cancelled.state is RunState.CANCEL_FAILED
    assert calls == {"observe": 0, "cancel": 0, "lookup": 0}

    second_grant = ExecutionGrant("grant://wrong-scope-reconcile")
    second_binding = replace(harness.binding, operation_key="op-wrong-scope-reconcile")
    create_grant_registrar(harness.store).register(
        harness.access, second_grant, second_binding
    )
    harness.provider.arm(FakeFault.AFTER_SUBMIT)
    second = harness.service.start(
        harness.access, harness.plan, second_grant, second_binding, _auth(harness)
    )
    monkeypatch.setattr(harness.provider, "lookup_submission", counted("lookup"))
    status = harness.service.reconcile(harness.access, second.run, wrong)
    assert status.state is RunState.SUBMISSION_AMBIGUOUS
    assert calls == {"observe": 0, "cancel": 0, "lookup": 0}


@pytest.mark.parametrize(
    "bad_job,bad_digest",
    [
        ("provider job with spaces", HEX_A),
        ("provider/job", "NOT-A-DIGEST"),
        ("x" * 257, HEX_A),
    ],
)
def test_hostile_provider_receipt_values_persist_nothing(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, bad_job: str, bad_digest: str,
) -> None:
    def hostile_submit(_auth_value, request):
        return SimpleNamespace(
            identity=request.identity,
            job=SimpleNamespace(provider_job_id=bad_job),
            receipt_digest=bad_digest,
        )

    monkeypatch.setattr(harness.provider, "submit", hostile_submit)
    result = _start(harness)
    assert result.state is RunState.SUBMISSION_AMBIGUOUS
    effect = harness.store.effect(harness.access, result.run, EffectKind.SUBMIT)
    assert effect.provider_job is None
    connection = sqlite3.connect(harness.store_path)
    row = connection.execute(
        "SELECT provider_job_id,receipt_digest FROM effects WHERE effect_id=?",
        (effect.identity.effect_id,),
    ).fetchone()
    connection.close()
    assert row == (None, None)


def test_dangling_database_link_is_refused_without_creating_target(tmp_path: Path) -> None:
    host = tmp_path / "host"
    state = host / "state"
    engine = host / "engine"
    state.mkdir(parents=True)
    engine.mkdir()
    target = tmp_path / "outside.sqlite3"
    link = state / "jobs.sqlite3"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"platform cannot create an unprivileged dangling symlink: {exc}")
    with pytest.raises(CorruptStore):
        JobStore(
            JobStoreLocation(link, state, (engine,)),
            clock=lambda: NOW,
            id_factory=SequentialIds(),
        )
    assert not target.exists()


def test_cancel_claimed_recovery_routes_as_cancellation_without_provider_effect(
    harness: Harness,
) -> None:
    started = _start(harness)
    cancel = harness.store.create_cancel_effect(harness.access, started.run)
    assert cancel is not None
    harness.store.recover()
    assert harness.store.status(harness.access, started.run).state is RunState.CANCEL_FAILED
    assert harness.provider.mutation_count(cancel.identity) == 0


def test_cancel_started_recovery_reconciles_cancellation_absent(harness: Harness) -> None:
    started = _start(harness)
    cancel = harness.store.create_cancel_effect(harness.access, started.run)
    assert cancel is not None
    harness.store.mark_effect_started(harness.access, started.run, EffectKind.CANCEL)
    harness.store.recover()
    assert harness.store.status(harness.access, started.run).state is RunState.RECONCILE_REQUIRED
    status = harness.service.reconcile(
        harness.access, started.run, _auth(harness), kind=EffectKind.CANCEL
    )
    assert status.state is RunState.CANCEL_FAILED
    assert harness.provider.mutation_count(cancel.identity) == 0


def test_cancel_found_for_different_target_stays_ambiguous(
    harness: Harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = _start(harness)
    harness.provider.arm(FakeFault.AFTER_CANCEL)
    assert harness.service.cancel(harness.access, started.run, _auth(harness)).state is RunState.CANCEL_AMBIGUOUS

    def wrong_target(_auth_value, identity):
        return EffectObservation(
            LookupResult.FOUND, identity, ProviderJobRef("provider_job-other"), HEX_A
        )

    monkeypatch.setattr(harness.provider, "lookup_cancellation", wrong_target)
    status = harness.service.reconcile(
        harness.access, started.run, _auth(harness), kind=EffectKind.CANCEL
    )
    assert status.state is RunState.CANCEL_AMBIGUOUS
    assert status.message_code == "provider_protocol_violation"


@pytest.mark.parametrize(
    "issued,expires",
    [
        ("2026-08-25T16:00:01Z", "2026-08-25T17:00:00Z"),
        ("2026-08-25T15:00:00Z", NOW),
    ],
)
def test_future_issued_and_exact_expiry_authority_are_rejected(
    harness: Harness, issued: str, expires: str,
) -> None:
    grant = ExecutionGrant("grant://time-" + issued[-3:-1] + expires[-3:-1])
    binding = replace(
        harness.binding,
        operation_key="op-time-" + issued[-3:-1] + expires[-3:-1],
        issued_at=issued,
        expires_at=expires,
    )
    create_grant_registrar(harness.store).register(harness.access, grant, binding)
    with pytest.raises(GrantRejected, match="lifetime"):
        harness.service.start(harness.access, harness.plan, grant, binding, _auth(harness))


def test_grant_and_store_times_are_canonical_utc(harness: Harness) -> None:
    binding = replace(
        harness.binding,
        operation_key="op-offset-time",
        issued_at="2026-08-25T11:00:00-04:00",
        expires_at="2026-08-25T13:00:00-04:00",
    )
    assert binding.issued_at == "2026-08-25T15:00:00Z"
    assert binding.expires_at == "2026-08-25T17:00:00Z"


def test_cursor_exact_ttl_expiry_is_rejected(harness: Harness) -> None:
    run = _start(harness).run
    page = harness.service.logs(harness.access, run, limit=1)
    assert page.next_cursor is not None
    connection = sqlite3.connect(harness.store_path)
    connection.execute(
        "UPDATE cursors SET created_at='2026-08-24T16:00:00Z'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="invalid log cursor"):
        harness.service.logs(harness.access, run, cursor=page.next_cursor)


def test_cursor_rows_are_capped_at_1024_per_run(harness: Harness) -> None:
    run = _start(harness).run
    with harness.store._transaction() as connection:
        for sequence in range(store_module.MAX_CURSOR_ROWS_PER_RUN + 5):
            harness.store._encode_cursor(connection, run, sequence)
        count = connection.execute(
            "SELECT COUNT(*) FROM cursors WHERE run_id=?", (run.run_id,)
        ).fetchone()[0]
    assert count == store_module.MAX_CURSOR_ROWS_PER_RUN


def test_test_fake_cancel_rollback_is_atomic(harness: Harness) -> None:
    started = _start(harness)
    submit = harness.store.effect(harness.access, started.run, EffectKind.SUBMIT)
    assert submit.provider_job is not None
    harness.provider.arm(FakeFault.CANCEL_ROLLBACK)
    result = harness.service.cancel(harness.access, started.run, _auth(harness))
    assert result.state is RunState.CANCEL_AMBIGUOUS
    cancel = harness.store.effect(harness.access, started.run, EffectKind.CANCEL)
    assert harness.provider.mutation_count(cancel.identity) == 0
    assert harness.provider.job_state(submit.provider_job) is ProviderRunState.SUBMITTED


def test_production_tree_contains_no_fake_provider_module() -> None:
    package_root = Path(store_module.__file__).parent
    assert not (package_root / "_fake_provider.py").exists()
    import synaptic_tuner._next_api_v1 as candidate

    assert "FakeExecutionProvider" not in candidate.__all__
    assert "create_grant_registrar" not in candidate.__all__
    assert "GrantRegistrar" not in candidate.__all__
