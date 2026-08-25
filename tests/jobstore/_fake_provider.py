"""Deterministic file-backed execution fake owned exclusively by tests."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from enum import Enum
from pathlib import Path
from typing import Callable

from synaptic_tuner._next_api_v1._provider import (
    CancelRequest,
    DefinitiveNoEffect,
    EffectIdentity,
    EffectObservation,
    EffectOutcomeUnknown,
    EffectReceipt,
    ExecutionScope,
    LookupResult,
    ProtocolViolation,
    ProviderAuth,
    ProviderJobRef,
    ProviderRunState,
    RunObservation,
    SubmitRequest,
)


class FakeFault(str, Enum):
    BEFORE_SUBMIT = "before_submit"
    AFTER_SUBMIT = "after_submit"
    WRONG_RECEIPT = "wrong_receipt"
    COLLISION = "collision"
    RECONCILE_ABSENT = "reconcile_absent"
    RECONCILE_INDETERMINATE = "reconcile_indeterminate"
    RECONCILE_COLLISION = "reconcile_collision"
    BEFORE_CANCEL = "before_cancel"
    AFTER_CANCEL = "after_cancel"
    CANCEL_ROLLBACK = "cancel_rollback"
    MALFORMED_STATUS = "malformed_status"
    SUBMIT_BARRIER = "submit_barrier"
    CANCEL_BARRIER = "cancel_barrier"


class FakeExecutionProvider:
    """Provider state is isolated from JobStore state and fully transactional."""

    def __init__(
        self,
        database_path: str | Path,
        scope: ExecutionScope,
        *,
        clock: Callable[[], str],
        id_factory: Callable[[str], str],
    ) -> None:
        path = Path(database_path)
        if not path.is_absolute() or not path.parent.exists():
            raise ValueError("fake provider database path must be explicit and absolute")
        if not isinstance(scope, ExecutionScope):
            raise TypeError("scope must be ExecutionScope")
        self._path = path
        self._scope = scope
        self._clock = clock
        self._id_factory = id_factory
        self._barriers = {
            FakeFault.SUBMIT_BARRIER: (threading.Event(), threading.Event()),
            FakeFault.CANCEL_BARRIER: (threading.Event(), threading.Event()),
        }
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def scope(self) -> ExecutionScope:
        return self._scope

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), isolation_level=None, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS jobs(
                  provider_job_id TEXT PRIMARY KEY, state TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS effects(
                  effect_key TEXT PRIMARY KEY, effect_id TEXT NOT NULL,
                  kind TEXT NOT NULL, provider_job_id TEXT NOT NULL,
                  receipt_digest TEXT NOT NULL, mutation_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS faults(
                  name TEXT PRIMARY KEY, remaining INTEGER NOT NULL
                );
                COMMIT;
                """
            )
        finally:
            connection.close()

    def arm(self, fault: FakeFault, count: int = 1) -> None:
        if not isinstance(fault, FakeFault) or not isinstance(count, int) or count < 0:
            raise ValueError("fault and non-negative count are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO faults(name,remaining) VALUES (?,?) "
                "ON CONFLICT(name) DO UPDATE SET remaining=excluded.remaining",
                (fault.value, count),
            )
            connection.execute("COMMIT")
        finally:
            connection.close()

    def barrier(self, fault: FakeFault) -> tuple[threading.Event, threading.Event]:
        try:
            return self._barriers[fault]
        except KeyError as exc:
            raise ValueError("fault is not a barrier") from exc

    def _take(self, fault: FakeFault) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT remaining FROM faults WHERE name=?", (fault.value,)
            ).fetchone()
            taken = row is not None and row["remaining"] > 0
            if taken:
                connection.execute(
                    "UPDATE faults SET remaining=remaining-1 WHERE name=?", (fault.value,)
                )
            connection.execute("COMMIT")
            return taken
        finally:
            connection.close()

    def _wait(self, fault: FakeFault) -> None:
        if not self._take(fault):
            return
        entered, released = self._barriers[fault]
        entered.set()
        if not released.wait(timeout=30):
            raise EffectOutcomeUnknown("fake barrier timed out")

    def submit(self, auth: ProviderAuth, request: SubmitRequest) -> EffectReceipt:
        self._require_auth(auth)
        if request.identity.scope != self.scope:
            raise ProtocolViolation("submission scope mismatch")
        if self._take(FakeFault.BEFORE_SUBMIT):
            raise DefinitiveNoEffect("injected before-submit fault")
        self._wait(FakeFault.SUBMIT_BARRIER)
        receipt = self._submit_mutation(request.identity)
        if self._take(FakeFault.AFTER_SUBMIT):
            raise EffectOutcomeUnknown("injected after-submit fault")
        if self._take(FakeFault.WRONG_RECEIPT):
            wrong = EffectIdentity(
                "wrong", request.identity.effect_key, request.identity.kind,
                request.identity.scope,
            )
            return EffectReceipt(wrong, receipt.job, receipt.receipt_digest)
        return receipt

    def cancel(self, auth: ProviderAuth, request: CancelRequest) -> EffectReceipt:
        self._require_auth(auth)
        if request.identity.scope != self.scope:
            raise ProtocolViolation("cancellation scope mismatch")
        if self._take(FakeFault.BEFORE_CANCEL):
            raise DefinitiveNoEffect("injected before-cancel fault")
        self._wait(FakeFault.CANCEL_BARRIER)
        rollback = self._take(FakeFault.CANCEL_ROLLBACK)
        receipt = self._cancel_mutation(request.identity, request.job, rollback=rollback)
        if self._take(FakeFault.AFTER_CANCEL):
            raise EffectOutcomeUnknown("injected after-cancel fault")
        return receipt

    def _submit_mutation(self, identity: EffectIdentity) -> EffectReceipt:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = self._prior_or_collision(connection, identity)
            if prior is not None:
                connection.execute("COMMIT")
                return self._receipt(identity, prior)
            job = ProviderJobRef(self._id_factory("provider_job"))
            now = self._clock()
            connection.execute(
                "INSERT INTO jobs VALUES (?,?,?,?)",
                (job.provider_job_id, ProviderRunState.SUBMITTED.value, now, now),
            )
            receipt = self._insert_effect(connection, identity, job)
            connection.execute("COMMIT")
            return receipt
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _cancel_mutation(
        self, identity: EffectIdentity, job: ProviderJobRef, *, rollback: bool
    ) -> EffectReceipt:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = self._prior_or_collision(connection, identity)
            if prior is not None:
                connection.execute("COMMIT")
                return self._receipt(identity, prior)
            exists = connection.execute(
                "SELECT 1 FROM jobs WHERE provider_job_id=?", (job.provider_job_id,)
            ).fetchone()
            if exists is None:
                raise ProtocolViolation("unknown cancellation target")
            receipt = self._insert_effect(connection, identity, job)
            connection.execute(
                "UPDATE jobs SET state=?,updated_at=? WHERE provider_job_id=?",
                (ProviderRunState.CANCELLED.value, self._clock(), job.provider_job_id),
            )
            if rollback:
                raise EffectOutcomeUnknown("injected atomic cancel rollback")
            connection.execute("COMMIT")
            return receipt
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _prior_or_collision(
        connection: sqlite3.Connection, identity: EffectIdentity
    ) -> sqlite3.Row | None:
        prior = connection.execute(
            "SELECT * FROM effects WHERE effect_key=?", (identity.effect_key,)
        ).fetchone()
        if prior is not None and prior["effect_id"] != identity.effect_id:
            raise ProtocolViolation("effect-key collision")
        return prior

    @staticmethod
    def _receipt(identity: EffectIdentity, row: sqlite3.Row) -> EffectReceipt:
        return EffectReceipt(
            identity, ProviderJobRef(row["provider_job_id"]), row["receipt_digest"]
        )

    @staticmethod
    def _insert_effect(
        connection: sqlite3.Connection,
        identity: EffectIdentity,
        job: ProviderJobRef,
    ) -> EffectReceipt:
        digest = hashlib.sha256(
            ("synaptic.test-fake-receipt/v1\0" + identity.effect_key
             + job.provider_job_id).encode("utf-8")
        ).hexdigest()
        connection.execute(
            "INSERT INTO effects VALUES (?,?,?,?,?,1)",
            (identity.effect_key, identity.effect_id, identity.kind.value,
             job.provider_job_id, digest),
        )
        return EffectReceipt(identity, job, digest)

    def lookup_submission(
        self, auth: ProviderAuth, identity: EffectIdentity
    ) -> EffectObservation:
        return self._lookup(auth, identity)

    def lookup_cancellation(
        self, auth: ProviderAuth, identity: EffectIdentity
    ) -> EffectObservation:
        return self._lookup(auth, identity)

    def _lookup(
        self, auth: ProviderAuth, identity: EffectIdentity
    ) -> EffectObservation:
        self._require_auth(auth)
        if identity.scope != self.scope:
            raise ProtocolViolation("lookup scope mismatch")
        if self._take(FakeFault.RECONCILE_ABSENT):
            return EffectObservation(LookupResult.DEFINITIVELY_ABSENT, identity)
        if self._take(FakeFault.RECONCILE_INDETERMINATE):
            return EffectObservation(LookupResult.INDETERMINATE, identity)
        if self._take(FakeFault.RECONCILE_COLLISION) or self._take(FakeFault.COLLISION):
            return EffectObservation(LookupResult.COLLISION, identity)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM effects WHERE effect_key=?", (identity.effect_key,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return EffectObservation(LookupResult.DEFINITIVELY_ABSENT, identity)
        if row["effect_id"] != identity.effect_id:
            return EffectObservation(LookupResult.COLLISION, identity)
        return EffectObservation(
            LookupResult.FOUND, identity, ProviderJobRef(row["provider_job_id"]),
            row["receipt_digest"],
        )

    def observe(self, auth: ProviderAuth, job: ProviderJobRef) -> RunObservation:
        self._require_auth(auth)
        if self._take(FakeFault.MALFORMED_STATUS):
            raise ProtocolViolation("injected malformed status")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT state,updated_at FROM jobs WHERE provider_job_id=?",
                (job.provider_job_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ProtocolViolation("unknown provider job")
        digest = hashlib.sha256(
            (row["state"] + row["updated_at"]).encode("utf-8")
        ).hexdigest()
        return RunObservation(job, ProviderRunState(row["state"]), digest)

    def set_state(self, job: ProviderJobRef, state: ProviderRunState) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE jobs SET state=?,updated_at=? WHERE provider_job_id=?",
                (state.value, self._clock(), job.provider_job_id),
            )
        finally:
            connection.close()

    def mutation_count(self, identity: EffectIdentity) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT mutation_count FROM effects WHERE effect_key=?",
                (identity.effect_key,),
            ).fetchone()
            return 0 if row is None else int(row["mutation_count"])
        finally:
            connection.close()

    def job_state(self, job: ProviderJobRef) -> ProviderRunState:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT state FROM jobs WHERE provider_job_id=?", (job.provider_job_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise LookupError(job.provider_job_id)
        return ProviderRunState(row["state"])

    def _require_auth(self, auth: ProviderAuth) -> None:
        if not isinstance(auth, ProviderAuth) or auth.scope != self.scope:
            raise TypeError("fake provider requires exact-scope ProviderAuth")


__all__ = ["FakeExecutionProvider", "FakeFault"]
