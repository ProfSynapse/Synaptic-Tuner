"""Read-only public Runs operations for verified Modal training outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from threading import RLock
from typing import Callable

from synaptic_tuner.api.v1.results import (
    TrainingRunRef,
    TrainingRunState,
    VerifiedArtifact,
)
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunListRequest,
    RunLogsRequest,
    RunOperationCode,
    RunOperationError,
    RunOutcome,
    RunVerification,
)
from tuner.execution.contracts import LifecyclePhase, VerificationStatus
from tuner.project.context import ProjectContext

from .contracts import ArtifactMemberV1
from .control import TerminalControlPlane
from .logs import LogControlPlane
from .manifest import CompletionControlPlane
from .facade import ExplicitModal154ReadFacade
from .training import (
    ModalDurablePreparationV1,
    ModalPreparedRunV1,
    _ExpectationStore,
    _prepared_record_prefix,
    _validated_current_record,
)


_MAX_CHUNK_BYTES = 1_048_576


def _error(code: RunOperationCode) -> RunOperationError:
    return RunOperationError(code)


def _one_use_claim() -> Callable[[], bool]:
    lock = RLock()
    consumed = False

    def claim() -> bool:
        nonlocal consumed
        with lock:
            if consumed:
                return False
            consumed = True
            return True

    return claim


@dataclass(frozen=True, slots=True)
class _VerifiedMemberV1:
    role: str
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_member(cls, value: ArtifactMemberV1) -> "_VerifiedMemberV1":
        if type(value) is not ArtifactMemberV1:
            raise ValueError("Modal artifact manifest member is invalid")
        return cls(value.role.value, value.path, value.sha256, value.size)

    @property
    def descriptor(self) -> VerifiedArtifact:
        return VerifiedArtifact(self.role, self.sha256, self.size_bytes)


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshotV1:
    run: TrainingRunRef
    preparation_bytes: bytes
    record_bytes: bytes
    volume_id: str
    members: tuple[_VerifiedMemberV1, ...]


@dataclass(frozen=True, slots=True)
class _ModalArtifactStreamV1:
    run: TrainingRunRef
    artifact: VerifiedArtifact
    maximum_bytes: int
    _facade: object
    _volume_id: str
    _path: str
    _durable_guard: object
    _claim: Callable[[], bool] = field(
        default_factory=_one_use_claim, repr=False, compare=False
    )

    def iter_bytes(self):
        if not self._claim():
            raise _error(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        return self._consume()

    def _consume(self):
        run = TrainingRunRef.from_dict(self.run.to_dict())
        artifact = VerifiedArtifact.from_dict(self.artifact.to_dict())
        maximum_bytes = self.maximum_bytes
        facade = self._facade
        volume_id = self._volume_id
        path = self._path
        durable_guard = self._durable_guard
        digest = hashlib.sha256()
        total = 0
        try:
            durable_guard()
            iterator = iter(
                facade.iter_complete(
                    volume_id, path, max_bytes=maximum_bytes
                )
            )
            for chunk in iterator:
                if type(chunk) is not bytes or not chunk or len(chunk) > _MAX_CHUNK_BYTES:
                    raise ValueError
                total += len(chunk)
                if total > maximum_bytes or total > artifact.size_bytes:
                    raise ValueError
                digest.update(chunk)
                yield chunk
            if (
                total != artifact.size_bytes
                or digest.hexdigest() != artifact.sha256
                or self.run != run
                or self.artifact != artifact
                or self.maximum_bytes != maximum_bytes
                or self._facade is not facade
                or self._volume_id != volume_id
                or self._path != path
                or self._durable_guard is not durable_guard
            ):
                raise ValueError
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise _error(RunOperationCode.ARTIFACT_CONTENT_INVALID) from None


class ModalVerifiedRunsOperationsV1:
    """Bounded reads from already verified, host-persisted Modal runs."""

    __slots__ = (
        "_context", "_repository", "_facade", "_completion", "_clock",
        "_cache", "_lock",
    )

    def __init__(
        self,
        *,
        context: ProjectContext,
        repository: object,
        authenticator: object,
        modal_reads: ExplicitModal154ReadFacade,
        clock: object,
    ) -> None:
        if not isinstance(context, ProjectContext) or context.mode != "host":
            raise ValueError("Modal run reads require a host project context")
        required = (
            "load", "load_modal_preparation", "load_modal_preparation_by_effect"
        )
        if any(not callable(getattr(repository, name, None)) for name in required):
            raise TypeError("repository must provide exact Modal run reads")
        if not callable(getattr(authenticator, "verify", None)):
            raise TypeError("authenticator must provide evidence verification")
        if type(modal_reads) is not ExplicitModal154ReadFacade:
            raise TypeError("modal_reads must be the exact explicit Modal facade")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._context = context
        self._repository = repository
        self._facade = modal_reads
        self._clock = clock
        expectations = _ExpectationStore(repository)
        terminal = TerminalControlPlane(expectations, authenticator, modal_reads)
        logs = LogControlPlane(expectations, authenticator, modal_reads)
        self._completion = CompletionControlPlane(
            expectations, authenticator, modal_reads, terminal, logs
        )
        self._cache: dict[tuple[str, str], _VerifiedSnapshotV1] = {}
        self._lock = RLock()

    @staticmethod
    def _run(value: object) -> TrainingRunRef:
        if type(value) is not TrainingRunRef:
            raise _error(RunOperationCode.PROVIDER_READ_INVALID)
        return TrainingRunRef.from_dict(value.to_dict())

    def _durable(self, public_run: TrainingRunRef):
        run = self._run(public_run)
        repository = self._repository
        try:
            preparation = repository.load_modal_preparation(
                run.project_ref, run.run_id
            )
            record = repository.load(run.project_ref, run.run_id)
            if type(preparation) is not ModalDurablePreparationV1:
                raise ValueError
            prepared = ModalPreparedRunV1(
                _prepared_record_prefix(record), preparation
            )
            current = _validated_current_record(
                record,
                prepared=prepared,
                expected_effect=preparation.operation.effect,
                require_ready_grant_ref=None,
            )
            if (
                current.project_ref != run.project_ref
                or current.run_id != run.run_id
                or preparation.context.project_ref != run.project_ref
                or preparation.operation.project_ref != run.project_ref
                or preparation.operation.run_id != run.run_id
                or current.phase is not LifecyclePhase.SUCCEEDED
                or current.verification is not VerificationStatus.VERIFIED
            ):
                raise ValueError
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise _error(RunOperationCode.READ_INELIGIBLE) from None
        return run, preparation, current

    def _snapshot(self, public_run: TrainingRunRef, *, refresh: bool = False):
        requested = self._run(public_run)
        key = (requested.project_ref, requested.run_id)
        if refresh:
            with self._lock:
                self._cache.pop(key, None)
        try:
            run, preparation, current = self._durable(requested)
        except BaseException:
            with self._lock:
                self._cache.pop(key, None)
            raise
        repository = self._repository
        preparation_bytes = preparation.canonical_bytes
        record_bytes = current.canonical_bytes
        with self._lock:
            cached = self._cache.get(key)
        if (
            not refresh
            and cached is not None
            and cached.preparation_bytes == preparation_bytes
            and cached.record_bytes == record_bytes
        ):
            return cached
        if cached is not None:
            with self._lock:
                self._cache.pop(key, None)
        try:
            manifest = self._completion.validate(
                preparation.operation.effect.effect_id
            )
            members = tuple(
                sorted(
                    (_VerifiedMemberV1.from_member(item) for item in manifest.members),
                    key=lambda item: item.role,
                )
            )
            after_preparation = repository.load_modal_preparation(
                run.project_ref, run.run_id
            )
            after_record = repository.load(run.project_ref, run.run_id)
            if (
                after_preparation != preparation
                or after_record.canonical_bytes != current.canonical_bytes
                or len(members) != 5
                or len({item.role for item in members}) != 5
            ):
                raise ValueError
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            with self._lock:
                self._cache.pop(key, None)
            raise _error(RunOperationCode.PROVIDER_READ_INVALID) from None
        snapshot = _VerifiedSnapshotV1(
            run,
            bytes(preparation_bytes),
            bytes(record_bytes),
            preparation.context.artifact_volume_id,
            members,
        )
        with self._lock:
            self._cache[key] = snapshot
        return snapshot

    def _guard_snapshot(self, snapshot: _VerifiedSnapshotV1) -> None:
        key = (snapshot.run.project_ref, snapshot.run.run_id)
        try:
            run, preparation, current = self._durable(snapshot.run)
            if (
                run != snapshot.run
                or preparation.canonical_bytes != snapshot.preparation_bytes
                or current.canonical_bytes != snapshot.record_bytes
            ):
                raise _error(RunOperationCode.READ_INELIGIBLE)
        except BaseException:
            with self._lock:
                self._cache.pop(key, None)
            raise

    @staticmethod
    def _outcome(
        run: TrainingRunRef, members: tuple[_VerifiedMemberV1, ...]
    ) -> RunOutcome:
        return RunOutcome(
            "synaptic-run-outcome/v1",
            run,
            TrainingRunState.SUCCEEDED,
            tuple(item.descriptor for item in members),
        )

    def show(self, run: TrainingRunRef) -> RunOutcome:
        snapshot = self._snapshot(run)
        return self._outcome(snapshot.run, snapshot.members)

    def outcome(self, run: TrainingRunRef) -> RunOutcome:
        return self.show(run)

    def verify(self, run: TrainingRunRef) -> RunVerification:
        snapshot = self._snapshot(run)
        return RunVerification(snapshot.run, True, self._clock())

    def reverify(self, run: TrainingRunRef) -> RunVerification:
        snapshot = self._snapshot(run, refresh=True)
        return RunVerification(snapshot.run, True, self._clock())

    def artifacts(self, request: RunArtifactRequest):
        if type(request) is not RunArtifactRequest:
            raise _error(RunOperationCode.ARTIFACT_CONTENT_INVALID)
        snapshot = self._snapshot(request.run)
        matches = tuple(item for item in snapshot.members if item.role == request.role)
        if len(matches) != 1:
            raise _error(RunOperationCode.ARTIFACT_ROLE_MISSING)
        member = matches[0]
        if member.size_bytes > request.maximum_bytes:
            raise _error(RunOperationCode.ARTIFACT_LIMIT_EXCEEDED)
        return _ModalArtifactStreamV1(
            snapshot.run,
            member.descriptor,
            request.maximum_bytes,
            self._facade,
            snapshot.volume_id,
            member.path,
            lambda: self._guard_snapshot(snapshot),
        )

    @staticmethod
    def list(_request: RunListRequest):
        raise _error(RunOperationCode.CAPABILITY_UNAVAILABLE)

    @staticmethod
    def logs(_request: RunLogsRequest):
        raise _error(RunOperationCode.CAPABILITY_UNAVAILABLE)

    @staticmethod
    def cancel(_run: TrainingRunRef, _reason: str):
        raise _error(RunOperationCode.CAPABILITY_UNAVAILABLE)

    @staticmethod
    def reconcile(_run: TrainingRunRef):
        raise _error(RunOperationCode.CAPABILITY_UNAVAILABLE)


def compose_modal_verified_run_reads(
    *,
    context: ProjectContext,
    repository: object,
    authenticator: object,
    modal_reads: ExplicitModal154ReadFacade,
    clock: object,
) -> ModalVerifiedRunsOperationsV1:
    """Compose read-only Runs operations over existing Modal run state."""
    return ModalVerifiedRunsOperationsV1(
        context=context,
        repository=repository,
        authenticator=authenticator,
        modal_reads=modal_reads,
        clock=clock,
    )


__all__ = ["ModalVerifiedRunsOperationsV1", "compose_modal_verified_run_reads"]
