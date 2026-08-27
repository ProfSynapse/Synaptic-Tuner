"""Thread-safe in-memory stores with coordinator-owned transition validation."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from threading import RLock

from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.foundation_v2.authority import (
    AuthenticatedGrantV2,
    AuthenticatedReconciliationGrantV1,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.repository import (
    DispatchState,
    EffectRecordV2,
    EffectState,
    ReconciliationGrantBindingV2,
    ReconciliationOwnershipV2,
)

from .coordinator import (
    ApplyStageEffectTransitionV1,
    ApplySubmitEffectTransitionV1,
    BeginPreparationTransitionV1,
    CoordinatorTransitionV1,
    ExecutionGrantSlotV1,
    ReconciliationGrantSlotV1,
    RecordStageIntentTransitionV1,
    RecordSubmitIntentTransitionV1,
)
from .model import WorkflowPhaseV1, WorkflowRecordV1
from .state_machine import (
    apply_stage_effect_record,
    apply_submit_effect_record,
    begin_preparation,
    record_stage_intent,
    record_submit_intent,
)


class CoordinatorStoreCode(str, Enum):
    AUTHORITY_INVALID = "authority_invalid"
    BINDING_MISMATCH = "binding_mismatch"
    CONFLICT = "conflict"
    INTEGRITY_ERROR = "integrity_error"
    TRANSITION_INVALID = "transition_invalid"


class CoordinatorStoreError(ValueError):
    """Closed coordinator persistence/integrity failure."""

    def __init__(self, code: CoordinatorStoreCode):
        if type(code) is not CoordinatorStoreCode:
            raise TypeError("code must be exact CoordinatorStoreCode")
        self.code = code
        super().__init__(code.value)


def _closed(code: CoordinatorStoreCode):
    return CoordinatorStoreError(code)


def _workflow_key(run: TrainingRunRef) -> tuple[str, str]:
    if type(run) is not TrainingRunRef:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
    return run.project_ref, run.run_id


def _revalidate_workflow(record: WorkflowRecordV1) -> WorkflowRecordV1:
    if type(record) is not WorkflowRecordV1:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    try:
        rebuilt = WorkflowRecordV1(
            **{field.name: getattr(record, field.name) for field in fields(WorkflowRecordV1)}
        )
        matches = rebuilt == record and rebuilt.record_digest == record.record_digest
    except Exception:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
    if not matches:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    return rebuilt


class InMemoryWorkflowStoreV1:
    def __init__(self, foundation_authenticator, assessment_authenticator):
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._records: dict[tuple[str, str], WorkflowRecordV1] = {}
        self._digests: dict[tuple[str, str], str] = {}
        self._plans: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = RLock()

    def _retained(self, key: tuple[str, str]) -> WorkflowRecordV1:
        try:
            retained = _revalidate_workflow(self._records[key])
        except (KeyError, CoordinatorStoreError):
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
        index = (retained.run.project_ref, retained.plan_fingerprint)
        if (
            _workflow_key(retained.run) != key
            or self._digests.get(key) != retained.record_digest
            or self._plans.get(index) != key
        ):
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
        return retained

    def create(self, record: WorkflowRecordV1) -> bool:
        candidate = _revalidate_workflow(record)
        if candidate.phase is not WorkflowPhaseV1.PLANNED or candidate.revision != 0:
            raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
        key = _workflow_key(candidate.run)
        plan_index = (candidate.run.project_ref, candidate.plan_fingerprint)
        with self._lock:
            existing = self._records.get(key)
            plan_key = self._plans.get(plan_index)
            if plan_key is None and any(
                self._retained(retained_key).run.project_ref == plan_index[0]
                and self._retained(retained_key).plan_fingerprint == plan_index[1]
                for retained_key in self._records
            ):
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            if existing is not None:
                existing = self._retained(key)
                if existing == candidate:
                    return False
                raise _closed(CoordinatorStoreCode.CONFLICT)
            if plan_key is not None:
                plan_record = self._retained(plan_key)
                if (
                    plan_key[0] != plan_index[0]
                    or plan_record.plan_fingerprint != plan_index[1]
                ):
                    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
                if plan_record == candidate:
                    return False
                raise _closed(CoordinatorStoreCode.CONFLICT)
            self._records[key] = candidate
            self._digests[key] = candidate.record_digest
            self._plans[plan_index] = key
            return True

    def get(self, run: TrainingRunRef) -> WorkflowRecordV1 | None:
        key = _workflow_key(run)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            return self._retained(key)

    def get_by_plan(
        self, project_ref: str, plan_fingerprint: str
    ) -> WorkflowRecordV1 | None:
        try:
            safe_ref(project_ref, "project_ref")
            digest_text(plan_fingerprint, "plan_fingerprint")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        with self._lock:
            index = (project_ref, plan_fingerprint)
            key = self._plans.get(index)
            if key is None:
                if any(
                    self._retained(record_key).run.project_ref == project_ref
                    and self._retained(record_key).plan_fingerprint == plan_fingerprint
                    for record_key in self._records
                ):
                    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
                return None
            if key[0] != project_ref:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            retained = self._retained(key)
            if (
                retained.plan_fingerprint != plan_fingerprint
                or (retained.run.project_ref, retained.plan_fingerprint) != index
            ):
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            return retained

    def list(self, project_ref: str) -> tuple[WorkflowRecordV1, ...]:
        try:
            safe_ref(project_ref, "project_ref")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        with self._lock:
            records = []
            for key in sorted(self._records):
                retained = self._retained(key)
                if retained.run.project_ref == project_ref:
                    records.append(retained)
            return tuple(records)

    def _replay(
        self, current: WorkflowRecordV1, transition: CoordinatorTransitionV1
    ) -> WorkflowRecordV1:
        failed = False
        try:
            if type(transition) is BeginPreparationTransitionV1:
                return begin_preparation(current)
            if type(transition) is RecordStageIntentTransitionV1:
                return record_stage_intent(
                    current, transition.preparation, transition.intent
                )
            if type(transition) is ApplyStageEffectTransitionV1:
                return apply_stage_effect_record(
                    current,
                    transition.record,
                    transition.assessment,
                    self._foundation_authenticator,
                    self._assessment_authenticator,
                )
            if type(transition) is RecordSubmitIntentTransitionV1:
                return record_submit_intent(current, transition.intent)
            if type(transition) is ApplySubmitEffectTransitionV1:
                return apply_submit_effect_record(
                    current,
                    transition.record,
                    transition.assessment,
                    self._foundation_authenticator,
                    self._assessment_authenticator,
                )
        except Exception:
            failed = True
        if failed:
            raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
        raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)

    def compare_and_swap(
        self,
        expected: WorkflowRecordV1,
        replacement: WorkflowRecordV1,
        *,
        transition: CoordinatorTransitionV1,
    ) -> bool:
        expected = _revalidate_workflow(expected)
        replacement = _revalidate_workflow(replacement)
        key = _workflow_key(expected.run)
        if _workflow_key(replacement.run) != key:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        if replacement.revision != expected.revision + 1:
            raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
        with self._lock:
            retained = self._records.get(key)
            if retained is None:
                return False
            retained = self._retained(key)
            if retained.revision != expected.revision:
                return False
            if retained.record_digest != expected.record_digest or retained != expected:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            replayed = self._replay(retained, transition)
            if replayed.revision != retained.revision + 1 or replayed != replacement:
                raise _closed(CoordinatorStoreCode.TRANSITION_INVALID)
            self._records[key] = replacement
            self._digests[key] = replacement.record_digest
            return True


class InMemoryPreparationStoreV1:
    def __init__(self):
        self._values: dict[str, CanonicalPreparationV2] = {}
        self._lock = RLock()

    @staticmethod
    def _validate(value: CanonicalPreparationV2) -> CanonicalPreparationV2:
        if type(value) is not CanonicalPreparationV2:
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
        try:
            parsed = CanonicalPreparationV2.parse(value.canonical_bytes)
        except (TypeError, ValueError, KeyError, AttributeError):
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
        if parsed != value:
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
        return parsed

    def put_if_absent(self, preparation: CanonicalPreparationV2) -> bool:
        candidate = self._validate(preparation)
        key = candidate.preparation_digest
        with self._lock:
            existing = self._values.get(key)
            if existing is None:
                self._values[key] = candidate
                return True
            if self._validate(existing) == candidate:
                return False
            raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, preparation_digest: str) -> CanonicalPreparationV2 | None:
        try:
            digest_text(preparation_digest, "preparation_digest")
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        with self._lock:
            value = self._values.get(preparation_digest)
            if value is None:
                return None
            retained = self._validate(value)
            if retained.preparation_digest != preparation_digest:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            return retained


def _command_bytes_digest(command_bytes: bytes) -> str:
    if type(command_bytes) is not bytes:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
    return domain_digest("synaptic-foundation-command-bytes/v1", command_bytes)


def _authenticate(authority, method_name: str, *args) -> None:
    try:
        authenticated = getattr(authority, method_name)(*args)
    except Exception:
        authenticated = False
    if authenticated is not True:
        raise _closed(CoordinatorStoreCode.AUTHORITY_INVALID)


def _revalidate_record(record: EffectRecordV2) -> EffectRecordV2:
    if type(record) is not EffectRecordV2:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    try:
        for claim in record.reconciliation_claims:
            ReconciliationOwnershipV2.__post_init__(claim)
            for binding in claim.grant_lineage:
                ReconciliationGrantBindingV2.__post_init__(binding)
        EffectRecordV2.__post_init__(record)
        rebuilt = EffectRecordV2(
            **{field.name: getattr(record, field.name) for field in fields(EffectRecordV2)}
        )
        matches = rebuilt == record and rebuilt.record_digest == record.record_digest
    except Exception:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
    if not matches:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    return rebuilt


class InMemoryExecutionGrantStoreV1:
    def __init__(self, authenticator):
        self._authenticator = authenticator
        self._values: dict[ExecutionGrantSlotV1, AuthenticatedGrantV2] = {}
        self._canonical: dict[ExecutionGrantSlotV1, bytes] = {}
        self._lock = RLock()

    def _validate(self, slot, grant, command_bytes):
        if type(slot) is not ExecutionGrantSlotV1 or type(grant) is not AuthenticatedGrantV2:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        try:
            command = parse_exact_command(command_bytes)
            content = grant.content
            expected = ExecutionGrantSlotV1(
                command.operation.effect.effect_id,
                command.digest,
                _command_bytes_digest(command_bytes),
            )
            if slot != expected or (
                content.effect_id,
                content.command_digest,
                content.preparation_digest,
            ) != (
                expected.effect_id,
                expected.command_digest,
                command.preparation.preparation_digest,
            ):
                raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        except CoordinatorStoreError:
            raise
        except Exception:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
        _authenticate(self._authenticator, "authenticate", grant, command_bytes)
        return grant

    def put_if_absent(self, slot, grant, command_bytes) -> bool:
        candidate = self._validate(slot, grant, command_bytes)
        with self._lock:
            existing = self._values.get(slot)
            if existing is None:
                self._values[slot] = candidate
                self._canonical[slot] = candidate.canonical_bytes
                return True
            existing = self._validate(slot, existing, command_bytes)
            if self._canonical.get(slot) != existing.canonical_bytes:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            if existing == candidate and existing.canonical_bytes == candidate.canonical_bytes:
                return False
            raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, slot, command_bytes):
        if type(slot) is not ExecutionGrantSlotV1:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        with self._lock:
            value = self._values.get(slot)
            if value is None:
                return None
            value = self._validate(slot, value, command_bytes)
            if self._canonical.get(slot) != value.canonical_bytes:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            return value


def _reconciliation_target(content) -> str:
    return domain_digest(
        "synaptic-reconciliation-target/v2",
        canonical_bytes(
            {
                "command_digest": content.command_digest,
                "effect_id": content.effect_id,
                "preparation_digest": content.preparation_digest,
                "adapter_digest": content.adapter_digest,
                "provider_id": content.provider_id,
                "profile_ref": content.profile_ref,
                "account_ref": content.account_ref,
                "namespace_ref": content.namespace_ref,
                "owner_ref": content.owner_ref,
                "policy_digest": content.policy_digest,
                "requirement_digest": content.requirement_digest,
            }
        ),
    )


def _lineage_for_retained(record, grant):
    digest = grant.authenticated_grant_digest
    for claim_index, claim in enumerate(record.reconciliation_claims):
        for binding_index, binding in enumerate(claim.grant_lineage):
            if (binding.grant_ref, binding.grant_digest) != (
                grant.content.grant_ref,
                digest,
            ):
                continue
            content = grant.content
            if (
                content.owner_ref,
                content.generation,
                content.ownership_epoch,
                _reconciliation_target(content),
            ) != (
                claim.owner_ref,
                claim.generation,
                claim.ownership_epoch,
                claim.target_digest,
            ):
                raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
            if binding_index:
                return claim.claim_digest, claim.grant_lineage[binding_index - 1].grant_digest
            if claim_index:
                predecessor = record.reconciliation_claims[claim_index - 1]
                return predecessor.claim_digest, predecessor.grant_digest
            return None, None
    return None


def _derive_reconciliation_slot(record, grant, command_bytes):
    record = _revalidate_record(record)
    try:
        command = parse_exact_command(command_bytes)
    except Exception:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH) from None
    if record.command_bytes != command_bytes:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
    content = grant.content
    actual = (
        content.command_digest,
        content.effect_id,
        content.preparation_digest,
        content.provider_id,
        content.profile_ref,
        content.account_ref,
        content.namespace_ref,
    )
    expected = (
        command.digest,
        command.operation.effect.effect_id,
        command.preparation.preparation_digest,
        command.preparation.provider.provider_id,
        command.preparation.provider.profile_ref,
        command.preparation.scope.account_ref,
        command.preparation.scope.namespace_ref,
    )
    if actual != expected:
        raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
    retained = _lineage_for_retained(record, grant)
    if retained is not None:
        prior_claim, predecessor_grant = retained
    else:
        current = record.reconciliation
        eligible = record.dispatch in {DispatchState.RELINQUISHED, DispatchState.QUIESCENCE_PROVEN}
        eligible = eligible and (
            record.state is EffectState.INDETERMINATE
            or (
                record.dispatch is DispatchState.QUIESCENCE_PROVEN
                and record.state is EffectState.UNRESOLVED
            )
        )
        if not eligible:
            raise _closed(CoordinatorStoreCode.CONFLICT)
        target = _reconciliation_target(content)
        digest = grant.authenticated_grant_digest
        if current is None:
            if (content.generation, content.ownership_epoch) != (1, 1):
                raise _closed(CoordinatorStoreCode.CONFLICT)
            prior_claim = predecessor_grant = None
        elif not current.active and not current.completed:
            if (
                content.owner_ref,
                content.generation,
                content.ownership_epoch,
                target,
            ) != (
                current.owner_ref,
                current.generation,
                current.ownership_epoch,
                current.target_digest,
            ) or content.grant_ref == current.grant_ref or digest == current.grant_digest:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            prior_claim, predecessor_grant = current.claim_digest, current.grant_digest
        elif current.completed:
            if (
                content.owner_ref,
                content.generation,
                content.ownership_epoch,
                target,
            ) != (
                current.owner_ref,
                current.generation + 1,
                current.ownership_epoch + 1,
                current.target_digest,
            ) or content.grant_ref == current.grant_ref or digest == current.grant_digest:
                raise _closed(CoordinatorStoreCode.CONFLICT)
            prior_claim, predecessor_grant = current.claim_digest, current.grant_digest
        else:
            raise _closed(CoordinatorStoreCode.CONFLICT)
    return ReconciliationGrantSlotV1(
        content.effect_id,
        content.command_digest,
        _command_bytes_digest(command_bytes),
        content.generation,
        content.ownership_epoch,
        prior_claim,
        predecessor_grant,
    )


class InMemoryReconciliationGrantStoreV1:
    def __init__(self, authenticator):
        self._authenticator = authenticator
        self._values: dict[
            ReconciliationGrantSlotV1, AuthenticatedReconciliationGrantV1
        ] = {}
        self._canonical: dict[ReconciliationGrantSlotV1, bytes] = {}
        self._lock = RLock()

    def _validate(self, slot, grant, command_bytes, record):
        if (
            type(slot) is not ReconciliationGrantSlotV1
            or type(grant) is not AuthenticatedReconciliationGrantV1
        ):
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        _authenticate(
            self._authenticator, "authenticate_reconciliation", grant, command_bytes
        )
        if slot != _derive_reconciliation_slot(record, grant, command_bytes):
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        return grant

    def put_if_absent(self, slot, grant, command_bytes, record) -> bool:
        candidate = self._validate(slot, grant, command_bytes, record)
        with self._lock:
            existing = self._values.get(slot)
            if existing is None:
                self._values[slot] = candidate
                self._canonical[slot] = candidate.canonical_bytes
                return True
            existing = self._validate(slot, existing, command_bytes, record)
            if self._canonical.get(slot) != existing.canonical_bytes:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            if existing == candidate and existing.canonical_bytes == candidate.canonical_bytes:
                return False
            raise _closed(CoordinatorStoreCode.CONFLICT)

    def get(self, slot, *, command_bytes, record):
        if type(slot) is not ReconciliationGrantSlotV1:
            raise _closed(CoordinatorStoreCode.BINDING_MISMATCH)
        with self._lock:
            value = self._values.get(slot)
            if value is None:
                return None
            value = self._validate(slot, value, command_bytes, record)
            if self._canonical.get(slot) != value.canonical_bytes:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            return value


__all__ = [
    "CoordinatorStoreCode",
    "CoordinatorStoreError",
    "InMemoryExecutionGrantStoreV1",
    "InMemoryPreparationStoreV1",
    "InMemoryReconciliationGrantStoreV1",
    "InMemoryWorkflowStoreV1",
]
