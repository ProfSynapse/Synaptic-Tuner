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
    ApplyArtifactVerificationTransitionV1,
    ApplyCancelEffectTransitionV1,
    ApplyProviderObservationTransitionV1,
    ApplyReverificationTransitionV1,
    ApplyStageEffectTransitionV1,
    ApplySubmitEffectTransitionV1,
    BeginPreparationTransitionV1,
    CoordinatorTransitionV1,
    ExecutionGrantSlotV1,
    ReconciliationGrantSlotV1,
    RecordStageIntentTransitionV1,
    RecordSubmitIntentTransitionV1,
    RecordCancelIntentTransitionV1,
)
from .model import WorkflowPhaseV1, WorkflowRecordV1
from .state_machine import (
    apply_artifact_verification,
    apply_cancel_effect_record,
    apply_provider_observation,
    apply_reverification,
    apply_stage_effect_record,
    apply_submit_effect_record,
    begin_preparation,
    record_stage_intent,
    record_submit_intent,
    record_cancel_intent,
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


_TRANSITION_TYPES = (
    BeginPreparationTransitionV1,
    RecordStageIntentTransitionV1,
    ApplyStageEffectTransitionV1,
    RecordSubmitIntentTransitionV1,
    ApplySubmitEffectTransitionV1,
    RecordCancelIntentTransitionV1,
    ApplyCancelEffectTransitionV1,
    ApplyProviderObservationTransitionV1,
    ApplyArtifactVerificationTransitionV1,
    ApplyReverificationTransitionV1,
)


def _intent_identity(intent):
    return {
        "kind": intent.kind.value,
        "effect_id": intent.effect_id,
        "command_digest": intent.command_digest,
        "command_bytes_digest": domain_digest(
            "synaptic-foundation-command-bytes/v1",
            intent.canonical_command_bytes,
        ),
        "foundation_binding_digests": [
            value.binding_digest for value in intent.foundation_bindings
        ],
        "foundation_outcome_digests": [
            value.outcome_digest for value in intent.foundation_outcomes
        ],
    }


def _transition_document(transition):
    document = {"kind": transition.kind.value}
    if type(transition) is BeginPreparationTransitionV1:
        return document
    if type(transition) is RecordStageIntentTransitionV1:
        document.update(
            preparation_digest=transition.preparation.preparation_digest,
            preparation_bytes_digest=domain_digest(
                "synaptic-coordinator-preparation-bytes/v1",
                transition.preparation.canonical_bytes,
            ),
            intent=_intent_identity(transition.intent),
        )
        return document
    if type(transition) in {
        ApplyStageEffectTransitionV1,
        ApplySubmitEffectTransitionV1,
        ApplyCancelEffectTransitionV1,
    }:
        document.update(
            foundation_record_digest=transition.record.record_digest,
            assessment_digest=transition.assessment.authenticated_assessment_digest,
            assessment_bytes_digest=domain_digest(
                "synaptic-coordinator-assessment-bytes/v1",
                transition.assessment.canonical_bytes,
            ),
        )
        return document
    if type(transition) in {
        RecordSubmitIntentTransitionV1,
        RecordCancelIntentTransitionV1,
    }:
        document["intent"] = _intent_identity(transition.intent)
        return document
    if type(transition) is ApplyProviderObservationTransitionV1:
        document.update(
            request_digest=transition.request.request_digest,
            request_bytes_digest=domain_digest(
                "synaptic-coordinator-provider-read-request-bytes/v1",
                transition.request.canonical_bytes,
            ),
            observation_digest=transition.observation.authenticated_observation_digest,
            observation_bytes_digest=domain_digest(
                "synaptic-coordinator-provider-observation-bytes/v1",
                transition.observation.canonical_bytes,
            ),
        )
        return document
    if type(transition) in {
        ApplyArtifactVerificationTransitionV1,
        ApplyReverificationTransitionV1,
    }:
        document.update(
            manifest_digest=transition.manifest.manifest_digest,
            manifest_evidence_digest=domain_digest(
                "synaptic-coordinator-manifest-evidence-bytes/v1",
                transition.manifest.canonical_evidence,
            ),
            verification_receipt_digest=(
                transition.receipt.authenticated_receipt_digest
            ),
            verification_receipt_bytes_digest=domain_digest(
                "synaptic-coordinator-verification-receipt-bytes/v1",
                transition.receipt.canonical_bytes,
            ),
        )
        return document
    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)


def _revalidate_transition(transition):
    if type(transition) not in _TRANSITION_TYPES:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    try:
        rebuilt = type(transition)(
            **{
                field.name: getattr(transition, field.name)
                for field in fields(type(transition))
            }
        )
        fingerprint = domain_digest(
            "synaptic-coordinator-transition/v1",
            canonical_bytes(_transition_document(rebuilt)),
        )
    except CoordinatorStoreError:
        raise
    except Exception:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
    if rebuilt != transition:
        raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
    return rebuilt, fingerprint


class InMemoryWorkflowStoreV1:
    def __init__(
        self,
        foundation_authenticator,
        assessment_authenticator,
        observation_authenticator,
        artifact_verifier,
    ):
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._observation_authenticator = observation_authenticator
        self._artifact_verifier = artifact_verifier
        self._records: dict[tuple[str, str], WorkflowRecordV1] = {}
        self._digests: dict[tuple[str, str], str] = {}
        self._plans: dict[tuple[str, str], tuple[str, str]] = {}
        self._history: dict[tuple[str, str], tuple[WorkflowRecordV1, ...]] = {}
        self._history_digests: dict[tuple[str, str], tuple[str, ...]] = {}
        self._transitions: dict[tuple[str, str], tuple[CoordinatorTransitionV1, ...]] = {}
        self._transition_digests: dict[tuple[str, str], tuple[str, ...]] = {}
        self._lock = RLock()

    def _validate_history(
        self, key: tuple[str, str], retained: WorkflowRecordV1
    ) -> tuple[WorkflowRecordV1, ...]:
        history = self._history.get(key)
        history_digests = self._history_digests.get(key)
        transitions = self._transitions.get(key)
        transition_digests = self._transition_digests.get(key)
        if (
            type(history) is not tuple
            or type(history_digests) is not tuple
            or type(transitions) is not tuple
            or type(transition_digests) is not tuple
            or not history
            or len(history_digests) != len(history)
            or len(transitions) != len(history) - 1
            or len(transition_digests) != len(transitions)
        ):
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
        try:
            genesis = _revalidate_workflow(history[0])
            if (
                genesis.phase is not WorkflowPhaseV1.PLANNED
                or genesis.revision != 0
                or _workflow_key(genesis.run) != key
            ):
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            previous = genesis
            if history_digests[0] != genesis.record_digest:
                raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
            for index, transition in enumerate(transitions):
                transition, fingerprint = _revalidate_transition(transition)
                if transition_digests[index] != fingerprint:
                    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
                following = _revalidate_workflow(history[index + 1])
                if (
                    following.revision != previous.revision + 1
                    or following.run != genesis.run
                    or following.plan_fingerprint != genesis.plan_fingerprint
                    or history_digests[index + 1] != following.record_digest
                ):
                    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
                replayed = self._replay(previous, transition)
                if replayed != following or replayed.record_digest != following.record_digest:
                    raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
                previous = following
        except CoordinatorStoreError:
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
        except Exception:
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR) from None
        if previous != retained or previous.record_digest != retained.record_digest:
            raise _closed(CoordinatorStoreCode.INTEGRITY_ERROR)
        return history

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
        self._validate_history(key, retained)
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
            self._history[key] = (candidate,)
            self._history_digests[key] = (candidate.record_digest,)
            self._transitions[key] = ()
            self._transition_digests[key] = ()
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

    def is_descendant(
        self, ancestor: WorkflowRecordV1, descendant: WorkflowRecordV1
    ) -> bool:
        ancestor = _revalidate_workflow(ancestor)
        descendant = _revalidate_workflow(descendant)
        ancestor_key = _workflow_key(ancestor.run)
        if _workflow_key(descendant.run) != ancestor_key:
            return False
        with self._lock:
            if ancestor_key not in self._records:
                return False
            retained = self._retained(ancestor_key)
            history = self._validate_history(ancestor_key, retained)
            if (
                ancestor.revision >= descendant.revision
                or descendant.revision >= len(history)
            ):
                return False
            stored_ancestor = history[ancestor.revision]
            stored_descendant = history[descendant.revision]
            return bool(
                stored_ancestor == ancestor
                and stored_ancestor.record_digest == ancestor.record_digest
                and stored_descendant == descendant
                and stored_descendant.record_digest == descendant.record_digest
            )

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
            if type(transition) is RecordCancelIntentTransitionV1:
                return record_cancel_intent(current, transition.intent)
            if type(transition) is ApplyCancelEffectTransitionV1:
                return apply_cancel_effect_record(
                    current,
                    transition.record,
                    transition.assessment,
                    self._foundation_authenticator,
                    self._assessment_authenticator,
                )
            if type(transition) is ApplyProviderObservationTransitionV1:
                return apply_provider_observation(
                    current,
                    transition.request,
                    transition.observation,
                    self._observation_authenticator,
                )
            if type(transition) is ApplyArtifactVerificationTransitionV1:
                return apply_artifact_verification(
                    current,
                    transition.manifest,
                    transition.receipt,
                    self._artifact_verifier,
                )
            if type(transition) is ApplyReverificationTransitionV1:
                return apply_reverification(
                    current,
                    transition.manifest,
                    transition.receipt,
                    self._artifact_verifier,
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
            transition, transition_digest = _revalidate_transition(transition)
            history = self._history[key]
            history_digests = self._history_digests[key]
            transitions = self._transitions[key]
            transition_digests = self._transition_digests[key]
            self._records[key] = replacement
            self._digests[key] = replacement.record_digest
            self._history[key] = history + (replacement,)
            self._history_digests[key] = history_digests + (replacement.record_digest,)
            self._transitions[key] = transitions + (transition,)
            self._transition_digests[key] = transition_digests + (transition_digest,)
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
