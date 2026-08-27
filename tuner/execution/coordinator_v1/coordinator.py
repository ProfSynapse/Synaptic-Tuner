"""Non-authoritative coordinator transition and durable-slot descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderDescriptor
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.foundation_v2.canonical import (
    DiagnosticCode,
    FoundationError,
    canonical_bytes,
    digest_text,
    domain_digest,
    exact_integer,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import (
    build_stage_command,
    build_submit_command,
    parse_exact_command,
)
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import AuthenticatedReceiptV2
from tuner.execution.foundation_v2.references import StagePredecessorV2
from tuner.execution.foundation_v2.repository import (
    DispatchState,
    EffectRecordV2,
    EffectState,
)

from .model import (
    AuthenticatedFoundationRecordAssessmentV1,
    EffectIntentV1,
    ReceiptFreshnessV1,
    WorkflowPhaseV1,
    WorkflowRecordV1,
)
from .state_machine import (
    apply_stage_effect_record,
    apply_submit_effect_record,
    begin_preparation,
    record_stage_intent,
    record_submit_intent,
)


class CoordinatorTransitionKindV1(str, Enum):
    BEGIN_PREPARATION = "begin_preparation"
    RECORD_STAGE_INTENT = "record_stage_intent"
    APPLY_STAGE_EFFECT = "apply_stage_effect"
    RECORD_SUBMIT_INTENT = "record_submit_intent"
    APPLY_SUBMIT_EFFECT = "apply_submit_effect"


@dataclass(frozen=True, slots=True)
class BeginPreparationTransitionV1:
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.BEGIN_PREPARATION

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.BEGIN_PREPARATION:
            raise ValueError("invalid begin-preparation transition")


@dataclass(frozen=True, slots=True)
class RecordStageIntentTransitionV1:
    preparation: CanonicalPreparationV2
    intent: EffectIntentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.RECORD_STAGE_INTENT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.RECORD_STAGE_INTENT:
            raise ValueError("invalid stage-intent transition")
        if type(self.preparation) is not CanonicalPreparationV2:
            raise TypeError("exact preparation required")
        if type(self.intent) is not EffectIntentV1:
            raise TypeError("exact effect intent required")


@dataclass(frozen=True, slots=True)
class ApplyStageEffectTransitionV1:
    record: EffectRecordV2
    assessment: AuthenticatedFoundationRecordAssessmentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.APPLY_STAGE_EFFECT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.APPLY_STAGE_EFFECT:
            raise ValueError("invalid stage-effect transition")
        if type(self.record) is not EffectRecordV2:
            raise TypeError("exact Foundation record required")
        if type(self.assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation assessment required")


@dataclass(frozen=True, slots=True)
class RecordSubmitIntentTransitionV1:
    intent: EffectIntentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.RECORD_SUBMIT_INTENT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.RECORD_SUBMIT_INTENT:
            raise ValueError("invalid submit-intent transition")
        if type(self.intent) is not EffectIntentV1:
            raise TypeError("exact effect intent required")


@dataclass(frozen=True, slots=True)
class ApplySubmitEffectTransitionV1:
    record: EffectRecordV2
    assessment: AuthenticatedFoundationRecordAssessmentV1
    kind: CoordinatorTransitionKindV1 = CoordinatorTransitionKindV1.APPLY_SUBMIT_EFFECT

    def __post_init__(self) -> None:
        if self.kind is not CoordinatorTransitionKindV1.APPLY_SUBMIT_EFFECT:
            raise ValueError("invalid submit-effect transition")
        if type(self.record) is not EffectRecordV2:
            raise TypeError("exact Foundation record required")
        if type(self.assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation assessment required")


CoordinatorTransitionV1: TypeAlias = (
    BeginPreparationTransitionV1
    | RecordStageIntentTransitionV1
    | ApplyStageEffectTransitionV1
    | RecordSubmitIntentTransitionV1
    | ApplySubmitEffectTransitionV1
)


@dataclass(frozen=True, slots=True)
class ExecutionGrantSlotV1:
    effect_id: str
    command_digest: str
    command_bytes_digest: str

    def __post_init__(self) -> None:
        safe_ref(self.effect_id, "effect_id")
        digest_text(self.command_digest, "command_digest")
        digest_text(self.command_bytes_digest, "command_bytes_digest")


@dataclass(frozen=True, slots=True)
class ReconciliationGrantSlotV1:
    effect_id: str
    command_digest: str
    command_bytes_digest: str
    generation: int
    ownership_epoch: int
    prior_claim_digest: str | None
    predecessor_grant_digest: str | None

    def __post_init__(self) -> None:
        safe_ref(self.effect_id, "effect_id")
        digest_text(self.command_digest, "command_digest")
        digest_text(self.command_bytes_digest, "command_bytes_digest")
        exact_integer(self.generation, "generation", minimum=1)
        exact_integer(self.ownership_epoch, "ownership_epoch", minimum=1)
        if (self.prior_claim_digest is None) != (self.predecessor_grant_digest is None):
            raise ValueError("reconciliation lineage must be paired")
        if self.prior_claim_digest is not None:
            digest_text(self.prior_claim_digest, "prior_claim_digest")
            digest_text(self.predecessor_grant_digest, "predecessor_grant_digest")


class CoordinatorCodeV1(str, Enum):
    INVALID_INPUT = "invalid_input"
    PLAN_MISSING = "plan_missing"
    CONTEXT_MISSING = "context_missing"
    PREFLIGHT_INVALID = "preflight_invalid"
    BINDING_MISMATCH = "binding_mismatch"
    WORKFLOW_CONFLICT = "workflow_conflict"
    STORE_INTEGRITY = "store_integrity"
    GRANT_MISSING = "grant_missing"
    GRANT_INVALID = "grant_invalid"
    QUIESCENCE_UNPROVEN = "quiescence_unproven"
    FOUNDATION_INTERRUPTED = "foundation_interrupted"
    RETRY_EXHAUSTED = "retry_exhausted"


class CoordinatorErrorV1(ValueError):
    """Closed internal coordinator failure."""

    def __init__(self, code: CoordinatorCodeV1):
        if type(code) is not CoordinatorCodeV1:
            raise TypeError("code must be exact CoordinatorCodeV1")
        self.code = code
        super().__init__(code.value)


def _error(code: CoordinatorCodeV1) -> CoordinatorErrorV1:
    return CoordinatorErrorV1(code)


def _call(operation, code: CoordinatorCodeV1 = CoordinatorCodeV1.STORE_INTEGRITY):
    failed = None
    value = None
    try:
        value = operation()
    except CoordinatorErrorV1 as error:
        failed = CoordinatorErrorV1(error.code)
    except FoundationError as error:
        mapped = (
            CoordinatorCodeV1.FOUNDATION_INTERRUPTED
            if error.code is DiagnosticCode.RECONCILIATION_INTERRUPTED
            else code
        )
        failed = CoordinatorErrorV1(mapped)
    except Exception:
        failed = CoordinatorErrorV1(code)
    if failed is not None:
        raise failed
    return value


def _command_bytes_digest(command_bytes: bytes) -> str:
    if type(command_bytes) is not bytes:
        raise _error(CoordinatorCodeV1.BINDING_MISMATCH)
    return domain_digest("synaptic-foundation-command-bytes/v1", command_bytes)


def _intent_prefix(before: EffectIntentV1 | None, after: EffectIntentV1 | None) -> bool:
    if before is None:
        return True
    if after is None:
        return False
    return (
        before.kind,
        before.effect_id,
        before.command_digest,
        before.canonical_command_bytes,
    ) == (
        after.kind,
        after.effect_id,
        after.command_digest,
        after.canonical_command_bytes,
    ) and after.foundation_bindings[: len(before.foundation_bindings)] == before.foundation_bindings and after.foundation_outcomes[: len(before.foundation_outcomes)] == before.foundation_outcomes


def _workflow_descends(before: WorkflowRecordV1, after: WorkflowRecordV1) -> bool:
    if type(before) is not WorkflowRecordV1 or type(after) is not WorkflowRecordV1:
        return False
    if after.revision < before.revision:
        return False
    if after.revision == before.revision:
        return after == before
    immutable = (
        "schema_version", "run", "plan_fingerprint", "preflight_digest",
        "provider", "provider_context_digest", "provider_descriptor_digest",
    )
    if any(getattr(before, name) != getattr(after, name) for name in immutable):
        return False
    if before.preparation_digest is not None and after.preparation_digest != before.preparation_digest:
        return False
    if not _intent_prefix(before.stage, after.stage) or not _intent_prefix(before.submit, after.submit):
        return False
    if before.provider_stage_ref is not None and after.provider_stage_ref != before.provider_stage_ref:
        return False
    if before.provider_run_ref is not None and after.provider_run_ref != before.provider_run_ref:
        return False
    prefix_fields = (
        "run_observation_digests", "provider_run_observations",
        "verification_receipts", "verification_receipt_digests", "diagnostic_codes",
    )
    if any(
        getattr(after, name)[: len(getattr(before, name))] != getattr(before, name)
        for name in prefix_fields
    ):
        return False
    ranks = {
        WorkflowPhaseV1.PLANNED: 0,
        WorkflowPhaseV1.PREPARING: 1,
        WorkflowPhaseV1.STAGE_INTENT_RECORDED: 2,
        WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED: 2,
        WorkflowPhaseV1.STAGED: 3,
        WorkflowPhaseV1.SUBMIT_INTENT_RECORDED: 4,
        WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED: 4,
        WorkflowPhaseV1.QUEUED: 5,
        WorkflowPhaseV1.FAILED: 5,
        WorkflowPhaseV1.CONTRADICTED: 5,
    }
    return before.phase in ranks and after.phase in ranks and ranks[after.phase] >= ranks[before.phase]


class TrainingCoordinatorV1:
    """Deterministic internal stage-to-submit coordinator."""

    _MAX_STEPS = 32
    _START_TERMINALS = frozenset(
        {
            WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED,
            WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED,
            WorkflowPhaseV1.QUEUED,
            WorkflowPhaseV1.FAILED,
            WorkflowPhaseV1.CONTRADICTED,
        }
    )

    def __init__(
        self,
        planning,
        planning_store,
        workflow_store,
        preparation_store,
        execution_grant_store,
        reconciliation_grant_store,
        binding_resolver,
        materializer,
        authorization,
        foundation,
        foundation_authenticator,
        clock,
        identity,
    ):
        self._planning = planning
        self._plans = planning_store
        self._workflows = workflow_store
        self._preparations = preparation_store
        self._execution_grants = execution_grant_store
        self._reconciliation_grants = reconciliation_grant_store
        self._bindings = binding_resolver
        self._materializer = materializer
        self._authorization = authorization
        self._foundation = foundation
        self._foundation_authenticator = foundation_authenticator
        self._clock = clock
        self._identity = identity

    def _load_basis(self, plan_fingerprint, *, expected_plan=None):
        plan = _call(
            lambda: self._plans.get_plan(plan_fingerprint),
            CoordinatorCodeV1.PLAN_MISSING,
        )
        if plan is None:
            raise _error(CoordinatorCodeV1.PLAN_MISSING)
        if type(plan) is not TrainingPlan or plan.plan_fingerprint != plan_fingerprint:
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        if expected_plan is not None and plan != expected_plan:
            raise _error(CoordinatorCodeV1.WORKFLOW_CONFLICT)
        context = _call(
            lambda: self._plans.get_context(plan.provider_plan.context_digest),
            CoordinatorCodeV1.CONTEXT_MISSING,
        )
        if context is None:
            raise _error(CoordinatorCodeV1.CONTEXT_MISSING)
        if type(context) is not ProviderPlanContextV1 or (
            context.provider_context_digest != plan.provider_plan.context_digest
            or context.basis_digest != plan.basis.basis_digest
        ):
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        descriptor = _call(
            lambda: self._planning.describe(context.provider),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        binding = _call(
            lambda: self._bindings.resolve(context.provider, context),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        if type(descriptor) is not ProviderDescriptor or (
            descriptor.descriptor_digest != context.descriptor_digest
            or binding.provider != context.provider
            or binding.provider_descriptor_digest != descriptor.descriptor_digest
            or binding.profile_digest != context.profile_digest
        ):
            raise _error(CoordinatorCodeV1.BINDING_MISMATCH)
        return plan, context, descriptor, binding

    def _cas(self, current, replacement, transition):
        if replacement is current:
            return current
        for _ in range(3):
            failure = None
            swapped = False
            try:
                swapped = self._workflows.compare_and_swap(
                    current, replacement, transition=transition
                )
            except Exception:
                failure = CoordinatorErrorV1(CoordinatorCodeV1.STORE_INTEGRITY)
            if swapped is True:
                return replacement
            retained = _call(
                lambda: self._workflows.get(current.run),
                CoordinatorCodeV1.STORE_INTEGRITY,
            )
            if retained == replacement:
                return retained
            if retained == current:
                continue
            if type(retained) is WorkflowRecordV1 and retained.revision > replacement.revision and _workflow_descends(replacement, retained):
                return retained
            if failure is not None:
                raise failure
        raise _error(CoordinatorCodeV1.RETRY_EXHAUSTED)

    @staticmethod
    def _validate_preparation(candidate, plan, workflow, binding):
        if type(candidate) is not CanonicalPreparationV2:
            raise _error(CoordinatorCodeV1.BINDING_MISMATCH)
        basis = plan.basis
        actual = (
            candidate.provider, candidate.scope, candidate.project_ref,
            candidate.run_id, candidate.plan_fingerprint, candidate.source_digest,
            candidate.workload_digest, candidate.runtime_digest,
            candidate.resource_digest, candidate.artifact_contract_digest,
            candidate.quote_digest, candidate.secret_requirements_digest,
        )
        expected = (
            binding.provider, binding.scope, workflow.run.project_ref,
            workflow.run.run_id, plan.plan_fingerprint, basis.source_digest,
            basis.workload_digest, basis.runtime_digest,
            binding.resource_digest, basis.artifact_policy_digest,
            binding.quote_digest, binding.secret_requirements_digest,
        )
        if actual != expected:
            raise _error(CoordinatorCodeV1.BINDING_MISMATCH)
        return candidate

    def _preparation(self, plan, workflow, binding):
        if workflow.preparation_digest is not None:
            retained = _call(
                lambda: self._preparations.get(workflow.preparation_digest),
                CoordinatorCodeV1.STORE_INTEGRITY,
            )
            if retained is None or retained.preparation_digest != workflow.preparation_digest:
                raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
            return self._validate_preparation(retained, plan, workflow, binding)
        candidate = _call(
            lambda: self._materializer.prepare(plan, workflow.run, binding),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        self._validate_preparation(candidate, plan, workflow, binding)
        _call(lambda: self._preparations.put_if_absent(candidate), CoordinatorCodeV1.STORE_INTEGRITY)
        retained = _call(
            lambda: self._preparations.get(candidate.preparation_digest),
            CoordinatorCodeV1.STORE_INTEGRITY,
        )
        if retained != candidate:
            raise _error(CoordinatorCodeV1.WORKFLOW_CONFLICT)
        return retained

    @staticmethod
    def _nonce(kind, workflow, preparation):
        return domain_digest(
            "synaptic-coordinator-invocation-nonce/v1",
            canonical_bytes(
                {
                    "kind": kind.value,
                    "run": workflow.run.to_dict(),
                    "plan_fingerprint": workflow.plan_fingerprint,
                    "preparation_digest": preparation.preparation_digest,
                }
            ),
        )

    def _stage_intent(self, workflow, preparation, binding):
        payload = _call(
            lambda: self._materializer.payload(preparation, EffectKind.STAGE),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        command = _call(
            lambda: build_stage_command(
                preparation,
                self._nonce(EffectKind.STAGE, workflow, preparation),
                payload,
                binding.executor_descriptor,
            ),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        return EffectIntentV1.from_command_bytes(command.canonical_bytes)

    def _fresh_stage_predecessor_unchecked(self, workflow, record, assessment, preparation):
        if (
            type(record) is not EffectRecordV2
            or record.state is not EffectState.FOUND
            or workflow.provider_stage_ref is None
            or self._foundation.authenticate(assessment) is not True
        ):
            raise _error(CoordinatorCodeV1.GRANT_INVALID)
        digest = workflow.provider_stage_ref.authenticated_receipt_digest
        matches = []
        for receipt, assessed in zip(
            record.results, assessment.content.receipt_assessments, strict=True
        ):
            if receipt.authenticated_receipt_digest != digest:
                continue
            if (
                type(receipt) is not AuthenticatedReceiptV2
                or self._foundation_authenticator.authenticate_receipt(receipt) is not True
                or assessed.authenticated_receipt_digest != digest
                or assessed.freshness is not ReceiptFreshnessV1.FRESH
                or receipt.content.disposition is not ObservationDisposition.FOUND
                or receipt.content.stage_ref != workflow.provider_stage_ref.reference
                or receipt.content.semantic_digest not in record.terminal_content_digests
            ):
                raise _error(CoordinatorCodeV1.GRANT_INVALID)
            matches.append(receipt)
        if len(matches) != 1:
            raise _error(CoordinatorCodeV1.GRANT_INVALID)
        command = parse_exact_command(record.command_bytes)
        receipt = matches[0]
        return StagePredecessorV2(
            preparation.provider.provider_id,
            preparation.provider.profile_ref,
            preparation.scope.account_ref,
            preparation.scope.namespace_ref,
            workflow.run.project_ref,
            workflow.run.run_id,
            workflow.plan_fingerprint,
            preparation.preparation_digest,
            preparation.workload_digest,
            command.operation.effect.effect_id,
            receipt.authenticated_receipt_digest,
            record.record_digest,
        )

    def _fresh_stage_predecessor(self, workflow, record, assessment, preparation):
        return _call(
            lambda: self._fresh_stage_predecessor_unchecked(
                workflow, record, assessment, preparation
            ),
            CoordinatorCodeV1.GRANT_INVALID,
        )

    def _submit_intent(self, workflow, preparation, binding):
        stage_record = _call(
            lambda: self._foundation.get(workflow.stage.effect_id),
            CoordinatorCodeV1.GRANT_INVALID,
        )
        if stage_record is None:
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        assessment = _call(
            lambda: self._foundation.assess(stage_record),
            CoordinatorCodeV1.GRANT_INVALID,
        )
        predecessor = self._fresh_stage_predecessor(
            workflow, stage_record, assessment, preparation
        )
        payload = _call(
            lambda: self._materializer.payload(preparation, EffectKind.SUBMIT),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        command = _call(
            lambda: build_submit_command(
                preparation,
                self._nonce(EffectKind.SUBMIT, workflow, preparation),
                payload,
                binding.executor_descriptor,
                predecessor,
            ),
            CoordinatorCodeV1.BINDING_MISMATCH,
        )
        return EffectIntentV1.from_command_bytes(command.canonical_bytes)

    def _execution_grant(self, intent, preflight_digest):
        slot = ExecutionGrantSlotV1(
            intent.effect_id,
            intent.command_digest,
            _command_bytes_digest(intent.canonical_command_bytes),
        )
        winner = _call(
            lambda: self._execution_grants.get(slot, intent.canonical_command_bytes),
            CoordinatorCodeV1.GRANT_INVALID,
        )
        if winner is None:
            candidate = _call(
                lambda: self._authorization.issue_effect_grant(
                    intent.canonical_command_bytes,
                    preflight_digest=preflight_digest,
                    now_epoch=self._clock.now_epoch(),
                ),
                CoordinatorCodeV1.GRANT_INVALID,
            )
            try:
                self._execution_grants.put_if_absent(
                    slot, candidate, intent.canonical_command_bytes
                )
            except Exception:
                pass
            winner = _call(
                lambda: self._execution_grants.get(slot, intent.canonical_command_bytes),
                CoordinatorCodeV1.GRANT_INVALID,
            )
        if winner is None:
            raise _error(CoordinatorCodeV1.GRANT_MISSING)
        return winner

    def _apply_effect(self, workflow, *, stage):
        intent = workflow.stage if stage else workflow.submit
        if intent is None:
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        record = _call(lambda: self._foundation.get(intent.effect_id))
        if record is not None and record.dispatch in {
            DispatchState.OWNED_NOT_STARTED,
            DispatchState.OWNED_IN_FLIGHT,
        }:
            raise _error(CoordinatorCodeV1.FOUNDATION_INTERRUPTED)
        if record is None:
            grant = self._execution_grant(intent, workflow.preflight_digest)
            _call(
                lambda: self._foundation.execute(
                    intent.canonical_command_bytes,
                    grant,
                    now_epoch=self._clock.now_epoch(),
                ),
                CoordinatorCodeV1.FOUNDATION_INTERRUPTED,
            )
            record = _call(lambda: self._foundation.get(intent.effect_id))
        if record is None:
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        assessment = _call(lambda: self._foundation.assess(record))
        reduction_failed = False
        replacement = None
        if stage:
            try:
                replacement = apply_stage_effect_record(
                    workflow, record, assessment,
                    self._foundation_authenticator, self._foundation,
                )
            except Exception:
                reduction_failed = True
            transition = ApplyStageEffectTransitionV1(record, assessment)
        else:
            try:
                replacement = apply_submit_effect_record(
                    workflow, record, assessment,
                    self._foundation_authenticator, self._foundation,
                )
            except Exception:
                reduction_failed = True
            transition = ApplySubmitEffectTransitionV1(record, assessment)
        if reduction_failed:
            retained = _call(
                lambda: self._workflows.get(workflow.run),
                CoordinatorCodeV1.STORE_INTEGRITY,
            )
            if (
                type(retained) is WorkflowRecordV1
                and retained.revision > workflow.revision
                and _workflow_descends(workflow, retained)
            ):
                return retained
            if retained == workflow:
                raise _error(CoordinatorCodeV1.FOUNDATION_INTERRUPTED)
            raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
        return self._cas(workflow, replacement, transition)

    def _bootstrap(self, plan, preflight):
        if type(plan) is not TrainingPlan or type(preflight) is not TrainingPreflight:
            raise _error(CoordinatorCodeV1.INVALID_INPUT)
        if not preflight.binds(plan) or preflight.ready is not True or preflight.is_expired(self._clock.now_iso()):
            raise _error(CoordinatorCodeV1.PREFLIGHT_INVALID)
        plan, context, descriptor, binding = self._load_basis(
            plan.plan_fingerprint, expected_plan=plan
        )
        preflight_digest = _call(
            lambda: self._authorization.commit_preflight(plan, preflight),
            CoordinatorCodeV1.PREFLIGHT_INVALID,
        )
        try:
            digest_text(preflight_digest, "preflight_digest")
        except Exception:
            raise _error(CoordinatorCodeV1.PREFLIGHT_INVALID) from None
        run = _call(lambda: self._identity.for_plan(plan), CoordinatorCodeV1.BINDING_MISMATCH)
        if type(run) is not TrainingRunRef or run.project_ref != plan.basis.project_ref:
            raise _error(CoordinatorCodeV1.BINDING_MISMATCH)
        planned = WorkflowRecordV1.planned(
            run=run, plan=plan, preflight_digest=preflight_digest,
            context=context, provider=context.provider, descriptor=descriptor,
        )
        existing = _call(
            lambda: self._workflows.get_by_plan(run.project_ref, plan.plan_fingerprint),
            CoordinatorCodeV1.STORE_INTEGRITY,
        )
        if existing is None:
            create_failed = False
            try:
                self._workflows.create(planned)
            except Exception:
                create_failed = True
        workflow = _call(
            lambda: self._workflows.get_by_plan(run.project_ref, plan.plan_fingerprint),
            CoordinatorCodeV1.STORE_INTEGRITY,
        )
        if workflow is None and existing is None and create_failed:
            raise _error(CoordinatorCodeV1.WORKFLOW_CONFLICT)
        if type(workflow) is not WorkflowRecordV1 or workflow.run != run or workflow.preflight_digest != preflight_digest:
            raise _error(CoordinatorCodeV1.WORKFLOW_CONFLICT)
        if not _workflow_descends(planned, workflow):
            raise _error(CoordinatorCodeV1.WORKFLOW_CONFLICT)
        return workflow, plan, binding

    def _progress_start(self, workflow, plan, binding):
        for _ in range(self._MAX_STEPS):
            if workflow.phase in self._START_TERMINALS:
                return workflow
            if workflow.phase is WorkflowPhaseV1.PLANNED:
                workflow = self._cas(
                    workflow, begin_preparation(workflow),
                    BeginPreparationTransitionV1(),
                )
                continue
            preparation = self._preparation(plan, workflow, binding)
            if workflow.phase is WorkflowPhaseV1.PREPARING:
                intent = self._stage_intent(workflow, preparation, binding)
                workflow = self._cas(
                    workflow, record_stage_intent(workflow, preparation, intent),
                    RecordStageIntentTransitionV1(preparation, intent),
                )
                continue
            if workflow.phase is WorkflowPhaseV1.STAGE_INTENT_RECORDED:
                workflow = self._apply_effect(workflow, stage=True)
                continue
            if workflow.phase is WorkflowPhaseV1.STAGED:
                intent = self._submit_intent(workflow, preparation, binding)
                workflow = self._cas(
                    workflow, record_submit_intent(workflow, intent),
                    RecordSubmitIntentTransitionV1(intent),
                )
                continue
            if workflow.phase is WorkflowPhaseV1.SUBMIT_INTENT_RECORDED:
                workflow = self._apply_effect(workflow, stage=False)
                continue
            raise _error(CoordinatorCodeV1.INVALID_INPUT)
        raise _error(CoordinatorCodeV1.RETRY_EXHAUSTED)

    def start(self, plan: TrainingPlan, preflight: TrainingPreflight) -> WorkflowRecordV1:
        workflow, retained_plan, binding = self._bootstrap(plan, preflight)
        return self._progress_start(workflow, retained_plan, binding)

    @staticmethod
    def _reconciliation_slot(record, intent):
        command_digest = _command_bytes_digest(intent.canonical_command_bytes)
        current = record.reconciliation
        claims = record.reconciliation_claims
        if current is None:
            generation = ownership_epoch = 1
            prior_claim = predecessor = None
        elif current.active:
            generation, ownership_epoch = current.generation, current.ownership_epoch
            if len(current.grant_lineage) > 1:
                prior_claim = current.claim_digest
                predecessor = current.grant_lineage[-2].grant_digest
            elif len(claims) > 1:
                prior_claim = claims[-2].claim_digest
                predecessor = claims[-2].grant_digest
            else:
                prior_claim = predecessor = None
        elif current.completed:
            generation, ownership_epoch = current.generation + 1, current.ownership_epoch + 1
            prior_claim, predecessor = current.claim_digest, current.grant_digest
        else:
            generation, ownership_epoch = current.generation, current.ownership_epoch
            prior_claim, predecessor = current.claim_digest, current.grant_digest
        return ReconciliationGrantSlotV1(
            intent.effect_id, intent.command_digest, command_digest,
            generation, ownership_epoch, prior_claim, predecessor,
        )

    def _reconciliation_grant(self, record, intent, binding):
        slot = self._reconciliation_slot(record, intent)
        winner = _call(
            lambda: self._reconciliation_grants.get(
                slot, command_bytes=intent.canonical_command_bytes, record=record
            ),
            CoordinatorCodeV1.GRANT_INVALID,
        )
        if winner is None:
            candidate = _call(
                lambda: self._authorization.issue_reconciliation_grant(
                    record, binding, slot=slot, now_epoch=self._clock.now_epoch()
                ),
                CoordinatorCodeV1.GRANT_INVALID,
            )
            try:
                self._reconciliation_grants.put_if_absent(
                    slot, candidate, intent.canonical_command_bytes, record
                )
            except Exception:
                pass
            winner = _call(
                lambda: self._reconciliation_grants.get(
                    slot, command_bytes=intent.canonical_command_bytes, record=record
                ),
                CoordinatorCodeV1.GRANT_INVALID,
            )
        if winner is None:
            raise _error(CoordinatorCodeV1.GRANT_MISSING)
        return winner

    def reconcile(self, run: TrainingRunRef) -> WorkflowRecordV1:
        if type(run) is not TrainingRunRef:
            raise _error(CoordinatorCodeV1.INVALID_INPUT)
        workflow = _call(lambda: self._workflows.get(run), CoordinatorCodeV1.STORE_INTEGRITY)
        if type(workflow) is not WorkflowRecordV1:
            raise _error(CoordinatorCodeV1.INVALID_INPUT)
        plan, _, _, binding = self._load_basis(workflow.plan_fingerprint)
        looked_up: set[str] = set()
        for _ in range(self._MAX_STEPS):
            if workflow.phase in {
                WorkflowPhaseV1.STAGED, WorkflowPhaseV1.SUBMIT_INTENT_RECORDED,
                WorkflowPhaseV1.QUEUED, WorkflowPhaseV1.FAILED,
                WorkflowPhaseV1.CONTRADICTED,
            }:
                # Reconciliation never re-executes the existing effect. A newly
                # discovered stage may, however, advance into the distinct submit
                # effect through the normal durable get-before-execute path.
                if workflow.phase in {
                    WorkflowPhaseV1.STAGED,
                    WorkflowPhaseV1.SUBMIT_INTENT_RECORDED,
                }:
                    return self._progress_start(workflow, plan, binding)
                return workflow
            stage = workflow.phase in {
                WorkflowPhaseV1.STAGE_INTENT_RECORDED,
                WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED,
            }
            submit = workflow.phase is WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED
            if not stage and not submit:
                raise _error(CoordinatorCodeV1.INVALID_INPUT)
            intent = workflow.stage if stage else workflow.submit
            record = _call(lambda: self._foundation.get(intent.effect_id))
            if record is None:
                raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
            if record.dispatch is DispatchState.ORPHANED_UNPROVEN:
                _call(
                    lambda: self._foundation.recover_orphan(
                        intent.effect_id, now_epoch=self._clock.now_epoch()
                    ),
                    CoordinatorCodeV1.QUIESCENCE_UNPROVEN,
                )
                record = _call(lambda: self._foundation.get(intent.effect_id))
                if record is None:
                    raise _error(CoordinatorCodeV1.STORE_INTEGRITY)
            assessment = _call(lambda: self._foundation.assess(record))
            reducer = apply_stage_effect_record if stage else apply_submit_effect_record
            transition_type = ApplyStageEffectTransitionV1 if stage else ApplySubmitEffectTransitionV1
            replacement = _call(
                lambda: reducer(
                    workflow, record, assessment,
                    self._foundation_authenticator, self._foundation,
                ),
                CoordinatorCodeV1.STORE_INTEGRITY,
            )
            retained_receipts = tuple(
                value.authenticated_receipt_digest for value in record.results
            )
            workflow_receipts = (
                () if not intent.foundation_bindings
                else intent.foundation_bindings[-1].authenticated_receipt_digests
            )
            claim_only_interruption = (
                record.reconciliation is not None
                and not record.reconciliation.active
                and not record.reconciliation.completed
                and retained_receipts == workflow_receipts
            )
            if replacement is not workflow and not claim_only_interruption:
                workflow = self._cas(
                    workflow, replacement, transition_type(record, assessment)
                )
                continue
            if workflow.phase not in {
                WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED,
                WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED,
            }:
                return workflow
            if intent.effect_id in looked_up:
                return workflow
            grant = self._reconciliation_grant(record, intent, binding)
            continuation = record.reconciliation
            if continuation is not None and continuation.completed:
                continuation = None
            looked_up.add(intent.effect_id)
            interrupted = None
            try:
                _call(
                    lambda: self._foundation.reconcile(
                        intent.canonical_command_bytes, grant,
                        now_epoch=self._clock.now_epoch(), continuation=continuation,
                    ),
                    CoordinatorCodeV1.FOUNDATION_INTERRUPTED,
                )
            except CoordinatorErrorV1 as error:
                interrupted = error
            if interrupted is not None:
                durable = _call(
                    lambda: self._foundation.get(intent.effect_id),
                    CoordinatorCodeV1.FOUNDATION_INTERRUPTED,
                )
                claim = None if durable is None else durable.reconciliation
                if (
                    durable is not None
                    and claim is not None
                    and not claim.active
                    and not claim.completed
                ):
                    return workflow
                raise interrupted
            newer = _call(lambda: self._foundation.get(intent.effect_id))
            if newer is None or newer.record_digest == record.record_digest:
                raise _error(CoordinatorCodeV1.FOUNDATION_INTERRUPTED)
        raise _error(CoordinatorCodeV1.RETRY_EXHAUSTED)


__all__ = [
    "ApplyStageEffectTransitionV1",
    "ApplySubmitEffectTransitionV1",
    "BeginPreparationTransitionV1",
    "CoordinatorTransitionKindV1",
    "CoordinatorTransitionV1",
    "CoordinatorCodeV1",
    "CoordinatorErrorV1",
    "ExecutionGrantSlotV1",
    "ReconciliationGrantSlotV1",
    "RecordStageIntentTransitionV1",
    "RecordSubmitIntentTransitionV1",
    "TrainingCoordinatorV1",
]
