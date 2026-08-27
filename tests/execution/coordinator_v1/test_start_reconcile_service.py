from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.coordinator_v1.coordinator import (
    ApplyStageEffectTransitionV1,
    CoordinatorCodeV1,
    CoordinatorErrorV1,
    TrainingCoordinatorV1,
)
from tuner.execution.coordinator_v1.foundation import (
    ComposedEffectFoundationV1,
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1, WorkflowPhaseV1
from tuner.execution.coordinator_v1.stores import (
    InMemoryExecutionGrantStoreV1,
    InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1,
    InMemoryWorkflowStoreV1,
)
from tuner.execution.foundation_v2.authority import ReconciliationGrantContentV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1, parse_exact_command
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.references import ProviderStageRefV1, ScopedProviderRunRefV1
from tuner.execution.foundation_v2.repository import DispatchState

from tests.execution.coordinator_v1.test_state_machine import (
    CONTEXT,
    D,
    DESC,
    PLAN,
    PROVIDER,
    RUN,
    SCOPE,
)
from tests.execution.foundation_v2.helpers import Adapter, AdapterResolver, environment


class Clock:
    def now_epoch(self):
        return 150

    def now_iso(self):
        return "2026-08-27T00:00:00Z"


class PlanningStore:
    def get_plan(self, fingerprint):
        return PLAN if fingerprint == PLAN.plan_fingerprint else None

    def get_context(self, digest):
        return CONTEXT if digest == CONTEXT.provider_context_digest else None


class Planning:
    def describe(self, provider):
        assert provider == PROVIDER
        return DESC


class Identity:
    def for_plan(self, plan):
        return RUN


class Resolver:
    def __init__(self, binding):
        self.binding = binding

    def resolve(self, provider, context):
        assert provider == PROVIDER and context == CONTEXT
        return self.binding


class Materializer:
    def __init__(self, binding):
        self.binding = binding
        self.prepare_calls = 0

    def prepare(self, plan, run, binding):
        self.prepare_calls += 1
        assert binding == self.binding
        return CanonicalPreparationV2.build(
            provider=binding.provider,
            scope=binding.scope,
            project_ref=run.project_ref,
            run_id=run.run_id,
            plan_fingerprint=plan.plan_fingerprint,
            source_digest=plan.basis.source_digest,
            workload_digest=plan.basis.workload_digest,
            runtime_digest=plan.basis.runtime_digest,
            resource_digest=binding.resource_digest,
            artifact_contract_digest=plan.basis.artifact_policy_digest,
            quote_digest=binding.quote_digest,
            secret_requirements_digest=binding.secret_requirements_digest,
        )

    def payload(self, preparation, kind):
        return CanonicalProviderPayloadV1.build(
            preparation.provider.provider_id,
            f"{kind.value}-payload/v2",
            preparation.workload_digest,
        )


class DynamicExecutor:
    def __init__(self, descriptor, outcomes):
        self.descriptor = descriptor
        self.provider_id = PROVIDER.provider_id
        self.profile_ref = PROVIDER.profile_ref
        self.account_ref = SCOPE.account_ref
        self.namespace_ref = SCOPE.namespace_ref
        self.effect_kinds = ("stage", "submit", "cancel")
        self.payload_schemas = ("stage-payload/v2", "submit-payload/v2", "cancel-payload/v2")
        self.outcomes = outcomes
        self.effects = {}
        self.calls = []

    def execute_once(self, payload, request):
        self.calls.append(request.effect_kind)
        effect_id = self.effects[request.command_digest]
        disposition = self.outcomes[request.effect_kind]
        values = {}
        if disposition is ObservationDisposition.FOUND:
            if request.effect_kind == "stage":
                values["stage_ref"] = ProviderStageRefV1(
                    self.provider_id, self.profile_ref, self.account_ref,
                    self.namespace_ref, "stage-output",
                )
            elif request.effect_kind == "submit":
                values["provider_run"] = ScopedProviderRunRefV1(
                    self.provider_id, self.profile_ref, self.account_ref,
                    self.namespace_ref, "job-1",
                )
        return ProviderObservationV1(
            effect_id, request.command_digest, request.descriptor_digest,
            disposition, request.digest, 1, **values,
        )


class FoundationAuthenticator:
    def __init__(self, grants, receipts, invalid):
        self.grants = grants
        self.receipts = receipts
        self.invalid = invalid

    def authenticate_grant(self, grant, command_bytes):
        return self.grants.authenticate(grant, command_bytes)

    def authenticate_receipt(self, receipt):
        return self.receipts.verify(receipt)

    def authenticate_invalid_evidence(self, evidence):
        return self.invalid.verify(evidence)


class Authorization:
    def __init__(self, grants, executor, adapter_descriptor):
        self.grants = grants
        self.executor = executor
        self.adapter_descriptor = adapter_descriptor
        self.effect_issues = 0
        self.reconciliation_issues = 0
        self.reconciliation_slots = []

    def commit_preflight(self, plan, preflight):
        assert preflight.binds(plan)
        return D[6]

    def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
        command = parse_exact_command(command_bytes)
        self.executor.effects[command.digest] = command.operation.effect.effect_id
        self.effect_issues += 1
        return self.grants.issue(
            command_bytes,
            grant_ref=f"effect-grant-{command.operation.effect.kind.value}",
            policy_digest=preflight_digest,
            requirement_digest=D[10],
            not_before_epoch=100,
            expires_at_epoch=200,
        )

    def issue_reconciliation_grant(self, record, binding, *, slot, now_epoch):
        command = parse_exact_command(record.command_bytes)
        current = record.reconciliation
        if current is None:
            generation = epoch = 1
        elif current.completed:
            generation, epoch = current.generation + 1, current.ownership_epoch + 1
        else:
            generation, epoch = current.generation, current.ownership_epoch
        self.reconciliation_issues += 1
        self.reconciliation_slots.append(slot)
        content = ReconciliationGrantContentV1(
            f"reconciliation-{generation}-{epoch}-{self.reconciliation_issues}",
            command.digest,
            command.operation.effect.effect_id,
            command.preparation.preparation_digest,
            self.adapter_descriptor.digest,
            command.preparation.provider.provider_id,
            command.preparation.provider.profile_ref,
            command.preparation.scope.account_ref,
            command.preparation.scope.namespace_ref,
            "coordinator-owner",
            generation,
            epoch,
            D[9],
            D[10],
            100,
            200,
            self.grants.epoch,
            self.grants.revocation_generation,
        )
        assert (content.generation, content.ownership_epoch) == (
            slot.generation, slot.ownership_epoch
        )
        return self.grants.issue_reconciliation(content)


class TrustedEvidence:
    def __init__(self, repository, verifier):
        self.repository = repository
        self.verifier = verifier
        self.calls = 0

    def obtain(self, request, *, now_epoch):
        self.calls += 1
        return self.verifier.proof(self.repository.get(request.effect_id), "quiescent", now_epoch)


class Harness:
    def __init__(self, *, stage=ObservationDisposition.FOUND, submit=ObservationDisposition.FOUND):
        self.adapter_descriptor = AdapterDescriptorV1(PROVIDER.provider_id, "adapter-a", "1.0.0")
        descriptor = ExecutorDescriptorV1(PROVIDER.provider_id, "executor-a", "1.0.0")
        self.binding = ProviderExecutionBindingV1(
            PROVIDER, DESC.descriptor_digest, CONTEXT.profile_digest, SCOPE,
            descriptor, self.adapter_descriptor.digest, D[7], D[8], D[9],
        )
        self.executor = DynamicExecutor(descriptor, {"stage": stage, "submit": submit})
        repository, grants, receipts, invalid, verifier, executor_resolver = environment(self.executor)
        self.repository = repository
        self.grants = grants
        self.receipts = receipts
        self.invalid = invalid
        self.adapter = Adapter(
            ProviderObservationV1(
                "placeholder", D[0], self.adapter_descriptor.digest,
                ObservationDisposition.INDETERMINATE, D[1], 1,
            ),
            provider_id=PROVIDER.provider_id,
            profile_ref=PROVIDER.profile_ref,
            account_ref=SCOPE.account_ref,
            namespace_ref=SCOPE.namespace_ref,
        )
        self.adapter.descriptor = self.adapter_descriptor
        broker = EffectBrokerV2(repository, executor_resolver, grants, receipts, invalid)
        reconciliation = ReconciliationServiceV1(
            repository, grants, AdapterResolver(self.adapter), receipts, invalid
        )
        assessments = FoundationRecordAssessmentAuthorityV1(
            "assessment-authority", "assessment-key", b"a" * 32,
            assessor_ref="foundation-assessor", assessor_version="1.0.0",
            clock=Clock(), receipt_authority=receipts,
            invalid_evidence_authority=invalid, grant_authority=grants,
        )
        self.trusted_evidence = TrustedEvidence(repository, verifier)
        self.foundation = ComposedEffectFoundationV1(
            repository, broker, reconciliation,
            grant_authority=grants, receipt_authority=receipts,
            invalid_evidence_authority=invalid,
            assessment_authority=assessments,
            trusted_quiescence_evidence=self.trusted_evidence,
        )
        self.authenticator = FoundationAuthenticator(grants, receipts, invalid)
        self.workflows = InMemoryWorkflowStoreV1(
            self.authenticator, assessments, assessments, assessments
        )
        self.execution_grants = InMemoryExecutionGrantStoreV1(grants)
        self.reconciliation_grants = InMemoryReconciliationGrantStoreV1(grants)
        self.authorization = Authorization(grants, self.executor, self.adapter_descriptor)
        self.materializer = Materializer(self.binding)
        self.service = TrainingCoordinatorV1(
            Planning(), PlanningStore(), self.workflows,
            InMemoryPreparationStoreV1(),
            self.execution_grants,
            self.reconciliation_grants,
            Resolver(self.binding), self.materializer, self.authorization,
            self.foundation, self.authenticator, Clock(), Identity(),
        )


def preflight():
    return TrainingPreflight(
        PLAN.plan_fingerprint, True,
        "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
    )


def test_start_runs_durable_stage_then_submit_and_is_restart_idempotent():
    harness = Harness()
    result = harness.service.start(PLAN, preflight())
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.executor.calls == ["stage", "submit"]
    assert harness.authorization.effect_issues == 2
    assert harness.materializer.prepare_calls == 1
    assert harness.service.start(PLAN, preflight()) == result
    assert harness.executor.calls == ["stage", "submit"]
    assert harness.authorization.effect_issues == 2
    assert harness.materializer.prepare_calls == 1


def test_stage_indeterminate_stops_before_submit_and_reconcile_resumes_submit():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    assert waiting.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.executor.calls == ["stage"]

    stage_command = parse_exact_command(waiting.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        stage_command.operation.effect.effect_id,
        stage_command.digest,
        stage_command.executor.digest,
        ObservationDisposition.FOUND,
        D[1],
        1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    resumed = harness.service.reconcile(RUN)
    assert resumed.phase is WorkflowPhaseV1.QUEUED
    assert harness.executor.calls == ["stage", "submit"]
    assert harness.authorization.reconciliation_issues == 1


def test_start_rejects_expired_preflight_before_any_effect():
    harness = Harness()
    expired = TrainingPreflight(
        PLAN.plan_fingerprint, True,
        "2026-08-25T00:00:00Z", "2026-08-26T00:00:00Z",
    )
    with pytest.raises(CoordinatorErrorV1) as caught:
        harness.service.start(PLAN, expired)
    assert caught.value.code is CoordinatorCodeV1.PREFLIGHT_INVALID
    assert harness.executor.calls == []


def test_start_uses_existing_foundation_record_before_execute():
    harness = Harness()
    first = harness.service.start(PLAN, preflight())
    assert first.phase is WorkflowPhaseV1.QUEUED
    calls = tuple(harness.executor.calls)
    assert harness.service.start(PLAN, preflight()) == first
    assert tuple(harness.executor.calls) == calls


def test_concurrent_starts_converge_without_duplicate_provider_effects():
    harness = Harness()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(harness.service.start, PLAN, preflight()) for _ in range(2))
    results = []
    retries = 0
    for future in futures:
        try:
            results.append(future.result())
        except CoordinatorErrorV1 as error:
            assert error.code is CoordinatorCodeV1.FOUNDATION_INTERRUPTED
            retries += 1
    for _ in range(retries):
        results.append(harness.service.start(PLAN, preflight()))
    assert len(results) == 2
    assert all(result.phase is WorkflowPhaseV1.QUEUED for result in results)
    assert results[0] == results[1]
    assert sorted(harness.executor.calls) == ["stage", "submit"]


def test_cas_applied_then_raised_converges_from_durable_replacement():
    harness = Harness()
    retained = harness.workflows

    class LostAckWorkflowStore:
        def __init__(self):
            self.raised = False

        def __getattr__(self, name):
            return getattr(retained, name)

        def compare_and_swap(self, expected, replacement, *, transition):
            result = retained.compare_and_swap(expected, replacement, transition=transition)
            if result and not self.raised:
                self.raised = True
                raise RuntimeError("lost acknowledgement secret")
            return result

    wrapper = LostAckWorkflowStore()
    harness.service._workflows = wrapper
    result = harness.service.start(PLAN, preflight())
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert wrapper.raised


@pytest.mark.parametrize("decision", ["truthy", "throwing"])
def test_coordinator_accepts_only_exact_true_store_ancestry(decision):
    harness = Harness()
    source = harness.service.start(PLAN, preflight())
    retained = harness.workflows

    class AncestryWorkflowStore:
        def __getattr__(self, name):
            return getattr(retained, name)

        def is_descendant(self, ancestor, descendant):
            if decision == "throwing":
                raise RuntimeError("ancestry-secret")
            return "truthy"

    harness.service._workflows = AncestryWorkflowStore()
    with pytest.raises(CoordinatorErrorV1) as caught:
        harness.service._stored_descendant(source, source)
    assert caught.value.code is CoordinatorCodeV1.STORE_INTEGRITY
    assert "ancestry-secret" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert harness.executor.calls == ["stage", "submit"]


def test_cas_false_with_unchanged_current_retries_same_direct_step():
    harness = Harness()
    retained = harness.workflows

    class FalseOnceWorkflowStore:
        def __init__(self):
            self.calls = 0

        def __getattr__(self, name):
            return getattr(retained, name)

        def compare_and_swap(self, expected, replacement, *, transition):
            self.calls += 1
            if self.calls == 1:
                return False
            return retained.compare_and_swap(expected, replacement, transition=transition)

    wrapper = FalseOnceWorkflowStore()
    harness.service._workflows = wrapper
    result = harness.service.start(PLAN, preflight())
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert wrapper.calls >= 2
    assert harness.executor.calls == ["stage", "submit"]


def test_execution_grant_loser_reloads_durable_winner_after_put_conflict():
    harness = Harness()
    retained = harness.execution_grants

    class AppliedThenConflictGrantStore:
        def get(self, slot, command_bytes):
            return retained.get(slot, command_bytes)

        def put_if_absent(self, slot, grant, command_bytes):
            retained.put_if_absent(slot, grant, command_bytes)
            raise RuntimeError("losing candidate")

    harness.service._execution_grants = AppliedThenConflictGrantStore()
    result = harness.service.start(PLAN, preflight())
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.executor.calls == ["stage", "submit"]


def test_existing_foundation_record_is_applied_before_poisoned_grant_store_access():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    retained = harness.workflows

    class RefuseStageApplication:
        def __getattr__(self, name):
            return getattr(retained, name)

        def compare_and_swap(self, expected, replacement, *, transition):
            if type(transition) is ApplyStageEffectTransitionV1:
                return False
            return retained.compare_and_swap(expected, replacement, transition=transition)

    harness.service._workflows = RefuseStageApplication()
    with pytest.raises(CoordinatorErrorV1) as stopped:
        harness.service.start(PLAN, preflight())
    assert stopped.value.code is CoordinatorCodeV1.RETRY_EXHAUSTED
    assert harness.executor.calls == ["stage"]

    class PoisonedGrantStore:
        def get(self, slot, command_bytes):
            raise RuntimeError("must not read grant when Foundation record exists")

    harness.service._workflows = retained
    harness.service._execution_grants = PoisonedGrantStore()
    resumed = harness.service.start(PLAN, preflight())
    assert resumed.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.executor.calls == ["stage"]


def test_exact_false_foundation_authentication_is_interruption_not_progress():
    harness = Harness()

    class ExactFalseFoundationAuthenticator:
        def authenticate_grant(self, grant, command_bytes):
            return False

        def authenticate_receipt(self, receipt):
            return False

        def authenticate_invalid_evidence(self, evidence):
            return False

    harness.service._foundation_authenticator = ExactFalseFoundationAuthenticator()
    for _ in range(2):
        with pytest.raises(CoordinatorErrorV1) as caught:
            harness.service.start(PLAN, preflight())
        assert caught.value.code is CoordinatorCodeV1.FOUNDATION_INTERRUPTED
        retained = harness.workflows.get(RUN)
        assert retained.phase is WorkflowPhaseV1.STAGE_INTENT_RECORDED
        assert harness.executor.calls == ["stage"]


def test_submit_indeterminate_reconciles_to_queued_without_reexecuting_submit():
    harness = Harness(submit=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    assert waiting.phase is WorkflowPhaseV1.SUBMIT_RECONCILE_REQUIRED
    submit_command = parse_exact_command(waiting.submit.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        submit_command.operation.effect.effect_id,
        submit_command.digest,
        submit_command.executor.digest,
        ObservationDisposition.FOUND,
        D[1],
        1,
        provider_run=ScopedProviderRunRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "job-1",
        ),
    )
    before = tuple(harness.executor.calls)
    queued = harness.service.reconcile(RUN)
    assert queued.phase is WorkflowPhaseV1.QUEUED
    assert tuple(harness.executor.calls) == before
    assert harness.authorization.reconciliation_issues == 1


def test_existing_reconciliation_completion_is_applied_before_another_lookup():
    harness = Harness(submit=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    submit_command = parse_exact_command(waiting.submit.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        submit_command.operation.effect.effect_id,
        submit_command.digest,
        submit_command.executor.digest,
        ObservationDisposition.FOUND,
        D[1],
        1,
        provider_run=ScopedProviderRunRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "job-1",
        ),
    )
    raw = harness.foundation.get(waiting.submit.effect_id)
    grant = harness.service._reconciliation_grant(raw, waiting.submit, harness.binding)
    harness.foundation.reconcile(
        waiting.submit.canonical_command_bytes, grant, now_epoch=150, continuation=None
    )
    calls = harness.adapter.calls
    queued = harness.service.reconcile(RUN)
    assert queued.phase is WorkflowPhaseV1.QUEUED
    assert harness.adapter.calls == calls


def test_reconciliation_grant_loser_reloads_winner_and_completes():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    command = parse_exact_command(waiting.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[1], 1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    retained = harness.reconciliation_grants

    class AppliedThenConflictReconciliationStore:
        def get(self, slot, *, command_bytes, record):
            return retained.get(slot, command_bytes=command_bytes, record=record)

        def put_if_absent(self, slot, grant, command_bytes, record):
            retained.put_if_absent(slot, grant, command_bytes, record)
            raise RuntimeError("losing reconciliation candidate")

    harness.service._reconciliation_grants = AppliedThenConflictReconciliationStore()
    result = harness.service.reconcile(RUN)
    assert result.phase is WorkflowPhaseV1.QUEUED


def test_fresh_stage_predecessor_closes_throwing_receipt_authenticator():
    harness = Harness()
    queued = harness.service.start(PLAN, preflight())
    stage_record = harness.foundation.get(queued.stage.effect_id)
    assessment = harness.foundation.assess(stage_record)
    preparation = harness.service._preparation(PLAN, queued, harness.binding)

    class ThrowingAuthenticator:
        def authenticate_receipt(self, receipt):
            raise RuntimeError("secret receipt authority")

    harness.service._foundation_authenticator = ThrowingAuthenticator()
    with pytest.raises(CoordinatorErrorV1) as caught:
        harness.service._fresh_stage_predecessor(
            queued, stage_record, assessment, preparation
        )
    assert caught.value.code is CoordinatorCodeV1.GRANT_INVALID
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_orphaned_dispatch_uses_trusted_recovery_then_reconciles():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    durable_foundation = harness.foundation

    class OrphaningFoundation:
        def __getattr__(self, name):
            return getattr(durable_foundation, name)

        def execute(self, command_bytes, grant, *, now_epoch):
            command = parse_exact_command(command_bytes)
            harness.repository.consume_attempt(command_bytes, grant, now_epoch=now_epoch)
            harness.repository.begin_dispatch(command.operation.effect.effect_id)
            harness.repository.orphan(command.operation.effect.effect_id)
            raise RuntimeError("provider outcome unknown")

    harness.service._foundation = OrphaningFoundation()
    with pytest.raises(CoordinatorErrorV1):
        harness.service.start(PLAN, preflight())
    retained = harness.workflows.get(RUN)
    assert retained.phase is WorkflowPhaseV1.STAGE_INTENT_RECORDED
    assert harness.repository.get(retained.stage.effect_id).dispatch is DispatchState.ORPHANED_UNPROVEN

    command = parse_exact_command(retained.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[1], 1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    harness.service._foundation = durable_foundation
    result = harness.service.reconcile(RUN)
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.trusted_evidence.calls == 1
    assert harness.executor.calls == ["submit"]


def test_active_claim_restart_reuses_durable_grant_and_continuation():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    command = parse_exact_command(waiting.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[1], 1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    durable_foundation = harness.foundation

    class AcquireThenLoseAck:
        def __getattr__(self, name):
            return getattr(durable_foundation, name)

        def reconcile(self, command_bytes, grant, *, now_epoch, continuation=None):
            harness.repository.acquire_reconciliation(
                command_bytes, grant, now_epoch=now_epoch, continuation=continuation
            )
            raise RuntimeError("lookup acknowledgement lost")

    harness.service._foundation = AcquireThenLoseAck()
    with pytest.raises(CoordinatorErrorV1):
        harness.service.reconcile(RUN)
    active = harness.repository.get(waiting.stage.effect_id)
    assert active.reconciliation.active
    assert harness.authorization.reconciliation_issues == 1

    harness.service._foundation = durable_foundation
    result = harness.service.reconcile(RUN)
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.authorization.reconciliation_issues == 1


def test_interrupted_claim_resume_issues_new_slot_grant_and_completes():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    command = parse_exact_command(waiting.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[1], 1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    harness.adapter.fail = True
    returned = harness.service.reconcile(RUN)
    assert returned.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    interrupted = harness.repository.get(waiting.stage.effect_id)
    assert not interrupted.reconciliation.active and not interrupted.reconciliation.completed
    assert harness.authorization.reconciliation_issues == 1

    harness.adapter.fail = False
    result = harness.service.reconcile(RUN)
    assert result.phase is WorkflowPhaseV1.QUEUED
    assert harness.authorization.reconciliation_issues == 2


def test_repeated_indeterminate_uses_one_lookup_per_call_then_completed_retries():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    command = parse_exact_command(waiting.stage.canonical_command_bytes)

    def observation(disposition):
        values = {}
        if disposition is ObservationDisposition.FOUND:
            values["stage_ref"] = ProviderStageRefV1(
                PROVIDER.provider_id, PROVIDER.profile_ref,
                SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
            )
        harness.adapter.observation = ProviderObservationV1(
            command.operation.effect.effect_id, command.digest,
            command.executor.digest, disposition, D[1], 1, **values,
        )

    durable = harness.foundation

    class RecordingFoundation:
        def __init__(self):
            self.continuations = []

        def __getattr__(self, name):
            return getattr(durable, name)

        def reconcile(self, command_bytes, grant, *, now_epoch, continuation=None):
            self.continuations.append(continuation)
            return durable.reconcile(
                command_bytes, grant, now_epoch=now_epoch, continuation=continuation
            )

    recording = RecordingFoundation()
    harness.service._foundation = recording
    observation(ObservationDisposition.INDETERMINATE)

    first = harness.service.reconcile(RUN)
    assert first.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.adapter.calls == 1
    assert harness.authorization.reconciliation_issues == 1
    first_claim = harness.repository.get(waiting.stage.effect_id).reconciliation
    assert first_claim.completed and (first_claim.generation, first_claim.ownership_epoch) == (1, 1)

    second = harness.service.reconcile(RUN)
    assert second.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.adapter.calls == 2
    assert harness.authorization.reconciliation_issues == 2
    second_claim = harness.repository.get(waiting.stage.effect_id).reconciliation
    assert second_claim.completed and (second_claim.generation, second_claim.ownership_epoch) == (2, 2)

    observation(ObservationDisposition.FOUND)
    terminal = harness.service.reconcile(RUN)
    assert terminal.phase is WorkflowPhaseV1.QUEUED
    assert harness.adapter.calls == 3
    assert harness.authorization.reconciliation_issues == 3
    third_claim = harness.repository.get(waiting.stage.effect_id).reconciliation
    assert third_claim.completed and (third_claim.generation, third_claim.ownership_epoch) == (3, 3)
    assert [(slot.generation, slot.ownership_epoch) for slot in harness.authorization.reconciliation_slots] == [(1, 1), (2, 2), (3, 3)]
    assert recording.continuations == [None, None, None]


def test_always_interrupted_performs_one_lookup_and_one_slot_per_invocation():
    harness = Harness(stage=ObservationDisposition.INDETERMINATE)
    waiting = harness.service.start(PLAN, preflight())
    command = parse_exact_command(waiting.stage.canonical_command_bytes)
    harness.adapter.observation = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[1], 1,
        stage_ref=ProviderStageRefV1(
            PROVIDER.provider_id, PROVIDER.profile_ref,
            SCOPE.account_ref, SCOPE.namespace_ref, "stage-output",
        ),
    )
    harness.adapter.fail = True
    one = harness.service.reconcile(RUN)
    assert one.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.adapter.calls == 1
    assert harness.authorization.reconciliation_issues == 1
    two = harness.service.reconcile(RUN)
    assert two.phase is WorkflowPhaseV1.STAGE_RECONCILE_REQUIRED
    assert harness.adapter.calls == 2
    assert harness.authorization.reconciliation_issues == 2
    assert len(harness.authorization.reconciliation_slots) == 2


def test_closed_taxonomy_and_representative_boundary_mappings():
    assert {code.value for code in CoordinatorCodeV1} == {
        "invalid_input", "plan_missing", "context_missing", "preflight_invalid",
        "binding_mismatch", "workflow_conflict", "store_integrity",
        "grant_missing", "grant_invalid", "quiescence_unproven",
        "foundation_interrupted", "retry_exhausted",
    }
    harness = Harness()
    with pytest.raises(CoordinatorErrorV1) as invalid:
        harness.service.start(object(), preflight())
    assert invalid.value.code is CoordinatorCodeV1.INVALID_INPUT

    class MissingPlan:
        def get_plan(self, fingerprint):
            return None

    harness.service._plans = MissingPlan()
    with pytest.raises(CoordinatorErrorV1) as missing_plan:
        harness.service.start(PLAN, preflight())
    assert missing_plan.value.code is CoordinatorCodeV1.PLAN_MISSING

    harness = Harness()

    class MissingContext(PlanningStore):
        def get_context(self, digest):
            return None

    harness.service._plans = MissingContext()
    with pytest.raises(CoordinatorErrorV1) as missing_context:
        harness.service.start(PLAN, preflight())
    assert missing_context.value.code is CoordinatorCodeV1.CONTEXT_MISSING

    harness = Harness()

    def invalid_grant(*args, **kwargs):
        raise RuntimeError("secret authority failure")

    harness.authorization.issue_effect_grant = invalid_grant
    with pytest.raises(CoordinatorErrorV1) as grant_invalid:
        harness.service.start(PLAN, preflight())
    assert grant_invalid.value.code is CoordinatorCodeV1.GRANT_INVALID
    assert grant_invalid.value.__cause__ is None and grant_invalid.value.__context__ is None
