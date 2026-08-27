from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from synaptic_tuner.api.v1.providers import (
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRef,
)
from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1,
    ProviderPlanRef,
    TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel
from tuner.execution.fake_provider_v1 import (
    FakeArtifactV1,
    FakeEffectResultV1,
    FakeEffectScriptV1,
    FakeProviderConfigV1,
    FakeProviderFamilyV1,
)
from tuner.execution.foundation_v2.executors import (
    AdapterDescriptorV1,
    ExecutionResolutionRequestV2,
    ExecutorDescriptorV1,
    ReconciliationResolutionRequestV2,
)
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import (
    ObservationDisposition,
    ProviderObservationV1,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.authority import (
    GrantAuthorityV2,
    ReconciliationGrantContentV1,
)
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
    ReceiptContentV2,
)
from tuner.execution.foundation_v2.repository import EffectState, InMemoryEffectRepositoryV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1, ScopedProviderRunRefV1
from tuner.execution.foundation_v2.registry import (
    LazyProviderRegistryV2,
    ProviderReaderFactoryRequestV1,
    ProviderRegistrationV2,
)
from tuner.execution.coordinator_v1.model import (
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    ProviderRunObservationContentV1,
    ProviderRunPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import (
    apply_provider_observation,
    provider_run_read_request,
)
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.cursors import HMACCursorAuthorityV1
from tuner.execution.coordinator_v1.foundation import (
    ComposedEffectFoundationV1,
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.coordinator_v1.operations import TrainingOperationsV1
from tuner.execution.coordinator_v1.stores import (
    InMemoryExecutionGrantStoreV1,
    InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1,
    InMemoryWorkflowStoreV1,
)
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest,
    RunLogsRequest,
    RunOperationCode,
    RunOperationError,
)
from tests.execution.coordinator_v1.test_state_machine import (
    AssessmentAuth,
    Auth,
    DESC,
    PROVIDER,
    SCOPE,
    assessment,
    queued_evidence,
)
from tests.execution.coordinator_v1.test_start_reconcile_service import (
    Clock,
    FoundationAuthenticator,
    TrustedEvidence,
)
from tests.execution.foundation_v2.helpers import StrongVerifier


PROFILES = ("docker", "modal", "hf_jobs", "runpod")
D = tuple(character * 64 for character in "123456789abcdef")


class UnusedFoundationAuth:
    def authenticate_grant(self, grant, command_bytes):
        raise AssertionError("reader was not invoked")

    def authenticate_receipt(self, receipt):
        raise AssertionError("reader was not invoked")

    def authenticate_invalid_evidence(self, evidence):
        raise AssertionError("reader was not invoked")


class UnusedAssessmentAuth:
    def authenticate(self, assessment):
        raise AssertionError("reader was not invoked")


def make_family(configuration):
    return FakeProviderFamilyV1(
        configuration,
        evidence_key=b"e" * 32,
        foundation_authenticator=UnusedFoundationAuth(),
        assessment_authenticator=UnusedAssessmentAuth(),
    )


def config(provider_id: str) -> FakeProviderConfigV1:
    provider = ProviderRef(provider_id, "profile")
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1",
        provider_id,
        f"Configured {provider_id}",
        "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    executor = ExecutorDescriptorV1(provider_id, "executor", "1.0.0")
    adapter = AdapterDescriptorV1(provider_id, "adapter", "1.0.0")
    effects = (
        FakeEffectScriptV1(
            "stage-effect",
            D[0],
            executor.digest,
            EffectKind.STAGE,
            (FakeEffectResultV1(ObservationDisposition.FOUND),),
        ),
        FakeEffectScriptV1(
            "submit-effect",
            D[1],
            executor.digest,
            EffectKind.SUBMIT,
            (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),),
            (FakeEffectResultV1(ObservationDisposition.FOUND),),
        ),
        FakeEffectScriptV1(
            "cancel-effect",
            D[2],
            executor.digest,
            EffectKind.CANCEL,
            (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),),
            (FakeEffectResultV1(ObservationDisposition.FOUND),),
            cancel_reason_digest=D[3],
        ),
    )
    content = b"adapter-data"
    artifact = FakeArtifactV1(
        VerifiedArtifact("adapter", hashlib.sha256(content).hexdigest(), len(content)),
        content,
    )
    logs = (
        RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "queued", "q", 1),
        RunLogEntry(3, "2026-08-27T12:00:01Z", RunLogLevel.INFO, "running", "r", 1),
    )
    return FakeProviderConfigV1(
        provider,
        descriptor,
        D[4],
        "account",
        "namespace",
        executor,
        adapter,
        effects,
        (
            ProviderRunPhaseV1.QUEUED,
            ProviderRunPhaseV1.RUNNING,
            ProviderRunPhaseV1.SUCCEEDED,
        ),
        logs,
        (artifact,),
    )


class ProfileStack:
    def __init__(self, provider_id: str, *, policies=None, capabilities=None):
        self.policies = {"stage": "found", "submit": "found", "cancel": "found"}
        if policies:
            self.policies.update(policies)
        self.provider = ProviderRef(provider_id, "profile")
        self.scope = ExecutionScopeV1("account", "namespace")
        self.run = TrainingRunRef("run", "project")
        self.descriptor = ProviderDescriptor(
            "synaptic-provider-descriptor/v1",
            provider_id,
            f"Configured {provider_id}",
            "1.0.0",
            capabilities
            or ProviderCapabilities(True, True, True, True, True, False),
        )
        self.basis = TrainingPlanBasisV1(
            "synaptic-training-plan-basis/v1",
            "request",
            self.run.project_ref,
            *D[:5],
        )
        self.context = ProviderPlanContextV1(
            "synaptic-provider-plan-context/v1",
            self.provider,
            self.basis.basis_digest,
            self.descriptor.descriptor_digest,
            D[5],
        )
        self.plan = TrainingPlan(
            "synaptic-training-plan/v2",
            self.basis,
            ProviderPlanRef(self.context.provider_context_digest),
        )
        executor = ExecutorDescriptorV1(provider_id, "executor", "1.0.0")
        adapter = AdapterDescriptorV1(provider_id, "adapter", "1.0.0")
        self.binding = ProviderExecutionBindingV1(
            self.provider,
            self.descriptor.descriptor_digest,
            self.context.profile_digest,
            self.scope,
            executor,
            adapter.digest,
            D[7],
            D[8],
            D[9],
        )
        artifact_bytes = b"adapter-data"
        artifact = FakeArtifactV1(
            VerifiedArtifact(
                "adapter",
                hashlib.sha256(artifact_bytes).hexdigest(),
                len(artifact_bytes),
            ),
            artifact_bytes,
        )
        fake_config = FakeProviderConfigV1(
            self.provider,
            self.descriptor,
            self.context.profile_digest,
            self.scope.account_ref,
            self.scope.namespace_ref,
            executor,
            adapter,
            (),
            (ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED),
            (
                RunLogEntry(
                    1,
                    "2026-08-27T12:00:00Z",
                    RunLogLevel.INFO,
                    "progress",
                    "ok",
                    2,
                ),
            ),
            (artifact,),
        )
        self.grants = GrantAuthorityV2("grants", b"g" * 32)
        self.receipts = ReceiptAuthorityV2("receipts", b"r" * 32)
        self.invalid = InvalidEvidenceAuthorityV2("invalid", b"i" * 32)
        verifier = StrongVerifier()
        self.repository = InMemoryEffectRepositoryV2(
            self.receipts, self.invalid, verifier, verifier, self.grants
        )
        self.foundation_auth = FoundationAuthenticator(
            self.grants, self.receipts, self.invalid
        )
        assessments = FoundationRecordAssessmentAuthorityV1(
            "assessment-authority",
            "assessment-key",
            b"a" * 32,
            assessor_ref="foundation-assessor",
            assessor_version="1.0.0",
            clock=Clock(),
            receipt_authority=self.receipts,
            invalid_evidence_authority=self.invalid,
            grant_authority=self.grants,
        )
        self.assessments = assessments
        self.family = FakeProviderFamilyV1(
            fake_config,
            evidence_key=b"e" * 32,
            foundation_authenticator=self.foundation_auth,
            assessment_authenticator=assessments,
        )
        broker = EffectBrokerV2(
            self.repository,
            self.family.executor_resolver,
            self.grants,
            self.receipts,
            self.invalid,
        )
        reconciliation = ReconciliationServiceV1(
            self.repository,
            self.grants,
            self.family.reconciliation_resolver,
            self.receipts,
            self.invalid,
        )
        trusted = TrustedEvidence(self.repository, verifier)
        self.foundation = ComposedEffectFoundationV1(
            self.repository,
            broker,
            reconciliation,
            grant_authority=self.grants,
            receipt_authority=self.receipts,
            invalid_evidence_authority=self.invalid,
            assessment_authority=assessments,
            trusted_quiescence_evidence=trusted,
        )

        stack = self

        class Planning:
            def describe(self, provider):
                assert provider == stack.provider
                return stack.descriptor

        class PlanStore:
            def get_plan(self, fingerprint):
                return stack.plan if fingerprint == stack.plan.plan_fingerprint else None

            def get_context(self, digest):
                return stack.context if digest == stack.context.provider_context_digest else None

        class Resolver:
            def resolve(self, provider, context):
                assert (provider, context) == (stack.provider, stack.context)
                return stack.binding

        class Materializer:
            def prepare(self, plan, run, binding):
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
                from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1

                return CanonicalProviderPayloadV1.build(
                    stack.provider.provider_id,
                    f"{kind.value}-payload/v2",
                    preparation.workload_digest,
                )

        class Identity:
            def for_plan(self, plan):
                return stack.run

        class Authorization:
            def commit_preflight(self, plan, preflight):
                assert preflight.binds(plan)
                return D[6]

            def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
                command = parse_exact_command(command_bytes)
                policy = stack.policies[command.operation.effect.kind.value]
                if policy == "found":
                    dispatch = (FakeEffectResultV1(ObservationDisposition.FOUND),)
                    reconciliation_results = ()
                elif policy == "indeterminate_found":
                    dispatch = (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),)
                    reconciliation_results = (
                        FakeEffectResultV1(ObservationDisposition.FOUND),
                    )
                elif policy == "indeterminate_interrupted":
                    dispatch = (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),)
                    reconciliation_results = ()
                elif policy == "indeterminate_absent":
                    proof = verifier.proof(
                        SimpleNamespace(command=command), "final_absent", now_epoch
                    )
                    dispatch = (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),)
                    reconciliation_results = (
                        FakeEffectResultV1(
                            ObservationDisposition.DEFINITELY_ABSENT, proof
                        ),
                    )
                elif policy == "absent":
                    proof = verifier.proof(
                        SimpleNamespace(command=command), "final_absent", now_epoch
                    )
                    dispatch = (
                        FakeEffectResultV1(
                            ObservationDisposition.DEFINITELY_ABSENT, proof
                        ),
                    )
                    reconciliation_results = ()
                else:
                    raise AssertionError("unsupported test policy")
                stack.family.register_command(
                    command,
                    dispatch=dispatch,
                    reconciliation=reconciliation_results,
                )
                return stack.grants.issue(
                    command_bytes,
                    grant_ref=f"grant-{command.operation.effect.kind.value}",
                    policy_digest=preflight_digest,
                    requirement_digest=D[10],
                    not_before_epoch=100,
                    expires_at_epoch=200,
                )

            def issue_reconciliation_grant(self, record, binding, *, slot, now_epoch):
                command = parse_exact_command(record.command_bytes)
                content = ReconciliationGrantContentV1(
                    f"reconcile-{slot.generation}-{slot.ownership_epoch}",
                    command.digest,
                    command.operation.effect.effect_id,
                    command.preparation.preparation_digest,
                    adapter.digest,
                    stack.provider.provider_id,
                    stack.provider.profile_ref,
                    stack.scope.account_ref,
                    stack.scope.namespace_ref,
                    "owner",
                    slot.generation,
                    slot.ownership_epoch,
                    D[9],
                    D[10],
                    100,
                    200,
                    stack.grants.epoch,
                    stack.grants.revocation_generation,
                )
                return stack.grants.issue_reconciliation(content)

        planning, plans = Planning(), PlanStore()
        self.workflows = InMemoryWorkflowStoreV1(
            self.foundation_auth, assessments, self.family.evidence_authority, self.family.artifact_verifier
        )
        self.coordinator = TrainingCoordinatorV1(
            planning,
            plans,
            self.workflows,
            InMemoryPreparationStoreV1(),
            InMemoryExecutionGrantStoreV1(self.grants),
            InMemoryReconciliationGrantStoreV1(self.grants),
            Resolver(),
            Materializer(),
            Authorization(),
            self.foundation,
            self.foundation_auth,
            Clock(),
            Identity(),
        )
        registry = LazyProviderRegistryV2()
        registry.register(
            ProviderRegistrationV2(
                self.descriptor,
                executor,
                adapter,
                self.family.executor_resolver,
                self.family.reconciliation_resolver,
                self.family.reader_factory,
            )
        )
        assert registry.list() == (self.descriptor,)
        assert self.family.trace.snapshot() == ()
        reader = registry.resolve_reader(
            ProviderReaderFactoryRequestV1(
                self.provider,
                self.descriptor.descriptor_digest,
                self.context.profile_digest,
                self.scope.account_ref,
                self.scope.namespace_ref,
            )
        ).reader
        self.operations = TrainingOperationsV1(
            planning,
            plans,
            self.workflows,
            self.coordinator,
            self.foundation,
            self.foundation_auth,
            assessments,
            reader,
            self.family.evidence_authority,
            self.family.evidence_authority,
            self.family.artifact_verifier,
            HMACCursorAuthorityV1(
                "cursor-authority", {1: b"c" * 32}, active_generation=1
            ),
            Clock(),
        )

    def preflight(self):
        return TrainingPreflight(
            self.plan.plan_fingerprint,
            True,
            "2026-08-26T00:00:00Z",
            "2026-08-28T00:00:00Z",
        )


def execution_request(configuration, effect):
    return ExecutionResolutionRequestV2(
        effect.command_digest,
        configuration.executor_descriptor.digest,
        configuration.provider.provider_id,
        configuration.provider.profile_ref,
        configuration.account_ref,
        configuration.namespace_ref,
        effect.kind.value,
        f"{effect.kind.value}-payload/v2",
        D[5],
    )


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_runs_real_start_outcome_logs_verify_and_artifacts(provider_id) -> None:
    stack = ProfileStack(provider_id)
    queued = stack.coordinator.start(stack.plan, stack.preflight())
    assert queued.phase.value == "queued"
    with pytest.raises(RunOperationError) as hidden:
        stack.operations.artifacts(RunArtifactRequest(stack.run, "adapter", 100))
    assert hidden.value.code is RunOperationCode.ARTIFACTS_UNVERIFIED

    running = stack.operations.outcome(stack.run)
    assert running.state.value == "running"
    logs = stack.operations.logs(
        RunLogsRequest(stack.run, limit=1, maximum_bytes=4096)
    )
    assert logs.entries[0].message == "ok"
    succeeded = stack.operations.outcome(stack.run)
    assert succeeded.state.value == "succeeded"
    verified = stack.operations.verify(stack.run)
    assert verified.verified is True
    reader_calls = stack.family.trace.snapshot().count(("reader", "artifacts"))
    assert stack.operations.reverify(stack.run).verified is True
    assert stack.family.trace.snapshot().count(("reader", "artifacts")) == reader_calls
    stream = stack.operations.artifacts(
        RunArtifactRequest(stack.run, "adapter", 100)
    )
    assert b"".join(stream.iter_bytes()) == b"adapter-data"


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_runs_real_cancel_found_and_is_idempotent(provider_id) -> None:
    stack = ProfileStack(provider_id)
    stack.coordinator.start(stack.plan, stack.preflight())
    requested = stack.operations.cancel(stack.run, "requested")
    assert requested.state.value == "cancel_requested"
    trace = stack.family.trace.snapshot()
    assert stack.operations.cancel(stack.run, "requested") == requested
    assert stack.family.trace.snapshot() == trace


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_restores_exact_phase_after_cancel_final_absence(provider_id) -> None:
    stack = ProfileStack(provider_id, policies={"cancel": "absent"})
    queued = stack.coordinator.start(stack.plan, stack.preflight())
    restored = stack.operations.cancel(stack.run, "requested")
    assert restored.state.value == "queued"
    retained = stack.workflows.get(stack.run)
    assert retained.phase.value == "queued"
    assert retained.cancel is not None and retained.cancel.foundation_outcomes


@pytest.mark.parametrize("effect_kind", ("submit", "cancel"))
@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_runs_real_indeterminate_then_found_reconciliation(
    provider_id, effect_kind
) -> None:
    stack = ProfileStack(provider_id, policies={effect_kind: "indeterminate_found"})
    waiting = stack.coordinator.start(stack.plan, stack.preflight())
    if effect_kind == "cancel":
        assert waiting.phase.value == "queued"
        waiting = stack.coordinator.cancel(stack.run, "requested")
    assert waiting.phase.value == f"{effect_kind}_reconcile_required"
    resumed = stack.operations.reconcile(stack.run)
    assert resumed.state.value == (
        "queued" if effect_kind == "submit" else "cancel_requested"
    )


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_concurrent_real_starts_converge_without_duplicate_effects(
    provider_id,
) -> None:
    stack = ProfileStack(provider_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(stack.coordinator.start, stack.plan, stack.preflight())
            for _ in range(2)
        )
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception:
            results.append(stack.coordinator.start(stack.plan, stack.preflight()))
    assert all(result.phase.value == "queued" for result in results)
    trace = stack.family.trace.snapshot()
    assert trace.count(("executor", "stage")) == 1
    assert trace.count(("executor", "submit")) == 1


@pytest.mark.parametrize("provider_id", PROFILES)
@pytest.mark.parametrize("disabled", ("observe", "logs", "cancel", "reconcile", "artifact_streaming"))
def test_each_profile_pinned_capability_false_stops_before_downstream_effect(
    provider_id, disabled
) -> None:
    values = dict(
        observe=True,
        logs=True,
        cancel=True,
        reconcile=True,
        artifact_streaming=True,
        cost_quote=False,
    )
    values[disabled] = False
    policies = {"submit": "indeterminate_found"} if disabled == "reconcile" else None
    stack = ProfileStack(
        provider_id,
        policies=policies,
        capabilities=ProviderCapabilities(**values),
    )
    stack.coordinator.start(stack.plan, stack.preflight())
    before = stack.family.trace.snapshot()
    with pytest.raises(RunOperationError) as caught:
        if disabled == "observe":
            stack.operations.outcome(stack.run)
        elif disabled == "logs":
            stack.operations.logs(
                RunLogsRequest(stack.run, limit=1, maximum_bytes=4096)
            )
        elif disabled == "cancel":
            stack.operations.cancel(stack.run, "requested")
        elif disabled == "reconcile":
            stack.operations.reconcile(stack.run)
        else:
            stack.operations.outcome(stack.run)
            stack.operations.outcome(stack.run)
            before = stack.family.trace.snapshot()
            stack.operations.verify(stack.run)
    assert caught.value.code is RunOperationCode.CAPABILITY_UNAVAILABLE
    assert stack.family.trace.snapshot() == before


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_not_ready_preflight_has_zero_provider_or_factory_effects(
    provider_id,
) -> None:
    stack = ProfileStack(provider_id)
    before = stack.family.trace.snapshot()
    not_ready = TrainingPreflight(
        stack.plan.plan_fingerprint,
        False,
        "2026-08-26T00:00:00Z",
        "2026-08-28T00:00:00Z",
        (),
        ("cost_quote_unavailable",),
    )
    with pytest.raises(Exception):
        stack.coordinator.start(stack.plan, not_ready)
    assert stack.family.trace.snapshot() == before


@pytest.mark.parametrize("provider_id", PROFILES)
@pytest.mark.parametrize(
    "field",
    ("provider_id", "profile_ref", "account_ref", "namespace_ref", "provider_job_ref"),
)
def test_each_profile_rejects_authenticated_cross_scope_observation_without_cas(
    provider_id, field
) -> None:
    stack = ProfileStack(provider_id)
    queued = stack.coordinator.start(stack.plan, stack.preflight())
    reader = stack.family.reader

    class SubstitutingReader:
        def observe(self, request):
            valid = reader.observe(request)
            content = replace(valid.content, **{field: f"foreign-{field}"})
            return stack.family.evidence_authority.observation(content)

        def __getattr__(self, name):
            return getattr(reader, name)

    stack.operations._reader = SubstitutingReader()
    with pytest.raises(RunOperationError) as caught:
        stack.operations.outcome(stack.run)
    assert caught.value.code is RunOperationCode.PROVIDER_READ_INVALID
    assert stack.workflows.get(stack.run) == queued


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_rejects_forged_log_and_verification_tags(provider_id) -> None:
    stack = ProfileStack(provider_id)
    stack.coordinator.start(stack.plan, stack.preflight())
    reader = stack.family.reader

    class ForgedLogReader:
        def logs(self, request, query):
            return replace(reader.logs(request, query), tag="0" * 64)

        def __getattr__(self, name):
            return getattr(reader, name)

    stack.operations._reader = ForgedLogReader()
    with pytest.raises(RunOperationError) as log_error:
        stack.operations.logs(
            RunLogsRequest(stack.run, limit=1, maximum_bytes=4096)
        )
    assert log_error.value.code is RunOperationCode.PROVIDER_READ_INVALID

    stack.operations._reader = reader
    stack.operations.outcome(stack.run)
    stack.operations.outcome(stack.run)
    verifier = stack.family.artifact_verifier

    class ForgedVerifier:
        def verify(self, workflow, manifest):
            return replace(verifier.verify(workflow, manifest), tag="0" * 64)

        def replay(self, workflow, manifest, prior_receipt):
            return verifier.replay(workflow, manifest, prior_receipt)

        def authenticate(self, receipt):
            return verifier.authenticate(receipt)

    stack.operations._artifact_verifier = ForgedVerifier()
    with pytest.raises(RunOperationError) as verification_error:
        stack.operations.verify(stack.run)
    assert verification_error.value.code is RunOperationCode.PROVIDER_READ_INVALID
    assert stack.workflows.get(stack.run).phase.value == "succeeded_unverified"


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_reconciliation_interruption_stops_one_lookup_and_replays_safely(
    provider_id,
) -> None:
    stack = ProfileStack(
        provider_id, policies={"submit": "indeterminate_interrupted"}
    )
    waiting = stack.coordinator.start(stack.plan, stack.preflight())
    assert waiting.phase.value == "submit_reconcile_required"
    before = stack.family.trace.snapshot().count(("adapter", "lookup"))
    interrupted = stack.operations.reconcile(stack.run)
    assert interrupted.state.value == "reconcile_required"
    after = stack.family.trace.snapshot().count(("adapter", "lookup"))
    assert after == before + 1
    assert stack.workflows.get(stack.run).phase.value == "submit_reconcile_required"


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_authenticated_reconciliation_absence_is_terminal(provider_id) -> None:
    stack = ProfileStack(provider_id, policies={"submit": "indeterminate_absent"})
    waiting = stack.coordinator.start(stack.plan, stack.preflight())
    assert waiting.phase.value == "submit_reconcile_required"
    failed = stack.operations.reconcile(stack.run)
    assert failed.state.value == "failed"


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_concurrent_outcomes_converge_without_history_fork(provider_id) -> None:
    stack = ProfileStack(provider_id)
    stack.coordinator.start(stack.plan, stack.preflight())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(stack.operations.outcome, stack.run) for _ in range(2))
    results = []
    for future in futures:
        try:
            results.append(future.result())
        except RunOperationError:
            results.append(stack.operations.outcome(stack.run))
    retained = stack.workflows.get(stack.run)
    if retained.phase.value == "running":
        stack.operations.outcome(stack.run)
        retained = stack.workflows.get(stack.run)
    assert retained.phase.value == "succeeded_unverified"
    assert len(retained.provider_run_observations) == 2


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_concurrent_cancel_converges_to_one_effect(provider_id) -> None:
    stack = ProfileStack(provider_id)
    stack.coordinator.start(stack.plan, stack.preflight())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(stack.operations.cancel, stack.run, "requested")
            for _ in range(2)
        )
    for future in futures:
        try:
            future.result()
        except RunOperationError:
            stack.operations.cancel(stack.run, "requested")
    assert stack.workflows.get(stack.run).phase.value == "cancel_requested"
    assert stack.family.trace.snapshot().count(("executor", "cancel")) == 1


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_concurrent_reconcile_converges_to_one_lookup(provider_id) -> None:
    stack = ProfileStack(provider_id, policies={"submit": "indeterminate_found"})
    stack.coordinator.start(stack.plan, stack.preflight())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(stack.operations.reconcile, stack.run) for _ in range(2)
        )
    for future in futures:
        try:
            future.result()
        except RunOperationError:
            stack.operations.reconcile(stack.run)
    assert stack.workflows.get(stack.run).phase.value == "queued"
    assert stack.family.trace.snapshot().count(("adapter", "lookup")) == 1


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_concurrent_verify_converges_to_one_receipt_history(provider_id) -> None:
    stack = ProfileStack(provider_id)
    stack.coordinator.start(stack.plan, stack.preflight())
    stack.operations.outcome(stack.run)
    stack.operations.outcome(stack.run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(stack.operations.verify, stack.run) for _ in range(2))
    for future in futures:
        try:
            future.result()
        except RunOperationError:
            pass
    retained = stack.workflows.get(stack.run)
    assert retained.phase.value == "verified"
    assert len(retained.verification_receipts) == 1


@pytest.mark.parametrize("provider_id", PROFILES)
def test_each_profile_reconciliation_conflicting_terminal_evidence_contradicts(
    provider_id,
) -> None:
    stack = ProfileStack(provider_id, policies={"submit": "indeterminate_found"})
    stack.coordinator.start(stack.plan, stack.preflight())
    stack.operations.reconcile(stack.run)
    workflow = stack.workflows.get(stack.run)
    command = parse_exact_command(workflow.submit.canonical_command_bytes)
    record = stack.repository.get(command.operation.effect.effect_id)
    claim = record.reconciliation
    conflicting = ProviderObservationV1(
        command.operation.effect.effect_id,
        command.digest,
        command.executor.digest,
        ObservationDisposition.FOUND,
        D[12],
        claim.ownership_epoch,
        provider_run=ScopedProviderRunRefV1(
            stack.provider.provider_id,
            stack.provider.profile_ref,
            stack.scope.account_ref,
            stack.scope.namespace_ref,
            "conflicting-job",
        ),
    )
    receipt = stack.receipts.issue(
        ReceiptContentV2.from_observation(
            conflicting,
            source_kind="reconciliation",
            source_owner_ref=claim.owner_ref,
            source_generation=claim.generation,
            source_ownership_epoch=claim.ownership_epoch,
            source_claim_digest=claim.claim_digest,
            source_grant_ref=claim.grant_ref,
            source_grant_digest=claim.grant_digest,
        )
    )
    contradicted = stack.repository.complete_reconciliation(
        command.operation.effect.effect_id,
        claim,
        receipt,
        None,
        now_epoch=150,
    )
    assert contradicted.state is EffectState.CONTRADICTED


def test_real_stack_semantic_traces_match_after_identity_is_configuration_only() -> None:
    traces = []
    for provider_id in PROFILES:
        stack = ProfileStack(provider_id)
        stack.coordinator.start(stack.plan, stack.preflight())
        stack.operations.outcome(stack.run)
        stack.operations.logs(
            RunLogsRequest(stack.run, limit=1, maximum_bytes=4096)
        )
        stack.operations.outcome(stack.run)
        stack.operations.verify(stack.run)
        tuple(
            stack.operations.artifacts(
                RunArtifactRequest(stack.run, "adapter", 100)
            ).iter_bytes()
        )
        traces.append(stack.family.trace.snapshot())
    assert traces[1:] == [traces[0]] * 3


@pytest.mark.parametrize("provider_id", PROFILES)
def test_all_profiles_share_stage_submit_cancel_and_reconcile_semantics(provider_id) -> None:
    configuration = config(provider_id)
    family = make_family(configuration)
    stage, submit, cancel = configuration.effects

    stage_request = execution_request(configuration, stage)
    stage_observation = family.executor_resolver.resolve(stage_request).executor.execute_once(
        object(), stage_request
    )
    assert stage_observation.disposition is ObservationDisposition.FOUND
    assert stage_observation.stage_ref.provider_id == provider_id

    for script in (submit, cancel):
        request = execution_request(configuration, script)
        first = family.executor_resolver.resolve(request).executor.execute_once(object(), request)
        assert first.disposition is ObservationDisposition.INDETERMINATE
        lookup_request = ReconciliationResolutionRequestV2(
            script.command_digest,
            configuration.adapter_descriptor.digest,
            provider_id,
            configuration.provider.profile_ref,
            configuration.account_ref,
            configuration.namespace_ref,
        )
        adapter = family.reconciliation_resolver.resolve(lookup_request).adapter
        found = adapter.lookup(
            SimpleNamespace(
                command_digest=script.command_digest,
                resolution_digest=lookup_request.digest,
                ownership_epoch=2,
            ),
            object(),
        )
        assert found.disposition is ObservationDisposition.FOUND
        assert (found.provider_run is not None) is (script.kind is EffectKind.SUBMIT)
        assert (found.cancellation is not None) is (script.kind is EffectKind.CANCEL)


def test_normalized_role_trace_is_identical_for_all_configured_profiles() -> None:
    traces = []
    for provider_id in PROFILES:
        configuration = config(provider_id)
        family = make_family(configuration)
        for effect in configuration.effects:
            request = execution_request(configuration, effect)
            family.executor_resolver.resolve(request).executor.execute_once(object(), request)
        traces.append(family.trace.snapshot())
    assert traces[1:] == [traces[0]] * (len(traces) - 1)


@pytest.mark.parametrize("provider_id", PROFILES)
def test_registry_discovery_is_zero_invocation_then_exact_resolution(provider_id) -> None:
    configuration = config(provider_id)
    family = make_family(configuration)
    registry = LazyProviderRegistryV2()
    registry.register(
        ProviderRegistrationV2(
            configuration.descriptor,
            configuration.executor_descriptor,
            configuration.adapter_descriptor,
            family.executor_resolver,
            family.reconciliation_resolver,
            family.reader_factory,
        )
    )
    assert registry.list() == (configuration.descriptor,)
    assert registry.reader_factory(configuration.provider) is family.reader_factory
    assert family.trace.snapshot() == ()
    request = ProviderReaderFactoryRequestV1(
        configuration.provider,
        configuration.descriptor.descriptor_digest,
        configuration.profile_digest,
        configuration.account_ref,
        configuration.namespace_ref,
    )
    resolved = registry.resolve_reader(request)
    assert resolved.reader is family.reader
    assert family.trace.snapshot() == (("reader_factory", "create"),)


@pytest.mark.parametrize("provider_id", PROFILES)
def test_reader_factory_rejects_cross_scope_substitution(provider_id) -> None:
    configuration = config(provider_id)
    family = make_family(configuration)
    request = ProviderReaderFactoryRequestV1(
        configuration.provider,
        configuration.descriptor.descriptor_digest,
        configuration.profile_digest,
        "foreign-account",
        configuration.namespace_ref,
    )
    with pytest.raises(ValueError, match="request mismatch"):
        family.reader_factory.create(request)


def test_profile_identifiers_are_only_configuration_data() -> None:
    source = __import__("pathlib").Path(
        "tuner/execution/fake_provider_v1.py"
    ).read_text(encoding="utf-8")
    assert all(name not in source for name in PROFILES)


def test_script_cursor_is_thread_safe_and_replay_stabilizes_at_last_result() -> None:
    configuration = config("configured-provider")
    stage = replace(
        configuration.effects[0],
        dispatch=(
            FakeEffectResultV1(ObservationDisposition.INDETERMINATE),
            FakeEffectResultV1(ObservationDisposition.FOUND),
        ),
    )
    configuration = replace(
        configuration, effects=(stage,) + configuration.effects[1:]
    )
    family = make_family(configuration)
    request = execution_request(configuration, stage)

    def execute():
        return family.executor.execute_once(object(), request).disposition

    with ThreadPoolExecutor(max_workers=2) as pool:
        dispositions = tuple(pool.map(lambda _: execute(), range(2)))
    assert set(dispositions) == {
        ObservationDisposition.INDETERMINATE,
        ObservationDisposition.FOUND,
    }
    assert execute() is ObservationDisposition.FOUND


def test_authenticated_fake_evidence_rejects_tag_or_authority_substitution() -> None:
    family = make_family(config("configured-provider"))
    content = ProviderRunObservationContentV1(
        "synaptic-provider-run-observation-content/v1",
        D[0],
        D[1],
        7,
        TrainingRunRef("run", "project"),
        D[2],
        "configured-provider",
        "profile",
        "account",
        "namespace",
        "job-1",
        ProviderRunPhaseV1.RUNNING,
        canonical_bytes({"phase": "running"}),
        None,
        "fake-reader",
        "1.0.0",
        "2026-08-27T12:00:00Z",
    )
    envelope = family.evidence_authority.observation(content)
    assert family.evidence_authority.authenticate(envelope) is True
    assert family.evidence_authority.authenticate(replace(envelope, tag="0" * 64)) is False
    foreign = FakeProviderFamilyV1(
        config("other-provider"),
        evidence_key=b"z" * 32,
        foundation_authenticator=UnusedFoundationAuth(),
        assessment_authenticator=UnusedAssessmentAuth(),
    )
    assert foreign.evidence_authority.authenticate(envelope) is False


def test_fake_reader_and_verifier_compose_with_exact_coordinator_evidence() -> None:
    queued, foundation_record = queued_evidence()
    submit_command = queued.submit
    executor = parse_exact_command(submit_command.canonical_command_bytes).executor
    content = b"adapter-data"
    artifact = FakeArtifactV1(
        VerifiedArtifact("adapter", hashlib.sha256(content).hexdigest(), len(content)),
        content,
    )
    configuration = FakeProviderConfigV1(
        PROVIDER,
        DESC,
        D[4],
        SCOPE.account_ref,
        SCOPE.namespace_ref,
        executor,
        AdapterDescriptorV1(PROVIDER.provider_id, "adapter", "1.0.0"),
        (
            FakeEffectScriptV1(
                submit_command.effect_id,
                submit_command.command_digest,
                executor.digest,
                EffectKind.SUBMIT,
                (FakeEffectResultV1(ObservationDisposition.FOUND),),
            ),
        ),
        (ProviderRunPhaseV1.SUCCEEDED,),
        (
            RunLogEntry(
                1,
                "2026-08-27T12:00:00Z",
                RunLogLevel.INFO,
                "complete",
                "ok",
                2,
            ),
        ),
        (artifact,),
    )
    family = FakeProviderFamilyV1(
        configuration,
        evidence_key=b"e" * 32,
        foundation_authenticator=Auth(),
        assessment_authenticator=AssessmentAuth(),
    )
    assessed = assessment(foundation_record)

    observe_request = provider_run_read_request(
        queued,
        foundation_record,
        assessed,
        Auth(),
        AssessmentAuth(),
        purpose=ProviderReadPurposeV1.OBSERVE,
    )
    observation = family.reader.observe(observe_request)
    assert family.evidence_authority.authenticate(observation) is True
    succeeded = apply_provider_observation(
        queued, observe_request, observation, family.evidence_authority
    )

    log_request = provider_run_read_request(
        succeeded,
        foundation_record,
        assessed,
        Auth(),
        AssessmentAuth(),
        purpose=ProviderReadPurposeV1.LOGS,
    )
    log_page = family.reader.logs(log_request, ProviderLogQueryV1(None, 1, 4096))
    assert family.evidence_authority.authenticate(log_page) is True
    assert log_page.content.entries[0].message == "ok"

    artifact_request = provider_run_read_request(
        succeeded,
        foundation_record,
        assessed,
        Auth(),
        AssessmentAuth(),
        purpose=ProviderReadPurposeV1.ARTIFACTS,
    )
    manifest = family.reader.artifacts(artifact_request)
    receipt = family.artifact_verifier.verify(succeeded, manifest)
    assert family.artifact_verifier.authenticate(receipt) is True
    assert tuple(
        family.reader.iter_artifact_bytes(
            artifact_request,
            manifest,
            "adapter",
            maximum_bytes=len(content),
        )
    ) == (content,)
