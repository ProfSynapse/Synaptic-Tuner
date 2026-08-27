from concurrent.futures import ThreadPoolExecutor

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, ProviderPlanRef, TrainingPlan
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from synaptic_tuner.api.v1.runs_facade import RunArtifactRequest, RunLogsRequest
from tuner.execution.coordinator_v1.coordinator import TrainingCoordinatorV1
from tuner.execution.coordinator_v1.cursors import HMACCursorAuthorityV1
from tuner.execution.coordinator_v1.foundation import (
    ComposedEffectFoundationV1, FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.stores import (
    InMemoryExecutionGrantStoreV1, InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1, InMemoryWorkflowStoreV1,
)
from tuner.execution.coordinator_v1.operations import TrainingOperationsV1
from tuner.execution.foundation_v2.authority import GrantAuthorityV2, ReconciliationGrantContentV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1, build_cancel_command, build_stage_command,
    build_submit_command,
)
from tuner.execution.foundation_v2.executors import ExecutionResolutionRequestV2
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import InvalidEvidenceAuthorityV2, ReceiptAuthorityV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import InMemoryEffectRepositoryV2
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import (
    CancellationRefV1, ProviderRunRefV1, StagePredecessorV2,
)
from tuner.execution.providers.docker_provider_v1.effects import DockerEffectExecutorV1
from tuner.execution.providers.docker_provider_v1.effects import (
    DockerExecutorResolverV1, DockerReconciliationAdapterV1,
    DockerReconciliationResolverV1,
)
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerCommandBindingV1, DockerCommandBindingV1,
    DockerCreateDispositionV1, DockerProviderError,
    DockerEffectIdentityV1, PreparedDockerPlanV1, labels_for,
)
from tests.execution.providers.docker_provider_v1.conftest import Authority, D
from tests.execution.coordinator_v1.test_start_reconcile_service import (
    Clock, FoundationAuthenticator, TrustedEvidence,
)
from tests.execution.foundation_v2.helpers import StrongVerifier
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerBindingResolverV1, DockerPreparationMaterializerV1,
)
from tuner.execution.providers.docker_provider_v1.reader import DockerProviderRunReaderV1
from tuner.execution.fake_provider_v1 import (
    FakeArtifactVerifierV1, FakeProviderEvidenceAuthorityV1, FakeTraceV1,
)
from tests.execution.providers.docker_provider_v1.test_reader import (
    Authority as ReadAuthority, ReadPort,
)


def prepared(profile, plan, run):
    preparation = CanonicalPreparationV2.build(
        provider=profile.provider, scope=profile.scope,
        project_ref=run.project_ref, run_id=run.run_id,
        plan_fingerprint=plan.plan_fingerprint,
        source_digest=plan.basis.source_digest,
        workload_digest=profile.workload.workload_digest,
        runtime_digest=profile.runtime.digest,
        resource_digest=profile.resource_digest,
        artifact_contract_digest=profile.artifacts.digest,
        quote_digest=profile.quote_digest,
        secret_requirements_digest=profile.secret_requirements_digest,
    )
    return PreparedDockerPlanV1(
        profile, run.project_ref, run.run_id, plan.plan_fingerprint,
        plan.basis.source_digest, preparation.preparation_digest,
    )


def make_binding(kind, digest, effect_id, prepared_plan, **values):
    profile = prepared_plan.profile
    preparation = CanonicalPreparationV2.build(
        provider=profile.provider, scope=profile.scope,
        project_ref=prepared_plan.project_ref, run_id=prepared_plan.run_id,
        plan_fingerprint=prepared_plan.plan_fingerprint,
        source_digest=prepared_plan.source_digest,
        workload_digest=profile.workload.workload_digest,
        runtime_digest=profile.runtime.digest,
        resource_digest=profile.resource_digest,
        artifact_contract_digest=profile.artifacts.digest,
        quote_digest=profile.quote_digest,
        secret_requirements_digest=profile.secret_requirements_digest,
    )
    provider_payload = payload(profile, kind)
    if kind == "stage":
        command = build_stage_command(
            preparation, effect_id, provider_payload, profile.executor_descriptor,
        )
    elif kind == "submit":
        predecessor = StagePredecessorV2(
            profile.provider.provider_id, profile.provider.profile_ref,
            profile.scope.account_ref, profile.scope.namespace_ref,
            prepared_plan.project_ref, prepared_plan.run_id,
            prepared_plan.plan_fingerprint, preparation.preparation_digest,
            profile.workload.workload_digest, "stage-effect", D[0], D[1],
        )
        command = build_submit_command(
            preparation, effect_id, provider_payload,
            profile.executor_descriptor, predecessor,
        )
    else:
        command = build_cancel_command(
            preparation, effect_id, provider_payload, profile.executor_descriptor,
            CancellationRefV1(
                ProviderRunRefV1(values["cancel_container_ref"]),
                values["cancel_reason_digest"],
            ),
        )
    identity = DockerEffectIdentityV1(
        command.digest, command.operation.effect.effect_id, kind, prepared_plan,
    )
    return DockerCommandBindingV1(
        identity, command.canonical_bytes,
        values.pop("original_submit_command_bytes", None), **values,
    )


def request(profile, binding):
    return ExecutionResolutionRequestV2(
        binding.command_digest, profile.executor_descriptor.digest, profile.provider.provider_id,
        profile.provider.profile_ref, profile.scope.account_ref, profile.scope.namespace_ref,
        binding.effect_kind, f"{binding.effect_kind}-payload/v2", profile.workload.workload_digest,
    )


def payload(profile, kind):
    return CanonicalProviderPayloadV1.build("docker", f"{kind}-payload/v2", profile.workload.workload_digest)


def executor(profile, catalog, images, source, control, cancellations):
    return DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, Authority()
    )


def test_stage_only_checks_image_and_read_only_source(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    binding = make_binding("stage", D[9], "stage-effect", prepared(profile, plan, run))
    catalog.values[binding.command_digest] = binding
    result = executor(profile, catalog, images, source, control, cancellations).execute_once(payload(profile, "stage"), request(profile, binding))
    assert result.disposition is ObservationDisposition.FOUND
    assert result.stage_ref.stage_ref == "stage-sealed"
    assert images.calls == source.calls == 1
    assert control.trace == []


def test_submit_is_exactly_create_then_start_and_replay_does_not_mutate(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    binding = make_binding("submit", D[10], "submit-effect", prepared(profile, plan, run))
    catalog.values[binding.command_digest] = binding
    effect = executor(profile, catalog, images, source, control, cancellations)
    first = effect.execute_once(payload(profile, "submit"), request(profile, binding))
    second = effect.execute_once(payload(profile, "submit"), request(profile, binding))
    assert first.disposition is ObservationDisposition.FOUND
    assert second.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in control.trace) == ("create", "start")


def test_concurrent_submit_has_one_create_and_one_start(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    binding = make_binding("submit", D[11], "submit-effect", prepared(profile, plan, run))
    catalog.values[binding.command_digest] = binding
    effect = executor(profile, catalog, images, source, control, cancellations)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = tuple(pool.map(lambda _: effect.execute_once(payload(profile, "submit"), request(profile, binding)), range(8)))
    assert sum(v.disposition is ObservationDisposition.FOUND for v in values) == 1
    assert tuple(event[0] for event in control.trace) == ("create", "start")


def test_reconstructed_executor_relies_on_idempotent_control_mutations(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    binding = make_binding("submit", D[11], "submit-effect", prepared(profile, plan, run))
    catalog.values[binding.command_digest] = binding
    first = executor(profile, catalog, images, source, control, cancellations)
    second = executor(profile, catalog, images, source, control, cancellations)
    assert first.execute_once(payload(profile, "submit"), request(profile, binding)).disposition is ObservationDisposition.FOUND
    assert second.execute_once(payload(profile, "submit"), request(profile, binding)).disposition is ObservationDisposition.FOUND
    assert control.create_mutations == control.start_mutations == 1
    assert tuple(event[0] for event in control.trace) == ("create", "start", "create", "start")


def test_collision_never_starts_and_partial_created_is_never_restarted(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    collision = make_binding("submit", D[12], "submit-collision", prepared(profile, plan, run))
    catalog.values[collision.command_digest] = collision
    control.create_disposition = DockerCreateDispositionV1.COLLISION
    with pytest.raises(DockerProviderError):
        executor(profile, catalog, images, source, control, cancellations).execute_once(payload(profile, "submit"), request(profile, collision))
    assert tuple(event[0] for event in control.trace) == ("create",)
    control.trace.clear(); control.create_disposition = DockerCreateDispositionV1.CREATED; control.start_result = False
    partial = make_binding("submit", D[13], "submit-partial", prepared(profile, plan, run))
    catalog.values[partial.command_digest] = partial
    effect = executor(profile, catalog, images, source, control, cancellations)
    assert effect.execute_once(payload(profile, "submit"), request(profile, partial)).disposition is ObservationDisposition.INDETERMINATE
    assert effect.execute_once(payload(profile, "submit"), request(profile, partial)).disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in control.trace) == ("create", "start")


def test_cancel_calls_stop_once(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prepared_plan = prepared(profile, plan, run)
    submit = make_binding("submit", D[2], "submit-effect", prepared_plan)
    binding = make_binding(
        "cancel", D[0], "cancel-effect", prepared_plan,
        cancel_container_ref="container-1", cancel_reason_digest=D[1],
        cancel_submit_labels=labels_for(submit.identity),
        original_submit_command_bytes=submit.command_bytes,
        cancel_authorization_digest=D[5],
    )
    catalog.values[binding.command_digest] = binding
    result = executor(profile, catalog, images, source, control, cancellations).execute_once(payload(profile, "cancel"), request(profile, binding))
    assert result.disposition is ObservationDisposition.FOUND
    assert tuple(event[0] for event in cancellations.trace) == ("stop",)
    assert cancellations.requests[0].submit_labels.effect_kind == "submit"
    assert cancellations.requests[0].cancellation_identity.effect_kind == "cancel"
    assert cancellations.requests[0].submit_labels.effect_identity_digest == submit.identity.digest


def test_stage_rejects_wrong_effect_bound_source_seal(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    bound = make_binding("stage", D[5], "stage-effect", prepared(profile, plan, run))
    other = make_binding("stage", D[6], "other-stage", bound.plan)
    catalog.values[bound.command_digest] = bound
    original = source.seal_read_only
    def forged(request):
        value = original(request)
        from dataclasses import replace
        return replace(value, content=replace(value.content, effect_identity_digest=other.identity.digest))
    source.seal_read_only = forged
    with pytest.raises(DockerProviderError) as caught:
        executor(profile, catalog, images, source, control, cancellations).execute_once(
            payload(profile, "stage"), request(profile, bound)
        )
    assert str(caught.value) == "docker_source_unsealed"


def test_cancel_rejects_alternate_signer_without_second_stop(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prepared_plan = prepared(profile, plan, run)
    submit = make_binding("submit", D[2], "submit-effect", prepared_plan)
    cancel = make_binding(
        "cancel", D[3], "cancel-effect", prepared_plan,
        cancel_container_ref="container-1", cancel_reason_digest=D[4],
        cancel_submit_labels=labels_for(submit.identity),
        original_submit_command_bytes=submit.command_bytes,
        cancel_authorization_digest=D[5],
    )
    catalog.values[cancel.command_digest] = cancel
    authority = Authority()
    authority.authenticate_cancellation = lambda value: False
    effect = DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, authority
    )
    first = effect.execute_once(payload(profile, "cancel"), request(profile, cancel))
    second = effect.execute_once(payload(profile, "cancel"), request(profile, cancel))
    assert first.disposition is second.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in cancellations.trace) == ("stop",)


def test_malformed_exact_cancellation_evidence_after_mutation_is_indeterminate(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prepared_plan = prepared(profile, plan, run)
    submit = make_binding("submit", D[2], "submit-effect", prepared_plan)
    cancel = make_binding(
        "cancel", D[3], "cancel-effect", prepared_plan,
        cancel_container_ref="container-1", cancel_reason_digest=D[4],
        cancel_submit_labels=labels_for(submit.identity),
        original_submit_command_bytes=submit.command_bytes,
        cancel_authorization_digest=D[5],
    )
    catalog.values[cancel.command_digest] = cancel
    original = cancellations.stop_once
    def malformed(request_value):
        evidence = original(request_value)
        object.__setattr__(evidence.content, "authorization_digest", "raw-secret-sentinel")
        return evidence
    cancellations.stop_once = malformed
    effect = executor(profile, catalog, images, source, control, cancellations)
    first = effect.execute_once(payload(profile, "cancel"), request(profile, cancel))
    second = effect.execute_once(payload(profile, "cancel"), request(profile, cancel))
    assert first.disposition is second.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in cancellations.trace) == ("stop",)


@pytest.mark.parametrize("failure", (
    "authority", "key", "authentication", "command", "original_submit",
    "cancel_authorization", "cancel_target", "original_labels",
))
def test_binding_validation_fails_before_every_effect_port(profile, plan, run, seams, failure):
    catalog, images, source, control, cancellations = seams
    prepared_plan = prepared(profile, plan, run)
    submit = make_binding("submit", D[2], "submit-effect", prepared_plan)
    cancel_failure = failure in {
        "original_submit", "cancel_authorization", "cancel_target", "original_labels",
    }
    binding = (submit if not cancel_failure else make_binding(
        "cancel", D[3], "cancel-effect", prepared_plan,
        cancel_container_ref="container-1", cancel_reason_digest=D[4],
        cancel_submit_labels=labels_for(submit.identity),
        original_submit_command_bytes=submit.command_bytes,
        cancel_authorization_digest=D[5],
    ))
    authority = Authority()
    binding_authority = catalog.binding_authority
    owned = binding_authority.issue(binding)
    if failure == "authority":
        owned = AuthenticatedDockerCommandBindingV1(
            binding, binding.binding_digest, "other-authority", owned.key_ref, D[14],
        )
    elif failure == "key":
        owned = AuthenticatedDockerCommandBindingV1(
            binding, binding.binding_digest, owned.authority_ref, "other-key", D[14],
        )
    elif failure == "authentication":
        owned = AuthenticatedDockerCommandBindingV1(
            binding, binding.binding_digest, owned.authority_ref, owned.key_ref, D[14],
        )
    elif failure == "command":
        other = make_binding(binding.effect_kind, D[6], "other-effect", prepared_plan,
            **({} if binding.effect_kind != "cancel" else {
                "cancel_container_ref": "container-1", "cancel_reason_digest": D[4],
                "cancel_submit_labels": labels_for(submit.identity),
                "original_submit_command_bytes": submit.command_bytes,
                "cancel_authorization_digest": D[5],
            }))
        altered = __import__("dataclasses").replace(binding, command_bytes=other.command_bytes)
        owned = AuthenticatedDockerCommandBindingV1(
            altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
        )
    elif failure == "original_submit":
        other_submit = make_binding("submit", D[7], "other-submit", prepared_plan)
        altered = __import__("dataclasses").replace(
            binding, original_submit_command_bytes=other_submit.command_bytes,
        )
        owned = AuthenticatedDockerCommandBindingV1(
            altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
        )
    elif failure == "cancel_authorization":
        altered = __import__("dataclasses").replace(binding, cancel_authorization_digest=D[6])
        owned = AuthenticatedDockerCommandBindingV1(
            altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
        )
    elif failure == "cancel_target":
        altered = __import__("dataclasses").replace(binding, cancel_container_ref="container-2")
        owned = AuthenticatedDockerCommandBindingV1(
            altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
        )
    elif failure == "original_labels":
        other_submit = make_binding("submit", D[7], "other-submit", prepared_plan)
        altered = __import__("dataclasses").replace(
            binding, cancel_submit_labels=labels_for(other_submit.identity),
        )
        owned = AuthenticatedDockerCommandBindingV1(
            altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
        )
    catalog.values[binding.command_digest] = owned
    effect = DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, authority,
    )
    with pytest.raises(DockerProviderError) as caught:
        effect.execute_once(payload(profile, binding.effect_kind), request(profile, binding))
    assert str(caught.value) == "docker_binding_mismatch"
    assert images.calls == source.calls == 0
    assert control.trace == cancellations.trace == []


@pytest.mark.parametrize("scenario", ("lifecycle", "cancel", "stage_recovery", "cancel_recovery"))
def test_both_opaque_profiles_run_unchanged_coordinator_foundation_and_operations(profile, plan, run, seams, scenario):
    catalog, images, source, control, cancellations = seams
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        profile.descriptor.descriptor_digest, profile.profile_digest,
    )
    bound_plan = TrainingPlan(
        "synaptic-training-plan/v2", plan.basis,
        ProviderPlanRef(context.provider_context_digest),
    )
    materializer = DockerPreparationMaterializerV1(profile)
    docker_authority = Authority()
    docker_executor = DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, docker_authority
    )
    docker_adapter = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, docker_authority
    )
    grants = GrantAuthorityV2("grants", b"g" * 32)
    receipts = ReceiptAuthorityV2("receipts", b"r" * 32)
    invalid = InvalidEvidenceAuthorityV2("invalid", b"i" * 32)
    verifier = StrongVerifier()
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, verifier, verifier, grants
    )
    foundation_auth = FoundationAuthenticator(grants, receipts, invalid)
    assessments = FoundationRecordAssessmentAuthorityV1(
        "assessment-authority", "assessment-key", b"a" * 32,
        assessor_ref="foundation-assessor", assessor_version="1.0.0",
        clock=Clock(), receipt_authority=receipts,
        invalid_evidence_authority=invalid, grant_authority=grants,
    )
    broker = EffectBrokerV2(
        repository, DockerExecutorResolverV1(docker_executor), grants,
        receipts, invalid,
    )
    reconciliation = ReconciliationServiceV1(
        repository, grants, DockerReconciliationResolverV1(docker_adapter),
        receipts, invalid,
    )
    foundation = ComposedEffectFoundationV1(
        repository, broker, reconciliation, grant_authority=grants,
        receipt_authority=receipts, invalid_evidence_authority=invalid,
        assessment_authority=assessments,
        trusted_quiescence_evidence=TrustedEvidence(repository, verifier),
    )

    class Planning:
        def describe(self, provider):
            assert provider == profile.provider
            return profile.descriptor
    class Plans:
        def get_plan(self, fingerprint):
            return bound_plan if fingerprint == bound_plan.plan_fingerprint else None
        def get_context(self, digest):
            return context if digest == context.provider_context_digest else None
    class Identity:
        def for_plan(self, selected):
            assert selected == bound_plan
            return run
    class Authorization:
        def commit_preflight(self, selected, preflight):
            assert selected == bound_plan and preflight.binds(selected)
            return D[6]
        def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
            command = parse_exact_command(command_bytes)
            prepared_plan = materializer.prepared(command.preparation.preparation_digest)
            identity = DockerEffectIdentityV1(
                command.digest, command.operation.effect.effect_id,
                command.operation.effect.kind.value, prepared_plan,
            )
            if identity.effect_kind == "submit":
                submitted["binding"] = DockerCommandBindingV1(
                    identity, command.canonical_bytes,
                )
                binding_value = submitted["binding"]
            elif identity.effect_kind == "cancel":
                cancellation = command.to_dict()["cancellation"]
                binding_value = DockerCommandBindingV1(
                    identity, command.canonical_bytes,
                    submitted["binding"].command_bytes,
                    cancel_container_ref=cancellation["provider_job_ref"],
                    cancel_reason_digest=cancellation["reason_digest"],
                    cancel_submit_labels=labels_for(submitted["binding"].identity),
                    cancel_authorization_digest=D[11],
                )
                submitted["cancel_binding"] = binding_value
            else:
                binding_value = DockerCommandBindingV1(
                    identity, command.canonical_bytes,
                )
            catalog.values[command.digest] = binding_value
            grant = grants.issue(
                command_bytes, grant_ref="grant-" + identity.effect_kind,
                policy_digest=preflight_digest, requirement_digest=D[10],
                not_before_epoch=100, expires_at_epoch=200,
            )
            issued[command.digest] = grant
            return grant
        def issue_reconciliation_grant(self, record, binding, *, slot, now_epoch):
            command = parse_exact_command(record.command_bytes)
            content = ReconciliationGrantContentV1(
                "docker-reconciliation", command.digest,
                command.operation.effect.effect_id,
                command.preparation.preparation_digest,
                profile.adapter_descriptor.digest,
                profile.provider.provider_id, profile.provider.profile_ref,
                profile.scope.account_ref, profile.scope.namespace_ref,
                "docker-coordinator-owner", slot.generation, slot.ownership_epoch,
                D[9], D[10], 100, 200, grants.epoch,
                grants.revocation_generation,
            )
            return grants.issue_reconciliation(content)
    class Unused:
        def authenticate(self, value): return True

    submitted = {}
    issued = {}
    workflows = InMemoryWorkflowStoreV1(
        foundation_auth, assessments, Unused(), Unused()
    )
    preparation_store = InMemoryPreparationStoreV1()
    execution_grant_store = InMemoryExecutionGrantStoreV1(grants)
    reconciliation_grant_store = InMemoryReconciliationGrantStoreV1(grants)
    authorization = Authorization()
    coordinator = TrainingCoordinatorV1(
        Planning(), Plans(), workflows, preparation_store,
        execution_grant_store, reconciliation_grant_store,
        DockerBindingResolverV1(profile), materializer, authorization,
        foundation, foundation_auth, Clock(), Identity(),
    )
    preflight = TrainingPreflight(
        bound_plan.plan_fingerprint, True,
        "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
    )
    if scenario == "stage_recovery":
        source.lost_return = True
    result = coordinator.start(bound_plan, preflight)
    def reconcile_concurrently():
        def attempt(_):
            try:
                return coordinator.reconcile(run)
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            return tuple(pool.map(attempt, range(8)))
    if scenario == "stage_recovery":
        assert result.phase.value == "stage_reconcile_required"
        recovered = reconcile_concurrently()
        result = next(value for value in recovered if value is not None and value.phase.value == "queued")
        assert source.lookup_calls == 1
    assert result.phase.value == "queued"
    assert tuple(event[0] for event in control.trace) == ("create", "start")
    assert images.calls == source.calls == 1
    if scenario == "lifecycle":
        before = tuple(control.trace)
        reconstructed = DockerEffectExecutorV1(
            profile, catalog, catalog.binding_authority,
            images, source, control, cancellations, docker_authority,
        )
        reconstructed_broker = EffectBrokerV2(
            repository, DockerExecutorResolverV1(reconstructed), grants,
            receipts, invalid,
        )
        submit_binding = submitted["binding"]
        replay = reconstructed_broker.execute(
            submit_binding.command_bytes, issued[submit_binding.command_digest],
            now_epoch=150,
        )
        assert replay.command.digest == submit_binding.command_digest
        assert reconstructed._attempted == set()
        assert tuple(control.trace) == before
    if scenario in {"cancel", "cancel_recovery"}:
        if scenario == "cancel_recovery":
            cancellations.lost_return = True
        cancelled = coordinator.cancel(run, "requested")
        if scenario == "cancel_recovery":
            assert cancelled.phase.value == "cancel_reconcile_required"
            reconstructed_executor = DockerEffectExecutorV1(
                profile, catalog, catalog.binding_authority,
                images, source, control, cancellations, docker_authority,
            )
            reconstructed_adapter = DockerReconciliationAdapterV1(
                profile, catalog, catalog.binding_authority,
                control, source, cancellations, docker_authority,
            )
            reconstructed_broker = EffectBrokerV2(
                repository, DockerExecutorResolverV1(reconstructed_executor),
                grants, receipts, invalid,
            )
            reconstructed_reconciliation = ReconciliationServiceV1(
                repository, grants,
                DockerReconciliationResolverV1(reconstructed_adapter),
                receipts, invalid,
            )
            reconstructed_foundation = ComposedEffectFoundationV1(
                repository, reconstructed_broker, reconstructed_reconciliation,
                grant_authority=grants, receipt_authority=receipts,
                invalid_evidence_authority=invalid,
                assessment_authority=assessments,
                trusted_quiescence_evidence=TrustedEvidence(repository, verifier),
            )
            reconstructed_coordinator = TrainingCoordinatorV1(
                Planning(), Plans(), workflows, preparation_store,
                execution_grant_store, reconciliation_grant_store,
                DockerBindingResolverV1(profile), materializer, authorization,
                reconstructed_foundation, foundation_auth, Clock(), Identity(),
            )
            cancel_binding = submitted["cancel_binding"]
            replayed_ambiguous = reconstructed_broker.execute(
                cancel_binding.command_bytes, issued[cancel_binding.command_digest],
                now_epoch=150,
            )
            assert replayed_ambiguous.command.digest == cancel_binding.command_digest
            assert reconstructed_executor._attempted == set()
            assert tuple(event[0] for event in cancellations.trace) == ("stop",)
            cancelled = reconstructed_coordinator.reconcile(run)
            before_repeat = tuple(cancellations.trace)
            repeated = reconstructed_coordinator.reconcile(run)
            assert repeated == cancelled
            assert tuple(cancellations.trace) == before_repeat
            assert reconstructed_executor._attempted == set()
            assert cancellations.mutations == 1
            assert tuple(event[0] for event in cancellations.trace) == ("stop", "cancel_lookup")
            cancel_record = repository.get(cancel_binding.effect_id)
            assert cancel_record.state.value == "found"
            assert cancel_record.results[-1].content.source_kind == "reconciliation"
        assert cancelled.phase.value == "cancel_requested"
        assert tuple(event[0] for event in control.trace) == ("create", "start")
        if scenario == "cancel":
            assert tuple(event[0] for event in cancellations.trace) == ("stop",)
        return

    class AllowRead:
        def authenticate(self, value): return True
    submit_labels = labels_for(submitted["binding"].identity)
    read_port = ReadPort(submit_labels, b"fixture-result")
    from tuner.execution.providers.docker_provider_v1.model import DockerRunPhaseV1
    read_port.phase = DockerRunPhaseV1.SUCCEEDED
    read_authority = ReadAuthority()
    artifact_authority = FakeProviderEvidenceAuthorityV1(
        "docker-artifact-evidence", b"e" * 32
    )
    artifact_verifier = FakeArtifactVerifierV1(
        artifact_authority, FakeTraceV1(), "2026-08-27T12:00:00Z"
    )
    reader = DockerProviderRunReaderV1(
        profile, catalog, catalog.binding_authority,
        AllowRead(), read_authority, read_port,
        observed_at="2026-08-27T12:00:00Z",
    )
    operations = TrainingOperationsV1(
        Planning(), Plans(), workflows, coordinator, foundation,
        foundation_auth, assessments, reader, read_authority, read_authority,
        artifact_verifier,
        HMACCursorAuthorityV1("cursor-authority", {1: b"c" * 32}, active_generation=1),
        Clock(),
    )
    assert operations.outcome(run).state.value == "succeeded"
    assert operations.logs(RunLogsRequest(run, limit=1, maximum_bytes=4096)).entries[0].message == "q"
    assert operations.verify(run).verified is True
    artifact_stream = operations.artifacts(RunArtifactRequest(run, "result", 1024))
    assert b"".join(artifact_stream.iter_bytes()) == b"fixture-result"
    assert tuple(event[0] for event in control.trace) == ("create", "start")
