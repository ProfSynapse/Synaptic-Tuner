from types import SimpleNamespace

import pytest

from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1, build_cancel_command, build_stage_command,
    build_submit_command,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.executors import ExecutionResolutionRequestV2
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import (
    CancellationRefV1, ProviderRunRefV1, StagePredecessorV2,
)
from tuner.execution.providers.docker_provider_v1.effects import (
    DockerEffectExecutorV1, DockerReconciliationAdapterV1,
)
from tuner.execution.providers.docker_provider_v1.model import (
    AuthenticatedDockerAbsenceV1, AuthenticatedDockerCommandBindingV1,
    DockerAbsenceContentV1,
    DockerCommandBindingV1, DockerLookupDispositionV1, DockerLookupPurposeV1,
    DockerLookupResultV1,
    DockerEffectIdentityV1, PreparedDockerPlanV1, labels_for,
)
from tests.execution.providers.docker_provider_v1.conftest import Authority, D


def resolution(profile, binding):
    return ExecutionResolutionRequestV2(
        binding.command_digest, profile.executor_descriptor.digest,
        profile.provider.provider_id, profile.provider.profile_ref,
        profile.scope.account_ref, profile.scope.namespace_ref,
        binding.effect_kind, f"{binding.effect_kind}-payload/v2",
        profile.workload.workload_digest,
    )


def recovery_payload(profile, kind):
    return CanonicalProviderPayloadV1.build(
        "docker", f"{kind}-payload/v2", profile.workload.workload_digest,
    )


def execution_binding(profile):
    return ProviderExecutionBindingV1(
        profile.provider, profile.descriptor.descriptor_digest,
        profile.profile_digest, profile.scope, profile.executor_descriptor,
        profile.adapter_descriptor.digest, profile.resource_digest,
        profile.quote_digest, profile.secret_requirements_digest,
    )


def setup_effect_recovery(profile, plan, run, catalog, kind):
    prep = CanonicalPreparationV2.build(
        provider=profile.provider, scope=profile.scope, project_ref=run.project_ref,
        run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
        source_digest=plan.basis.source_digest,
        workload_digest=profile.workload.workload_digest,
        runtime_digest=profile.runtime.digest, resource_digest=profile.resource_digest,
        artifact_contract_digest=profile.artifacts.digest,
        quote_digest=profile.quote_digest,
        secret_requirements_digest=profile.secret_requirements_digest,
        execution_binding_digest=execution_binding(profile).binding_digest,
    )
    provider_payload = recovery_payload(profile, kind)
    if kind == "stage":
        command = build_stage_command(
            prep, "nonce", provider_payload, profile.executor_descriptor,
        )
    else:
        command = build_cancel_command(
            prep, "nonce", provider_payload, profile.executor_descriptor,
            CancellationRefV1(ProviderRunRefV1("container-1"), D[1]),
        )
    prepared = PreparedDockerPlanV1(
        profile, run.project_ref, run.run_id, plan.plan_fingerprint,
        plan.basis.source_digest, prep.preparation_digest,
    )
    identity = DockerEffectIdentityV1(
        command.digest, command.operation.effect.effect_id, kind, prepared,
    )
    if kind == "stage":
        binding = DockerCommandBindingV1(identity, command.canonical_bytes)
    else:
        predecessor = StagePredecessorV2(
            "docker", profile.provider.profile_ref, profile.scope.account_ref,
            profile.scope.namespace_ref, run.project_ref, run.run_id,
            plan.plan_fingerprint, prep.preparation_digest,
            profile.workload.workload_digest, "stage-effect", D[0], D[1],
        )
        submit_command = build_submit_command(
            prep, "submit-nonce", recovery_payload(profile, "submit"),
            profile.executor_descriptor, predecessor,
        )
        submit = DockerEffectIdentityV1(
            submit_command.digest, submit_command.operation.effect.effect_id,
            "submit", prepared,
        )
        binding = DockerCommandBindingV1(
            identity, command.canonical_bytes, submit_command.canonical_bytes,
            cancel_container_ref="container-1",
            cancel_reason_digest=D[1], cancel_submit_labels=labels_for(submit),
            cancel_authorization_digest=D[5],
        )
    catalog.values[command.digest] = binding
    target = SimpleNamespace(
        command_bytes=command.canonical_bytes, command_digest=command.digest,
        effect_id=binding.effect_id, resolution_digest=D[3], ownership_epoch=3,
    )
    return prep, binding, target


def setup_recovery(profile, plan, run, catalog, control):
    prep = CanonicalPreparationV2.build(
        provider=profile.provider, scope=profile.scope, project_ref=run.project_ref,
        run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
        source_digest=plan.basis.source_digest, workload_digest=profile.workload.workload_digest,
        runtime_digest=profile.runtime.digest, resource_digest=profile.resource_digest,
        artifact_contract_digest=profile.artifacts.digest, quote_digest=profile.quote_digest,
        secret_requirements_digest=profile.secret_requirements_digest,
        execution_binding_digest=execution_binding(profile).binding_digest,
    )
    payload = CanonicalProviderPayloadV1.build("docker", "submit-payload/v2", profile.workload.workload_digest)
    predecessor = StagePredecessorV2(
        "docker", profile.provider.profile_ref, profile.scope.account_ref, profile.scope.namespace_ref,
        run.project_ref, run.run_id, plan.plan_fingerprint, prep.preparation_digest,
        profile.workload.workload_digest, "stage-effect", D[0], D[1],
    )
    command = build_submit_command(prep, "nonce", payload, profile.executor_descriptor, predecessor)
    prepared = PreparedDockerPlanV1(
        profile, run.project_ref, run.run_id, plan.plan_fingerprint,
        plan.basis.source_digest, prep.preparation_digest,
    )
    binding = DockerCommandBindingV1(DockerEffectIdentityV1(
        command.digest, command.operation.effect.effect_id, "submit", prepared
    ), command.canonical_bytes)
    catalog.values[command.digest] = binding
    target = SimpleNamespace(
        command_bytes=command.canonical_bytes, command_digest=command.digest,
        effect_id=binding.effect_id, resolution_digest=D[2], ownership_epoch=3,
    )
    return prep, binding, target


@pytest.mark.parametrize("disposition,expected", (
    (DockerLookupDispositionV1.FOUND, ObservationDisposition.FOUND),
    (DockerLookupDispositionV1.DEFINITELY_ABSENT, ObservationDisposition.DEFINITELY_ABSENT),
    (DockerLookupDispositionV1.INDETERMINATE, ObservationDisposition.INDETERMINATE),
    (DockerLookupDispositionV1.MULTIPLE, ObservationDisposition.INDETERMINATE),
))
def test_recovery_is_lookup_only_for_all_nonabsence_outcomes(profile, plan, run, seams, disposition, expected):
    catalog, _, source, control, cancellations = seams
    prep, binding, target = setup_recovery(profile, plan, run, catalog, control)
    labels = labels_for(binding.identity)
    if disposition is DockerLookupDispositionV1.FOUND:
        control.lookup_result = DockerLookupResultV1(disposition, labels, "container-1", __import__(
            "tuner.execution.providers.docker_provider_v1.model", fromlist=["DockerRunPhaseV1"]
        ).DockerRunPhaseV1.RUNNING)
    else:
        control.lookup_disposition = disposition
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority()
    ).lookup(target, prep)
    assert result.disposition is expected
    assert tuple(event[0] for event in control.trace) == ("lookup",)


def test_recovery_rejects_found_container_with_wrong_exact_labels(profile, plan, run, seams):
    catalog, _, source, control, cancellations = seams
    prep, binding, target = setup_recovery(profile, plan, run, catalog, control)
    wrong_identity = DockerEffectIdentityV1(D[3], binding.effect_id, "submit", binding.plan)
    wrong = labels_for(wrong_identity)
    phase = __import__(
        "tuner.execution.providers.docker_provider_v1.model", fromlist=["DockerRunPhaseV1"]
    ).DockerRunPhaseV1.RUNNING
    control.lookup_result = DockerLookupResultV1(
        DockerLookupDispositionV1.FOUND, wrong, "container-1", phase
    )
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority()
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in control.trace) == ("lookup",)


def test_reconciliation_rejects_separate_foreign_execution_binding_before_lookup_ports(
        profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prep, _, target = setup_recovery(profile, plan, run, catalog, control)
    document = prep.to_dict()
    document["execution_binding_digest"] = D[14]
    hostile = CanonicalPreparationV2.parse(canonical_bytes(document))
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, hostile)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert images.calls == source.lookup_calls == 0
    assert control.trace == cancellations.trace == []


@pytest.mark.parametrize("failure", ("purpose", "generation", "signer"))
def test_absence_requires_exact_authenticated_purpose_and_generation(profile, plan, run, seams, failure):
    catalog, _, source, control, cancellations = seams
    prep, _, target = setup_recovery(profile, plan, run, catalog, control)
    original = control.lookup
    def forged(request):
        value = original(request)
        content = value.absence.content
        if failure == "purpose":
            content = DockerAbsenceContentV1(
                content.request_digest, content.labels_digest,
                DockerLookupPurposeV1.OBSERVE, content.generation, content.evidence_digest,
            )
        elif failure == "generation":
            content = DockerAbsenceContentV1(
                content.request_digest, content.labels_digest,
                content.purpose, content.generation + 1, content.evidence_digest,
            )
        return DockerLookupResultV1(
            DockerLookupDispositionV1.DEFINITELY_ABSENT,
            absence=AuthenticatedDockerAbsenceV1(content, "docker-test", "key-v1", D[13]),
        )
    control.lookup_disposition = DockerLookupDispositionV1.DEFINITELY_ABSENT
    control.lookup = forged
    authority = Authority()
    if failure == "signer": authority.authenticate_absence = lambda value: False
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, authority
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in control.trace) == ("lookup",)


def test_stage_lost_return_recovers_only_from_exact_source_seal(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prep, binding, target = setup_effect_recovery(profile, plan, run, catalog, "stage")
    source.lost_return = True
    execution = DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, Authority(),
    ).execute_once(recovery_payload(profile, "stage"), resolution(profile, binding))
    assert execution.disposition is ObservationDisposition.INDETERMINATE
    recovered = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert recovered.disposition is ObservationDisposition.FOUND
    assert recovered.stage_ref.stage_ref == "stage-sealed"
    assert source.mutations == 1
    assert source.lookup_calls == 1
    assert control.trace == []


def test_cancel_lost_return_recovers_only_from_retained_stop_evidence(profile, plan, run, seams):
    catalog, images, source, control, cancellations = seams
    prep, binding, target = setup_effect_recovery(profile, plan, run, catalog, "cancel")
    cancellations.lost_return = True
    execution = DockerEffectExecutorV1(
        profile, catalog, catalog.binding_authority,
        images, source, control, cancellations, Authority(),
    ).execute_once(recovery_payload(profile, "cancel"), resolution(profile, binding))
    assert execution.disposition is ObservationDisposition.INDETERMINATE
    recovered = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert recovered.disposition is ObservationDisposition.FOUND
    assert recovered.cancellation.run.provider_job_ref == "container-1"
    assert cancellations.mutations == 1
    assert tuple(event[0] for event in cancellations.trace) == ("stop", "cancel_lookup")
    assert control.trace == []


def test_running_container_without_cancellation_evidence_is_not_cancel_success(profile, plan, run, seams):
    catalog, _, source, control, cancellations = seams
    prep, _, target = setup_effect_recovery(profile, plan, run, catalog, "cancel")
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert tuple(event[0] for event in cancellations.trace) == ("cancel_lookup",)
    assert control.trace == []


@pytest.mark.parametrize("disposition,expected", (
    (DockerLookupDispositionV1.DEFINITELY_ABSENT, ObservationDisposition.DEFINITELY_ABSENT),
    (DockerLookupDispositionV1.INDETERMINATE, ObservationDisposition.INDETERMINATE),
    (DockerLookupDispositionV1.MULTIPLE, ObservationDisposition.INDETERMINATE),
))
def test_cancel_recovery_uses_only_purpose_specific_evidence(profile, plan, run, seams, disposition, expected):
    catalog, _, source, control, cancellations = seams
    prep, _, target = setup_effect_recovery(profile, plan, run, catalog, "cancel")
    cancellations.lookup_disposition = disposition
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert result.disposition is expected
    assert control.trace == []


@pytest.mark.parametrize("kind", ("stage", "submit", "cancel"))
def test_every_effect_specific_lookup_exception_is_closed_and_indeterminate(profile, plan, run, seams, kind):
    catalog, _, source, control, cancellations = seams
    if kind == "submit":
        prep, _, target = setup_recovery(profile, plan, run, catalog, control)
        control.lookup = lambda request: (_ for _ in ()).throw(RuntimeError("raw-secret-sentinel"))
    else:
        prep, _, target = setup_effect_recovery(profile, plan, run, catalog, kind)
        port = source if kind == "stage" else cancellations
        port.lookup = lambda request: (_ for _ in ()).throw(RuntimeError("raw-secret-sentinel"))
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE


@pytest.mark.parametrize("kind", ("stage", "submit", "cancel"))
def test_recovery_rejects_unpinned_binding_before_every_lookup_port(profile, plan, run, seams, kind):
    catalog, _, source, control, cancellations = seams
    if kind == "submit":
        prep, binding, target = setup_recovery(profile, plan, run, catalog, control)
    else:
        prep, binding, target = setup_effect_recovery(profile, plan, run, catalog, kind)
    catalog.values[binding.command_digest] = AuthenticatedDockerCommandBindingV1(
        binding, binding.binding_digest, "rebound-authority",
        catalog.binding_authority.key_ref, D[14],
    )
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert source.lookup_calls == 0
    assert control.trace == cancellations.trace == []


@pytest.mark.parametrize("failure", ("authorization", "target", "original_labels"))
def test_recovery_rejects_rewrapped_cancel_binding_before_lookup(profile, plan, run, seams, failure):
    catalog, _, source, control, cancellations = seams
    prep, binding, target = setup_effect_recovery(profile, plan, run, catalog, "cancel")
    owned = catalog.resolve(binding.command_digest)
    if failure == "authorization":
        altered = __import__("dataclasses").replace(
            binding, cancel_authorization_digest=D[6],
        )
    elif failure == "target":
        altered = __import__("dataclasses").replace(
            binding, cancel_container_ref="container-2",
        )
    else:
        other_submit = DockerEffectIdentityV1(
            D[6], "other-submit", "submit", binding.plan,
        )
        altered = __import__("dataclasses").replace(
            binding, cancel_submit_labels=labels_for(other_submit),
        )
    catalog.values[binding.command_digest] = AuthenticatedDockerCommandBindingV1(
        altered, altered.binding_digest, owned.authority_ref, owned.key_ref, owned.tag,
    )
    result = DockerReconciliationAdapterV1(
        profile, catalog, catalog.binding_authority,
        control, source, cancellations, Authority(),
    ).lookup(target, prep)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    assert source.lookup_calls == 0
    assert control.trace == cancellations.trace == []
