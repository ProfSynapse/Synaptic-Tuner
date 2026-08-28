from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1
from synaptic_tuner.api.v1.providers import ProviderRef
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.providers.docker_provider_v1.model import (
    DockerArtifactContractV1, DockerImageV1, DockerProviderError, DockerRootsV1,
    PreparedDockerPlanV1,
)
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerBindingResolverV1, DockerPreparationMaterializerV1,
)


def test_binding_and_preparation_exactly_translate_b3_contracts(profile, plan, run):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        profile.descriptor.descriptor_digest, profile.profile_digest,
    )
    binding = DockerBindingResolverV1(profile).resolve(profile.provider, context)
    materializer = DockerPreparationMaterializerV1(profile)
    preparation = materializer.prepare(plan, run, binding)
    assert preparation.source_digest == plan.basis.source_digest
    assert preparation.artifact_contract_digest == profile.artifacts.digest
    assert type(materializer.prepared(preparation)) is PreparedDockerPlanV1
    assert materializer.payload(preparation, EffectKind.SUBMIT).payload_kind == "submit-payload/v2"


def test_preparation_reconstructs_identically_after_materializer_restart(profile, plan, run):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        profile.descriptor.descriptor_digest, profile.profile_digest,
    )
    binding = DockerBindingResolverV1(profile).resolve(profile.provider, context)
    preparation = DockerPreparationMaterializerV1(profile).prepare(plan, run, binding)

    original = DockerPreparationMaterializerV1(profile).prepared(preparation)
    restarted = DockerPreparationMaterializerV1(profile)
    reconstructed = restarted.prepared(preparation)
    assert reconstructed == original
    assert reconstructed.digest == original.digest
    assert restarted.payload(preparation, EffectKind.STAGE).canonical_bytes == (
        DockerPreparationMaterializerV1(profile).payload(preparation, EffectKind.STAGE).canonical_bytes
    )


def _preparation(profile, plan, run, **changes):
    values = {
        "provider": profile.provider,
        "scope": profile.scope,
        "project_ref": run.project_ref,
        "run_id": run.run_id,
        "plan_fingerprint": plan.plan_fingerprint,
        "source_digest": plan.basis.source_digest,
        "workload_digest": profile.workload.workload_digest,
        "runtime_digest": profile.runtime.digest,
        "resource_digest": profile.resource_digest,
        "artifact_contract_digest": profile.artifacts.digest,
        "quote_digest": profile.quote_digest,
        "secret_requirements_digest": profile.secret_requirements_digest,
        "execution_binding_digest": _binding(profile).binding_digest,
    }
    values.update(changes)
    return CanonicalPreparationV2.build(**values)


def _binding(profile):
    return ProviderExecutionBindingV1(
        profile.provider, profile.descriptor.descriptor_digest,
        profile.profile_digest, profile.scope, profile.executor_descriptor,
        profile.adapter_descriptor.digest, profile.resource_digest,
        profile.quote_digest, profile.secret_requirements_digest,
    )


@pytest.mark.parametrize("changes", (
    {"provider": ProviderRef("other", "opaque/local-cpu")},
    {"scope": ExecutionScopeV1("other-account", "namespace")},
    {"workload_digest": "a" * 64},
    {"runtime_digest": "b" * 64},
    {"resource_digest": "c" * 64},
    {"artifact_contract_digest": "d" * 64},
    {"quote_digest": "e" * 64},
    {"secret_requirements_digest": "f" * 64},
))
def test_reconstruction_rejects_foreign_profile_binding(profile, plan, run, changes):
    foreign = _preparation(profile, plan, run, **changes)
    materializer = DockerPreparationMaterializerV1(profile)
    with pytest.raises(DockerProviderError) as caught:
        materializer.prepared(foreign)
    assert str(caught.value) == "docker_binding_mismatch"
    with pytest.raises(DockerProviderError) as caught:
        materializer.payload(foreign, EffectKind.SUBMIT)
    assert str(caught.value) == "docker_binding_mismatch"


def test_reconstruction_rejects_wrong_exact_type_and_mutated_sealed_bytes(profile, plan, run):
    materializer = DockerPreparationMaterializerV1(profile)
    with pytest.raises(DockerProviderError) as caught:
        materializer.prepared(object())
    assert str(caught.value) == "docker_binding_mismatch"

    preparation = _preparation(profile, plan, run)
    foreign = _preparation(profile, plan, run, workload_digest="a" * 64)
    object.__setattr__(preparation, "_raw", foreign.canonical_bytes)
    with pytest.raises(DockerProviderError) as caught:
        materializer.prepared(preparation)
    assert str(caught.value) == "docker_binding_mismatch"

    malformed = _preparation(profile, plan, run)
    object.__setattr__(malformed, "_raw", b"{}")
    with pytest.raises(DockerProviderError) as caught:
        materializer.payload(malformed, EffectKind.STAGE)
    assert str(caught.value) == "docker_binding_mismatch"


def test_reconstruction_rejects_profile_with_different_binding_identity(profile, plan, run):
    preparation = _preparation(profile, plan, run)
    foreign_profile = replace(profile, provider=ProviderRef("docker", "other-profile"))
    with pytest.raises(DockerProviderError) as caught:
        DockerPreparationMaterializerV1(foreign_profile).prepared(preparation)
    assert str(caught.value) == "docker_binding_mismatch"


@pytest.mark.parametrize("substitute", (
    lambda p: replace(p, descriptor=replace(p.descriptor, display_name="Other Docker")),
    lambda p: replace(p, executor_descriptor=ExecutorDescriptorV1("docker", "other-executor", "1.0.0")),
    lambda p: replace(p, adapter_descriptor=AdapterDescriptorV1("docker", "other-adapter", "1.0.0")),
    lambda p: replace(p, image=DockerImageV1("other-image", p.image.image_digest)),
    lambda p: replace(p, roots=DockerRootsV1("other-source", "artifact-root")),
    lambda p: replace(p, workload=replace(p.workload, arguments=("python", "/source/other.py"))),
    lambda p: replace(p, runtime=replace(p.runtime, cpu_count=3)),
    lambda p: replace(p, artifacts=DockerArtifactContractV1(("other-result",), 1_048_576, 1_048_576)),
    lambda p: replace(p, scope=ExecutionScopeV1("other-account", "namespace")),
    lambda p: replace(p, resource_digest="a" * 64),
    lambda p: replace(p, quote_digest="b" * 64),
    lambda p: replace(p, secret_requirements_digest="c" * 64),
))
def test_restart_rejects_every_structural_profile_substitution(profile, plan, run, substitute):
    preparation = _preparation(profile, plan, run)
    foreign_profile = substitute(profile)
    assert foreign_profile.provider == profile.provider
    with pytest.raises(DockerProviderError) as caught:
        DockerPreparationMaterializerV1(foreign_profile).prepared(preparation)
    assert str(caught.value) == "docker_binding_mismatch"


def test_binding_resolution_revalidates_internal_profile_snapshot(profile, plan):
    resolver = DockerBindingResolverV1(profile)
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        profile.descriptor.descriptor_digest, profile.profile_digest,
    )
    object.__setattr__(resolver._profile.runtime, "network_mode", "bridge")
    with pytest.raises(DockerProviderError) as caught:
        resolver.resolve(profile.provider, context)
    assert str(caught.value) == "docker_binding_mismatch"


def test_reconstruction_and_payload_revalidate_internal_profile_snapshot(profile, plan, run):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        profile.descriptor.descriptor_digest, profile.profile_digest,
    )
    binding = DockerBindingResolverV1(profile).resolve(profile.provider, context)
    materializer = DockerPreparationMaterializerV1(profile)
    preparation = materializer.prepare(plan, run, binding)
    object.__setattr__(materializer._profile.workload, "arguments", ["mutable"])
    with pytest.raises(DockerProviderError) as caught:
        materializer.prepared(preparation)
    assert str(caught.value) == "docker_binding_mismatch"
    with pytest.raises(DockerProviderError) as caught:
        materializer.payload(preparation, EffectKind.SUBMIT)
    assert str(caught.value) == "docker_binding_mismatch"


def test_binding_mismatch_is_closed(profile, plan):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        "f" * 64, profile.profile_digest,
    )
    with pytest.raises(DockerProviderError) as caught:
        DockerBindingResolverV1(profile).resolve(profile.provider, context)
    assert str(caught.value) == "docker_binding_mismatch"
