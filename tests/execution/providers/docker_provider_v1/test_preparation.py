import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.providers.docker_provider_v1.model import DockerProviderError, PreparedDockerPlanV1
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
    assert type(materializer.prepared(preparation.preparation_digest)) is PreparedDockerPlanV1
    assert materializer.payload(preparation, EffectKind.SUBMIT).payload_kind == "submit-payload/v2"


def test_binding_mismatch_is_closed(profile, plan):
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", profile.provider, plan.basis.basis_digest,
        "f" * 64, profile.profile_digest,
    )
    with pytest.raises(DockerProviderError) as caught:
        DockerBindingResolverV1(profile).resolve(profile.provider, context)
    assert str(caught.value) == "docker_binding_mismatch"
