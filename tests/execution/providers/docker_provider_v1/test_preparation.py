from dataclasses import replace
import hashlib
from pathlib import PurePosixPath

import pytest

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training import (
    AcceleratorDeviceRequestV1,
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingPlan as PublicTrainingPlan,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1
from tuner.execution.providers.docker_provider_v1.model import (
    DockerArtifactContractV1,
    DockerImageV1,
    DockerProviderError,
    DockerRootsV1,
    DockerRuntimeV1,
    DockerWorkloadV1,
    PreparedDockerPlanV1,
)
from tuner.execution.providers.docker_provider_v1.preparation import (
    DockerBindingResolverV1,
    DockerPreparationMaterializerV1,
    DockerTrainingPreparationBridgeV1,
)
from tuner.runtime.dispatch import (
    CanonicalWorkloadFileLocationV1,
    WorkerControlLocationV1,
    build_source_worker_invocation,
    materialize_worker_bundle,
)
from tuner.training.methods.sft import compile_sft_workload
from tests.training.test_sft_compilation import _config, _execution_source


PUBLIC_ARTIFACT_ROLES = (
    "final_model",
    "tokenizer",
    "training_lineage",
    "training_metrics",
    "workload_record",
)


def _document_digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _public_bridge_values(profile):
    no_secrets = _document_digest(
        {
            "schema_version": "synaptic-docker-secret-requirements/v1",
            "secrets": [],
        }
    )
    zero_quote = _document_digest(
        {
            "schema_version": "synaptic-docker-local-quote/v1",
            "currency": "USD",
            "amount": "0",
        }
    )
    source = replace(_execution_source(), secret_requirements_digest=no_secrets)
    config = _config()
    compiled = compile_sft_workload(
        resolved_config=config,
        execution_source=source,
    )
    resources = ResourceSpec("cpu", 1, profile.runtime.timeout_seconds)
    plan = PublicTrainingPlan(
        execution_source=source,
        execution_context=CanonicalDocument.from_mapping(
            {"schema_version": "test-docker-execution-context/v1"}
        ),
        resolved_config=config,
        workload=CanonicalDocument(compiled.canonical_bytes.decode("utf-8")),
        runtime=RuntimeSpec(
            f"{profile.image.image_ref}@{profile.image.image_digest}",
            "9" * 64,
            source.python_version,
        ),
        resources=resources,
        artifact_policy=ArtifactPolicy(required_kinds=PUBLIC_ARTIFACT_ROLES),
    )
    bundle = materialize_worker_bundle(
        build_source_worker_invocation(
            plan,
            WorkerControlLocationV1(PurePosixPath("/source/control")),
            CanonicalWorkloadFileLocationV1(PurePosixPath("/source/control")),
        )
    )
    resource_digest = _document_digest(
        {
            "accelerator": resources.accelerator,
            "accelerator_count": resources.accelerator_count,
            "timeout_seconds": resources.timeout_seconds,
        }
    )
    aligned = replace(
        profile,
        workload=DockerWorkloadV1(
            bundle.dispatch.argv,
            tuple(sorted(dict(bundle.dispatch.environment))),
            bundle.workload_sha256,
        ),
        artifacts=DockerArtifactContractV1(
            PUBLIC_ARTIFACT_ROLES,
            profile.artifacts.maximum_artifact_bytes,
            profile.artifacts.maximum_total_bytes,
        ),
        resource_digest=resource_digest,
        quote_digest=zero_quote,
        secret_requirements_digest=no_secrets,
    )
    return plan, aligned, TrainingRunRef(source.run_id, "project"), "8" * 64


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


def test_public_training_bridge_preserves_the_exact_plan_and_recomputes_preparation(
    profile,
):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    first = DockerTrainingPreparationBridgeV1(aligned)
    preparation = first.prepare(public_plan, public_run, source_digest)
    second = DockerTrainingPreparationBridgeV1(aligned)

    assert preparation.plan_fingerprint == public_plan.fingerprint
    assert preparation.workload_digest == aligned.workload.workload_digest
    assert preparation.runtime_digest == aligned.runtime.digest
    assert preparation.resource_digest == aligned.resource_digest
    assert preparation.artifact_contract_digest == aligned.artifacts.digest
    assert second.prepare(public_plan, public_run, source_digest) == preparation
    assert "--canonical-workload-file" in aligned.workload.arguments
    assert "--canonical-workload-stdin" not in aligned.workload.arguments
    assert "/source/control/workload.json" in aligned.workload.arguments
    assert all(
        "/workspace/control" not in value for value in aligned.workload.arguments
    )

    prepared = second.prepared(
        preparation=preparation,
        plan=public_plan,
        run=public_run,
        source_digest=source_digest,
    )
    assert type(prepared) is PreparedDockerPlanV1
    assert prepared.plan_fingerprint == public_plan.fingerprint
    assert prepared.source_digest == source_digest
    assert prepared.profile == aligned


def test_public_training_bridge_rejects_private_plan_and_wrong_exact_types(
    profile,
    plan,
):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    bridge = DockerTrainingPreparationBridgeV1(aligned)

    for foreign in (plan, object()):
        with pytest.raises(DockerProviderError) as caught:
            bridge.prepare(foreign, public_run, source_digest)
        assert str(caught.value) == "docker_invalid_plan"

    class ForeignPublicPlan(PublicTrainingPlan):
        pass

    foreign = object.__new__(ForeignPublicPlan)
    with pytest.raises(DockerProviderError) as caught:
        bridge.prepare(foreign, public_run, source_digest)
    assert str(caught.value) == "docker_invalid_plan"

    with pytest.raises(DockerProviderError) as caught:
        bridge.prepare(public_plan, object(), source_digest)
    assert str(caught.value) == "docker_invalid_plan"


@pytest.mark.parametrize(
    "substitute",
    (
        lambda p: replace(p, image=DockerImageV1("other-image", p.image.image_digest)),
        lambda p: replace(
            p,
            runtime=DockerRuntimeV1(
                p.runtime.cpu_count,
                p.runtime.memory_bytes,
                p.runtime.timeout_seconds + 1,
                p.runtime.accelerator_devices,
            ),
        ),
        lambda p: replace(
            p,
            runtime=DockerRuntimeV1(
                p.runtime.cpu_count,
                p.runtime.memory_bytes,
                p.runtime.timeout_seconds,
                AcceleratorDeviceRequestV1("nvidia", (0,), ("gpu",)),
            ),
        ),
        lambda p: replace(
            p,
            workload=replace(p.workload, workload_digest="1" * 64),
        ),
        lambda p: replace(
            p,
            workload=replace(p.workload, arguments=p.workload.arguments[:-1]),
        ),
        lambda p: replace(
            p,
            workload=replace(
                p.workload, environment_keys=p.workload.environment_keys[:-1]
            ),
        ),
        lambda p: replace(
            p,
            artifacts=DockerArtifactContractV1(
                ("final_model",),
                p.artifacts.maximum_artifact_bytes,
                p.artifacts.maximum_total_bytes,
            ),
        ),
        lambda p: replace(p, resource_digest="2" * 64),
        lambda p: replace(p, quote_digest="3" * 64),
        lambda p: replace(p, secret_requirements_digest="4" * 64),
        lambda p: replace(
            p,
            descriptor=replace(
                p.descriptor,
                capabilities=ProviderCapabilities(True, True, True, True, True, True),
            ),
        ),
    ),
)
def test_public_training_bridge_rejects_each_profile_projection_mismatch(
    profile,
    substitute,
):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    with pytest.raises(DockerProviderError) as caught:
        DockerTrainingPreparationBridgeV1(substitute(aligned)).prepare(
            public_plan, public_run, source_digest
        )
    assert str(caught.value) == "docker_invalid_plan"


def test_public_training_bridge_rejects_network_tamper(profile):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    bridge = DockerTrainingPreparationBridgeV1(aligned)
    object.__setattr__(bridge._profile.runtime, "network_mode", "bridge")
    with pytest.raises(DockerProviderError) as caught:
        bridge.prepare(public_plan, public_run, source_digest)
    assert str(caught.value) == "docker_invalid_plan"


def test_public_training_bridge_accepts_opaque_roots_and_contextually_binds_them(
    profile,
):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    original = DockerTrainingPreparationBridgeV1(aligned).prepare(
        public_plan, public_run, source_digest
    )
    alternate_profile = replace(
        aligned,
        roots=DockerRootsV1("alternate-source-ref", "alternate-artifact-ref"),
    )
    alternate_bridge = DockerTrainingPreparationBridgeV1(alternate_profile)
    alternate = alternate_bridge.prepare(public_plan, public_run, source_digest)

    assert original.execution_binding_digest != alternate.execution_binding_digest
    assert original.preparation_digest != alternate.preparation_digest
    with pytest.raises(DockerProviderError) as caught:
        alternate_bridge.prepared(
            preparation=original,
            plan=public_plan,
            run=public_run,
            source_digest=source_digest,
        )
    assert str(caught.value) == "docker_binding_mismatch"


def test_public_training_bridge_rejects_stdin_and_workspace_profile_arguments(
    profile,
):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    byte_bundle = materialize_worker_bundle(
        build_source_worker_invocation(
            public_plan,
            WorkerControlLocationV1(PurePosixPath("/source/control")),
        )
    )
    byte_profile = replace(
        aligned,
        workload=replace(aligned.workload, arguments=byte_bundle.dispatch.argv),
    )
    workspace_profile = replace(
        aligned,
        workload=replace(
            aligned.workload,
            arguments=tuple(
                value.replace("/source/control", "/workspace/control")
                for value in aligned.workload.arguments
            ),
        ),
    )

    for foreign in (byte_profile, workspace_profile):
        with pytest.raises(DockerProviderError) as caught:
            DockerTrainingPreparationBridgeV1(foreign).prepare(
                public_plan, public_run, source_digest
            )
        assert str(caught.value) == "docker_invalid_plan"


def test_public_training_bridge_rejects_plan_run_source_and_preparation_swaps(profile):
    public_plan, aligned, public_run, source_digest = _public_bridge_values(profile)
    bridge = DockerTrainingPreparationBridgeV1(aligned)
    preparation = bridge.prepare(public_plan, public_run, source_digest)
    foreign_plan = replace(
        public_plan,
        runtime=replace(public_plan.runtime, dependency_lock_digest="a" * 64),
    )
    foreign_run = TrainingRunRef("other-run", public_run.project_ref)

    for changes in (
        {"plan": foreign_plan},
        {"run": foreign_run},
        {"source_digest": "b" * 64},
    ):
        values = {
            "preparation": preparation,
            "plan": public_plan,
            "run": public_run,
            "source_digest": source_digest,
        }
        values.update(changes)
        with pytest.raises(DockerProviderError) as caught:
            bridge.prepared(**values)
        assert str(caught.value) == "docker_binding_mismatch"

    fabricated = CanonicalPreparationV2.build(
        provider=preparation.provider,
        scope=preparation.scope,
        project_ref=preparation.project_ref,
        run_id=preparation.run_id,
        plan_fingerprint="c" * 64,
        source_digest=preparation.source_digest,
        workload_digest=preparation.workload_digest,
        runtime_digest=preparation.runtime_digest,
        resource_digest=preparation.resource_digest,
        artifact_contract_digest=preparation.artifact_contract_digest,
        quote_digest=preparation.quote_digest,
        secret_requirements_digest=preparation.secret_requirements_digest,
        execution_binding_digest=preparation.execution_binding_digest,
    )
    with pytest.raises(DockerProviderError) as caught:
        bridge.prepared(
            preparation=fabricated,
            plan=public_plan,
            run=public_run,
            source_digest=source_digest,
        )
    assert str(caught.value) == "docker_binding_mismatch"
