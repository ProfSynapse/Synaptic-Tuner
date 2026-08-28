from dataclasses import FrozenInstanceError, fields, replace

import pytest

from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.references import ExecutionScopeV1
from tuner.execution.providers.docker_provider_v1.model import (
    DockerArtifactContractV1, DockerImageV1, DockerProfileV1, DockerRootsV1,
    DockerRuntimeV1, DockerWorkloadV1, validated_profile_snapshot,
)


def test_profile_is_canonical_immutable_and_opaque_profiles_share_one_type(profile):
    assert profile.provider.profile_ref.startswith("opaque/")
    with pytest.raises(FrozenInstanceError):
        profile.resource_digest = "0" * 64
    assert "profile_digest" not in profile.__dataclass_fields__


def test_profile_digest_cannot_be_injected(profile):
    values = {name: getattr(profile, name) for name in profile.__dataclass_fields__}
    values["profile_digest"] = "0" * 64
    with pytest.raises(TypeError):
        DockerProfileV1.build(**values)


def _with_provider_id(profile, provider_id):
    return replace(
        profile,
        provider=ProviderRef(provider_id, profile.provider.profile_ref),
        descriptor=replace(profile.descriptor, provider_id=provider_id),
        executor_descriptor=replace(profile.executor_descriptor, provider_id=provider_id),
        adapter_descriptor=replace(profile.adapter_descriptor, provider_id=provider_id),
    )


@pytest.mark.parametrize("mutate", (
    lambda p: _with_provider_id(p, "other-docker"),
    lambda p: replace(p, provider=ProviderRef("docker", "opaque/other")),
    lambda p: replace(p, descriptor=replace(p.descriptor, display_name="Other Docker")),
    lambda p: replace(p, descriptor=replace(p.descriptor, implementation_version="2.0.0")),
    lambda p: replace(p, descriptor=replace(p.descriptor, capabilities=replace(p.descriptor.capabilities, observe=False))),
    lambda p: replace(p, descriptor=replace(
        p.descriptor, capabilities=replace(p.descriptor.capabilities, logs=False),
    )),
    lambda p: replace(p, descriptor=replace(p.descriptor, capabilities=replace(p.descriptor.capabilities, cancel=False))),
    lambda p: replace(p, descriptor=replace(p.descriptor, capabilities=replace(p.descriptor.capabilities, reconcile=False))),
    lambda p: replace(p, descriptor=replace(p.descriptor, capabilities=replace(p.descriptor.capabilities, artifact_streaming=False))),
    lambda p: replace(p, descriptor=replace(p.descriptor, capabilities=replace(p.descriptor.capabilities, cost_quote=True))),
    lambda p: replace(p, scope=ExecutionScopeV1("other-account", "namespace")),
    lambda p: replace(p, scope=ExecutionScopeV1("account", "other-namespace")),
    lambda p: replace(p, executor_descriptor=ExecutorDescriptorV1("docker", "other-executor", "1.0.0")),
    lambda p: replace(p, executor_descriptor=replace(p.executor_descriptor, implementation_version="2.0.0")),
    lambda p: replace(p, adapter_descriptor=AdapterDescriptorV1("docker", "other-adapter", "1.0.0")),
    lambda p: replace(p, adapter_descriptor=replace(p.adapter_descriptor, implementation_version="2.0.0")),
    lambda p: replace(p, image=DockerImageV1("other-image", p.image.image_digest)),
    lambda p: replace(p, image=DockerImageV1(p.image.image_ref, "sha256:" + "b" * 64)),
    lambda p: replace(p, runtime=replace(p.runtime, cpu_count=3)),
    lambda p: replace(p, runtime=replace(p.runtime, memory_bytes=2_147_483_648)),
    lambda p: replace(p, runtime=replace(p.runtime, timeout_seconds=7200)),
    lambda p: replace(p, workload=replace(
        p.workload, arguments=("python", "/source/other.py"),
    )),
    lambda p: replace(p, workload=replace(p.workload, environment_keys=("TOKEN",))),
    lambda p: replace(p, workload=replace(p.workload, workload_digest="d" * 64)),
    lambda p: replace(p, roots=DockerRootsV1("other-source", "artifact-root")),
    lambda p: replace(p, roots=DockerRootsV1("source-root", "other-artifact")),
    lambda p: replace(p, artifacts=DockerArtifactContractV1(("other-result",), 1_048_576, 1_048_576)),
    lambda p: replace(p, artifacts=replace(p.artifacts, maximum_artifact_bytes=524_288)),
    lambda p: replace(p, artifacts=replace(p.artifacts, maximum_total_bytes=2_097_152)),
    lambda p: replace(p, resource_digest="a" * 64),
    lambda p: replace(p, quote_digest="b" * 64),
    lambda p: replace(p, secret_requirements_digest="c" * 64),
))
def test_profile_digest_covers_every_structural_binding(profile, mutate):
    changed = mutate(profile)
    assert changed.profile_digest != profile.profile_digest
    assert replace(changed).profile_digest == changed.profile_digest


def test_profile_rejects_hostile_capabilities_subclass(profile):
    class HostileCapabilities(ProviderCapabilities):
        pass

    hostile = HostileCapabilities(True, True, True, True, True, False)
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", "docker", "Docker", "1.0.0", hostile,
    )
    with pytest.raises(TypeError):
        replace(profile, descriptor=descriptor)


@pytest.mark.parametrize("field", (
    "provider", "descriptor", "scope", "executor_descriptor",
    "adapter_descriptor", "image", "runtime", "workload", "roots",
    "artifacts",
))
def test_profile_rejects_every_hostile_nested_subclass(profile, field):
    value = getattr(profile, field)
    hostile_type = type(f"Hostile{type(value).__name__}", (type(value),), {})
    hostile = hostile_type(**{
        item.name: getattr(value, item.name) for item in fields(value)
    })
    with pytest.raises(TypeError):
        replace(profile, **{field: hostile})


@pytest.mark.parametrize(("path", "attribute", "hostile"), (
    (("provider",), "provider_id", ""),
    (("descriptor",), "display_name", ""),
    (("descriptor", "capabilities"), "observe", "not-boolean"),
    (("scope",), "account_ref", ""),
    (("executor_descriptor",), "executor_id", ""),
    (("adapter_descriptor",), "adapter_id", ""),
    (("image",), "presence_policy", "pull"),
    (("runtime",), "network_mode", "bridge"),
    (("workload",), "arguments", ["mutable"]),
    (("roots",), "source_read_only", False),
    (("artifacts",), "roles", ("z", "a")),
))
def test_profile_snapshot_revalidates_postconstruction_behavior_leaf_mutation(
        profile, path, attribute, hostile):
    target = profile
    for part in path:
        target = getattr(target, part)
    object.__setattr__(target, attribute, hostile)
    with pytest.raises((TypeError, ValueError)):
        validated_profile_snapshot(profile)


@pytest.mark.parametrize("field", (
    "provider", "descriptor", "scope", "executor_descriptor",
    "adapter_descriptor", "image", "runtime", "workload", "roots",
    "artifacts",
))
def test_profile_snapshot_rejects_postconstruction_same_projection_hostile_wrapper(profile, field):
    value = getattr(profile, field)

    class SameProjectionHostile(type(value)):
        def to_dict(self):
            return value.to_dict()

    hostile = SameProjectionHostile(**{
        item.name: getattr(value, item.name) for item in fields(value)
    })
    object.__setattr__(profile, field, hostile)
    with pytest.raises(TypeError):
        validated_profile_snapshot(profile)


@pytest.mark.parametrize("bad", ("repo:latest", "latest"))
def test_image_rejects_latest_and_requires_present_only(bad):
    with pytest.raises(ValueError):
        DockerImageV1(bad, "sha256:" + "a" * 64)
    with pytest.raises(ValueError):
        DockerImageV1("repo", "sha256:" + "a" * 64, "pull")


def test_cpu_network_and_workload_bounds_are_closed():
    with pytest.raises(ValueError):
        DockerRuntimeV1(1, 1024, 10, "bridge", False)
    with pytest.raises(ValueError):
        DockerRuntimeV1(1, 1024, 10, "none", True)
    with pytest.raises(ValueError):
        DockerWorkloadV1(("x" * 40_000,), (), "1" * 64)
    with pytest.raises(ValueError):
        DockerArtifactContractV1(("z", "a"))
