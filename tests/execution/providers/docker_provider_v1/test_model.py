from dataclasses import FrozenInstanceError

import pytest

from tuner.execution.providers.docker_provider_v1.model import (
    DockerArtifactContractV1, DockerImageV1, DockerRuntimeV1, DockerWorkloadV1,
)


def test_profile_is_canonical_immutable_and_opaque_profiles_share_one_type(profile):
    assert profile.provider.profile_ref.startswith("opaque/")
    with pytest.raises(FrozenInstanceError):
        profile.profile_digest = "0" * 64


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
