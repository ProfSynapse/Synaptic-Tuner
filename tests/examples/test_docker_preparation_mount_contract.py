# ruff: noqa: F811
"""Actual existing source/artifact ports; no daemon or model requests."""

import pytest
from tests.execution.providers.docker_provider_v1.conftest import profile, plan, run  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_source import material  # noqa: F401
from tests.execution.providers.docker_provider_v1.test_local_artifacts import stack  # noqa: F401
from tuner.execution.providers.docker_provider_v1.model import DockerProviderError


def test_artifact_cache_survives_real_source_seal(material):
    port, request, labels, args, _ = material
    cache = args["artifact_root"] / "run" / "cache"
    cache.parent.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    (cache / "synthetic-model").write_bytes(b"fixture-cache-only")
    port.seal_read_only(request)
    port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)
    assert (cache / "synthetic-model").read_bytes() == b"fixture-cache-only"


def test_cache_never_widens_source_inventory(material):
    port, request, labels, args, _ = material
    port.seal_read_only(request)
    (args["source_root"] / "model-cache").mkdir(mode=0o700)
    with pytest.raises(DockerProviderError):
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)


def test_existing_prepared_slot_still_rejects_with_cache(material):
    port, request, _, _, _ = material
    root, relative = port._input_key
    (port._roots[root].path / relative).parent.mkdir(parents=True, mode=0o700)
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)


def test_model_cache_is_not_a_declared_native_role(stack):
    port, request, root, signer, _ = stack
    cache = root / "run" / "cache"
    cache.mkdir(parents=True)
    (cache / "synthetic-model").write_bytes(b"not-an-artifact")
    inventory = port.artifact_inventory(request)
    assert signer.authenticate_inventory(inventory)
    assert [entry.descriptor.role for entry in inventory.content.entries] == ["result"]


@pytest.mark.parametrize(
    "mutation", ["file", "directory", "symlink", "replacement", "mode"]
)
def test_authenticated_project_root_remains_empty_and_owned(material, mutation):
    port, request, labels, args, _ = material
    port.seal_read_only(request)
    root = args["source_root"] / port._project_relative
    if mutation == "file":
        (root / "unexpected").write_bytes(b"reject")
    elif mutation == "directory":
        (root / "unexpected").mkdir(mode=0o700)
    elif mutation == "symlink":
        root.rmdir()
        root.symlink_to(args["artifact_root"], target_is_directory=True)
    elif mutation == "replacement":
        root.rename(root.with_name("old-project"))
        root.mkdir(mode=0o700)
    else:
        root.chmod(0o755)
    with pytest.raises(DockerProviderError):
        port.revalidate_for_submit(labels=labels, source_ref=request.source_ref)


def test_existing_project_directory_is_not_adopted(material):
    port, request, _, args, _ = material
    (args["source_root"] / port._project_relative).mkdir(mode=0o700)
    with pytest.raises(DockerProviderError):
        port.seal_read_only(request)
