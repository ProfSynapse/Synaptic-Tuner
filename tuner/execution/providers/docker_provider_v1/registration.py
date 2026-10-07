"""Side-effect-free Docker provider discovery and registration."""

from __future__ import annotations

from ...foundation_v2.registry import LazyProviderRegistryV2, ProviderRegistrationV2
from .model import DockerProfileV1


def docker_registration_v1(
    profile: DockerProfileV1, *, executor_factory_ref: object,
    adapter_factory_ref: object, reader_factory_ref: object,
) -> ProviderRegistrationV2:
    """Build metadata only; none of the supplied factories is invoked."""
    if type(profile) is not DockerProfileV1:
        raise TypeError("exact Docker profile required")
    return ProviderRegistrationV2(
        profile.descriptor, profile.executor_descriptor, profile.adapter_descriptor,
        executor_factory_ref, adapter_factory_ref, reader_factory_ref,
    )


def register_docker_v1(registry: LazyProviderRegistryV2, registration: ProviderRegistrationV2) -> None:
    if type(registry) is not LazyProviderRegistryV2 or type(registration) is not ProviderRegistrationV2:
        raise TypeError("exact registry values required")
    registry.register(registration)


__all__: list[str] = []
