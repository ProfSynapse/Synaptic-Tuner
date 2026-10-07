from tuner.execution.foundation_v2.registry import LazyProviderRegistryV2
from tuner.execution.providers.docker_provider_v1.registration import docker_registration_v1, register_docker_v1


class CounterFactory:
    def __init__(self): self.calls = 0
    def __call__(self, *args): self.calls += 1; raise AssertionError


class ReaderFactory:
    def __init__(self): self.calls = 0
    def create(self, request): self.calls += 1; raise AssertionError


def test_discovery_registers_metadata_without_factory_or_control_invocation(profile):
    executor, adapter, reader = CounterFactory(), CounterFactory(), ReaderFactory()
    registration = docker_registration_v1(
        profile, executor_factory_ref=executor, adapter_factory_ref=adapter,
        reader_factory_ref=reader,
    )
    registry = LazyProviderRegistryV2()
    register_docker_v1(registry, registration)
    assert registry.list() == (profile.descriptor,)
    assert (executor.calls, adapter.calls, reader.calls) == (0, 0, 0)
