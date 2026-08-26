"""Zero-invocation registry for descriptors and factory references."""

from __future__ import annotations

from dataclasses import dataclass

from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef

from .executors import AdapterDescriptorV1, ExecutorDescriptorV1


@dataclass(frozen=True, slots=True)
class ProviderRegistrationV2:
    provider: ProviderDescriptor
    executor: ExecutorDescriptorV1
    adapter: AdapterDescriptorV1
    executor_factory_ref: object
    adapter_factory_ref: object

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderDescriptor):
            raise TypeError("provider must be ProviderDescriptor")
        if self.executor.provider_id != self.provider.provider_id:
            raise ValueError("executor provider mismatch")
        if self.adapter.provider_id != self.provider.provider_id:
            raise ValueError("adapter provider mismatch")
        if self.executor_factory_ref is None or self.adapter_factory_ref is None:
            raise ValueError("factory references are required")


class LazyProviderRegistryV2:
    def __init__(self) -> None:
        self._values: dict[str, ProviderRegistrationV2] = {}

    def register(self, value: ProviderRegistrationV2) -> None:
        if value.provider.provider_id in self._values:
            raise ValueError("provider already registered")
        self._values[value.provider.provider_id] = value

    def list(self) -> tuple[ProviderDescriptor, ...]:
        return tuple(self._values[key].provider for key in sorted(self._values))

    def registration(self, provider: ProviderRef) -> ProviderRegistrationV2:
        try:
            return self._values[provider.provider_id]
        except KeyError as exc:
            raise KeyError("provider not registered") from exc

    def executor_factory(self, provider: ProviderRef) -> object:
        return self.registration(provider).executor_factory_ref

    def adapter_factory(self, provider: ProviderRef) -> object:
        return self.registration(provider).adapter_factory_ref
