from __future__ import annotations

from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.providers import (
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRef,
)
from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.executors import (
    AdapterDescriptorV1,
    ExecutorDescriptorV1,
)
from tuner.execution.foundation_v2.registry import (
    LazyProviderRegistryV2,
    ProviderReaderFactoryRequestV1,
    ProviderRegistrationV2,
    ResolvedProviderReaderV1,
)


DIGEST = "a" * 64


class Reader:
    pass


class Factory:
    def __init__(self, *, mutate=None, failure=None):
        self.calls = 0
        self.mutate = mutate
        self.failure = failure
        self.reader = Reader()

    def create(self, request):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        resolved = ResolvedProviderReaderV1(
            request.request_digest,
            request.provider,
            request.provider_descriptor_digest,
            request.profile_digest,
            request.account_ref,
            request.namespace_ref,
            self.reader,
        )
        return self.mutate(resolved) if self.mutate else resolved


def configured(factory):
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1",
        "configured-provider",
        "Configured Provider",
        "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    registration = ProviderRegistrationV2(
        descriptor,
        ExecutorDescriptorV1("configured-provider", "executor", "1.0.0"),
        AdapterDescriptorV1("configured-provider", "adapter", "1.0.0"),
        object(),
        object(),
        factory,
    )
    registry = LazyProviderRegistryV2()
    registry.register(registration)
    request = ProviderReaderFactoryRequestV1(
        ProviderRef("configured-provider", "profile-a"),
        descriptor.descriptor_digest,
        DIGEST,
        "account-a",
        "namespace-a",
    )
    return registry, descriptor, request


def test_discovery_and_factory_inspection_never_invoke_reader_factory() -> None:
    factory = Factory()
    registry, descriptor, request = configured(factory)
    assert registry.list() == (descriptor,)
    assert registry.registration(request.provider).reader_factory_ref is factory
    assert registry.reader_factory(request.provider) is factory
    assert factory.calls == 0


def test_explicit_reader_resolution_invokes_once_and_binds_every_field() -> None:
    factory = Factory()
    registry, _, request = configured(factory)
    resolved = registry.resolve_reader(request)
    assert resolved.reader is factory.reader
    assert resolved.request_digest == request.request_digest
    assert factory.calls == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: replace(value, request_digest="b" * 64),
        lambda value: replace(value, provider=ProviderRef("foreign", "profile-a")),
        lambda value: replace(value, profile_digest="b" * 64),
        lambda value: replace(value, account_ref="foreign"),
        lambda value: replace(value, namespace_ref="foreign"),
    ],
)
def test_malformed_reader_resolution_closes(mutate) -> None:
    factory = Factory(mutate=mutate)
    registry, _, request = configured(factory)
    with pytest.raises(FoundationError) as caught:
        registry.resolve_reader(request)
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert factory.calls == 1


def test_throwing_reader_factory_closes_without_raw_context() -> None:
    factory = Factory(failure=RuntimeError("credential secret"))
    registry, _, request = configured(factory)
    with pytest.raises(FoundationError) as caught:
        registry.resolve_reader(request)
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_descriptor_substitution_rejects_before_factory_invocation() -> None:
    factory = Factory()
    registry, _, request = configured(factory)
    with pytest.raises(FoundationError) as caught:
        registry.resolve_reader(replace(request, provider_descriptor_digest="b" * 64))
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert factory.calls == 0


def test_unknown_provider_resolution_closes_without_raw_context_or_factory_call() -> None:
    factory = Factory()
    registry, _, request = configured(factory)
    unknown = replace(
        request,
        provider=ProviderRef("secret-unknown-provider", request.provider.profile_ref),
    )

    with pytest.raises(FoundationError) as caught:
        registry.resolve_reader(unknown)

    error = caught.value
    assert error.code is DiagnosticCode.BINDING_MISMATCH
    assert "secret-unknown-provider" not in str(error)
    assert "secret-unknown-provider" not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert factory.calls == 0
