from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.registry import (
    LazyProviderRegistryV2, ProviderReaderFactoryRequestV1, ResolvedProviderReaderV1,
)
from tuner.execution.providers.modal.coordinator_effects import (
    ModalEffectOutcome, ModalFoundationReconciliationAdapter,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.providers.modal.coordinator_factories import modal_coordinator_registration
from tuner.execution.providers.modal.coordinator_reader import ModalCoordinatorRunReader
from tests.execution.providers.test_modal_coordinator_adapter import composed
from tests.execution.providers.test_modal_coordinator_effects import (
    Authority, Catalog, Transport, executor, stage_case,
)


class NoCalls:
    def __init__(self): self.calls = 0
    def __getattr__(self, name):
        def call(*args, **kwargs): self.calls += 1; raise AssertionError(name)
        return call


def configured():
    command, payload, execution_request, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    effect_executor = executor(binding, transport)
    reconciliation = ModalFoundationReconciliationAdapter(
        profile_ref=binding.profile_ref, account_ref=binding.account_ref,
        namespace_ref=binding.namespace_ref, catalog=Catalog(binding),
        authority=Authority(), transport=transport,
    )
    dependencies = [NoCalls() for _ in range(6)]
    reader = ModalCoordinatorRunReader(
        catalog=dependencies[0], binding_authority=dependencies[1],
        foundation_authenticator=dependencies[2], assessment_authenticator=dependencies[3],
        evidence_authority=dependencies[4], transport=dependencies[5],
        observed_at="2026-09-09T12:00:00Z",
    )
    preparation = composed()[0]
    registration = modal_coordinator_registration(
        preparation, effect_executor, reconciliation, reader,
    )
    registry = LazyProviderRegistryV2(); registry.register(registration)
    _, execution, _ = preparation._snapshot()
    request = ProviderReaderFactoryRequestV1(
        execution.provider, registration.provider.descriptor_digest,
        execution.profile_digest, execution.scope.account_ref, execution.scope.namespace_ref,
    )
    return registry, registration, request, reader, dependencies, execution_request


def test_registration_and_listing_are_zero_invocation_and_keep_false_capabilities():
    registry, registration, request, _, dependencies, _ = configured()
    assert registry.list() == (registration.provider,)
    assert registry.registration(request.provider) is registration
    assert registry.executor_factory(request.provider) is registration.executor_factory_ref
    assert registry.adapter_factory(request.provider) is registration.adapter_factory_ref
    assert registry.reader_factory(request.provider) is registration.reader_factory_ref
    assert all(item.calls == 0 for item in dependencies)
    assert not any(registration.provider.capabilities.to_dict().values())


def test_reader_factory_returns_registry_native_exact_binding():
    registry, _, request, reader, dependencies, _ = configured()
    resolved = registry.resolve_reader(request)
    assert type(resolved) is ResolvedProviderReaderV1 and resolved.reader is reader
    assert resolved.request_digest == request.request_digest
    assert all(item.calls == 0 for item in dependencies)


def test_reader_factory_rejects_nonexact_request_before_reading_its_fields():
    _, registration, _, _, dependencies, _ = configured()
    touched = []

    class ForgedRequest:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError("untrusted request field read")

    with pytest.raises(FoundationError) as caught:
        registration.reader_factory_ref.create(ForgedRequest())
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert touched == []
    assert all(item.calls == 0 for item in dependencies)


@pytest.mark.parametrize("field,value", [
    ("provider_descriptor_digest", "f" * 64), ("profile_digest", "f" * 64),
    ("account_ref", "alien"), ("namespace_ref", "alien"),
])
def test_reader_request_substitution_fails_closed(field, value):
    registry, _, request, _, dependencies, _ = configured()
    with pytest.raises(FoundationError) as caught:
        registry.resolve_reader(replace(request, **{field: value}))
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert all(item.calls == 0 for item in dependencies)


def test_reader_provider_profile_substitution_fails_closed():
    from synaptic_tuner.api.v1.providers import ProviderRef
    registry, _, request, _, dependencies, _ = configured()
    with pytest.raises(FoundationError):
        registry.resolve_reader(replace(request, provider=ProviderRef("modal", "alien")))
    assert all(item.calls == 0 for item in dependencies)


def test_executor_factory_uses_existing_registry_native_minting():
    registry, registration, request, _, dependencies, execution_request = configured()
    resolved = registry.executor_factory(request.provider).resolve(execution_request)
    assert resolved.request_digest == execution_request.digest
    assert resolved.executor is not None
    assert all(item.calls == 0 for item in dependencies)


def test_reconciliation_factory_uses_existing_registry_native_minting():
    from tuner.execution.foundation_v2.executors import ReconciliationResolutionRequestV2
    registry, registration, request, _, dependencies, execution_request = configured()
    resolution = ReconciliationResolutionRequestV2(
        execution_request.command_digest, registration.adapter.digest,
        execution_request.provider_id, execution_request.profile_ref,
        execution_request.account_ref, execution_request.namespace_ref,
    )
    resolved = registry.adapter_factory(request.provider).resolve(resolution)
    assert resolved.request_digest == resolution.digest and resolved.adapter is not None
    assert all(item.calls == 0 for item in dependencies)


def test_registration_rejects_runtime_scope_or_reader_substitution():
    _, registration, _, reader, _, _ = configured()
    preparation = composed()[0]
    executor_value = registration.executor_factory_ref._executor
    reconciliation = registration.adapter_factory_ref._adapter
    executor_value.account_ref = "alien"
    with pytest.raises(FoundationError):
        modal_coordinator_registration(preparation, executor_value, reconciliation, reader)
    with pytest.raises(TypeError):
        modal_coordinator_registration(preparation, executor_value, reconciliation, object())
