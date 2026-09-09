"""Internal lazy-registry factories for Foundation-native Modal components."""
from __future__ import annotations

from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.registry import (
    ProviderReaderFactoryRequestV1, ProviderRegistrationV2, ResolvedProviderReaderV1,
)

from .coordinator_adapter import ModalPreparationAdapter
from .coordinator_effects import (
    ModalFoundationEffectExecutor, ModalFoundationExecutorResolver,
    ModalFoundationReconciliationAdapter, ModalFoundationReconciliationResolver,
)
from .coordinator_reader import ModalCoordinatorRunReader


class ModalCoordinatorReaderFactory:
    __slots__ = ("_expected", "_reader")

    def __init__(
        self, *, provider, provider_descriptor_digest: str, profile_digest: str,
        account_ref: str, namespace_ref: str, reader: ModalCoordinatorRunReader,
    ) -> None:
        if type(reader) is not ModalCoordinatorRunReader:
            raise TypeError("exact Modal coordinator reader is required")
        expected = ProviderReaderFactoryRequestV1(
            provider, provider_descriptor_digest, profile_digest, account_ref, namespace_ref,
        )
        self._expected = expected
        self._reader = reader

    def create(self, request: ProviderReaderFactoryRequestV1) -> ResolvedProviderReaderV1:
        try:
            if type(request) is not ProviderReaderFactoryRequestV1:
                raise ValueError
            expected = self._expected
            actual = (
                request.provider, request.provider_descriptor_digest, request.profile_digest,
                request.account_ref, request.namespace_ref,
            )
            bound = (
                expected.provider, expected.provider_descriptor_digest, expected.profile_digest,
                expected.account_ref, expected.namespace_ref,
            )
            if actual != bound:
                raise ValueError
            return ResolvedProviderReaderV1(
                request.request_digest, request.provider,
                request.provider_descriptor_digest, request.profile_digest,
                request.account_ref, request.namespace_ref, self._reader,
            )
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None


def modal_coordinator_registration(
    preparation: ModalPreparationAdapter,
    executor: ModalFoundationEffectExecutor,
    reconciliation: ModalFoundationReconciliationAdapter,
    reader: ModalCoordinatorRunReader,
) -> ProviderRegistrationV2:
    if type(preparation) is not ModalPreparationAdapter:
        raise TypeError("exact Modal preparation adapter is required")
    if type(executor) is not ModalFoundationEffectExecutor:
        raise TypeError("exact Modal Foundation executor is required")
    if type(reconciliation) is not ModalFoundationReconciliationAdapter:
        raise TypeError("exact Modal Foundation reconciliation adapter is required")
    if type(reader) is not ModalCoordinatorRunReader:
        raise TypeError("exact Modal coordinator reader is required")
    context, execution, _ = preparation._snapshot()
    provider = context.provider
    descriptor = preparation.describe(provider)
    expected_runtime = (
        provider.provider_id, provider.profile_ref,
        execution.scope.account_ref, execution.scope.namespace_ref,
    )
    if (
        (executor.provider_id, executor.profile_ref, executor.account_ref, executor.namespace_ref)
        != expected_runtime
        or (reconciliation.provider_id, reconciliation.profile_ref,
            reconciliation.account_ref, reconciliation.namespace_ref) != expected_runtime
        or executor.descriptor != execution.executor_descriptor
        or reconciliation.descriptor.digest != execution.reconciliation_adapter_digest
        or descriptor.descriptor_digest != execution.provider_descriptor_digest
    ):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    reader_factory = ModalCoordinatorReaderFactory(
        provider=provider, provider_descriptor_digest=descriptor.descriptor_digest,
        profile_digest=execution.profile_digest,
        account_ref=execution.scope.account_ref,
        namespace_ref=execution.scope.namespace_ref, reader=reader,
    )
    return ProviderRegistrationV2(
        descriptor, executor.descriptor, reconciliation.descriptor,
        ModalFoundationExecutorResolver(executor),
        ModalFoundationReconciliationResolver(reconciliation),
        reader_factory,
    )


__all__: list[str] = []
