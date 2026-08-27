"""Zero-invocation registry for descriptors and factory references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from synaptic_tuner.api.v1.providers import ProviderDescriptor, ProviderRef

from .canonical import DiagnosticCode, FoundationError, canonical_bytes, digest_text, domain_digest, safe_ref
from .executors import AdapterDescriptorV1, ExecutorDescriptorV1


@dataclass(frozen=True, slots=True)
class ProviderReaderFactoryRequestV1:
    provider: ProviderRef
    provider_descriptor_digest: str
    profile_digest: str
    account_ref: str
    namespace_ref: str

    def __post_init__(self) -> None:
        if type(self.provider) is not ProviderRef:
            raise TypeError("provider must be exact ProviderRef")
        digest_text(self.provider_descriptor_digest, "provider_descriptor_digest")
        digest_text(self.profile_digest, "profile_digest")
        safe_ref(self.account_ref, "account_ref")
        safe_ref(self.namespace_ref, "namespace_ref")

    @property
    def request_digest(self) -> str:
        return domain_digest(
            "synaptic-provider-reader-factory-request/v1",
            canonical_bytes(
                {
                    "schema_version": "synaptic-provider-reader-factory-request/v1",
                    "provider": self.provider.to_dict(),
                    "provider_descriptor_digest": self.provider_descriptor_digest,
                    "profile_digest": self.profile_digest,
                    "account_ref": self.account_ref,
                    "namespace_ref": self.namespace_ref,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedProviderReaderV1:
    request_digest: str
    provider: ProviderRef
    provider_descriptor_digest: str
    profile_digest: str
    account_ref: str
    namespace_ref: str
    reader: object

    def __post_init__(self) -> None:
        digest_text(self.request_digest, "request_digest")
        if type(self.provider) is not ProviderRef:
            raise TypeError("provider must be exact ProviderRef")
        digest_text(self.provider_descriptor_digest, "provider_descriptor_digest")
        digest_text(self.profile_digest, "profile_digest")
        safe_ref(self.account_ref, "account_ref")
        safe_ref(self.namespace_ref, "namespace_ref")
        if self.reader is None:
            raise ValueError("reader is required")


class ProviderReaderFactoryV1(Protocol):
    def create(self, request: ProviderReaderFactoryRequestV1) -> ResolvedProviderReaderV1: ...


@dataclass(frozen=True, slots=True)
class ProviderRegistrationV2:
    provider: ProviderDescriptor
    executor: ExecutorDescriptorV1
    adapter: AdapterDescriptorV1
    executor_factory_ref: object
    adapter_factory_ref: object
    reader_factory_ref: ProviderReaderFactoryV1

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderDescriptor):
            raise TypeError("provider must be ProviderDescriptor")
        if self.executor.provider_id != self.provider.provider_id:
            raise ValueError("executor provider mismatch")
        if self.adapter.provider_id != self.provider.provider_id:
            raise ValueError("adapter provider mismatch")
        if (
            self.executor_factory_ref is None
            or self.adapter_factory_ref is None
            or self.reader_factory_ref is None
            or not callable(getattr(self.reader_factory_ref, "create", None))
        ):
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

    def reader_factory(self, provider: ProviderRef) -> ProviderReaderFactoryV1:
        return self.registration(provider).reader_factory_ref

    def resolve_reader(
        self, request: ProviderReaderFactoryRequestV1
    ) -> ResolvedProviderReaderV1:
        if type(request) is not ProviderReaderFactoryRequestV1:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        registration = None
        try:
            registration = self._values[request.provider.provider_id]
        except KeyError:
            pass
        if registration is None:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        if registration.provider.descriptor_digest != request.provider_descriptor_digest:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        failure: DiagnosticCode | None = None
        resolved = None
        try:
            resolved = registration.reader_factory_ref.create(request)
        except FoundationError as error:
            failure = error.code
        except Exception:
            failure = DiagnosticCode.BINDING_MISMATCH
        if failure is not None:
            raise FoundationError(failure)
        if type(resolved) is not ResolvedProviderReaderV1:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        actual = (
            resolved.request_digest,
            resolved.provider,
            resolved.provider_descriptor_digest,
            resolved.profile_digest,
            resolved.account_ref,
            resolved.namespace_ref,
        )
        expected = (
            request.request_digest,
            request.provider,
            request.provider_descriptor_digest,
            request.profile_digest,
            request.account_ref,
            request.namespace_ref,
        )
        if actual != expected:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return resolved


__all__ = [
    "LazyProviderRegistryV2",
    "ProviderReaderFactoryRequestV1",
    "ProviderReaderFactoryV1",
    "ProviderRegistrationV2",
    "ResolvedProviderReaderV1",
]
