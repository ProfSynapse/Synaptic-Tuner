"""Foundation effect adapters for separately authorized Modal chat commands."""

from __future__ import annotations

from inspect import getattr_static
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    DiagnosticCode,
    FoundationError,
    digest_text,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1,
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)
from tuner.execution.foundation_v2.executors import (
    AdapterDescriptorV1,
    ExecutionResolutionRequestV2,
    ExecutorDescriptorV1,
    ReconciliationResolutionRequestV2,
    mint_resolved_adapter,
    mint_resolved_executor,
)
from tuner.execution.foundation_v2.observations import (
    ObservationDisposition,
    ProviderObservationV1,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.foundation_v2.references import (
    ProviderStageRefV1,
    ScopedProviderRunRefV1,
)

from .coordinator_effects import ModalEffectOutcome
from .inference_commands import ModalInferenceCommandBinding
from .inference_retention import load_modal_chat_command

_KINDS = ("stage", "submit")
_PAYLOADS = ("stage-payload/v2", "submit-payload/v2")


class _ChatCatalog(Protocol):
    def resolve(self, command_digest: str) -> object | None: ...


class _BindingAuthority(Protocol):
    def authenticate(self, binding: ModalInferenceCommandBinding) -> bool: ...


class _ChatTransport(Protocol):
    def execute_once(
        self, binding: ModalInferenceCommandBinding, command: object
    ) -> ModalEffectOutcome: ...

    def lookup_once(
        self, binding: ModalInferenceCommandBinding, command: object
    ) -> ModalEffectOutcome: ...


def _method(value: object, name: str) -> object:
    member = getattr_static(type(value), name, None)
    if (
        member is None
        or not callable(member)
        or getattr_static(value, name, None) is not member
    ):
        raise TypeError("chat collaborator method is invalid")
    return member


def _binding_snapshot(binding: ModalInferenceCommandBinding) -> tuple[bytes, bytes]:
    if type(binding) is not ModalInferenceCommandBinding:
        raise TypeError("exact chat command binding required")
    return binding.command_bytes, binding.preparation_snapshot


def _request_snapshot(request: ExecutionResolutionRequestV2) -> tuple[object, ...]:
    if type(request) is not ExecutionResolutionRequestV2:
        raise TypeError("exact execution resolution request required")
    values = tuple(getattr(request, name) for name in request.__dataclass_fields__)
    for name, value in zip(request.__dataclass_fields__, values):
        if name in {"command_digest", "descriptor_digest", "input_digest"}:
            digest_text(value, name)
        else:
            safe_ref(value, name)
    return values


def _target_snapshot(target: ReconciliationTargetV1) -> tuple[object, ...]:
    if type(target) is not ReconciliationTargetV1:
        raise TypeError("exact reconciliation target required")
    values = tuple(getattr(target, name) for name in target.__dataclass_fields__)
    if type(target.command_bytes) is not bytes:
        raise TypeError("exact reconciliation command bytes required")
    for name in ("command_digest", "resolution_digest"):
        digest_text(getattr(target, name), name)
    for name in ("effect_id", "owner_ref"):
        safe_ref(getattr(target, name), name)
    for name in ("generation", "ownership_epoch", "claimed_at_epoch"):
        value = getattr(target, name)
        if type(value) is not int or value < 1:
            raise ValueError("invalid reconciliation epoch")
    return values


def _reconciliation_request_snapshot(
    request: ReconciliationResolutionRequestV2,
) -> tuple[object, ...]:
    if type(request) is not ReconciliationResolutionRequestV2:
        raise TypeError("exact reconciliation resolution request required")
    values = tuple(getattr(request, name) for name in request.__dataclass_fields__)
    for name, value in zip(request.__dataclass_fields__, values):
        if name in {"command_digest", "adapter_digest"}:
            digest_text(value, name)
        else:
            safe_ref(value, name)
    return values


def _adapter_snapshot(value: object) -> tuple[object, ...]:
    transport_method = (
        "execute_once" if hasattr(value, "effect_kinds") else "lookup_once"
    )
    return (
        value.provider_id,
        value.profile_ref,
        value.account_ref,
        value.namespace_ref,
        value.descriptor.to_dict(),
        getattr(value, "effect_kinds", None),
        getattr(value, "payload_schemas", None),
        getattr(value, "capabilities", None),
        id(value._catalog),
        id(value._authority),
        id(value._transport),
        _method(value._catalog, "resolve"),
        _method(value._authority, "authenticate"),
        _method(value._transport, transport_method),
    )


def _load(catalog, authority, digest: str):
    retained = load_modal_chat_command(digest, catalog=catalog, authority=authority)
    binding = ModalInferenceCommandBinding(*_binding_snapshot(retained))
    command = parse_exact_command(binding.command_bytes)
    if (
        type(command) not in (StageCommandV2, SubmitCommandV2)
        or command.digest != digest
    ):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return binding, command


def _validate_scope(adapter: object, command: object) -> None:
    preparation = command.preparation
    if (
        adapter.provider_id,
        adapter.profile_ref,
        adapter.account_ref,
        adapter.namespace_ref,
    ) != (
        preparation.provider.provider_id,
        preparation.provider.profile_ref,
        preparation.scope.account_ref,
        preparation.scope.namespace_ref,
    ):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _validate_request(command: object, request: ExecutionResolutionRequestV2) -> None:
    preparation = command.preparation
    actual = (
        command.digest,
        command.executor.digest,
        preparation.provider.provider_id,
        preparation.provider.profile_ref,
        preparation.scope.account_ref,
        preparation.scope.namespace_ref,
        command.operation.effect.kind.value,
        command.payload.payload_kind,
        command.payload.input_digest,
    )
    if actual != _request_snapshot(request):
        raise FoundationError(DiagnosticCode.BINDING_MISMATCH)


def _outcome(command: object, resolution_digest: str, epoch: int, value: object):
    if type(value) is not ModalEffectOutcome:
        raise FoundationError(DiagnosticCode.EVIDENCE_INVALID)
    try:
        value = ModalEffectOutcome(
            value.disposition, value.provider_ref, value.finality_proof
        )
    except Exception:
        raise FoundationError(DiagnosticCode.EVIDENCE_INVALID) from None
    values: dict[str, object] = {}
    preparation = command.preparation
    if value.disposition is ObservationDisposition.FOUND:
        if type(command) is StageCommandV2:
            values["stage_ref"] = ProviderStageRefV1(
                preparation.provider.provider_id,
                preparation.provider.profile_ref,
                preparation.scope.account_ref,
                preparation.scope.namespace_ref,
                value.provider_ref,
            )
        elif type(command) is SubmitCommandV2:
            values["provider_run"] = ScopedProviderRunRefV1(
                preparation.provider.provider_id,
                preparation.provider.profile_ref,
                preparation.scope.account_ref,
                preparation.scope.namespace_ref,
                value.provider_ref,
            )
        else:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
    return ProviderObservationV1(
        command.operation.effect.effect_id,
        command.digest,
        command.executor.digest,
        value.disposition,
        resolution_digest,
        epoch,
        finality_proof=value.finality_proof,
        **values,
    )


class ModalChatEffectExecutor:
    effect_kinds = _KINDS
    payload_schemas = _PAYLOADS

    def __init__(
        self,
        *,
        profile_ref: str,
        account_ref: str,
        namespace_ref: str,
        implementation_version: str,
        catalog: _ChatCatalog,
        authority: _BindingAuthority,
        transport: _ChatTransport,
    ) -> None:
        for name, value in (
            ("profile_ref", profile_ref),
            ("account_ref", account_ref),
            ("namespace_ref", namespace_ref),
            ("implementation_version", implementation_version),
        ):
            safe_ref(value, name)
        _method(catalog, "resolve")
        _method(authority, "authenticate")
        _method(transport, "execute_once")
        self.descriptor = ExecutorDescriptorV1(
            "modal", "modal-chat-executor", implementation_version
        )
        self.provider_id = "modal"
        self.profile_ref = profile_ref
        self.account_ref = account_ref
        self.namespace_ref = namespace_ref
        self._catalog = catalog
        self._authority = authority
        self._transport = transport
        self._initial_snapshot = _adapter_snapshot(self)

    def execute_once(self, payload: object, request: ExecutionResolutionRequestV2):
        transport_started = False
        transport_completed = False
        try:
            request_snapshot = _request_snapshot(request)
            adapter_snapshot = _adapter_snapshot(self)
            if adapter_snapshot != self._initial_snapshot:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            if type(payload) is not CanonicalProviderPayloadV1:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            payload_snapshot = payload.canonical_bytes
            owned_payload = CanonicalProviderPayloadV1.parse(payload_snapshot)
            if (
                request.descriptor_digest != self.descriptor.digest
                or request.provider_id != self.provider_id
                or request.profile_ref != self.profile_ref
                or request.account_ref != self.account_ref
                or request.namespace_ref != self.namespace_ref
                or request.effect_kind not in self.effect_kinds
                or request.payload_schema != owned_payload.payload_kind
                or request.input_digest != owned_payload.input_digest
                or owned_payload.provider_id != self.provider_id
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            binding, command = _load(
                self._catalog, self._authority, request.command_digest
            )
            binding_snapshot = _binding_snapshot(binding)
            if (
                _request_snapshot(request) != request_snapshot
                or payload.canonical_bytes != payload_snapshot
                or _adapter_snapshot(self) != adapter_snapshot
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            if type(payload) is not type(owned_payload) or payload != owned_payload:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            _validate_scope(self, command)
            if command.executor != self.descriptor:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            _validate_request(command, request)
            transport_binding = ModalInferenceCommandBinding(*binding_snapshot)
            transport_command = parse_exact_command(command.canonical_bytes)
            transport_binding_snapshot = _binding_snapshot(transport_binding)
            transport_command_bytes = transport_command.canonical_bytes
            transport_method = _method(self._transport, "execute_once")
            transport_started = True
            result = transport_method(
                self._transport,
                transport_binding,
                transport_command,
            )
            transport_completed = True
            if (
                _request_snapshot(request) != request_snapshot
                or payload.canonical_bytes != payload_snapshot
                or _binding_snapshot(binding) != binding_snapshot
                or _adapter_snapshot(self) != adapter_snapshot
                or _binding_snapshot(transport_binding) != transport_binding_snapshot
                or transport_command.canonical_bytes != transport_command_bytes
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            return _outcome(command, request.digest, 1, result)
        except (KeyboardInterrupt, SystemExit, FoundationError):
            raise
        except Exception:
            if transport_started and not transport_completed:
                raise
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None


class ModalChatReconciliationAdapter:
    capabilities = ("lookup",)

    def __init__(
        self,
        *,
        profile_ref: str,
        account_ref: str,
        namespace_ref: str,
        implementation_version: str,
        catalog: _ChatCatalog,
        authority: _BindingAuthority,
        transport: _ChatTransport,
    ) -> None:
        for name, value in (
            ("profile_ref", profile_ref),
            ("account_ref", account_ref),
            ("namespace_ref", namespace_ref),
            ("implementation_version", implementation_version),
        ):
            safe_ref(value, name)
        _method(catalog, "resolve")
        _method(authority, "authenticate")
        _method(transport, "lookup_once")
        self.descriptor = AdapterDescriptorV1(
            "modal", "modal-chat-lookup", implementation_version
        )
        self.provider_id = "modal"
        self.profile_ref = profile_ref
        self.account_ref = account_ref
        self.namespace_ref = namespace_ref
        self._catalog = catalog
        self._authority = authority
        self._transport = transport
        self._initial_snapshot = _adapter_snapshot(self)

    def lookup(self, target: ReconciliationTargetV1, preparation: object):
        transport_started = False
        transport_completed = False
        try:
            target_snapshot = _target_snapshot(target)
            adapter_snapshot = _adapter_snapshot(self)
            if adapter_snapshot != self._initial_snapshot:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            target_command = parse_exact_command(target.command_bytes)
            if type(preparation) is not type(target_command.preparation):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            if (
                type(target_command) not in (StageCommandV2, SubmitCommandV2)
                or target_command.digest != target.command_digest
                or target_command.operation.effect.effect_id != target.effect_id
                or preparation != target_command.preparation
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            expected_resolution = ReconciliationResolutionRequestV2(
                target.command_digest,
                self.descriptor.digest,
                self.provider_id,
                self.profile_ref,
                self.account_ref,
                self.namespace_ref,
            )
            if target.resolution_digest != expected_resolution.digest:
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            preparation_bytes = preparation.canonical_bytes
            binding, command = _load(
                self._catalog, self._authority, target.command_digest
            )
            binding_snapshot = _binding_snapshot(binding)
            if (
                _target_snapshot(target) != target_snapshot
                or _adapter_snapshot(self) != adapter_snapshot
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            _validate_scope(self, command)
            if (
                target.command_bytes != command.canonical_bytes
                or preparation != command.preparation
                or target.effect_id != command.operation.effect.effect_id
                or command.executor
                != ExecutorDescriptorV1(
                    "modal",
                    "modal-chat-executor",
                    self.descriptor.implementation_version,
                )
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            transport_binding = ModalInferenceCommandBinding(*binding_snapshot)
            transport_command = parse_exact_command(command.canonical_bytes)
            transport_binding_snapshot = _binding_snapshot(transport_binding)
            transport_command_bytes = transport_command.canonical_bytes
            transport_method = _method(self._transport, "lookup_once")
            transport_started = True
            result = transport_method(
                self._transport,
                transport_binding,
                transport_command,
            )
            transport_completed = True
            if (
                _target_snapshot(target) != target_snapshot
                or preparation.canonical_bytes != preparation_bytes
                or _binding_snapshot(binding) != binding_snapshot
                or _adapter_snapshot(self) != adapter_snapshot
                or _binding_snapshot(transport_binding) != transport_binding_snapshot
                or transport_command.canonical_bytes != transport_command_bytes
            ):
                raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
            return _outcome(
                command, target.resolution_digest, target.ownership_epoch, result
            )
        except (KeyboardInterrupt, SystemExit, FoundationError):
            raise
        except Exception:
            if transport_started and not transport_completed:
                raise
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None


class ModalChatExecutorResolver:
    def __init__(self, executor: ModalChatEffectExecutor) -> None:
        if type(executor) is not ModalChatEffectExecutor:
            raise TypeError("exact Modal chat executor required")
        self._executor = executor

    def resolve(self, request: ExecutionResolutionRequestV2):
        snapshot = _request_snapshot(request)
        adapter = _adapter_snapshot(self._executor)
        if adapter != self._executor._initial_snapshot:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        result = mint_resolved_executor(request, self._executor)
        if (
            _request_snapshot(request) != snapshot
            or _adapter_snapshot(self._executor) != adapter
        ):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return result


class ModalChatReconciliationResolver:
    def __init__(self, adapter: ModalChatReconciliationAdapter) -> None:
        if type(adapter) is not ModalChatReconciliationAdapter:
            raise TypeError("exact Modal chat reconciliation adapter required")
        self._adapter = adapter

    def resolve(self, request: ReconciliationResolutionRequestV2):
        try:
            snapshot = _reconciliation_request_snapshot(request)
        except Exception:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH) from None
        adapter = _adapter_snapshot(self._adapter)
        if adapter != self._adapter._initial_snapshot:
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        result = mint_resolved_adapter(request, self._adapter)
        if (
            _reconciliation_request_snapshot(request) != snapshot
            or _adapter_snapshot(self._adapter) != adapter
        ):
            raise FoundationError(DiagnosticCode.BINDING_MISMATCH)
        return result


__all__: list[str] = []
