"""Exact, explicit-client Modal 1.5.4 read adapter.

Importing the engine never imports Modal. A host constructs and authenticates
the SDK client, then injects provider scope and deployment observers. There is
deliberately no profile, token, or ``Client.from_env`` fallback here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Protocol

from ...contracts import safe_ref
from .binding import CapabilityProofV1, ModalClientBinding
from .contracts import canonical_path, provider_entry_identity, strict_int
from .resolution import ModalDeploymentSelectionV1

EXACT_MODAL_SDK_VERSION = "1.5.4"
MODAL_VOLUME_V1 = 1


class ModalFunctionCallState(str, Enum):
    PENDING = "pending"
    RETURNED = "returned"
    UNKNOWN = "unknown"


class ModalFacadeError(RuntimeError):
    """Stable non-secret provider read failure."""


class ModalFacade(Protocol):
    def bound_scope(self) -> tuple[str, str, str, str]: ...
    def capability_proof(self, binding: ModalClientBinding) -> CapabilityProofV1: ...
    def read_complete(self, volume_id: str, path: str, *, max_bytes: int) -> bytes: ...
    def list_prefix(self, volume_id: str, prefix: str, *, max_entries: int) -> tuple[tuple[str, int, str], ...]: ...
    def inspect_deployment(self, *, app_name: str, function_name: str) -> ModalDeploymentSelectionV1: ...


def _closed_prefix(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prefix must be a string")
    stripped = value[:-1] if value.endswith("/") else value
    canonical_path(stripped)
    return stripped + "/"


class ExplicitModal154ReadFacade:
    """Read-only Modal adapter bound to one authenticated explicit client."""

    __slots__ = (
        "binding", "sdk", "client", "_scope_observer", "_deployment_observer",
        "_volume_names",
    )

    def __init__(
        self,
        binding: ModalClientBinding,
        *,
        sdk: object,
        client: object,
        scope_observer: Callable[[object], tuple[str, str, str, str]],
        deployment_observer: Callable[..., ModalDeploymentSelectionV1],
        volume_names: Mapping[str, str],
    ) -> None:
        if type(binding) is not ModalClientBinding:
            raise TypeError("binding must be ModalClientBinding")
        if binding.sdk_version != EXACT_MODAL_SDK_VERSION:
            raise ModalFacadeError("modal_sdk_version_mismatch")
        if sdk is None or getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
            raise ModalFacadeError("modal_sdk_version_mismatch")
        if client is None:
            raise TypeError("an explicit Modal client is required")
        if not callable(scope_observer) or not callable(deployment_observer):
            raise TypeError("explicit scope and deployment observers are required")
        names = dict(volume_names)
        if not names:
            raise ValueError("at least one exact Modal volume binding is required")
        normalized: dict[str, str] = {}
        for volume_id, name in names.items():
            normalized[safe_ref(volume_id, "volume_id")] = safe_ref(name, "volume_name")
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("Modal volume names must be unique")
        self.binding = binding
        self.sdk = sdk
        self.client = client
        self._scope_observer = scope_observer
        self._deployment_observer = deployment_observer
        self._volume_names = normalized

    def bound_scope(self) -> tuple[str, str, str, str]:
        try:
            scope = self._scope_observer(self.client)
        except Exception:
            raise ModalFacadeError("modal_scope_unavailable") from None
        expected = (
            self.binding.account_ref, self.binding.workspace_ref,
            self.binding.environment_ref, self.binding.client_ref,
        )
        if scope != expected:
            raise ModalFacadeError("modal_scope_mismatch")
        return expected

    def capability_proof(self, binding: ModalClientBinding) -> CapabilityProofV1:
        if type(binding) is not ModalClientBinding or binding != self.binding:
            return CapabilityProofV1(False, False, False, False, False, False, False)
        try:
            self.bound_scope()
            volume = self.sdk.Volume
            function = self.sdk.Function
            volume_io = all(hasattr(volume, member) for member in ("from_name", "read_file"))
            volume_listing = hasattr(volume, "iterdir")
            function_version = hasattr(function, "from_name")
        except Exception:
            return CapabilityProofV1(False, False, False, False, False, False, False)
        return CapabilityProofV1(
            True, True, volume_io, volume_listing, True, True, function_version
        )

    def _volume(self, volume_id: str):
        volume_id = safe_ref(volume_id, "volume_id")
        try:
            name = self._volume_names[volume_id]
            return self.sdk.Volume.from_name(
                name,
                environment_name=self.binding.environment_ref,
                create_if_missing=False,
                version=MODAL_VOLUME_V1,
                client=self.client,
            )
        except Exception:
            raise ModalFacadeError("modal_volume_unavailable") from None

    def volume_name(self, volume_id: str) -> str:
        """Return the configured opaque name for one exact provider volume ID."""
        try:
            return self._volume_names[safe_ref(volume_id, "volume_id")]
        except (KeyError, ValueError):
            raise ModalFacadeError("modal_volume_unavailable") from None

    @staticmethod
    def _assert_volume_id(volume: object, expected: str) -> None:
        try:
            actual = volume.object_id
        except Exception:
            raise ModalFacadeError("modal_volume_identity_unavailable") from None
        if actual != expected:
            raise ModalFacadeError("modal_volume_identity_mismatch")

    def read_complete(self, volume_id: str, path: str, *, max_bytes: int) -> bytes:
        canonical_path(path)
        strict_int(max_bytes, "max_bytes", minimum=1)
        volume = self._volume(volume_id)
        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in volume.read_file(path):
                if not isinstance(chunk, bytes):
                    raise ValueError
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError
                chunks.append(chunk)
            self._assert_volume_id(volume, volume_id)
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_volume_read_failed") from None
        return b"".join(chunks)

    def list_prefix(self, volume_id: str, prefix: str, *, max_entries: int) -> tuple[tuple[str, int, str], ...]:
        prefix = _closed_prefix(prefix)
        strict_int(max_entries, "max_entries", minimum=1)
        volume = self._volume(volume_id)
        result: list[tuple[str, int, str]] = []
        seen: set[str] = set()
        try:
            for entry in volume.iterdir(prefix, recursive=True):
                if len(result) >= max_entries:
                    raise ValueError
                path = canonical_path(entry.path)
                if not path.startswith(prefix) or path in seen or int(entry.type) != 1:
                    raise ValueError
                size = strict_int(entry.size, "entry_size")
                strict_int(entry.mtime, "entry_mtime")
                identity = provider_entry_identity(volume_id, path, size)
                seen.add(path)
                result.append((path, size, identity))
            self._assert_volume_id(volume, volume_id)
        except ModalFacadeError:
            raise
        except Exception:
            raise ModalFacadeError("modal_volume_list_failed") from None
        return tuple(result)

    def inspect_deployment(self, *, app_name: str, function_name: str) -> ModalDeploymentSelectionV1:
        app_name = safe_ref(app_name, "app_name")
        function_name = safe_ref(function_name, "function_name")
        try:
            observed = self._deployment_observer(
                client=self.client,
                app_name=app_name,
                function_name=function_name,
                environment_name=self.binding.environment_ref,
            )
        except Exception:
            raise ModalFacadeError("modal_deployment_unavailable") from None
        if type(observed) is not ModalDeploymentSelectionV1:
            raise ModalFacadeError("modal_deployment_invalid")
        if (
            observed.app_name != app_name or observed.function_name != function_name
            or observed.account_ref != self.binding.account_ref
            or observed.workspace_ref != self.binding.workspace_ref
            or observed.environment_ref != self.binding.environment_ref
            or observed.client_ref != self.binding.client_ref
            or observed.sdk_version != EXACT_MODAL_SDK_VERSION
        ):
            raise ModalFacadeError("modal_deployment_identity_mismatch")
        try:
            function = self._function(
                app_name=app_name,
                function_name=function_name,
                function_version=observed.function_version,
            )
            function.hydrate(self.client)
            safe_ref(function.object_id, "function_id")
        except Exception:
            raise ModalFacadeError("modal_function_identity_unavailable") from None
        return observed

    def _function(self, *, app_name: str, function_name: str, function_version: str):
        """Private exact function resolver used only by the sole-submit adapter."""
        try:
            version = int(function_version)
            if str(version) != function_version or version < 1:
                raise ValueError
            return self.sdk.Function.from_name(
                safe_ref(app_name, "app_name"),
                safe_ref(function_name, "function_name"),
                version=version,
                environment_name=self.binding.environment_ref,
                client=self.client,
            )
        except Exception:
            raise ModalFacadeError("modal_function_unavailable") from None

    def observe_function_call(self, provider_job_ref: str) -> ModalFunctionCallState:
        """Poll one exact call without treating its result as completion evidence."""
        provider_job_ref = safe_ref(provider_job_ref, "provider_job_ref")
        try:
            call = self.sdk.FunctionCall.from_id(provider_job_ref, client=self.client)
            result = call.get(timeout=0)
        except TimeoutError:
            return ModalFunctionCallState.PENDING
        except Exception:
            return ModalFunctionCallState.UNKNOWN
        if not isinstance(result, Mapping) or set(result) != {
            "schema_version", "effect_id", "returncode", "status_code"
        }:
            return ModalFunctionCallState.UNKNOWN
        if (
            result.get("schema_version") != "synaptic-modal-worker-result/v1"
            or not isinstance(result.get("effect_id"), str)
            or type(result.get("returncode")) is not int
            or result.get("status_code") not in {"completed", "failed"}
        ):
            return ModalFunctionCallState.UNKNOWN
        return ModalFunctionCallState.RETURNED


__all__ = [
    "EXACT_MODAL_SDK_VERSION", "ExplicitModal154ReadFacade", "ModalFacade",
    "ModalFacadeError", "ModalFunctionCallState",
]
