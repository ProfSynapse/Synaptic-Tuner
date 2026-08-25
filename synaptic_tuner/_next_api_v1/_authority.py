"""Private authority-only grant registration capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .execution import AccessContext, ExecutionGrant

if TYPE_CHECKING:
    from ._store import GrantBinding, JobStore


_CAPABILITY_SEAL = object()


class _RegistrarCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _CAPABILITY_SEAL:
            raise TypeError("registrar capabilities are authority-created")
        self._seal = seal

    def __repr__(self) -> str:
        return "_RegistrarCapability(<opaque>)"

    def __reduce__(self) -> object:
        raise TypeError("registrar capabilities cannot be serialized")


def _is_registrar_capability(value: object) -> bool:
    return isinstance(value, _RegistrarCapability) and value._seal is _CAPABILITY_SEAL


class GrantRegistrar:
    """Authority composition surface intentionally absent from lifecycle APIs."""

    __slots__ = ("_store", "_capability")

    def __init__(self, store: JobStore, capability: _RegistrarCapability) -> None:
        if not _is_registrar_capability(capability):
            raise TypeError("a valid registrar capability is required")
        self._store = store
        self._capability = capability

    def register(
        self, access: AccessContext, grant: ExecutionGrant, binding: GrantBinding
    ) -> None:
        self._store._register_grant_from_authority(
            self._capability, access, grant, binding
        )


def create_grant_registrar(store: JobStore) -> GrantRegistrar:
    """Trusted composition root entrypoint; never exposed by JobStore/JobService."""

    return GrantRegistrar(store, _RegistrarCapability(_CAPABILITY_SEAL))


__all__ = ["GrantRegistrar", "create_grant_registrar"]
