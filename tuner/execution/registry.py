"""Registry for provider read-only reconciliation adapters."""

from __future__ import annotations

from .contracts import EffectIdentity, ReconciliationAdapter, safe_ref


class AdapterAlreadyRegistered(RuntimeError):
    pass


class AdapterNotRegistered(RuntimeError):
    pass


class ReconciliationRegistry:
    """Maps provider names to lookup-only adapters.

    Mutation clients are intentionally outside this registry so resolution of
    persisted state can never synthesize execution authority.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, ReconciliationAdapter] = {}

    def register(self, adapter: ReconciliationAdapter) -> None:
        if not isinstance(adapter, ReconciliationAdapter):
            raise TypeError("adapter must implement ReconciliationAdapter")
        provider = safe_ref(adapter.provider.lower(), "provider")
        if provider in self._adapters:
            raise AdapterAlreadyRegistered("reconciliation adapter is already registered")
        self._adapters[provider] = adapter

    def resolve(self, identity: EffectIdentity) -> ReconciliationAdapter:
        try:
            return self._adapters[identity.scope.provider]
        except KeyError as exc:
            raise AdapterNotRegistered("reconciliation adapter is not registered") from exc

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


__all__ = [
    "AdapterAlreadyRegistered",
    "AdapterNotRegistered",
    "ReconciliationRegistry",
]
