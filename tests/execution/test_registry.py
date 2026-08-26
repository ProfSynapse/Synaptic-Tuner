from __future__ import annotations

import pytest

from tuner.execution.contracts import (
    EffectDisposition,
    EffectIdentity,
    EffectKind,
    EffectObservation,
    ExecutionScope,
)
from tuner.execution.registry import (
    AdapterAlreadyRegistered,
    AdapterNotRegistered,
    ReconciliationRegistry,
)


class LookupOnlyAdapter:
    provider = "modal"

    def lookup_effect(self, identity: EffectIdentity) -> EffectObservation:
        return EffectObservation(identity, EffectDisposition.INDETERMINATE)


def effect(provider: str = "modal") -> EffectIdentity:
    return EffectIdentity(
        "effect-1",
        "operation-1",
        EffectKind.SUBMIT,
        ExecutionScope(provider, "account-1", "namespace-1"),
    )


def test_registry_resolves_lookup_only_adapter() -> None:
    registry = ReconciliationRegistry()
    adapter = LookupOnlyAdapter()
    registry.register(adapter)
    assert registry.resolve(effect()) is adapter
    assert registry.providers() == ("modal",)
    assert not hasattr(adapter, "submit")
    assert not hasattr(adapter, "cancel")


def test_registry_rejects_duplicates_and_missing_provider() -> None:
    registry = ReconciliationRegistry()
    registry.register(LookupOnlyAdapter())
    with pytest.raises(AdapterAlreadyRegistered):
        registry.register(LookupOnlyAdapter())
    with pytest.raises(AdapterNotRegistered):
        registry.resolve(effect("runpod"))
