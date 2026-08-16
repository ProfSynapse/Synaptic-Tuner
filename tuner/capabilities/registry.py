"""Deterministic, side-effect-free capability registry."""

from __future__ import annotations

from collections.abc import Iterable

from synaptic_tuner.api.v1 import CapabilityDescriptor

from .schema import validate_descriptor


class CapabilityRegistry:
    def __init__(self, descriptors: Iterable[CapabilityDescriptor] = ()) -> None:
        self._items: dict[str, CapabilityDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor) -> None:
        validate_descriptor(descriptor)
        if descriptor.id in self._items:
            raise ValueError(f"Duplicate capability id: {descriptor.id}")
        self._items[descriptor.id] = descriptor

    def list(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def describe(self, capability_id: str) -> CapabilityDescriptor:
        try:
            return self._items[capability_id]
        except KeyError as exc:
            raise KeyError(f"Unknown capability: {capability_id}") from exc


def builtin_registry() -> CapabilityRegistry:
    from .builtins import builtin_descriptors

    return CapabilityRegistry(builtin_descriptors())


__all__ = ["CapabilityRegistry", "builtin_registry"]
