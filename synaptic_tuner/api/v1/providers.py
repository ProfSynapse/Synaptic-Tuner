"""Provider-neutral identities, descriptors, and closed capabilities.

This module is the staged B1 contract foundation.  It is deliberately not
re-exported by :mod:`synaptic_tuner.api.v1` until the operational internals can
move atomically to the new contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._contract import contract_digest, exact_fields, required_text


@dataclass(frozen=True, slots=True)
class ProviderRef:
    provider_id: str
    profile_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", required_text(self.provider_id, "provider_id"))
        object.__setattr__(self, "profile_ref", required_text(self.profile_ref, "profile_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"provider_id": self.provider_id, "profile_ref": self.profile_ref}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderRef":
        exact_fields(value, frozenset({"provider_id", "profile_ref"}), "provider_ref")
        return cls(provider_id=value["provider_id"], profile_ref=value["profile_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Closed lifecycle claims; absence of a capability is explicit."""

    observe: bool
    logs: bool
    cancel: bool
    reconcile: bool
    artifact_streaming: bool
    cost_quote: bool

    def __post_init__(self) -> None:
        for name in (
            "observe",
            "logs",
            "cancel",
            "reconcile",
            "artifact_streaming",
            "cost_quote",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "observe": self.observe,
            "logs": self.logs,
            "cancel": self.cancel,
            "reconcile": self.reconcile,
            "artifact_streaming": self.artifact_streaming,
            "cost_quote": self.cost_quote,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderCapabilities":
        fields = frozenset(
            {"observe", "logs", "cancel", "reconcile", "artifact_streaming", "cost_quote"}
        )
        exact_fields(value, fields, "provider_capabilities")
        return cls(**{name: value[name] for name in fields})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    schema_version: str
    provider_id: str
    display_name: str
    implementation_version: str
    capabilities: ProviderCapabilities

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-provider-descriptor/v1":
            raise ValueError("unsupported provider descriptor schema version")
        object.__setattr__(self, "provider_id", required_text(self.provider_id, "provider_id"))
        object.__setattr__(self, "display_name", required_text(self.display_name, "display_name"))
        object.__setattr__(
            self,
            "implementation_version",
            required_text(self.implementation_version, "implementation_version"),
        )
        if not isinstance(self.capabilities, ProviderCapabilities):
            raise TypeError("capabilities must be ProviderCapabilities")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "implementation_version": self.implementation_version,
            "capabilities": self.capabilities.to_dict(),
        }

    @property
    def descriptor_digest(self) -> str:
        return contract_digest("synaptic-provider-descriptor/v1", self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProviderDescriptor":
        exact_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "provider_id",
                    "display_name",
                    "implementation_version",
                    "capabilities",
                }
            ),
            "provider_descriptor",
        )
        capabilities = value["capabilities"]
        if not isinstance(capabilities, Mapping):
            raise TypeError("capabilities must be an object")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            provider_id=value["provider_id"],  # type: ignore[arg-type]
            display_name=value["display_name"],  # type: ignore[arg-type]
            implementation_version=value["implementation_version"],  # type: ignore[arg-type]
            capabilities=ProviderCapabilities.from_dict(capabilities),
        )


__all__ = ["ProviderCapabilities", "ProviderDescriptor", "ProviderRef"]
