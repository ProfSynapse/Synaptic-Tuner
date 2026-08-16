"""Agent-discoverable capability metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    summary: str
    command: Sequence[str]
    config_schema: str | None = None
    inputs: Sequence[Mapping[str, Any]] = ()
    outputs: Sequence[Mapping[str, Any]] = ()
    effects: Mapping[str, Any] = field(default_factory=dict)
    confirmation: Mapping[str, Any] = field(default_factory=dict)
    resumable: bool = False
    supports: Mapping[str, bool] = field(default_factory=dict)
    schema_version: str = "synaptic-capability/v1"

    def __post_init__(self) -> None:
        if not self.id or not self.summary or not self.command:
            raise ValueError("Capability id, summary, and command are required")
        if self.schema_version != "synaptic-capability/v1":
            raise ValueError("Unsupported capability schema version")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "summary": self.summary,
            "command": list(self.command),
            "inputs": [dict(item) for item in self.inputs],
            "outputs": [dict(item) for item in self.outputs],
            "effects": dict(self.effects),
            "confirmation": dict(self.confirmation),
            "resumable": self.resumable,
            "supports": dict(self.supports),
        }
        if self.config_schema is not None:
            result["config_schema"] = self.config_schema
        return result


__all__ = ["CapabilityDescriptor"]
