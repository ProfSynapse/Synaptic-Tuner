"""Machine-readable result and JSONL event envelope contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ResultEnvelope:
    success: bool
    capability: str
    run_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Sequence[Mapping[str, Any]] = ()
    warnings: Sequence[str] = ()
    timestamp: str = field(default_factory=_timestamp)
    schema_version: str = "synaptic-result/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-result/v1":
            raise ValueError("Unsupported result schema version")
        if not self.capability or not self.run_id:
            raise ValueError("Result capability and run_id are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "capability": self.capability,
            "run_id": self.run_id,
            "data": dict(self.data),
            "artifacts": [dict(item) for item in self.artifacts],
            "warnings": list(self.warnings),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EventEnvelope:
    event: str
    capability: str
    run_id: str
    sequence: int
    data: Mapping[str, Any] = field(default_factory=dict)
    final: bool = False
    result: ResultEnvelope | None = None
    timestamp: str = field(default_factory=_timestamp)
    schema_version: str = "synaptic-event/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-event/v1":
            raise ValueError("Unsupported event schema version")
        if not self.event or not self.capability or not self.run_id:
            raise ValueError("Event name, capability, and run_id are required")
        if self.sequence < 0:
            raise ValueError("Event sequence must be non-negative")
        if self.final and self.result is None:
            raise ValueError("A final event must contain a result envelope")
        if not self.final and self.result is not None:
            raise ValueError("Only a final event may contain a result envelope")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event": self.event,
            "capability": self.capability,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "data": dict(self.data),
            "final": self.final,
            "timestamp": self.timestamp,
        }
        if self.result is not None:
            payload["result"] = self.result.to_dict()
        return payload


__all__ = ["EventEnvelope", "ResultEnvelope"]
