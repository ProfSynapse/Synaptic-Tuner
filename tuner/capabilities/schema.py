"""Validation adapters for the public v1 capability contracts."""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from synaptic_tuner.api.v1 import CapabilityDescriptor, EventEnvelope, ResultEnvelope

_SCHEMA_FILES = {
    "capability": "synaptic-capability-v1.schema.json",
    "result": "synaptic-result-v1.schema.json",
    "event": "synaptic-event-v1.schema.json",
}


def _schema_root() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"


@lru_cache(maxsize=None)
def _validator(kind: str) -> Draft202012Validator:
    path = _schema_root() / _SCHEMA_FILES[kind]
    schema = json.loads(path.read_text(encoding="utf-8"))
    resources: list[tuple[str, Resource[Any]]] = []
    for schema_file in _SCHEMA_FILES.values():
        dependency_path = _schema_root() / schema_file
        dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(dependency)
        resources.append((dependency_path.as_uri(), resource))
        if "$id" in dependency:
            resources.append((dependency["$id"], resource))
    registry = Registry().with_resources(resources)
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def _payload(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    raise TypeError("Contract value must be a public v1 envelope or mapping")


def validate_descriptor(value: CapabilityDescriptor | Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_payload(value))
    _validator("capability").validate(payload)
    return payload


def validate_result(value: ResultEnvelope | Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_payload(value))
    _validator("result").validate(payload)
    _validate_timestamp(payload.get("timestamp"))
    return payload


def validate_event(value: EventEnvelope | Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_payload(value))
    _validator("event").validate(payload)
    _validate_timestamp(payload.get("timestamp"))
    result = payload.get("result")
    if isinstance(result, Mapping):
        _validate_timestamp(result.get("timestamp"))
    return payload


def _validate_timestamp(value: Any) -> None:
    from jsonschema import ValidationError

    if not isinstance(value, str):
        raise ValidationError("timestamp must be an RFC 3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("timestamp must be RFC 3339 date-time") from exc
    if parsed.tzinfo is None:
        raise ValidationError("timestamp must include a timezone")


__all__ = ["validate_descriptor", "validate_event", "validate_result"]
