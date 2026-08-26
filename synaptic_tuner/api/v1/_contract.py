"""Small canonical-JSON primitives shared only by public contract modules."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} is required")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def canonical_integer(value: int | float, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{name} must be a finite integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def digest_text(value: str, name: str) -> str:
    value = required_text(value, name)
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def exact_fields(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("contract must contain only canonical JSON values") from exc


def contract_digest(domain: str, value: Mapping[str, object]) -> str:
    domain = required_text(domain, "domain")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + canonical_bytes(value)).hexdigest()


__all__: list[str] = []
