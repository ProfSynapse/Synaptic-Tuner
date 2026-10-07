"""Bounded canonical primitives shared only by the B2 foundation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from enum import Enum


MAX_CANONICAL_BYTES = 16 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")


class DiagnosticCode(str, Enum):
    AUTHORITY_INVALID = "authority_invalid"
    BINDING_MISMATCH = "binding_mismatch"
    EVIDENCE_INVALID = "evidence_invalid"
    EFFECT_CONFLICT = "effect_conflict"
    EFFECT_INELIGIBLE = "effect_ineligible"
    EFFECT_AMBIGUOUS = "effect_ambiguous"
    RECONCILIATION_CONFLICT = "reconciliation_conflict"
    RECONCILIATION_INTERRUPTED = "reconciliation_interrupted"
    FINALITY_UNPROVEN = "finality_unproven"
    STALE_RESULT = "stale_result"


class FoundationError(RuntimeError):
    def __init__(self, code: DiagnosticCode):
        if not isinstance(code, DiagnosticCode):
            raise TypeError("code must be DiagnosticCode")
        self.code = code
        super().__init__(code.value)


def safe_ref(value: str, name: str) -> str:
    if not isinstance(value, str) or _REF_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded safe reference")
    return value


def digest_text(value: str, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def exact_integer(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer of at least {minimum}")
    return value


def exact_fields(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != expected:
        raise ValueError(f"{name} contains missing or unknown fields")


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise ValueError("canonical value exceeds the byte limit")
    return encoded


def parse_canonical_object(raw: bytes, *, name: str) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_CANONICAL_BYTES:
        raise ValueError(f"{name} must be bounded nonempty bytes")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(_: str) -> object:
        raise ValueError(f"{name} contains a non-finite number")

    def reject_float(_: str) -> object:
        raise ValueError(f"{name} contains a non-integer number")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_constant, parse_float=reject_float,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def domain_digest(domain: str, payload: bytes) -> str:
    safe_ref(domain, "domain")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
