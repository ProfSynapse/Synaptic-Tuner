"""Small canonical-JSON primitives shared only by public contract modules."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Mapping


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def required_text(value: str, name: str) -> str:
    if type(value) is not str:
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


def exact_integer(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def digest_text(value: str, name: str) -> str:
    value = required_text(value, name)
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def exact_fields(
    value: dict[str, object], expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    keys = tuple(dict.keys(value))
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{name} field names must be exact strings")
    actual = frozenset(keys)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")
    return {key: dict.__getitem__(value, key) for key in keys}


def canonical_bytes(value: dict[str, object]) -> bytes:
    if type(value) is not dict:
        raise TypeError("contract must be an exact object")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("contract must contain only canonical JSON values") from None


def contract_digest(domain: str, value: dict[str, object]) -> str:
    domain = required_text(domain, "domain")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + canonical_bytes(value)).hexdigest()


_PREPARED_PREFIX = "prepared://sha256/"
_MAX_PREPARED_INPUT_BYTES = 64 * 1024 * 1024
_FORMAT_SEGMENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class PreparedTrainingInputIdentity:
    """Complete canonical identity consumed by packaged execution bindings."""

    ref: str
    revision: str
    content_digest: str
    size_bytes: int
    format: str

    @classmethod
    def validate_format(cls, value: str) -> str:
        format_value = required_text(value, "format")
        segments = format_value.split("/")
        if (
            len(format_value.encode("utf-8")) > 128
            or not format_value[0].isalpha()
            or any(
                segment in {"", ".", ".."}
                or _FORMAT_SEGMENT_PATTERN.fullmatch(segment) is None
                for segment in segments
            )
        ):
            raise ValueError("prepared training input format must be a logical identifier")
        return format_value

    def __post_init__(self) -> None:
        if type(self.ref) is not str or not self.ref.startswith(_PREPARED_PREFIX):
            raise ValueError("prepared training input reference is invalid")
        semantic_digest = digest_text(
            self.ref[len(_PREPARED_PREFIX):], "prepared input semantic digest"
        )
        if type(self.revision) is not str or self.revision != semantic_digest:
            raise ValueError("prepared training input revision is invalid")
        object.__setattr__(
            self, "content_digest", digest_text(self.content_digest, "content_digest")
        )
        if (
            type(self.size_bytes) is not int
            or not 1 <= self.size_bytes <= _MAX_PREPARED_INPUT_BYTES
        ):
            raise ValueError("prepared training input size is invalid")
        object.__setattr__(self, "format", self.validate_format(self.format))

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "revision": self.revision,
            "content_digest": self.content_digest,
            "size_bytes": self.size_bytes,
            "format": self.format,
        }

    def execution_binding_fields(self) -> dict[str, object]:
        """Map the complete identity onto packaged execution-binding fields."""

        return {
            "prepared_input_ref": self.ref,
            "prepared_input_revision": self.revision,
            "prepared_input_content_digest": self.content_digest,
            "prepared_input_size_bytes": self.size_bytes,
            "prepared_input_format": self.format,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PreparedTrainingInputIdentity":
        if type(value) is not dict or set(value) != {
            "ref", "revision", "content_digest", "size_bytes", "format",
        }:
            raise ValueError("prepared training input identity is malformed")
        return cls(**value)  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PreparedTrainingInputIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("prepared training input identity is malformed")
        return cls.from_dict(dict(value))

# Preserve public identity, repr and pickle lookup while sharing implementation.
PreparedTrainingInputIdentity.__module__ = "synaptic_tuner.api.v1.execution"


__all__: list[str] = []
