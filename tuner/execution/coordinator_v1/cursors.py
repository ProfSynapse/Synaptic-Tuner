"""Canonical authenticated cursors for internal run operations."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import hmac
from types import MappingProxyType
from typing import Mapping

from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest, safe_ref


_VERSION = 1
_PREFIX = "sc1."
_HMAC_DOMAIN = b"synaptic-authenticated-cursor/v1\0"
_ENVELOPE_BYTES = 134
_PAYLOAD_CHARACTERS = 179


class CursorKindV1(IntEnum):
    RUN_LIST = 1
    RUN_LOGS = 2


@dataclass(frozen=True, slots=True)
class CursorContentV1:
    kind: CursorKindV1
    query_digest: bytes
    after_run_key: bytes | None = None
    after_sequence: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not CursorKindV1:
            raise TypeError("kind must be exact CursorKindV1")
        if type(self.query_digest) is not bytes or len(self.query_digest) != 32:
            raise ValueError("query_digest must be exactly 32 bytes")
        if self.kind is CursorKindV1.RUN_LIST:
            if type(self.after_run_key) is not bytes or len(self.after_run_key) != 32:
                raise ValueError("list cursor requires an exact 32-byte run key")
            if self.after_sequence is not None:
                raise ValueError("list cursor cannot carry a log sequence")
        else:
            if self.after_run_key is not None:
                raise ValueError("log cursor cannot carry a run key")
            if (
                type(self.after_sequence) is not int
                or not 0 <= self.after_sequence <= 2**64 - 1
            ):
                raise ValueError("log cursor requires an unsigned 64-bit sequence")

    @property
    def boundary(self) -> bytes:
        if self.kind is CursorKindV1.RUN_LIST:
            return self.after_run_key
        return b"\0" * 24 + self.after_sequence.to_bytes(8, "big")


@dataclass(frozen=True, slots=True)
class AuthenticatedCursorV1:
    content: CursorContentV1
    authority_digest: bytes
    key_generation: int
    tag: bytes

    def __post_init__(self) -> None:
        if type(self.content) is not CursorContentV1:
            raise TypeError("content must be exact CursorContentV1")
        if type(self.authority_digest) is not bytes or len(self.authority_digest) != 32:
            raise ValueError("authority_digest must be exactly 32 bytes")
        if (
            type(self.key_generation) is not int
            or not 1 <= self.key_generation <= 2**32 - 1
        ):
            raise ValueError("key_generation must be an unsigned nonzero 32-bit integer")
        if type(self.tag) is not bytes or len(self.tag) != 32:
            raise ValueError("tag must be exactly 32 bytes")

    @property
    def unsigned_bytes(self) -> bytes:
        return b"".join(
            (
                bytes((_VERSION, int(self.content.kind))),
                self.authority_digest,
                self.key_generation.to_bytes(4, "big"),
                self.content.query_digest,
                self.content.boundary,
            )
        )

    @property
    def canonical_bytes(self) -> bytes:
        raw = self.unsigned_bytes + self.tag
        if len(raw) != _ENVELOPE_BYTES:
            raise ValueError("cursor envelope length invalid")
        return raw


def encode_cursor(cursor: AuthenticatedCursorV1) -> str:
    if type(cursor) is not AuthenticatedCursorV1:
        raise TypeError("cursor must be exact AuthenticatedCursorV1")
    payload = base64.urlsafe_b64encode(cursor.canonical_bytes).decode("ascii").rstrip("=")
    if len(payload) != _PAYLOAD_CHARACTERS:
        raise ValueError("cursor payload length invalid")
    return _PREFIX + payload


def decode_cursor(token: str) -> AuthenticatedCursorV1:
    if (
        type(token) is not str
        or len(token) != len(_PREFIX) + _PAYLOAD_CHARACTERS
        or not token.startswith(_PREFIX)
    ):
        raise ValueError("cursor token shape invalid")
    payload = token[len(_PREFIX) :]
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for character in payload):
        raise ValueError("cursor token alphabet invalid")
    try:
        raw = base64.b64decode(payload + "=", altchars=b"-_", validate=True)
    except Exception:
        raise ValueError("cursor token encoding invalid") from None
    if len(raw) != _ENVELOPE_BYTES:
        raise ValueError("cursor envelope length invalid")
    if encode_cursor_bytes(raw) != token:
        raise ValueError("cursor token is not canonical")
    if raw[0] != _VERSION:
        raise ValueError("cursor version invalid")
    try:
        kind = CursorKindV1(raw[1])
    except ValueError:
        raise ValueError("cursor kind invalid") from None
    authority_digest = raw[2:34]
    generation = int.from_bytes(raw[34:38], "big")
    query_digest = raw[38:70]
    boundary = raw[70:102]
    if kind is CursorKindV1.RUN_LIST:
        content = CursorContentV1(kind, query_digest, after_run_key=boundary)
    else:
        if boundary[:24] != b"\0" * 24:
            raise ValueError("log cursor boundary padding invalid")
        content = CursorContentV1(
            kind, query_digest, after_sequence=int.from_bytes(boundary[24:], "big")
        )
    return AuthenticatedCursorV1(content, authority_digest, generation, raw[102:])


def encode_cursor_bytes(raw: bytes) -> str:
    if type(raw) is not bytes or len(raw) != _ENVELOPE_BYTES:
        raise ValueError("cursor envelope length invalid")
    return _PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class HMACCursorAuthorityV1:
    __slots__ = (
        "authority_ref",
        "authority_digest",
        "active_generation",
        "_keys",
        "_revoked",
    )

    def __init__(
        self,
        authority_ref: str,
        keys: Mapping[int, bytes],
        *,
        active_generation: int,
        revoked_generations: frozenset[int] = frozenset(),
    ) -> None:
        self.authority_ref = safe_ref(authority_ref, "authority_ref")
        if type(keys) is not dict and not isinstance(keys, Mapping):
            raise TypeError("keys must be a mapping")
        copied: dict[int, bytes] = {}
        for generation, key in keys.items():
            if type(generation) is not int or not 1 <= generation <= 2**32 - 1:
                raise ValueError("cursor key generation invalid")
            if type(key) is not bytes or len(key) < 32:
                raise ValueError("cursor HMAC key must contain at least 32 bytes")
            copied[generation] = bytes(key)
        if len(set(copied.values())) != len(copied):
            raise ValueError("cursor HMAC key material must not be reused")
        if (
            type(active_generation) is not int
            or active_generation not in copied
            or active_generation in revoked_generations
        ):
            raise ValueError("active cursor generation invalid")
        if type(revoked_generations) is not frozenset or any(
            type(value) is not int or value not in copied for value in revoked_generations
        ):
            raise ValueError("revoked cursor generations invalid")
        self.active_generation = active_generation
        self._keys = MappingProxyType(copied)
        self._revoked = revoked_generations
        self.authority_digest = bytes.fromhex(
            domain_digest(
                "synaptic-cursor-authority/v1",
                canonical_bytes({"authority_ref": self.authority_ref}),
            )
        )

    def __repr__(self) -> str:
        return (
            f"HMACCursorAuthorityV1(authority_ref={self.authority_ref!r}, "
            f"active_generation={self.active_generation!r}, keys=<redacted>)"
        )

    def issue(self, content: CursorContentV1) -> AuthenticatedCursorV1:
        if type(content) is not CursorContentV1:
            raise TypeError("content must be exact CursorContentV1")
        unsigned = AuthenticatedCursorV1(
            content, self.authority_digest, self.active_generation, b"\0" * 32
        ).unsigned_bytes
        tag = hmac.new(
            self._keys[self.active_generation], _HMAC_DOMAIN + unsigned, hashlib.sha256
        ).digest()
        return AuthenticatedCursorV1(
            content, self.authority_digest, self.active_generation, tag
        )

    def verify(self, cursor: AuthenticatedCursorV1) -> bool:
        try:
            if (
                type(cursor) is not AuthenticatedCursorV1
                or cursor.authority_digest != self.authority_digest
                or cursor.key_generation in self._revoked
            ):
                return False
            key = self._keys.get(cursor.key_generation)
            if key is None:
                return False
            expected = hmac.new(
                key, _HMAC_DOMAIN + cursor.unsigned_bytes, hashlib.sha256
            ).digest()
            return hmac.compare_digest(cursor.tag, expected)
        except Exception:
            return False


__all__: list[str] = []
