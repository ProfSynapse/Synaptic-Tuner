"""Packaged host-only clock and authenticated reader evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
from types import MappingProxyType
from typing import Mapping

from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1, AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1, ProviderRunObservationContentV1,
)
from tuner.execution.foundation_v2.canonical import safe_ref


class UTCClock:
    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def now_iso(self) -> str:
        return self.now()

    def now_epoch(self) -> int:
        return int(datetime.now(timezone.utc).timestamp())


class HMACAuthenticator:
    def __init__(self, keys: Mapping[str, bytes], *, allowed_purposes: frozenset[str]):
        if type(allowed_purposes) is not frozenset or not allowed_purposes:
            raise ValueError("host HMAC purposes required")
        copied = {}
        for ref, key in keys.items():
            if type(key) is not bytes or len(key) < 32:
                raise ValueError("host HMAC key invalid")
            copied[safe_ref(ref, "key_ref")] = bytes(key)
        if not copied:
            raise ValueError("host HMAC key required")
        self._keys = MappingProxyType(copied)
        self._purposes = frozenset(safe_ref(value, "purpose") for value in allowed_purposes)

    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        if purpose not in self._purposes or type(payload) is not bytes or key_ref not in self._keys:
            raise ValueError("host HMAC authorization invalid")
        return hmac.new(self._keys[key_ref], purpose.encode("ascii") + b"\0" + payload,
                        hashlib.sha256).digest()

    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool:
        try:
            return type(tag) is bytes and hmac.compare_digest(
                tag, self.sign(purpose, payload, key_ref),
            )
        except Exception:
            return False


class ReaderEvidenceAuthority:
    def __init__(self, authority_ref: str, key_ref: str, authenticator: HMACAuthenticator):
        self.authority_ref = safe_ref(authority_ref, "authority_ref")
        self.key_ref = safe_ref(key_ref, "key_ref")
        if type(authenticator) is not HMACAuthenticator:
            raise TypeError("exact HMAC authenticator required")
        self._hmac = authenticator

    def _tag(self, purpose: str, digest: str) -> str:
        return self._hmac.sign(purpose, bytes.fromhex(digest), self.key_ref).hex()

    def observation(self, content):
        if type(content) is not ProviderRunObservationContentV1:
            raise TypeError("exact observation content required")
        envelope = AuthenticatedProviderRunObservationV1(
            content, self.authority_ref, self.key_ref,
            self._tag("provider-run-observation/v1", content.content_digest),
        )
        return AuthenticatedProviderRunObservationV1.parse(envelope.canonical_bytes)

    def log_page(self, content):
        if type(content) is not ProviderLogPageContentV1:
            raise TypeError("exact log content required")
        envelope = AuthenticatedProviderLogPageV1(
            content, self.authority_ref, self.key_ref,
            self._tag("provider-log-page/v1", content.content_digest),
        )
        return AuthenticatedProviderLogPageV1.parse(envelope.canonical_bytes)

    def authenticate_observation(self, value) -> bool:
        try:
            owned = AuthenticatedProviderRunObservationV1.parse(value.canonical_bytes)
            return (type(value) is AuthenticatedProviderRunObservationV1 and owned == value
                    and owned.authority_ref == self.authority_ref
                    and owned.key_ref == self.key_ref
                    and hmac.compare_digest(owned.tag, self._tag(
                        "provider-run-observation/v1", owned.content.content_digest)))
        except Exception:
            return False

    def authenticate_log_page(self, value) -> bool:
        try:
            owned = AuthenticatedProviderLogPageV1.parse(value.canonical_bytes)
            return (type(value) is AuthenticatedProviderLogPageV1 and owned == value
                    and owned.authority_ref == self.authority_ref
                    and owned.key_ref == self.key_ref
                    and hmac.compare_digest(owned.tag, self._tag(
                        "provider-log-page/v1", owned.content.content_digest)))
        except Exception:
            return False


class ObservationAuthenticator:
    def __init__(self, authority: ReaderEvidenceAuthority):
        self._authority = authority

    def authenticate(self, value):
        return self._authority.authenticate_observation(value)


class LogAuthenticator:
    def __init__(self, authority: ReaderEvidenceAuthority):
        self._authority = authority

    def authenticate(self, value):
        return self._authority.authenticate_log_page(value)
