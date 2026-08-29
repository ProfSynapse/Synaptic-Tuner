"""Shared authenticated-evidence validity and replay boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from .contracts import digest, safe_ref


_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def canonical_utc(value: object, name: str) -> str:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must use canonical whole-second UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{name} must use canonical whole-second UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{name} must use canonical whole-second UTC")
    return value


def parse_utc(value: str) -> datetime:
    canonical_utc(value, "timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class EvidenceFreshnessPolicyV1:
    maximum_age_seconds: int
    maximum_lifetime_seconds: int
    future_skew_seconds: int = 30

    def __post_init__(self) -> None:
        for name in (
            "maximum_age_seconds", "maximum_lifetime_seconds", "future_skew_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 3600:
                raise ValueError(f"{name} must be a bounded exact integer")
        if self.maximum_age_seconds == 0 or self.maximum_lifetime_seconds == 0:
            raise ValueError("evidence age and lifetime must be positive")


SOURCE_EVIDENCE_POLICY = EvidenceFreshnessPolicyV1(600, 600, 30)
SOURCE_EVIDENCE_PURPOSE = "source-lock-evidence/v1"
DEPLOYMENT_EVIDENCE_POLICY = EvidenceFreshnessPolicyV1(300, 300, 30)


def validate_evidence_window(
    *, verified_at: str, expires_at: str, now: str,
    policy: EvidenceFreshnessPolicyV1,
) -> None:
    verified = parse_utc(verified_at)
    expiry = parse_utc(expires_at)
    current = parse_utc(now)
    if verified > current + timedelta(seconds=policy.future_skew_seconds):
        raise ValueError("evidence is too far in the future")
    if expiry <= verified:
        raise ValueError("evidence expiry must follow verification")
    if expiry - verified > timedelta(seconds=policy.maximum_lifetime_seconds):
        raise ValueError("evidence lifetime exceeds policy")
    age = max(timedelta(0), current - verified)
    if age > timedelta(seconds=policy.maximum_age_seconds):
        raise ValueError("evidence is stale")
    if current >= expiry:
        raise ValueError("evidence is expired")


class ReplayDisposition(str, Enum):
    ADMITTED = "admitted"
    IDEMPOTENT = "idempotent"
    COLLISION = "collision"


@runtime_checkable
class EvidenceReplayRepository(Protocol):
    def admit(
        self, *, purpose: str, issuer_ref: str, evidence_ref: str,
        challenge_nonce: str, audience_ref: str, payload_digest: str,
        expires_at: str,
    ) -> ReplayDisposition: ...


@runtime_checkable
class EvidenceAuthenticator(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


def admit_evidence(
    repository: EvidenceReplayRepository, *, purpose: str, issuer_ref: str,
    evidence_ref: str, challenge_nonce: str, audience_ref: str,
    payload_digest: str, expires_at: str,
) -> None:
    if not isinstance(repository, EvidenceReplayRepository):
        raise TypeError("evidence replay repository is required")
    disposition = repository.admit(
        purpose=safe_ref(purpose, "purpose"), issuer_ref=safe_ref(issuer_ref, "issuer_ref"),
        evidence_ref=safe_ref(evidence_ref, "evidence_ref"),
        challenge_nonce=safe_ref(challenge_nonce, "challenge_nonce"),
        audience_ref=safe_ref(audience_ref, "audience_ref"),
        payload_digest=digest(payload_digest, "payload_digest"),
        expires_at=canonical_utc(expires_at, "expires_at"),
    )
    if disposition not in {ReplayDisposition.ADMITTED, ReplayDisposition.IDEMPOTENT}:
        raise ValueError("evidence replay collision")


__all__ = [
    "DEPLOYMENT_EVIDENCE_POLICY", "EvidenceAuthenticator", "EvidenceFreshnessPolicyV1",
    "EvidenceReplayRepository", "ReplayDisposition", "SOURCE_EVIDENCE_POLICY",
    "SOURCE_EVIDENCE_PURPOSE",
    "admit_evidence", "canonical_utc", "parse_utc", "validate_evidence_window",
]
