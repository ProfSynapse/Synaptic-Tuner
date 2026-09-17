"""Engine-owned authorities for the reference host, keyed from one named secret.

Location: ``synaptic_tuner/api/v1/reference/authority.py``.

The coordinator trusts several HMAC authorities (execution grants, receipts,
invalid evidence, record assessments, list cursors, publication evidence). A
host names one secret (``SecretRef``); ``compose_reference_authority``
resolves it once through ``SecretResolverPort`` at composition time and
derives one independent key per purpose with HMAC-SHA256 over a fixed domain
string. The resolved value is never stored, logged or placed in a dataclass
field; each authority object holds only its derived key and redacts it from
``repr``.

Also here: ``ReferenceClockV1`` (the coordinator's ``now``/``now_iso``/
``now_epoch`` over the public two-method ``ClockPort``),
``ReferenceFoundationAuthenticatorV1`` (grant, receipt and invalid-evidence
authentication for the workflow replay) and the two unavailable proof
implementations composed when a family brings no quiescence evidence.

Consumed by ``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac

from synaptic_tuner.api.v1._timestamps import require_rfc3339
from synaptic_tuner.api.v1.ports import ClockPort, SecretResolverPort
from synaptic_tuner.api.v1.secrets import SecretRef
from tuner.execution.coordinator_v1.cursors import HMACCursorAuthorityV1
from tuner.execution.coordinator_v1.foundation import FoundationRecordAssessmentAuthorityV1
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import safe_ref
from tuner.execution.foundation_v2.receipts import InvalidEvidenceAuthorityV2, ReceiptAuthorityV2

from .provider_family import require_methods


MINIMUM_SECRET_CHARACTERS = 32
_KEY_DOMAIN = b"synaptic-reference-authority-key/v1\0"


class ReferenceCompositionError(RuntimeError):
    """Closed reference composition failure."""


class ReferenceClockV1:
    """Coordinator clock over the public ``ClockPort``."""

    __slots__ = ("_port",)

    def __init__(self, port: ClockPort) -> None:
        require_methods(port, "now", "now_epoch")
        self._port = port

    def now(self) -> str:
        return require_rfc3339(self._port.now(), "now")

    def now_iso(self) -> str:
        return self.now()

    def now_epoch(self) -> int:
        value = self._port.now_epoch()
        if type(value) is not int or value < 0:
            raise ReferenceCompositionError("clock epoch invalid")
        return value


class ReferenceFoundationAuthenticatorV1:
    """``FoundationEvidenceAuthenticatorPortV1`` over the three foundation authorities."""

    __slots__ = ("_grants", "_receipts", "_invalid")

    def __init__(self, grants, receipts, invalid_evidence) -> None:
        self._grants, self._receipts, self._invalid = grants, receipts, invalid_evidence

    def authenticate_grant(self, grant, command_bytes) -> bool:
        return self._grants.authenticate(grant, command_bytes) is True

    def authenticate_receipt(self, receipt) -> bool:
        return self._receipts.verify(receipt) is True

    def authenticate_invalid_evidence(self, evidence) -> bool:
        return self._invalid.verify(evidence) is True


class UnavailableRecoveryVerifierV1:
    """Refuses every quiescence or finality proof."""

    def verify_quiescence(self, *args, **kwargs) -> bool:
        return False

    def verify_finality(self, *args, **kwargs) -> bool:
        return False


class UnavailableQuiescenceEvidenceV1:
    """No trusted quiescence evidence: orphan recovery is refused, never guessed."""

    def obtain(self, *args, **kwargs):
        raise ReferenceCompositionError("quiescence evidence unavailable")


class ReferencePublicationEvidenceAuthorityV1:
    """``EvidenceAuthorityPortV1`` for publication descriptors and sources."""

    __slots__ = ("key_ref", "_key")

    def __init__(self, key_ref: str, key: bytes) -> None:
        self.key_ref = safe_ref(key_ref, "key_ref")
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("publication evidence key too short")
        self._key = bytes(key)

    def __repr__(self) -> str:
        return f"ReferencePublicationEvidenceAuthorityV1(key_ref={self.key_ref!r}, key=<redacted>)"

    def sign(self, purpose: str, payload: bytes) -> str:
        safe_ref(purpose, "purpose")
        if type(payload) is not bytes:
            raise TypeError("payload must be bytes")
        return hmac.new(self._key, purpose.encode("ascii") + b"\0" + payload, hashlib.sha256).hexdigest()

    def verify(self, purpose: str, payload: bytes, tag: str, key_ref: str) -> bool:
        try:
            return key_ref == self.key_ref and type(tag) is str and hmac.compare_digest(
                tag, self.sign(purpose, payload)
            )
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class ReferenceAuthorityV1:
    """Every engine-owned authority the reference composition trusts."""

    authority_ref: str
    clock: ReferenceClockV1
    grant_authority: GrantAuthorityV2
    receipt_authority: ReceiptAuthorityV2
    invalid_evidence_authority: InvalidEvidenceAuthorityV2
    assessment_authority: FoundationRecordAssessmentAuthorityV1
    foundation_authenticator: ReferenceFoundationAuthenticatorV1
    cursor_authority: HMACCursorAuthorityV1
    publication_authority: ReferencePublicationEvidenceAuthorityV1


def derive_authority_key(master: bytes, purpose: str) -> bytes:
    """One 32-byte key per purpose from the resolved master secret."""
    if type(master) is not bytes or len(master) < MINIMUM_SECRET_CHARACTERS:
        raise ReferenceCompositionError("authority secret too short")
    return hmac.new(master, _KEY_DOMAIN + safe_ref(purpose, "purpose").encode("ascii"), hashlib.sha256).digest()


def compose_reference_authority(
    *,
    clock: ClockPort,
    secrets: SecretResolverPort,
    authority_secret: SecretRef,
    authority_ref: str = "synaptic-reference",
) -> ReferenceAuthorityV1:
    """Resolve the named secret once and derive every authority from it."""
    if type(authority_secret) is not SecretRef:
        raise TypeError("authority_secret must be exact SecretRef")
    require_methods(secrets, "resolve")
    authority_ref = safe_ref(authority_ref, "authority_ref")
    reference_clock = ReferenceClockV1(clock)
    resolved = secrets.resolve(authority_secret)
    if type(resolved) is not str or len(resolved) < MINIMUM_SECRET_CHARACTERS:
        raise ReferenceCompositionError("authority secret too short")
    master = resolved.encode("utf-8")
    del resolved
    grants = GrantAuthorityV2(f"{authority_ref}-grants", derive_authority_key(master, "grants"))
    receipts = ReceiptAuthorityV2(f"{authority_ref}-receipts", derive_authority_key(master, "receipts"))
    invalid = InvalidEvidenceAuthorityV2(
        f"{authority_ref}-invalid-evidence", derive_authority_key(master, "invalid-evidence")
    )
    assessments = FoundationRecordAssessmentAuthorityV1(
        f"{authority_ref}-assessments",
        f"{authority_ref}-assessment-key",
        derive_authority_key(master, "assessments"),
        assessor_ref=f"{authority_ref}-assessor",
        assessor_version="1.0.0",
        clock=reference_clock,
        receipt_authority=receipts,
        invalid_evidence_authority=invalid,
        grant_authority=grants,
    )
    cursors = HMACCursorAuthorityV1(
        f"{authority_ref}-cursors", {1: derive_authority_key(master, "cursors")}, active_generation=1
    )
    publication = ReferencePublicationEvidenceAuthorityV1(
        f"{authority_ref}-publication-key", derive_authority_key(master, "publication")
    )
    return ReferenceAuthorityV1(
        authority_ref,
        reference_clock,
        grants,
        receipts,
        invalid,
        assessments,
        ReferenceFoundationAuthenticatorV1(grants, receipts, invalid),
        cursors,
        publication,
    )


__all__ = [
    "MINIMUM_SECRET_CHARACTERS",
    "ReferenceAuthorityV1",
    "ReferenceClockV1",
    "ReferenceCompositionError",
    "ReferenceFoundationAuthenticatorV1",
    "ReferencePublicationEvidenceAuthorityV1",
    "UnavailableQuiescenceEvidenceV1",
    "UnavailableRecoveryVerifierV1",
    "compose_reference_authority",
    "derive_authority_key",
]
