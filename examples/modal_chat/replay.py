"""Durable source/deployment replay admission in the consumer's existing catalog."""

from __future__ import annotations

import hashlib

from tuner.execution.contracts import digest, safe_ref
from tuner.execution.evidence import ReplayDisposition, canonical_utc
from tuner.execution.foundation_v2.canonical import canonical_bytes

from .storage import ModalChatStorage, ModalChatStorageError


class ModalChatEvidenceReplay:
    """Bind each purpose/challenge pair to one exact evidence record forever.

    Authentication and freshness checks belong to the existing source finalizer.
    Admission does not authenticate evidence, expire claims, or permit reuse of
    an old challenge with a new audience, issuer, payload, or expiry.
    """

    def __init__(self, storage: ModalChatStorage) -> None:
        if type(storage) is not ModalChatStorage:
            raise TypeError("exact consumer storage required")
        self._catalog = storage.catalog(
            "source-evidence-replay", encode=bytes, decode=bytes
        )

    def admit(
        self,
        *,
        purpose: str,
        issuer_ref: str,
        evidence_ref: str,
        challenge_nonce: str,
        audience_ref: str,
        payload_digest: str,
        expires_at: str,
    ) -> ReplayDisposition:
        record = {
            "purpose": safe_ref(purpose, "purpose"),
            "issuer_ref": safe_ref(issuer_ref, "issuer_ref"),
            "evidence_ref": safe_ref(evidence_ref, "evidence_ref"),
            "challenge_nonce": safe_ref(challenge_nonce, "challenge_nonce"),
            "audience_ref": safe_ref(audience_ref, "audience_ref"),
            "payload_digest": digest(payload_digest, "payload_digest"),
            "expires_at": canonical_utc(expires_at, "expires_at"),
        }
        key = hashlib.sha256(
            canonical_bytes({"purpose": purpose, "challenge_nonce": challenge_nonce})
        ).hexdigest()
        payload = canonical_bytes(record)
        previous = self._catalog.resolve(key)
        if previous is not None:
            return (
                ReplayDisposition.IDEMPOTENT
                if previous == payload
                else ReplayDisposition.COLLISION
            )
        inserted = self._catalog.publish_if_absent(key, payload)
        if type(inserted) is not bool or self._catalog.resolve(key) != payload:
            raise ModalChatStorageError("modal_chat_storage_conflict")
        return ReplayDisposition.ADMITTED if inserted else ReplayDisposition.IDEMPOTENT


__all__ = ["ModalChatEvidenceReplay"]
