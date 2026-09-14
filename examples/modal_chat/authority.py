"""Strict same-process authorities and stores for the Modal chat example."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping

from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1,
    ProviderRunObservationContentV1,
)
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_retention import (
    ModalRetainedPreparation,
)


class ModalChatAuthorityError(RuntimeError):
    """Closed local composition failure."""


class UTCClock:
    def now(self) -> str:
        return self.now_iso()

    def now_iso(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    def now_epoch(self) -> int:
        return int(datetime.now(timezone.utc).timestamp())


class HMACAuthenticator:
    """Purpose-separated HMAC with explicit key references and rotation support."""

    def __init__(self, keys: Mapping[str, bytes], *, allowed_purposes: frozenset[str]):
        copied: dict[str, bytes] = {}
        for key_ref, key in keys.items():
            ref = safe_ref(key_ref, "key_ref")
            if type(key) is not bytes or len(key) < 32:
                raise ValueError("HMAC key must contain at least 32 bytes")
            copied[ref] = bytes(key)
        if (
            not copied
            or type(allowed_purposes) is not frozenset
            or not allowed_purposes
        ):
            raise ValueError("explicit HMAC keys and purposes are required")
        self._keys = MappingProxyType(copied)
        self._purposes = frozenset(safe_ref(x, "purpose") for x in allowed_purposes)

    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        if purpose not in self._purposes or type(payload) is not bytes:
            raise ModalChatAuthorityError("modal_chat_authority_invalid")
        key = self._keys.get(key_ref)
        if key is None:
            raise ModalChatAuthorityError("modal_chat_authority_invalid")
        return hmac.new(
            key, purpose.encode("ascii") + b"\0" + payload, hashlib.sha256
        ).digest()

    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool:
        try:
            return type(tag) is bytes and hmac.compare_digest(
                tag, self.sign(purpose, payload, key_ref)
            )
        except Exception:
            return False


class ReaderEvidenceAuthority:
    """Issue and authenticate exact coordinator observation and log envelopes."""

    def __init__(
        self, authority_ref: str, key_ref: str, authenticator: HMACAuthenticator
    ):
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
            content,
            self.authority_ref,
            self.key_ref,
            self._tag("provider-run-observation/v1", content.content_digest),
        )
        return AuthenticatedProviderRunObservationV1.parse(envelope.canonical_bytes)

    def log_page(self, content):
        if type(content) is not ProviderLogPageContentV1:
            raise TypeError("exact log page content required")
        envelope = AuthenticatedProviderLogPageV1(
            content,
            self.authority_ref,
            self.key_ref,
            self._tag("provider-log-page/v1", content.content_digest),
        )
        return AuthenticatedProviderLogPageV1.parse(envelope.canonical_bytes)

    def authenticate_observation(self, value) -> bool:
        try:
            if type(value) is not AuthenticatedProviderRunObservationV1:
                return False
            owned = AuthenticatedProviderRunObservationV1.parse(value.canonical_bytes)
            return (
                owned == value
                and owned.authority_ref == self.authority_ref
                and owned.key_ref == self.key_ref
                and hmac.compare_digest(
                    owned.tag,
                    self._tag(
                        "provider-run-observation/v1", owned.content.content_digest
                    ),
                )
            )
        except Exception:
            return False

    def authenticate_log_page(self, value) -> bool:
        try:
            if type(value) is not AuthenticatedProviderLogPageV1:
                return False
            owned = AuthenticatedProviderLogPageV1.parse(value.canonical_bytes)
            return (
                owned == value
                and owned.authority_ref == self.authority_ref
                and owned.key_ref == self.key_ref
                and hmac.compare_digest(
                    owned.tag,
                    self._tag("provider-log-page/v1", owned.content.content_digest),
                )
            )
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


class PlanningStore:
    def __init__(self):
        self._plans: dict[str, bytes] = {}
        self._contexts: dict[str, bytes] = {}
        self._lock = RLock()

    @staticmethod
    def _plan(value: TrainingPlan) -> TrainingPlan:
        if type(value) is not TrainingPlan:
            raise ModalChatAuthorityError("modal_chat_planning_invalid")
        return TrainingPlan.from_dict(value.to_dict())

    @staticmethod
    def _context(value: ProviderPlanContextV1) -> ProviderPlanContextV1:
        if type(value) is not ProviderPlanContextV1:
            raise ModalChatAuthorityError("modal_chat_planning_invalid")
        return ProviderPlanContextV1.from_dict(value.to_dict())

    def _put(self, values, key, payload):
        with self._lock:
            prior = values.get(key)
            if prior is None:
                values[key] = payload
                return True
            if prior == payload:
                return False
            raise ModalChatAuthorityError("modal_chat_planning_conflict")

    def put_plan_if_absent(self, value):
        owned = self._plan(value)
        return self._put(
            self._plans, owned.plan_fingerprint, canonical_bytes(owned.to_dict())
        )

    def get_plan(self, key):
        with self._lock:
            raw = self._plans.get(key)
        return None if raw is None else TrainingPlan.from_dict(json.loads(raw))

    def put_context_if_absent(self, value):
        owned = self._context(value)
        return self._put(
            self._contexts,
            owned.provider_context_digest,
            canonical_bytes(owned.to_dict()),
        )

    def get_context(self, key):
        with self._lock:
            raw = self._contexts.get(key)
        return None if raw is None else ProviderPlanContextV1.from_dict(json.loads(raw))


class CanonicalCatalog:
    def __init__(self, expected_type: type, rebuild: Callable[[object], object]):
        self._type, self._rebuild = expected_type, rebuild
        self._values: dict[str, tuple[bytes, object]] = {}
        self._lock = RLock()

    def _owned(self, value):
        if type(value) is not self._type:
            raise ModalChatAuthorityError("modal_chat_catalog_invalid")
        rebuilt = self._rebuild(value)
        if type(rebuilt) is not self._type or rebuilt != value:
            raise ModalChatAuthorityError("modal_chat_catalog_invalid")
        raw = rebuilt.canonical_bytes
        if type(raw) is not bytes:
            raise ModalChatAuthorityError("modal_chat_catalog_invalid")
        return raw, rebuilt

    def publish_if_absent(self, key: str, value) -> bool:
        key = safe_ref(key, "catalog_key")
        raw, owned = self._owned(value)
        with self._lock:
            prior = self._values.get(key)
            if prior is None:
                self._values[key] = (raw, owned)
                return True
            if prior[0] == raw:
                return False
            raise ModalChatAuthorityError("modal_chat_catalog_conflict")

    def resolve(self, key: str):
        key = safe_ref(key, "catalog_key")
        with self._lock:
            prior = self._values.get(key)
        if prior is None:
            return None
        raw, value = self._owned(prior[1])
        if raw != prior[0]:
            raise ModalChatAuthorityError("modal_chat_catalog_invalid")
        return value


class RetainedBindingAuthority:
    def __init__(self, catalog: CanonicalCatalog):
        self._catalog = catalog

    def authenticate(self, value) -> bool:
        try:
            if type(value) is not ModalCommandBinding:
                return False
            retained = self._catalog.resolve(value.command_digest)
            return type(retained) is ModalCommandBinding and hmac.compare_digest(
                retained.canonical_bytes, value.canonical_bytes
            )
        except Exception:
            return False


class RetainedInputs:
    def __init__(self, preparation_digest: str, value: ModalRetainedPreparation):
        digest_text(preparation_digest, "preparation_digest")
        if type(value) is not ModalRetainedPreparation:
            raise TypeError("exact retained preparation required")
        owned = ModalRetainedPreparation(
            **{name: getattr(value, name) for name in value.__dataclass_fields__}
        )
        self._digest = preparation_digest
        self._value = owned

    def resolve(self, preparation_digest: str):
        return self._value if preparation_digest == self._digest else None


class FoundationAuthenticator:
    def __init__(self, grants, receipts, invalid_evidence):
        self._grants, self._receipts, self._invalid = grants, receipts, invalid_evidence

    def authenticate_grant(self, grant, command_bytes):
        return self._grants.authenticate(grant, command_bytes)

    def authenticate_receipt(self, receipt):
        return self._receipts.verify(receipt)

    def authenticate_invalid_evidence(self, evidence):
        return self._invalid.verify(evidence)


class BoundedAuthorization:
    def __init__(
        self,
        plan: TrainingPlan,
        preflight: TrainingPreflight,
        preparation: CanonicalPreparationV2,
        grants,
        clock,
        *,
        maximum_grant_seconds: int = 900,
    ):
        if (
            type(plan) is not TrainingPlan
            or type(preflight) is not TrainingPreflight
            or type(preparation) is not CanonicalPreparationV2
            or preparation.plan_fingerprint != plan.plan_fingerprint
            or not preflight.ready
            or not preflight.binds(plan)
        ):
            raise ValueError("exact ready plan and preflight required")
        if (
            type(maximum_grant_seconds) is not int
            or not 1 <= maximum_grant_seconds <= 3600
        ):
            raise ValueError("grant bound invalid")
        self._plan, self._preflight = plan, preflight
        self._preparation = CanonicalPreparationV2.parse(preparation.canonical_bytes)
        self._grants, self._clock = grants, clock
        self._authority_state = (grants.epoch, grants.revocation_generation)
        self._maximum = maximum_grant_seconds
        self._expires = int(
            datetime.fromisoformat(
                preflight.expires_at.replace("Z", "+00:00")
            ).timestamp()
        )
        self._policy = domain_digest(
            "synaptic-modal-chat-preflight/v1", canonical_bytes(preflight.to_dict())
        )
        self._requirements = domain_digest(
            "synaptic-modal-chat-requirements/v1",
            canonical_bytes(
                {"authorization": [x.to_dict() for x in preflight.authorization]}
            ),
        )
        self._committed = False
        self._commands: dict[str, bytes] = {}
        self._lock = RLock()

    def commit_preflight(self, plan, preflight):
        if (
            plan != self._plan
            or preflight != self._preflight
            or self._clock.now_epoch() >= self._expires
        ):
            raise ModalChatAuthorityError("modal_chat_authorization_invalid")
        self._committed = True
        return self._policy

    def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
        if (
            not self._committed
            or type(command_bytes) is not bytes
            or preflight_digest != self._policy
            or now_epoch != self._clock.now_epoch()
            or (self._grants.epoch, self._grants.revocation_generation)
            != self._authority_state
        ):
            raise ModalChatAuthorityError("modal_chat_authorization_invalid")
        command = parse_exact_command(command_bytes)
        kind = command.operation.effect.kind.value
        if (
            kind not in {"stage", "submit"}
            or command.preparation != self._preparation
            or now_epoch >= self._expires
        ):
            raise ModalChatAuthorityError("modal_chat_authorization_invalid")
        with self._lock:
            prior = self._commands.get(kind)
            if prior is not None and prior != command.canonical_bytes:
                raise ModalChatAuthorityError("modal_chat_authorization_invalid")
            self._commands[kind] = command.canonical_bytes
        return self._grants.issue(
            command.canonical_bytes,
            grant_ref=f"modal-{kind}-{command.digest[:24]}",
            policy_digest=self._policy,
            requirement_digest=self._requirements,
            not_before_epoch=now_epoch,
            expires_at_epoch=min(self._expires, now_epoch + self._maximum),
        )

    def issue_reconciliation_grant(self, *args, **kwargs):
        raise ModalChatAuthorityError("modal_chat_reconciliation_unsupported")


class UnavailableRecoveryVerifier:
    def verify_quiescence(self, *args, **kwargs):
        return False

    def verify_finality(self, *args, **kwargs):
        return False


class UnavailableQuiescenceEvidence:
    def obtain(self, *args, **kwargs):
        raise ModalChatAuthorityError("modal_chat_quiescence_unavailable")


__all__ = [
    "BoundedAuthorization",
    "CanonicalCatalog",
    "FoundationAuthenticator",
    "HMACAuthenticator",
    "LogAuthenticator",
    "ModalChatAuthorityError",
    "ObservationAuthenticator",
    "PlanningStore",
    "ReaderEvidenceAuthority",
    "RetainedBindingAuthority",
    "RetainedInputs",
    "UTCClock",
    "UnavailableQuiescenceEvidence",
    "UnavailableRecoveryVerifier",
]
