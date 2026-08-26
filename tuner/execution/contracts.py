"""Provider-neutral contracts for durable execution lifecycle coordination.

The engine owns lifecycle semantics but deliberately owns no database and no
provider mutation client.  Hosts inject a repository implementation and call
providers outside this boundary after receiving a durable effect claim.
"""

from __future__ import annotations

import hashlib
import json
import re
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable


MAX_LIST_LIMIT = 100
MAX_REF_LENGTH = 256
_SAFE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if (
        not value
        or len(value) > MAX_REF_LENGTH
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError(f"{name} must be bounded text without controls")
    return value


def safe_ref(value: str, name: str) -> str:
    value = required_text(value, name)
    if _SAFE_REF_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded safe reference")
    return value


def digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical SHA-256 digest")
    return value


def timestamp(value: str, name: str) -> str:
    value = required_text(value, name)
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class LifecyclePhase(str, Enum):
    PLANNED = "planned"
    READY = "ready"
    PREPARING = "preparing"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    RECONCILE_REQUIRED = "reconcile_required"


class VerificationStatus(str, Enum):
    NOT_READY = "not_ready"
    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


class EffectKind(str, Enum):
    SUBMIT = "submit"
    CANCEL = "cancel"


class EffectState(str, Enum):
    CLAIMED = "claimed"
    ATTEMPTED = "attempted"
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"


class EffectDisposition(str, Enum):
    FOUND = "found"
    DEFINITELY_ABSENT = "definitely_absent"
    INDETERMINATE = "indeterminate"


class ProviderRunPhase(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class EventCode(str, Enum):
    RUN_PLANNED = "run_planned"
    AUTHORITY_ACCEPTED = "authority_accepted"
    AUTHORIZATION_REJECTED = "authorization_rejected"
    PREPARATION_STARTED = "preparation_started"
    PREPARATION_COMPLETED = "preparation_completed"
    EFFECT_CLAIMED = "effect_claimed"
    EFFECT_ATTEMPTED = "effect_attempted"
    EFFECT_FOUND = "effect_found"
    EFFECT_DEFINITELY_ABSENT = "effect_definitely_absent"
    EFFECT_INDETERMINATE = "effect_indeterminate"
    PROVIDER_QUEUED = "provider_queued"
    PROVIDER_RUNNING = "provider_running"
    PROVIDER_SUCCEEDED = "provider_succeeded"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_CANCELLED = "provider_cancelled"
    PROVIDER_UNKNOWN = "provider_unknown"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_VERIFIED = "verification_verified"
    VERIFICATION_INVALID = "verification_invalid"
    VERIFICATION_INCONCLUSIVE = "verification_inconclusive"
    VERIFICATION_REOPENED = "verification_reopened"


class MessageCode(str, Enum):
    PLANNED = "planned"
    AUTHORITY_BOUND = "authority_bound"
    AUTHORIZATION_MISMATCH = "authorization_mismatch"
    PREPARING = "preparing"
    READY = "ready"
    EFFECT_DURABLY_CLAIMED = "effect_durably_claimed"
    EFFECT_MUTATION_ATTEMPTED = "effect_mutation_attempted"
    EFFECT_CONFIRMED = "effect_confirmed"
    EFFECT_ABSENT = "effect_absent"
    EFFECT_OUTCOME_UNKNOWN = "effect_outcome_unknown"
    PROVIDER_STATE_OBSERVED = "provider_state_observed"
    SEMANTIC_VERIFICATION_PENDING = "semantic_verification_pending"
    SEMANTIC_VERIFICATION_STARTED = "semantic_verification_started"
    SEMANTIC_VERIFICATION_PASSED = "semantic_verification_passed"
    SEMANTIC_VERIFICATION_FAILED = "semantic_verification_failed"
    SEMANTIC_VERIFICATION_INCONCLUSIVE = "semantic_verification_inconclusive"
    SEMANTIC_VERIFICATION_REOPENED = "semantic_verification_reopened"


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    provider: str
    account_ref: str
    namespace_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", safe_ref(self.provider.lower(), "provider"))
        object.__setattr__(self, "account_ref", safe_ref(self.account_ref, "account_ref"))
        object.__setattr__(self, "namespace_ref", safe_ref(self.namespace_ref, "namespace_ref"))


@dataclass(frozen=True, slots=True)
class EffectIdentity:
    effect_id: str
    effect_key: str
    kind: EffectKind
    scope: ExecutionScope

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", safe_ref(self.effect_id, "effect_id"))
        object.__setattr__(self, "effect_key", safe_ref(self.effect_key, "effect_key"))
        if not isinstance(self.kind, EffectKind):
            raise TypeError("kind must be EffectKind")
        if not isinstance(self.scope, ExecutionScope):
            raise TypeError("scope must be ExecutionScope")


@dataclass(frozen=True, slots=True)
class GrantBinding:
    """Non-secret, immutable description of authority granted by the host."""

    operation: object
    issued_at: str
    expires_at: str

    def __post_init__(self) -> None:
        from .operation import OperationBindingV1
        if not isinstance(self.operation, OperationBindingV1):
            raise TypeError("operation must be OperationBindingV1")
        issued_at = timestamp(self.issued_at, "issued_at")
        expires_at = timestamp(self.expires_at, "expires_at")
        if expires_at <= issued_at:
            raise ValueError("grant expiry must be after issue time")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)

    def __getattr__(self, name: str):
        aliases = {
            "grant_ref": "grant_ref", "project_ref": "project_ref",
            "operation_key": "effect", "effect_kind": "effect", "scope": "effect",
            "plan_fingerprint": "plan_fingerprint", "source_digest": "source_digest",
            "workload_digest": "workload_digest", "artifact_slot_ref": "artifact_slot_ref",
            "quote_digest": "quote_digest", "resource_digest": "resource_digest",
            "allowed_secret_refs_digest": "allowed_secret_refs_digest",
            "operation_binding_digest": "digest",
        }
        if name not in aliases:
            raise AttributeError(name)
        value = getattr(self.operation, aliases[name])
        if name == "operation_key":
            return value.effect_key
        if name == "effect_kind":
            return value.kind
        if name == "scope":
            return value.scope
        return value

    @classmethod
    def from_operation(cls, operation, *, issued_at: str, expires_at: str) -> "GrantBinding":
        return cls(operation=operation, issued_at=issued_at, expires_at=expires_at)

    @property
    def fingerprint(self) -> str:
        document = {"operation_binding": self.operation.to_dict(), "issued_at": self.issued_at,
                    "expires_at": self.expires_at}
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"synaptic.execution-grant/v2\0" + payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation.to_dict(),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> "GrantBinding":
        if not isinstance(value, dict) or set(value) != {
            "operation", "issued_at", "expires_at"
        }:
            raise ValueError("grant binding contains missing or unknown fields")
        from .operation import OperationBindingV1
        return cls(
            OperationBindingV1.from_dict(value["operation"]),
            value["issued_at"],
            value["expires_at"],
        )


@dataclass(frozen=True, slots=True)
class EffectRecord:
    identity: EffectIdentity
    grant_fingerprint: str
    state: EffectState = EffectState.CLAIMED
    provider_job_ref: str | None = None
    receipt_digest: str | None = None
    grant_ref: str | None = None
    command_digest: str | None = None
    canonical_command: bytes | None = None
    attempt_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EffectIdentity):
            raise TypeError("identity must be EffectIdentity")
        object.__setattr__(self, "grant_fingerprint", digest(self.grant_fingerprint, "grant_fingerprint"))
        if not isinstance(self.state, EffectState):
            raise TypeError("state must be EffectState")
        if self.state is EffectState.FOUND:
            if self.provider_job_ref is None or self.receipt_digest is None:
                raise ValueError("found effects require provider job and receipt references")
        elif self.provider_job_ref is not None or self.receipt_digest is not None:
            raise ValueError("only found effects may contain provider result references")
        if self.provider_job_ref is not None:
            object.__setattr__(self, "provider_job_ref", safe_ref(self.provider_job_ref, "provider_job_ref"))
        if self.receipt_digest is not None:
            object.__setattr__(self, "receipt_digest", digest(self.receipt_digest, "receipt_digest"))
        if self.grant_ref is not None:
            object.__setattr__(self, "grant_ref", safe_ref(self.grant_ref, "grant_ref"))
        if self.command_digest is not None:
            object.__setattr__(self, "command_digest", digest(self.command_digest, "command_digest"))
        if self.canonical_command is not None and not isinstance(self.canonical_command, bytes):
            raise TypeError("canonical_command must be bytes")
        if self.state is EffectState.ATTEMPTED and (
            self.grant_ref is None or self.command_digest is None
            or self.canonical_command is None or self.attempt_count != 1
        ):
            raise ValueError("attempted effects require one canonical durable attempt")

    def to_dict(self) -> dict[str, object]:
        scope = self.identity.scope
        command = (
            None
            if self.canonical_command is None
            else base64.b64encode(self.canonical_command).decode("ascii")
        )
        return {
            "identity": {
                "effect_id": self.identity.effect_id,
                "effect_key": self.identity.effect_key,
                "kind": self.identity.kind.value,
                "scope": {
                    "provider": scope.provider,
                    "account_ref": scope.account_ref,
                    "namespace_ref": scope.namespace_ref,
                },
            },
            "grant_fingerprint": self.grant_fingerprint,
            "state": self.state.value,
            "provider_job_ref": self.provider_job_ref,
            "receipt_digest": self.receipt_digest,
            "grant_ref": self.grant_ref,
            "command_digest": self.command_digest,
            "canonical_command_base64": command,
            "attempt_count": self.attempt_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EffectRecord":
        expected = {
            "identity", "grant_fingerprint", "state", "provider_job_ref",
            "receipt_digest", "grant_ref", "command_digest",
            "canonical_command_base64", "attempt_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("effect record contains missing or unknown fields")
        identity = value["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "effect_id", "effect_key", "kind", "scope"
        }:
            raise ValueError("effect identity is malformed")
        scope = identity["scope"]
        if not isinstance(scope, dict) or set(scope) != {
            "provider", "account_ref", "namespace_ref"
        }:
            raise ValueError("effect scope is malformed")
        encoded = value["canonical_command_base64"]
        if encoded is None:
            command = None
        elif isinstance(encoded, str) and encoded.isascii():
            try:
                command = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("canonical command Base64 is invalid") from exc
            if base64.b64encode(command).decode("ascii") != encoded:
                raise ValueError("canonical command Base64 is not canonical")
        else:
            raise ValueError("canonical command Base64 is invalid")
        return cls(
            identity=EffectIdentity(
                identity["effect_id"],
                identity["effect_key"],
                EffectKind(identity["kind"]),
                ExecutionScope(**scope),
            ),
            grant_fingerprint=value["grant_fingerprint"],
            state=EffectState(value["state"]),
            provider_job_ref=value["provider_job_ref"],
            receipt_digest=value["receipt_digest"],
            grant_ref=value["grant_ref"],
            command_digest=value["command_digest"],
            canonical_command=command,
            attempt_count=value["attempt_count"],
        )


@dataclass(frozen=True, slots=True)
class EffectObservation:
    identity: EffectIdentity
    disposition: EffectDisposition
    provider_job_ref: str | None = None
    receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EffectIdentity):
            raise TypeError("identity must be EffectIdentity")
        if not isinstance(self.disposition, EffectDisposition):
            raise TypeError("disposition must be EffectDisposition")
        if self.disposition is EffectDisposition.FOUND:
            if self.provider_job_ref is None or self.receipt_digest is None:
                raise ValueError("FOUND requires provider job and receipt references")
            object.__setattr__(self, "provider_job_ref", safe_ref(self.provider_job_ref, "provider_job_ref"))
            object.__setattr__(self, "receipt_digest", digest(self.receipt_digest, "receipt_digest"))
        elif self.provider_job_ref is not None or self.receipt_digest is not None:
            raise ValueError("only FOUND may contain provider result references")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    code: EventCode
    occurred_at: str
    message_code: MessageCode
    effect: EffectRecord | None = None
    grant_binding: GrantBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, EventCode):
            raise TypeError("code must be EventCode")
        if not isinstance(self.message_code, MessageCode):
            raise TypeError("message_code must be MessageCode")
        object.__setattr__(self, "occurred_at", timestamp(self.occurred_at, "occurred_at"))
        if self.effect is not None and not isinstance(self.effect, EffectRecord):
            raise TypeError("effect must be EffectRecord")
        if self.grant_binding is not None and not isinstance(self.grant_binding, GrantBinding):
            raise TypeError("grant_binding must be GrantBinding")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "occurred_at": self.occurred_at,
            "message_code": self.message_code.value,
            "effect": None if self.effect is None else self.effect.to_dict(),
            "grant_binding": (
                None if self.grant_binding is None else self.grant_binding.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "LifecycleEvent":
        if not isinstance(value, dict) or set(value) != {
            "code", "occurred_at", "message_code", "effect", "grant_binding"
        }:
            raise ValueError("lifecycle event contains missing or unknown fields")
        return cls(
            EventCode(value["code"]),
            value["occurred_at"],
            MessageCode(value["message_code"]),
            None if value["effect"] is None else EffectRecord.from_dict(value["effect"]),
            (
                None
                if value["grant_binding"] is None
                else GrantBinding.from_dict(value["grant_binding"])
            ),
        )


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    run_id: str
    project_ref: str
    revision: int
    phase: LifecyclePhase
    verification: VerificationStatus
    updated_at: str
    message_code: MessageCode
    events: tuple[LifecycleEvent, ...]
    effects: tuple[EffectRecord, ...] = ()
    grant_binding: GrantBinding | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", safe_ref(self.run_id, "run_id"))
        object.__setattr__(self, "project_ref", safe_ref(self.project_ref, "project_ref"))
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.phase, LifecyclePhase):
            raise TypeError("phase must be LifecyclePhase")
        if not isinstance(self.verification, VerificationStatus):
            raise TypeError("verification must be VerificationStatus")
        if not isinstance(self.message_code, MessageCode):
            raise TypeError("message_code must be MessageCode")
        object.__setattr__(self, "updated_at", timestamp(self.updated_at, "updated_at"))
        events = tuple(self.events)
        effects = tuple(self.effects)
        if not events or len(events) != self.revision:
            raise ValueError("revision must equal the durable event count")
        if any(not isinstance(event, LifecycleEvent) for event in events):
            raise TypeError("events must contain LifecycleEvent values")
        if any(not isinstance(effect, EffectRecord) for effect in effects):
            raise TypeError("effects must contain EffectRecord values")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "effects", effects)
        if self.grant_binding is not None and self.grant_binding.project_ref != self.project_ref:
            raise ValueError("grant binding project does not match run project")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "synaptic-lifecycle-record/v1",
            "run_id": self.run_id,
            "project_ref": self.project_ref,
            "revision": self.revision,
            "phase": self.phase.value,
            "verification": self.verification.value,
            "updated_at": self.updated_at,
            "message_code": self.message_code.value,
            "events": [event.to_dict() for event in self.events],
            "effects": [effect.to_dict() for effect in self.effects],
            "grant_binding": (
                None if self.grant_binding is None else self.grant_binding.to_dict()
            ),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: object) -> "LifecycleRecord":
        expected = {
            "schema_version", "run_id", "project_ref", "revision", "phase",
            "verification", "updated_at", "message_code", "events", "effects",
            "grant_binding",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("lifecycle record contains missing or unknown fields")
        if value["schema_version"] != "synaptic-lifecycle-record/v1":
            raise ValueError("unsupported lifecycle record schema")
        events = value["events"]
        effects = value["effects"]
        if not isinstance(events, list) or not isinstance(effects, list):
            raise ValueError("lifecycle record collections must be arrays")
        return cls(
            run_id=value["run_id"],
            project_ref=value["project_ref"],
            revision=value["revision"],
            phase=LifecyclePhase(value["phase"]),
            verification=VerificationStatus(value["verification"]),
            updated_at=value["updated_at"],
            message_code=MessageCode(value["message_code"]),
            events=tuple(LifecycleEvent.from_dict(item) for item in events),
            effects=tuple(EffectRecord.from_dict(item) for item in effects),
            grant_binding=(
                None
                if value["grant_binding"] is None
                else GrantBinding.from_dict(value["grant_binding"])
            ),
        )

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> "LifecycleRecord":
        if not isinstance(value, bytes) or not value or len(value) > 16 * 1024 * 1024:
            raise ValueError("lifecycle record must be bounded canonical JSON bytes")
        try:
            document = json.loads(value.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("lifecycle record must be canonical JSON") from exc
        result = cls.from_dict(document)
        if result.canonical_bytes != value:
            raise ValueError("lifecycle record is not canonical")
        return result


class AttemptDisposition(str, Enum):
    EXECUTE_NOW = "execute_now"
    LOOKUP_ONLY = "lookup_only"

@dataclass(frozen=True, slots=True)
class AttemptAdmission:
    record: LifecycleRecord
    effect: EffectRecord
    disposition: AttemptDisposition


@dataclass(frozen=True, slots=True)
class RunPage:
    items: tuple[LifecycleRecord, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if any(not isinstance(item, LifecycleRecord) for item in self.items):
            raise TypeError("items must contain LifecycleRecord values")
        if self.next_cursor is not None:
            object.__setattr__(self, "next_cursor", safe_ref(self.next_cursor, "next_cursor"))


class LifecycleError(RuntimeError):
    """Base error with no provider payload or credential-derived detail."""


class RunAlreadyExists(LifecycleError):
    pass


class RunNotFound(LifecycleError):
    pass


class RevisionConflict(LifecycleError):
    pass


class InvalidTransition(LifecycleError):
    pass


class EffectCollision(LifecycleError):
    pass


class AuthorizationMismatch(LifecycleError):
    pass


@runtime_checkable
class LifecycleRepository(Protocol):
    """Host persistence boundary; each write is required to be atomic."""

    def create(self, record: LifecycleRecord) -> LifecycleRecord: ...

    def load(self, project_ref: str, run_id: str) -> LifecycleRecord | None: ...

    def append(
        self,
        project_ref: str,
        run_id: str,
        *,
        expected_revision: int,
        event: LifecycleEvent,
    ) -> LifecycleRecord: ...

    def compare_and_consume_attempt(
        self, project_ref: str, run_id: str, *, expected_revision: int,
        grant_ref: str, canonical_command: object,
    ) -> AttemptAdmission: ...

    def record_attempt_outcome(
        self, project_ref: str, run_id: str, *, expected_revision: int,
        command_digest: str, observation: EffectObservation,
    ) -> LifecycleRecord: ...

    def list_runs(
        self,
        project_ref: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> RunPage: ...


@runtime_checkable
class ReconciliationAdapter(Protocol):
    """Read-only provider seam: mutation methods intentionally do not exist."""

    @property
    def provider(self) -> str: ...

    def lookup_effect(self, identity: EffectIdentity) -> EffectObservation: ...


__all__ = [
    "AttemptAdmission",
    "AttemptDisposition",
    "AuthorizationMismatch",
    "EffectCollision",
    "EffectDisposition",
    "EffectIdentity",
    "EffectKind",
    "EffectObservation",
    "EffectRecord",
    "EffectState",
    "EventCode",
    "ExecutionScope",
    "GrantBinding",
    "InvalidTransition",
    "LifecycleEvent",
    "LifecyclePhase",
    "LifecycleRecord",
    "LifecycleRepository",
    "MAX_LIST_LIMIT",
    "MessageCode",
    "ProviderRunPhase",
    "ReconciliationAdapter",
    "RevisionConflict",
    "RunAlreadyExists",
    "RunNotFound",
    "RunPage",
    "VerificationStatus",
    "timestamp",
]
