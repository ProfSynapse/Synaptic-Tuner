"""Chat session and turn facade (contract only).

Location: ``synaptic_tuner/api/v1/chat_facade.py``.

A chat session is **not** a run. ``ChatSessionRef{session_id, project_ref}`` is a
first-class entity outside the lifecycle model (architecture §7.3 Q1): a
session has no paid provider submission to reconcile and a 10,000-turn session
would make one lifecycle event per turn quadratic. A session is either
*model-first* (a canonical local model directory or a Hub ref pinned to a
40-hex commit, ``ChatModelSource``) or *run-first* (a ``TrainingRunRef`` whose
verified artifacts the runtime serves); it binds a ``ChatModelIdentity`` once
the runtime is ready and a bounded ``ChatSessionPolicyV1``.

A turn is ``ChatTurnRef{session, request_id}`` with ``request_id`` a strictly
consecutive positive integer starting at 1 (§7.3 Q2). ``turn()`` returns a
completed ``ChatTurn{ref, state, content_digest, content, usage}``; token
streaming later binds ``chat_token`` observations to the **same** ref through
the shared observation stream, so nothing here changes when it lands. ``usage``
is present only when the runtime measured it and is absent, never null,
otherwise (§9).

Closed vocabularies: ``ChatSessionState`` (``opening, ready, serving, closing,
closed, cleanup_unresolved``), ``ChatModelKind`` (``full, lora``) and
``ChatOperationCode``. ``cleanup_unresolved`` is the public shape of an
owned-process family whose teardown did not converge (§7.3 Q5): such a session
is terminal-uncertain and serves no further turn. A non-Linux host receives
``host_unsupported`` from the local runtime arm (§7.3 Q4). Chat has ``close``,
not cancel. The ``synaptic-modal-chat-channel/v1`` frames are provider-local and
are not this vocabulary (§7.3 Q3).

Every type is a frozen slotted dataclass validating in ``__post_init__`` and
round-tripping through ``exact_fields``. ``ChatAPI`` applies the
``RunsAPI._call`` discipline verbatim. Runtime selection is host-supplied by
name in ``ChatOpenRequest.runtime``; nothing here reads the environment.

Registered in the lazy export table of ``synaptic_tuner/api/v1/__init__.py``
and in both import-closure gates. Imports nothing from ``tuner.*``; the
reference implementation over ``tuner/inference/`` lives in ``api/v1/reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
from typing import Protocol

from ._contract import digest_text, exact_fields, exact_integer, required_text
from .observations import (
    ObservationFamily,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from .results import TrainingRunRef
from .usage import UsageAvailability, UsageRecordV1


CHAT_SESSION_SCHEMA_VERSION = "synaptic-chat-session/v1"
CHAT_TURN_SCHEMA_VERSION = "synaptic-chat-turn/v1"

CHAT_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
CHAT_MAX_TURNS = 10_000
CHAT_MAX_HISTORY_BYTES = 64 * 1024 * 1024

_MAX_LIST_LIMIT = 100
_HEX_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HUB_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_HUB_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


# --- exact-value helpers ------------------------------------------------------


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _ascii_cursor(value: str | None, name: str = "cursor") -> str | None:
    if value is None:
        return None
    value = _text(value, name)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be ASCII") from None
    if len(encoded) > 256:
        raise ValueError(f"{name} exceeds 256 bytes")
    return value


def _parse_enum(enum_type: type, value: object, name: str):
    value = _text(value, name)
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"unknown {name}") from None


def _revision(value: object, name: str) -> str:
    value = _text(value, name)
    if _HEX_REVISION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 40- or 64-hex lowercase revision")
    return value


def _utf8_bytes(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("content must be valid Unicode text") from None


# --- identities and closed vocabularies ----------------------------------------


class ChatSessionState(str, Enum):
    OPENING = "opening"
    READY = "ready"
    SERVING = "serving"
    CLOSING = "closing"
    CLOSED = "closed"
    CLEANUP_UNRESOLVED = "cleanup_unresolved"


CHAT_TERMINAL_STATES = frozenset({ChatSessionState.CLOSED, ChatSessionState.CLEANUP_UNRESOLVED})
_TURN_STATES = frozenset(ChatSessionState) - {ChatSessionState.OPENING, ChatSessionState.SERVING}


class ChatModelKind(str, Enum):
    FULL = "full"
    LORA = "lora"


class ChatOperationCode(str, Enum):
    SESSION_MISSING = "session_missing"
    SESSION_CLOSED = "session_closed"
    SESSION_BUSY = "session_busy"
    TURN_BOUNDS_INVALID = "turn_bounds_invalid"
    MODEL_INELIGIBLE = "model_ineligible"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    CLEANUP_UNRESOLVED = "cleanup_unresolved"
    HOST_UNSUPPORTED = "host_unsupported"
    INTEGRITY_ERROR = "integrity_error"


class ChatOperationError(ValueError):
    def __init__(self, code: ChatOperationCode) -> None:
        if type(code) is not ChatOperationCode:
            raise TypeError("code must be exact ChatOperationCode")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class ChatSessionRef:
    """Identity of one chat session; never a run and never interchangeable with one."""

    session_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"session_id": self.session_id, "project_ref": self.project_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatSessionRef":
        value = exact_fields(value, frozenset({"session_id", "project_ref"}), "chat_session_ref")
        return cls(value["session_id"], value["project_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatTurnRef:
    """One turn of one session. ``request_id`` counts admitted turns from 1 upward.

    Consecutiveness is a property of the sequence an implementation issues:
    the first turn of a session is 1 and every later turn is exactly one more
    than the previous one. ``follows`` states that rule so conformance tests
    and streaming consumers share one definition.
    """

    session: ChatSessionRef
    request_id: int

    def __post_init__(self) -> None:
        if type(self.session) is not ChatSessionRef:
            raise TypeError("session must be exact ChatSessionRef")
        object.__setattr__(self, "session", ChatSessionRef.from_dict(self.session.to_dict()))
        _bounded_integer(self.request_id, "request_id", minimum=1, maximum=CHAT_MAX_TURNS)

    def follows(self, previous: "ChatTurnRef | None") -> bool:
        if previous is None:
            return self.request_id == 1
        if type(previous) is not ChatTurnRef:
            raise TypeError("previous must be exact ChatTurnRef or None")
        return previous.session == self.session and self.request_id == previous.request_id + 1

    def to_dict(self) -> dict[str, object]:
        return {"session": self.session.to_dict(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTurnRef":
        value = exact_fields(value, frozenset({"session", "request_id"}), "chat_turn_ref")
        return cls(ChatSessionRef.from_dict(value["session"]), value["request_id"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatModelIdentity:
    """The served model as a consistency projection: pinned revisions, closed kind."""

    model_ref: str
    model_revision: str
    tokenizer_revision: str
    model_kind: ChatModelKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_ref", _text(self.model_ref, "model_ref"))
        object.__setattr__(self, "model_revision", _revision(self.model_revision, "model_revision"))
        object.__setattr__(self, "tokenizer_revision", _revision(self.tokenizer_revision, "tokenizer_revision"))
        if type(self.model_kind) is not ChatModelKind:
            raise TypeError("model_kind must be exact ChatModelKind")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_ref": self.model_ref,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "model_kind": self.model_kind.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatModelIdentity":
        value = exact_fields(
            value, frozenset({"model_ref", "model_revision", "tokenizer_revision", "model_kind"}), "chat_model",
        )
        return cls(
            value["model_ref"],  # type: ignore[arg-type]
            value["model_revision"],  # type: ignore[arg-type]
            value["tokenizer_revision"],  # type: ignore[arg-type]
            _parse_enum(ChatModelKind, value["model_kind"], "model_kind"),
        )


# --- policy, model source, open request -------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatSessionPolicyV1:
    """Bounds of one session; a value outside its bound is ``turn_bounds_invalid``.

    Timeouts are finite positive seconds of at most one day, ``max_turns`` is
    1..10000 and ``max_history_bytes`` is 1..64 MiB, matching the runtime's own
    ``ChatSessionPolicy`` so the contract can never admit a session the runtime
    refuses.
    """

    request_timeout_seconds: float
    idle_timeout_seconds: float
    absolute_lifetime_seconds: float
    max_turns: int
    max_history_bytes: int

    def __post_init__(self) -> None:
        for name in ("request_timeout_seconds", "idle_timeout_seconds", "absolute_lifetime_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or not 0 < value <= CHAT_MAX_TIMEOUT_SECONDS:
                raise ChatOperationError(ChatOperationCode.TURN_BOUNDS_INVALID)
        for name, maximum in (("max_turns", CHAT_MAX_TURNS), ("max_history_bytes", CHAT_MAX_HISTORY_BYTES)):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= maximum:
                raise ChatOperationError(ChatOperationCode.TURN_BOUNDS_INVALID)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_timeout_seconds": self.request_timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "absolute_lifetime_seconds": self.absolute_lifetime_seconds,
            "max_turns": self.max_turns,
            "max_history_bytes": self.max_history_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatSessionPolicyV1":
        value = exact_fields(
            value,
            frozenset({
                "request_timeout_seconds", "idle_timeout_seconds", "absolute_lifetime_seconds",
                "max_turns", "max_history_bytes",
            }),
            "chat_session_policy",
        )
        return cls(
            value["request_timeout_seconds"],  # type: ignore[arg-type]
            value["idle_timeout_seconds"],  # type: ignore[arg-type]
            value["absolute_lifetime_seconds"],  # type: ignore[arg-type]
            value["max_turns"],  # type: ignore[arg-type]
            value["max_history_bytes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ChatModelSource:
    """Model-first selection: a canonical local directory or a Hub ref pinned to a commit.

    A local directory is an absolute path and carries a host-asserted 64-hex
    content digest as ``model_revision``; a Hub ref is ``owner/name`` pinned to
    a 40-hex commit. ``tokenizer_revision`` defaults to ``model_revision``. A
    ``lora`` kind names the adapter's canonical local directory; a ``full`` kind
    never does. Whether the directory exists and is canonical is checked by the
    runtime at open time (``model_ineligible``), not here.
    """

    model_ref: str
    model_revision: str
    tokenizer_revision: str | None = None
    model_kind: ChatModelKind = ChatModelKind.FULL
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        model_ref = _text(self.model_ref, "model_ref")
        revision = _revision(self.model_revision, "model_revision")
        if model_ref.startswith("/"):
            if _LOCAL_DIGEST.fullmatch(revision) is None:
                raise ValueError("a local model directory requires a 64-hex model_revision digest")
        elif _HUB_REF.fullmatch(model_ref) is None:
            raise ValueError("model_ref must be an absolute directory or an owner/name Hub ref")
        elif _HUB_COMMIT.fullmatch(revision) is None:
            raise ValueError("a Hub model requires a 40-hex commit as model_revision")
        if self.tokenizer_revision is not None:
            _revision(self.tokenizer_revision, "tokenizer_revision")
        if type(self.model_kind) is not ChatModelKind:
            raise TypeError("model_kind must be exact ChatModelKind")
        adapter = _optional_text(self.adapter_path, "adapter_path")
        if adapter is not None and not adapter.startswith("/"):
            raise ValueError("adapter_path must be an absolute local directory")
        if (self.model_kind is ChatModelKind.LORA) != (adapter is not None):
            raise ValueError("a lora model names exactly one adapter_path; a full model names none")

    def identity(self) -> ChatModelIdentity:
        return ChatModelIdentity(
            self.model_ref,
            self.model_revision,
            self.model_revision if self.tokenizer_revision is None else self.tokenizer_revision,
            self.model_kind,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_ref": self.model_ref,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "model_kind": self.model_kind.value,
            "adapter_path": self.adapter_path,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatModelSource":
        value = exact_fields(
            value,
            frozenset({"model_ref", "model_revision", "tokenizer_revision", "model_kind", "adapter_path"}),
            "chat_model_source",
        )
        return cls(
            value["model_ref"],  # type: ignore[arg-type]
            value["model_revision"],  # type: ignore[arg-type]
            value["tokenizer_revision"],  # type: ignore[arg-type]
            _parse_enum(ChatModelKind, value["model_kind"], "model_kind"),
            value["adapter_path"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ChatOpenRequest:
    """Open one session: exactly one of ``model`` (model-first) or ``run`` (run-first).

    ``runtime`` names a host-composed runtime; an unknown name is
    ``runtime_unavailable``. Nothing about the runtime is ambient: GPU, ports,
    interpreter and credentials all live behind the host's runtime port.
    """

    project_ref: str
    runtime: str
    policy: ChatSessionPolicyV1
    model: ChatModelSource | None = None
    run: TrainingRunRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "runtime", _text(self.runtime, "runtime"))
        if type(self.policy) is not ChatSessionPolicyV1:
            raise TypeError("policy must be exact ChatSessionPolicyV1")
        object.__setattr__(self, "policy", ChatSessionPolicyV1.from_dict(self.policy.to_dict()))
        if (self.model is None) == (self.run is None):
            raise ValueError("exactly one of model or run selects the session's model")
        if self.model is not None:
            if type(self.model) is not ChatModelSource:
                raise TypeError("model must be exact ChatModelSource")
            object.__setattr__(self, "model", ChatModelSource.from_dict(self.model.to_dict()))
        if self.run is not None:
            if type(self.run) is not TrainingRunRef:
                raise TypeError("run must be exact TrainingRunRef")
            run = TrainingRunRef.from_dict(self.run.to_dict())
            if run.project_ref != self.project_ref:
                raise ValueError("run project does not match the open request")
            object.__setattr__(self, "run", run)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_ref": self.project_ref,
            "runtime": self.runtime,
            "policy": self.policy.to_dict(),
            "model": None if self.model is None else self.model.to_dict(),
            "run": None if self.run is None else self.run.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatOpenRequest":
        value = exact_fields(value, frozenset({"project_ref", "runtime", "policy", "model", "run"}), "chat_open_request")
        model = value["model"]
        run = value["run"]
        return cls(
            value["project_ref"],  # type: ignore[arg-type]
            value["runtime"],  # type: ignore[arg-type]
            ChatSessionPolicyV1.from_dict(value["policy"]),  # type: ignore[arg-type]
            None if model is None else ChatModelSource.from_dict(model),  # type: ignore[arg-type]
            None if run is None else TrainingRunRef.from_dict(run),  # type: ignore[arg-type]
        )


# --- session record --------------------------------------------------------------------


_SESSION_FIELDS = frozenset({
    "schema_version", "ref", "state", "model", "run", "turns", "history_bytes", "policy", "diagnostic_code",
})


@dataclass(frozen=True, slots=True)
class ChatSession:
    """The durable view of one session: identity, state, counts, bounds, close diagnostic.

    ``model`` is ``None`` only while the runtime is still ``opening`` or when a
    session closed before it ever became ready (a terminal state with a
    diagnostic and zero turns), because a run-first identity is only known once
    the runtime has admitted the run's artifacts. A ``diagnostic_code`` is
    present only on a terminal state and is mandatory for
    ``cleanup_unresolved``. ``turns`` and ``history_bytes`` never exceed the
    policy.
    """

    schema_version: str
    ref: ChatSessionRef
    state: ChatSessionState
    model: ChatModelIdentity | None
    run: TrainingRunRef | None
    turns: int
    history_bytes: int
    policy: ChatSessionPolicyV1
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != CHAT_SESSION_SCHEMA_VERSION:
            raise ValueError("unsupported chat session schema version")
        if type(self.ref) is not ChatSessionRef or type(self.state) is not ChatSessionState:
            raise TypeError("ref/state have invalid types")
        object.__setattr__(self, "ref", ChatSessionRef.from_dict(self.ref.to_dict()))
        if self.model is not None:
            if type(self.model) is not ChatModelIdentity:
                raise TypeError("model must be exact ChatModelIdentity or None")
            object.__setattr__(self, "model", ChatModelIdentity.from_dict(self.model.to_dict()))
        if self.run is not None:
            if type(self.run) is not TrainingRunRef:
                raise TypeError("run must be exact TrainingRunRef or None")
            run = TrainingRunRef.from_dict(self.run.to_dict())
            if run.project_ref != self.ref.project_ref:
                raise ValueError("bound run project does not match the session")
            object.__setattr__(self, "run", run)
        if type(self.policy) is not ChatSessionPolicyV1:
            raise TypeError("policy must be exact ChatSessionPolicyV1")
        object.__setattr__(self, "policy", ChatSessionPolicyV1.from_dict(self.policy.to_dict()))
        _bounded_integer(self.turns, "turns", minimum=0, maximum=self.policy.max_turns)
        _bounded_integer(self.history_bytes, "history_bytes", minimum=0, maximum=self.policy.max_history_bytes)
        object.__setattr__(self, "diagnostic_code", _optional_text(self.diagnostic_code, "diagnostic_code"))
        terminal = self.state in CHAT_TERMINAL_STATES
        if not terminal and self.diagnostic_code is not None:
            raise ValueError("diagnostic_code belongs to a closed or cleanup_unresolved session")
        if self.state is ChatSessionState.CLEANUP_UNRESOLVED and self.diagnostic_code is None:
            raise ValueError("cleanup_unresolved requires a diagnostic_code")
        if self.model is None:
            never_ready = terminal and self.diagnostic_code is not None
            if (self.state is not ChatSessionState.OPENING and not never_ready) or self.turns or self.history_bytes:
                raise ValueError("model identity is required once a session has been ready")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": self.ref.to_dict(),
            "state": self.state.value,
            "model": None if self.model is None else self.model.to_dict(),
            "run": None if self.run is None else self.run.to_dict(),
            "turns": self.turns,
            "history_bytes": self.history_bytes,
            "policy": self.policy.to_dict(),
            "diagnostic_code": self.diagnostic_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatSession":
        value = exact_fields(value, _SESSION_FIELDS, "chat_session")
        model = value["model"]
        run = value["run"]
        return cls(
            _text(value["schema_version"], "schema_version"),
            ChatSessionRef.from_dict(value["ref"]),  # type: ignore[arg-type]
            _parse_enum(ChatSessionState, value["state"], "state"),
            None if model is None else ChatModelIdentity.from_dict(model),  # type: ignore[arg-type]
            None if run is None else TrainingRunRef.from_dict(run),  # type: ignore[arg-type]
            value["turns"],  # type: ignore[arg-type]
            value["history_bytes"],  # type: ignore[arg-type]
            ChatSessionPolicyV1.from_dict(value["policy"]),  # type: ignore[arg-type]
            value["diagnostic_code"],  # type: ignore[arg-type]
        )


# --- turn ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatTurnRequest:
    """One user message for one session; bounded by the largest admissible history."""

    session: ChatSessionRef
    content: str

    def __post_init__(self) -> None:
        if type(self.session) is not ChatSessionRef:
            raise TypeError("session must be exact ChatSessionRef")
        object.__setattr__(self, "session", ChatSessionRef.from_dict(self.session.to_dict()))
        if type(self.content) is not str:
            raise TypeError("content must be an exact string")
        if not self.content or _utf8_bytes(self.content) > CHAT_MAX_HISTORY_BYTES:
            raise ChatOperationError(ChatOperationCode.TURN_BOUNDS_INVALID)

    def to_dict(self) -> dict[str, object]:
        return {"session": self.session.to_dict(), "content": self.content}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTurnRequest":
        value = exact_fields(value, frozenset({"session", "content"}), "chat_turn_request")
        return cls(ChatSessionRef.from_dict(value["session"]), value["content"])  # type: ignore[arg-type]


_TURN_FIELDS = frozenset({"schema_version", "ref", "state", "content_digest", "content", "usage"})


def content_digest(content: str) -> str:
    """SHA-256 of the UTF-8 content; the binding between a turn and its observations."""
    if type(content) is not str:
        raise TypeError("content must be an exact string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """A completed turn: its ref, the session state it left behind, and the reply.

    ``state`` is the session state observed as the turn completed and is never
    ``opening`` or ``serving``. ``content_digest`` is the SHA-256 of ``content``
    and equals the digest carried by the ``chat_turn_completed`` observation.
    ``usage`` is present only when measured (``UsageAvailability.MEASURED``);
    an unavailable usage is refused here and the field is omitted from the
    canonical document rather than published as null.
    """

    schema_version: str
    ref: ChatTurnRef
    state: ChatSessionState
    content_digest: str
    content: str
    usage: UsageRecordV1 | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != CHAT_TURN_SCHEMA_VERSION:
            raise ValueError("unsupported chat turn schema version")
        if type(self.ref) is not ChatTurnRef:
            raise TypeError("ref must be exact ChatTurnRef")
        object.__setattr__(self, "ref", ChatTurnRef.from_dict(self.ref.to_dict()))
        if type(self.state) is not ChatSessionState:
            raise TypeError("state must be exact ChatSessionState")
        if self.state not in _TURN_STATES:
            raise ValueError("a completed turn leaves the session ready, closing, closed or cleanup_unresolved")
        if type(self.content) is not str:
            raise TypeError("content must be an exact string")
        if _utf8_bytes(self.content) > CHAT_MAX_HISTORY_BYTES:
            raise ValueError("content exceeds the largest admissible history")
        object.__setattr__(self, "content_digest", digest_text(_text(self.content_digest, "content_digest"), "content_digest"))
        if self.content_digest != content_digest(self.content):
            raise ValueError("content_digest does not match content")
        if self.usage is not None:
            if type(self.usage) is not UsageRecordV1:
                raise TypeError("usage must be exact UsageRecordV1 or None")
            if self.usage.availability is not UsageAvailability.MEASURED:
                raise ValueError("unavailable usage must be omitted, not published")
            object.__setattr__(self, "usage", UsageRecordV1.from_dict(self.usage.to_dict()))

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": self.schema_version,
            "ref": self.ref.to_dict(),
            "state": self.state.value,
            "content_digest": self.content_digest,
            "content": self.content,
        }
        if self.usage is not None:
            document["usage"] = self.usage.to_dict()
        return document

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ChatTurn":
        if type(value) is not dict:
            raise TypeError("chat_turn must be an exact object")
        keys = tuple(dict.keys(value))
        if any(type(key) is not str for key in keys):
            raise TypeError("chat_turn field names must be exact strings")
        expected = _TURN_FIELDS if "usage" in keys else _TURN_FIELDS - {"usage"}
        value = exact_fields(value, expected, "chat_turn")
        usage = value.get("usage")
        if "usage" in value and usage is None:
            raise ValueError("usage must be omitted when unavailable, not null")
        return cls(
            _text(value["schema_version"], "schema_version"),
            ChatTurnRef.from_dict(value["ref"]),  # type: ignore[arg-type]
            _parse_enum(ChatSessionState, value["state"], "state"),
            value["content_digest"],  # type: ignore[arg-type]
            value["content"],  # type: ignore[arg-type]
            None if usage is None else UsageRecordV1.from_dict(usage),  # type: ignore[arg-type]
        )


# --- paging ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = _MAX_LIST_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _text(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _ascii_cursor(self.cursor))
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")


@dataclass(frozen=True, slots=True)
class ChatSessionPage:
    request: ChatListRequest
    sessions: tuple[ChatSession, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not ChatListRequest:
            raise TypeError("request must be exact ChatListRequest")
        if type(self.sessions) is not tuple or any(type(item) is not ChatSession for item in self.sessions):
            raise TypeError("sessions must be an exact tuple of ChatSession")
        if len(self.sessions) > self.request.limit:
            raise ValueError("sessions exceed requested limit")
        if any(item.ref.project_ref != self.request.project_ref for item in self.sessions):
            raise ValueError("session project does not match list request")
        _bool(self.truncated, "truncated")
        object.__setattr__(self, "next_cursor", _ascii_cursor(self.next_cursor, "next_cursor"))
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and not self.sessions:
            raise ValueError("a truncated page must contain a session")


# --- operations and facade -------------------------------------------------------


class ChatOperations(Protocol):
    def open(self, request: ChatOpenRequest) -> ChatSession: ...
    def turn(self, request: ChatTurnRequest) -> ChatTurn: ...
    def show(self, session: ChatSessionRef) -> ChatSession: ...
    def list(self, request: ChatListRequest) -> ChatSessionPage: ...
    def close(self, session: ChatSessionRef) -> ChatSession: ...
    def observations(self, request: ObservationsRequest) -> ObservationPage: ...


class ChatAPI:
    """Public chat facade over host-composed ``ChatOperations``.

    Every verb rebuilds its input, presents a detached copy to the callback,
    re-validates the original and the copy after return or raise, and rebuilds
    the callback's result so nothing the callback retains can alias the values
    handed back to the caller. A result must bind the request it answers.
    """

    __slots__ = ("_operations",)

    def __init__(self, operations: ChatOperations) -> None:
        self._operations = operations

    # -- input rebuilders (one per accepted input type) --------------------------

    @staticmethod
    def _session(value: ChatSessionRef) -> ChatSessionRef:
        if type(value) is not ChatSessionRef:
            raise TypeError("session must be exact ChatSessionRef")
        return ChatSessionRef.from_dict(value.to_dict())

    @staticmethod
    def _open_request(value: ChatOpenRequest) -> ChatOpenRequest:
        if type(value) is not ChatOpenRequest:
            raise TypeError("request must be exact ChatOpenRequest")
        return ChatOpenRequest.from_dict(value.to_dict())

    @staticmethod
    def _turn_request(value: ChatTurnRequest) -> ChatTurnRequest:
        if type(value) is not ChatTurnRequest:
            raise TypeError("request must be exact ChatTurnRequest")
        return ChatTurnRequest.from_dict(value.to_dict())

    @staticmethod
    def _list_request(value: ChatListRequest) -> ChatListRequest:
        if type(value) is not ChatListRequest:
            raise TypeError("request must be exact ChatListRequest")
        return ChatListRequest(value.project_ref, value.cursor, value.limit)

    @staticmethod
    def _observations_request(value: ObservationsRequest) -> ObservationsRequest:
        if type(value) is not ObservationsRequest:
            raise TypeError("request must be exact ObservationsRequest")
        stream = ObservationStreamRef.from_dict(value.stream.to_dict())
        if stream.family is not ObservationFamily.CHAT:
            raise ValueError("observations request must name a chat stream")
        return ObservationsRequest(stream, value.after_sequence, value.limit)

    # -- detach-and-revalidate discipline (RunsAPI._call, verbatim) --------------

    @staticmethod
    def _matches(current: object, baseline: object, rebuild) -> bool:
        try:
            return rebuild(current) == baseline
        except BaseException:
            return False

    @staticmethod
    def _changed() -> None:
        raise ValueError("chat operation input changed during callback") from None

    @classmethod
    def _unchanged(cls, current: object, baseline: object, rebuild) -> None:
        if not cls._matches(current, baseline, rebuild):
            cls._changed()

    @classmethod
    def _call(cls, callback, original, baseline, presentation, rebuild, *extra):
        mutation_after_failure = False
        try:
            result = callback(presentation, *extra)
        except BaseException:
            mutation_after_failure = (
                not cls._matches(original, baseline, rebuild)
                or not cls._matches(presentation, baseline, rebuild)
            )
            if not mutation_after_failure:
                raise
        if mutation_after_failure:
            cls._changed()
        cls._unchanged(original, baseline, rebuild)
        cls._unchanged(presentation, baseline, rebuild)
        return result

    # -- result rebuilders ----------------------------------------------------------

    @staticmethod
    def _session_record(value: object, session: ChatSessionRef) -> ChatSession:
        if type(value) is not ChatSession:
            raise TypeError("chat result must be exact ChatSession")
        rebuilt = ChatSession.from_dict(value.to_dict())
        if rebuilt.ref != session:
            raise ValueError("chat session does not bind the request")
        return rebuilt

    # -- verbs -----------------------------------------------------------------------

    def open(self, request: ChatOpenRequest) -> ChatSession:
        baseline = self._open_request(request)
        presented = self._open_request(baseline)
        result = self._call(self._operations.open, request, baseline, presented, self._open_request)
        if type(result) is not ChatSession:
            raise TypeError("chat open result must be exact ChatSession")
        rebuilt = ChatSession.from_dict(result.to_dict())
        if (
            rebuilt.ref.project_ref != baseline.project_ref
            or rebuilt.policy != baseline.policy
            or rebuilt.run != baseline.run
            or rebuilt.state not in (ChatSessionState.OPENING, ChatSessionState.READY)
            or (baseline.model is not None and rebuilt.model not in (None, baseline.model.identity()))
        ):
            raise ValueError("chat session does not bind the open request")
        self._unchanged(request, baseline, self._open_request)
        self._unchanged(presented, baseline, self._open_request)
        return rebuilt

    def turn(self, request: ChatTurnRequest) -> ChatTurn:
        baseline = self._turn_request(request)
        presented = self._turn_request(baseline)
        result = self._call(self._operations.turn, request, baseline, presented, self._turn_request)
        if type(result) is not ChatTurn:
            raise TypeError("chat turn result must be exact ChatTurn")
        rebuilt = ChatTurn.from_dict(result.to_dict())
        if rebuilt.ref.session != baseline.session:
            raise ValueError("chat turn does not bind the request")
        self._unchanged(request, baseline, self._turn_request)
        self._unchanged(presented, baseline, self._turn_request)
        return rebuilt

    def show(self, session: ChatSessionRef) -> ChatSession:
        baseline = self._session(session)
        presented = self._session(baseline)
        rebuilt = self._session_record(self._call(self._operations.show, session, baseline, presented, self._session), baseline)
        self._unchanged(session, baseline, self._session)
        self._unchanged(presented, baseline, self._session)
        return rebuilt

    def list(self, request: ChatListRequest) -> ChatSessionPage:
        baseline = self._list_request(request)
        presented = self._list_request(baseline)
        result = self._call(self._operations.list, request, baseline, presented, self._list_request)
        if type(result) is not ChatSessionPage:
            raise TypeError("chat list result must be exact ChatSessionPage")
        rebuilt = ChatSessionPage(
            ChatListRequest(result.request.project_ref, result.request.cursor, result.request.limit),
            tuple(ChatSession.from_dict(item.to_dict()) for item in result.sessions),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("chat list result does not bind the request")
        self._unchanged(request, baseline, self._list_request)
        self._unchanged(presented, baseline, self._list_request)
        return rebuilt

    def close(self, session: ChatSessionRef) -> ChatSession:
        baseline = self._session(session)
        presented = self._session(baseline)
        rebuilt = self._session_record(self._call(self._operations.close, session, baseline, presented, self._session), baseline)
        self._unchanged(session, baseline, self._session)
        self._unchanged(presented, baseline, self._session)
        return rebuilt

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        baseline = self._observations_request(request)
        presented = self._observations_request(baseline)
        result = self._call(self._operations.observations, request, baseline, presented, self._observations_request)
        if type(result) is not ObservationPage:
            raise TypeError("observation result must be exact ObservationPage")
        rebuilt = ObservationPage(
            self._observations_request(result.request),
            tuple(ObservationRecordV1.from_dict(item.to_dict()) for item in result.records),
            result.next_cursor,
            result.truncated,
        )
        if rebuilt.request != baseline:
            raise ValueError("observation result does not bind the request")
        self._unchanged(request, baseline, self._observations_request)
        self._unchanged(presented, baseline, self._observations_request)
        return rebuilt


__all__ = [
    "ChatAPI", "ChatListRequest", "ChatModelIdentity", "ChatModelKind", "ChatModelSource",
    "ChatOpenRequest", "ChatOperationCode", "ChatOperationError", "ChatOperations",
    "ChatSession", "ChatSessionPage", "ChatSessionPolicyV1", "ChatSessionRef",
    "ChatSessionState", "ChatTurn", "ChatTurnRef", "ChatTurnRequest",
]
