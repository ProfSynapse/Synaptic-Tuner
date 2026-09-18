"""Chat reference composition: ``ChatOperations`` over ``tuner/inference/``.

Location: ``synaptic_tuner/api/v1/reference/chat.py``.

``ChatOperations`` (``api/v1/chat_facade.py``: open, turn, show, list, close,
observations) is implemented here by ``ReferenceChatOperationsV1`` over the
engine's bounded chat session (``tuner/inference/chat_session.py``) reached
through ``open_model_chat`` (model-first) and ``open_run_chat`` (run-first).
Which runtime serves a session is host-supplied by name through
``ReferenceChatPortsV1.runtimes``; nothing here reads the environment, derives
a port, chooses a GPU or resolves a credential. Nothing of the legacy coupling
(§10 of ``docs/architecture/api-facade-v1.md``) is inherited: the failure
signal is the typed ``ModelChatFailure`` and no exception text is ever stored.

Durable layout (partition -> key -> document):

- ``chat_session`` / ``session/<project_digest>/<session_key>``: one small
  compare-and-swap head per session on the record port, holding only the
  current ``synaptic-chat-session/v1`` record, its revision and digest. A
  session is not a lifecycle entity (§7.3 Q1): there is NO lifecycle record and
  NO durable event per turn, so the head never grows with the conversation
  and is bounded by ``CHAT_HEAD_MAXIMUM_BYTES``.
- ``observation`` / ``chat/<project_digest>/<session_key>``: the public
  ``chat_turn_started`` / ``chat_turn_completed`` observation records (§4),
  exactly two per admitted turn. The per-session observation budget is
  therefore ``2 * policy.max_turns`` and at most ``CHAT_OBSERVATION_BUDGET``
  (20,000); the writer asserts the budget before every append. They carry no
  authority; a host may truncate them. Token streaming later appends
  ``chat_token`` records to the same stream without changing this layout.

Liveness lives in this process only: the live engine session, its exit stack
and an in-flight guard are held in memory per session. A second turn while one
is in flight is refused with ``session_busy`` under a per-session lock. A head
whose session is not live here (another process, or a restart) answers
``show``/``list`` normally and ``runtime_unavailable`` for ``turn``/``close``,
because whether its process family still holds a GPU is unknown here.

``cleanup_unresolved`` (§7.3 Q5) is the public shape of a teardown that did
not converge: an ``OwnedProcessError`` whose cleanup remains unresolved, a
``ModelChatFailure`` carrying a pending lease, or a live session still reporting
``cleanup_pending``. The session becomes terminal-uncertain, every further turn
raises ``cleanup_unresolved``, and the host is told through the record that a
process family may still hold a GPU. A host that does not support owned
process families (non-Linux, §7.3 Q4) receives ``host_unsupported`` from the
local runtime arm before anything is spawned.

Consumed by ``synaptic_tuner/api/v1/reference/composition.py`` and exported
lazily through ``synaptic_tuner/api/v1/reference/__init__.py``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import secrets
from threading import Lock, RLock
from typing import Protocol

from synaptic_tuner.api.v1._contract import contract_digest, digest_text
from synaptic_tuner.api.v1.chat_facade import (
    CHAT_MAX_HISTORY_BYTES,
    CHAT_MAX_TURNS,
    CHAT_SESSION_SCHEMA_VERSION,
    CHAT_TURN_SCHEMA_VERSION,
    ChatListRequest,
    ChatModelIdentity,
    ChatModelKind,
    ChatModelSource,
    ChatOpenRequest,
    ChatOperationCode,
    ChatOperationError,
    ChatSession,
    ChatSessionPage,
    ChatSessionPolicyV1,
    ChatSessionRef,
    ChatSessionState,
    ChatTurn,
    ChatTurnRef,
    ChatTurnRequest,
    content_digest,
)
from synaptic_tuner.api.v1.observations import (
    OBSERVATION_SCHEMA_VERSION,
    ChatTurnCompletedPayloadV1,
    ChatTurnStartedPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationsRequest,
    ObservationStreamRef,
)
from synaptic_tuner.api.v1.ports import (
    DurableRecordStorePort,
    DurableStreamStorePort,
    StoragePartition,
    StoredRecordV1,
    StoredStreamPageV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunOperationError, RunsAPI
from synaptic_tuner.api.v1.usage import USAGE_SCHEMA_VERSION, UsageAvailability, UsageRecordV1
from tuner.execution.foundation_v2.canonical import domain_digest
from tuner.execution.redaction import redact
from tuner.inference.chat_session import ChatSession as EngineChatSession
from tuner.inference.chat_session import ChatSessionError, ChatSessionPolicy
from tuner.inference.owned_process import OwnedProcessError
from tuner.inference.run_chat import PreparedRunChat, RunChatRuntime, open_run_chat

from .provider_family import require_methods


# --- constants ---------------------------------------------------------------------

CHAT_HEAD_SCHEMA = "synaptic-reference-chat-head/v1"
CHAT_RECORD_DOMAIN = "synaptic-reference-chat-record/v1"
CHAT_SESSION_DOMAIN = "synaptic-reference-chat-session/v1"
CHAT_PROJECT_DOMAIN = "synaptic-reference-chat-project-key/v1"
CHAT_HEAD_MAXIMUM_BYTES = 16 * 1024
CHAT_OBSERVATION_BUDGET = 2 * CHAT_MAX_TURNS
CHAT_OBSERVATIONS_PER_TURN = 2
HISTORY_PAGE_LIMIT = 200
APPEND_ATTEMPTS = 4

_CHAT_SESSION = StoragePartition.CHAT_SESSION.value
_OBSERVATION = StoragePartition.OBSERVATION.value
_HEAD_KEYS = frozenset({"schema_version", "revision", "record_digest", "record"})
_TERMINAL = frozenset({ChatSessionState.CLOSED, ChatSessionState.CLEANUP_UNRESOLVED})
_REDACT_DEPTH = 8
_REDACT_ITEMS = 1024


def _error(code: ChatOperationCode) -> ChatOperationError:
    return ChatOperationError(code)


def _integrity() -> ChatOperationError:
    return _error(ChatOperationCode.INTEGRITY_ERROR)


# --- host ports ---------------------------------------------------------------------


class ChatRuntimePort(Protocol):
    """One chat runtime a host offers under a name.

    ``open_model`` yields a live engine ``ChatSession`` for a model-first
    source; ``open_run`` yields a ``PreparedRunChat`` for a run-first request.
    Both are context managers whose exit tears the process family down. Where
    and how a model is served (local vLLM, a Modal sandbox, ...) is entirely
    the runtime's business; the operations layer only sees the bounded session.
    """

    def open_model(
        self, source: ChatModelSource, policy: ChatSessionPolicyV1
    ) -> AbstractContextManager[EngineChatSession]: ...

    def open_run(
        self, runs: RunsAPI, run: TrainingRunRef, policy: ChatSessionPolicyV1
    ) -> AbstractContextManager[PreparedRunChat]: ...


class ChatRuntimeRegistryPort(Protocol):
    def resolve(self, runtime: str) -> ChatRuntimePort | None: ...


class HostInfoPort(Protocol):
    """Whether this host can own process families (Linux procfs + POSIX groups)."""

    def supports_owned_processes(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ReferenceChatPortsV1:
    """What a host injects to compose the chat family."""

    runtimes: ChatRuntimeRegistryPort

    def __post_init__(self) -> None:
        require_methods(self.runtimes, "resolve")


# --- reference port implementations ------------------------------------------------


class ChatRuntimeRegistryV1:
    """``ChatRuntimeRegistryPort`` over a fixed name -> runtime mapping."""

    __slots__ = ("_runtimes",)

    def __init__(self, runtimes: dict[str, ChatRuntimePort]) -> None:
        if type(runtimes) is not dict or any(type(name) is not str or not name for name in runtimes):
            raise TypeError("runtimes must map non-empty names to runtimes")
        for runtime in runtimes.values():
            require_methods(runtime, "open_model", "open_run")
        self._runtimes = dict(runtimes)

    def resolve(self, runtime: str) -> ChatRuntimePort | None:
        if type(runtime) is not str:
            return None
        return self._runtimes.get(runtime)


class LocalHostInfoV1:
    """``HostInfoPort`` mirroring the engine's owned-process host requirement."""

    __slots__ = ()

    def supports_owned_processes(self) -> bool:
        return os.name == "posix" and Path("/proc/self/stat").is_file() and hasattr(os, "killpg")


class LocalVLLMChatRuntimeV1:
    """``ChatRuntimePort`` serving on this machine through the engine's local vLLM arm.

    Model-first sessions go through ``open_model_chat``; run-first sessions are
    delegated to the host-injected ``RunChatRuntime`` (``run_runtime``), which
    owns authentication, artifact admission and the served session's bounds;
    without one, run-first requests are ``runtime_unavailable``. A host whose
    ``HostInfoPort`` reports no owned-process support receives
    ``host_unsupported`` before any process is spawned. Every knob (working
    directory, allow-listed environment, ports, GPU fraction, generation
    parameters) is supplied by the host at composition time.
    """

    __slots__ = (
        "_cwd", "_environment", "_host", "_run_runtime", "_served_model_name",
        "_startup", "_generation",
    )

    def __init__(
        self,
        *,
        cwd: Path,
        environment: dict[str, str],
        host: HostInfoPort | None = None,
        run_runtime: RunChatRuntime | None = None,
        served_model_name: str = "synaptic-chat",
        startup: dict[str, object] | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        if not isinstance(cwd, Path):
            raise TypeError("cwd must be a Path")
        if type(environment) is not dict:
            raise TypeError("environment must be a dict")
        if type(served_model_name) is not str or not served_model_name:
            raise TypeError("served_model_name must be a non-empty string")
        if startup is not None and type(startup) is not dict:
            raise TypeError("startup overrides must be a dict")
        self._cwd = cwd
        self._environment = dict(environment)
        self._host = LocalHostInfoV1() if host is None else host
        require_methods(self._host, "supports_owned_processes")
        if run_runtime is not None:
            require_methods(run_runtime, "open")
        self._run_runtime = run_runtime
        self._served_model_name = served_model_name
        self._startup = {} if startup is None else dict(startup)
        self._generation = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}

    def _require_host(self) -> None:
        if self._host.supports_owned_processes() is not True:
            raise _error(ChatOperationCode.HOST_UNSUPPORTED)

    def open_model(
        self, source: ChatModelSource, policy: ChatSessionPolicyV1
    ) -> AbstractContextManager[EngineChatSession]:
        self._require_host()
        # Imported here so composing a host that never serves locally does not
        # load the vLLM client stack.
        from tuner.inference.model_chat import open_model_chat
        from tuner.inference.vllm_runtime import (
            ExplicitNetworkLoRA,
            ExplicitNetworkVLLMSource,
            VLLMStartupSpec,
        )

        local = source.model_ref.startswith("/")
        adapter = None
        if source.model_kind is ChatModelKind.LORA:
            adapter = ExplicitNetworkLoRA(self._served_model_name, Path(source.adapter_path))
        vllm_source = ExplicitNetworkVLLMSource(
            source.model_ref, revision=None if local else source.model_revision, tokenizer_ref=None, lora=adapter
        )
        startup = VLLMStartupSpec(vllm_source, self._served_model_name, **self._startup)
        return open_model_chat(
            startup, _engine_policy(policy), cwd=self._cwd, environment=dict(self._environment), **self._generation
        )

    def open_run(
        self, runs: RunsAPI, run: TrainingRunRef, policy: ChatSessionPolicyV1
    ) -> AbstractContextManager[PreparedRunChat]:
        self._require_host()
        if self._run_runtime is None:
            raise _error(ChatOperationCode.RUNTIME_UNAVAILABLE)
        return open_run_chat(runs, run, runtime=self._run_runtime)


def _engine_policy(policy: ChatSessionPolicyV1) -> ChatSessionPolicy:
    return ChatSessionPolicy(
        request_timeout_seconds=policy.request_timeout_seconds,
        idle_timeout_seconds=policy.idle_timeout_seconds,
        absolute_lifetime_seconds=policy.absolute_lifetime_seconds,
        max_turns=policy.max_turns,
        max_history_bytes=policy.max_history_bytes,
    )


# --- canonical documents ------------------------------------------------------------


def _dump(document: dict[str, object], maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise _integrity() from None
    if len(encoded) > maximum:
        raise _integrity()
    return encoded


def _load_head(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > CHAT_HEAD_MAXIMUM_BYTES:
        raise _integrity()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _integrity() from None
    if type(document) is not dict or set(document) != _HEAD_KEYS:
        raise _integrity()
    if document["schema_version"] != CHAT_HEAD_SCHEMA or _dump(document, CHAT_HEAD_MAXIMUM_BYTES) != raw:
        raise _integrity()
    return document


def _project_digest(project_ref: str) -> str:
    return domain_digest(CHAT_PROJECT_DOMAIN, project_ref.encode("utf-8"))


def _session_key(session_id: str) -> str:
    return domain_digest(CHAT_SESSION_DOMAIN, session_id.encode("utf-8"))


def _head_key(session: ChatSessionRef) -> str:
    return f"session/{_project_digest(session.project_ref)}/{_session_key(session.session_id)}"


def _observation_key(session: ChatSessionRef) -> str:
    return f"chat/{_project_digest(session.project_ref)}/{_session_key(session.session_id)}"


def _stream_ref(session: ChatSessionRef) -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.CHAT, session.project_ref, session.session_id)


def _record_digest(record: ChatSession) -> str:
    return contract_digest(CHAT_RECORD_DOMAIN, record.to_dict())


def _head_document(record: ChatSession, revision: int) -> bytes:
    return _dump(
        {
            "schema_version": CHAT_HEAD_SCHEMA,
            "revision": revision,
            "record_digest": _record_digest(record),
            "record": record.to_dict(),
        },
        CHAT_HEAD_MAXIMUM_BYTES,
    )


def _redacted_text(value: object) -> str:
    """Bounded, secret-redacted copy of one reply; never the raw backend text."""
    try:
        text = json.loads(
            redact(value, max_depth=_REDACT_DEPTH, max_string_bytes=CHAT_MAX_HISTORY_BYTES, max_items=_REDACT_ITEMS)
        )
    except ValueError:
        raise _integrity() from None
    if type(text) is not str:
        raise _integrity()
    return text


def _reply_text(response: object) -> str:
    message = getattr(response, "message", None)
    if type(message) is str:
        return message
    if type(message) is dict:
        try:
            return json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise _integrity() from None
    raise _integrity()


def _measured_usage(response: object) -> UsageRecordV1 | None:
    """``UsageRecordV1`` when the backend reported token counts, else ``None`` (omitted)."""
    raw = getattr(response, "raw", None)
    usage = raw.get("usage") if type(raw) is dict else None
    if type(usage) is not dict:
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0:
        return None
    return UsageRecordV1(USAGE_SCHEMA_VERSION, UsageAvailability.MEASURED, prompt, completion, None, None, None)


def _identity(model: object) -> ChatModelIdentity:
    try:
        return ChatModelIdentity(
            model.model_ref, model.model_revision, model.tokenizer_revision, ChatModelKind(model.model_kind)  # type: ignore[attr-defined]
        )
    except (AttributeError, TypeError, ValueError):
        raise _error(ChatOperationCode.MODEL_INELIGIBLE) from None


# --- observation stream -------------------------------------------------------------


def _read_page(
    streams: DurableStreamStorePort, key: str, after_sequence: int | None, limit: int
) -> StoredStreamPageV1:
    try:
        page = streams.read_page(partition=_OBSERVATION, stream_key=key, after_sequence=after_sequence, limit=limit)
    except Exception:
        raise _integrity() from None
    if type(page) is not StoredStreamPageV1:
        raise _integrity()
    return page


def _tail_sequence(streams: DurableStreamStorePort, key: str) -> int:
    tail = 0
    cursor = None
    while True:
        page = _read_page(streams, key, cursor, HISTORY_PAGE_LIMIT)
        if page.entries:
            tail = page.entries[-1].sequence
        if not page.truncated or page.next_cursor is None:
            return tail
        cursor = page.next_cursor


class _ObservationWriter:
    """Appends ``chat_turn_*`` records for one session within its observation budget."""

    __slots__ = ("_streams", "_key", "_stream", "_clock", "_budget", "_next")

    def __init__(self, streams: DurableStreamStorePort, session: ChatSessionRef, clock, budget: int) -> None:
        self._streams = streams
        self._key = _observation_key(session)
        self._stream = _stream_ref(session)
        self._clock = clock
        self._budget = min(budget, CHAT_OBSERVATION_BUDGET)
        self._next = _tail_sequence(streams, self._key) + 1

    def emit(self, kind: ObservationKind, payload: object) -> None:
        for _ in range(APPEND_ATTEMPTS):
            if self._next > self._budget:
                raise _integrity()
            record = ObservationRecordV1(
                OBSERVATION_SCHEMA_VERSION, self._stream, self._next, self._clock.now(), kind, payload
            )
            canonical = _dump(record.to_dict(), CHAT_HEAD_MAXIMUM_BYTES)
            try:
                appended = self._streams.append(
                    partition=_OBSERVATION, stream_key=self._key, sequence=self._next, canonical=canonical
                )
            except Exception:
                raise _integrity() from None
            if appended is True:
                self._next += 1
                return
            if appended is not False:
                raise _integrity()
            self._next = _tail_sequence(self._streams, self._key) + 1
        raise _integrity()


# --- liveness ----------------------------------------------------------------------------


@dataclass(slots=True)
class _Head:
    record: ChatSession
    revision: int


class _Live:
    """This process's hold on one open session: engine session, exit stack, in-flight guard."""

    __slots__ = ("session", "stack", "writer", "guard", "in_flight")

    def __init__(self, session: EngineChatSession, stack: ExitStack, writer: _ObservationWriter) -> None:
        self.session = session
        self.stack = stack
        self.writer = writer
        self.guard = Lock()
        self.in_flight = False

    def acquire(self) -> None:
        with self.guard:
            if self.in_flight:
                raise _error(ChatOperationCode.SESSION_BUSY)
            self.in_flight = True

    def release(self) -> None:
        with self.guard:
            self.in_flight = False


# --- operations ---------------------------------------------------------------------


class ReferenceChatOperationsV1:
    """``ChatOperations`` over the host stores, the runs facade and the chat ports."""

    __slots__ = ("_records", "_streams", "_clock", "_runs", "_ports", "_live", "_lock")

    def __init__(
        self,
        *,
        records: DurableRecordStorePort,
        streams: DurableStreamStorePort,
        clock,
        runs: RunsAPI,
        ports: ReferenceChatPortsV1,
    ) -> None:
        require_methods(records, "create", "read", "compare_and_swap", "list_page")
        require_methods(streams, "append", "read_page")
        require_methods(clock, "now")
        if type(runs) is not RunsAPI:
            raise TypeError("runs must be an exact RunsAPI")
        if type(ports) is not ReferenceChatPortsV1:
            raise TypeError("ports must be exact ReferenceChatPortsV1")
        self._records = records
        self._streams = streams
        self._clock = clock
        self._runs = runs
        self._ports = ports
        self._live: dict[str, _Live] = {}
        self._lock = RLock()

    # -- head documents ------------------------------------------------------------

    def _read_head(self, session: ChatSessionRef) -> _Head | None:
        key = _head_key(session)
        try:
            stored = self._records.read(partition=_CHAT_SESSION, key=key)
        except Exception:
            raise _integrity() from None
        if stored is None:
            return None
        return self._validate(key, stored, session)

    @staticmethod
    def _validate(key: str, stored: StoredRecordV1, session: ChatSessionRef | None) -> _Head:
        if type(stored) is not StoredRecordV1 or stored.key != key:
            raise _integrity()
        document = _load_head(stored.canonical)
        if type(document["revision"]) is not int or document["revision"] < 1:
            raise _integrity()
        try:
            record = ChatSession.from_dict(document["record"])  # type: ignore[arg-type]
            digest_text(document["record_digest"], "record_digest")
        except Exception:
            raise _integrity() from None
        if document["record_digest"] != _record_digest(record) or _head_key(record.ref) != key:
            raise _integrity()
        if session is not None and record.ref != session:
            raise _integrity()
        return _Head(record, stored.revision)

    def _require_head(self, session: ChatSessionRef) -> _Head:
        head = self._read_head(session)
        if head is None:
            raise _error(ChatOperationCode.SESSION_MISSING)
        return head

    def _create(self, record: ChatSession) -> _Head:
        raw = _head_document(record, 1)
        try:
            created = self._records.create(partition=_CHAT_SESSION, key=_head_key(record.ref), canonical=raw)
        except Exception:
            raise _integrity() from None
        if created is not True:
            raise _integrity()
        return _Head(record, 1)

    def _swap(self, head: _Head, record: ChatSession) -> _Head:
        """Commit the next record through the head compare-and-swap; no event is appended."""
        revision = head.revision + 1
        raw = _head_document(record, revision)
        with self._lock:
            try:
                swapped = self._records.compare_and_swap(
                    partition=_CHAT_SESSION, key=_head_key(record.ref), expected_revision=head.revision, canonical=raw
                )
            except Exception:
                raise _integrity() from None
        if swapped is not True:
            raise _integrity()
        return _Head(record, revision)

    # -- runtime resolution ------------------------------------------------------------

    def _runtime(self, name: str) -> ChatRuntimePort:
        runtime = self._ports.runtimes.resolve(name)
        if runtime is None:
            raise _error(ChatOperationCode.RUNTIME_UNAVAILABLE)
        require_methods(runtime, "open_model", "open_run")
        return runtime

    @staticmethod
    def _open_failure(error: BaseException) -> ChatOperationCode:
        if isinstance(error, ChatOperationError):
            return error.code
        if _leaked(error):
            return ChatOperationCode.CLEANUP_UNRESOLVED
        if isinstance(error, RunOperationError):
            return ChatOperationCode.MODEL_INELIGIBLE
        if isinstance(error, (TypeError, ValueError)) and not isinstance(error, ChatSessionError):
            return ChatOperationCode.MODEL_INELIGIBLE
        return ChatOperationCode.RUNTIME_UNAVAILABLE

    # -- verbs -----------------------------------------------------------------------

    def open(self, request: ChatOpenRequest) -> ChatSession:
        runtime = self._runtime(request.runtime)
        ref = ChatSessionRef("cs-" + secrets.token_hex(16), request.project_ref)
        opening = ChatSession(
            CHAT_SESSION_SCHEMA_VERSION, ref, ChatSessionState.OPENING, None, request.run, 0, 0, request.policy
        )
        head = self._create(opening)
        stack = ExitStack()
        try:
            if request.model is not None:
                session = stack.enter_context(runtime.open_model(request.model, request.policy))
                identity = request.model.identity()
            else:
                prepared = stack.enter_context(runtime.open_run(self._runs, request.run, request.policy))
                if type(prepared) is not PreparedRunChat:
                    raise TypeError("runtime must yield an exact PreparedRunChat")
                session = prepared.session
                identity = _identity(prepared.model)
            if type(session) is not EngineChatSession:
                raise TypeError("runtime must yield an exact engine ChatSession")
        except BaseException as error:
            # Classify inside the handler, raise outside it: the public error must
            # carry neither the runtime's exception as cause nor as context.
            code = self._open_failure(error)
            terminal = (
                ChatSessionState.CLEANUP_UNRESOLVED
                if code is ChatOperationCode.CLEANUP_UNRESOLVED
                else ChatSessionState.CLOSED
            )
            self._swap(head, replace(opening, state=terminal, diagnostic_code=code.value))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            failure = code
        else:
            failure = None
        if failure is not None:
            raise _error(failure)
        ready = replace(opening, state=ChatSessionState.READY, model=identity)
        writer = _ObservationWriter(self._streams, ref, self._clock, CHAT_OBSERVATIONS_PER_TURN * request.policy.max_turns)
        live = _Live(session, stack, writer)
        try:
            head = self._swap(head, ready)
        except BaseException:
            self._teardown(head, live, ChatOperationCode.INTEGRITY_ERROR)
            raise
        with self._lock:
            self._live[_head_key(ref)] = live
        return ready

    def turn(self, request: ChatTurnRequest) -> ChatTurn:
        head = self._require_head(request.session)
        record = head.record
        if record.state is ChatSessionState.CLEANUP_UNRESOLVED:
            raise _error(ChatOperationCode.CLEANUP_UNRESOLVED)
        if record.state in (ChatSessionState.CLOSED, ChatSessionState.CLOSING):
            raise _error(ChatOperationCode.SESSION_CLOSED)
        if record.state is not ChatSessionState.READY:
            raise _error(ChatOperationCode.SESSION_BUSY)
        key = _head_key(request.session)
        with self._lock:
            live = self._live.get(key)
        if live is None:
            raise _error(ChatOperationCode.RUNTIME_UNAVAILABLE)
        live.acquire()
        try:
            return self._serve(head, live, request)
        finally:
            live.release()

    def _serve(self, head: _Head, live: _Live, request: ChatTurnRequest) -> ChatTurn:
        record = head.record
        policy = record.policy
        content_bytes = len(request.content.encode("utf-8"))
        if record.turns + 1 > policy.max_turns or record.history_bytes + content_bytes > policy.max_history_bytes:
            self._teardown(head, live, ChatOperationCode.TURN_BOUNDS_INVALID)
            raise _error(ChatOperationCode.TURN_BOUNDS_INVALID)
        request_id = record.turns + 1
        head = self._swap(head, replace(record, state=ChatSessionState.SERVING))
        live.writer.emit(ObservationKind.CHAT_TURN_STARTED, ChatTurnStartedPayloadV1(request_id))
        failure = None
        try:
            response = live.session.chat(request.content)
        except (KeyboardInterrupt, SystemExit):
            raise
        except ChatSessionError:
            failure = self._teardown(head, live, ChatOperationCode.SESSION_CLOSED) or ChatOperationCode.SESSION_CLOSED
        except Exception:
            failure = self._teardown(head, live, ChatOperationCode.RUNTIME_UNAVAILABLE) or ChatOperationCode.RUNTIME_UNAVAILABLE
        if failure is not None:
            raise _error(failure)
        text = _redacted_text(_reply_text(response))
        usage = _measured_usage(response)
        digest = content_digest(text)
        state = self._live_state(live)
        if (
            state is None
            or state.closed
            or state.turns > policy.max_turns
            or state.history_bytes > policy.max_history_bytes
        ):
            code = ChatOperationCode.TURN_BOUNDS_INVALID if state is not None and not state.closed else ChatOperationCode.SESSION_CLOSED
            final = self._teardown(head, live, code)
            turn_state = ChatSessionState.CLEANUP_UNRESOLVED if final is ChatOperationCode.CLEANUP_UNRESOLVED else ChatSessionState.CLOSED
        else:
            head = self._swap(
                head, replace(record, state=ChatSessionState.READY, turns=state.turns, history_bytes=state.history_bytes)
            )
            turn_state = ChatSessionState.READY
        live.writer.emit(
            ObservationKind.CHAT_TURN_COMPLETED,
            ChatTurnCompletedPayloadV1(request_id, digest, len(text.encode("utf-8"))),
        )
        return ChatTurn(CHAT_TURN_SCHEMA_VERSION, ChatTurnRef(record.ref, request_id), turn_state, digest, text, usage)

    def show(self, session: ChatSessionRef) -> ChatSession:
        return self._require_head(session).record

    def list(self, request: ChatListRequest) -> ChatSessionPage:
        prefix = f"session/{_project_digest(request.project_ref)}/"
        after_key = None
        if request.cursor is not None:
            # The cursor is the session key of the last listed session; a cursor
            # naming no stored session is the closest closed code, ``session_missing``.
            try:
                digest_text(request.cursor, "cursor")
            except ValueError:
                raise _error(ChatOperationCode.SESSION_MISSING) from None
            after_key = prefix + request.cursor
            try:
                stored = self._records.read(partition=_CHAT_SESSION, key=after_key)
            except Exception:
                raise _integrity() from None
            if stored is None:
                raise _error(ChatOperationCode.SESSION_MISSING)
        try:
            page = self._records.list_page(partition=_CHAT_SESSION, prefix=prefix, after_key=after_key, limit=request.limit)
        except Exception:
            raise _integrity() from None
        sessions = []
        for stored in page.records:
            if type(stored) is not StoredRecordV1 or not stored.key.startswith(prefix):
                raise _integrity()
            record = self._validate(stored.key, stored, None).record
            if record.ref.project_ref != request.project_ref:
                raise _integrity()
            sessions.append(record)
        next_cursor = None
        if page.truncated:
            if type(page.next_cursor) is not str or not page.next_cursor.startswith(prefix):
                raise _integrity()
            next_cursor = page.next_cursor[len(prefix):]
        return ChatSessionPage(request, tuple(sessions), next_cursor, page.truncated is True)

    def close(self, session: ChatSessionRef) -> ChatSession:
        head = self._require_head(session)
        record = head.record
        if record.state in _TERMINAL:
            return record
        if record.state is ChatSessionState.OPENING:
            raise _error(ChatOperationCode.SESSION_BUSY)
        key = _head_key(session)
        with self._lock:
            live = self._live.get(key)
        if live is None:
            raise _error(ChatOperationCode.RUNTIME_UNAVAILABLE)
        live.acquire()
        try:
            self._teardown(head, live, None)
        finally:
            live.release()
        return self._require_head(session).record

    def observations(self, request: ObservationsRequest) -> ObservationPage:
        session = ChatSessionRef(request.stream.entity_id, request.stream.project_ref)
        self._require_head(session)
        page = _read_page(self._streams, _observation_key(session), request.after_sequence, request.limit)
        records = []
        for entry in page.entries:
            try:
                record = ObservationRecordV1.from_dict(json.loads(entry.canonical.decode("utf-8")))
            except Exception:
                raise _integrity() from None
            if record.sequence != entry.sequence or record.stream != request.stream:
                raise _integrity()
            records.append(record)
        return ObservationPage(request, tuple(records), page.next_cursor, page.truncated is True)

    # -- teardown -------------------------------------------------------------------

    @staticmethod
    def _live_state(live: _Live):
        try:
            return live.session.state
        except Exception:
            return None

    def _teardown(self, head: _Head, live: _Live, diagnostic: ChatOperationCode | None) -> ChatOperationCode | None:
        """Close the process family and commit the terminal record.

        Returns the diagnostic actually recorded: ``cleanup_unresolved`` when the
        teardown did not converge (the session is terminal-uncertain and a
        process family may still hold a GPU), otherwise ``diagnostic`` itself,
        or ``runtime_unavailable`` when a clean close raised without a leak.
        """
        key = _head_key(head.record.ref)
        with self._lock:
            if self._live.get(key) is live:
                del self._live[key]
        current = self._read_head(head.record.ref) or head
        if current.record.state in _TERMINAL:
            return None if current.record.diagnostic_code is None else ChatOperationCode(current.record.diagnostic_code)
        try:
            closing = self._swap(current, replace(current.record, state=ChatSessionState.CLOSING))
        except ChatOperationError:
            closing = current
        leaked = False
        failed = False
        try:
            live.stack.close()
        except (KeyboardInterrupt, SystemExit):
            leaked = True
            self._commit_terminal(closing, leaked=True, diagnostic=diagnostic, failed=False)
            raise
        except BaseException as error:
            failed = True
            leaked = _leaked(error)
        state = self._live_state(live)
        if state is None or state.cleanup_pending or state.close_error is not None:
            leaked = True
        return self._commit_terminal(closing, leaked=leaked, diagnostic=diagnostic, failed=failed)

    def _commit_terminal(
        self, head: _Head, *, leaked: bool, diagnostic: ChatOperationCode | None, failed: bool
    ) -> ChatOperationCode | None:
        if leaked:
            code: ChatOperationCode | None = ChatOperationCode.CLEANUP_UNRESOLVED
            state = ChatSessionState.CLEANUP_UNRESOLVED
        else:
            code = diagnostic if diagnostic is not None else (ChatOperationCode.RUNTIME_UNAVAILABLE if failed else None)
            state = ChatSessionState.CLOSED
        record = replace(head.record, state=state, diagnostic_code=None if code is None else code.value)
        self._swap(head, record)
        return code


def _leaked(error: BaseException) -> bool:
    """Whether ``error`` reports an owned process family whose cleanup did not converge."""
    from tuner.inference.model_chat import ModelChatError

    if isinstance(error, OwnedProcessError):
        return "cleanup remains unresolved" in str(error)
    if isinstance(error, ModelChatError):
        return error.failure.cleanup_lease is not None
    if isinstance(error, ChatSessionError):
        return "cleanup remains unresolved" in str(error)
    return False


def compose_reference_chat(
    *,
    records: DurableRecordStorePort,
    streams: DurableStreamStorePort,
    clock,
    runs: RunsAPI,
    chat: ReferenceChatPortsV1,
) -> ReferenceChatOperationsV1:
    """The ``ChatOperations`` implementation over the host stores, runs facade and chat ports."""
    return ReferenceChatOperationsV1(records=records, streams=streams, clock=clock, runs=runs, ports=chat)


__all__ = [
    "CHAT_HEAD_MAXIMUM_BYTES",
    "CHAT_OBSERVATION_BUDGET",
    "CHAT_OBSERVATIONS_PER_TURN",
    "ChatRuntimePort",
    "ChatRuntimeRegistryPort",
    "ChatRuntimeRegistryV1",
    "HostInfoPort",
    "LocalHostInfoV1",
    "LocalVLLMChatRuntimeV1",
    "ReferenceChatOperationsV1",
    "ReferenceChatPortsV1",
    "compose_reference_chat",
]
