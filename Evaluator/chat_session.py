"""Bounded in-memory conversation control for an already-ready backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, cast, runtime_checkable

from .protocols import BackendClient, BackendResponse

_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_TURNS = 10_000
_MAX_HISTORY_BYTES = 64 * 1024 * 1024
_CONTEXT_CLOSE_WAIT_SECONDS = 10.5


class ChatSessionError(RuntimeError):
    """A closed or failed bounded chat session."""


@runtime_checkable
class OwnedRuntimeLease(Protocol):
    """The ready runtime exclusively owned by one controller."""

    def close(
        self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
    ) -> bool: ...

    @property
    def cleanup_pending(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ChatSessionPolicy:
    request_timeout_seconds: float
    idle_timeout_seconds: float
    absolute_lifetime_seconds: float
    max_turns: int
    max_history_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "request_timeout_seconds",
            "idle_timeout_seconds",
            "absolute_lifetime_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite number")
            if not 0 < value <= _MAX_TIMEOUT_SECONDS:
                raise ValueError(f"{name} is outside its bound")
        if type(self.max_turns) is not int or not 0 < self.max_turns <= _MAX_TURNS:
            raise ValueError("max_turns is outside its bound")
        if (
            type(self.max_history_bytes) is not int
            or not 0 < self.max_history_bytes <= _MAX_HISTORY_BYTES
        ):
            raise ValueError("max_history_bytes is outside its bound")


@dataclass(frozen=True, slots=True)
class ChatSessionState:
    closed: bool
    cleanup_pending: bool
    request_inflight: bool
    turns: int
    history_bytes: int
    close_error: str | None


Clock = Callable[[], float]
Waiter = Callable[[threading.Event, float], bool]


def _wait(event: threading.Event, timeout: float) -> bool:
    return event.wait(timeout)


def _now(clock: Clock) -> float:
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ChatSessionError("session clock returned an invalid value")
    return float(value)


def _response_text(response: BackendResponse, maximum_bytes: int) -> tuple[str, int]:
    if type(response) is not BackendResponse:
        raise ChatSessionError("backend returned the wrong response type")
    if type(response.raw) is not dict:
        raise ChatSessionError("backend response raw value is invalid")
    if type(response.latency_s) not in (int, float) or not math.isfinite(
        response.latency_s
    ):
        raise ChatSessionError("backend response latency is invalid")
    if type(response.message) is str:
        if len(response.message) > maximum_bytes:
            raise ChatSessionError("backend response exceeds history bound")
        encoded = response.message.encode("utf-8")
        if len(encoded) > maximum_bytes:
            raise ChatSessionError("backend response exceeds history bound")
        return response.message, len(encoded)
    if type(response.message) is dict:
        try:
            _preflight_json(response.message, maximum_bytes)
            encoder = json.JSONEncoder(
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            chunks: list[str] = []
            size = 0
            for chunk in encoder.iterencode(response.message):
                if len(chunk) > maximum_bytes - size:
                    raise ChatSessionError("backend response exceeds history bound")
                encoded = chunk.encode("utf-8")
                size += len(encoded)
                if size > maximum_bytes:
                    raise ChatSessionError("backend response exceeds history bound")
                chunks.append(chunk)
            return "".join(chunks), size
        except ChatSessionError:
            raise
        except (TypeError, ValueError):
            raise ChatSessionError("backend response message is invalid") from None
    raise ChatSessionError("backend response message is invalid")


def _preflight_json(value: object, maximum_bytes: int) -> None:
    """Bound a JSON value before the standard encoder allocates escaped chunks."""

    remaining_nodes = maximum_bytes
    active: set[int] = set()

    def string_size(item: str) -> int:
        size = 2
        for character in item:
            codepoint = ord(character)
            if character in {'"', "\\"} or character in "\b\f\n\r\t":
                size += 2
            elif codepoint < 0x20:
                size += 6
            elif codepoint <= 0x7F:
                size += 1
            elif codepoint <= 0x7FF:
                size += 2
            elif 0xD800 <= codepoint <= 0xDFFF:
                raise ChatSessionError("backend response message is invalid")
            elif codepoint <= 0xFFFF:
                size += 3
            else:
                size += 4
            if size > maximum_bytes:
                raise ChatSessionError("backend response exceeds history bound")
        return size

    def visit(item: object, depth: int) -> int:
        nonlocal remaining_nodes
        if depth > 64 or remaining_nodes <= 0:
            raise ChatSessionError("backend response exceeds history bound")
        remaining_nodes -= 1
        if item is None:
            return 4
        if type(item) is bool:
            return 4 if item else 5
        if type(item) is str:
            return string_size(item)
        if type(item) is int:
            if item.bit_length() > maximum_bytes * 4 + 1:
                raise ChatSessionError("backend response exceeds history bound")
            try:
                size = len(str(item))
            except ValueError:
                raise ChatSessionError(
                    "backend response exceeds history bound"
                ) from None
            if size > maximum_bytes:
                raise ChatSessionError("backend response exceeds history bound")
            return size
        if type(item) is float:
            if not math.isfinite(item):
                raise ChatSessionError("backend response message is invalid")
            return len(repr(item))
        if type(item) not in (dict, list):
            raise ChatSessionError("backend response message is invalid")
        identity = id(item)
        if identity in active:
            raise ChatSessionError("backend response message is invalid")
        active.add(identity)
        try:
            if type(item) is dict:
                size = 2
                for key, child in item.items():
                    if type(key) is not str:
                        raise ChatSessionError("backend response message is invalid")
                    if size > 2:
                        size += 1
                    size += visit(key, depth + 1) + 1
                    size += visit(child, depth + 1)
                    if size > maximum_bytes:
                        raise ChatSessionError("backend response exceeds history bound")
            else:
                size = 2
                for child in item:
                    if size > 2:
                        size += 1
                    size += visit(child, depth + 1)
                    if size > maximum_bytes:
                        raise ChatSessionError("backend response exceeds history bound")
            return size
        finally:
            active.remove(identity)

    if visit(value, 0) > maximum_bytes:
        raise ChatSessionError("backend response exceeds history bound")


class ChatSession:
    """Bound one generic backend conversation to one owned runtime lifetime."""

    def __init__(
        self,
        client: BackendClient,
        runtime: OwnedRuntimeLease,
        policy: ChatSessionPolicy,
        *,
        clock: Clock = time.monotonic,
        waiter: Waiter = _wait,
    ) -> None:
        if not isinstance(client, BackendClient):
            raise TypeError("client must implement BackendClient")
        if not isinstance(runtime, OwnedRuntimeLease):
            raise TypeError("runtime must implement OwnedRuntimeLease")
        if type(policy) is not ChatSessionPolicy:
            raise TypeError("policy must be an exact ChatSessionPolicy")
        if not callable(clock) or not callable(waiter):
            raise TypeError("clock and waiter must be callable")
        self._client = client
        self._runtime = runtime
        self._policy = policy
        self._clock = clock
        self._waiter = waiter
        self._condition = threading.Condition()
        self._wake = threading.Event()
        self._closed = False
        self._close_started = False
        self._close_finished = False
        self._close_error: str | None = None
        self._inflight = False
        self._turns = 0
        self._history_bytes = 0
        self._history: tuple[Mapping[str, str], ...] = ()
        self._request_event: threading.Event | None = None
        self._cleanup_thread: threading.Thread | None = None
        try:
            started = _now(clock)
        except (KeyboardInterrupt, SystemExit):
            with self._condition:
                self._begin_close_locked()
            raise
        except BaseException:
            with self._condition:
                self._begin_close_locked()
            raise ChatSessionError("session clock failed") from None
        self._started = started
        self._last_activity = started
        self._watchdog = threading.Thread(
            target=self._watch,
            name="chat-session-watchdog",
            daemon=True,
        )
        try:
            self._watchdog.start()
        except (KeyboardInterrupt, SystemExit):
            with self._condition:
                self._begin_close_locked()
            raise
        except BaseException:
            with self._condition:
                self._begin_close_locked()
            raise ChatSessionError("session watchdog failed") from None

    def _expired(self, now: float) -> bool:
        return (
            now - self._started >= self._policy.absolute_lifetime_seconds
            or now - self._last_activity >= self._policy.idle_timeout_seconds
        )

    def _watch(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                    now = _now(self._clock)
                    remaining = min(
                        self._policy.absolute_lifetime_seconds - (now - self._started),
                        self._policy.idle_timeout_seconds - (now - self._last_activity),
                    )
                    if remaining <= 0:
                        self._begin_close_locked()
                        return
                waited = self._waiter(self._wake, remaining)
                if type(waited) is not bool:
                    raise ChatSessionError("session waiter returned an invalid value")
                self._wake.clear()
        except BaseException:
            with self._condition:
                if not self._closed:
                    self._begin_close_locked()

    def _cleanup(self, *, thread_start_failed: bool = False) -> None:
        try:
            closed = self._runtime.close(term_timeout=5.0, kill_timeout=5.0)
            if type(closed) is not bool:
                error = "runtime_close_invalid"
            else:
                error = None if closed else "runtime_unresolved"
                if closed and thread_start_failed:
                    error = "cleanup_thread_start_failed"
        except BaseException:
            error = "runtime_close_failed"
        with self._condition:
            self._close_error = error
            self._close_finished = True
            self._condition.notify_all()

    def _begin_close_locked(self) -> None:
        self._closed = True
        self._wake.set()
        if self._request_event is not None:
            self._request_event.set()
        if self._close_started:
            return
        self._close_started = True
        cleanup = threading.Thread(
            target=self._cleanup,
            name="chat-session-runtime-cleanup",
            daemon=True,
        )
        self._cleanup_thread = cleanup
        try:
            cleanup.start()
        except BaseException:
            self._cleanup(thread_start_failed=True)

    def close(self) -> None:
        with self._condition:
            self._begin_close_locked()

    def wait_closed(
        self, timeout: float = _CONTEXT_CLOSE_WAIT_SECONDS
    ) -> ChatSessionState:
        if type(timeout) not in (int, float) or not math.isfinite(timeout):
            raise TypeError("close wait timeout must be a finite number")
        if not 0 <= timeout <= _MAX_TIMEOUT_SECONDS:
            raise ValueError("close wait timeout is outside its bound")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._close_started and not self._close_finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        return self.state

    def chat(self, content: str) -> BackendResponse:
        if type(content) is not str or not content:
            raise ValueError("chat content must be a non-empty string")
        if len(content) > self._policy.max_history_bytes:
            self.close()
            raise ChatSessionError("chat session history bound reached")
        try:
            content_bytes = len(content.encode("utf-8"))
        except (KeyboardInterrupt, SystemExit):
            self.close()
            raise
        except BaseException:
            self.close()
            raise ChatSessionError("chat content encoding failed") from None
        completed = threading.Event()
        with self._condition:
            try:
                now = _now(self._clock)
            except (KeyboardInterrupt, SystemExit):
                self._begin_close_locked()
                raise
            except BaseException:
                self._begin_close_locked()
                raise ChatSessionError("session clock failed") from None
            if self._expired(now):
                self._begin_close_locked()
            if self._closed:
                raise ChatSessionError("chat session is closed")
            if self._inflight:
                raise ChatSessionError("a chat request is already in flight")
            if self._turns >= self._policy.max_turns:
                self._begin_close_locked()
                raise ChatSessionError("chat session turn bound reached")
            if self._history_bytes + content_bytes > self._policy.max_history_bytes:
                self._begin_close_locked()
                raise ChatSessionError("chat session history bound reached")
            messages = self._history + (
                MappingProxyType({"role": "user", "content": content}),
            )
            self._inflight = True
            self._request_event = completed
            self._last_activity = now
            self._wake.set()

        result: list[object] = []

        def request() -> None:
            try:
                result.append(self._client.chat(messages))
            except BaseException as error:
                result.append(error)
            finally:
                with self._condition:
                    if self._closed:
                        self._inflight = False
                        self._condition.notify_all()
                completed.set()

        worker = threading.Thread(
            target=request,
            name="chat-session-request",
            daemon=True,
        )
        try:
            worker.start()
        except (KeyboardInterrupt, SystemExit):
            with self._condition:
                self._begin_close_locked()
            raise
        except BaseException:
            with self._condition:
                self._begin_close_locked()
            raise ChatSessionError("chat request worker failed") from None
        try:
            finished = self._waiter(completed, self._policy.request_timeout_seconds)
        except BaseException as error:
            with self._condition:
                self._begin_close_locked()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ChatSessionError("session waiter failed") from None
        if type(finished) is not bool:
            with self._condition:
                self._begin_close_locked()
            raise ChatSessionError("session waiter returned an invalid value")
        with self._condition:
            try:
                now = _now(self._clock)
            except (KeyboardInterrupt, SystemExit):
                self._begin_close_locked()
                raise
            except BaseException:
                self._begin_close_locked()
                raise ChatSessionError("session clock failed") from None
            if not finished or not completed.is_set() or self._expired(now):
                self._begin_close_locked()
                raise ChatSessionError("chat request deadline expired")
            self._request_event = None
            self._inflight = False
            if self._closed:
                raise ChatSessionError("chat session closed before response")
            if len(result) != 1:
                self._begin_close_locked()
                raise ChatSessionError("chat request did not return exactly once")
            if isinstance(result[0], BaseException):
                self._begin_close_locked()
                if isinstance(result[0], (KeyboardInterrupt, SystemExit)):
                    raise result[0]
                raise ChatSessionError("backend chat request failed") from None
            response = result[0]
            try:
                remaining = self._policy.max_history_bytes - (
                    self._history_bytes + content_bytes
                )
                assistant, assistant_bytes = _response_text(
                    cast(BackendResponse, response), remaining
                )
            except ChatSessionError:
                self._begin_close_locked()
                raise
            except (KeyboardInterrupt, SystemExit):
                self._begin_close_locked()
                raise
            except BaseException:
                self._begin_close_locked()
                raise ChatSessionError("backend response validation failed") from None
            new_size = self._history_bytes + content_bytes + assistant_bytes
            if new_size > self._policy.max_history_bytes:
                self._begin_close_locked()
                raise ChatSessionError("chat session history bound reached")
            self._history = messages + (
                MappingProxyType({"role": "assistant", "content": assistant}),
            )
            self._history_bytes = new_size
            self._turns += 1
            self._last_activity = now
            self._wake.set()
            return response

    @property
    def state(self) -> ChatSessionState:
        with self._condition:
            runtime_pending = False
            if self._close_started and self._close_finished:
                try:
                    runtime_pending = self._runtime.cleanup_pending
                    if type(runtime_pending) is not bool:
                        runtime_pending = True
                except BaseException:
                    runtime_pending = True
            return ChatSessionState(
                closed=self._closed,
                cleanup_pending=self._close_started
                and (not self._close_finished or runtime_pending),
                request_inflight=self._inflight,
                turns=self._turns,
                history_bytes=self._history_bytes,
                close_error=self._close_error,
            )

    def __enter__(self) -> ChatSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
        state = self.wait_closed(_CONTEXT_CLOSE_WAIT_SECONDS)
        if exc_type is None and (
            state.cleanup_pending or state.close_error is not None
        ):
            raise ChatSessionError("runtime cleanup remains unresolved") from None
