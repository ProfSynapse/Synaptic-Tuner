from __future__ import annotations

import threading
import time
import json

import pytest

from Evaluator import chat_session as chat_session_module
from Evaluator.chat_session import (
    ChatSession,
    ChatSessionError,
    ChatSessionPolicy,
)
from Evaluator.protocols import BackendResponse


class Runtime:
    def __init__(self) -> None:
        self.calls = 0
        self.close_arguments = []
        self.pending = False

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        self.calls += 1
        self.close_arguments.append((term_timeout, kill_timeout))
        return True

    @property
    def cleanup_pending(self) -> bool:
        return self.pending


class Client:
    def __init__(self, response: object = None) -> None:
        self.response = response or BackendResponse("answer", {}, 0.1)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def policy(**changes) -> ChatSessionPolicy:
    values = dict(
        request_timeout_seconds=1.0,
        idle_timeout_seconds=1.0,
        absolute_lifetime_seconds=2.0,
        max_turns=2,
        max_history_bytes=100,
    )
    values.update(changes)
    return ChatSessionPolicy(**values)


def wait_closed(session: ChatSession) -> None:
    deadline = time.monotonic() + 1
    while not session.state.closed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert session.state.closed


def test_chat_uses_generic_history_and_returns_exact_response() -> None:
    runtime = Runtime()
    client = Client()
    session = ChatSession(client, runtime, policy())
    first = session.chat("hello")
    second = session.chat("again")
    assert type(first) is BackendResponse
    assert second.message == "answer"
    assert tuple(dict(message) for message in client.calls[1]) == (
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "again"},
    )
    assert session.state.turns == 2
    session.close()


def test_request_deadline_discards_late_result_and_closes_once() -> None:
    release = threading.Event()

    class Slow(Client):
        def chat(self, messages):
            release.wait()
            return BackendResponse("late", {}, 1.0)

    runtime = Runtime()
    session = ChatSession(Slow(), runtime, policy(request_timeout_seconds=0.01))
    with pytest.raises(ChatSessionError, match="deadline"):
        session.chat("hello")
    assert session.state.request_inflight
    release.set()
    wait_closed(session)
    session.close()
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.calls == 1

    assert session.state.turns == 0


def test_backend_error_and_keyboard_interrupt_close_runtime() -> None:
    for error in (RuntimeError("private detail"), KeyboardInterrupt()):
        runtime = Runtime()
        session = ChatSession(Client(error), runtime, policy())
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                session.chat("hello")
        else:
            with pytest.raises(ChatSessionError, match="failed") as captured:
                session.chat("hello")
            assert captured.value.__cause__ is None
            assert "private detail" not in str(captured.value)
        wait_closed(session)
        assert session.state.turns == 0


def test_caller_keyboard_interrupt_is_preserved_and_closes() -> None:
    runtime = Runtime()

    def interrupt(event, timeout):
        if timeout == 0.123:
            raise KeyboardInterrupt()
        return event.wait(min(timeout, 0.001))

    session = ChatSession(
        Client(), runtime, policy(request_timeout_seconds=0.123), waiter=interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        session.chat("hello")
    wait_closed(session)


def test_ordinary_waiter_failure_is_sanitized_and_closes() -> None:
    runtime = Runtime()

    def fail_request_wait(event, timeout):
        if timeout == 0.123:
            raise RuntimeError("private waiter detail")
        return event.wait(min(timeout, 0.001))

    session = ChatSession(
        Client(),
        runtime,
        policy(request_timeout_seconds=0.123),
        waiter=fail_request_wait,
    )
    with pytest.raises(ChatSessionError, match="waiter failed") as captured:
        session.chat("hello")
    assert captured.value.__cause__ is None
    assert "private" not in str(captured.value)
    wait_closed(session)


def test_constructor_clock_and_watchdog_start_failures_cleanup_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime()

    def bad_clock():
        raise RuntimeError("private clock detail")

    with pytest.raises(ChatSessionError, match="clock failed") as captured:
        ChatSession(Client(), runtime, policy(), clock=bad_clock)
    assert captured.value.__cause__ is None
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.calls == 1

    original_start = threading.Thread.start

    def fail_watchdog(thread):
        if thread.name == "chat-session-watchdog":
            raise RuntimeError("private thread detail")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_watchdog)
    runtime = Runtime()
    with pytest.raises(ChatSessionError, match="watchdog failed") as captured:
        ChatSession(Client(), runtime, policy())
    assert captured.value.__cause__ is None
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.calls == 1


@pytest.mark.parametrize("failure", (RuntimeError("private"), KeyboardInterrupt()))
def test_global_thread_start_failure_uses_direct_cleanup_and_preserves_control(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    runtime = Runtime()

    def fail_every_start(thread):
        raise failure

    monkeypatch.setattr(threading.Thread, "start", fail_every_start)
    expected = (
        KeyboardInterrupt
        if isinstance(failure, KeyboardInterrupt)
        else ChatSessionError
    )
    with pytest.raises(expected) as captured:
        ChatSession(Client(), runtime, policy())
    assert runtime.calls == 1
    assert runtime.close_arguments == [(5.0, 5.0)]
    if expected is ChatSessionError:
        assert captured.value.__cause__ is None
        assert "private" not in str(captured.value)


def test_request_worker_start_failure_and_invalid_waiter_close_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_start = threading.Thread.start

    def fail_request(thread):
        if thread.name == "chat-session-request":
            raise RuntimeError("private thread detail")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_request)
    runtime = Runtime()
    session = ChatSession(Client(), runtime, policy())
    with pytest.raises(ChatSessionError, match="worker failed") as captured:
        session.chat("hello")
    assert captured.value.__cause__ is None
    wait_closed(session)

    monkeypatch.setattr(threading.Thread, "start", original_start)
    runtime = Runtime()
    session = ChatSession(
        Client(),
        runtime,
        policy(request_timeout_seconds=0.123),
        waiter=lambda event, timeout: (
            1 if timeout == 0.123 else event.wait(min(timeout, 0.001))
        ),
    )
    with pytest.raises(ChatSessionError, match="invalid value"):
        session.chat("hello")
    wait_closed(session)


def test_watchdog_rejects_non_bool_waiter_and_closes() -> None:
    runtime = Runtime()
    session = ChatSession(Client(), runtime, policy(), waiter=lambda event, timeout: 1)
    wait_closed(session)
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.calls == 1


def test_invalid_utf8_input_closes_before_backend_call() -> None:
    runtime = Runtime()
    client = Client()
    session = ChatSession(client, runtime, policy())
    with pytest.raises(ChatSessionError, match="encoding failed") as captured:
        session.chat("\ud800")
    assert captured.value.__cause__ is None
    assert client.calls == []
    wait_closed(session)


def test_turn_and_history_bounds_close_without_extra_backend_call() -> None:
    runtime = Runtime()
    client = Client()
    session = ChatSession(client, runtime, policy(max_turns=1))
    session.chat("one")
    with pytest.raises(ChatSessionError, match="turn"):
        session.chat("two")
    assert len(client.calls) == 1

    runtime = Runtime()
    client = Client(BackendResponse("large", {}, 0.1))
    session = ChatSession(client, runtime, policy(max_history_bytes=5))
    with pytest.raises(ChatSessionError, match="bound"):
        session.chat("x")
    assert session.state.turns == 0


def test_oversized_input_and_mapping_response_close_with_zero_history() -> None:
    runtime = Runtime()
    client = Client()
    session = ChatSession(client, runtime, policy(max_history_bytes=4))
    with pytest.raises(ChatSessionError, match="history"):
        session.chat("x" * 5)
    assert client.calls == []
    assert session.state.history_bytes == 0

    runtime = Runtime()
    client = Client(BackendResponse({"content": "x" * 100}, {}, 0.1))
    session = ChatSession(client, runtime, policy(max_history_bytes=10))
    with pytest.raises(ChatSessionError, match="bound"):
        session.chat("x")
    assert session.state.history_bytes == 0
    assert session.state.closed


@pytest.mark.parametrize(
    "message",
    (
        {"content": "a" * 20},
        {"content": 'quote=" slash=\\ newline=\n'},
        {"content": "café 東京"},
    ),
)
def test_mapping_response_accepts_exact_ascii_escape_and_utf8_byte_boundary(
    message,
) -> None:
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    runtime = Runtime()
    session = ChatSession(
        Client(BackendResponse(message, {}, 0.1)),
        runtime,
        policy(max_history_bytes=1 + len(encoded)),
    )
    response = session.chat("x")
    assert response.message == message
    assert session.state.history_bytes == 1 + len(encoded)
    session.close()


def test_at_most_one_inflight_request() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Blocking(Client):
        def chat(self, messages):
            entered.set()
            release.wait()
            return BackendResponse("done", {}, 0.1)

    session = ChatSession(Blocking(), Runtime(), policy())
    result = []
    thread = threading.Thread(target=lambda: result.append(session.chat("one")))
    thread.start()
    assert entered.wait(1)
    with pytest.raises(ChatSessionError, match="already"):
        session.chat("two")
    release.set()
    thread.join(1)
    assert len(result) == 1
    session.close()


def test_explicit_close_during_request_denies_late_history_and_closes_once() -> None:
    entered = threading.Event()
    release = threading.Event()
    errors = []

    class Blocking(Client):
        def chat(self, messages):
            entered.set()
            release.wait()
            return BackendResponse("late", {}, 0.1)

    runtime = Runtime()
    session = ChatSession(Blocking(), runtime, policy())

    def invoke() -> None:
        try:
            session.chat("one")
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(1)
    session.close()
    release.set()
    thread.join(1)
    assert len(errors) == 1
    assert type(errors[0]) is ChatSessionError
    assert session.state.turns == 0
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    session.close()
    assert runtime.calls == 1
    assert runtime.close_arguments == [(5.0, 5.0)]


def test_real_watchdog_closes_idle_session_without_user_action() -> None:
    runtime = Runtime()
    session = ChatSession(
        Client(),
        runtime,
        policy(idle_timeout_seconds=0.02, absolute_lifetime_seconds=1.0),
    )
    wait_closed(session)
    deadline = time.monotonic() + 1
    while runtime.calls != 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert runtime.calls == 1
    assert runtime.close_arguments == [(5.0, 5.0)]


def test_cleanup_pending_and_close_error_are_honest() -> None:
    release = threading.Event()

    class BlockingRuntime(Runtime):
        def close(
            self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
        ) -> bool:
            self.calls += 1
            release.wait()
            return True

    runtime = BlockingRuntime()
    session = ChatSession(Client(), runtime, policy())
    session.close()
    assert session.state.cleanup_pending
    release.set()

    class FailedRuntime(Runtime):
        def close(
            self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
        ) -> bool:
            self.calls += 1
            raise RuntimeError("private")

    failed = FailedRuntime()
    failed_session = ChatSession(Client(), failed, policy())
    failed_session.close()
    deadline = time.monotonic() + 1
    while not failed_session.state.close_error and time.monotonic() < deadline:
        time.sleep(0.005)
    assert failed_session.state.close_error == "runtime_close_failed"
    assert failed.calls == 1

    class UnresolvedRuntime(Runtime):
        def close(
            self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
        ) -> bool:
            self.calls += 1
            self.pending = True
            return False

    unresolved = UnresolvedRuntime()
    unresolved_session = ChatSession(Client(), unresolved, policy())
    unresolved_session.close()
    deadline = time.monotonic() + 1
    while unresolved_session.state.close_error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert unresolved_session.state.close_error == "runtime_unresolved"
    assert unresolved_session.state.cleanup_pending
    unresolved_session.close()
    assert unresolved.calls == 1


def test_context_exit_waits_for_cleanup_and_bounds_unresolved_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    class BlockingRuntime(Runtime):
        def close(
            self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
        ) -> bool:
            self.calls += 1
            self.close_arguments.append((term_timeout, kill_timeout))
            release.wait()
            return True

    runtime = BlockingRuntime()
    releaser = threading.Thread(target=lambda: (time.sleep(0.02), release.set()))
    releaser.start()
    started = time.monotonic()
    with ChatSession(Client(), runtime, policy()):
        pass
    assert time.monotonic() - started >= 0.01
    assert runtime.calls == 1
    releaser.join(1)

    monkeypatch.setattr(chat_session_module, "_CONTEXT_CLOSE_WAIT_SECONDS", 0.01)
    release = threading.Event()
    runtime = BlockingRuntime()
    with pytest.raises(ChatSessionError, match="cleanup remains unresolved"):
        with ChatSession(Client(), runtime, policy()):
            pass
    assert runtime.calls == 1
    release.set()


def test_context_cleanup_failure_does_not_mask_active_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    class BlockingRuntime(Runtime):
        def close(
            self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0
        ) -> bool:
            self.calls += 1
            release.wait()
            return True

    monkeypatch.setattr(chat_session_module, "_CONTEXT_CLOSE_WAIT_SECONDS", 0.01)
    runtime = BlockingRuntime()
    with pytest.raises(RuntimeError, match="active"):
        with ChatSession(Client(), runtime, policy()):
            raise RuntimeError("active")
    release.set()


@pytest.mark.parametrize(
    "changes",
    (
        {"request_timeout_seconds": 0},
        {"idle_timeout_seconds": float("nan")},
        {"absolute_lifetime_seconds": 100_000},
        {"max_turns": True},
        {"max_history_bytes": 0},
    ),
)
def test_policy_rejects_invalid_or_unbounded_values(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        policy(**changes)
