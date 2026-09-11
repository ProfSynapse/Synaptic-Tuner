"""Provider-free tests for the bounded Modal chat stdio channel."""

from __future__ import annotations

from io import BytesIO
import json
import threading

import pytest

from Evaluator.chat_session import ChatSession, ChatSessionPolicy
from Evaluator.protocols import BackendResponse
from tuner.inference.run_chat import PreparedModelIdentity
from tuner.execution.providers.modal import inference_channel as channel

_ARGUMENT = b"authenticated-launch"
_SESSION = "chat-session-1"
_DIGEST = "4ecf96be6ab8a8450fcf5a59db93fa2e1b27418c12d8085d876e6e631a59b525"
_MODEL = PreparedModelIdentity("fixture/model", "a" * 40, "a" * 40, "lora")


class Runtime:
    def __init__(self):
        self.closed = threading.Event()

    def close(self, *, term_timeout=5.0, kill_timeout=5.0):
        self.closed.set()
        return True

    @property
    def cleanup_pending(self):
        return not self.closed.is_set()


class Client:
    def __init__(self, *, failure=None, reply="reply"):
        self.failure = failure
        self.reply = reply
        self.calls = []

    def chat(self, messages):
        self.calls.append(tuple(dict(item) for item in messages))
        if self.failure is not None:
            raise self.failure
        return BackendResponse(self.reply, {"value": "retained"}, 0.01)


def _session(*, client=None, idle=60.0):
    runtime = Runtime()
    client = client or Client()
    session = ChatSession(
        client,
        runtime,
        ChatSessionPolicy(1.0, idle, 60.0, 4, 4096),
    )
    return session, client, runtime


def _frame(kind, request_id=None, **values):
    value = {
        "schema_version": "synaptic-modal-chat-channel/v1",
        "kind": kind,
        "session_id": _SESSION,
        "launch_digest": _DIGEST,
        **values,
    }
    if request_id is not None:
        value["request_id"] = request_id
    if kind == "ready":
        value["model"] = {
            name: getattr(_MODEL, name) for name in _MODEL.__dataclass_fields__
        }
    return channel.encode_modal_chat_frame(value, 4096)


def _serve(session, source, *, request_max=4096, response_max=4096):
    output = BytesIO()
    channel.serve_modal_chat_channel(
        session,
        source,
        output,
        model=_MODEL,
        session_id=_SESSION,
        argument_bytes=_ARGUMENT,
        max_request_bytes=request_max,
        max_response_bytes=response_max,
    )
    return [
        channel.decode_modal_chat_frame(line + b"\n", response_max)
        for line in output.getvalue().splitlines()
    ]


def test_chat_then_stop_uses_one_session_and_exact_bindings():
    session, client, runtime = _session()
    frames = _serve(
        session,
        BytesIO(_frame("chat", 1, content="hello") + _frame("stop", 2)),
    )
    assert [item["kind"] for item in frames] == ["ready", "chat", "closed"]
    assert frames[1] == {
        "schema_version": "synaptic-modal-chat-channel/v1",
        "kind": "chat",
        "session_id": _SESSION,
        "launch_digest": _DIGEST,
        "request_id": 1,
        "content": "reply",
    }
    assert client.calls == [({"role": "user", "content": "hello"},)]
    assert runtime.closed.wait(1)


def test_eof_closes_without_a_backend_request():
    session, client, runtime = _session()
    frames = _serve(session, BytesIO())
    assert [item["kind"] for item in frames] == ["ready", "closed"]
    assert frames[-1]["request_id"] == 1
    assert client.calls == []
    assert runtime.closed.wait(1)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"kind":"chat","kind":"chat"}\n',
        b'{"content":1,"kind":"chat","launch_digest":"'
        + _DIGEST.encode()
        + b'","request_id":1,"schema_version":"synaptic-modal-chat-channel/v1",'
        + b'"session_id":"chat-session-1"}\n',
        b'{"content":"x","kind":"chat","launch_digest":"'
        + _DIGEST.encode()
        + b'","request_id":1.0,"schema_version":"synaptic-modal-chat-channel/v1",'
        + b'"session_id":"chat-session-1"}\n',
        b'{"content":{"nested":true},"kind":"chat"}\n',
    ],
)
def test_malformed_partial_duplicate_or_nonprimitive_frame_is_closed(payload):
    session, client, runtime = _session()
    frames = _serve(session, BytesIO(payload))
    assert [item["kind"] for item in frames] == ["ready", "error"]
    assert frames[-1]["code"] == "modal_chat_channel_error"
    assert client.calls == []
    assert runtime.closed.wait(1)


@pytest.mark.parametrize(
    "payload",
    [
        _frame("chat", 2, content="skipped"),
        _frame("chat", 1, content="wrong").replace(_SESSION.encode(), b"other-session"),
        _frame("chat", 1, content="wrong").replace(_DIGEST.encode(), b"f" * 64),
        _frame("ready"),
    ],
)
def test_wrong_order_binding_or_direction_is_denied(payload):
    session, client, runtime = _session()
    frames = _serve(session, BytesIO(payload))
    assert [item["kind"] for item in frames] == ["ready", "error"]
    assert client.calls == []
    assert runtime.closed.wait(1)


def test_oversize_input_is_read_with_a_bounded_readline_and_denied():
    class Input:
        def __init__(self):
            self.sizes = []

        def readline(self, size):
            self.sizes.append(size)
            return b"x" * size

    source = Input()
    session, client, runtime = _session()
    frames = _serve(session, source, request_max=128)
    assert 1 <= len(source.sizes) <= 2
    assert set(source.sizes) == {129}
    assert [item["kind"] for item in frames] == ["ready", "error"]
    assert client.calls == []
    assert runtime.closed.wait(1)


def test_backend_failure_is_sanitized_and_closes():
    client = Client(failure=RuntimeError("private backend detail"))
    session, _, runtime = _session(client=client)
    frames = _serve(session, BytesIO(_frame("chat", 1, content="hello")))
    encoded = json.dumps(frames)
    assert [item["kind"] for item in frames] == ["ready", "error"]
    assert "private" not in encoded and "backend" not in encoded
    assert runtime.closed.wait(1)


def test_oversize_response_becomes_a_closed_error_frame():
    session, _, runtime = _session(client=Client(reply="x" * 4096))
    frames = _serve(
        session,
        BytesIO(_frame("chat", 1, content="hello")),
        response_max=512,
    )
    assert [item["kind"] for item in frames] == ["ready", "error"]
    assert runtime.closed.wait(1)


def test_session_watchdog_closes_while_input_read_remains_blocked():
    release = threading.Event()

    class Input:
        def readline(self, size):
            release.wait(2)
            return b""

    session, client, runtime = _session(idle=0.05)
    try:
        frames = _serve(session, Input())
        assert [item["kind"] for item in frames] == ["ready", "closed"]
        assert client.calls == []
        assert runtime.closed.wait(1)
    finally:
        release.set()


def test_session_watchdog_releases_loop_while_output_write_remains_blocked():
    release = threading.Event()

    class Output:
        def write(self, payload):
            release.wait(2)
            return len(payload)

        def flush(self):
            pass

    session, client, runtime = _session(idle=0.05)
    try:
        channel.serve_modal_chat_channel(
            session,
            BytesIO(),
            Output(),
            model=_MODEL,
            session_id=_SESSION,
            argument_bytes=_ARGUMENT,
            max_request_bytes=4096,
            max_response_bytes=4096,
        )
        assert client.calls == []
        assert runtime.closed.wait(1)
    finally:
        release.set()


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
def test_input_control_flow_is_preserved_and_session_is_closed(control):
    class Input:
        def readline(self, size):
            raise control("control")

    session, _, runtime = _session()
    with pytest.raises(control):
        _serve(session, Input())
    assert runtime.closed.wait(1)


def test_codec_rejects_noncanonical_and_nonfinite_frames():
    with pytest.raises(ValueError):
        channel.decode_modal_chat_frame(
            b'{"schema_version":"synaptic-modal-chat-channel/v1","kind":"chat",'
            b'"session_id":"chat-session-1","launch_digest":"'
            + _DIGEST.encode()
            + b'","request_id":1,"content":NaN}\n',
            4096,
        )
    with pytest.raises(ValueError):
        channel.decode_modal_chat_frame(
            _frame("chat", 1, content="hello").replace(b'"chat"', b'"chat" '),
            4096,
        )
