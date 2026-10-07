from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Evaluator import base_client
from Evaluator.base_client import BaseBackendClient
from Evaluator.protocols import BackendError, BackendResponse


class _Error(BackendError):
    pass


class _Client(BaseBackendClient):
    def __init__(self, payload, **kwargs):
        self.payload = payload
        super().__init__(SimpleNamespace(api_key=None), **kwargs)

    @property
    def _client_name(self):
        return "bounded-fixture"

    def _build_payload(self, messages):
        return self.payload

    def _get_chat_url(self):
        return "http://127.0.0.1:8765/v1/chat/completions"

    def _extract_response(self, data, latency_s):
        return BackendResponse(message=data["answer"], raw=data, latency_s=latency_s)

    def _create_error(self, message):
        return _Error(message)


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"answer": "ok"}

    def close(self):
        return None


class _Session:
    instances = []
    failure = None
    callback = None

    def __init__(self):
        self.calls = []
        self.closed = False
        self.trust_env = None
        self.__class__.instances.append(self)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if type(self).callback is not None:
            type(self).callback()
        if type(self).failure is not None:
            raise type(self).failure
        return _Response()

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_session(monkeypatch):
    _Session.instances = []
    _Session.failure = None
    _Session.callback = None
    monkeypatch.setattr(base_client.requests, "Session", _Session)


@pytest.mark.parametrize("value", [True, False, 0, -1, 64 * 1024 * 1024 + 1, 1.0, "1"])
def test_request_bound_requires_bounded_exact_integer(value):
    with pytest.raises(ValueError):
        _Client({}, max_request_bytes=value)


def test_bounded_chat_sends_exact_once_encoded_full_payload():
    payload = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "é\n\""}],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 17,
        "seed": 4,
    }
    expected = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    result = _Client(payload, max_request_bytes=len(expected)).chat([])
    assert result.message == "ok"
    [(method, _, kwargs)] = _Session.instances[0].calls
    assert method == "POST"
    assert kwargs["data"] == expected
    assert "json" not in kwargs
    assert kwargs["headers"] == {"Content-Type": "application/json"}

    _Session.instances = []
    with pytest.raises(ValueError, match="invalid or exceeds"):
        _Client(payload, retries=4, max_request_bytes=len(expected) - 1).chat([])
    assert _Session.instances == []


@pytest.mark.parametrize(
    "payload,bound",
    [
        ({"number": float("nan")}, 1000),
        ({"number": float("inf")}, 1000),
        ({"value": object()}, 1000),
        ({1: "coerced-key"}, 1000),
        ({"number": 1 << 5000}, 32),
    ],
)
def test_invalid_json_is_closed_before_session_without_payload_leak(payload, bound):
    with pytest.raises(ValueError) as caught:
        _Client(payload, retries=5, max_request_bytes=bound).chat([])
    assert _Session.instances == []
    assert "coerced-key" not in str(caught.value)


def test_cycles_and_excessive_depth_are_denied_before_session():
    cyclic = []
    cyclic.append(cyclic)
    deep = None
    for _ in range(66):
        deep = [deep]
    for payload in (cyclic, deep):
        with pytest.raises(ValueError, match="invalid or exceeds"):
            _Client(payload, max_request_bytes=4096).chat([])
    assert _Session.instances == []


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": ["x" * 30, "x" * 30]},
        {"rows": ["\x00" * 10, "\x00" * 10]},
    ],
)
def test_aggregate_and_escaped_size_are_bounded_before_encoding(monkeypatch, payload):
    calls = []
    original = base_client.json.dumps

    def dumps(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(base_client.json, "dumps", dumps)
    with pytest.raises(ValueError, match="invalid or exceeds"):
        _Client(payload, max_request_bytes=50).chat([])
    assert calls == []
    assert _Session.instances == []


def test_payload_mutation_during_request_cannot_change_wire_bytes():
    payload = {"model": "before", "messages": []}
    expected = b'{"model":"before","messages":[]}'
    _Session.callback = lambda: payload.update(model="after", added="large")
    _Client(payload, max_request_bytes=len(expected)).chat([])
    sent = _Session.instances[0].calls[0][2]["data"]
    assert sent == expected
    assert payload["model"] == "after"


def test_mutation_between_preflight_and_encoding_cannot_expand_body(monkeypatch):
    payload = {"model": "before", "messages": []}
    expected = b'{"model":"before","messages":[]}'
    original = base_client.json.dumps
    calls = 0

    def dumps(value, **kwargs):
        nonlocal calls
        calls += 1
        payload["model"] = "x" * 100_000
        return original(value, **kwargs)

    monkeypatch.setattr(base_client.json, "dumps", dumps)
    _Client(payload, max_request_bytes=len(expected)).chat([])
    assert calls == 1
    assert _Session.instances[0].calls[0][2]["data"] == expected


def test_retry_reuses_exact_body_after_caller_mutation(monkeypatch):
    payload = {"model": "before", "messages": []}
    expected = b'{"model":"before","messages":[]}'
    attempts = 0

    def callback():
        nonlocal attempts
        attempts += 1
        payload["model"] = "after"
        if attempts == 1:
            raise base_client.requests.ConnectionError("transient")

    _Session.callback = callback
    monkeypatch.setattr(base_client.time, "sleep", lambda _: None)
    result = _Client(payload, retries=1, max_request_bytes=len(expected)).chat([])
    assert result.message == "ok"
    assert attempts == 2
    assert [session.calls[0][2]["data"] for session in _Session.instances] == [
        expected,
        expected,
    ]
    assert all(session.closed for session in _Session.instances)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        {},
        [],
        (),
        -17,
        -0.0,
        1.25,
        "\"\\\b\f\n\r\t",
        "\x00\x1f",
        "¢",
        "€",
        "😀",
    ],
)
def test_exact_json_byte_boundaries_cover_primitives_escapes_and_unicode(value):
    expected = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    assert base_client._bounded_json_bytes(value, len(expected)) == expected
    with pytest.raises(ValueError, match="invalid or exceeds"):
        base_client._bounded_json_bytes(value, len(expected) - 1)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(3)])
def test_bounded_transport_preserves_control_failures_and_closes_session(failure):
    _Session.failure = failure
    with pytest.raises(type(failure)):
        _Client({}, max_request_bytes=2).chat([])
    assert len(_Session.instances) == 1
    assert _Session.instances[0].closed is True


def test_bounded_policy_keeps_get_bodyless_and_managed():
    client = _Client({}, max_request_bytes=2)
    assert client._request_status("GET", "http://127.0.0.1/health", timeout=1) == 200
    kwargs = _Session.instances[0].calls[0][2]
    assert "data" not in kwargs
    assert "json" not in kwargs
    assert "Content-Type" not in kwargs["headers"]


def test_low_level_bounded_request_encodes_payload_and_rejects_invalid_body():
    client = _Client({}, max_request_bytes=8)
    assert client._request_json("POST", "http://127.0.0.1", payload={"x": 1}) == {
        "answer": "ok"
    }
    assert _Session.instances[0].calls[0][2]["data"] == b'{"x":1}'
    _Session.instances = []
    with pytest.raises(ValueError, match="invalid or exceeds"):
        client._request_json(
            "POST", "http://127.0.0.1", request_body=bytearray(b"{}")
        )
    with pytest.raises(ValueError, match="invalid or exceeds"):
        client._request_json("POST", "http://127.0.0.1", request_body=b"123456789")
    assert _Session.instances == []


def test_unbounded_default_retains_direct_json_request(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(base_client.requests, "post", post)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    _Client(payload).chat([])
    assert calls[0][1]["json"] is payload
    assert _Session.instances == []
