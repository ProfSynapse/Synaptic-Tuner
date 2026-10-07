from __future__ import annotations

import json
import io
from types import MappingProxyType

import pytest
import requests

from Evaluator.config import VLLMSettings
from Evaluator.vllm_client import VLLMClient, VLLMError


def _settings() -> VLLMSettings:
    return VLLMSettings(
        model="fixture",
        scheme="http",
        host="127.0.0.1",
        port=8000,
        api_key=None,
        max_tokens=8,
    )


class Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("closed fixture")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size: int):
        yield json.dumps(self.payload).encode()

    def close(self) -> None:
        self.closed = True


class Session:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.trust_env = True
        self.closed = False
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def close(self) -> None:
        self.closed = True


def test_defaults_preserve_module_requests_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    response = Response({"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(
        "Evaluator.base_client.requests.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or response,
    )
    client = VLLMClient(_settings(), retries=0)
    assert client.chat([{"role": "user", "content": "hi"}]).message == "ok"
    assert calls[0][1]["json"]["model"] == "fixture"


def test_default_list_and_health_preserve_module_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [Response({"data": [{"id": "fixture"}]}), Response({}, 200)]
    calls = []
    monkeypatch.setattr(
        "Evaluator.base_client.requests.get",
        lambda url, **kwargs: calls.append((url, kwargs)) or responses.pop(0),
    )
    client = VLLMClient(_settings(), retries=0)
    assert client.list_models() == ["fixture"]
    assert client.is_server_running()
    assert len(calls) == 2


def test_restricted_chat_disables_environment_redirects_and_bounds_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response({"choices": [{"message": {"content": "ok"}}]})
    session = Session(response)
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=1024,
    )
    assert client.chat([{"role": "user", "content": "hi"}]).message == "ok"
    assert session.trust_env is False
    assert session.calls[0][1]["allow_redirects"] is False
    assert session.calls[0][1]["headers"] == {}
    assert response.closed and session.closed
    with pytest.raises(AttributeError):
        client.trust_environment = True


@pytest.mark.parametrize("value", [True, 0, -1, 64 * 1024 * 1024 + 1, 1.5])
def test_invalid_response_bounds_rejected_before_io(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VLLMClient(_settings(), max_response_bytes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["trust_environment", "allow_redirects"])
def test_transport_flags_require_exact_bool(field: str) -> None:
    with pytest.raises(TypeError):
        VLLMClient(_settings(), **{field: 1})


def test_redirect_and_oversize_close_response_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for response, message in (
        (Response({}, 302), "redirect"),
        (Response("x" * 100), "bound"),
    ):
        session = Session(response)
        monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
        client = VLLMClient(
            _settings(),
            retries=0,
            trust_environment=False,
            allow_redirects=False,
            max_response_bytes=8,
        )
        with pytest.raises(VLLMError, match=message):
            client.chat([{"role": "user", "content": "hi"}])
        assert response.closed and session.closed


def test_keyboard_interrupt_during_stream_still_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response({})
    response.iter_content = lambda chunk_size: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )
    session = Session(response)
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=8,
    )
    with pytest.raises(KeyboardInterrupt):
        client.chat([{"role": "user", "content": "hi"}])
    assert response.closed and session.closed


def test_transport_failure_before_response_still_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(Response({}))
    session.request = lambda *args, **kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("closed fixture")
    )
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=8,
    )
    with pytest.raises(VLLMError):
        client.chat([{"role": "user", "content": "hi"}])
    assert session.closed


def test_list_and_health_share_restricted_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [Response({"data": [{"id": "fixture"}]}), Response({}, 204)]
    sessions = []

    def factory():
        session = Session(responses[len(sessions)])
        sessions.append(session)
        return session

    monkeypatch.setattr("Evaluator.base_client.requests.Session", factory)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=1024,
    )
    assert client.list_models() == ["fixture"]
    assert not client.is_server_running()
    assert all(
        not item.trust_env and item.closed and item.response.closed for item in sessions
    )


def test_list_interrupt_and_health_failure_close_owned_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response({})
    response.iter_content = lambda chunk_size: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )
    first = Session(response)
    second = Session(Response({}))
    second.request = lambda *args, **kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("closed")
    )
    sessions = iter((first, second))
    monkeypatch.setattr(
        "Evaluator.base_client.requests.Session", lambda: next(sessions)
    )
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=8,
    )
    with pytest.raises(KeyboardInterrupt):
        client.list_models()
    assert first.closed and response.closed
    assert not client.is_server_running()
    assert second.closed


@pytest.mark.parametrize("close_target", ["response", "session"])
def test_cleanup_failure_attempts_both_closes_and_does_not_mask_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    close_target: str,
) -> None:
    response = Response({})
    session = Session(response)
    response.iter_content = lambda chunk_size: (_ for _ in ()).throw(
        KeyboardInterrupt()
    )
    response_closed = []
    session_closed = []
    response.close = lambda: (
        (
            response_closed.append(1),
            (_ for _ in ()).throw(RuntimeError("response close")),
        )[-1]
        if close_target == "response"
        else response_closed.append(1)
    )
    session.close = lambda: (
        (
            session_closed.append(1),
            (_ for _ in ()).throw(RuntimeError("session close")),
        )[-1]
        if close_target == "session"
        else session_closed.append(1)
    )
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=8,
    )
    with pytest.raises(KeyboardInterrupt):
        client.chat([{"role": "user", "content": "hi"}])
    assert response_closed == [1]
    assert session_closed == [1]


def test_cleanup_failure_without_active_error_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Response({"choices": [{"message": {"content": "ok"}}]})
    session = Session(response)
    session_closed = []
    response.close = lambda: (_ for _ in ()).throw(RuntimeError("sensitive"))
    session.close = lambda: session_closed.append(1)
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=1024,
    )
    with pytest.raises(VLLMError, match="transport cleanup failed"):
        client.chat([{"role": "user", "content": "hi"}])
    assert session_closed == [1]


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit()])
@pytest.mark.parametrize("ordinary_first", [False, True])
def test_cleanup_control_signal_is_preserved_after_both_close_attempts(
    monkeypatch: pytest.MonkeyPatch,
    control: BaseException,
    ordinary_first: bool,
) -> None:
    response = Response({"choices": [{"message": {"content": "ok"}}]})
    session = Session(response)
    response_calls = []
    session_calls = []
    if ordinary_first:
        response.close = lambda: (
            response_calls.append(1),
            (_ for _ in ()).throw(RuntimeError("ordinary")),
        )[-1]
        session.close = lambda: (
            session_calls.append(1),
            (_ for _ in ()).throw(control),
        )[-1]
    else:
        response.close = lambda: (
            response_calls.append(1),
            (_ for _ in ()).throw(control),
        )[-1]
        session.close = lambda: session_calls.append(1)
    monkeypatch.setattr("Evaluator.base_client.requests.Session", lambda: session)
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=1024,
    )
    with pytest.raises(type(control)):
        client.chat([{"role": "user", "content": "hi"}])
    assert response_calls == [1]
    assert session_calls == [1]


def test_real_session_restricted_mode_ignores_proxy_and_netrc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')
    monkeypatch.setenv("HTTP_PROXY", "http://ambient-proxy")
    monkeypatch.setenv("NETRC", "/ambient/netrc")
    monkeypatch.setattr(
        requests.sessions,
        "get_netrc_auth",
        lambda url: (_ for _ in ()).throw(AssertionError("netrc consulted")),
    )
    monkeypatch.setattr(
        requests.adapters.HTTPAdapter,
        "send",
        lambda adapter, request, **kwargs: seen.append((request, kwargs)) or response,
    )
    client = VLLMClient(
        _settings(),
        retries=0,
        trust_environment=False,
        allow_redirects=False,
        max_response_bytes=1024,
    )
    assert (
        client.chat([MappingProxyType({"role": "user", "content": "hi"})]).message
        == "ok"
    )
    request, kwargs = seen[0]
    assert "Authorization" not in request.headers
    assert kwargs["proxies"] == {}
    encoded = json.loads(request.body)
    assert encoded["model"] == "fixture"
    assert encoded["messages"] == [{"role": "user", "content": "hi"}]
