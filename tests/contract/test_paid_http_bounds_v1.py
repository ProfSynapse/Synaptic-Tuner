"""Actual local HTTP only: byte bounds and caller wait do not prove hard abort."""

from contextlib import contextmanager
from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import requests

import test_reference_evaluation_v1 as evaluation
import test_reference_spend_v1 as spend
from shared.llm.providers.openrouter import OpenRouterClient
from shared.llm.providers.openai_responses import OpenAIResponsesClient
from shared.llm.metering import MeteredLLMClient
from shared.llm.usage import UsageAccumulator
from synaptic_tuner.api.v1.reference.evaluation import MeteredPaidBackendV1, MeteredPaidJudgeV1
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationResultRequest,
    EvaluationRunState,
    EvaluationModelRef,
)
from synaptic_tuner.api.v1.reference.spend import SpendEffectState
from synaptic_tuner.api.v1.secrets import SecretRef


@contextmanager
def local_server(body=b"{}", *, status=200, encoding=None, trickle=False):
    state = SimpleNamespace(
        body=body,
        status=status,
        encoding=encoding,
        requests=[],
        started=Event(),
        release=Event(),
        finished=Event(),
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            state.requests.append(
                (self.path, self.rfile.read(size), dict(self.headers))
            )
            if self.path == "/destination":
                state.requests.append(("unexpected-destination", b"", {}))
            self.send_response(state.status)
            self.send_header("Content-Length", str(len(state.body)))
            if state.status == 307:
                self.send_header("Location", "/destination")
            if state.encoding:
                self.send_header("Content-Encoding", state.encoding)
            self.end_headers()
            state.started.set()
            try:
                if trickle:
                    offset = 0
                    while not state.release.wait(0.025) and offset < len(state.body):
                        self.wfile.write(state.body[offset : offset + 1])
                        self.wfile.flush()
                        offset += 1
                    self.wfile.write(state.body[offset:])
                else:
                    self.wfile.write(state.body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                state.finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def recorded_transport(monkeypatch):
    original = requests.Session
    state = SimpleNamespace(
        closed=[], responses=[], closed_event=Event(), redirect=None
    )

    class Session(original):
        def request(self, method, url, *args, **kwargs):
            if state.redirect is not None and url in (
                "https://openrouter.ai/api/v1/chat/completions",
                "https://api.openai.com/v1/responses",
            ):
                url = state.redirect + "/completion"
            assert urlparse(url).hostname == "127.0.0.1", "external HTTP forbidden"
            response = super().request(method, url, *args, **kwargs)
            state.responses.append(response)
            original_close = response.close

            def close():
                state.closed.append("response")
                return original_close()

            response.close = close
            return response

        def close(self):
            try:
                return super().close()
            finally:
                state.closed.append("session")
                state.closed_event.set()

    monkeypatch.setattr(requests, "Session", Session)
    return state


def wire(provider, text="été🍃"):
    usage = (
        {"prompt_tokens": 3, "completion_tokens": 5}
        if provider == "openrouter"
        else {"input_tokens": 3, "output_tokens": 5}
    )
    document = (
        {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": usage,
        }
        if provider == "openrouter"
        else {"output_text": text, "usage": usage}
    )
    return json.dumps(document, ensure_ascii=False).encode("utf-8")


def client(provider, server, **kwargs):
    cls = OpenRouterClient if provider == "openrouter" else OpenAIResponsesClient
    result = cls("synthetic-token", "fixture-model", **kwargs)
    result.api_url = server.url + "/completion"
    return result


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_exact_complete_utf8_response_and_null_tokens(provider, recorded_transport):
    body = wire(provider)
    with local_server(body) as server:
        inner = client(
            provider, server, max_response_bytes=len(body), max_request_bytes=1024
        )
        reply = inner.chat([{"role": "user", "content": "été"}], max_tokens=None)
        assert (
            reply.text == "été🍃"
            and reply.usage.input_tokens == 3
            and reply.usage.output_tokens == 5
        )
        assert len(server.requests) == 1
        sent = json.loads(server.requests[0][1])
        key = "max_tokens" if provider == "openrouter" else "max_output_tokens"
        assert (
            sent[key] is None and server.requests[0][2]["Accept-Encoding"] == "identity"
        )
    assert recorded_transport.closed == ["response", "session"]


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_serialized_request_utf8_boundary_rejects_before_post(
    provider, recorded_transport
):
    with local_server(wire(provider)) as server:
        messages = [{"role": "user", "content": "é🍃"}]
        inner = client(provider, server, max_request_bytes=1024)
        inner.chat(messages, max_tokens=None)
        exact = len(server.requests[0][1])
        client(provider, server, max_request_bytes=exact).chat(
            messages, max_tokens=None
        )
        with pytest.raises(Exception):
            client(provider, server, max_request_bytes=exact - 1).chat(
                messages, max_tokens=None
            )
        assert len(server.requests) == 2
    assert recorded_transport.closed.count("session") == 2


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
@pytest.mark.parametrize(
    "failure",
    ["oversize", "error", "gzip", "redirect", "invalid-json", "NaN", "overflow"],
)
def test_bounded_failure_unknown_and_no_second_paid_dispatch(
    provider, failure, recorded_transport
):
    body = wire(provider)
    status, encoding, cap = 200, None, len(body)
    if failure == "oversize":
        cap -= 1
    if failure == "error":
        status, body, cap = 500, b"private error body" * 1000, 64
    if failure == "gzip":
        encoding, body = "gzip", b"not decoded"
    if failure == "redirect":
        status = 307
    if failure == "invalid-json":
        body = b"{broken"
    if failure == "NaN":
        body = b'{"value":NaN}'
    if failure == "overflow":
        body = b'{"value":1e999}'
    with local_server(body, status=status, encoding=encoding) as server:
        meter = UsageAccumulator()
        wrapped = MeteredLLMClient(
            client(provider, server, max_response_bytes=cap),
            meter,
            stop_on_unknown=True,
        )
        with pytest.raises(Exception):
            wrapped.chat([])
        assert meter.calls == 1 and meter.indeterminate
        with pytest.raises(Exception):
            wrapped.structured_output([], {})
        assert len(server.requests) == 1 and meter.calls == 1
        assert all(row[0] != "/destination" for row in server.requests)
    assert recorded_transport.closed_event.wait(5)
    assert recorded_transport.closed == ["response", "session"]


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_absolute_expired_deadline_never_posts(provider, recorded_transport):
    with local_server(wire(provider)) as server:
        inner = client(provider, server, deadline=time.monotonic() - 1)
        with pytest.raises(Exception):
            inner.chat([])
        assert not server.requests and not recorded_transport.closed


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_slow_trickle_deadline_unknown_late_worker_closes_without_meter_mutation(
    provider, recorded_transport
):
    with local_server(wire(provider, "x" * 1000), trickle=True) as server:
        meter = UsageAccumulator()
        wrapped = MeteredLLMClient(
            client(provider, server, max_response_bytes=4096, timeout_seconds=1),
            meter,
            stop_on_unknown=True,
        )
        started = time.monotonic()
        with pytest.raises(Exception):
            wrapped.chat([])
        elapsed = time.monotonic() - started
        assert server.started.is_set() and elapsed < 5
        assert not server.release.is_set() and not server.finished.is_set()
        assert meter.calls == 1 and meter.indeterminate
        with pytest.raises(Exception):
            wrapped.chat([])
        assert len(server.requests) == 1
        server.release.set()
        assert server.finished.wait(5) and recorded_transport.closed_event.wait(5)
        assert recorded_transport.closed == ["response", "session"]
        assert meter.calls == 1 and meter.indeterminate


@pytest.mark.parametrize(
    "control,value",
    [
        ("max_request_bytes", True),
        ("max_request_bytes", 0),
        ("max_response_bytes", -1),
        ("max_response_bytes", 64 * 1024 * 1024 + 1),
        ("max_response_bytes", 1.5),
        ("deadline", float("inf")),
        ("deadline", True),
    ],
)
def test_invalid_controls_before_credentials_or_dispatch(control, value):
    with pytest.raises((TypeError, ValueError)):
        OpenRouterClient("key", "model", **{control: value})

    class Secrets:
        def resolve(self, reference):
            pytest.fail("invalid controls resolve credentials")

    with pytest.raises((TypeError, ValueError)):
        MeteredPaidBackendV1(
            "openrouter",
            api_key=SecretRef("env", "KEY"),
            secrets=Secrets(),
            account_ref="account",
            namespace_ref="scope",
            **{control: value},
        )


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_public_paid_api_complete_utf8_artifacts_and_late_readback(
    tmp_path, provider, recorded_transport, monkeypatch
):
    class Secrets:
        def resolve(self, reference):
            assert reference == SecretRef("env", "KEY")
            return "synthetic-key"

    with local_server(wire(provider, "hello été🍃")) as server:
        recorded_transport.redirect = server.url
        backend = MeteredPaidBackendV1(
            provider,
            api_key=SecretRef("env", "KEY"),
            secrets=Secrets(),
            account_ref="account",
            namespace_ref="scope",
            max_request_bytes=4096,
            max_response_bytes=4096,
            deadline=time.monotonic() + 30,
        )
        _, api, _, records, _, sink = evaluation._host(
            tmp_path, backends={provider: backend}
        )
        request = replace(
            evaluation._request("alpha.yaml", backend=provider), case_limit=1
        )
        run = api.evaluation.start(api.evaluation.plan(request)).run
        result = api.evaluation.result(EvaluationResultRequest(run))
        assert result.state is EvaluationRunState.SUCCEEDED
        (effect,) = spend._spend_records(records)
        assert effect.state is SpendEffectState.FOUND and result.usage == effect.usage
        assert len(server.requests) == 1
        # Public artifact reader verifies existing role hashes before comparing.
        for artifact in result.artifacts:
            raw = evaluation._artifact_bytes(sink, run, artifact.role)
            import hashlib

            assert (
                len(raw) == artifact.size_bytes
                and hashlib.sha256(raw).hexdigest() == artifact.sha256
            )
        report = json.loads(evaluation._artifact_bytes(sink, run, "evaluation_results"))
        assert "été🍃" in json.dumps(report, ensure_ascii=False)
        recorded_transport.redirect = None
        monkeypatch.setattr(
            MeteredPaidBackendV1,
            "open",
            lambda *args: pytest.fail("result reopened backend"),
        )
        assert api.evaluation.result(EvaluationResultRequest(run)) == result


def test_worker_captures_immutable_request_and_headers(recorded_transport, monkeypatch):
    import threading
    from shared.llm.base import BaseLLMClient

    original = threading.Thread
    ready, release = Event(), Event()

    def delayed(*args, **kwargs):
        if kwargs.get("name") == "bounded-llm-http":
            target = kwargs["target"]

            def run():
                ready.set()
                assert release.wait(5)
                target()

            kwargs["target"] = run
        return original(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", delayed)
    payload, headers, results = (
        {"text": "original"},
        {"Authorization": "original-header"},
        [],
    )
    with local_server(b'{"ok":true}') as server:

        def call():
            results.append(
                BaseLLMClient._bounded_json_post(
                    server.url + "/completion",
                    payload,
                    headers,
                    timeout_seconds=5,
                    controls=(1024, 1024, None),
                )
            )

        owner = original(target=call, daemon=True)
        owner.start()
        try:
            assert ready.wait(5)
            payload["text"] = "changed"
            headers["Authorization"] = "changed"
        finally:
            release.set()
            owner.join(timeout=5)
        assert not owner.is_alive() and results == [{"ok": True}]
        assert json.loads(server.requests[0][1]) == {"text": "original"}
        assert server.requests[0][2]["Authorization"] == "original-header"


def test_delayed_worker_never_dispatches_after_expired_admission(
    recorded_transport, monkeypatch
):
    import threading

    original = threading.Thread
    targets = []

    def delayed(*args, **kwargs):
        if kwargs.get("name") == "bounded-llm-http":
            return SimpleNamespace(start=lambda: targets.append(kwargs["target"]))
        return original(*args, **kwargs)

    with local_server(wire("openrouter")) as server:
        monkeypatch.setattr(threading, "Thread", delayed)
        with pytest.raises(Exception):
            client(
                "openrouter", server, max_response_bytes=1024, timeout_seconds=0.1
            ).chat([])
        assert len(targets) == 1
        targets[0]()
        assert not server.requests and not recorded_transport.closed


def test_modified_retry_adapter_rejected_without_post(recorded_transport, monkeypatch):
    from urllib3.util import Retry

    original = requests.Session

    def unsafe():
        session = original()
        session.adapters["http://"].max_retries = Retry(total=3)
        return session

    monkeypatch.setattr(requests, "Session", unsafe)
    with local_server(wire("openrouter")) as server:
        with pytest.raises(Exception):
            client("openrouter", server, max_response_bytes=1024).chat([])
        assert not server.requests
    assert recorded_transport.closed == ["session"]


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_actual_paid_judge_forwarding_and_complete_account_usage(
    tmp_path, provider, recorded_transport
):
    from shared.judge import RubricDef, JudgeConfig

    class Secrets:
        def resolve(self, reference):
            assert reference == SecretRef("env", "JUDGE")
            return "synthetic-judge-key"

    scores = json.dumps({"quality_score": 0.95, "overall_feedback": "ok"})
    body = wire(provider, scores)
    with local_server(body) as server:
        recorded_transport.redirect = server.url
        port = MeteredPaidJudgeV1(
            provider,
            EvaluationModelRef("judge", "revision"),
            api_key=SecretRef("env", "JUDGE"),
            secrets=Secrets(),
            account_ref="account",
            namespace_ref="scope",
            rubrics=[
                RubricDef(
                    "quality",
                    "Quality",
                    "fixture",
                    "response",
                    0.8,
                    "Score {response}.",
                    {"type": "object"},
                )
            ],
            judge_config=JudgeConfig(max_tokens=123),
            max_request_bytes=4096,
            max_response_bytes=len(body),
            deadline=time.monotonic() + 30,
        )
        _, api, _, records, _, _ = evaluation._host(
            tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=port
        )
        run = api.evaluation.start(
            api.evaluation.plan(evaluation._request("alpha.yaml"))
        ).run
        result = api.evaluation.result(EvaluationResultRequest(run))
        assert result.state is EvaluationRunState.SUCCEEDED
        (effect,) = spend._spend_records(records)
        assert effect.state is SpendEffectState.FOUND and result.usage == effect.usage
        assert (result.usage.input_tokens, result.usage.output_tokens) == (6, 10)
        assert len(server.requests) == 2
        assert all(row[2]["Accept-Encoding"] == "identity" for row in server.requests)
