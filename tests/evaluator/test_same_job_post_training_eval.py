from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import threading

import pytest

from Evaluator.protocols import BackendResponse, BackendError, RequestFailureCode
from Evaluator import vllm_client
from tuner.inference import vllm_runtime
from tuner.runtime import post_training_eval
from tuner.training.post_training import validate_post_training_config


def _config():
    return {
        "mode": "same_job",
        "evaluation": {
            "scenarios": [
                {"id": "one", "question": "Say ready", "correct": {
                    "assertions": [{"type": "jsonpath_equals", "path": "$.content", "value": "ready"}]
                }},
                {"id": "two", "question": "Say ready again", "correct": {
                    "assertions": [{"type": "jsonpath_equals", "path": "$.content", "value": "ready"}]
                }},
            ],
            "min_pass_rate": 1.0,
            "max_cases": 2,
            "startup_timeout_seconds": 10,
            "timeout_seconds": 30,
            "served_model_name": "new-adapter",
            "generation": {"max_tokens": 32, "temperature": 0, "top_p": 1},
            "vllm": {
                "expected_version": "0.23.0", "dtype": "bfloat16",
                "max_model_len": 1024, "tensor_parallel_size": 1,
                "max_num_seqs": 4, "max_num_batched_tokens": 1024,
                "language_model_only": True, "max_lora_rank": 64,
            },
        },
    }


def _bindings():
    return {key: character * 64 for key, character in (
        ("workload_digest", "a"), ("model_snapshot_digest", "b"),
        ("adapter_digest", "c"),
    )}


def _metric_body():
    return (b'vllm:num_requests_running{model_name="private"} 2\n'
            b'vllm:num_requests_waiting 0\n'
            b'vllm:generation_tokens_total 1203\n'
            b'vllm:kv_cache_usage_perc 0.25\n')


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nan", "fraction", "negative", "huge", "oversized", "line"])
def test_serving_metrics_parser_rejects_ambiguous_or_unbounded_data(mutation):
    raw = _metric_body()
    if mutation == "duplicate": raw += b'vllm:num_requests_running{model_name="other"} 1\n'
    if mutation == "missing": raw = raw.replace(b'vllm:num_requests_waiting 0\n', b'')
    if mutation == "nan": raw = raw.replace(b'1203', b'NaN')
    if mutation == "fraction": raw = raw.replace(b'0.25', b'1.1')
    if mutation == "negative": raw = raw.replace(b'1203', b'-1')
    if mutation == "huge": raw = raw.replace(b'1203', b'9007199254740992')
    if mutation == "oversized": raw = b'x' * 262145
    if mutation == "line": raw += b'#' + b'x' * 4096
    with pytest.raises(ValueError):
        post_training_eval._parse_serving_metrics(raw)


@pytest.mark.parametrize("mode", ["valid", "redirect", "duplicate_length", "oversized", "headers", "truncated", "trickle"])
def test_serving_metrics_real_socket_fixed_request_and_bounds(mode):
    import socketserver
    import time
    requests = []
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            requests.append(self.request.recv(4096))
            body = _metric_body()
            header = b'HTTP/1.0 200 OK\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n'
            if mode == "redirect": header = b'HTTP/1.0 302 Found\r\nContent-Length: 0\r\n\r\n'; body = b''
            if mode == "duplicate_length": header = header.replace(b'\r\n\r\n', b'\r\nContent-Length: 1\r\n\r\n')
            if mode == "oversized": header = b'HTTP/1.0 200 OK\r\nContent-Length: 262145\r\n\r\n'
            if mode == "headers": header = b'HTTP/1.0 200 OK\r\nX: ' + b'x' * 5000
            if mode == "truncated": body = body[:-1]
            try:
                if mode == "trickle":
                    for _ in range(30):
                        self.request.sendall(b'H')
                        time.sleep(0.05)
                else:
                    self.request.sendall(header + body)
            except OSError:
                pass
    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        if mode == "valid":
            assert post_training_eval._read_serving_metrics(server.server_address[1], threading.Event()) == {
                "running_requests": 2, "waiting_requests": 0, "generation_tokens": 1203, "kv_cache_usage": .25}
        else:
            with pytest.raises((ValueError, OSError)):
                post_training_eval._read_serving_metrics(server.server_address[1], threading.Event())
        assert time.monotonic() - started < 1.5
        assert requests == [b'GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n']
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


def test_serving_metrics_real_prometheus_exposition_discards_labels():
    prometheus = pytest.importorskip("prometheus_client")
    registry = prometheus.CollectorRegistry()
    for name, value in (("num_requests_running", 2), ("num_requests_waiting", 1), ("kv_cache_usage_perc", .5)):
        prometheus.Gauge("vllm:" + name, "private help", ["model_name"], registry=registry).labels("private model").set(value)
    prometheus.Counter("vllm:generation_tokens", "private help", ["model_name"], registry=registry).labels("private model").inc(15)
    raw = prometheus.generate_latest(registry)
    assert b"vllm:generation_tokens_total" in raw
    values = post_training_eval._parse_serving_metrics(raw)
    assert values == {"running_requests": 2, "waiting_requests": 1, "generation_tokens": 15, "kv_cache_usage": .5}
    assert "private" not in json.dumps(values)


def test_serving_metrics_actual_instrumentator_http_route_contract():
    # Local-library contract coverage, not pinned remote vLLM qualification.
    import http.client
    import re
    import socket
    import time
    fastapi = pytest.importorskip("fastapi")
    prometheus = pytest.importorskip("prometheus_client")
    instrumentator = pytest.importorskip("prometheus_fastapi_instrumentator")
    uvicorn = pytest.importorskip("uvicorn")
    from starlette.routing import Mount
    registry = prometheus.CollectorRegistry()
    for name, value in (("num_requests_running", 2), ("num_requests_waiting", 0), ("kv_cache_usage_perc", .25)):
        prometheus.Gauge("vllm:" + name, "help", ["model_name"], registry=registry).labels("private").set(value)
    prometheus.Counter("vllm:generation_tokens", "help", registry=registry).inc(1203)
    class PrometheusResponse(fastapi.Response):
        media_type = prometheus.CONTENT_TYPE_LATEST
    app = fastapi.FastAPI()
    # Match vLLM0.26 attach_router: expose first, then patched ASGI Mount.
    instrumentator.Instrumentator(excluded_handlers=["/metrics", "/health", "/load", "/ping", "/version", "/server_info"], registry=registry).add().instrument(app).expose(app, response_class=PrometheusResponse)
    route = Mount("/metrics", prometheus.make_asgi_app(registry=registry))
    route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(route)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False, lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/metrics")
            response = connection.getresponse()
            assert response.status == 200  # no 307 slash redirect
            assert response.getheader("Content-Length") is not None
            assert response.getheader("Content-Encoding") is None
            assert response.getheader("Transfer-Encoding") is None
            response.read()
        finally:
            connection.close()
        assert post_training_eval._read_serving_metrics(port, threading.Event()) == {
            "running_requests": 2, "waiting_requests": 0, "generation_tokens": 1203, "kv_cache_usage": .25}
    finally:
        server.should_exit = True
        thread.join(3)
        listener.close()
        assert not thread.is_alive()


def test_serving_sampler_schedule_cap_and_errors_are_best_effort(monkeypatch):
    class Lease:
        host, port = "127.0.0.1", 8000
    events = []
    sampler = post_training_eval._ServingSampler(Lease(), lambda kind, values: events.append(values))
    class Stop:
        def is_set(self): return False
        def wait(self, seconds):
            assert seconds == 15
            return False
    sampler._stop = Stop()
    monkeypatch.setattr(post_training_eval, "_read_serving_metrics", lambda *args: post_training_eval._parse_serving_metrics(_metric_body()))
    sampler._run()
    assert len(events) == 64
    monkeypatch.setattr(post_training_eval, "_read_serving_metrics", lambda *args: (_ for _ in ()).throw(OSError("private")))
    sampler._run()
    assert len(events) == 64


@pytest.mark.parametrize("outcome", [True, False, "exception"])
def test_serving_cleanup_projection_only_reports_normal_boolean_return(tmp_path, monkeypatch, outcome):
    events = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def chat(self, messages): return BackendResponse(message="ready", raw={}, latency_s=.1)
    class Lease:
        served_model_name, host, port = "new-adapter", "127.0.0.1", 8000
        def close(self):
            if outcome == "exception": raise OSError("private cleanup")
            return outcome
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    monkeypatch.setattr(post_training_eval._ServingSampler, "start", lambda self: None)
    monkeypatch.setattr(post_training_eval._ServingSampler, "finish", lambda self: None)
    def run():
        return post_training_eval.execute_post_training_evaluation(
            _config(), base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
            validate=lambda: None, environment={}, cwd=tmp_path, python_executable=sys.executable,
            bindings=_bindings(), serving_callback=lambda kind, values: events.append((kind, values)))
    if outcome is True:
        assert run()["gate_passed"] is True
    else:
        with pytest.raises(post_training_eval.PostTrainingCleanupUnresolved): run()
    assert events == ([] if outcome == "exception" else [("CLEANUP", {"cleanup_resolved": outcome})])


def test_real_trace_blocked_metrics_sink_does_not_block_evaluation_cleanup(tmp_path, monkeypatch):
    from tuner.execution.providers.modal.packaged_worker import _PackagedPhaseTrace
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def sink(line):
        if line.startswith("SYNAPTIC_SERVING "):
            entered.set()
            assert release.wait(5)
    trace = _PackagedPhaseTrace(sink=sink)
    monkeypatch.setattr(post_training_eval, "_read_serving_metrics", lambda *args: post_training_eval._parse_serving_metrics(_metric_body()))
    class Client:
        def __init__(self, *args, **kwargs): pass
        def chat(self, messages):
            assert entered.wait(2)
            return BackendResponse(message="ready", raw={}, latency_s=.1)
    class Lease:
        host, port, served_model_name = "127.0.0.1", 8000, "new-adapter"
        def close(self):
            assert entered.is_set() and not release.is_set()
            closed.set()
            return True
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    finished = threading.Event()
    outcomes = []
    def evaluate():
        try:
            outcomes.append(post_training_eval.execute_post_training_evaluation(
                _config(), base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
                validate=lambda: None, environment={}, cwd=tmp_path, python_executable=sys.executable,
                bindings=_bindings(), phase_callback=trace.emit, serving_callback=trace.emit_serving))
        except BaseException as error:
            outcomes.append(error)
        finally:
            finished.set()
    evaluator = threading.Thread(target=evaluate, daemon=True)
    try:
        evaluator.start()
        assert finished.wait(2), "diagnostic sink blocked evaluation cleanup/return"
        assert closed.is_set() and outcomes[0]["gate_passed"] is True
        assert not release.is_set()
    finally:
        release.set()
        evaluator.join(3)
        assert not evaluator.is_alive()


def test_serving_sampler_stall_cannot_block_cleanup(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def stalled(*args):
        entered.set()
        release.wait(5)
        return post_training_eval._parse_serving_metrics(_metric_body())
    monkeypatch.setattr(post_training_eval, "_read_serving_metrics", stalled)
    events = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def chat(self, messages):
            assert entered.wait(2)
            return BackendResponse(message="ready", raw={}, latency_s=.1)
    class Lease:
        host, port, served_model_name = "127.0.0.1", 8000, "new-adapter"
        def close(self):
            assert not release.is_set()
            events.append("closed")
            return True
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    try:
        record = post_training_eval.execute_post_training_evaluation(
            _config(), base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
            validate=lambda: None, environment={}, cwd=tmp_path, python_executable=sys.executable,
            bindings=_bindings(), serving_callback=lambda kind, values: events.append((kind, values)))
        assert record["gate_passed"] is True
        assert events == ["closed", ("CLEANUP", {"cleanup_resolved": True})]
    finally:
        release.set()


@pytest.mark.parametrize("fast_error", [False, True])
def test_phase_trace_request_edges_are_immediate_and_batch_drains_before_cleanup(tmp_path, monkeypatch, fast_error):
    events = []
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)
    fast_traced = threading.Event()
    def trace(phase, edge, ordinal):
        with lock:
            events.append((phase, edge, ordinal))
        if phase == "CHAT_REQUEST" and edge in {"RETURN", "ERROR"}:
            fast_traced.set()
    class Client:
        def __init__(self, *_args, **_kwargs): pass
        def chat(self, messages):
            barrier.wait()
            if messages[-1]["content"] == "Say ready":
                assert fast_traced.wait(5), "fast edge was blocked by ordered case emission"
            elif fast_error:
                raise BackendError("private prompt/token/path")
            return BackendResponse(message="ready", raw={}, latency_s=0.1)
    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000
        def close(self):
            assert events[-1] == ("VLLM_CLEANUP", "START", None)
            assert ("CHAT_BATCH", "RETURN", None) in events
            assert len([event for event in events if event[0] == "CHAT_REQUEST" and event[1] != "START"]) == 2
            return True
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *_args, **_kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        _config(), base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
        validate=lambda: None, environment={}, cwd=tmp_path, python_executable=sys.executable,
        bindings=_bindings(), phase_callback=trace)
    starts = [ordinal for phase, edge, ordinal in events if phase == "CHAT_REQUEST" and edge == "START"]
    assert sorted(starts) == [1, 2]
    assert ("VLLM_CLEANUP", "RETURN", None) in events
    assert record["passed_count"] == (1 if fast_error else 2)
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("fault", ["callback", "validate", "cleanup"])
def test_phase_callback_errors_preserve_evaluation_and_cleanup_outcomes(tmp_path, monkeypatch, fault):
    events = []
    def trace(*event):
        if fault == "callback": raise OSError("private callback")
        events.append(event)
    class Client:
        def __init__(self, *_args, **_kwargs): pass
        def chat(self, _messages): return BackendResponse(message="ready", raw={}, latency_s=0.1)
    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000
        def close(self):
            if fault == "cleanup": raise OSError("private cleanup")
            return True
    def validate():
        if fault == "validate": raise ValueError("private model identity")
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *_args, **_kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    args = dict(base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
        validate=validate, environment={}, cwd=tmp_path, python_executable=sys.executable,
        bindings=_bindings(), phase_callback=trace)
    if fault == "cleanup":
        with pytest.raises(post_training_eval.PostTrainingCleanupUnresolved):
            post_training_eval.execute_post_training_evaluation(_config(), **args)
        assert ("VLLM_CLEANUP", "ERROR", None) in events
    else:
        record = post_training_eval.execute_post_training_evaluation(_config(), **args)
        assert record["gate_passed"] is (fault == "callback")
        if fault == "validate":
            assert ("EVALUATION_IDENTITY_VALIDATE", "ERROR", None) in events
            assert ("EVALUATION_PREPARE", "ERROR", None) in events
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("failures,expected", [
    (("timeout", "connection", "http400"), ("request_timeout", "request_connection", "http_400")),
    (("http_other", "value", "backend"), ("http_other", "request_validation", "request_backend")),
    (("request", "unknown", "hostile_status"), ("request_transport", "request_unknown", "request_unknown")),
])
def test_real_http_failures_have_closed_ordered_signed_codes_and_cleanup(tmp_path, monkeypatch, failures, expected):
    import requests
    from types import SimpleNamespace
    from Evaluator import base_client
    from tests.evaluator.test_http_transport_policy import Session, Response
    secret = "HF_TOKEN=private https://private.example/customer /home/private prompt-body"
    sessions = []
    class HostileStatus:
        @property
        def status_code(self):
            raise ValueError(secret)
    errors = {
        "timeout": requests.Timeout(secret), "connection": requests.ConnectionError(secret),
        "http400": requests.HTTPError(secret, response=SimpleNamespace(status_code=400)),
        "http_other": requests.HTTPError(secret, response=SimpleNamespace(status_code=499)),
        "value": ValueError(secret), "backend": BackendError(secret),
        "request": requests.RequestException(secret), "unknown": RuntimeError(secret),
        "hostile_status": requests.HTTPError(secret, response=HostileStatus()),
    }
    class FailingSession(Session):
        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            payload = json.loads(kwargs["data"])
            assert "max_tokens" not in payload and "max_completion_tokens" not in payload
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            index = int(payload["messages"][-1]["content"])
            raise errors[failures[index]]
    def session():
        value = FailingSession(Response({}))
        sessions.append(value)
        return value
    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000
        def close(self):
            assert len(sessions) == 3 and all(item.closed for item in sessions)
            return True
    monkeypatch.setattr(base_client.requests, "Session", session)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    config = _config()
    evaluation = config["evaluation"]
    evaluation["generation"].update(max_tokens=None, chat_template_kwargs={"enable_thinking": False})
    evaluation["max_cases"] = evaluation["vllm"]["max_num_seqs"] = 3
    original = evaluation["scenarios"][0]
    evaluation["scenarios"] = [dict(original, id=f"case_{index}", question=str(index)) for index in range(3)]
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
        validate=lambda: None, environment={}, cwd=tmp_path,
        python_executable=sys.executable, bindings=_bindings())
    assert record["failure_code"] == "evaluation_error" and record["gate_passed"] is False
    assert [case["error_code"] for case in record["cases"]] == list(expected)
    assert [case["id"] for case in record["cases"]] == [f"case_{index}" for index in range(3)]
    assert all(0 <= case["latency_seconds"] <= 3600 for case in record["cases"])
    assert sum(len(item.calls) for item in sessions) == 3
    assert secret not in json.dumps(record)
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


def test_closed_request_record_codes_match_enum_and_reject_hostile_values():
    from Evaluator.protocols import closed_request_failure_code
    assert post_training_eval._REQUEST_FAILURE_CODES == {item.value for item in RequestFailureCode}
    error = BackendError("private", request_failure_code="private-code")
    assert closed_request_failure_code(error) is RequestFailureCode.BACKEND
    class HostileError(BackendError):
        @property
        def request_failure_code(self):
            raise RuntimeError("private-code")
        @request_failure_code.setter
        def request_failure_code(self, value):
            raise RuntimeError("private-setter-code")
    assert closed_request_failure_code(HostileError("private")) is RequestFailureCode.BACKEND


@pytest.mark.parametrize("elapsed", [float("nan"), -1.0, 3601.0, "clock_failure"])
def test_failed_request_elapsed_is_optional_and_bounded(tmp_path, monkeypatch, elapsed):
    from Evaluator import runner
    from Evaluator.config_loader import ConfigLoader
    case = ConfigLoader(tmp_path)._test_to_case(_config()["evaluation"]["scenarios"][0], {})
    class Client:
        def chat(self, messages):
            raise BackendError("legacy-private-text", request_failure_code=RequestFailureCode.TIMEOUT)
    values = iter([0.0, elapsed])
    def clock():
        value = next(values)
        if value == "clock_failure":
            raise RuntimeError("private-clock-text")
        return value
    monkeypatch.setattr(runner.time, "monotonic", clock)
    result = runner._evaluate_single_case(case, Client(), False)
    assert result.latency_s is None
    assert result.request_failure_code is RequestFailureCode.TIMEOUT
    assert result.error == "legacy-private-text"


@pytest.mark.parametrize("status,expected", [(400, "http_400"), (401, "http_401"),
    (403, "http_403"), (404, "http_404"), (408, "http_408"), (413, "http_413"),
    (422, "http_422"), (429, "http_429"), (500, "http_500"), (502, "http_502"),
    (503, "http_503"), (504, "http_504"), (499, "http_other"),
    (True, "http_other"), ("HF_TOKEN=private", "http_other")])
def test_http_status_diagnostics_are_exact_finite_codes(status, expected):
    import requests
    from types import SimpleNamespace
    from Evaluator.base_client import _typed_request_failure_code
    error = requests.HTTPError("private-message", response=SimpleNamespace(status_code=status))
    assert _typed_request_failure_code(error).value == expected


@pytest.mark.parametrize("template_options", [False, True])
def test_record_validation_import_stays_host_light(template_options: bool):
    config = _config()
    if template_options:
        config["evaluation"]["generation"].update(max_tokens=None, chat_template_kwargs={"enable_thinking": False})
    record = {
        "schema_version": "synaptic-post-training-evaluation/v1",
        "status": "failed", "gate_passed": False,
        "failure_code": "startup_failed", "case_count": 2,
        "passed_count": 0, "pass_rate": 0.0, "min_pass_rate": 1.0,
        "cases": [], "bindings": _bindings(),
    }
    probe = """
import builtins
import json
import sys

payload = json.load(sys.stdin)
original_import = builtins.__import__
blocked = {"Evaluator", "torch", "numpy", "pandas", "transformers", "vllm", "unsloth"}
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ImportError("execution-only dependency imported by host validation")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from tuner.runtime.post_training_eval import MAX_EVALUATION_RECORD_BYTES, validate_evaluation_record
assert MAX_EVALUATION_RECORD_BYTES == 16 * 1024 * 1024
assert validate_evaluation_record(payload["record"], config=payload["config"]) == payload["record"]
assert not blocked.intersection(sys.modules)
"""
    interpreter = os.environ.get("SYNAPTIC_HOST_LIGHT_PYTHON", sys.executable)
    result = subprocess.run(
        [interpreter, "-c", probe],
        input=json.dumps({"config": config, "record": record}),
        text=True, capture_output=True, timeout=30,
        cwd=Path(__file__).resolve().parents[2], check=False,
    )
    assert result.returncode == 0, result.stderr


def test_validator_rejects_unbounded_or_nonasserted_cases():
    assert validate_post_training_config(None) is None
    config = _config()
    config["evaluation"]["scenarios"][0]["correct"] = {}
    with pytest.raises(ValueError):
        validate_post_training_config(config)
    config = _config()
    config["evaluation"]["vllm"]["max_model_len"] = True
    with pytest.raises(ValueError):
        validate_post_training_config(config)
    config = _config()
    config["evaluation"]["scenarios"][0]["environment"] = {"enabled": True}
    with pytest.raises(ValueError):
        validate_post_training_config(config)


def test_same_job_source_projects_offline_local_lora_and_pinned_options(tmp_path: Path, monkeypatch):
    base, adapter, tokenizer = (tmp_path / name for name in ("base", "adapter", "tokenizer"))
    for path in (base, adapter, tokenizer):
        path.mkdir()
    checks = []
    monkeypatch.setattr(vllm_runtime.metadata, "version", lambda name: "0.23.0")
    spec = vllm_runtime.VLLMStartupSpec(
        source=vllm_runtime.VerifiedInJobVLLMSource(base, adapter, tokenizer, lambda: checks.append(1), "0.23.0"),
        served_model_name="new-adapter", python_executable="/usr/bin/python3",
        dtype="bfloat16", max_model_len=1024, max_num_seqs=4,
        max_num_batched_tokens=1024, language_model_only=True,
    )
    projected = vllm_runtime._projection(spec, cwd=tmp_path, environment={"PATH": "/usr/bin"})
    assert checks == [1]
    assert ("--model", str(base)) == projected.argv[projected.argv.index("--model"):projected.argv.index("--model") + 2]
    assert ("--lora-modules", f"new-adapter={adapter}") == projected.argv[-2:]
    assert "--language-model-only" in projected.argv
    assert projected.environment["HF_HUB_OFFLINE"] == "1"
    assert projected.expected_model_names == ("new-adapter", "synaptic-base")


@pytest.mark.parametrize("generation_options", [{}, {"max_tokens": None, "chat_template_kwargs": {"enable_thinking": False}}, {"max_tokens": 8192}])
def test_run_uses_existing_assertion_runner_and_closes_runtime(tmp_path: Path, monkeypatch, generation_options):
    config = _config()
    config["evaluation"]["generation"].update(generation_options)
    config["evaluation"]["vllm"]["max_model_len"] = 16384
    paths = [tmp_path / name for name in ("base", "adapter", "tokenizer")]
    for path in paths:
        path.mkdir()
    events = []

    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000
        cleanup_pending = False

        def close(self):
            events.append("close")
            return True

    class Client:
        def __init__(self, settings, **kwargs):
            assert settings.api_key is None
            assert settings.max_tokens == config["evaluation"]["generation"]["max_tokens"]
            assert settings.chat_template_kwargs == config["evaluation"]["generation"].get("chat_template_kwargs")
            assert kwargs["trust_environment"] is False
            assert kwargs["retries"] == 0

        def chat(self, messages):
            events.append("chat")
            return BackendResponse(message="ready", raw={"choices": [{"message": {"content": "ready"}}]}, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=paths[0], adapter_path=paths[1], tokenizer_path=paths[2],
        validate=lambda: events.append("validate"), environment={"PATH": "/usr/bin"},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert record["status"] == "completed"
    assert record["gate_passed"] is True
    assert record["passed_count"] == 2
    assert events[-2:] == ["close", "validate"]
    assert post_training_eval.validate_evaluation_record(record, config=_config(), bindings=_bindings()) is record
    record["cases"][0]["id"] = "other"
    with pytest.raises(ValueError):
        post_training_eval.validate_evaluation_record(record, config=_config())


def test_parallel_requests_overlap_are_bounded_ordered_and_join_before_cleanup(tmp_path: Path, monkeypatch):
    config = _config()
    config["evaluation"]["scenarios"].append({
        "id": "three", "question": "Say ready a third time", "correct": {
            "assertions": [{"type": "jsonpath_equals", "path": "$.content", "value": "ready"}]
        },
    })
    config["evaluation"]["max_cases"] = 3
    config["evaluation"]["vllm"]["max_num_seqs"] = 2
    barrier = threading.Barrier(2, timeout=5)
    second_done = threading.Event()
    guard = threading.Lock()
    state = {"active": 0, "peak": 0, "started": [], "finished": [], "closed": False}
    starts = []

    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            with guard:
                assert state["active"] == 0
                state["closed"] = True
            return True

    class Client:
        def __init__(self, settings, **kwargs):
            assert settings.model == "new-adapter"
            assert kwargs["retries"] == 0
            assert 0 < kwargs["timeout"] <= config["evaluation"]["timeout_seconds"]

        def chat(self, messages):
            question = messages[-1]["content"]
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["started"].append(question)
            try:
                if question in ("Say ready", "Say ready again"):
                    barrier.wait()
                if question == "Say ready":
                    assert second_done.wait(5)
                elif question == "Say ready again":
                    second_done.set()
                return BackendResponse(message="ready", raw={}, latency_s=0.1)
            finally:
                with guard:
                    state["finished"].append(question)
                    state["active"] -= 1

    def start(*args, **kwargs):
        starts.append(1)
        return Lease()

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", start)
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert starts == [1]
    assert state["peak"] == 2
    assert state["finished"].index("Say ready again") < state["finished"].index("Say ready")
    assert state["closed"] is True
    assert record["status"] == "completed"
    assert [case["id"] for case in record["cases"]] == ["one", "two", "three"]
    assert record["passed_count"] == 3


def test_parallel_backend_error_keeps_ordered_partial_count_and_waits_for_cleanup(tmp_path: Path, monkeypatch):
    config = _config()
    config["evaluation"]["min_pass_rate"] = 0
    finished = threading.Event()
    closed = []

    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            assert finished.is_set()
            closed.append(True)
            return True

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, messages):
            if messages[-1]["content"] == "Say ready again":
                finished.set()
                raise RuntimeError("private backend detail")
            assert finished.wait(5)
            return BackendResponse(message="ready", raw={}, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert closed == [True]
    assert record["status"] == "failed"
    assert record["gate_passed"] is False
    assert record["failure_code"] == "evaluation_error"
    assert [case["id"] for case in record["cases"]] == ["one", "two"]
    assert record["passed_count"] == 1
    assert record["pass_rate"] == 0.5
    assert "private backend detail" not in str(record)
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


def test_first_case_error_retains_later_finished_chapter(tmp_path: Path, monkeypatch):
    config = _config()
    config["evaluation"]["min_pass_rate"] = 0
    later_finished = threading.Event()
    closed = []

    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            assert later_finished.is_set()
            closed.append(True)
            return True

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, messages):
            if messages[-1]["content"] == "Say ready":
                assert later_finished.wait(5)
                raise RuntimeError("private backend detail")
            later_finished.set()
            return BackendResponse(message="ready", raw={}, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert closed == [True]
    assert record["status"] == "failed"
    assert record["gate_passed"] is False
    assert record["failure_code"] == "evaluation_error"
    assert [case["id"] for case in record["cases"]] == ["one", "two"]
    assert record["cases"][0]["error_code"] == "request_unknown"
    assert record["cases"][1]["response"] == "ready"
    assert record["passed_count"] == 1
    assert "private backend detail" not in str(record)
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


def test_queued_case_cannot_start_request_after_global_deadline(tmp_path: Path, monkeypatch):
    config = _config()
    config["evaluation"]["vllm"]["max_num_seqs"] = 1
    calls = []

    class Clock:
        expired = False

        def monotonic(self):
            return 31.0 if self.expired else 0.0

    clock = Clock()

    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            return True

    class Client:
        def __init__(self, settings, **kwargs):
            assert 0 < kwargs["timeout"] <= 30

        def chat(self, messages):
            calls.append(messages[-1]["content"])
            clock.expired = True
            return BackendResponse(message="ready", raw={}, latency_s=0.1)

    monkeypatch.setattr(post_training_eval, "time", clock)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert calls == ["Say ready"]
    assert record["failure_code"] == "deadline"
    assert record["passed_count"] == 0
    assert record["gate_passed"] is False
    assert all(case["response"] is None for case in record["cases"])


@pytest.mark.parametrize("startup_elapsed,read_gaps,expected_timeouts,passed", (
    (20.0, ((150.0,), (50.0,)), (280.0, 130.0), 2),
    (300.0, (), (), 0),
    # Requests permits a body that trickles for longer than its inactivity
    # timeout when every gap is shorter. Our deadline rejects that late body.
    (20.0, ((90.0, 90.0, 90.0, 90.0),), (280.0,), 0),
))
def test_real_client_uses_remaining_deadline_and_rejects_late_bodies(
    tmp_path, monkeypatch, startup_elapsed, read_gaps, expected_timeouts, passed,
):
    import requests
    from types import SimpleNamespace
    from Evaluator import base_client
    from tests.evaluator.test_http_transport_policy import Session, Response
    clock = SimpleNamespace(now=0.0)
    timers = SimpleNamespace(monotonic=lambda: clock.now, perf_counter=lambda: clock.now)
    config = _config()
    config["evaluation"]["timeout_seconds"] = 300
    config["evaluation"]["vllm"]["max_num_seqs"] = 1
    config["evaluation"]["generation"].update(max_tokens=None, chat_template_kwargs={"enable_thinking": False})
    sessions, timeouts, closed = [], [], []

    class TimedResponse(Response):
        def iter_content(self, chunk_size):
            body = json.dumps(self.payload).encode()
            for index, gap in enumerate(self.gaps):
                if gap > self.timeout:
                    raise requests.ReadTimeout("closed fixture")
                clock.now += gap
                low = len(body) * index // len(self.gaps)
                high = len(body) * (index + 1) // len(self.gaps)
                yield body[low:high]

    class TimedSession(Session):
        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            assert self.trust_env is False and kwargs["allow_redirects"] is False
            assert kwargs["stream"] is True
            payload = json.loads(kwargs["data"])
            assert "max_tokens" not in payload and "max_completion_tokens" not in payload
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            timeouts.append(kwargs["timeout"])
            self.response.timeout = kwargs["timeout"]
            return self.response

    def session():
        index = len(sessions)
        assert index < len(read_gaps), "expired queued case performed HTTP I/O"
        response = TimedResponse({"choices": [{"message": {"content": "ready"}}]})
        response.gaps = read_gaps[index]
        value = TimedSession(response)
        sessions.append(value)
        return value

    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000
        def close(self):
            assert all(item.closed and item.response.closed for item in sessions)
            closed.append(True)
            return True

    def start(*args, **kwargs):
        assert kwargs["deadline"] == 300.0
        clock.now += startup_elapsed
        return Lease()

    monkeypatch.setattr(post_training_eval, "time", timers)
    monkeypatch.setattr(base_client, "time", timers)
    monkeypatch.setattr(base_client.requests, "Session", session)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", start)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
        validate=lambda: None, environment={}, cwd=tmp_path,
        python_executable=sys.executable, bindings=_bindings(),
    )
    assert timeouts == list(expected_timeouts)
    assert record["passed_count"] == passed and record["gate_passed"] is (passed == 2)
    assert closed == [True]
    assert [case["id"] for case in record["cases"]] == ["one", "two"]
    if passed:
        assert [case["latency_seconds"] for case in record["cases"]] == [150.0, 50.0]
        assert record["status"] == "completed"
    else:
        assert record["failure_code"] == "deadline"
        assert all(case["status"] == "fail" and case["response"] is None for case in record["cases"])
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


@pytest.mark.parametrize("decode_controls", [{}, {
    "presence_penalty": 1.5, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
}])
def test_real_requests_socket_accepts_elapsed_over_120_with_evaluation_decode_controls(tmp_path, monkeypatch, decode_controls):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace
    from Evaluator import base_client
    clock = SimpleNamespace(now=0.0)
    timers = SimpleNamespace(monotonic=lambda: clock.now, perf_counter=lambda: clock.now)
    config = _config()
    config["evaluation"].update(timeout_seconds=300, max_cases=1)
    config["evaluation"]["generation"].update(decode_controls)
    config["evaluation"]["scenarios"] = config["evaluation"]["scenarios"][:1]
    requests_seen, timeouts, closed = [], [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append(payload)
            # Simulate elapsed generation time while retaining a real local
            # TCP request/response through the unmodified Requests Session.
            clock.now += 150.0
            body = json.dumps({"choices": [{"message": {"content": "ready"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_client = vllm_client.VLLMClient

    def observe_client(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return real_client(*args, **kwargs)

    class Lease:
        served_model_name = "new-adapter"
        host, port = server.server_address
        def close(self):
            assert len(requests_seen) == 1
            closed.append(True)
            return True

    def start(*args, **kwargs):
        assert kwargs["deadline"] == 300.0
        clock.now += 20.0
        return Lease()

    monkeypatch.setattr(post_training_eval, "time", timers)
    monkeypatch.setattr(base_client, "time", timers)
    monkeypatch.setattr(vllm_client, "VLLMClient", observe_client)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", start)
    try:
        record = post_training_eval.execute_post_training_evaluation(
            config, base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
            validate=lambda: None, environment={}, cwd=tmp_path,
            python_executable=sys.executable, bindings=_bindings(),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert timeouts == [280.0] and closed == [True]
    assert record["gate_passed"] is True and record["passed_count"] == 1
    assert record["cases"][0]["response"] == "ready"
    assert record["cases"][0]["latency_seconds"] == 150.0
    assert requests_seen[0]["model"] == "new-adapter"
    for name in ("presence_penalty", "top_k", "min_p", "repetition_penalty"):
        if name in decode_controls:
            assert requests_seen[0][name] == decode_controls[name]
        else:
            assert name not in requests_seen[0]
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


def test_late_parallel_http_results_drain_before_runtime_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from Evaluator import base_client
    from tests.evaluator.test_http_transport_policy import Session, Response
    config = _config()
    barrier = threading.Barrier(2, timeout=5)
    guard = threading.Lock()
    state = {"now": 0.0, "active": 0, "peak": 0, "closed": False}
    sessions = []

    class ParallelSession(Session):
        def request(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            assert kwargs["timeout"] == 30.0
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            barrier.wait()
            state["now"] = 31.0
            return self.response
        def close(self):
            super().close()
            with guard:
                state["active"] -= 1

    def session():
        value = ParallelSession(Response({"choices": [{"message": {"content": "ready"}}]}))
        sessions.append(value)
        return value

    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000
        def close(self):
            assert state["active"] == 0
            assert len(sessions) == 2 and all(item.closed and item.response.closed for item in sessions)
            state["closed"] = True
            return True

    monkeypatch.setattr(post_training_eval, "time", SimpleNamespace(monotonic=lambda: state["now"]))
    monkeypatch.setattr(base_client.requests, "Session", session)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path, tokenizer_path=tmp_path,
        validate=lambda: None, environment={}, cwd=tmp_path,
        python_executable=sys.executable, bindings=_bindings(),
    )
    assert state["peak"] == 2 and state["closed"] is True
    assert record["failure_code"] == "deadline" and record["gate_passed"] is False
    assert record["passed_count"] == 0
    assert all(case["status"] == "fail" and case["response"] is None for case in record["cases"])


def test_cleanup_uncertainty_cannot_return_success(tmp_path: Path, monkeypatch):
    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            return False

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", lambda *args, **kwargs: object())
    with pytest.raises(post_training_eval.PostTrainingCleanupUnresolved) as caught:
        post_training_eval.execute_post_training_evaluation(
            _config(), base_model_path=tmp_path, adapter_path=tmp_path,
            tokenizer_path=tmp_path, validate=lambda: None, environment={},
            cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
        )
    assert caught.value.cleanup_lease is not None


@pytest.mark.parametrize("message,expected_code", [
    (RuntimeError("private backend detail"), "evaluation_error"),
    ("x" * (64 * 1024 + 1), "oversized_response"),
], ids=["backend-error", "oversized-response"])
def test_failed_generation_has_closed_code_and_cannot_pass_zero_gate(
    tmp_path: Path, monkeypatch, message, expected_code
):
    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            return True

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, messages):
            if isinstance(message, Exception):
                raise message
            return BackendResponse(message=message, raw={}, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    config = _config()
    config["evaluation"]["min_pass_rate"] = 0
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert record["status"] == "failed"
    assert record["gate_passed"] is False
    assert record["failure_code"] == expected_code
    assert "private backend detail" not in str(record)
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())


@pytest.mark.parametrize("interruption", ["identity", "deadline"])
def test_partial_pass_counts_survive_later_failure(tmp_path: Path, monkeypatch, interruption):
    class Lease:
        served_model_name = "new-adapter"
        host = "127.0.0.1"
        port = 8000

        def close(self):
            return True

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, messages):
            return BackendResponse(message="ready", raw={}, latency_s=0.1)

    checks = []

    def validate():
        checks.append(1)
        if interruption == "identity" and len(checks) >= 2:
            raise ValueError("changed")
        if interruption == "deadline" and len(checks) >= 2:
            clock.now = 31.0

    if interruption == "deadline":
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

        clock = Clock()
        monkeypatch.setattr(post_training_eval, "time", clock)
    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    config = _config()
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=validate, environment={},
        cwd=tmp_path, python_executable="/usr/bin/python3", bindings=_bindings(),
    )
    assert record["status"] == "failed"
    assert record["failure_code"] == ("identity_changed" if interruption == "identity" else "deadline")
    expected_count = 1 if interruption == "identity" else 2
    assert len(record["cases"]) == expected_count
    assert record["passed_count"] == expected_count
    assert record["pass_rate"] == expected_count / 2
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())
