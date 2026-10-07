from __future__ import annotations

import json
import io
import io
import threading

import pytest

from tuner.inference import vllm_runtime as runtime


class _Response:
    status = 200
    body = {"object": "list", "data": [{"id": "served"}]}
    length = None

    def getheader(self, _name):
        return self.length

    def read(self, _amount):
        if isinstance(self.body, bytes):
            return self.body
        return json.dumps(self.body).encode()


class _Connection:
    response = _Response()
    failure = None

    def __init__(self, *_args, **_kwargs):
        pass

    def request(self, *_args, **_kwargs):
        if self.failure is not None:
            raise self.failure

    def getresponse(self):
        return self.response

    def close(self):
        pass


@pytest.mark.parametrize(
    ("status", "body", "length", "failure", "expected"),
    [
        (200, {"object": "list", "data": [{"id": "served"}]}, None, None, "ready"),
        (503, b"private server body", None, None, "http_non_200"),
        (200, {"object": "list", "data": [{"id": "other"}]}, None, None, "names_mismatch"),
        (200, b"private malformed body", None, None, "invalid_response"),
        (200, {"object": "list", "data": [{"id": "served"}]}, str((1 << 20) + 1), None, "invalid_response"),
        (200, None, None, ConnectionRefusedError("private host"), "no_connect"),
        (200, None, None, TimeoutError("private timeout"), "unknown"),
    ],
)
def test_probe_reports_only_closed_categories(monkeypatch, status, body, length, failure, expected):
    response = _Response()
    response.status, response.body, response.length = status, body, length
    _Connection.response, _Connection.failure = response, failure
    monkeypatch.setattr(runtime.http.client, "HTTPConnection", _Connection)
    assert runtime._probe_ready("127.0.0.1", 8000, ("served",), 1) == expected
    assert runtime._ready("127.0.0.1", 8000, ("served",), 1) is (expected == "ready")
    assert runtime._readiness_state.last_probe == expected


def test_readiness_state_is_thread_local_and_overwritten_before_each_probe(monkeypatch):
    _Connection.response, _Connection.failure = _Response(), None
    monkeypatch.setattr(runtime.http.client, "HTTPConnection", _Connection)
    runtime._readiness_state.last_probe = "names_mismatch"
    seen = []

    def other_thread():
        seen.append(getattr(runtime._readiness_state, "last_probe", "unknown"))
        runtime._ready("127.0.0.1", 8000, ("served",), 1)
        seen.append(runtime._readiness_state.last_probe)

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert seen == ["unknown", "ready"]
    assert runtime._readiness_state.last_probe == "names_mismatch"


class _Process:
    pid = 10
    _pgid = 10
    _start_time = 1
    cleanup_pending = False

    def close(self, **_kwargs):
        return True


def _spec():
    return runtime.VLLMStartupSpec(
        runtime.ExplicitNetworkVLLMSource("org/model", "a" * 40),
        "served", startup_timeout_s=1, readiness_request_timeout_s=0.1,
        python_executable="/usr/bin/python3",
    )


@pytest.mark.parametrize(("alive", "failure"), [(True, "deadline"), (False, "unknown")])
def test_startup_error_carries_closed_evidence_without_stale_probe(tmp_path, monkeypatch, alive, failure):
    current = [0.0]
    monkeypatch.setattr(runtime, "_monotonic", lambda: current[0])
    monkeypatch.setattr(runtime, "_port_available", lambda *_args: True)
    monkeypatch.setattr(runtime, "_spawn", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(runtime, "_leader_alive", lambda _process: alive)
    monkeypatch.setattr(runtime, "_sleep", lambda _seconds: current.__setitem__(0, 1.0))
    monkeypatch.setattr(runtime, "_ready", lambda *_args: False)
    runtime._readiness_state.last_probe = "names_mismatch"
    with pytest.raises(runtime.VLLMRuntimeError) as caught:
        runtime.start_vllm_runtime(_spec(), cwd=tmp_path, environment={})
    diagnostic = caught.value.startup_diagnostic
    assert type(diagnostic) is runtime.VLLMStartupDiagnostic
    assert diagnostic.failure == failure
    assert diagnostic.last_probe == "unknown"
    assert diagnostic.leader_alive is (True if alive else None)
    assert diagnostic.probe_count == (1 if alive else 0)
    assert diagnostic.elapsed_seconds == (1.0 if alive else 0.0)
    assert "private" not in repr(diagnostic)


def test_probe_count_saturation_loses_specific_last_probe(tmp_path, monkeypatch):
    current = [0.0]
    monkeypatch.setattr(runtime, "_MAX_DIAGNOSTIC_PROBES", 1)
    monkeypatch.setattr(runtime, "_monotonic", lambda: current[0])
    monkeypatch.setattr(runtime, "_port_available", lambda *_args: True)
    monkeypatch.setattr(runtime, "_spawn", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(runtime, "_leader_alive", lambda _process: True)
    monkeypatch.setattr(runtime, "_sleep", lambda _seconds: current.__setitem__(0, current[0] + 0.5))

    def ready(*_args):
        runtime._readiness_state.last_probe = "names_mismatch"
        return False

    monkeypatch.setattr(runtime, "_ready", ready)
    with pytest.raises(runtime.VLLMRuntimeError) as caught:
        runtime.start_vllm_runtime(_spec(), cwd=tmp_path, environment={})
    assert caught.value.startup_diagnostic.probe_count is None
    assert caught.value.startup_diagnostic.last_probe == "unknown"


def test_elapsed_is_finite_and_bounded():
    assert runtime._bounded_elapsed(float("nan"), 0) is None
    assert runtime._bounded_elapsed(100000.0, 0) is None
    assert runtime._bounded_elapsed(-2.0, 0) is None
    assert runtime._bounded_elapsed(1.25, 0) == 1.25


def test_optional_startup_log_sink_routes_both_child_streams(tmp_path, monkeypatch):
    captured = {}
    sink = io.BytesIO()
    monkeypatch.setattr(runtime, "_port_available", lambda *_args: True)
    monkeypatch.setattr(runtime, "_spawn", lambda *_args, **kwargs: captured.update(kwargs) or _Process())
    monkeypatch.setattr(runtime, "_leader_alive", lambda _process: True)
    monkeypatch.setattr(runtime, "_ready", lambda *_args: True)
    lease = runtime.start_vllm_runtime(_spec(), cwd=tmp_path, environment={}, startup_log=sink)
    assert captured["stdout"] is sink
    assert captured["stderr"] is sink
    assert lease.close()


def test_optional_startup_log_sink_routes_both_child_streams(tmp_path, monkeypatch):
    captured = {}
    sink = io.BytesIO()
    monkeypatch.setattr(runtime, "_port_available", lambda *_args: True)
    monkeypatch.setattr(runtime, "_spawn", lambda *_args, **kwargs: captured.update(kwargs) or _Process())
    monkeypatch.setattr(runtime, "_leader_alive", lambda _process: True)
    monkeypatch.setattr(runtime, "_ready", lambda *_args: True)
    lease = runtime.start_vllm_runtime(_spec(), cwd=tmp_path, environment={}, startup_log=sink)
    assert captured["stdout"] is sink
    assert captured["stderr"] is sink
    assert lease.close()
