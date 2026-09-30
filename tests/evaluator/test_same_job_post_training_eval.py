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
    assert record["passed_count"] == 1
    assert record["gate_passed"] is False


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

    if interruption == "deadline":
        class Clock:
            readings = iter((0.0, 0.0, 0.0, 31.0))

            def monotonic(self):
                return next(self.readings)

        monkeypatch.setattr(post_training_eval, "time", Clock())
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
