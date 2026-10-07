"""One bounded local evaluation after packaged SFT saves its adapter."""

from __future__ import annotations

import math
import json
from contextlib import contextmanager
from pathlib import Path
import re
import time
import threading
import socket
from typing import Callable, Mapping

from tuner.training.post_training import validate_post_training_config


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_KEYS = {"workload_digest", "model_snapshot_digest", "adapter_digest"}
MAX_EVALUATION_RESPONSE_BYTES = 1 << 20
_REQUEST_FAILURE_CODES = frozenset({
    "request_timeout", "request_connection", "request_transport", "request_validation",
    "request_backend", "request_unknown", "http_other", "http_400", "http_401",
    "http_403", "http_404", "http_408", "http_413", "http_422", "http_429",
    "http_500", "http_502", "http_503", "http_504",
})
MAX_EVALUATION_RECORD_BYTES = 16 * 1024 * 1024
_FAILURE_CODES = {"deadline", "incomplete", "startup_failed", "runtime_failed", "identity_changed", "gate_failed", "oversized_response", "evaluation_error"}
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
_MAX_TOKEN_COUNT = (2**53) - 1


def _validate_startup_diagnostic(value):
    """Validate closed readiness observations without importing the serving runtime."""
    if type(value) is not dict or set(value) != {
        "failure", "last_probe", "leader_alive", "elapsed_seconds", "probe_count",
    }:
        raise ValueError("evaluation startup diagnostic fields are invalid")
    if type(value["failure"]) is not str or value["failure"] not in {
        "deadline", "untimely", "unknown",
    }:
        raise ValueError("evaluation startup failure is invalid")
    if type(value["last_probe"]) is not str or value["last_probe"] not in {
        "ready", "no_connect", "http_non_200", "invalid_response", "names_mismatch", "unknown",
    }:
        raise ValueError("evaluation startup probe is invalid")
    if value["leader_alive"] is not None and type(value["leader_alive"]) is not bool:
        raise ValueError("evaluation startup liveness is invalid")
    elapsed = value["elapsed_seconds"]
    if elapsed is not None and (
        type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= 86400
    ):
        raise ValueError("evaluation startup elapsed time is invalid")
    count = value["probe_count"]
    if count is not None and (type(count) is not int or not 0 <= count <= 1_000_000):
        raise ValueError("evaluation startup probe count is invalid")
    return value


def _startup_diagnostic(error):
    """Project only the exact serving diagnostic type, never exception text."""
    try:
        from tuner.inference.vllm_runtime import VLLMRuntimeError, VLLMStartupDiagnostic
        if type(error) is not VLLMRuntimeError:
            return None
        diagnostic = error.startup_diagnostic
        if type(diagnostic) is not VLLMStartupDiagnostic:
            return None
        value = {key: getattr(diagnostic, key) for key in (
            "failure", "last_probe", "leader_alive", "elapsed_seconds", "probe_count",
        )}
        return _validate_startup_diagnostic(value)
    except Exception:
        return None


class PostTrainingCleanupUnresolved(RuntimeError):
    """The vLLM process family may still own GPU resources."""


_SERVING_METRICS = {
    "vllm:num_requests_running": "running_requests",
    "vllm:num_requests_waiting": "waiting_requests",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
}


def _read_serving_metrics(port, stop, connected=None):
    """Fixed loopback HTTP only; deadlines checked between bounded socket reads.

    No Requests inactivity timeout is treated as a whole-call deadline. Each
    socket operation is bounded separately; a stalled diagnostic is disposable.
    """
    if type(port) is not int or not 1 <= port <= 65535 or stop.is_set():
        raise ValueError("diagnostic endpoint invalid")
    deadline = time.monotonic() + 1.0
    with socket.create_connection(("127.0.0.1", port), timeout=0.25) as connection:
        if connected is not None:
            connected(connection)
        connection.sendall(b"GET /metrics HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        data = bytearray()
        header_end = None
        size = None
        while True:
            remaining = deadline - time.monotonic()
            if stop.is_set() or remaining <= 0:
                raise ValueError("diagnostic read expired")
            connection.settimeout(min(0.25, remaining))
            chunk = connection.recv(1024)
            if not chunk:
                break
            data.extend(chunk)
            if header_end is None:
                marker = data.find(b"\r\n\r\n")
                if marker == -1:
                    if len(data) > 4096:
                        raise ValueError("diagnostic headers oversized")
                    continue
                header_end = marker + 4
                if header_end > 4096:
                    raise ValueError("diagnostic headers oversized")
                headers = bytes(data[:marker]).split(b"\r\n")
                if not re.fullmatch(rb"HTTP/1\.[01] 200(?: [^\r\n]*)?", headers[0]):
                    raise ValueError("diagnostic response unavailable")
                lengths = []
                for header in headers[1:]:
                    name, separator, value = header.partition(b":")
                    if not separator:
                        raise ValueError("diagnostic headers invalid")
                    if name.lower() == b"transfer-encoding":
                        raise ValueError("diagnostic encoding unsupported")
                    if name.lower() == b"content-length":
                        if not re.fullmatch(rb"[0-9]{1,6}", value.strip()):
                            raise ValueError("diagnostic length invalid")
                        lengths.append(int(value))
                if len(lengths) != 1 or lengths[0] > 262144:
                    raise ValueError("diagnostic body bound invalid")
                size = lengths[0]
            if len(data) - header_end > size:
                raise ValueError("diagnostic body oversized")
            if len(data) - header_end == size:
                break
        if header_end is None or len(data) - header_end != size:
            raise ValueError("diagnostic response truncated")
        return _parse_serving_metrics(bytes(data[header_end:]))


def _parse_serving_metrics(raw):
    if type(raw) is not bytes or len(raw) > 262144:
        raise ValueError("diagnostic body invalid")
    lines = raw.splitlines()
    if len(lines) > 4096 or any(len(line) > 4096 for line in lines):
        raise ValueError("diagnostic lines oversized")
    values = {}
    for line in lines:
        name = re.split(rb"[ {\t]", line, maxsplit=1)[0].decode("ascii", errors="ignore")
        if name not in _SERVING_METRICS:
            continue
        match = re.fullmatch(rb"[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^\r\n]*\})?[ \t]+([^ \t]+)", line)
        if match is None or name in values:
            raise ValueError("diagnostic series ambiguous")
        value = float(match[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError("diagnostic value invalid")
        if name == "vllm:kv_cache_usage_perc":
            if value > 1:
                raise ValueError("diagnostic fraction invalid")
        else:
            if value > 2**53 - 1 or not value.is_integer():
                raise ValueError("diagnostic counter invalid")
            value = int(value)
        values[name] = value
    if set(values) != set(_SERVING_METRICS):
        raise ValueError("diagnostic series unavailable")
    return {_SERVING_METRICS[name]: value for name, value in values.items()}


def _serving_diagnostic(callback, kind, values):
    try:
        if callable(callback):
            callback(kind, values)
    except Exception:
        pass


class _ServingSampler:
    def __init__(self, runtime, callback):
        self._runtime, self._callback = runtime, callback
        self._stop = threading.Event()
        self._connection = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        # Never join before GPU cleanup, nor give this observer ownership of it.
        self._stop.set()
        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def finish(self):
        self._thread.join(timeout=0.05)

    def _run(self):
        for _ in range(64):
            if self._stop.is_set():
                return
            try:
                if self._runtime.host != "127.0.0.1":
                    return
                values = _read_serving_metrics(self._runtime.port, self._stop, self._connected)
                if not self._stop.is_set():
                    _serving_diagnostic(self._callback, "METRICS", values)
            except Exception:
                pass
            if self._stop.wait(15):
                return

    def _connected(self, connection):
        with self._lock:
            self._connection = connection
        if self._stop.is_set():
            connection.shutdown(socket.SHUT_RDWR)


def _emit_phase(callback, phase, edge, ordinal=None):
    if phase == "CHAT_REQUEST" and ordinal is None:
        return
    try:
        if callable(callback):
            callback(phase, edge, ordinal)
    except Exception:
        pass


@contextmanager
def _phase_span(callback, phase, ordinal=None):
    _emit_phase(callback, phase, "START", ordinal)
    try:
        yield
    except BaseException:
        _emit_phase(callback, phase, "ERROR", ordinal)
        raise
    _emit_phase(callback, phase, "RETURN", ordinal)


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != _BINDING_KEYS or any(
        type(item) is not str or _DIGEST.fullmatch(item) is None for item in value.values()
    ):
        raise ValueError("post-training identity bindings are invalid")
    return dict(value)


def _response(value: object) -> str | None:
    if type(value) is not str:
        return None
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) > MAX_EVALUATION_RESPONSE_BYTES:
        raise ValueError("evaluation response exceeds its bound")
    return value


def _completion_metadata(raw: object) -> tuple[str | None, dict[str, int | None]]:
    """Retain only closed, bounded diagnostics from an actual API reply."""
    finish_reason = None
    usage = {key: None for key in _USAGE_KEYS}
    if type(raw) is not dict:
        return finish_reason, usage
    choices = raw.get("choices")
    if type(choices) is list and choices and type(choices[0]) is dict:
        candidate = choices[0].get("finish_reason")
        if type(candidate) is str and candidate in _FINISH_REASONS:
            finish_reason = candidate
    reported = raw.get("usage")
    if type(reported) is dict:
        for key in _USAGE_KEYS:
            candidate = reported.get(key)
            if type(candidate) is int and 0 <= candidate <= _MAX_TOKEN_COUNT:
                usage[key] = candidate
    return finish_reason, usage


def canonical_evaluation_document_bytes(value: Mapping[str, object]) -> bytes:
    """Canonical evaluation publication bytes under the aggregate record bound."""
    if not isinstance(value, Mapping):
        raise TypeError("evaluation document root must be a mapping")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("evaluation document is not finite canonical JSON") from exc
    if not encoded or len(encoded) > MAX_EVALUATION_RECORD_BYTES:
        raise ValueError("evaluation document exceeds its aggregate bound")
    return encoded


def execute_post_training_evaluation(
    config: dict,
    *,
    base_model_path: Path,
    adapter_path: Path,
    tokenizer_path: Path,
    validate: Callable[[], None],
    environment: dict[str, str],
    cwd: Path,
    python_executable: str,
    bindings: dict[str, str],
    phase_callback: Callable | None = None,
    serving_callback: Callable | None = None,
) -> dict:
    """Run evaluator assertions on the locally prepared base and new adapter.

    The caller retains model and adapter identity handles for the whole call.
    This function neither retrieves model bytes nor writes an artifact.
    """
    normalized = validate_post_training_config(config)
    if normalized is None:
        raise ValueError("post-training evaluation requires an enabled config")
    identities = _bindings(bindings)
    evaluation = normalized["evaluation"]
    vllm = evaluation["vllm"]
    generation = evaluation["generation"]
    if not callable(validate):
        raise TypeError("retained identity validator is required")
    if type(environment) is not dict:
        raise TypeError("explicit environment is required")
    started = time.monotonic()
    deadline = started + evaluation["timeout_seconds"]
    record = {
        "schema_version": "synaptic-post-training-evaluation/v1",
        "status": "failed",
        "gate_passed": False,
        "failure_code": None,
        "case_count": len(evaluation["scenarios"]),
        "passed_count": 0,
        "pass_rate": 0.0,
        "min_pass_rate": evaluation["min_pass_rate"],
        "cases": [],
        "bindings": identities,
    }
    runtime = None
    sampler = None
    failure_code = None
    original_validate = validate
    def validate():
        with _phase_span(phase_callback, "EVALUATION_IDENTITY_VALIDATE"):
            original_validate()
    request_lock, request_count = threading.Lock(), 0
    preparing = True
    _emit_phase(phase_callback, "EVALUATION_PREPARE", "START")
    try:
        from Evaluator.config import VLLMSettings
        from Evaluator.protocols import RequestFailureCode
        from Evaluator.config_loader import ConfigLoader
        from Evaluator.runner import evaluate_cases
        from Evaluator.vllm_client import VLLMClient
        from tuner.inference.vllm_runtime import (
            VLLMStartupSpec, VerifiedInJobVLLMSource, start_vllm_runtime,
        )
        validate()
        loader = ConfigLoader(cwd)
        cases = [loader._test_to_case(item, {}) for item in evaluation["scenarios"]]
        startup = VLLMStartupSpec(
            source=VerifiedInJobVLLMSource(
                base_model_path, adapter_path, tokenizer_path, validate,
                vllm["expected_version"],
            ),
            served_model_name=evaluation["served_model_name"],
            python_executable=python_executable,
            startup_timeout_s=evaluation["startup_timeout_seconds"],
            tensor_parallel_size=vllm["tensor_parallel_size"],
            dtype=vllm["dtype"],
            max_model_len=vllm["max_model_len"],
            max_num_seqs=vllm["max_num_seqs"],
            max_num_batched_tokens=vllm["max_num_batched_tokens"],
            language_model_only=vllm["language_model_only"],
            max_lora_rank=vllm["max_lora_rank"],
        )
        preparing = False
        _emit_phase(phase_callback, "EVALUATION_PREPARE", "RETURN")
        runtime = start_vllm_runtime(
            startup, cwd=cwd, environment=environment, deadline=deadline,
            **({"phase_callback": phase_callback} if callable(phase_callback) else {}),
        )
        if callable(serving_callback):
            try:
                sampler = _ServingSampler(runtime, serving_callback)
                sampler.start()
            except Exception:
                sampler = None
        settings = VLLMSettings(
            model=runtime.served_model_name,
            scheme="http", host=runtime.host, port=runtime.port, api_key=None,
            temperature=generation["temperature"], top_p=generation["top_p"],
            max_tokens=generation["max_tokens"], model_path=None,
            lora_adapter=None,
            chat_template_kwargs=generation.get("chat_template_kwargs"),
            presence_penalty=generation.get("presence_penalty"),
            top_k=generation.get("top_k"),
            min_p=generation.get("min_p"),
            repetition_penalty=generation.get("repetition_penalty"),
        )

        class DeadlineClient:
            def chat(self, messages):
                nonlocal request_count
                with request_lock:
                    request_count += 1
                    ordinal = request_count if request_count <= 32 else None
                with _phase_span(phase_callback, "CHAT_REQUEST", ordinal):
                    return self._chat(messages)

            def _chat(self, messages):
                remaining = deadline - time.monotonic()
                if not math.isfinite(remaining) or remaining <= 0:
                    raise TimeoutError("evaluation deadline reached")
                client = VLLMClient(
                    settings, timeout=remaining, retries=0,
                    trust_environment=False, allow_redirects=False,
                    max_request_bytes=1 << 20,
                    max_response_bytes=MAX_EVALUATION_RESPONSE_BYTES,
                )
                # Requests uses connect/read inactivity timeouts, not a total
                # wall-clock deadline. Drain and close its transport normally,
                # then reject a late reply before the assertion runner sees it.
                response = client.chat(messages)
                remaining = deadline - time.monotonic()
                if not math.isfinite(remaining) or remaining <= 0:
                    raise TimeoutError("evaluation deadline reached")
                return response

        with _phase_span(phase_callback, "CHAT_BATCH"):
            results = evaluate_cases(
                cases, DeadlineClient(), parallel=True,
                max_workers=min(len(cases), vllm["max_num_seqs"]),
            )
        for case, result in zip(cases, results):
            status = result.status
            if result.error is not None and failure_code is None:
                failure_code = (
                    "deadline" if time.monotonic() >= deadline
                    else "evaluation_error"
                )
            matched_path = (
                result.correctness.matched_path
                if result.correctness is not None else None
            )
            if type(matched_path) is not str or len(matched_path) > 256:
                matched_path = None
            oversized = False
            try:
                response = _response(result.response_text)
            except ValueError:
                oversized = True
                response = None
                status = "fail"
                if failure_code is None:
                    failure_code = "oversized_response"
            error_code = "evaluation_error" if result.error is not None or oversized else None
            if result.error is not None:
                try:
                    typed_code = getattr(result, "request_failure_code", None)
                    if type(typed_code) is RequestFailureCode:
                        error_code = typed_code.value
                except Exception:
                    pass
            finish_reason, token_usage = _completion_metadata(result.raw_response)
            record["cases"].append({
                "id": case.case_id,
                "status": status,
                "response": response,
                "latency_seconds": (
                    round(result.latency_s, 6)
                    if type(result.latency_s) in (int, float)
                    and math.isfinite(result.latency_s) and 0 <= result.latency_s <= 3600
                    else None
                ),
                "matched_path": matched_path,
                "error_code": error_code,
                "finish_reason": finish_reason,
                "token_usage": token_usage,
            })
            validate()
            if failure_code is None and time.monotonic() >= deadline:
                failure_code = "deadline"
        record["passed_count"] = sum(case["status"] == "pass" for case in record["cases"])
        record["pass_rate"] = record["passed_count"] / record["case_count"]
        if failure_code is None and len(record["cases"]) == record["case_count"]:
            record["gate_passed"] = record["pass_rate"] >= record["min_pass_rate"]
            record["status"] = "completed"
            if not record["gate_passed"]:
                record["failure_code"] = "gate_failed"
        else:
            record["failure_code"] = failure_code or "incomplete"
    except (KeyboardInterrupt, SystemExit):
        if preparing:
            _emit_phase(phase_callback, "EVALUATION_PREPARE", "ERROR")
        raise
    except Exception as error:
        if preparing:
            _emit_phase(phase_callback, "EVALUATION_PREPARE", "ERROR")
        if getattr(error, "cleanup_lease", None) is not None:
            raise
        record["failure_code"] = (
            "runtime_failed" if runtime is not None else "startup_failed"
        )
        if runtime is None:
            diagnostic = _startup_diagnostic(error)
            if diagnostic is not None:
                record["startup_diagnostic"] = diagnostic
    finally:
        if sampler is not None:
            sampler.stop()
        if runtime is not None:
            try:
                with _phase_span(phase_callback, "VLLM_CLEANUP"):
                    stopped = runtime.close()
            except BaseException:
                stopped = False
            else:
                if type(stopped) is bool:
                    _serving_diagnostic(serving_callback, "CLEANUP", {"cleanup_resolved": stopped})
            if sampler is not None:
                sampler.finish()
            if not stopped:
                error = PostTrainingCleanupUnresolved("vLLM process cleanup unresolved")
                error.cleanup_lease = runtime
                raise error
    try:
        validate()
    except Exception:
        record["status"] = "failed"
        record["gate_passed"] = False
        record["failure_code"] = "identity_changed"
    # Cases may already have been recorded when a later identity check or
    # deadline fails. Keep counts tied to those retained case outcomes.
    record["passed_count"] = sum(case["status"] == "pass" for case in record["cases"])
    record["pass_rate"] = record["passed_count"] / record["case_count"]
    validate_evaluation_record(record, config=normalized, bindings=identities)
    return record


def validate_evaluation_record(
    record: object, *, config: dict, bindings: dict[str, str] | None = None
) -> dict:
    """Validate the closed signed record shape before presenting a result."""
    normalized = validate_post_training_config(config)
    if normalized is None:
        raise ValueError("post-training config is absent")
    expected = normalized["evaluation"]
    fields = {
        "schema_version", "status", "gate_passed", "failure_code", "case_count",
        "passed_count", "pass_rate", "min_pass_rate", "cases", "bindings",
    }
    optional_fields = {"startup_diagnostic"}
    if type(record) is not dict or not fields <= set(record) or set(record) - fields - optional_fields:
        raise ValueError("evaluation record fields are invalid")
    if "startup_diagnostic" in record:
        _validate_startup_diagnostic(record["startup_diagnostic"])
        if record["status"] != "failed" or record["failure_code"] not in {"startup_failed", "identity_changed"}:
            raise ValueError("evaluation startup diagnostic outcome is invalid")
    if record["schema_version"] != "synaptic-post-training-evaluation/v1":
        raise ValueError("evaluation record schema is invalid")
    if record["status"] not in ("completed", "failed") or type(record["gate_passed"]) is not bool:
        raise ValueError("evaluation record outcome is invalid")
    if record["failure_code"] is not None and record["failure_code"] not in _FAILURE_CODES:
        raise ValueError("evaluation failure code is invalid")
    if type(record["case_count"]) is not int or record["case_count"] != len(expected["scenarios"]):
        raise ValueError("evaluation case count is invalid")
    if type(record["passed_count"]) is not int:
        raise ValueError("evaluation passed count is invalid")
    if record["min_pass_rate"] != expected["min_pass_rate"]:
        raise ValueError("evaluation gate threshold differs")
    cases = record["cases"]
    if type(cases) is not list or len(cases) > record["case_count"]:
        raise ValueError("evaluation case records are invalid")
    expected_ids = [item["id"] for item in expected["scenarios"]]
    for index, case in enumerate(cases):
        required_fields = {
            "id", "status", "response", "latency_seconds", "matched_path", "error_code",
        }
        optional_fields = {"finish_reason", "token_usage"}
        if type(case) is not dict or not required_fields <= set(case) or set(case) - required_fields - optional_fields or case["id"] != expected_ids[index] or case["status"] not in ("pass", "fail"):
            raise ValueError("evaluation case outcome is invalid")
        if "finish_reason" in case and case["finish_reason"] is not None and (
            type(case["finish_reason"]) is not str or case["finish_reason"] not in _FINISH_REASONS
        ):
            raise ValueError("evaluation finish reason is invalid")
        if "token_usage" in case:
            usage = case["token_usage"]
            if type(usage) is not dict or set(usage) != set(_USAGE_KEYS) or any(
                value is not None and (type(value) is not int or not 0 <= value <= _MAX_TOKEN_COUNT)
                for value in usage.values()
            ):
                raise ValueError("evaluation token usage is invalid")
        if case["response"] is not None and (
            type(case["response"]) is not str
            or len(case["response"].encode("utf-8")) > MAX_EVALUATION_RESPONSE_BYTES
        ):
            raise ValueError("evaluation response is invalid")
        latency = case["latency_seconds"]
        if latency is not None and (
            type(latency) not in (int, float) or not math.isfinite(latency)
            or not 0 <= latency <= 3600
        ):
            raise ValueError("evaluation latency is invalid")
        matched = case["matched_path"]
        if matched is not None and (type(matched) is not str or len(matched) > 256):
            raise ValueError("evaluation matched path is invalid")
        code = case["error_code"]
        if code is not None and (type(code) is not str
                or code not in _REQUEST_FAILURE_CODES | {"evaluation_error"}):
            raise ValueError("evaluation error code is invalid")
    passed = sum(item["status"] == "pass" for item in cases)
    if record["passed_count"] != passed or record["pass_rate"] != passed / record["case_count"]:
        raise ValueError("evaluation counts or rate are invalid")
    if record["status"] == "completed":
        gate = record["pass_rate"] >= expected["min_pass_rate"]
        if len(cases) != record["case_count"] or record["gate_passed"] != gate or record["failure_code"] != (None if gate else "gate_failed"):
            raise ValueError("completed evaluation gate is invalid")
    elif record["gate_passed"] or record["failure_code"] in (None, "gate_failed"):
        raise ValueError("failed evaluation gate is invalid")
    if any(case["error_code"] is not None for case in cases) and record["status"] == "completed":
        raise ValueError("evaluation transport error cannot complete the gate")
    actual_bindings = _bindings(record["bindings"])
    if bindings is not None and actual_bindings != _bindings(bindings):
        raise ValueError("evaluation bindings differ")
    return record
