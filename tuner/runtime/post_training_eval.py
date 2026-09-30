"""One bounded local evaluation after packaged SFT saves its adapter."""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Callable, Mapping

from tuner.training.post_training import validate_post_training_config


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_KEYS = {"workload_digest", "model_snapshot_digest", "adapter_digest"}
_RESPONSE_BYTES = 64 * 1024
MAX_EVALUATION_RECORD_BYTES = 16 * 1024 * 1024
_FAILURE_CODES = {"deadline", "incomplete", "startup_failed", "runtime_failed", "identity_changed", "gate_failed", "oversized_response", "evaluation_error"}


class PostTrainingCleanupUnresolved(RuntimeError):
    """The vLLM process family may still own GPU resources."""


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
    if len(encoded) > _RESPONSE_BYTES:
        raise ValueError("evaluation response exceeds its bound")
    return value


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
    failure_code = None
    try:
        from Evaluator.config import VLLMSettings
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
        runtime = start_vllm_runtime(
            startup, cwd=cwd, environment=environment, deadline=deadline,
        )
        for case in cases:
            remaining = deadline - time.monotonic()
            if not math.isfinite(remaining) or remaining <= 0:
                failure_code = "deadline"
                break
            settings = VLLMSettings(
                model=runtime.served_model_name,
                scheme="http", host=runtime.host, port=runtime.port, api_key=None,
                temperature=generation["temperature"], top_p=generation["top_p"],
                max_tokens=generation["max_tokens"], model_path=None,
                lora_adapter=None,
                chat_template_kwargs=generation.get("chat_template_kwargs"),
            )
            client = VLLMClient(
                settings, timeout=min(remaining, 120.0), retries=0,
                trust_environment=False, allow_redirects=False,
                max_request_bytes=1 << 20, max_response_bytes=1 << 20,
            )
            result = evaluate_cases([case], client)[0]
            status = result.status
            if result.error is not None:
                failure_code = "evaluation_error"
            matched_path = (
                result.correctness.matched_path
                if result.correctness is not None else None
            )
            if type(matched_path) is not str or len(matched_path) > 256:
                matched_path = None
            try:
                response = _response(result.response_text)
            except ValueError:
                failure_code = "oversized_response"
                break
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
                "error_code": "evaluation_error" if result.error is not None else None,
            })
            validate()
            if failure_code is not None:
                break
            if time.monotonic() >= deadline:
                failure_code = "deadline"
                break
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
        raise
    except Exception as error:
        if getattr(error, "cleanup_lease", None) is not None:
            raise
        record["failure_code"] = "runtime_failed" if runtime is not None else "startup_failed"
    finally:
        if runtime is not None:
            try:
                stopped = runtime.close()
            except BaseException:
                stopped = False
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
    if type(record) is not dict or set(record) != fields:
        raise ValueError("evaluation record fields are invalid")
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
        if type(case) is not dict or set(case) != {
            "id", "status", "response", "latency_seconds", "matched_path", "error_code",
        } or case["id"] != expected_ids[index] or case["status"] not in ("pass", "fail"):
            raise ValueError("evaluation case outcome is invalid")
        if case["response"] is not None and (
            type(case["response"]) is not str
            or len(case["response"].encode("utf-8")) > _RESPONSE_BYTES
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
        if case["error_code"] not in (None, "evaluation_error"):
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
