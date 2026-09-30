from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from Evaluator.protocols import BackendResponse
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


def test_record_validation_import_stays_host_light():
    config = _config()
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


def test_run_uses_existing_assertion_runner_and_closes_runtime(tmp_path: Path, monkeypatch):
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
            assert kwargs["trust_environment"] is False
            assert kwargs["retries"] == 0

        def chat(self, messages):
            events.append("chat")
            return BackendResponse(message="ready", raw={"choices": [{"message": {"content": "ready"}}]}, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        _config(), base_model_path=paths[0], adapter_path=paths[1], tokenizer_path=paths[2],
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
            readings = iter((0.0, 0.0, 31.0))

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
    assert len(record["cases"]) == 1
    assert record["passed_count"] == 1
    assert record["pass_rate"] == 0.5
    post_training_eval.validate_evaluation_record(record, config=config, bindings=_bindings())
