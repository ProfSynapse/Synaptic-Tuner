"""Same-job evaluation retains only bounded completion diagnostics."""

from __future__ import annotations

from copy import deepcopy
import json
import sys

import pytest

from Evaluator.protocols import BackendResponse
from tests.evaluator.test_same_job_post_training_eval import _bindings, _config
from tuner.inference import vllm_runtime
from tuner.runtime import post_training_eval
from Evaluator import vllm_client


def _run(tmp_path, monkeypatch, raw):
    config = _config()
    config["evaluation"]["scenarios"] = config["evaluation"]["scenarios"][:1]
    config["evaluation"]["max_cases"] = 1
    config["evaluation"]["generation"]["max_tokens"] = None

    class Lease:
        served_model_name = "new-adapter"
        host, port = "127.0.0.1", 8000

        def close(self):
            return True

    class Client:
        def __init__(self, settings, **kwargs):
            assert settings.max_tokens is None
            assert kwargs["max_response_bytes"] == post_training_eval.MAX_EVALUATION_RESPONSE_BYTES

        def chat(self, messages):
            return BackendResponse(message="ready", raw=raw, latency_s=0.1)

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", lambda *args, **kwargs: Lease())
    monkeypatch.setattr(vllm_client, "VLLMClient", Client)
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable=sys.executable, bindings=_bindings(),
    )
    return config, record


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_retains_actual_finish_reason_and_token_usage(tmp_path, monkeypatch, reason):
    raw = {"choices": [{"finish_reason": reason, "message": {"content": "ready"}}],
           "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
    config, record = _run(tmp_path, monkeypatch, raw)
    case = record["cases"][0]
    assert case["finish_reason"] == reason
    assert case["token_usage"] == raw["usage"]
    assert record["status"] == "completed" and record["gate_passed"] is True
    encoded = post_training_eval.canonical_evaluation_document_bytes(record)
    assert post_training_eval.validate_evaluation_record(
        json.loads(encoded), config=config, bindings=_bindings(),
    )["cases"][0] == case


def test_missing_and_hostile_metadata_becomes_null_without_raw_leak(tmp_path, monkeypatch):
    raw = {"choices": [{"finish_reason": "secret-response", "message": {"content": "private"}}],
           "usage": {"prompt_tokens": True, "completion_tokens": -1,
                     "total_tokens": 2**100, "secret": "private"},
           "private_payload": "private"}
    config, record = _run(tmp_path, monkeypatch, raw)
    case = record["cases"][0]
    assert case["finish_reason"] is None
    assert case["token_usage"] == dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"))
    assert "private" not in json.dumps(record)
    assert post_training_eval.validate_evaluation_record(record, config=config) is record
    config, missing = _run(tmp_path, monkeypatch, {})
    assert missing["cases"][0]["finish_reason"] is None
    assert all(value is None for value in missing["cases"][0]["token_usage"].values())


@pytest.mark.parametrize("field,value", [
    ("finish_reason", "hostile"), ("finish_reason", "x" * 10000),
    ("token_usage", {"prompt_tokens": 1.0, "completion_tokens": 2, "total_tokens": 3}),
    ("token_usage", {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 3}),
    ("token_usage", {"prompt_tokens": 2**53, "completion_tokens": 2, "total_tokens": 3}),
    ("token_usage", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "private": 1}),
])
def test_readback_rejects_invalid_optional_metadata(tmp_path, monkeypatch, field, value):
    config, record = _run(tmp_path, monkeypatch, {})
    bad = deepcopy(record)
    bad["cases"][0][field] = value
    with pytest.raises(ValueError):
        post_training_eval.validate_evaluation_record(bad, config=config)


def test_legacy_record_bytes_remain_unchanged(tmp_path, monkeypatch):
    config, record = _run(tmp_path, monkeypatch, {})
    legacy = deepcopy(record)
    del legacy["cases"][0]["finish_reason"]
    del legacy["cases"][0]["token_usage"]
    encoded = post_training_eval.canonical_evaluation_document_bytes(legacy)
    assert post_training_eval.validate_evaluation_record(
        json.loads(encoded), config=config, bindings=_bindings(),
    ) == legacy
    assert post_training_eval.canonical_evaluation_document_bytes(legacy) == encoded
