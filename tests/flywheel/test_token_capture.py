"""Tests for token-faithful rollout capture (proxy -> logger -> JSONL -> stager).

Covers the optional completion token-id / logprob capture added for a future
GRPO feed: extraction, record serialization back-compat, the capture config
gate, proxy logprob injection, and stager pass-through.
"""
from __future__ import annotations

import json

from unittest.mock import AsyncMock

import pytest

from shared.flywheel.catalog import InferenceLogRecord
from shared.flywheel.config import FlywheelConfig
from shared.flywheel.inference_logger import (
    InferenceLogger,
    _extract_completion_token_logprobs,
)
from shared.flywheel.stager import DatasetStager
from services.proxy.app import _inject_logprobs


# ---------------------------------------------------------------------------
# _extract_completion_token_logprobs
# ---------------------------------------------------------------------------

def _choice_with_logprobs(entries):
    return {"message": {"content": "x"}, "finish_reason": "stop", "logprobs": {"content": entries}}


def test_extract_token_ids_and_logprobs():
    choice = _choice_with_logprobs(
        [{"token": "token_id:101", "logprob": -0.5}, {"token": "token_id:202", "logprob": -1.5}]
    )
    ids, logps = _extract_completion_token_logprobs(choice)
    assert ids == [101, 202]
    assert logps == [-0.5, -1.5]


def test_extract_logprobs_only_when_tokens_not_ids():
    # Plain token strings (no return_tokens_as_token_ids) -> ids None, logprobs kept
    choice = _choice_with_logprobs([{"token": "hello", "logprob": -0.2}, {"token": " world", "logprob": -0.3}])
    ids, logps = _extract_completion_token_logprobs(choice)
    assert ids is None
    assert logps == [-0.2, -0.3]


def test_extract_none_when_no_logprobs():
    assert _extract_completion_token_logprobs({"message": {"content": "x"}}) == (None, None)
    assert _extract_completion_token_logprobs(_choice_with_logprobs([])) == (None, None)


def test_extract_handles_missing_logprob_value():
    choice = _choice_with_logprobs([{"token": "token_id:5"}])  # no "logprob" key
    ids, logps = _extract_completion_token_logprobs(choice)
    assert ids == [5]
    assert logps == [0.0]


# ---------------------------------------------------------------------------
# Record serialization / back-compat
# ---------------------------------------------------------------------------

def test_to_json_omits_unset_token_fields():
    rec = InferenceLogRecord(log_id="a", timestamp="t", model_id="m")
    data = json.loads(rec.to_json())
    assert "completion_token_ids" not in data
    assert "completion_logprobs" not in data
    assert "prompt_token_ids" not in data


def test_to_json_includes_token_fields_when_set():
    rec = InferenceLogRecord(
        log_id="a", timestamp="t", model_id="m",
        completion_token_ids=[1, 2], completion_logprobs=[-0.1, -0.2],
    )
    data = json.loads(rec.to_json())
    assert data["completion_token_ids"] == [1, 2]
    assert data["completion_logprobs"] == [-0.1, -0.2]


def test_from_dict_roundtrip_with_token_fields():
    rec = InferenceLogRecord(
        log_id="a", timestamp="t", model_id="m",
        completion_token_ids=[7], completion_logprobs=[-0.9],
    )
    restored = InferenceLogRecord.from_dict(json.loads(rec.to_json()))
    assert restored.completion_token_ids == [7]
    assert restored.completion_logprobs == [-0.9]


def test_from_dict_old_record_without_token_fields():
    # A pre-capture JSONL row deserializes with token fields defaulting to None
    old = {"log_id": "a", "timestamp": "t", "model_id": "m", "response_content": "hi"}
    rec = InferenceLogRecord.from_dict(old)
    assert rec.completion_token_ids is None
    assert rec.completion_logprobs is None


# ---------------------------------------------------------------------------
# Capture config gate in _build_record
# ---------------------------------------------------------------------------

def _response_with_logprobs():
    return {
        "choices": [_choice_with_logprobs(
            [{"token": "token_id:11", "logprob": -0.4}, {"token": "token_id:12", "logprob": -0.6}]
        )],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def _request():
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5, "max_tokens": 64}


def test_build_record_captures_when_enabled(tmp_path):
    logger = InferenceLogger(tmp_path, AsyncMock(), FlywheelConfig(capture_token_ids=True))
    rec = logger._build_record(_request(), _response_with_logprobs(), 10.0, "m", None)
    assert rec.completion_token_ids == [11, 12]
    assert rec.completion_logprobs == [-0.4, -0.6]


def test_build_record_skips_capture_when_disabled(tmp_path):
    logger = InferenceLogger(tmp_path, AsyncMock(), FlywheelConfig(capture_token_ids=False))
    rec = logger._build_record(_request(), _response_with_logprobs(), 10.0, "m", None)
    assert rec.completion_token_ids is None
    assert rec.completion_logprobs is None


# ---------------------------------------------------------------------------
# Proxy logprob injection
# ---------------------------------------------------------------------------

def test_inject_logprobs_adds_flags():
    body = json.dumps({"model": "m", "messages": []}).encode()
    out = json.loads(_inject_logprobs(body))
    assert out["logprobs"] is True
    assert out["return_tokens_as_token_ids"] is True


def test_inject_logprobs_respects_existing_values():
    body = json.dumps({"model": "m", "logprobs": False}).encode()
    out = json.loads(_inject_logprobs(body))
    assert out["logprobs"] is False  # not overridden


def test_inject_logprobs_passes_through_bad_json():
    body = b"not json"
    assert _inject_logprobs(body) == body


# ---------------------------------------------------------------------------
# Stager pass-through
# ---------------------------------------------------------------------------

def _stager():
    return DatasetStager(catalog=AsyncMock(), config=FlywheelConfig())


def test_grpo_example_includes_token_fields_when_present():
    stager = _stager()
    record = InferenceLogRecord(log_id="a", timestamp="t", model_id="m", fitness_score=0.7)
    content = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_content": "ok",
        "completion_token_ids": [3, 4],
        "completion_logprobs": [-0.1, -0.2],
    }
    example = stager._format_grpo_example(record, content)
    assert example["completion_token_ids"] == [3, 4]
    assert example["completion_logprobs"] == [-0.1, -0.2]
    assert "reward" in example and "conversations" in example


def test_grpo_example_omits_token_fields_when_absent():
    stager = _stager()
    record = InferenceLogRecord(log_id="a", timestamp="t", model_id="m", fitness_score=0.5)
    content = {"messages": [{"role": "user", "content": "hi"}], "response_content": "ok"}
    example = stager._format_grpo_example(record, content)
    assert "completion_token_ids" not in example
    assert "completion_logprobs" not in example
