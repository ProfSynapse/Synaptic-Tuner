"""Closed startup observations survive evaluation publication without text leaks."""

from copy import deepcopy
import json
import sys
import types

import pytest

from tests.evaluator.test_same_job_post_training_eval import _bindings, _config
from tuner.inference import vllm_runtime
from tuner.runtime import post_training_eval


def _failed(tmp_path, monkeypatch, error=None):
    if error is None:
        error = vllm_runtime.VLLMRuntimeError("private backend details")
        error.startup_diagnostic = vllm_runtime.VLLMStartupDiagnostic(
            failure="deadline", last_probe="no_connect", leader_alive=True,
            elapsed_seconds=300.0, probe_count=47,
        )

    called = []
    def fail(*args, **kwargs):
        called.append(True)
        raise error

    monkeypatch.setattr(vllm_runtime, "start_vllm_runtime", fail)
    loader = types.ModuleType("Evaluator.config_loader")
    class ConfigLoader:
        def __init__(self, _cwd):
            pass
        def _test_to_case(self, item, _defaults):
            return types.SimpleNamespace(case_id=item["id"])
    loader.ConfigLoader = ConfigLoader
    runner = types.ModuleType("Evaluator.runner")
    runner.evaluate_cases = lambda *_args, **_kwargs: pytest.fail("evaluation ran after startup failure")
    monkeypatch.setitem(sys.modules, "Evaluator.config_loader", loader)
    monkeypatch.setitem(sys.modules, "Evaluator.runner", runner)
    config = _config()
    record = post_training_eval.execute_post_training_evaluation(
        config, base_model_path=tmp_path, adapter_path=tmp_path,
        tokenizer_path=tmp_path, validate=lambda: None, environment={},
        cwd=tmp_path, python_executable=sys.executable, bindings=_bindings(),
    )
    assert called == [True]
    return config, record


def test_closed_startup_diagnostic_retained_and_round_trips(tmp_path, monkeypatch):
    config, record = _failed(tmp_path, monkeypatch)
    assert record["cases"] == []
    assert record["failure_code"] == "startup_failed"
    assert record["startup_diagnostic"] == {
        "failure": "deadline", "last_probe": "no_connect", "leader_alive": True,
        "elapsed_seconds": 300.0, "probe_count": 47,
    }
    raw = post_training_eval.canonical_evaluation_document_bytes(record)
    assert b"private" not in raw
    assert post_training_eval.validate_evaluation_record(
        json.loads(raw), config=config, bindings=_bindings(),
    ) == record


def test_legacy_error_and_arbitrary_diagnostic_are_not_projected(tmp_path, monkeypatch):
    for error in (RuntimeError("private"), vllm_runtime.VLLMRuntimeError("private")):
        error.startup_diagnostic = {"failure": "private"}
        config, record = _failed(tmp_path, monkeypatch, error)
        assert "startup_diagnostic" not in record
        raw = post_training_eval.canonical_evaluation_document_bytes(record)
        post_training_eval.validate_evaluation_record(record, config=config)
        assert post_training_eval.canonical_evaluation_document_bytes(record) == raw
        assert b"private" not in raw


@pytest.mark.parametrize("field,value", [
    ("failure", "private"), ("failure", "leader_exited"), ("last_probe", "private"),
    ("leader_alive", 1), ("elapsed_seconds", True),
    ("elapsed_seconds", -1), ("elapsed_seconds", float("nan")),
    ("elapsed_seconds", float("inf")), ("elapsed_seconds", 86401),
    ("probe_count", True), ("probe_count", -1), ("probe_count", 1000001),
    ("unexpected", "private"),
])
def test_readback_rejects_nonclosed_diagnostics(tmp_path, monkeypatch, field, value):
    config, record = _failed(tmp_path, monkeypatch)
    bad = deepcopy(record)
    bad["startup_diagnostic"][field] = value
    with pytest.raises(ValueError):
        post_training_eval.validate_evaluation_record(bad, config=config)


def test_readback_rejects_missing_field(tmp_path, monkeypatch):
    config, record = _failed(tmp_path, monkeypatch)
    del record["startup_diagnostic"]["probe_count"]
    with pytest.raises(ValueError):
        post_training_eval.validate_evaluation_record(record, config=config)


def test_unknown_startup_measurements_remain_null(tmp_path, monkeypatch):
    config, record = _failed(tmp_path, monkeypatch)
    record["startup_diagnostic"] = {
        "failure": "unknown", "last_probe": "unknown", "leader_alive": None,
        "elapsed_seconds": None, "probe_count": None,
    }
    raw = post_training_eval.canonical_evaluation_document_bytes(record)
    assert post_training_eval.validate_evaluation_record(json.loads(raw), config=config) == record


def test_identity_failure_preserves_startup_observation(tmp_path, monkeypatch):
    config, record = _failed(tmp_path, monkeypatch)
    record["failure_code"] = "identity_changed"
    assert post_training_eval.validate_evaluation_record(record, config=config) is record
