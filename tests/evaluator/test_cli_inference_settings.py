from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Evaluator import cli
from Evaluator.cli_utils import resolve_inference_settings, resolved_inference_metadata
from Evaluator.client_factory import create_settings
from Evaluator.config_loader import ConfigLoader
from Evaluator.prompt_sets import PromptCase
from Evaluator.protocols import BackendResponse
from Evaluator.reporting import record_to_dict, write_private_trace
from Evaluator.runner import EvaluationRecord
from Evaluator.vllm_client import VLLMClient


def _config(tmp_path, document):
    root = tmp_path / "config"
    root.mkdir(exist_ok=True)
    (root / "eval_run.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return root


def _args(*extra):
    return cli.parse_args(["--backend", "vllm", "--model", "served", "--scenario", "cases.yaml", *extra])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_yaml_controls_reach_existing_vllm_payload(tmp_path, nested, thinking):
    inference = {
        "max_tokens": None, "temperature": 0.6, "top_p": 0.95, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "presence_penalty": 1.5, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.1,
    }
    document = {"model": {"inference": inference}}
    if nested:
        document = {"run": document}
    loaded = ConfigLoader(_config(tmp_path, document)).load_eval_run()
    options = resolve_inference_settings(_args(), loaded)
    assert options == inference
    settings = create_settings("vllm", "served", **options)
    # Large input is transported intact; no prompt or output text truncation.
    messages = [{"role": "user", "content": "context " * 55000}]
    payload = VLLMClient(settings)._build_payload(messages)
    assert payload["messages"] == messages
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload
    for name, value in inference.items():
        if name != "max_tokens":
            assert payload[name] == value
    assert resolved_inference_metadata(settings) == inference


def test_checked_in_config_explicit_values_are_now_honored_with_cli_precedence():
    root = Path(__file__).resolve().parents[2] / "Evaluator" / "config"
    loaded = ConfigLoader(root).load_eval_run()
    # Intentional config-first fix: these values were previously ignored.
    options = resolve_inference_settings(_args(), loaded)
    assert options == {"temperature": 0.7, "top_p": 0.9, "max_tokens": 2048, "seed": 42}
    overridden = resolve_inference_settings(_args("--max-tokens", "1024", "--temperature", "0", "--seed", "0"), loaded)
    assert overridden == {"temperature": 0.0, "top_p": 0.9, "max_tokens": 1024, "seed": 0}


def test_absent_yaml_and_cli_preserve_legacy_request_for_missing_fields(tmp_path):
    loaded = ConfigLoader(_config(tmp_path, {"run": {}})).load_eval_run()
    assert loaded.inference_settings == {}
    options = resolve_inference_settings(_args(), loaded)
    assert options == {"temperature": None, "top_p": 0.9, "max_tokens": 1024, "seed": None}
    actual = VLLMClient(create_settings("vllm", "served", **options))._build_payload([])
    baseline = VLLMClient(create_settings("vllm", "served"))._build_payload([])
    assert actual == baseline
    assert not {"chat_template_kwargs", "presence_penalty", "top_k", "min_p", "repetition_penalty"}.intersection(actual)


def test_explicit_cli_overrides_yaml_and_preserves_zero_values(tmp_path):
    loaded = ConfigLoader(_config(tmp_path, {"model": {"inference": {
        "max_tokens": None, "temperature": 0.7, "top_p": 0.95, "seed": 42,
    }}})).load_eval_run()
    assert resolve_inference_settings(_args("--max-tokens", "8192", "--temperature", "0", "--top-p", "0", "--seed", "0"), loaded) == {
        "max_tokens": 8192, "temperature": 0.0, "top_p": 0.0, "seed": 0,
    }


def test_run_model_and_preset_override_top_level_without_dropping_other_keys(tmp_path):
    loaded = ConfigLoader(_config(tmp_path, {
        "model": {"inference": {"max_tokens": 12, "top_p": 0.95}},
        "run": {"model": {"inference": {"max_tokens": 25}}},
        "presets": {"unbounded": {"model": {"inference": {"max_tokens": None}}}},
    })).load_eval_run("unbounded")
    assert resolve_inference_settings(_args(), loaded)["max_tokens"] is None
    assert resolve_inference_settings(_args(), loaded)["top_p"] == 0.95


@pytest.mark.parametrize("bad", [[], "invalid", None])
def test_invalid_inference_mapping_rejects(tmp_path, bad):
    with pytest.raises(ValueError, match="mapping"):
        ConfigLoader(_config(tmp_path, {"model": {"inference": bad}})).load_eval_run()


@pytest.mark.parametrize("option,value", [
    ("top_k", True), ("min_p", float("nan")), ("presence_penalty", 3),
    ("repetition_penalty", 0), ("max_tokens", True), ("chat_template_kwargs", {"messages": []}),
])
def test_factory_keeps_existing_vllm_validation(option, value):
    with pytest.raises(ValueError):
        create_settings("vllm", "served", **{option: value})


def test_vllm_only_controls_cannot_silently_disappear_on_other_backends():
    with pytest.raises(ValueError, match="vllm"):
        create_settings("lmstudio", "served", chat_template_kwargs={"enable_thinking": False})
    assert create_settings("lmstudio", "served").max_tokens == 1024


def _record():
    raw = {"choices": [{"message": {"content": "prose", "reasoning_content": "private thinking"},
                        "finish_reason": "stop"}], "usage": {"completion_tokens": 70000}}
    return EvaluationRecord(PromptCase(case_id="case", question="prompt"), "prose", None, 0.1, raw_response=raw)


def test_private_trace_retains_reasoning_finish_usage_without_public_leak(tmp_path):
    record = _record()
    path = tmp_path / "private" / "trace.json"
    write_private_trace(path, [record])
    trace = json.loads(path.read_text(encoding="utf-8"))
    assert trace["records"][0]["raw_response"] == record.raw_response
    assert "private thinking" not in json.dumps(record_to_dict(record))
    assert record_to_dict(record)["response_text"] == "prose"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_private_trace(path, [])
    assert path.read_bytes() == before


def test_cli_connected_config_payload_lineage_and_private_trace(tmp_path, monkeypatch):
    root = _config(tmp_path, {"run": {"model": {"inference": {
        "max_tokens": None, "temperature": 0.6, "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True}, "top_k": 20,
    }}}})
    (root / "scenarios").mkdir()
    (root / "scenarios" / "cases.yaml").write_text(yaml.safe_dump({"tests": [{
        "id": "case", "question": "context " * 55000,
        "correct": {"any": [{"assertions": [{"type": "jsonpath_equals", "path": "$.content", "value": "prose"}]}]},
    }]}), encoding="utf-8")
    payloads = []

    def client(backend, settings, **kwargs):
        transport = VLLMClient(settings)
        def chat(messages):
            payloads.append(transport._build_payload(messages))
            return BackendResponse(message="prose", raw=_record().raw_response, latency_s=0.1)
        return SimpleNamespace(chat=chat)

    monkeypatch.setattr(cli, "create_client", client)
    monkeypatch.setattr("shared.experiment_tracking.registry.RunRegistry", lambda: SimpleNamespace(register_run=lambda value: None))
    output, trace, lineage = (tmp_path / name for name in ("public.json", "private.json", "lineage.json"))
    result = cli.main(["--backend", "vllm", "--model", "served", "--config-dir", str(root),
                       "--scenario", "cases.yaml", "--no-dashboard", "--output", str(output),
                       "--private-trace-json", str(trace), "--lineage", str(lineage)])
    assert result == 0
    assert len(payloads) == 1
    assert "max_tokens" not in payloads[0]
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert payloads[0]["messages"][-1]["content"] == "context " * 55000
    public = json.loads(output.read_text(encoding="utf-8"))
    assert public["metadata"]["max_tokens"] is None
    assert public["metadata"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "private thinking" not in output.read_text(encoding="utf-8")
    assert json.loads(trace.read_text(encoding="utf-8"))["records"][0]["raw_response"] == _record().raw_response
    lineage_text = lineage.read_text(encoding="utf-8")
    assert '"max_tokens": null' in lineage_text
    assert '"enable_thinking": true' in lineage_text
    assert '"top_k": 20' in lineage_text


def test_cli_rejects_private_public_path_collision_before_calls(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    calls = []
    monkeypatch.setattr(cli, "create_client", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="separate"):
        cli.main(["--backend", "vllm", "--model", "served", "--scenario", "cases.yaml",
                  "--output", str(output), "--private-trace-json", str(output)])
    assert calls == []
