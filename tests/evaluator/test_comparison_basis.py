"""Captured declarations and explicit nonsecret controls reach actual evaluation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from Evaluator.comparison_basis import GENERATION_KEYS, backend_controls, build_basis
from Evaluator.prompt_sets import PromptCase
from Evaluator.runner import evaluate_cases
from Evaluator.schema_validator import validate_assistant_response
from shared.validation import dataset_validator
from shared.validation.dataset_validator import ValidatorConfig


@pytest.mark.parametrize("response", [
    {"tool_calls": [{"type": "function", "function": {"name": "echo", "arguments": '{"value":"ok"}'}}]},
    'tool_call: echo\narguments: {"value":"ok"}',
    '[TOOL_CALLS] [{"name":"echo","arguments":{"value":"ok"}}]',
    '<tool_call>{"name":"echo","arguments":{"value":"ok"}}</tool_call>',
])
def test_validation_and_extraction_use_retained_declarations_without_global_fallback(monkeypatch, response):
    config = ValidatorConfig([], {"echo": {"parameters": [{"name": "value"}], "required_params": ["value"]}})
    monkeypatch.setattr(dataset_validator, "default_validator_config", lambda: pytest.fail("global fallback"))
    result = validate_assistant_response(response, validator_config=config)
    assert result.passed and [(call.name, call.arguments) for call in result.tool_calls] == [("echo", {"value": "ok"})]
    invalid = ValidatorConfig([], {"echo": {"parameters": [{"name": "value"}], "required_params": ["missing"]}})
    assert not validate_assistant_response(response, validator_config=invalid).passed


def test_actual_runner_consumes_retained_config_parallel_and_sequential(monkeypatch):
    config = ValidatorConfig([], {})
    monkeypatch.setattr(dataset_validator, "default_validator_config", lambda: pytest.fail("global fallback"))
    case = PromptCase("one", "hello", metadata={"correct": {"any": [{"assertions": [
        {"type": "jsonpath_regex", "path": "$.content", "pattern": "hello"}]}]}})
    class Client:
        def chat(self, messages):
            return SimpleNamespace(message="hello", raw={}, latency_s=0.1)
    for parallel in (False, True):
        results = evaluate_cases([deepcopy(case), deepcopy(case)], Client(), parallel=parallel, validator_config=config)
        assert all(record.passed for record in results)


@pytest.mark.parametrize("key,value", [("headers", {"Authorization": "secret"}), ("endpoint", "https://x?token=secret")])
def test_backend_control_allowlist_rejects_secret_or_endpoint_fields(key, value):
    class Client:
        def comparison_configuration(self):
            return {"backend": "fixture", "revision": "v1", "generation": {name: None for name in GENERATION_KEYS}, key: value}
    with pytest.raises(ValueError):
        backend_controls(Client())


@pytest.mark.parametrize("key", ["retrieval_config", "audio_config", "environment", "scoring"])
def test_unresolved_late_dependencies_remain_explicitly_unverified(key):
    case = PromptCase("one", "hello", metadata={key: {"configured": True}})
    controls = {"backend": "fixture", "revision": "v1", "generation": {name: None for name in GENERATION_KEYS}}
    basis = build_basis((("source", (case,)),), ValidatorConfig([], {}), controls)
    assert basis["status"] == "unverified" and basis["reasons"] == ["external_scoring_inputs_unavailable"]


def test_local_http_projection_matches_actual_request_settings_without_endpoint_or_key():
    from synaptic_tuner.api.v1.reference.evaluation import LocalHttpEvaluatorBackendV1
    from synaptic_tuner.api.v1.evaluation_facade import EvaluationModelRef
    client = LocalHttpEvaluatorBackendV1("vllm", host="localhost", port=8000).open(EvaluationModelRef("fixture", "revision"))
    controls = backend_controls(client)
    payload = client._build_payload([{"role": "user", "content": "hello"}])
    for key, value in controls["generation"].items():
        assert payload.get(key) == value
    assert controls["generation"]["max_tokens"] == 1024
    assert "host" not in controls and "port" not in controls and "api_key" not in controls
    client.settings.top_p = 0.5
    assert backend_controls(client)["generation"]["top_p"] == 0.5
