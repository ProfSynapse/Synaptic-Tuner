from Evaluator.config import VLLMSettings
from Evaluator.vllm_client import VLLMClient

import pytest


@pytest.mark.parametrize("limit", [None, 1, 8192])
def test_vllm_output_limit_null_omits_token_fields(limit):
    settings = VLLMSettings(model="generic-model", max_tokens=limit)
    payload = VLLMClient(settings)._build_payload([{"role": "user", "content": "hello"}])
    assert "max_completion_tokens" not in payload
    if limit is None:
        assert "max_tokens" not in payload
    else:
        assert payload["max_tokens"] == limit


def test_vllm_omitted_generation_options_preserve_default_payload():
    payload = VLLMClient(VLLMSettings(model="generic-model"))._build_payload([])
    assert payload["max_tokens"] == 1024
    assert "chat_template_kwargs" not in payload


def test_vllm_template_kwargs_are_copied_and_forwarded_in_request_json():
    kwargs = {"enable_thinking": False, "style": {"name": "brief", "options": [1, None]}}
    settings = VLLMSettings(model="generic-model", chat_template_kwargs=kwargs)
    payload = VLLMClient(settings)._build_payload([])
    assert payload["chat_template_kwargs"] == kwargs
    payload["chat_template_kwargs"]["style"]["name"] = "changed"
    assert settings.chat_template_kwargs["style"]["name"] == "brief"
    assert kwargs["style"]["name"] == "brief"


@pytest.mark.parametrize("kwargs", [{"messages": []}, {"tokenize": False},
    {"add_generation_prompt": False}, {"return_dict": True}, {"return_tensors": "pt"},
    {"continue_final_message": True}, {"chat_template": "custom"},
    {"value": float("nan")}, {"value": "x" * 4097}, []])
def test_vllm_template_kwargs_reject_reserved_controls_and_unbounded_json(kwargs):
    with pytest.raises(ValueError):
        VLLMSettings(model="generic-model", chat_template_kwargs=kwargs)


def test_vllm_mutated_template_settings_are_revalidated_before_request():
    settings = VLLMSettings(model="generic-model", chat_template_kwargs={"style": "brief"})
    settings.chat_template_kwargs["messages"] = []
    with pytest.raises(ValueError):
        VLLMClient(settings)._build_payload([])


def test_openai_compat_client_adds_bearer_header_when_api_key_present():
    settings = VLLMSettings(
        model="google/gemma-4-E4B-it",
        host="example.hf.space",
        port=443,
        scheme="https",
        api_key="hf_test",
    )
    client = VLLMClient(settings=settings)
    assert client._request_headers() == {"Authorization": "Bearer hf_test"}
