from Evaluator.config import LMStudioSettings, OllamaSettings, VLLMSettings


def test_vllm_settings_support_https_base_url():
    settings = VLLMSettings(model="google/gemma-4-E4B-it", host="example.hf.space", port=443, scheme="https")
    assert settings.base_url() == "https://example.hf.space"


def test_vllm_settings_can_carry_api_key():
    settings = VLLMSettings(model="google/gemma-4-E4B-it", host="example.hf.space", port=443, scheme="https", api_key="hf_test")
    assert settings.api_key == "hf_test"


def test_lmstudio_settings_default_to_http():
    settings = LMStudioSettings(model="foo", host="localhost", port=1234)
    assert settings.base_url() == "http://localhost:1234"


def test_ollama_settings_default_to_http():
    settings = OllamaSettings(model="foo", host="127.0.0.1", port=11434)
    assert settings.base_url() == "http://127.0.0.1:11434"


def test_http_default_port_is_omitted():
    settings = OllamaSettings(model="foo", host="127.0.0.1", port=80, scheme="http")
    assert settings.base_url() == "http://127.0.0.1"
