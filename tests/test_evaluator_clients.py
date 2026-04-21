from Evaluator.config import VLLMSettings
from Evaluator.vllm_client import VLLMClient


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
