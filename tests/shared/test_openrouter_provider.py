from __future__ import annotations

import pytest

from shared.llm.exceptions import LLMResponseError
from shared.llm.providers.openrouter import OpenRouterClient


def test_openrouter_structured_output_parse_failure_includes_raw_response():
    client = OpenRouterClient(api_key="test-key", model="test-model")

    def fake_make_request(payload):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"environment": {"fixture": {"files": [\n  {"path": "foo.md", "content": "bar"}\n'
                    }
                }
            ]
        }

    client._make_request = fake_make_request  # type: ignore[method-assign]

    with pytest.raises(LLMResponseError) as exc_info:
        client.structured_output(
            messages=[{"role": "user", "content": "Generate JSON"}],
            schema={"name": "response", "type": "object"},
        )

    err = exc_info.value
    assert err.raw_response is not None
    assert '"environment"' in err.raw_response
    assert "Response excerpt:" in str(err)
