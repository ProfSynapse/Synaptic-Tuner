from __future__ import annotations

from shared.llm.config import LLMConfig


def test_openrouter_provider_routing_preserves_privacy_controls(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    routing = {
        "order": ["Example Provider"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": ["Example Provider"],
        "ignore": ["Fallback Provider"],
    }

    config = LLMConfig.from_env(
        config_defaults={
            "provider": "openrouter",
            "model": "example/model",
            "provider_routing": routing,
        }
    )

    assert config.provider_routing == routing
    assert config.provider_routing is not routing


def test_openrouter_provider_routing_requires_mapping(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    try:
        LLMConfig.from_env(
            config_defaults={
                "provider_routing": ["zdr"],
            }
        )
    except ValueError as error:
        assert str(error) == "provider_routing must be a mapping"
    else:
        raise AssertionError("Expected provider_routing mapping validation")
