"""Test backward-compat re-exports from generator.py.

The decomposition must preserve all 5 private functions that were previously
imported from generator.py by tests and run.py. These re-exports guarantee
that existing callers continue working without import changes.
"""
from __future__ import annotations


def test_build_use_tools_response_schema_reexported():
    """_build_use_tools_response_schema must be importable from generator.py."""
    from SynthChat.generator import _build_use_tools_response_schema
    schema = _build_use_tools_response_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "tool_calls" in schema["properties"]


def test_build_use_tools_generation_prompt_reexported():
    """_build_use_tools_generation_prompt must be importable from generator.py."""
    from SynthChat.generator import _build_use_tools_generation_prompt
    prompt = _build_use_tools_generation_prompt(
        base_prompt="Do something",
        wrapper_name="useTools",
        allowed_tools=["fileManager_read"],
    )
    assert isinstance(prompt, str)
    assert "useTools" in prompt
    assert "fileManager_read" in prompt


def test_apply_stage_review_result_reexported():
    """_apply_stage_review_result must be importable from generator.py."""
    from SynthChat.generator import _apply_stage_review_result
    failures = []
    reviews = {}
    _apply_stage_review_result(
        failures, reviews, "test_stage",
        {"passed": False, "enforce": True},
    )
    assert "test_stage" in failures
    assert "test_stage" in reviews


def test_normalize_target_spec_reexported():
    """_normalize_target_spec must be importable from generator.py."""
    from SynthChat.generator import _normalize_target_spec
    result = _normalize_target_spec(5)
    assert result == {"seed_count": 5, "rollouts_per_seed": 1}


def test_extract_shared_seed_spec_reexported():
    """_extract_shared_seed_spec must be importable from generator.py."""
    from SynthChat.generator import _extract_shared_seed_spec
    spec, cleaned = _extract_shared_seed_spec({"scenario_a": 3})
    assert spec is None
    assert cleaned == {"scenario_a": 3}


def test_parse_assistant_response_reexported():
    """parse_assistant_response must be importable from generator.py."""
    from SynthChat.generator import parse_assistant_response
    result = parse_assistant_response("Hello there", {})
    assert result["role"] == "assistant"
    assert result["content"] == "Hello there"


def test_stringify_assistant_message_reexported():
    """stringify_assistant_message must be importable from generator.py."""
    from SynthChat.generator import stringify_assistant_message
    result = stringify_assistant_message("plain text")
    assert result == "plain text"


def test_normalize_generated_environment_reexported():
    """normalize_generated_environment must be importable from generator.py."""
    from SynthChat.generator import normalize_generated_environment
    result = normalize_generated_environment({"environment": {"fixture": {}}})
    assert "environment" in result


def test_build_canonical_environment_schema_reexported():
    """_build_canonical_environment_schema must be importable from generator.py."""
    from SynthChat.generator import _build_canonical_environment_schema
    schema = _build_canonical_environment_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "environment" in schema["properties"]


def test_build_metadata_labels_reexported():
    """build_metadata_labels must be importable from generator.py."""
    from SynthChat.generator import build_metadata_labels
    result = build_metadata_labels(
        scenario_key="test",
        scenario={"type": "tool"},
        environment_mode="generated",
        stage_failures=[],
        environment_trace=None,
        generated_environment={},
    )
    assert "flat" in result
    assert "filter" in result


def test_llm_client_pool_reexported():
    """LLMClientPool must be importable from generator.py."""
    from SynthChat.generator import LLMClientPool
    assert LLMClientPool is not None


def test_call_llm_functions_reexported():
    """call_llm and call_llm_structured must be importable from generator.py."""
    from SynthChat.generator import call_llm, call_llm_structured
    assert callable(call_llm)
    assert callable(call_llm_structured)
