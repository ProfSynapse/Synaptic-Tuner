"""Tests for SynthChat.agentic — turn judge template building and episode helpers."""
from __future__ import annotations

import json
import pytest
from SynthChat.agentic.episode import (
    build_turn_judge,
    build_turn_judge_template_vars,
    synthchat_loop_response,
)


# ---- build_turn_judge_template_vars ----

class TestBuildTurnJudgeTemplateVars:
    def _base_kwargs(self, **overrides):
        defaults = {
            "scenario_key": "test_sc",
            "scenario": {"type": "tool"},
            "assistant_prompt": "Help the user",
            "system_context": {"session_id": "s1"},
            "task_context": {"goal": "test"},
            "hard_requirements": [{"desc": "must read file"}],
            "quality_rubric": ["be helpful"],
            "turn_payload": {
                "messages": [
                    {"role": "user", "content": "Please read the file"},
                    {"role": "assistant", "content": "I will read it."},
                ],
                "response_message": {"role": "assistant", "content": "Done"},
                "turn_index": 2,
            },
        }
        defaults.update(overrides)
        return defaults

    def test_all_keys_present(self):
        result = build_turn_judge_template_vars(**self._base_kwargs())
        expected = {
            "scenario_key", "assistant_prompt", "scenario_json",
            "system_context_json", "task_context_json",
            "hard_requirements_json", "quality_rubric_json",
            "messages_json", "latest_user_message",
            "assistant_response_json", "validation_json",
            "environment_step_json", "environment_preview_json",
            "tool_feedback", "turn_index",
        }
        assert expected.issubset(set(result.keys()))

    def test_latest_user_message_extracted(self):
        result = build_turn_judge_template_vars(**self._base_kwargs())
        assert result["latest_user_message"] == "Please read the file"

    def test_no_user_messages(self):
        result = build_turn_judge_template_vars(**self._base_kwargs(
            turn_payload={"messages": [{"role": "assistant", "content": "hi"}]}
        ))
        assert result["latest_user_message"] == ""

    def test_values_are_strings(self):
        result = build_turn_judge_template_vars(**self._base_kwargs())
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is not a string"

    def test_turn_index_rendered(self):
        result = build_turn_judge_template_vars(**self._base_kwargs())
        assert result["turn_index"] == "2"


# ---- build_turn_judge ----

class TestBuildTurnJudge:
    def _base_kwargs(self, **overrides):
        defaults = {
            "scenario_key": "test",
            "scenario": {},
            "assistant_prompt": "help",
            "system_context": {},
            "task_context": {},
            "hard_requirements": [],
            "quality_rubric": [],
            "judge_config": {},
            "llm_client": None,
            "get_stage_llm_clients": lambda _: [],
        }
        defaults.update(overrides)
        return defaults

    def test_empty_config_returns_none(self):
        assert build_turn_judge(**self._base_kwargs()) is None

    def test_disabled_returns_none(self):
        config = {"enabled": False, "prompt": "judge"}
        assert build_turn_judge(**self._base_kwargs(judge_config=config)) is None

    def test_missing_prompt_returns_none(self):
        config = {"enabled": True, "prompt": ""}
        assert build_turn_judge(**self._base_kwargs(judge_config=config)) is None


# ---- synthchat_loop_response ----

class TestSynthchatLoopResponse:
    def test_basic_response(self):
        def mock_build_context(messages):
            return "context from messages"

        def mock_generate(scenario, system_context, assistant_context,
                         assistant_prompt, randomize_params, trace_label):
            return "I will help you."

        def mock_parse(content, scenario):
            return {"role": "assistant", "content": content}

        result = synthchat_loop_response(
            scenario={"type": "tool"},
            system_context={},
            messages=[{"role": "user", "content": "help"}],
            assistant_prompt="assist",
            randomize_params=False,
            scenario_key="sc",
            turn_index=0,
            thinking_content=None,
            build_loop_assistant_context=mock_build_context,
            generate_assistant_response=mock_generate,
            parse_response=mock_parse,
        )
        assert result.message["role"] == "assistant"
        assert result.message["content"] == "I will help you."

    def test_thinking_content_injected(self):
        context_received = {}

        def mock_build_context(messages):
            return "base context"

        def mock_generate(scenario, system_context, assistant_context,
                         assistant_prompt, randomize_params, trace_label):
            context_received["ctx"] = assistant_context
            return '{"content": null, "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}'

        def mock_parse(content, scenario):
            return {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "t"}}]}

        result = synthchat_loop_response(
            scenario={},
            system_context={},
            messages=[],
            assistant_prompt="assist",
            randomize_params=False,
            scenario_key="sc",
            turn_index=1,
            thinking_content="I need to read the file first",
            build_loop_assistant_context=mock_build_context,
            generate_assistant_response=mock_generate,
            parse_response=mock_parse,
        )
        assert "I need to read the file first" in context_received["ctx"]
        assert "<thinking>" in result.message["content"]
