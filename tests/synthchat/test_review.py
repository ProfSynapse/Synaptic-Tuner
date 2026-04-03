"""Tests for SynthChat.review — stage review and judge template building."""
from __future__ import annotations

import json
import pytest
from SynthChat.review import (
    build_environment_generation_review_payload,
    build_stage_judge_template_vars,
    run_configured_stage_judge,
    run_stage_review,
)


# ---- build_stage_judge_template_vars ----

class TestBuildStageJudgeTemplateVars:
    def test_all_keys_present(self):
        result = build_stage_judge_template_vars(
            stage_name="assistant",
            scenario_key="test_sc",
            scenario={"type": "tool"},
            task_context={"goal": "test"},
            payload={"value": "data", "text": "hello"},
        )
        expected_keys = {
            "stage_name", "scenario_key", "scenario_json", "task_context_json",
            "payload_json", "value_json", "text", "system_text", "user_text",
            "assistant_response_json", "environment_result_json",
            "conversation_trace_json", "hard_requirements_json", "quality_rubric_json",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_values_are_strings(self):
        result = build_stage_judge_template_vars(
            stage_name="user",
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
        )
        for key, value in result.items():
            assert isinstance(value, str), f"{key} is not a string"

    def test_scenario_json_is_valid(self):
        result = build_stage_judge_template_vars(
            stage_name="user",
            scenario_key="sc",
            scenario={"type": "tool", "tool": "test"},
            task_context={},
            payload={},
        )
        parsed = json.loads(result["scenario_json"])
        assert parsed["type"] == "tool"


# ---- build_environment_generation_review_payload ----

class TestBuildEnvironmentGenerationReviewPayload:
    def test_basic_structure(self):
        env = {
            "environment": {
                "fixture": {
                    "directories": ["notes/"],
                    "files": {"notes/a.md": "content"},
                },
                "assertions": [{"type": "path_exists", "path": "notes/a.md"}],
            },
            "system_context": {"session_id": "s1"},
            "task_context": {"goal": "test"},
        }
        result = build_environment_generation_review_payload(generated_environment=env)
        assert "value" in result
        assert "generated_environment" in result
        assert result["value"]["system_context"]["session_id"] == "s1"

    def test_seed_id_included(self):
        result = build_environment_generation_review_payload(
            generated_environment={"environment": {}},
            seed_id="seed_42",
        )
        assert result["seed_id"] == "seed_42"

    def test_non_dict_environment_handled(self):
        result = build_environment_generation_review_payload(
            generated_environment={"environment": None}
        )
        assert "value" in result

    def test_fixture_error_captured(self):
        """Invalid fixture config should not crash, but capture the error."""
        # environment with fixture that references missing data
        result = build_environment_generation_review_payload(
            generated_environment={"environment": {"fixture": {"notes": "invalid"}}}
        )
        # Should have a fixture_snapshot — may have error key if parsing failed
        assert "value" in result


# ---- run_configured_stage_judge ----

class TestRunConfiguredStageJudge:
    def test_disabled_returns_none(self):
        result = run_configured_stage_judge(
            stage_name="test",
            judge_config={"enabled": False, "prompt": "judge this"},
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
            llm_client=None,
            get_stage_llm_clients=lambda _: [],
        )
        assert result is None

    def test_none_config_returns_none(self):
        result = run_configured_stage_judge(
            stage_name="test",
            judge_config=None,
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
            llm_client=None,
            get_stage_llm_clients=lambda _: [],
        )
        assert result is None

    def test_missing_prompt_returns_none(self):
        result = run_configured_stage_judge(
            stage_name="test",
            judge_config={"enabled": True, "prompt": ""},
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
            llm_client=None,
            get_stage_llm_clients=lambda _: [],
        )
        assert result is None


# ---- run_stage_review ----

class TestRunStageReview:
    def test_none_config_returns_none(self):
        result = run_stage_review(
            stage_name="test",
            stage_config=None,
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
            llm_client=None,
            get_stage_llm_clients=lambda _: [],
        )
        assert result is None

    def test_empty_config_returns_none(self):
        result = run_stage_review(
            stage_name="test",
            stage_config={},
            scenario_key="sc",
            scenario={},
            task_context={},
            payload={},
            llm_client=None,
            get_stage_llm_clients=lambda _: [],
        )
        assert result is None
