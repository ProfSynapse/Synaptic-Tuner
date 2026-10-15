"""Agent-as-harness review of student proposals before execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from shared.agentic_loop import AgenticModelResponse
from SynthChat.agentic.harness_review import (
    HarnessReviewConfig,
    build_harnessed_respond,
    build_review_schema,
)
from SynthChat.parsing import stringify_assistant_message


@dataclass
class _Validation:
    passed: bool
    issues: List[Dict[str, str]] = field(default_factory=list)


def _validate(message: Any) -> _Validation:
    if not isinstance(message, dict):
        return _Validation(False, [{"message": "not a message"}])
    for call in message.get("tool_calls") or []:
        if not str(call["function"]["arguments"]).startswith("{\""):
            return _Validation(False, [{"message": "Failed to parse arguments JSON"}])
    return _Validation(True)


def _tool(command: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "useTools", "arguments": f'{{"tool": "{command}"}}'}}],
    }


def _text(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": None}


def _replacement(message: Dict[str, Any]) -> Dict[str, Any]:
    return {"content": message.get("content"), "tool_calls": message.get("tool_calls")}


def _config(**overrides) -> HarnessReviewConfig:
    raw = {
        "student": {"prompt": "next turn"},
        "reviewer": {"prompt": "{proposal_kind} | {triggered_middleware} | {structural_defects} | {new_events}"},
        "reference_harness": {
            "components": [{"id": "read-first", "content": "Read before answering."}],
            "middleware": [
                {"id": "final-before-read", "final_only": True, "unless_prior_pattern": "content read", "hint": "read first"}
            ],
        },
        "replace_budget": 1,
        "leak_patterns": ["\\breviewer\\b"],
        "reviewer_voice_patterns": ["\\bthe student\\b"],
    }
    raw.update(overrides)
    return HarnessReviewConfig.from_scenario(raw)


def _harness(config, proposals, submissions):
    prompts: List[str] = []
    proposal_iter = iter(proposals)
    submission_iter = iter(submissions)

    def student(messages, turn_index):
        item = next(proposal_iter)
        if isinstance(item, Exception):
            raise item
        return AgenticModelResponse(message=item)

    def review(prompt, system_prompt):
        prompts.append(prompt)
        item = next(submission_iter)
        if isinstance(item, Exception):
            raise item
        return item

    respond, state = build_harnessed_respond(
        config=config,
        scenario_key="s",
        action_space="useTools",
        student_respond=student,
        review=review,
        parse_replacement=lambda payload: {"role": "assistant", **payload},
        validate=_validate,
        stringify=stringify_assistant_message,
    )
    return respond, state, prompts


_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "find the note"}]


def test_pass_executes_student_proposal_unchanged():
    proposal = _tool('search content \\"reset\\"')
    respond, state, _ = _harness(_config(), [proposal], [{"decision": "PASS", "reason": "ok", "components_used": [], "replacement": None}])
    result = respond(_MESSAGES, 1)
    assert result.message == proposal
    assert state.turns[0]["decision"] == "PASS"
    assert state.replacements_used == 0


def test_replace_swaps_in_correction_and_records_proposal():
    proposal = _text("The note says to reset it.")
    correction = _tool('content read \\"docs/a.md\\" 1')
    respond, state, prompts = _harness(
        _config(),
        [proposal],
        [{"decision": "REPLACE", "reason": "unread", "components_used": ["read-first"], "replacement": _replacement(correction)}],
    )
    result = respond(_MESSAGES, 1)
    assert result.message["tool_calls"] == correction["tool_calls"]
    entry = state.turns[0]
    assert entry["proposal"] == proposal and entry["decision"] == "REPLACE"
    assert entry["triggered_middleware"] == ["final-before-read"]
    assert "final answer" in prompts[0] and "read first" in prompts[0]
    assert state.replacements_used == 1


def test_invalid_submissions_are_reprompted_with_feedback():
    proposal = _tool('search content \\"x\\"')
    respond, state, prompts = _harness(
        _config(),
        [proposal],
        [
            {"decision": "PASS", "reason": "", "components_used": [], "replacement": _replacement(proposal)},
            {"decision": "REPLACE", "reason": "", "components_used": [], "replacement": _replacement(_text("As the reviewer, I fixed this."))},
            {"decision": "PASS", "reason": "fine", "components_used": [], "replacement": None},
        ],
    )
    assert respond(_MESSAGES, 1).message == proposal
    assert len(prompts) == 3
    assert "PASS must set replacement to null" in prompts[1]
    assert "private review material" in prompts[2]


def test_budget_exhaustion_rejects_further_replacements():
    first, second = _tool('search content \\"a\\"'), _tool('search content \\"b\\"')
    fix = _replacement(_tool('search content \\"c\\"'))
    respond, state, prompts = _harness(
        _config(),
        [first, second],
        [
            {"decision": "REPLACE", "reason": "", "components_used": [], "replacement": fix},
            {"decision": "REPLACE", "reason": "", "components_used": [], "replacement": fix},
            {"decision": "PASS", "reason": "", "components_used": [], "replacement": None},
        ],
    )
    respond(_MESSAGES, 1)
    assert respond(_MESSAGES + [{"role": "user", "content": "results"}], 2).message == second
    assert "budget is exhausted" in prompts[2]


def test_structural_defects_force_replace_without_spending_budget():
    broken = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "useTools", "arguments": "{'tool': 'x'}"}}]}
    fix = _replacement(_tool('search content \\"x\\"'))
    respond, state, prompts = _harness(
        _config(replace_budget=0),
        [broken],
        [
            {"decision": "PASS", "reason": "", "components_used": [], "replacement": None},
            {"decision": "REPLACE", "reason": "fix json", "components_used": [], "replacement": fix},
        ],
    )
    assert respond(_MESSAGES, 1).message["tool_calls"] == fix["tool_calls"]
    assert "Failed to parse arguments JSON" in prompts[0]
    assert state.turns[0]["structural_replacement"] is True
    assert state.replacements_used == 0


def test_unparseable_student_turn_must_be_replaced():
    fix = _replacement(_tool('search content \\"x\\"'))
    respond, state, prompts = _harness(
        _config(),
        [ValueError("Failed to parse structured output")],
        [
            {"decision": "PASS", "reason": "", "components_used": [], "replacement": None},
            {"decision": "REPLACE", "reason": "", "components_used": [], "replacement": fix},
        ],
    )
    assert respond(_MESSAGES, 1).message["tool_calls"] == fix["tool_calls"]
    assert "no parseable response" in prompts[0]
    assert state.turns[0]["structural_replacement"] is True


def test_no_valid_submission_marks_review_failed_and_keeps_proposal():
    proposal = _tool('search content \\"x\\"')
    respond, state, _ = _harness(_config(), [proposal], [RuntimeError("down")] * 3)
    assert respond(_MESSAGES, 1).message == proposal
    assert state.failed and state.turns[0]["decision"] == "FAILED"


def test_reviewer_voice_is_flagged_and_new_events_advance():
    voiced = _replacement(_text("The student should read the note first."))
    respond, state, prompts = _harness(
        _config(reviewer_voice_patterns=["\\bthe student\\b"], leak_patterns=[]),
        [_tool('search content \\"a\\"'), _text("done")],
        [
            {"decision": "PASS", "reason": "", "components_used": [], "replacement": None},
            {"decision": "REPLACE", "reason": "", "components_used": [], "replacement": voiced},
        ],
    )
    respond(_MESSAGES, 1)
    later = _MESSAGES + [{"role": "assistant", "content": "Tool calls: [...]"}, {"role": "user", "content": "SEARCH RESULT"}]
    respond(later, 2)
    assert state.turns[1]["reviewer_voice"] is True
    assert "SEARCH RESULT" in prompts[1] and "find the note" not in prompts[1].split("|")[-1]


def test_review_schema_wraps_student_schema():
    student_schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    schema = build_review_schema(student_schema)
    assert schema["properties"]["replacement"]["anyOf"][0] is student_schema
    assert set(schema["required"]) == {"decision", "reason", "components_used", "replacement"}


def test_pilot_scenario_config_parses():
    path = Path(__file__).resolve().parents[2] / "SynthChat" / "scenarios" / "workspace_harness_review_pilot.yaml"
    scenario = yaml.safe_load(path.read_text())["scenarios"]["envfs_harness_review_search_then_read"]
    config = HarnessReviewConfig.from_scenario(scenario["harness_review"])
    assert config.student["model"] and config.reviewer["model"]
    assert {c["id"] for c in config.components} >= {"read-before-answering", "search-recovery"}
    rendered_system = yaml.safe_dump(scenario["system_context"])
    assert scenario["task_context"]["final_answer"] not in rendered_system


def test_caught_exception_bodies_never_enter_prompts_or_trace():
    import json
    sentinel = "PRIVATE_PROVIDER_CREDENTIAL_RESPONSE_SENTINEL"
    fix = _replacement(_text("safe replacement"))
    respond, state, prompts = _harness(
        _config(), [RuntimeError(sentinel)],
        [RuntimeError(sentinel), {"decision": "REPLACE", "reason": "safe", "components_used": [], "replacement": fix}],
    )
    assert respond(_MESSAGES, 1).message["content"] == "safe replacement"
    retained = json.dumps({"prompts": prompts, "trace": state.summary(1)})
    assert sentinel not in retained
    assert "STUDENT_RESPONSE_FAILURE" in retained
    assert "REVIEW_CALL_FAILURE" in retained
