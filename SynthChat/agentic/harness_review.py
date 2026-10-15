"""SynthChat Harness Review - agent-as-harness review of student rollouts.

Location: SynthChat/agentic/harness_review.py
Purpose: Wrap a student model's agentic turns with a reviewing agent that
         passes or replaces each proposal before it executes (Harness-Zero,
         arXiv:2609.24974). The reviewer is guided by a private reference
         harness from scenario config; only accepted responses enter the
         student-visible trajectory, and every review decision is traced.
Usage: Built by generate_agentic_episode when a scenario sets
       ``harness_review.enabled``. The returned ``respond`` callable plugs into
       run_environment_episode unchanged.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..template_utils import _make_json_safe

try:
    from shared.agentic_judge import render_agentic_judge_prompt
    from shared.agentic_loop import AgenticModelResponse
except ImportError:
    render_agentic_judge_prompt = None
    AgenticModelResponse = None


PASS = "PASS"
REPLACE = "REPLACE"


@dataclass
class HarnessMiddleware:
    """Deterministic review check that adds private guidance when it matches a proposal."""

    id: str
    hint: str
    pattern: Optional[str] = None
    final_only: bool = False
    tool_only: bool = False
    unless_prior_pattern: Optional[str] = None

    def matches(self, proposal_text: str, is_final: bool, prior_assistant_text: str) -> bool:
        if self.final_only and not is_final:
            return False
        if self.tool_only and is_final:
            return False
        if self.pattern and not re.search(self.pattern, proposal_text, flags=re.IGNORECASE):
            return False
        if self.unless_prior_pattern and re.search(self.unless_prior_pattern, prior_assistant_text, flags=re.IGNORECASE):
            return False
        return True


@dataclass
class HarnessReviewConfig:
    """Resolved ``harness_review`` scenario block."""

    student: Dict[str, Any]
    reviewer: Dict[str, Any]
    components: List[Dict[str, str]]
    middleware: List[HarnessMiddleware]
    replace_budget: int
    max_submission_attempts: int
    leak_patterns: List[str]
    reviewer_voice_patterns: List[str]

    @classmethod
    def from_scenario(cls, raw: Mapping[str, Any]) -> "HarnessReviewConfig":
        student = raw.get("student")
        reviewer = raw.get("reviewer")
        if not isinstance(student, dict) or not str(student.get("prompt") or "").strip():
            raise ValueError("harness_review.student must be a mapping with a non-empty prompt")
        if not isinstance(reviewer, dict) or not str(reviewer.get("prompt") or "").strip():
            raise ValueError("harness_review.reviewer must be a mapping with a non-empty prompt")
        reference = raw.get("reference_harness") if isinstance(raw.get("reference_harness"), dict) else {}
        components = []
        for item in reference.get("components") or []:
            if not isinstance(item, dict) or not item.get("id") or not str(item.get("content") or "").strip():
                raise ValueError("reference_harness.components entries need id and content")
            components.append(
                {"id": str(item["id"]), "kind": str(item.get("kind") or "skill"), "content": str(item["content"]).strip()}
            )
        middleware = []
        for item in reference.get("middleware") or []:
            if not isinstance(item, dict) or not item.get("id") or not str(item.get("hint") or "").strip():
                raise ValueError("reference_harness.middleware entries need id and hint")
            middleware.append(
                HarnessMiddleware(
                    id=str(item["id"]),
                    hint=str(item["hint"]).strip(),
                    pattern=item.get("pattern"),
                    final_only=bool(item.get("final_only", False)),
                    tool_only=bool(item.get("tool_only", False)),
                    unless_prior_pattern=item.get("unless_prior_pattern"),
                )
            )
        return cls(
            student=dict(student),
            reviewer=dict(reviewer),
            components=components,
            middleware=middleware,
            replace_budget=int(raw.get("replace_budget", 3)),
            max_submission_attempts=max(1, int(raw.get("max_submission_attempts", 3))),
            leak_patterns=[str(p) for p in raw.get("leak_patterns") or []],
            reviewer_voice_patterns=[str(p) for p in raw.get("reviewer_voice_patterns") or []],
        )


@dataclass
class HarnessReviewState:
    """Mutable per-episode review state and trace."""

    turns: List[Dict[str, Any]] = field(default_factory=list)
    replacements_used: int = 0
    events_reviewed: int = 0
    failed: bool = False

    def summary(self, replace_budget: int) -> Dict[str, Any]:
        decisions = [turn.get("decision") for turn in self.turns]
        return {
            "replace_budget": replace_budget,
            "reviews": len(self.turns),
            "passes": decisions.count(PASS),
            "replacements": decisions.count(REPLACE),
            "budgeted_replacements": self.replacements_used,
            "reviewer_voice_turns": [t["turn_index"] for t in self.turns if t.get("reviewer_voice")],
            "failed": self.failed,
            "turns": self.turns,
        }


def build_review_schema(response_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap the student's response schema in a PASS/REPLACE submission schema."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": [PASS, REPLACE]},
            "reason": {"type": "string"},
            "components_used": {"type": "array", "items": {"type": "string"}},
            "replacement": {"anyOf": [response_schema, {"type": "null"}]},
        },
        "required": ["decision", "reason", "components_used", "replacement"],
    }


def render_reference_harness(components: Sequence[Mapping[str, str]]) -> str:
    blocks = [f'<component id="{c["id"]}" kind="{c["kind"]}">\n{c["content"]}\n</component>' for c in components]
    return "\n\n".join(blocks)


def render_events(messages: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(_make_json_safe(content), ensure_ascii=False)
        if text and text.strip():
            parts.append(f"{role}:\n{text}")
    return "\n\n".join(parts) or "(no new events)"


def _validation_issues(validation: Any) -> List[str]:
    issues = getattr(validation, "issues", None)
    if issues is None and isinstance(validation, dict):
        issues = validation.get("issues")
    rendered = []
    for issue in issues or []:
        message = issue.get("message") if isinstance(issue, dict) else getattr(issue, "message", None)
        rendered.append(str(message or issue))
    return rendered


def _validation_passed(validation: Any) -> bool:
    passed = getattr(validation, "passed", None)
    if passed is None and isinstance(validation, dict):
        passed = validation.get("passed")
    return bool(passed)


def _is_final(message: Any) -> bool:
    return not (isinstance(message, dict) and message.get("tool_calls"))


def build_harnessed_respond(
    *,
    config: HarnessReviewConfig,
    scenario_key: str,
    action_space: str,
    student_respond: Callable[[Sequence[Mapping[str, Any]], int], Any],
    review: Callable[[str, Optional[str]], Dict[str, Any]],
    parse_replacement: Callable[[Dict[str, Any]], Dict[str, Any]],
    validate: Callable[[Any], Any],
    stringify: Callable[[Any], str],
    logger: Any = None,
):
    """Return ``(respond, state)`` where respond reviews each student proposal before execution.

    ``review(prompt, system_prompt)`` returns a raw submission dict matching
    build_review_schema. ``parse_replacement`` turns a replacement payload into
    an assistant message in the student's action space.
    """
    state = HarnessReviewState()
    reference_harness = render_reference_harness(config.components)
    leak_res = [re.compile(p, flags=re.IGNORECASE) for p in config.leak_patterns]
    voice_res = [re.compile(p, flags=re.IGNORECASE) for p in config.reviewer_voice_patterns]

    def check_submission(submission: Any, proposal_valid: bool, structural: bool) -> tuple:
        """Return (decision, accepted_message_or_None, error_or_None)."""
        if not isinstance(submission, dict):
            return None, None, "submission must be a JSON object"
        decision = str(submission.get("decision") or "").strip().upper()
        replacement = submission.get("replacement")
        if decision not in {PASS, REPLACE}:
            return None, None, "decision must be PASS or REPLACE"
        if decision == PASS:
            if replacement is not None:
                return None, None, "a PASS must set replacement to null"
            if not proposal_valid:
                return None, None, "the proposal has structural defects; PASS would preserve them, so REPLACE it"
            return PASS, None, None
        if not isinstance(replacement, dict):
            return None, None, "a REPLACE must include the complete replacement response"
        if not structural and state.replacements_used >= config.replace_budget:
            return None, None, "the replacement budget is exhausted; PASS this proposal"
        message = parse_replacement(replacement)
        content = message.get("content")
        if not (isinstance(content, str) and content.strip()) and not message.get("tool_calls"):
            return None, None, "a replacement needs visible content or a tool call"
        validation = validate(message)
        if not _validation_passed(validation):
            return None, None, "the replacement failed response validation: " + "; ".join(_validation_issues(validation))
        text = stringify(message)
        for leak in leak_res:
            if leak.search(text):
                return None, None, f"the replacement mentions private review material (matched {leak.pattern!r})"
        return REPLACE, message, None

    def respond(messages: Sequence[Mapping[str, Any]], turn_index: int):
        try:
            proposal_response = student_respond(messages, turn_index)
            proposal = proposal_response.message
            proposal_validation = validate(proposal)
            proposal_valid = _validation_passed(proposal_validation)
            defects = [] if proposal_valid else _validation_issues(proposal_validation) or ["response failed validation"]
        except Exception:  # an unparseable student turn is a structural defect to replace
            proposal_response = None
            proposal = {"role": "assistant", "content": ""}
            proposal_valid = False
            defects = ["the student produced no parseable response: STUDENT_RESPONSE_FAILURE"]
        proposal_text = stringify(proposal)
        prior_assistant = "\n".join(
            stringify(m.get("content")) for m in messages if str(m.get("role", "")).strip() == "assistant"
        )
        triggered = [
            {"id": mw.id, "hint": mw.hint}
            for mw in config.middleware
            if mw.matches(proposal_text, _is_final(proposal), prior_assistant)
        ]
        new_events = list(messages[state.events_reviewed:])
        history = [
            {"turn_index": t["turn_index"], "decision": t["decision"], "reason": t.get("reason", "")}
            for t in state.turns
        ]
        template_vars = {
            "scenario_key": scenario_key,
            "reference_harness": reference_harness,
            "student_action_space": action_space,
            "student_transcript": render_events(messages),
            "new_events": render_events(new_events),
            "proposal_json": json.dumps(_make_json_safe(proposal), ensure_ascii=False, indent=2),
            "proposal_kind": "final answer (no tool call)" if _is_final(proposal) else "tool call",
            "structural_defects": "\n".join(f"- {d}" for d in defects) or "(none)",
            "triggered_middleware": "\n".join(f"- [{t['id']}] {t['hint']}" for t in triggered) or "(none)",
            "review_history_json": json.dumps(history, ensure_ascii=False, indent=2),
            "replacements_used": str(state.replacements_used),
            "replace_budget": str(config.replace_budget),
            "replacements_remaining": str(max(0, config.replace_budget - state.replacements_used)),
            "turn_index": str(turn_index),
        }
        base_prompt = render_agentic_judge_prompt(str(config.reviewer["prompt"]), template_vars)
        system_prompt = config.reviewer.get("system")

        started = time.monotonic()
        attempts: List[Dict[str, Any]] = []
        decision, accepted, submission = None, None, None
        feedback = None
        for _ in range(config.max_submission_attempts):
            prompt = base_prompt if feedback is None else (
                f"{base_prompt}\n\nYour previous submission was rejected: {feedback}\nSubmit again."
            )
            try:
                submission = review(prompt, system_prompt)
            except Exception:  # provider failures count as invalid submissions
                feedback = "review call failed: REVIEW_CALL_FAILURE"
                attempts.append({"error": feedback})
                continue
            decision, accepted, feedback = check_submission(submission, proposal_valid, structural=not proposal_valid)
            attempts.append({"decision": decision, "error": feedback})
            if decision is not None:
                break

        entry: Dict[str, Any] = {
            "turn_index": turn_index,
            "proposal": proposal,
            "proposal_valid": proposal_valid,
            "structural_defects": defects,
            "triggered_middleware": [t["id"] for t in triggered],
            "attempts": attempts,
            "latency_s": round(time.monotonic() - started, 3),
        }
        if decision is None:
            state.failed = True
            entry.update({"decision": "FAILED", "reason": feedback or "no valid submission"})
            accepted = proposal
        else:
            entry.update(
                {
                    "decision": decision,
                    "reason": str(submission.get("reason") or "")[:500],
                    "components_used": list(submission.get("components_used") or []),
                }
            )
            if decision == REPLACE:
                if proposal_valid:
                    state.replacements_used += 1
                entry["structural_replacement"] = not proposal_valid
                entry["accepted"] = accepted
                content = accepted.get("content")
                entry["reviewer_voice"] = bool(
                    isinstance(content, str) and any(v.search(content) for v in voice_res)
                )
            else:
                accepted = proposal
        state.turns.append(entry)
        # The loop appends the accepted response and its observation after this call.
        state.events_reviewed = len(messages) + 1
        if logger:
            logger.info(
                f"[{scenario_key}] harness_review turn={turn_index} decision={entry['decision']} "
                f"replacements={state.replacements_used}/{config.replace_budget}"
            )
        return AgenticModelResponse(
            message=accepted,
            raw={"message": accepted, "harness_review": {"decision": entry["decision"]}},
            latency_s=float(getattr(proposal_response, "latency_s", 0.0) or 0.0),
        )

    return respond, state
