#!/usr/bin/env python3
"""Inspect saved SynthChat/Evaluator rollout traces from JSONL artifacts.

This script is designed for environment-backed artifacts that store:
- `conversation_trace`
- `metadata.environment`
- `metadata.environment.episode_trace`

It is intentionally tolerant of partial artifacts. If `episode_trace.steps[*]`
does not contain model-facing tool feedback, the script reconstructs it from
`conversation_trace` entries with `kind == "tool_feedback"`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to rollout JSONL artifact")
    parser.add_argument("--scenario", help="Only show rows whose metadata.scenario matches")
    parser.add_argument("--seed-id", help="Only show rows with this metadata.environment_seed.seed_id")
    parser.add_argument("--failed-only", action="store_true", help="Only show failed environment rollouts")
    parser.add_argument("--passed-only", action="store_true", help="Only show passed environment rollouts")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of rows to print")
    parser.add_argument("--show-system", action="store_true", help="Include full system prompt in conversation output")
    parser.add_argument("--show-all-messages", action="store_true", help="Print all conversation messages instead of assistant/tool-feedback only")
    parser.add_argument("--max-chars", type=int, default=1600, help="Max characters per printed message/result")
    parser.add_argument("--json-output", action="store_true", help="Emit selected rows as normalized JSON instead of text")
    return parser.parse_args()


def load_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "_meta" in payload:
                continue
            yield payload


def truncate(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def get_environment(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("metadata", {}).get("environment", {}) or {}


def row_passed(row: Dict[str, Any]) -> Optional[bool]:
    environment = get_environment(row)
    passed = environment.get("passed")
    if isinstance(passed, bool):
        return passed
    return None


def parse_tool_feedback_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if message.get("kind") != "tool_feedback":
        return None
    content = str(message.get("content") or "")
    prefix = "Tool execution results:"
    if not content.startswith(prefix):
        return None
    body = content[len(prefix):].strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def build_feedback_map(conversation_trace: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    feedback_by_turn: Dict[int, Dict[str, Any]] = {}
    for entry in conversation_trace:
        feedback = parse_tool_feedback_message(entry)
        if feedback is None:
            continue
        turn_index = entry.get("turn_index")
        if isinstance(turn_index, int):
            feedback_by_turn[turn_index] = feedback
    return feedback_by_turn


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata", {})
    environment = metadata.get("environment", {}) or {}
    episode_trace = environment.get("episode_trace", {}) or {}
    conversation_trace = row.get("conversation_trace", []) or []
    feedback_by_turn = build_feedback_map(conversation_trace)

    normalized_steps: List[Dict[str, Any]] = []
    for step in episode_trace.get("steps", []) or []:
        turn_index = step.get("turn_index")
        feedback = feedback_by_turn.get(turn_index, {})
        normalized_steps.append(
            {
                "turn_index": turn_index,
                "state_changed": step.get("state_changed"),
                "action_signature": step.get("action_signature"),
                "issues": step.get("issues", []),
                "executed_tools": step.get("executed_tools", []),
                "tool_feedback": feedback,
            }
        )

    return {
        "scenario": metadata.get("scenario"),
        "seed_id": metadata.get("environment_seed", {}).get("seed_id"),
        "passed": environment.get("passed"),
        "stop_reason": episode_trace.get("stop_reason"),
        "task_context": metadata.get("task_context", {}),
        "hard_requirements": metadata.get("hard_requirements", []),
        "quality_rubric": metadata.get("quality_rubric", []),
        "derivation_summary": metadata.get("derivation_summary", {}),
        "judge_trace": metadata.get("judge", {}).get("trace", []),
        "final_issues": environment.get("issues", []),
        "conversation_trace": conversation_trace,
        "steps": normalized_steps,
    }


def filter_rows(rows: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        normalized = normalize_row(row)
        if args.scenario and normalized.get("scenario") != args.scenario:
            continue
        if args.seed_id and normalized.get("seed_id") != args.seed_id:
            continue
        passed = normalized.get("passed")
        if args.failed_only and passed is not False:
            continue
        if args.passed_only and passed is not True:
            continue
        selected.append(normalized)
        if len(selected) >= args.limit:
            break
    return selected


def should_show_message(message: Dict[str, Any], show_system: bool, show_all_messages: bool) -> bool:
    if show_all_messages:
        return show_system or message.get("role") != "system"
    if message.get("kind") == "assistant_response":
        return True
    if message.get("kind") == "tool_feedback":
        return True
    if message.get("role") == "user":
        return True
    if show_system and message.get("role") == "system":
        return True
    return False


def print_text(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        print("No matching rollout records found.")
        return

    for index, row in enumerate(rows, start=1):
        print("=" * 100)
        print(f"Record {index}")
        print(f"scenario: {row.get('scenario')}")
        print(f"seed_id: {row.get('seed_id')}")
        print(f"passed: {row.get('passed')}")
        print(f"stop_reason: {row.get('stop_reason')}")
        print(f"task_context: {json.dumps(row.get('task_context'), ensure_ascii=False)}")
        print(f"hard_requirements: {json.dumps(row.get('hard_requirements'), ensure_ascii=False)}")
        print(f"quality_rubric: {json.dumps(row.get('quality_rubric'), ensure_ascii=False)}")
        print(f"derivation_summary: {json.dumps(row.get('derivation_summary'), ensure_ascii=False)}")
        print(f"final_issues: {json.dumps(row.get('final_issues'), ensure_ascii=False)}")
        print()
        print("Conversation trace:")
        for msg_index, message in enumerate(row.get("conversation_trace", []), start=1):
            if not should_show_message(message, args.show_system, args.show_all_messages):
                continue
            content = truncate(message.get("content") or "", args.max_chars)
            print(f"  [{msg_index}] {message.get('role')} / {message.get('kind')}:")
            print(f"  {content}")
        if row.get("steps"):
            print()
            print("Episode steps:")
            for step in row["steps"]:
                print(f"  turn {step.get('turn_index')}: state_changed={step.get('state_changed')}")
                print(f"    action_signature: {truncate(step.get('action_signature'), args.max_chars)}")
                if step.get("issues"):
                    print(f"    issues: {truncate(step.get('issues'), args.max_chars)}")
                if step.get("executed_tools"):
                    print(f"    executed_tools: {truncate(step.get('executed_tools'), args.max_chars)}")
                if step.get("tool_feedback"):
                    print(f"    tool_feedback: {truncate(step.get('tool_feedback'), args.max_chars)}")
        if row.get("judge_trace"):
            print()
            print("Judge trace:")
            for item in row["judge_trace"]:
                print(f"  turn {item.get('turn_index')}: passed={item.get('passed')} hard_failure={item.get('hard_failure')}")
                if item.get("feedback_to_model"):
                    print(f"    feedback_to_model: {truncate(item.get('feedback_to_model'), args.max_chars)}")
                if item.get("feedback_for_trace"):
                    print(f"    feedback_for_trace: {truncate(item.get('feedback_for_trace'), args.max_chars)}")
        print()


def main() -> None:
    args = parse_args()
    rows = filter_rows(load_rows(Path(args.input)), args)
    if args.json_output:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print_text(rows, args)


if __name__ == "__main__":
    main()
