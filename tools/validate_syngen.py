#!/usr/bin/env python3
"""
Validator for synthetic Claudesidian tool-calling datasets in ChatML format.

Validates JSONL files with the following structure:
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "label": true  // optional (true = desirable, false = undesirable)
}

Usage:
    python tools/validate_syngen.py Synthetic\\ Conversations/syngen_toolset_v1.0.0.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

SESSION_ID_RE = re.compile(r"^session_\d{13}_[a-z0-9]{9}$")
WORKSPACE_ID_RE = re.compile(r"^ws_\d{13}_[a-z0-9]{9}$")
# Labels are now boolean: true = desirable, false = undesirable
ALLOWED_ROLES = {"system", "user", "assistant"}
MIN_TOOL_CALLS = 2

# Load tool schemas
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}
SCHEMAS_FILE = Path(__file__).parent / "tool_schemas.json"

def load_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Load tool schemas from JSON file."""
    if SCHEMAS_FILE.exists():
        try:
            with open(SCHEMAS_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load tool schemas: {e}", file=sys.stderr)
    return {}

TOOL_SCHEMAS = load_tool_schemas()


@dataclass
class ValidationIssue:
    level: str  # "ERROR" or "WARN"
    message: str


@dataclass
class ExampleReport:
    index: int
    issues: List[ValidationIssue] = field(default_factory=list)
    label: Optional[bool] = None

    def add(self, level: str, message: str) -> None:
        self.issues.append(ValidationIssue(level, message))

    @property
    def is_valid(self) -> bool:
        # For undesirable examples (label=False), we expect them to have errors (they demonstrate bad behavior)
        # So we only fail validation if there are structural issues, not tool parameter issues
        if self.label is False:  # Explicitly check for False
            # Allow tool parameter errors and schema mismatches in undesirable examples
            structural_errors = [
                issue for issue in self.issues
                if issue.level == "ERROR" and not any(x in issue.message for x in [
                    "Missing required parameter",
                    "Unexpected parameter",
                    "Invalid parameter value",
                    "Missing required 'context'",  # Allow missing context in undesirable
                    "Context object must be the first field"  # Allow context ordering issues
                ])
            ]
            return len(structural_errors) == 0
        # For desirable examples (label=True), all errors are failures
        return all(issue.level != "ERROR" for issue in self.issues)


def load_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {idx}: invalid JSON - {exc}") from exc
            yield idx, payload


def validate_conversations_array(conversations: list, report: ExampleReport) -> None:
    if not isinstance(conversations, list):
        report.add("ERROR", "conversations must be an array")
        return
    if len(conversations) == 0:
        report.add("ERROR", "conversations array must not be empty")
        return

    # Check for required roles
    roles = [msg.get("role") for msg in conversations if isinstance(msg, dict)]
    # System message is no longer required (new format)
    if "user" not in roles:
        report.add("ERROR", "conversations must include at least one 'user' role message")

    # Validate each message
    for idx, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            report.add("ERROR", f"Message at index {idx} must be an object")
            continue

        role = msg.get("role")
        content = msg.get("content")

        if not role:
            report.add("ERROR", f"Message at index {idx} missing 'role' field")
        elif role not in ALLOWED_ROLES:
            report.add("ERROR", f"Message at index {idx} has invalid role '{role}' (must be system/user/assistant)")

        if not isinstance(content, str):
            report.add("ERROR", f"Message at index {idx} missing or invalid 'content' field (must be string)")


def extract_tool_calls(content: str) -> List[Tuple[str, dict]]:
    """Extract tool calls from assistant content, returning (tool_name, arguments_dict) tuples."""
    entries: List[Tuple[str, dict]] = []
    marker = "tool_call:"
    pos = 0
    while True:
        idx = content.find(marker, pos)
        if idx == -1:
            break
        start = idx + len(marker)
        rest = content[start:]
        parts = rest.split("arguments:", 1)
        if len(parts) != 2:
            break
        tool_name = parts[0].strip().splitlines()[0].strip()
        json_start = content.index("{", content.index("arguments:", idx))
        json_blob, end_index = extract_json_block(content, json_start)
        try:
            args = json.loads(json_blob, strict=False)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        entries.append((tool_name, args))
        pos = end_index
    return entries


def validate_tool_call_structure(content: str, report: ExampleReport) -> None:
    """Validate the structure and formatting of tool calls in assistant content."""
    tool_call_positions = []
    pos = 0

    # Find all tool_call markers
    while True:
        idx = content.find("tool_call:", pos)
        if idx == -1:
            break
        tool_call_positions.append(idx)
        pos = idx + 1

    if not tool_call_positions:
        return  # No tool calls to validate

    for call_idx, call_pos in enumerate(tool_call_positions):
        tool_call_num = call_idx + 1

        # Extract the section for this tool call
        next_pos = tool_call_positions[call_idx + 1] if call_idx + 1 < len(tool_call_positions) else len(content)
        section = content[call_pos:next_pos]

        # 1. Validate tool_call line format
        first_line_end = section.find("\n")
        if first_line_end == -1:
            report.add("ERROR", f"Tool call #{tool_call_num}: Missing newline after tool_call:")
            continue

        first_line = section[:first_line_end]
        if not first_line.startswith("tool_call: "):
            report.add("ERROR", f"Tool call #{tool_call_num}: Must have space after 'tool_call:'")

        tool_name = first_line.replace("tool_call:", "").strip()
        if not tool_name:
            report.add("ERROR", f"Tool call #{tool_call_num}: Missing tool name after 'tool_call:'")
        elif "_" not in tool_name:
            report.add("WARN", f"Tool call #{tool_call_num}: Tool name '{tool_name}' doesn't follow manager_mode convention")

        # 2. Validate arguments presence and format
        if "arguments:" not in section:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing 'arguments:' line")
            continue

        args_idx = section.find("arguments:")
        args_line_start = section.rfind("\n", 0, args_idx)
        args_line = section[args_line_start:args_idx + len("arguments:")].strip()

        if not args_line.startswith("arguments:"):
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'arguments:' must be at start of line")

        # 3. Validate arguments JSON structure
        try:
            json_start = section.index("{", args_idx)
            json_blob, _ = extract_json_block(section, json_start)
            args = json.loads(json_blob, strict=False)

            # Validate it's a dict
            if not isinstance(args, dict):
                report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Arguments must be a JSON object, not {type(args).__name__}")
        except ValueError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Invalid JSON in arguments - {e}")
        except json.JSONDecodeError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Failed to parse arguments JSON - {e}")

        # 4. Validate Result presence and format
        if "Result:" not in section:
            report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): Missing 'Result:' response")
            continue

        result_idx = section.find("Result:")

        # Check that Result comes after arguments
        if result_idx < args_idx:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'Result:' appears before 'arguments:'")

        # 5. Validate Result JSON structure
        try:
            result_json_start = section.index("{", result_idx)
            result_json_blob, _ = extract_json_block(section, result_json_start)
            result = json.loads(result_json_blob, strict=False)

            # Validate it's a dict
            if not isinstance(result, dict):
                report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Result must be a JSON object, not {type(result).__name__}")
        except ValueError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Invalid JSON in Result - {e}")
        except json.JSONDecodeError as e:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Failed to parse Result JSON - {e}")


def validate_tool_against_schema(tool_name: str, args: dict, report: ExampleReport, tool_call_num: int) -> None:
    """Validate tool call arguments against the tool's schema."""
    if not TOOL_SCHEMAS:
        report.add("WARN", "Tool schemas not loaded - skipping schema validation")
        return

    if tool_name not in TOOL_SCHEMAS:
        report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): No schema found for this tool")
        return

    schema = TOOL_SCHEMAS[tool_name]
    required_params = schema.get('required_params', [])
    all_params = {p['name']: p for p in schema.get('parameters', [])}

    # Special handling for get_tools meta-tool
    if tool_name == 'get_tools':
        # Validate 'managers' parameter is present and is an array
        if 'managers' not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Missing required parameter 'managers'")
        elif not isinstance(args.get('managers'), list):
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Parameter 'managers' must be an array")
        elif len(args.get('managers', [])) == 0:
            report.add("WARN", f"Tool call #{tool_call_num} (get_tools): Parameter 'managers' is an empty array")

        # Validate context is present (required for all tools)
        if 'context' not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} (get_tools): Missing required parameter 'context'")

        # Skip further validation for get_tools (it's a meta-tool with special handling)
        return

    # Check required parameters are present
    for req_param in required_params:
        if req_param not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing required parameter '{req_param}'")

    # Check for unexpected parameters (skip 'context' as it's standard across all tools via CommonParams)
    for arg_name in args.keys():
        if arg_name not in all_params and arg_name not in ['context', 'workspaceContext']:
            report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): Unexpected parameter '{arg_name}' not in schema")

    # Validate context structure if present
    if 'context' in args and 'context_schema' in schema:
        context = args['context']
        if not isinstance(context, dict):
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): 'context' must be an object")
        else:
            context_schema = schema['context_schema']
            if context_schema and 'fields' in context_schema:
                required_context_fields = [f['name'] for f in context_schema['fields'] if not f.get('optional', False)]
                for field in required_context_fields:
                    if field not in context:
                        report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing required context field '{field}'")


def extract_json_block(text: str, start_index: int) -> Tuple[str, int]:
    stack = []
    in_string = False
    escape = False
    i = start_index
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack.pop():
                    raise ValueError("Unbalanced JSON block")
                if not stack:
                    return text[start_index : i + 1], i + 1
        i += 1
    raise ValueError("Unterminated JSON block")


def validate_context(args: dict, report: ExampleReport) -> None:
    if not isinstance(args, dict) or not args:
        report.add("ERROR", "Arguments must be a JSON object")
        return

    # ALL tools require context from CommonParams - it must be present and first
    if "context" not in args:
        report.add("ERROR", "Missing required 'context' field in arguments (all tools require context from CommonParams)")
        return

    keys = list(args.keys())
    if keys[0] != "context":
        report.add("ERROR", "Context object must be the first field in arguments")
        return
    ctx = args.get("context")
    required_fields = [
        "sessionId",
        "workspaceId",
        "sessionDescription",
        "sessionMemory",
        "toolContext",
        "primaryGoal",
        "subgoal",
    ]
    if not isinstance(ctx, dict):
        report.add("ERROR", "context must be an object")
        return
    for field in required_fields:
        if field not in ctx:
            report.add("ERROR", f"context missing field '{field}'")
    sess_id = ctx.get("sessionId")
    if isinstance(sess_id, str) and not SESSION_ID_RE.match(sess_id):
        report.add("ERROR", f"sessionId '{sess_id}' does not match generator format")
    ws_id = ctx.get("workspaceId")
    if isinstance(ws_id, str) and not WORKSPACE_ID_RE.match(ws_id):
        report.add("ERROR", f"workspaceId '{ws_id}' does not match generator format")


def validate_assistant_content(content: str, report: ExampleReport) -> None:
    """Validate assistant message content, including tool calls if present."""
    if not content.strip():
        report.add("ERROR", "Assistant content may not be empty")
        return

    # Check if this message contains tool calls
    if "tool_call:" in content:
        tool_calls = extract_tool_calls(content)
        if not tool_calls:
            report.add("ERROR", "Assistant content has 'tool_call:' marker but no valid tool calls found")
            return

        # Validate each tool call
        for idx, (tool_name, args) in enumerate(tool_calls, 1):
            if not tool_name:
                report.add("ERROR", "Tool call missing name")
            validate_context(args, report)
            # Validate against actual tool schema
            validate_tool_against_schema(tool_name, args, report, idx)

        # Check that tool results are present if tool calls exist
        if "Result:" not in content:
            report.add("WARN", "Tool calls present but no 'Result:' markers found")


def validate_example(idx: int, example: dict) -> ExampleReport:
    label = example.get("label")
    report = ExampleReport(index=idx, label=label)
    conversations = example.get("conversations")

    # Validate conversations field
    if conversations is None:
        report.add("ERROR", "Missing 'conversations' field")
        return report
    if not isinstance(conversations, list):
        report.add("ERROR", "'conversations' must be an array")
        return report

    # Validate conversations array structure
    validate_conversations_array(conversations, report)

    # Validate assistant messages specifically
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            validate_assistant_content(content, report)

    # Label is optional in ChatML format
    # Labels should now be boolean: true = desirable, false = undesirable
    if label is not None:
        if not isinstance(label, bool):
            report.add("ERROR", f"Label must be a boolean (true/false) if present, got: {type(label).__name__}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Claudesidian synthetic data JSONL files")
    parser.add_argument("path", type=Path, help="Path to JSONL file")
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    reports: List[ExampleReport] = []
    try:
        for idx, payload in load_jsonl(args.path):
            reports.append(validate_example(idx, payload))
    except ValueError as exc:
        sys.exit(str(exc))

    invalid = [r for r in reports if not r.is_valid]
    for report in reports:
        if report.issues:
            print(f"Example line {report.index}:")
            for issue in report.issues:
                print(f"  [{issue.level}] {issue.message}")
            print()

    # Print schema validation status
    if TOOL_SCHEMAS:
        print(f"✓ Schema validation enabled ({len(TOOL_SCHEMAS)} tool schemas loaded)\n", file=sys.stderr)
    else:
        print(f"⚠ Schema validation disabled (tool_schemas.json not found)\n", file=sys.stderr)

    summary = f"Validated {len(reports)} example(s): {len(invalid)} failed."
    if invalid:
        sys.exit(summary)
    print(summary)


if __name__ == "__main__":
    main()
