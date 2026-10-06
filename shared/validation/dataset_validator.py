#!/usr/bin/env python3
"""
Validator for synthetic tool-calling datasets in ChatML format.

Validates JSONL files with the following structure:
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "label": true  // optional (true = desirable, false = undesirable)
}

Checks are config-first:
- A tool call whose name (or argument set) matches a wrapper in the tool-call
  format registry (SynthChat/config/tool_call_formats.yaml by default) must
  carry that format's required argument fields, each satisfying its configured
  property schema. Fields the format lists under ``prompt_bound_fields`` must
  hold a value the system prompt states. Calls that match no configured wrapper
  are direct (wrapper-less) tool calls.
- Every call is checked against its entry in the tool schema catalog
  (Tools/tool_schemas.json in the engine by default): required parameters and
  undeclared parameters.

Usage:
    python3 -m shared.validation.dataset_validator Datasets/your_dataset.jsonl
    python3 -m shared.validation.dataset_validator data.jsonl \\
        --tool-call-formats host/tool_call_formats.yaml --tool-schemas host/tool_schemas.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Iterable, List, Tuple, Dict, Any, Mapping, Optional, Sequence

from jsonschema import Draft202012Validator

from SynthChat.config.format_resolver import load_tool_call_formats
from shared.utilities.paths import get_engine_root
from shared.validation.parsing.configured_formats import (
    build_wrapper_specs,
    get_configured_wrapper_specs,
    match_configured_wrapper,
)

# Labels are now boolean: true = desirable, false = undesirable
ALLOWED_ROLES = {"system", "user", "assistant"}

# Engine tool schema catalog, resolved from the engine root (not the CWD).
SCHEMAS_FILE = get_engine_root() / "Tools" / "tool_schemas.json"


def load_tool_schemas(path: Path = SCHEMAS_FILE) -> Dict[str, Dict[str, Any]]:
    """Load a tool schema catalog (tool name -> schema) from JSON."""
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load tool schemas: {e}", file=sys.stderr)
    return {}


@dataclass(frozen=True)
class ValidatorConfig:
    """The configuration a dataset is validated against."""

    wrapper_specs: Sequence[Dict[str, Any]]
    tool_schemas: Mapping[str, Dict[str, Any]]


@lru_cache(maxsize=1)
def default_validator_config() -> ValidatorConfig:
    """The engine's configured tool-call formats and tool schema catalog."""
    return ValidatorConfig(
        wrapper_specs=get_configured_wrapper_specs(),
        tool_schemas=load_tool_schemas(),
    )


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
        issue = ValidationIssue(level, message)
        # The wrapper format and the tool schema can both require a field;
        # report each finding once.
        if issue not in self.issues:
            self.issues.append(issue)

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
                    "Invalid wrapper field",
                    "does not match system prompt"  # Allow ID mismatches in undesirable
                ])
            ]
            return len(structural_errors) == 0
        # For desirable examples (label=True), all errors are failures
        return all(issue.level != "ERROR" for issue in self.issues)


def extract_system_prompt(conversations: list) -> Optional[str]:
    """Return the first system message's content, if any."""
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


def _prompt_values(system_prompt: str, sources: Sequence[Mapping[str, Any]]) -> List[str]:
    """Values the system prompt states for one prompt-bound field."""
    values: List[str] = []
    for source in sources:
        text = system_prompt
        tag = source.get("in_tag")
        if tag:
            match = re.search(rf"<{re.escape(tag)}(?:\s[^>]*)?>(.*?)</{re.escape(tag)}>", system_prompt, re.DOTALL)
            if not match:
                continue
            text = match.group(1)
        for value in re.findall(source["pattern"], text):
            if value not in values:
                values.append(value)
    return values


def validate_prompt_bound_fields(
    spec: Mapping[str, Any],
    args: Mapping[str, Any],
    system_prompt: Optional[str],
    report: ExampleReport,
    tool_call_num: int,
) -> None:
    """Check the wrapper's prompt-bound fields against the values the system prompt states."""
    if not system_prompt:
        return
    for field_name, sources in spec.get("prompt_bound_fields", {}).items():
        value = args.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        allowed = _prompt_values(system_prompt, sources or [])
        if allowed and value not in allowed:
            report.add(
                "ERROR",
                f"Tool call #{tool_call_num} ({spec['wrapper_name']}): {field_name} '{value}' "
                f"does not match system prompt (expected one of: {allowed})",
            )


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
        tool_calls = msg.get("tool_calls")  # OpenAI format

        if not role:
            report.add("ERROR", f"Message at index {idx} missing 'role' field")
        elif role not in ALLOWED_ROLES:
            report.add("ERROR", f"Message at index {idx} has invalid role '{role}' (must be system/user/assistant)")

        # For OpenAI format, assistant can have null content if tool_calls present
        if role == "assistant" and tool_calls is not None:
            # OpenAI format: content can be null when tool_calls is present
            if content is not None and not isinstance(content, str):
                report.add("ERROR", f"Message at index {idx} has invalid 'content' field (must be string or null)")
        elif not isinstance(content, str):
            report.add("ERROR", f"Message at index {idx} missing or invalid 'content' field (must be string)")


def extract_tool_calls(content: str) -> List[Tuple[str, dict]]:
    """Extract tool calls from ChatML format assistant content, returning (tool_name, arguments_dict) tuples."""
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


def extract_tool_calls_openai(tool_calls_array: list) -> List[Tuple[str, dict]]:
    """Extract tool calls from OpenAI format, returning (tool_name, arguments_dict) tuples."""
    entries: List[Tuple[str, dict]] = []
    for tool_call in tool_calls_array:
        if not isinstance(tool_call, dict):
            continue

        # OpenAI format has nested function object
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue

        tool_name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        # Arguments might be a string that needs parsing
        if isinstance(arguments_str, str):
            try:
                args = json.loads(arguments_str, strict=False)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        elif isinstance(arguments_str, dict):
            args = arguments_str
        else:
            raise ValueError(f"Invalid arguments format for tool {tool_name}")

        entries.append((tool_name, args))
    return entries


def extract_tool_calls_mistral(content: str) -> List[Tuple[str, dict]]:
    """Extract tool calls from Mistral [TOOL_CALLS] format, returning (tool_name, arguments_dict) tuples.

    Mistral format: [TOOL_CALLS] [{"name": "tool_name", "arguments": "{...}", "id": "..."}]
    """
    entries: List[Tuple[str, dict]] = []

    if "[TOOL_CALLS]" not in content:
        return entries

    # Split on marker and get the JSON part
    parts = content.split("[TOOL_CALLS]", 1)
    if len(parts) < 2:
        return entries

    json_part = parts[1].strip()
    if not json_part.startswith("["):
        return entries

    # Find the matching closing bracket for the array
    try:
        json_blob, _ = extract_json_block(json_part, 0)
        tool_calls_array = json.loads(json_blob, strict=False)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse Mistral tool calls JSON: {e}")

    if not isinstance(tool_calls_array, list):
        raise ValueError("Mistral tool calls must be an array")

    for tool_call in tool_calls_array:
        if not isinstance(tool_call, dict):
            continue

        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # Arguments might be a string that needs parsing
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments, strict=False)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse arguments JSON for tool {tool_name}")
        elif isinstance(arguments, dict):
            args = arguments
        else:
            raise ValueError(f"Invalid arguments format for tool {tool_name}")

        entries.append((tool_name, args))

    return entries


def validate_tool_against_schema(
    tool_name: str,
    args: dict,
    report: ExampleReport,
    tool_call_num: int,
    tool_schemas: Mapping[str, Dict[str, Any]],
) -> None:
    """Validate tool call arguments against the tool's catalog schema.

    The schema's ``required_params`` must be present and every argument must be
    declared in its ``parameters``.
    """
    if not tool_schemas:
        report.add("WARN", "Tool schemas not loaded - skipping schema validation")
        return

    if tool_name not in tool_schemas:
        report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): No schema found for this tool")
        return

    schema = tool_schemas[tool_name]
    declared_params = {p["name"] for p in schema.get("parameters", [])}

    for req_param in schema.get("required_params", []):
        if req_param not in args:
            report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Missing required parameter '{req_param}'")

    for arg_name in args.keys():
        if arg_name not in declared_params:
            report.add("WARN", f"Tool call #{tool_call_num} ({tool_name}): Unexpected parameter '{arg_name}' not in schema")


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


def validate_wrapper_arguments(
    tool_name: str,
    args: Any,
    report: ExampleReport,
    tool_call_num: int,
    wrapper_specs: Sequence[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> None:
    """Check a tool call's arguments against the configured wrapper it uses.

    The wrapper is resolved with the shared ``match_configured_wrapper`` helper
    (by function name, then by argument set) over ``wrapper_specs``. A call that
    matches no wrapper is a direct tool call and has no wrapper fields to check.
    """
    if not isinstance(args, dict):
        report.add("ERROR", f"Tool call #{tool_call_num} ({tool_name}): Arguments must be a JSON object")
        return

    spec = match_configured_wrapper(args, function_name=tool_name, specs=wrapper_specs)
    if spec is None:
        return

    wrapper_name = spec["wrapper_name"]
    for field_name in spec["required_fields"]:
        if field_name not in args:
            report.add(
                "ERROR",
                f"Tool call #{tool_call_num} ({wrapper_name}): Missing required parameter '{field_name}'",
            )

    properties = {name: prop for name, prop in spec["properties"].items() if isinstance(prop, dict)}
    property_validator = Draft202012Validator({"type": "object", "properties": properties})
    for error in sorted(property_validator.iter_errors(args), key=lambda e: [str(part) for part in e.path]):
        field_path = ".".join(str(part) for part in error.path)
        report.add(
            "ERROR",
            f"Tool call #{tool_call_num} ({wrapper_name}): Invalid wrapper field '{field_path}' - {error.message}",
        )

    validate_prompt_bound_fields(spec, args, system_prompt, report, tool_call_num)


def validate_assistant_content(
    content: str,
    report: ExampleReport,
    system_prompt: Optional[str] = None,
    config: Optional[ValidatorConfig] = None,
) -> None:
    """Validate assistant message content, including tool calls if present.

    Supports multiple formats:
    - ChatML format: tool_call: toolName\\narguments: {...}
    - Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
    """
    config = config or default_validator_config()
    if not content.strip():
        report.add("ERROR", "Assistant content may not be empty")
        return

    tool_calls = []

    # Check for Mistral format first (more specific marker)
    if "[TOOL_CALLS]" in content:
        try:
            tool_calls = extract_tool_calls_mistral(content)
        except ValueError as e:
            report.add("ERROR", f"Invalid Mistral tool call format: {e}")
            return

        if not tool_calls:
            report.add("ERROR", "Assistant content has '[TOOL_CALLS]' marker but no valid tool calls found")
            return

    # Check for ChatML format
    elif "tool_call:" in content:
        try:
            tool_calls = extract_tool_calls(content)
        except ValueError as e:
            report.add("ERROR", f"Invalid ChatML tool call format: {e}")
            return

        if not tool_calls:
            report.add("ERROR", "Assistant content has 'tool_call:' marker but no valid tool calls found")
            return

    # Validate each tool call (if any were found)
    for idx, (tool_name, args) in enumerate(tool_calls, 1):
        if not tool_name:
            report.add("ERROR", f"Tool call #{idx} missing name")
        validate_wrapper_arguments(tool_name, args, report, idx, config.wrapper_specs, system_prompt)
        if isinstance(args, dict):
            validate_tool_against_schema(tool_name, args, report, idx, config.tool_schemas)


def validate_assistant_message_openai(
    msg: dict,
    report: ExampleReport,
    system_prompt: Optional[str] = None,
    config: Optional[ValidatorConfig] = None,
) -> None:
    """Validate OpenAI format assistant message with tool_calls array."""
    config = config or default_validator_config()
    tool_calls_array = msg.get("tool_calls")
    if not isinstance(tool_calls_array, list):
        report.add("ERROR", "tool_calls must be an array")
        return

    if len(tool_calls_array) == 0:
        report.add("ERROR", "tool_calls array must not be empty if present")
        return

    try:
        tool_calls = extract_tool_calls_openai(tool_calls_array)
    except ValueError as e:
        report.add("ERROR", str(e))
        return

    if not tool_calls:
        report.add("ERROR", "No valid tool calls found in tool_calls array")
        return

    # Validate each tool call
    for idx, (tool_name, args) in enumerate(tool_calls, 1):
        if not tool_name:
            report.add("ERROR", f"Tool call #{idx} missing name")
        validate_wrapper_arguments(tool_name, args, report, idx, config.wrapper_specs, system_prompt)
        if isinstance(args, dict):
            validate_tool_against_schema(tool_name, args, report, idx, config.tool_schemas)


def validate_example(
    idx: int,
    example: dict,
    config: Optional[ValidatorConfig] = None,
) -> ExampleReport:
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

    # The system prompt states the values of prompt-bound wrapper fields
    system_prompt = extract_system_prompt(conversations)

    # Validate assistant messages specifically
    for msg in conversations:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            # Check format: OpenAI (tool_calls array) or ChatML (content string)
            if "tool_calls" in msg:
                # OpenAI format
                validate_assistant_message_openai(msg, report, system_prompt, config)
            else:
                # ChatML format
                content = msg.get("content", "")
                validate_assistant_content(content, report, system_prompt, config)

    # Label is optional in ChatML format
    # Labels should now be boolean: true = desirable, false = undesirable
    if label is not None:
        if not isinstance(label, bool):
            report.add("ERROR", f"Label must be a boolean (true/false) if present, got: {type(label).__name__}")

    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic tool-calling JSONL files")
    parser.add_argument("path", type=Path, help="Path to JSONL file")
    parser.add_argument(
        "--tool-call-formats",
        type=Path,
        default=None,
        help="Tool-call format registry YAML (default: SynthChat/config/tool_call_formats.yaml)",
    )
    parser.add_argument(
        "--tool-schemas",
        type=Path,
        default=None,
        help="Tool schema catalog JSON (default: the engine's Tools/tool_schemas.json)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    default = default_validator_config()
    wrapper_specs = default.wrapper_specs
    if args.tool_call_formats is not None:
        if not args.tool_call_formats.is_file():
            sys.exit(f"Tool-call format registry not found: {args.tool_call_formats}")
        wrapper_specs = build_wrapper_specs(load_tool_call_formats(str(args.tool_call_formats)))
    tool_schemas = default.tool_schemas
    if args.tool_schemas is not None:
        if not args.tool_schemas.is_file():
            sys.exit(f"Tool schema catalog not found: {args.tool_schemas}")
        tool_schemas = load_tool_schemas(args.tool_schemas)
    config = ValidatorConfig(wrapper_specs=wrapper_specs, tool_schemas=tool_schemas)

    reports: List[ExampleReport] = []
    try:
        for idx, payload in load_jsonl(args.path):
            reports.append(validate_example(idx, payload, config))
    except ValueError as exc:
        sys.exit(str(exc))

    # Only count failures from label=true examples (or no label)
    # label=false examples are intentionally incorrect and should be ignored
    invalid = [r for r in reports if not r.is_valid and r.label is not False]

    for report in reports:
        if report.issues and report.label is not False:
            print(f"Example line {report.index}:")
            for issue in report.issues:
                print(f"  [{issue.level}] {issue.message}")
            print()

    # Print schema validation status
    if config.tool_schemas:
        print(f"✓ Schema validation enabled ({len(config.tool_schemas)} tool schemas loaded)\n", file=sys.stderr)
    else:
        print("⚠ Schema validation disabled (no tool schemas loaded)\n", file=sys.stderr)

    # Count label=false examples separately for informational purposes
    label_false_count = len([r for r in reports if r.label is False])
    summary = f"Validated {len(reports)} example(s): {len(invalid)} failed (ignoring {label_false_count} label=false examples)."
    if invalid:
        sys.exit(summary)
    print(summary)


if __name__ == "__main__":
    main()
