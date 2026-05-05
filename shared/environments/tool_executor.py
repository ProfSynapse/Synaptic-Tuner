"""Execute parsed tool calls against an environment runtime.

This module is intentionally schema-driven. It avoids hardcoded tool-name
switches and instead:

1. Loads tool definitions (agent + tool name + params) from config
2. Infers primitive filesystem actions from tool metadata + argument shape
3. Executes those primitives against the runtime
"""

from __future__ import annotations

import json
import re
import shlex
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared.validation.parsing.response_parser import parse_response
from shared.validation.parsing.configured_formats import match_configured_wrapper

from .base import EnvironmentRuntime
from .types import EnvironmentIssue, ExecutedToolCall

SUPPORTED_ACTIONS = {
    "mkdir",
    "append",
    "write",
    "replace",
    "read",
    "list",
    "move",
    "copy",
    "delete",
    "search",
    "simulate",
}

CLI_PARSE_ERRORS_KEY = "__cli_parse_errors__"


def execute_response_tool_calls(
    runtime: EnvironmentRuntime,
    response: Any,
    allowed_tools: Optional[Iterable[str]] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    action_hints: Optional[Dict[str, str]] = None,
    strict_schema: bool = False,
    invalid_cli_patterns: Optional[List[Dict[str, Any]]] = None,
    verb_rules: Optional[Dict[str, List[str]]] = None,
    key_hints: Optional[Dict[str, List[str]]] = None,
    mock_tool_outputs: Optional[List[Dict[str, Any]]] = None,
    default_action: str = "simulate",
) -> Tuple[List[ExecutedToolCall], List[EnvironmentIssue]]:
    """Parse a model response and execute supported tool calls."""
    parsed = parse_response(response)
    allowed = {str(tool).strip() for tool in (allowed_tools or []) if str(tool).strip()}
    allowed = _expand_allowed_tool_identifiers(allowed, tool_schema)
    enforce_allowlist = bool(allowed)
    schema_index = _build_tool_index(tool_schema)
    hints = _normalize_action_hints(action_hints or {})
    normalized_verb_rules = _normalize_rule_dict_lower(verb_rules or {})
    normalized_key_hints = _normalize_rule_dict_preserve(key_hints or {})
    normalized_default_action = _normalize_action(default_action, fallback="simulate")

    executions: List[ExecutedToolCall] = []
    issues: List[EnvironmentIssue] = []

    expanded_calls = _expand_wrapper_calls(parsed.tool_calls, invalid_cli_patterns=invalid_cli_patterns)

    for call in expanded_calls:
        name = call.name
        args = call.arguments if isinstance(call.arguments, dict) else {}
        record = ExecutedToolCall(name=name, arguments=args)
        schema_entry = schema_index.get(name)
        cli_parse_errors = []
        if isinstance(args, dict):
            raw_cli_errors = args.pop(CLI_PARSE_ERRORS_KEY, None)
            if isinstance(raw_cli_errors, list):
                cli_parse_errors = [str(item) for item in raw_cli_errors if str(item).strip()]

        if cli_parse_errors:
            record.status = "error"
            record.error = "; ".join(cli_parse_errors)
            record.recoverable = True
            issues.append(
                EnvironmentIssue(
                    "error",
                    f"Tool '{name}' has invalid CLI arguments: {'; '.join(cli_parse_errors)}",
                    code="invalid_cli_arguments",
                    recoverable=True,
                )
            )
            executions.append(record)
            continue

        if strict_schema and schema_index and schema_entry is None:
            record.status = "error"
            record.error = "Tool not present in schema"
            record.recoverable = True
            issues.append(
                EnvironmentIssue(
                    "error",
                    f"Tool '{name}' is not defined in configured tool schema",
                    code="tool_not_in_schema",
                    recoverable=True,
                )
            )
            executions.append(record)
            continue

        missing_required = _missing_required_args(args, schema_entry)
        if missing_required:
            record.status = "error"
            record.error = f"Missing required args: {', '.join(missing_required)}"
            record.recoverable = True
            issues.append(
                EnvironmentIssue(
                    "error",
                    f"Tool '{name}' missing required args: {', '.join(missing_required)}",
                    code="missing_required_args",
                    recoverable=True,
                )
            )
            executions.append(record)
            continue

        if enforce_allowlist and name not in allowed:
            record.status = "blocked"
            record.error = "Tool not in allowlist"
            record.recoverable = True
            issues.append(
                EnvironmentIssue(
                    "error",
                    f"Tool '{name}' not in environment allowlist",
                    code="tool_not_allowed",
                    recoverable=True,
                )
            )
            executions.append(record)
            continue

        mocked = _find_mock_tool_output(
            tool_name=name,
            args=args,
            schema_entry=schema_entry,
            mock_tool_outputs=mock_tool_outputs or [],
        )
        if mocked is not None:
            status = str(mocked.get("status") or "ok").strip().lower() or "ok"
            record.status = status
            record.output = _stringify_mock_value(mocked.get("output")) if "output" in mocked else None
            record.error = _stringify_mock_value(mocked.get("error")) if "error" in mocked else None
            record.recoverable = bool(mocked.get("recoverable", status != "ok"))
            if status != "ok":
                issues.append(
                    EnvironmentIssue(
                        "error",
                        record.error or f"Mocked tool '{name}' returned status '{status}'",
                        code="mock_tool_output_error",
                        recoverable=record.recoverable,
                    )
                )
            executions.append(record)
            continue

        try:
            action = _resolve_action(
                tool_name=name,
                args=args,
                schema_entry=schema_entry,
                action_hints=hints,
                verb_rules=normalized_verb_rules,
                key_hints=normalized_key_hints,
                default_action=normalized_default_action,
            )
            output = _execute_action(runtime, action, args, key_hints=normalized_key_hints)
            record.output = output
            record.status = "ok"
            record.recoverable = False
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
            record.recoverable = True
            issues.append(
                EnvironmentIssue(
                    "error",
                    f"Tool '{name}' failed: {exc}",
                    code="tool_execution_failed",
                    recoverable=True,
                )
            )

        executions.append(record)

    return executions, issues


def format_tool_results_message(
    executions: List[ExecutedToolCall],
    issues: List[EnvironmentIssue],
    format_name: str = "json",
    tool_schema: Optional[Dict[str, Any]] = None,
    tool_name_format: str = "executor",
) -> str:
    """Render executed tool results into a message for the next model turn.

    Model-facing feedback should describe what actually happened in the runtime,
    not prescribe the next tool. Internal issue codes remain available for
    analysis/reporting, but the conversation payload should stay close to real
    filesystem/tool output.
    """
    payload = {
        "tool_results": [
            {
                "name": _display_tool_name(tool.name, tool_schema, tool_name_format),
                "status": tool.status,
                **({"output": tool.output} if tool.output is not None else {}),
                **(
                    {
                        "error": _replace_tool_names_in_text(
                            tool.error,
                            tool_schema,
                            tool_name_format,
                        )
                    }
                    if tool.error is not None
                    else {}
                ),
            }
            for tool in executions
        ],
        "issues": [
            {
                "level": issue.level,
                "message": _replace_tool_names_in_text(
                    issue.message,
                    tool_schema,
                    tool_name_format,
                ),
            }
            for issue in issues
        ],
    }
    normalized = str(format_name or "json").strip().lower()
    if normalized == "json":
        return "Tool execution results:\n" f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    return (
        "Tool execution results:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def format_environment_payload_for_model(
    payload: Any,
    tool_schema: Optional[Dict[str, Any]] = None,
    tool_name_format: str = "executor",
) -> Any:
    """Return a model-facing copy with configured tool display names.

    Internal execution records continue to use executor identifiers. This
    formatter is only for prompt, feedback, and judge context surfaces.
    """
    normalized = str(tool_name_format or "executor").strip().lower()
    if normalized not in {"command", "cli", "cli_command"}:
        return payload

    if isinstance(payload, dict):
        rendered: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "name" and isinstance(value, str):
                rendered[key] = _display_tool_name(value, tool_schema, normalized)
            elif isinstance(value, str):
                rendered[key] = _replace_tool_names_in_text(value, tool_schema, normalized)
            else:
                rendered[key] = format_environment_payload_for_model(value, tool_schema, normalized)
        return rendered
    if isinstance(payload, list):
        return [format_environment_payload_for_model(item, tool_schema, normalized) for item in payload]
    if isinstance(payload, str):
        return _replace_tool_names_in_text(payload, tool_schema, normalized)
    return payload


def _looks_like_cli_wrapper_call(call) -> bool:
    args = call.arguments if isinstance(call.arguments, dict) else {}
    if not isinstance(args, dict):
        return False
    wrapper_spec = match_configured_wrapper(args, function_name=getattr(call, "name", None))
    if wrapper_spec is None:
        return False
    tool_value = args.get("tool")
    return isinstance(tool_value, str) and bool(tool_value.strip())


@lru_cache(maxsize=1)
def _load_cli_command_catalog() -> Dict[str, Tuple[str, List[Dict[str, Any]]]]:
    schema_path = Path(__file__).resolve().parents[2] / "tool-schemas.json"
    if not schema_path.exists():
        return {}

    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    catalog: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    for item in payload.get("tools", []):
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        tool = str(item.get("tool", "")).strip()
        command = str(item.get("command", "")).strip()
        if not agent or not tool or not command:
            continue
        catalog[command] = (f"{agent}_{tool}", item.get("arguments", []) or [])
    return catalog


def _split_cli_commands(tool_value: str) -> List[str]:
    commands: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    brace_depth = 0
    bracket_depth = 0

    for char in tool_value:
        current.append(char)
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "{":
            brace_depth += 1
            continue
        if char == "}":
            brace_depth = max(0, brace_depth - 1)
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            continue
        if char == "," and brace_depth == 0 and bracket_depth == 0:
            current.pop()
            segment = "".join(current).strip()
            if segment:
                commands.append(segment)
            current = []

    tail = "".join(current).strip()
    if tail:
        commands.append(tail)
    return commands


def _normalize_cli_whitespace(value: str) -> str:
    if not isinstance(value, str):
        return value
    quote_map = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    normalized_chars: List[str] = []
    for char in value:
        char = quote_map.get(char, char)
        if char.isspace() or unicodedata.category(char) == "Zs":
            normalized_chars.append(" ")
        else:
            normalized_chars.append(char)
    return "".join(normalized_chars)


def _parse_cli_value(raw_value: str, value_type: str) -> Any:
    lowered = (value_type or "").strip().lower()
    if lowered.startswith("array") or lowered == "object":
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            if lowered == "array<string>":
                parts = [part.strip() for part in raw_value.split(",") if part.strip()]
                if parts:
                    return parts
            return raw_value
    if lowered == "boolean":
        if raw_value.lower() in {"true", "1", "yes"}:
            return True
        if raw_value.lower() in {"false", "0", "no"}:
            return False
        return raw_value
    if lowered == "number":
        try:
            return int(raw_value) if "." not in raw_value else float(raw_value)
        except ValueError:
            return raw_value
    return raw_value


def _validate_cli_arg_value(value: Any, spec: Dict[str, Any]) -> Optional[str]:
    value_type = str(spec.get("type", "string") or "string").strip().lower()
    name = str(spec.get("name", "value") or "value")

    if value_type.startswith("array"):
        if not isinstance(value, list):
            return f"{name} must be valid JSON array"
        if value_type == "array<object>" and any(not isinstance(item, dict) for item in value):
            return f"{name} must be an array of objects"
        return None

    if value_type == "object" and not isinstance(value, dict):
        return f"{name} must be valid JSON object"

    if value_type == "number" and not isinstance(value, (int, float)):
        return f"{name} must be numeric"

    if value_type == "boolean" and not isinstance(value, bool):
        return f"{name} must be boolean"

    return None


def _expand_cli_wrapper_call(call, invalid_cli_patterns: Optional[List[Dict[str, Any]]] = None) -> List:
    args = call.arguments if isinstance(call.arguments, dict) else {}
    tool_value = args.get("tool")
    if not isinstance(tool_value, str) or not tool_value.strip():
        return [call]

    catalog = _load_cli_command_catalog()
    if not catalog:
        return [call]

    sorted_commands = sorted(catalog.keys(), key=lambda value: len(value.split()), reverse=True)
    expanded = []

    for raw_command_str in _split_cli_commands(tool_value):
        command_str = _normalize_cli_whitespace(raw_command_str)
        parse_errors: List[str] = _invalid_cli_pattern_errors(raw_command_str, invalid_cli_patterns)
        try:
            tokens = shlex.split(command_str)
        except ValueError as exc:
            parse_errors.append(f"CLI command could not be parsed: {exc}")
            tokens = []
        if not tokens:
            if parse_errors:
                expanded.append(
                    type(call)(
                        name=getattr(call, "name", "unknown"),
                        arguments={CLI_PARSE_ERRORS_KEY: parse_errors},
                        raw=call.raw,
                    )
                )
            continue

        matched_command = None
        matched_spec = None
        matched_prefix_len = 0
        for command in sorted_commands:
            command_tokens = command.split()
            if tokens[: len(command_tokens)] == command_tokens:
                matched_command = command
                matched_spec = catalog[command]
                matched_prefix_len = len(command_tokens)
                break

        if matched_command is None or matched_spec is None:
            if parse_errors:
                expanded.append(
                    type(call)(
                        name=getattr(call, "name", "unknown"),
                        arguments={CLI_PARSE_ERRORS_KEY: parse_errors},
                        raw=call.raw,
                    )
                )
                continue
            return [call]

        tool_name, argument_specs = matched_spec
        remaining = tokens[matched_prefix_len:]
        parsed_args: Dict[str, Any] = {}
        positional_specs = [arg for arg in argument_specs if arg.get("positional")]
        flag_specs = {
            str(arg.get("flag")).strip(): arg
            for arg in argument_specs
            if str(arg.get("flag", "")).strip()
        }

        positional_index = 0
        i = 0
        while i < len(remaining):
            token = remaining[i]
            if token.startswith("--"):
                flag_token = token
                inline_value = None
                if "=" in token:
                    flag_token, inline_value = token.split("=", 1)
                arg_spec = flag_specs.get(flag_token)
                if not arg_spec:
                    i += 1
                    continue
                value_type = arg_spec.get("type", "string")
                if value_type == "boolean":
                    parsed_args[arg_spec["name"]] = True
                    i += 1
                    continue
                if inline_value is not None:
                    parsed_value = _parse_cli_value(inline_value, value_type)
                    parsed_args[arg_spec["name"]] = parsed_value
                    validation_error = _validate_cli_arg_value(parsed_value, arg_spec)
                    if validation_error:
                        parse_errors.append(validation_error)
                    i += 1
                    continue
                if i + 1 < len(remaining):
                    parsed_value = _parse_cli_value(remaining[i + 1], value_type)
                    parsed_args[arg_spec["name"]] = parsed_value
                    validation_error = _validate_cli_arg_value(parsed_value, arg_spec)
                    if validation_error:
                        parse_errors.append(validation_error)
                    i += 2
                    continue
                i += 1
                continue

            if positional_index < len(positional_specs):
                arg_spec = positional_specs[positional_index]
                parsed_value = _parse_cli_value(token, arg_spec.get("type", "string"))
                parsed_args[arg_spec["name"]] = parsed_value
                validation_error = _validate_cli_arg_value(parsed_value, arg_spec)
                if validation_error:
                    parse_errors.append(validation_error)
                positional_index += 1
            i += 1

        if parse_errors:
            parsed_args[CLI_PARSE_ERRORS_KEY] = parse_errors
        expanded.append(type(call)(name=tool_name, arguments=parsed_args, raw=call.raw))

    return expanded or [call]


def _invalid_cli_pattern_errors(
    value: str,
    invalid_cli_patterns: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not isinstance(value, str) or not invalid_cli_patterns:
        return []

    errors: List[str] = []
    for index, rule in enumerate(invalid_cli_patterns, start=1):
        if isinstance(rule, str):
            pattern = rule
            name = f"rule_{index}"
            message = f"CLI command matched forbidden pattern {name}"
        elif isinstance(rule, dict):
            pattern = str(rule.get("pattern") or "")
            name = str(rule.get("name") or f"rule_{index}")
            message = str(rule.get("message") or f"CLI command matched forbidden pattern {name}")
        else:
            continue
        if not pattern:
            continue
        try:
            if re.search(pattern, value):
                errors.append(message)
        except re.error as exc:
            errors.append(f"Invalid CLI pattern config {name}: {exc}")
    return errors


def _expand_wrapper_calls(parsed_calls, invalid_cli_patterns: Optional[List[Dict[str, Any]]] = None) -> List:
    """Expand delegated wrapper calls into concrete tool calls."""
    expanded = []
    for call in parsed_calls:
        if _looks_like_cli_wrapper_call(call):
            expanded.extend(_expand_cli_wrapper_call(call, invalid_cli_patterns=invalid_cli_patterns))
            continue
        expanded.append(call)

    return expanded


def _find_mock_tool_output(
    *,
    tool_name: str,
    args: Dict[str, Any],
    schema_entry: Optional[Dict[str, Any]],
    mock_tool_outputs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a configured mock response for this call, if one matches.

    The rules are deliberately declarative: a scenario/generated environment
    chooses which tool identifiers and argument values should receive mocked
    output. The executor only transports that configured response.
    """
    if not isinstance(mock_tool_outputs, list):
        return None

    identifiers = _tool_identifiers_for_mock_match(tool_name, schema_entry)
    for item in mock_tool_outputs:
        if not isinstance(item, dict):
            continue
        rule_tools = _mock_rule_tools(item)
        if rule_tools and identifiers.isdisjoint(rule_tools):
            continue
        if not _mock_arguments_match(args, item.get("match", item.get("arguments"))):
            continue
        return item
    return None


def _tool_identifiers_for_mock_match(
    tool_name: str,
    schema_entry: Optional[Dict[str, Any]],
) -> set[str]:
    identifiers = {str(tool_name or "").strip()}
    if "_" in tool_name:
        identifiers.add(tool_name.split("_", 1)[1])
    if isinstance(schema_entry, dict):
        command = str(schema_entry.get("command") or "").strip()
        configured_name = str(schema_entry.get("name") or "").strip()
        if command:
            identifiers.add(command)
        if configured_name:
            identifiers.add(configured_name)
    return {item for item in identifiers if item}


def _mock_rule_tools(item: Dict[str, Any]) -> set[str]:
    values: List[Any] = []
    for key in ("tool", "name", "command"):
        if key in item:
            values.append(item[key])
    if isinstance(item.get("tools"), list):
        values.extend(item["tools"])
    return {str(value).strip() for value in values if str(value).strip()}


def _mock_arguments_match(args: Dict[str, Any], expected: Any) -> bool:
    if expected in (None, {}, []):
        return True
    if not isinstance(expected, dict):
        return False
    for key, value in expected.items():
        if key not in args:
            return False
        if args[key] != value:
            return False
    return True


def _stringify_mock_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, indent=2)


def _build_tool_index(tool_schema: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tool_schema, dict):
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    for agent, tools in (tool_schema.get("tools") or {}).items():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                continue
            full_name = f"{agent}_{tool_name}"
            index[full_name] = tool
            # Keep direct key too if a model emits unqualified names.
            index.setdefault(tool_name, tool)
    return index


def _expand_allowed_tool_identifiers(
    allowed: Iterable[str],
    tool_schema: Optional[Dict[str, Any]],
) -> set[str]:
    expanded = {str(item).strip() for item in allowed if str(item).strip()}
    if not expanded or not isinstance(tool_schema, dict):
        return expanded

    for agent, tools in (tool_schema.get("tools") or {}).items():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "").strip()
            command = str(tool.get("command") or "").strip()
            if not tool_name or not command or command not in expanded:
                continue
            expanded.add(f"{agent}_{tool_name}")
            expanded.add(tool_name)
    return expanded


def _tool_command_display_map(tool_schema: Optional[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not isinstance(tool_schema, dict):
        return mapping

    for agent, tools in (tool_schema.get("tools") or {}).items():
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "").strip()
            command = str(tool.get("command") or "").strip()
            if not tool_name or not command:
                continue
            mapping[f"{agent}_{tool_name}"] = command
            mapping[tool_name] = command
    return mapping


def _display_tool_name(
    name: str,
    tool_schema: Optional[Dict[str, Any]],
    tool_name_format: str,
) -> str:
    normalized = str(tool_name_format or "executor").strip().lower()
    if normalized not in {"command", "cli", "cli_command"}:
        return name
    return _tool_command_display_map(tool_schema).get(name, name)


def _replace_tool_names_in_text(
    text: Optional[str],
    tool_schema: Optional[Dict[str, Any]],
    tool_name_format: str,
) -> Optional[str]:
    if text is None:
        return None
    normalized = str(tool_name_format or "executor").strip().lower()
    if normalized not in {"command", "cli", "cli_command"}:
        return text

    rendered = str(text)
    for internal_name, command in sorted(
        (
            (name, command)
            for name, command in _tool_command_display_map(tool_schema).items()
            if "_" in name
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        rendered = rendered.replace(internal_name, command)
    return rendered


def _normalize_action_hints(hints: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in (hints or {}).items():
        if not isinstance(value, str):
            continue
        action = _normalize_action(value, fallback="")
        if not action:
            continue
        name = str(key).strip()
        if name:
            normalized[name] = action
    return normalized


def _normalize_rule_dict_lower(rule_map: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, values in (rule_map or {}).items():
        if not isinstance(values, list):
            continue
        tokens = [str(value).strip().lower() for value in values if str(value).strip()]
        if tokens:
            normalized[str(key).strip()] = tokens
    return normalized


def _normalize_rule_dict_preserve(rule_map: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, values in (rule_map or {}).items():
        if not isinstance(values, list):
            continue
        tokens = [str(value).strip() for value in values if str(value).strip()]
        if tokens:
            normalized[str(key).strip()] = tokens
    return normalized


def _missing_required_args(args: Dict[str, Any], schema_entry: Optional[Dict[str, Any]]) -> List[str]:
    if not schema_entry:
        return []
    required = ((schema_entry.get("params") or {}).get("required") or [])
    missing = []
    for key in required:
        if key == "context":
            continue
        if key not in args:
            missing.append(str(key))
    return missing


def _resolve_action(
    tool_name: str,
    args: Dict[str, Any],
    schema_entry: Optional[Dict[str, Any]],
    action_hints: Dict[str, str],
    verb_rules: Dict[str, List[str]],
    key_hints: Dict[str, List[str]],
    default_action: str,
) -> str:
    hint = action_hints.get(tool_name)
    if hint is None:
        unqualified = tool_name.split("_", 1)[1] if "_" in tool_name else tool_name
        hint = action_hints.get(unqualified)
    if isinstance(hint, str) and hint.strip():
        return _normalize_action(hint, fallback=default_action)

    verb = tool_name.split("_", 1)[1] if "_" in tool_name else tool_name
    verb_lower = verb.lower()

    # 1) Configured verb inference rules.
    inferred_by_rules = _infer_action_from_verb_rules(verb_lower, verb_rules)
    if inferred_by_rules:
        if _action_args_look_compatible(inferred_by_rules, args, key_hints):
            return inferred_by_rules

    # 2) Generic argument-shape inference.
    inferred_by_args = _infer_action_from_args(args, key_hints)
    if inferred_by_args:
        return inferred_by_args

    # 3) Schema signature fallback (still config/schema-driven, no tool-name map).
    inferred_by_schema = _infer_action_from_schema_signature(schema_entry)
    if inferred_by_schema:
        return inferred_by_schema

    return default_action


def _infer_action_from_verb_rules(verb_lower: str, verb_rules: Dict[str, List[str]]) -> Optional[str]:
    for action, tokens in verb_rules.items():
        for token in tokens:
            if token and token in verb_lower:
                return _normalize_action(action, fallback="")
    return None


def _infer_action_from_args(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    has_source_path = _extract_source_path(args, key_hints) is not None
    has_dest_path = _extract_destination_path(args, key_hints) is not None
    has_content = _extract_content(args, key_hints) is not None
    has_old_content = _extract_old_content(args, key_hints) is not None
    has_new_content = _extract_new_content(args, key_hints) is not None
    has_query = _extract_query(args, key_hints) is not None

    if has_query:
        return "search"
    if has_source_path and has_dest_path:
        return "move"
    if has_source_path and has_old_content and has_new_content:
        return "replace"
    if has_source_path and has_content:
        return "write"
    return None


def _infer_action_from_schema_signature(schema_entry: Optional[Dict[str, Any]]) -> Optional[str]:
    if not schema_entry:
        return None

    params = schema_entry.get("params")
    if not isinstance(params, dict):
        return None

    required = set(str(k) for k in (params.get("required") or []))
    optional = set(str(k) for k in (params.get("optional") or []))
    keys = required | optional

    if "path" in required and {"oldContent", "newContent"}.issubset(keys):
        return "replace"
    if {"path", "content"}.issubset(required):
        return "write"
    if {"query"}.issubset(required) or "query" in keys:
        return "search"
    if "path" in required and any(key in keys for key in {"startLine", "endLine", "mode", "focus"}):
        return "read"
    return None


def _action_args_look_compatible(action: str, args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> bool:
    if action in {"simulate", "list"}:
        return True
    if action in {"mkdir", "read", "delete"}:
        return _extract_source_path(args, key_hints) is not None
    if action in {"write", "append"}:
        return _extract_source_path(args, key_hints) is not None
    if action == "replace":
        return (
            _extract_source_path(args, key_hints) is not None
            and _extract_old_content(args, key_hints) is not None
            and _extract_new_content(args, key_hints) is not None
        )
    if action in {"move", "copy"}:
        return (
            _extract_source_path(args, key_hints) is not None
            and _extract_destination_path(args, key_hints) is not None
        )
    if action == "search":
        return _extract_query(args, key_hints) is not None or _extract_source_path(args, key_hints) is not None
    return False


def _execute_action(
    runtime: EnvironmentRuntime,
    action: str,
    args: Dict[str, Any],
    key_hints: Dict[str, List[str]],
) -> Optional[str]:
    if action == "mkdir":
        runtime.mkdir(_require_path(_extract_source_path(args, key_hints), "path"))
        return "created"
    if action == "list":
        path = _extract_source_path(args, key_hints) or "."
        return json.dumps(runtime.list_dir(path))
    if action == "read":
        return runtime.read_text(_require_path(_extract_source_path(args, key_hints), "path"))
    if action == "write":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        content = _extract_content(args, key_hints)
        runtime.write_text(path, "" if content is None else content)
        return "written"
    if action == "replace":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        old_content = _extract_old_content(args, key_hints)
        new_content = _extract_new_content(args, key_hints)
        if old_content is None:
            raise ValueError("Missing required old-content-like argument 'oldContent'")
        if new_content is None:
            raise ValueError("Missing required new-content-like argument 'newContent'")
        existing = runtime.read_text(path)
        if old_content not in existing:
            raise ValueError("oldContent was not found in the target file")
        runtime.write_text(path, existing.replace(old_content, new_content, 1))
        return "replaced"
    if action == "append":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        existing = runtime.read_text(path) if runtime.exists(path) else ""
        appended = _extract_content(args, key_hints) or ""
        runtime.write_text(path, existing + appended)
        return "appended"
    if action == "move":
        runtime.move(
            _require_path(_extract_source_path(args, key_hints), "path"),
            _require_path(_extract_destination_path(args, key_hints), "newPath"),
            overwrite=bool(args.get("overwrite", False)),
        )
        return "moved"
    if action == "copy":
        runtime.copy(
            _require_path(_extract_source_path(args, key_hints), "path"),
            _require_path(_extract_destination_path(args, key_hints), "newPath"),
            overwrite=bool(args.get("overwrite", False)),
        )
        return "copied"
    if action == "delete":
        runtime.delete(
            _require_path(_extract_source_path(args, key_hints), "path"),
            recursive=bool(args.get("recursive", True)),
        )
        return "deleted"
    if action == "search":
        query = _extract_query(args, key_hints) or ""
        path = _extract_source_path(args, key_hints) or "."
        return json.dumps(runtime.search(query=query, path=path))
    if action == "simulate":
        return "simulated"

    raise ValueError(f"Unsupported inferred action: {action}")


def _extract_source_path(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("source_path", []):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inferred = _extract_value_by_key_predicate(
        args,
        lambda key: "path" in key and not any(token in key for token in ("new", "dest", "destination", "output")),
    )
    if inferred:
        return inferred
    return _scan_for_path_like_value(args)


def _extract_destination_path(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("destination_path", []):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_value_by_key_predicate(
        args,
        lambda key: any(token in key for token in ("newpath", "dest", "destination", "output", "targetpath", "to")),
    )


def _extract_content(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("content", []):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return _extract_value_by_key_predicate(
        args,
        lambda key: any(token in key for token in ("content", "text", "body", "value")),
    )


def _extract_old_content(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("old_content", []):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return _extract_value_by_key_predicate(
        args,
        lambda key: any(token in key for token in ("oldcontent", "old_content", "oldtext", "old_text")),
    )


def _extract_new_content(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("new_content", []):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return _extract_value_by_key_predicate(
        args,
        lambda key: any(
            token in key
            for token in ("newcontent", "new_content", "replacement", "newtext", "new_text")
        ),
    )


def _extract_query(args: Dict[str, Any], key_hints: Dict[str, List[str]]) -> Optional[str]:
    for key in key_hints.get("query", []):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _extract_value_by_key_predicate(
        args,
        lambda key: any(token in key for token in ("query", "search", "term", "pattern", "needle")),
    )


def _extract_value_by_key_predicate(args: Dict[str, Any], predicate) -> Optional[str]:
    for key, value in args.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        key_normalized = key.strip().lower()
        if not predicate(key_normalized):
            continue
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _scan_for_path_like_value(args: Dict[str, Any]) -> Optional[str]:
    for value in args.values():
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if "/" in candidate or candidate.endswith(".md") or candidate.endswith(".txt"):
            return candidate
    return None


def _require_path(path: Optional[str], field_name: str) -> str:
    if path is None:
        raise ValueError(f"Missing required path-like argument '{field_name}'")
    clean = path.strip()
    if not clean:
        raise ValueError(f"Invalid empty path-like argument '{field_name}'")
    return clean


def _normalize_action(action: Any, fallback: str) -> str:
    value = str(action or "").strip().lower()
    if value in SUPPORTED_ACTIONS:
        return value
    return fallback
