"""Execute parsed tool calls against an environment runtime.

This module is intentionally schema-driven. It avoids hardcoded tool-name
switches and instead:

1. Loads tool definitions (agent + tool name + params) from config
2. Infers primitive filesystem actions from tool metadata + argument shape
3. Executes those primitives against the runtime
"""

from __future__ import annotations

import json
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
    verb_rules: Optional[Dict[str, List[str]]] = None,
    key_hints: Optional[Dict[str, List[str]]] = None,
    default_action: str = "simulate",
    read_output: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ExecutedToolCall], List[EnvironmentIssue]]:
    """Parse a model response and execute supported tool calls."""
    parsed = parse_response(response)
    allowed = set(allowed_tools or [])
    enforce_allowlist = bool(allowed)
    schema_index = _build_tool_index(tool_schema)
    hints = _normalize_action_hints(action_hints or {})
    normalized_verb_rules = _normalize_rule_dict_lower(verb_rules or {})
    normalized_key_hints = _normalize_rule_dict_preserve(key_hints or {})
    normalized_default_action = _normalize_action(default_action, fallback="simulate")

    executions: List[ExecutedToolCall] = []
    issues: List[EnvironmentIssue] = []

    expanded_calls = _expand_wrapper_calls(parsed.tool_calls)

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
            output = _execute_action(
                runtime,
                action,
                args,
                key_hints=normalized_key_hints,
                read_output=read_output or {},
            )
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
                "name": tool.name,
                "status": tool.status,
                **({"output": tool.output} if tool.output is not None else {}),
                **({"error": tool.error} if tool.error is not None else {}),
            }
            for tool in executions
        ],
        "issues": [
            {
                "level": issue.level,
                "message": issue.message,
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
    schema_path = Path(__file__).resolve().parents[2] / "cli-first-tool-schemas.json"
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
    tool_value = _normalize_cli_whitespace(tool_value)
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


def _expand_cli_wrapper_call(call) -> List:
    args = call.arguments if isinstance(call.arguments, dict) else {}
    tool_value = args.get("tool")
    if not isinstance(tool_value, str) or not tool_value.strip():
        return [call]

    catalog = _load_cli_command_catalog()
    if not catalog:
        return [call]

    sorted_commands = sorted(catalog.keys(), key=lambda value: len(value.split()), reverse=True)
    expanded = []

    for command_str in _split_cli_commands(tool_value):
        try:
            tokens = shlex.split(command_str)
        except ValueError:
            return [call]
        if not tokens:
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
            return [call]

        tool_name, argument_specs = matched_spec
        remaining = tokens[matched_prefix_len:]
        parsed_args: Dict[str, Any] = {}
        parse_errors: List[str] = []
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


def _expand_wrapper_calls(parsed_calls) -> List:
    """Expand delegated wrapper calls into concrete tool calls."""
    expanded = []
    for call in parsed_calls:
        if _looks_like_cli_wrapper_call(call):
            expanded.extend(_expand_cli_wrapper_call(call))
            continue
        expanded.append(call)

    return expanded


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
    has_query = _extract_query(args, key_hints) is not None

    if has_query:
        return "search"
    if has_source_path and has_dest_path:
        return "move"
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

    if {"path", "content"}.issubset(required):
        return "write"
    if {"path", "oldContent", "newContent", "startLine", "endLine"}.issubset(required):
        return "replace"
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
            and _extract_old_content(args) is not None
            and _extract_new_content(args) is not None
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
    read_output: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if action == "mkdir":
        runtime.mkdir(_require_path(_extract_source_path(args, key_hints), "path"))
        return "created"
    if action == "list":
        path = _extract_source_path(args, key_hints) or "."
        return json.dumps(runtime.list_dir(path))
    if action == "read":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        content = runtime.read_text(path)
        return _format_read_output(content, args=args, options=read_output or {})
    if action == "write":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        content = _extract_content(args, key_hints)
        runtime.write_text(path, "" if content is None else content)
        return "written"
    if action == "replace":
        path = _require_path(_extract_source_path(args, key_hints), "path")
        content = runtime.read_text(path)
        updated = _replace_text_range(
            content,
            old_content=_require_text(_extract_old_content(args), "oldContent"),
            new_content=_require_text(_extract_new_content(args), "newContent"),
            start_line=_extract_line_number(args, ["startLine", "start_line", "start", "line"]),
            end_line=_extract_line_number(args, ["endLine", "end_line", "end"]),
        )
        runtime.write_text(path, updated)
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


def _format_read_output(content: str, args: Dict[str, Any], options: Dict[str, Any]) -> str:
    include_line_numbers = bool(options.get("include_line_numbers", False))
    honor_line_range = bool(options.get("honor_line_range", False))
    if not include_line_numbers and not honor_line_range:
        return content

    lines = content.splitlines()
    start_line = _extract_line_number(args, ["startLine", "start_line", "start", "line"]) or 1
    end_line = _extract_line_number(args, ["endLine", "end_line", "end"])

    start_line = max(1, start_line)
    if end_line is None:
        end_line = len(lines)
    end_line = max(start_line, end_line)

    selected = lines[start_line - 1 : end_line] if honor_line_range else lines
    first_rendered_line = start_line if honor_line_range else 1

    if not include_line_numbers:
        rendered = "\n".join(selected)
        if content.endswith("\n") and end_line >= len(lines):
            rendered += "\n"
        return rendered

    separator = str(options.get("line_number_separator", ": "))
    width = int(options.get("line_number_width", 0) or 0)
    rendered_lines = []
    for offset, line in enumerate(selected):
        line_number = first_rendered_line + offset
        label = str(line_number).rjust(width) if width > 0 else str(line_number)
        rendered_lines.append(f"{label}{separator}{line}")
    return "\n".join(rendered_lines)


def _extract_line_number(args: Dict[str, Any], keys: List[str]) -> Optional[int]:
    for key in keys:
        value = args.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.isdigit():
                return int(candidate)
    return None


def _replace_text_range(
    content: str,
    *,
    old_content: str,
    new_content: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> str:
    if start_line is None or end_line is None:
        raise ValueError("Replace action requires startLine and endLine")
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid replace line range: {start_line}-{end_line}")

    lines = content.splitlines(keepends=True)
    if start_line > len(lines):
        raise ValueError(f"Replace startLine {start_line} exceeds file length {len(lines)}")
    end_line = min(end_line, len(lines))
    selected = "".join(lines[start_line - 1 : end_line])
    selected_compare = selected[:-1] if selected.endswith("\n") and not old_content.endswith("\n") else selected

    if selected_compare != old_content:
        found_line = _find_text_line(content, old_content)
        suffix = f"; matching text starts at line {found_line}" if found_line is not None else ""
        raise ValueError(f"Content at lines {start_line}-{end_line} did not match oldContent{suffix}")

    replacement = new_content
    if selected.endswith("\n") and replacement and not replacement.endswith("\n"):
        replacement += "\n"
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement_lines:
        replacement_lines = [replacement]
    updated_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]
    return "".join(updated_lines)


def _find_text_line(content: str, needle: str) -> Optional[int]:
    if not needle:
        return None
    index = content.find(needle)
    if index < 0:
        return None
    return content.count("\n", 0, index) + 1


def _extract_old_content(args: Dict[str, Any]) -> Optional[str]:
    return _extract_first_string(args, ["oldContent", "old_content", "old", "findText", "find_text"])


def _extract_new_content(args: Dict[str, Any]) -> Optional[str]:
    return _extract_first_string(args, ["newContent", "new_content", "new", "replaceText", "replace_text"])


def _extract_first_string(args: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None


def _require_text(value: Optional[str], label: str) -> str:
    if value is None:
        raise ValueError(f"Missing required text argument: {label}")
    return value


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
