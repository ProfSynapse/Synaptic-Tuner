"""Shared helpers for CLI-schema dataset migration scripts."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import bump_version, find_latest_version, read_jsonl


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_target_catalog(schema_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load the current CLI-oriented tool catalog."""
    with open(schema_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for tool in payload.get("tools", []):
        agent = tool["agent"]
        name = tool["tool"]
        arguments = tool.get("arguments", [])
        catalog[(agent, name)] = {
            "required_args": sorted(arg["name"] for arg in arguments if arg.get("required")),
            "all_args": sorted(arg["name"] for arg in arguments),
            "usage": tool.get("usage"),
            "command": tool.get("command"),
            "argument_specs": arguments,
        }
    return catalog


def build_command_lookup(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Tuple[str, str, Dict[str, Any]]]:
    lookup: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    for (agent, tool), spec in catalog.items():
        command = spec.get("command")
        if isinstance(command, str) and command.strip():
            lookup[command.strip()] = (agent, tool, spec)
    return lookup


def discover_latest_nonthinking_dataset_files(
    datasets_root: Path,
    agent_names: Iterable[str],
) -> Dict[str, Path]:
    """Return latest clean version file for each requested non-thinking agent."""
    results: Dict[str, Path] = {}
    nonthinking_root = datasets_root / "non_thinking"

    for agent in agent_names:
        folder = nonthinking_root / agent
        latest = find_latest_version(folder)
        if latest is not None:
            results[agent] = latest

    return results


def load_jsonl_with_line_numbers(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """Read JSONL and retain 1-based source line numbers."""
    items = read_jsonl(path)
    return list(enumerate(items, start=1))


def parse_arguments(arguments: Any) -> Dict[str, Any]:
    """Normalize tool-call arguments from either string or object form."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


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


def _parse_cli_value(raw_value: str, value_type: str) -> Any:
    lowered = (value_type or "").strip().lower()
    if lowered.startswith("array") or lowered == "object" or lowered == "array<object>" or lowered == "array<string>":
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
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


def parse_cli_tool_string(
    tool_value: str,
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    command_lookup = build_command_lookup(catalog)
    sorted_commands = sorted(command_lookup.keys(), key=lambda value: len(value.split()), reverse=True)
    normalized: List[Dict[str, Any]] = []

    for command_str in _split_cli_commands(tool_value):
        try:
            tokens = shlex.split(command_str)
        except ValueError:
            continue
        if not tokens:
            continue

        matched_command: Optional[str] = None
        matched_prefix_len = 0
        for command in sorted_commands:
            command_tokens = command.split()
            if tokens[: len(command_tokens)] == command_tokens:
                matched_command = command
                matched_prefix_len = len(command_tokens)
                break
        if matched_command is None:
            continue

        agent, tool, spec = command_lookup[matched_command]
        remaining = tokens[matched_prefix_len:]
        params: Dict[str, Any] = {}
        positional_specs = [arg for arg in spec.get("argument_specs", []) if arg.get("positional")]
        flag_specs = {arg.get("flag"): arg for arg in spec.get("argument_specs", []) if arg.get("flag")}

        positional_index = 0
        i = 0
        while i < len(remaining):
            token = remaining[i]
            if token.startswith("--"):
                arg_spec = flag_specs.get(token)
                if not arg_spec:
                    i += 1
                    continue
                name = arg_spec["name"]
                value_type = arg_spec.get("type", "string")
                if value_type == "boolean":
                    params[name] = True
                    i += 1
                    continue
                if i + 1 < len(remaining):
                    params[name] = _parse_cli_value(remaining[i + 1], value_type)
                    i += 2
                    continue
                i += 1
                continue

            if positional_index < len(positional_specs):
                arg_spec = positional_specs[positional_index]
                params[arg_spec["name"]] = _parse_cli_value(token, arg_spec.get("type", "string"))
                positional_index += 1
            i += 1

        normalized.append(
            {
                "source": "cli_wrapper",
                "function_name": "useTools",
                "agent": agent,
                "tool": tool,
                "params": params,
                "command": matched_command,
            }
        )

    return normalized


def extract_normalized_calls(
    example: Dict[str, Any],
    catalog: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Extract assistant tool calls into a normalized call list."""
    normalized: List[Dict[str, Any]] = []

    for message in example.get("conversations", []):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            function_name = function.get("name")
            arguments = parse_arguments(function.get("arguments", {}))

            if function_name == "useTools":
                if isinstance(arguments.get("tool"), str) and catalog is not None:
                    normalized.extend(parse_cli_tool_string(arguments["tool"], catalog))
                    continue
                for wrapped_call in arguments.get("calls", []) or []:
                    normalized.append(
                        {
                            "source": "wrapped_legacy",
                            "function_name": function_name,
                            "agent": wrapped_call.get("agent"),
                            "tool": wrapped_call.get("tool"),
                            "params": wrapped_call.get("params", {}) or {},
                        }
                    )
                continue

            if isinstance(function_name, str) and "_" in function_name:
                agent, tool = function_name.split("_", 1)
                normalized.append(
                    {
                        "source": "direct",
                        "function_name": function_name,
                        "agent": agent,
                        "tool": tool,
                        "params": arguments,
                    }
                )

    return normalized


def classify_example_bucket(call_buckets: Iterable[str]) -> str:
    """Collapse call-level buckets into one example-level bucket."""
    buckets = list(call_buckets)
    if not buckets:
        return "no_calls"
    if "regenerate" in buckets:
        return "regenerate"
    if "heuristic" in buckets:
        return "heuristic"
    if "out_of_scope" in buckets:
        return "out_of_scope"
    return "auto"


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def serialize_arguments_like(original_arguments: Any, payload: Dict[str, Any]) -> Any:
    """Preserve argument container style when rewriting a tool call."""
    if isinstance(original_arguments, str):
        return json.dumps(payload, ensure_ascii=False)
    return payload


def validate_call_shape(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    agent: str,
    tool: str,
    params: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate params against the target CLI schema surface."""
    key = (agent, tool)
    if key not in catalog:
        return False, [f"unknown_target_tool:{agent}.{tool}"]

    spec = catalog[key]
    required = set(spec["required_args"])
    allowed = set(spec["all_args"])
    present = set(params.keys())

    errors: List[str] = []
    missing = sorted(required - present)
    extra = sorted(present - allowed)

    if missing:
        errors.extend(f"missing_required:{name}" for name in missing)
    if extra:
        errors.extend(f"unexpected_param:{name}" for name in extra)

    return not errors, errors


def _quote_cli_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_cli_value(value: Any, value_type: str) -> str:
    lowered = (value_type or "").strip().lower()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if lowered.startswith("array") or lowered == "object":
        return _quote_cli_string(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return _quote_cli_string(str(value))


def render_cli_command(
    agent: str,
    tool: str,
    params: Dict[str, Any],
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    spec = catalog[(agent, tool)]
    pieces = [spec["command"]]
    arg_specs = spec.get("argument_specs", [])

    for arg in arg_specs:
        name = arg["name"]
        if name not in params:
            continue
        value = params[name]
        if arg.get("positional"):
            pieces.append(render_cli_value(value, arg.get("type", "string")))
            continue
        if arg.get("type") == "boolean":
            if value:
                pieces.append(arg["flag"])
            continue
        pieces.append(arg["flag"])
        pieces.append(render_cli_value(value, arg.get("type", "string")))

    return " ".join(pieces)


def next_version_path(dataset_path: Path) -> Path:
    """Return the next versioned dataset path in the same folder."""
    latest = find_latest_version(dataset_path.parent)
    base = latest if latest is not None else dataset_path
    return dataset_path.with_name(bump_version(base.name))
