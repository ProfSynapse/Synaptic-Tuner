"""Shared helpers for CLI-schema dataset migration scripts."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from utils import bump_version, find_latest_version, read_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.validation.parsing.cli_commands import CliCommandSpec, parse_cli_commands  # noqa: E402
from shared.validation.parsing.configured_formats import match_configured_wrapper  # noqa: E402


def get_repo_root() -> Path:
    return _REPO_ROOT


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


def parse_cli_tool_string(
    tool_value: str,
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    escapes: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Normalize the catalog commands in a CLI command string; skip unknown ones."""
    command_catalog = {
        spec["command"].strip(): CliCommandSpec(
            command=spec["command"].strip(),
            agent=agent,
            tool=tool,
            arguments=tuple(spec.get("argument_specs", [])),
        )
        for (agent, tool), spec in catalog.items()
        if isinstance(spec.get("command"), str) and spec["command"].strip()
    }
    try:
        commands = parse_cli_commands(tool_value, command_catalog, escapes)
    except ValueError:
        return []

    return [
        {
            "source": "cli_wrapper",
            "function_name": "useTools",
            "agent": command.spec.agent,
            "tool": command.spec.tool,
            "params": command.arguments,
            "command": command.spec.command,
        }
        for command in commands
        if command.spec is not None
    ]


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
                    wrapper_spec = match_configured_wrapper(arguments, function_name=function_name) or {}
                    normalized.extend(
                        parse_cli_tool_string(arguments["tool"], catalog, wrapper_spec.get("command_escapes") or {})
                    )
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
