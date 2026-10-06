#!/usr/bin/env python3
"""Analyze tool coverage using the current CLI-first schema.

This script is intentionally schema-driven:
- Valid tools come from cli-first-tool-schemas.json
- CLI commands inside useTools.tool are parsed from schema command metadata
- The active dataset path is expected to use CLI-first useTools payloads
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.validation.parsing.cli_commands import (  # noqa: E402
    CliCommandSpec,
    load_cli_command_catalog,
    parse_cli_commands,
)


def load_tool_schema(path: Path) -> Tuple[set[str], Dict[str, List[str]], Dict[str, CliCommandSpec]]:
    command_lookup = load_cli_command_catalog(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid_tools: set[str] = set()
    tools_by_agent: Dict[str, List[str]] = defaultdict(list)

    for item in payload.get("tools", []):
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", "")).strip()
        tool = str(item.get("tool", "")).strip()
        if not agent or not tool:
            continue

        full_name = f"{agent}_{tool}"
        valid_tools.add(full_name)
        tools_by_agent[agent].append(full_name)

    return valid_tools, {k: sorted(v) for k, v in tools_by_agent.items()}, command_lookup


def parse_arguments(arguments: Any) -> Dict[str, Any]:
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


def extract_from_cli(tool_value: str, command_lookup: Dict[str, CliCommandSpec]) -> List[str]:
    """Return the tool names of the catalog commands in a CLI command string."""
    try:
        # Only command names are read here, so argument escapes do not matter.
        commands = parse_cli_commands(tool_value, command_lookup, {})
    except ValueError:
        return []
    return [command.spec.tool_name for command in commands if command.spec is not None]


def extract_tools_from_assistant_message(
    message: Dict[str, Any],
    command_lookup: Dict[str, CliCommandSpec],
) -> List[str]:
    tools: List[str] = []

    for tool_call in message.get("tool_calls", []) or []:
        function = tool_call.get("function", {}) or {}
        function_name = str(function.get("name") or "").strip()
        arguments = parse_arguments(function.get("arguments", {}))

        if function_name == "useTools":
            tool_value = arguments.get("tool")
            if isinstance(tool_value, str) and tool_value.strip():
                tools.extend(extract_from_cli(tool_value, command_lookup))
            continue

        if function_name:
            tools.append(function_name)

    content = message.get("content", "") or ""
    tools.extend(re.findall(r"tool_call:\s*([a-zA-Z_][a-zA-Z0-9_]*)", content))
    return tools


def analyze_coverage(jsonl_file: Path, valid_tools: set[str], tools_by_agent: Dict[str, List[str]], command_lookup: Dict[str, CliCommandSpec]) -> Dict[str, Any]:
    tool_counter = Counter()
    label_tool_counter = defaultdict(Counter)
    invalid_tool_counter = Counter()
    invalid_examples = []
    conversation_count = 0
    total_tool_calls = 0
    good_example_count = 0
    bad_example_count = 0

    with jsonl_file.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Error parsing line {line_num}: {exc}", file=sys.stderr)
                continue

            conversations = data.get("conversations", [])
            label = data.get("label")

            conversation_count += 1
            if label is True:
                good_example_count += 1
            elif label is False:
                bad_example_count += 1

            for message in conversations:
                if message.get("role") != "assistant":
                    continue
                for tool in extract_tools_from_assistant_message(message, command_lookup):
                    total_tool_calls += 1
                    if tool in valid_tools:
                        tool_counter[tool] += 1
                        if label is not None:
                            label_tool_counter[label][tool] += 1
                    else:
                        invalid_tool_counter[tool] += 1
                        invalid_examples.append({"line": line_num, "tool": tool, "label": label})

    return {
        "tool_counter": tool_counter,
        "label_tool_counter": label_tool_counter,
        "invalid_tool_counter": invalid_tool_counter,
        "invalid_examples": invalid_examples,
        "conversation_count": conversation_count,
        "total_tool_calls": total_tool_calls,
        "good_example_count": good_example_count,
        "bad_example_count": bad_example_count,
        "tools_by_agent": tools_by_agent,
        "valid_tool_count": len(valid_tools),
    }


def print_report(results: Dict[str, Any]) -> None:
    tool_counter = results["tool_counter"]
    label_tool_counter = results["label_tool_counter"]
    invalid_tool_counter = results["invalid_tool_counter"]
    tools_by_agent = results["tools_by_agent"]

    print("=" * 80)
    print("TOOL COVERAGE REPORT")
    print("=" * 80)
    print(f"\nConversations analyzed: {results['conversation_count']}")
    print(f"Total tool calls:        {results['total_tool_calls']}")
    print(f"Good examples:           {results['good_example_count']}")
    print(f"Bad examples:            {results['bad_example_count']}")
    print(f"Schema-defined tools:    {results['valid_tool_count']}")

    print("\nTool usage by agent:")
    print("-" * 80)
    for agent in sorted(tools_by_agent):
        manager_tools = tools_by_agent[agent]
        used = [tool for tool in manager_tools if tool_counter[tool] > 0]
        print(f"\n{agent}:")
        print(f"  Covered: {len(used)}/{len(manager_tools)}")
        if used:
            for tool in used:
                print(
                    f"  - {tool}: total={tool_counter[tool]}, "
                    f"good={label_tool_counter[True][tool]}, bad={label_tool_counter[False][tool]}"
                )
        missing = [tool for tool in manager_tools if tool_counter[tool] == 0]
        if missing:
            print(f"  Missing: {', '.join(missing)}")

    if invalid_tool_counter:
        print("\nInvalid/unrecognized tool names:")
        print("-" * 80)
        for tool, count in invalid_tool_counter.most_common():
            print(f"  {tool}: {count}")
    else:
        print("\nNo invalid tool names found.")


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    if len(args) != 2:
        print("Usage: python3 tools/analyze_tool_coverage.py <dataset.jsonl>")
        return 1

    dataset_path = Path(args[1]).resolve()
    schema_path = _REPO_ROOT / "cli-first-tool-schemas.json"

    valid_tools, tools_by_agent, command_lookup = load_tool_schema(schema_path)
    results = analyze_coverage(dataset_path, valid_tools, tools_by_agent, command_lookup)
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
