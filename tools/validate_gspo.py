#!/usr/bin/env python3
"""
Validate GSPO (Group Sequence Policy Optimization) dataset format.

GSPO Format Requirements:
{
  "prompt": [
    {"role": "system", "content": "..."},  # Optional
    {"role": "user", "content": "..."}     # Required
  ],
  "ground_truth_tool": "toolName",          # Required, non-empty string
  "ground_truth_args": {...}                # Required, dict with context object
}

Key validations:
- prompt must be a list of messages with at least one user role
- prompt must NOT contain assistant role (GSPO generates completions)
- ground_truth_tool must be a valid tool name
- ground_truth_args.context must have all 7 required fields
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Required context fields
CONTEXT_FIELDS = [
    "sessionId", "workspaceId", "sessionDescription",
    "sessionMemory", "toolContext", "primaryGoal", "subgoal"
]

# Valid tool prefixes (agent families)
VALID_AGENTS = [
    "vaultManager", "contentManager", "memoryManager",
    "vaultLibrarian", "agentManager"
]


def validate_prompt(prompt: Any, line_num: int) -> List[str]:
    """Validate the prompt field."""
    errors = []

    if not isinstance(prompt, list):
        errors.append(f"Line {line_num}: 'prompt' must be a list, got {type(prompt).__name__}")
        return errors

    if len(prompt) == 0:
        errors.append(f"Line {line_num}: 'prompt' is empty")
        return errors

    roles = []
    for i, msg in enumerate(prompt):
        if not isinstance(msg, dict):
            errors.append(f"Line {line_num}: prompt[{i}] is not a dict")
            continue

        role = msg.get("role")
        if role not in ("system", "user"):
            if role == "assistant":
                errors.append(f"Line {line_num}: prompt contains 'assistant' role - GSPO should not include assistant responses")
            else:
                errors.append(f"Line {line_num}: prompt[{i}] has invalid role '{role}'")
            continue

        roles.append(role)

        if "content" not in msg:
            errors.append(f"Line {line_num}: prompt[{i}] missing 'content' field")

    if "user" not in roles:
        errors.append(f"Line {line_num}: prompt must contain at least one 'user' message")

    return errors


def validate_tool_name(tool_name: Any, line_num: int) -> List[str]:
    """Validate the ground_truth_tool field."""
    errors = []

    if not isinstance(tool_name, str):
        errors.append(f"Line {line_num}: 'ground_truth_tool' must be string, got {type(tool_name).__name__}")
        return errors

    if not tool_name:
        errors.append(f"Line {line_num}: 'ground_truth_tool' is empty")
        return errors

    # Check format: agentName_toolName
    if "_" not in tool_name:
        errors.append(f"Line {line_num}: 'ground_truth_tool' should be in format 'agentName_toolName', got '{tool_name}'")
        return errors

    agent = tool_name.split("_")[0]
    if agent not in VALID_AGENTS:
        errors.append(f"Line {line_num}: Unknown agent '{agent}' in tool '{tool_name}'")

    return errors


def validate_tool_args(args: Any, line_num: int) -> Tuple[List[str], List[str]]:
    """Validate the ground_truth_args field. Returns (errors, warnings)."""
    errors = []
    warnings = []

    if not isinstance(args, dict):
        errors.append(f"Line {line_num}: 'ground_truth_args' must be dict, got {type(args).__name__}")
        return errors, warnings

    # Check for context object
    if "context" not in args:
        warnings.append(f"Line {line_num}: 'ground_truth_args' missing 'context' field")
        return errors, warnings

    context = args["context"]
    if not isinstance(context, dict):
        errors.append(f"Line {line_num}: 'context' must be dict, got {type(context).__name__}")
        return errors, warnings

    # Check required context fields
    missing_fields = [f for f in CONTEXT_FIELDS if f not in context]
    if missing_fields:
        warnings.append(f"Line {line_num}: context missing fields: {missing_fields}")

    # Check for empty sessionMemory (critical field)
    if "sessionMemory" in context and not context["sessionMemory"]:
        warnings.append(f"Line {line_num}: context.sessionMemory is empty")

    return errors, warnings


def validate_example(example: Dict[str, Any], line_num: int) -> Tuple[List[str], List[str]]:
    """Validate a single GSPO example. Returns (errors, warnings)."""
    errors = []
    warnings = []

    required_fields = ["prompt", "ground_truth_tool", "ground_truth_args"]
    missing = [f for f in required_fields if f not in example]

    if missing:
        errors.append(f"Line {line_num}: Missing required fields: {missing}")
        return errors, warnings

    # Validate each field
    errors.extend(validate_prompt(example["prompt"], line_num))
    errors.extend(validate_tool_name(example["ground_truth_tool"], line_num))

    arg_errors, arg_warnings = validate_tool_args(example["ground_truth_args"], line_num)
    errors.extend(arg_errors)
    warnings.extend(arg_warnings)

    return errors, warnings


def validate_dataset(input_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Validate entire GSPO dataset."""

    input_file = Path(input_path)

    if not input_file.exists():
        return {"error": f"File not found: {input_path}"}

    total = 0
    valid = 0
    all_errors = []
    all_warnings = []
    tool_counts = defaultdict(int)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                example = json.loads(line)
                errors, warnings = validate_example(example, line_num)

                if errors:
                    all_errors.extend(errors)
                else:
                    valid += 1
                    # Count tools only for valid examples
                    if "ground_truth_tool" in example:
                        tool_counts[example["ground_truth_tool"]] += 1

                all_warnings.extend(warnings)

            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: Invalid JSON: {e}")

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "errors": all_errors,
        "warnings": all_warnings,
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "unique_tools": len(tool_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate GSPO dataset format"
    )
    parser.add_argument(
        "input",
        help="Path to GSPO dataset (JSONL)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all errors and warnings"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum errors to display (default: 20)"
    )

    args = parser.parse_args()

    print(f"Validating GSPO dataset: {args.input}")
    print("=" * 60)
    print()

    result = validate_dataset(args.input, verbose=args.verbose)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1

    # Summary
    valid_pct = (result["valid"] / result["total"] * 100) if result["total"] > 0 else 0
    print(f"Total examples:   {result['total']}")
    print(f"Valid examples:   {result['valid']} ({valid_pct:.1f}%)")
    print(f"Invalid examples: {result['invalid']}")
    print(f"Unique tools:     {result['unique_tools']}")
    print()

    # Errors
    if result["errors"]:
        print(f"ERRORS ({len(result['errors'])} total):")
        for err in result["errors"][:args.max_errors]:
            print(f"  {err}")
        if len(result["errors"]) > args.max_errors:
            print(f"  ... and {len(result['errors']) - args.max_errors} more errors")
        print()

    # Warnings
    if result["warnings"] and args.verbose:
        print(f"WARNINGS ({len(result['warnings'])} total):")
        for warn in result["warnings"][:args.max_errors]:
            print(f"  {warn}")
        if len(result["warnings"]) > args.max_errors:
            print(f"  ... and {len(result['warnings']) - args.max_errors} more warnings")
        print()
    elif result["warnings"]:
        print(f"Warnings: {len(result['warnings'])} (use -v to see details)")
        print()

    # Tool distribution
    print("Top 10 tools:")
    for tool, count in list(result["tool_counts"].items())[:10]:
        print(f"  {tool}: {count}")

    # Final verdict
    print()
    if result["invalid"] == 0:
        print("RESULT: Dataset is valid GSPO format")
        return 0
    else:
        print(f"RESULT: Dataset has {result['invalid']} invalid examples")
        return 1


if __name__ == "__main__":
    exit(main())
