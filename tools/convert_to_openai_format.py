#!/usr/bin/env python3
"""
Convert tool-calling dataset from text format to OpenAI Function Calling format.

Original format:
    tool_call: toolName
    arguments: {...}

Target format (OpenAI-compatible for Mistral):
    {
      "role": "assistant",
      "content": "Optional text response",
      "tool_calls": [
        {
          "id": "call_<unique_id>",
          "type": "function",
          "function": {
            "name": "toolName",
            "arguments": "{...}"  // JSON string, not dict!
          }
        }
      ]
    }

Usage:
    python tools/convert_to_openai_format.py input.jsonl output.jsonl
    python tools/convert_to_openai_format.py input.jsonl output.jsonl --validate
"""

import argparse
import json
import re
import sys
import random
import string
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Import the validator to reuse tool call extraction
sys.path.insert(0, str(Path(__file__).parent))
import validate_syngen


def generate_call_id() -> str:
    """Generate a unique tool call ID in OpenAI format: call_<random chars>"""
    # OpenAI uses various formats, but call_<random> is common
    chars = string.ascii_letters + string.digits
    random_part = ''.join(random.choices(chars, k=24))
    return f"call_{random_part}"


def convert_tool_call_to_openai(tool_name: str, arguments: dict) -> dict:
    """
    Convert a single tool call from text format to OpenAI format.

    Important: OpenAI format requires arguments as a JSON STRING, not a dict!
    """
    return {
        "id": generate_call_id(),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False)  # Must be string!
        }
    }


def convert_assistant_content(content: str) -> Tuple[str | None, List[Dict[str, Any]] | None]:
    """
    Convert assistant content from text format to OpenAI format.

    Returns (text_content, tool_calls):
    - text_content: String or None (text before tool calls)
    - tool_calls: List of tool call dicts or None

    In OpenAI format:
    - If no tool calls: return (full_content, None)
    - If tool calls only: return (None, tool_calls_list)
    - If text + tool calls: return (text_before, tool_calls_list)
    """
    # Use the existing validator's extraction function
    try:
        tool_calls = validate_syngen.extract_tool_calls(content)
    except Exception as e:
        # If extraction fails, return content as text only
        print(f"Warning: Could not extract tool calls: {e}", file=sys.stderr)
        return (content, None)

    if not tool_calls:
        # No tool calls, return as text only
        return (content, None)

    # Extract text before first tool call (if any)
    first_tool_idx = content.find("tool_call:")
    text_content = None
    if first_tool_idx > 0:
        text_before = content[:first_tool_idx].strip()
        if text_before:
            text_content = text_before

    # Convert all tool calls to OpenAI format
    openai_tool_calls = [
        convert_tool_call_to_openai(tool_name, arguments)
        for tool_name, arguments in tool_calls
    ]

    return (text_content, openai_tool_calls)


def convert_example(example: dict) -> dict:
    """Convert a single training example from text format to OpenAI format."""
    conversations = example.get("conversations", [])
    converted_conversations = []

    for msg in conversations:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant" and "tool_call:" in content:
            # Convert assistant message with tool calls
            text_content, tool_calls = convert_assistant_content(content)

            converted_msg = {"role": role}

            # Add content (can be null if only tool calls)
            converted_msg["content"] = text_content

            # Add tool_calls if present
            if tool_calls:
                converted_msg["tool_calls"] = tool_calls

        else:
            # Keep other messages as-is
            converted_msg = {
                "role": role,
                "content": content
            }

        converted_conversations.append(converted_msg)

    # Preserve label if present
    result = {"conversations": converted_conversations}
    if "label" in example:
        result["label"] = example["label"]

    return result


def validate_openai_format(converted_msg: dict) -> None:
    """
    Validate that a converted message follows OpenAI format.

    Raises AssertionError if validation fails.
    """
    if "tool_calls" in converted_msg:
        tool_calls = converted_msg["tool_calls"]
        assert isinstance(tool_calls, list), "tool_calls must be a list"

        for tc in tool_calls:
            assert "id" in tc, "Missing 'id' field in tool_call"
            assert "type" in tc, "Missing 'type' field in tool_call"
            assert tc["type"] == "function", "type must be 'function'"
            assert "function" in tc, "Missing 'function' field in tool_call"

            func = tc["function"]
            assert "name" in func, "Missing 'name' in function"
            assert "arguments" in func, "Missing 'arguments' in function"
            assert isinstance(func["arguments"], str), "arguments must be a JSON string, not dict!"

            # Verify arguments is valid JSON
            try:
                json.loads(func["arguments"])
            except json.JSONDecodeError:
                raise AssertionError(f"arguments is not valid JSON: {func['arguments'][:100]}")


def convert_dataset(input_path: Path, output_path: Path, validate: bool = False) -> Tuple[int, int]:
    """
    Convert entire dataset from text format to OpenAI format.

    Returns (total_examples, converted_examples).
    """
    total = 0
    converted = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total += 1

            try:
                example = json.loads(stripped)
                converted_example = convert_example(example)

                # Optionally validate the conversion
                if validate:
                    for msg in converted_example["conversations"]:
                        if msg["role"] == "assistant":
                            validate_openai_format(msg)

                outfile.write(json.dumps(converted_example, ensure_ascii=False) + "\n")
                converted += 1

            except Exception as e:
                print(f"Error converting example {idx}: {e}", file=sys.stderr)
                continue

    return total, converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert tool-calling dataset from text format to OpenAI format (for Mistral)"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file (text format)")
    parser.add_argument("output", type=Path, help="Output JSONL file (OpenAI format)")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted examples"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't write output file"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not args.dry_run:
        response = input(f"Output file '{args.output}' exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    print(f"Converting {args.input} -> {args.output}")
    print("Target format: OpenAI Function Calling (Mistral-compatible)")
    if args.validate:
        print("Validation enabled")
    if args.dry_run:
        print("DRY RUN - no output file will be written")

    if args.dry_run:
        # For dry run, just process without writing
        total = 0
        with args.input.open("r", encoding="utf-8") as infile:
            for idx, line in enumerate(infile, start=1):
                if line.strip():
                    total += 1
                    example = json.loads(line.strip())
                    converted = convert_example(example)
                    if idx <= 3:  # Show first 3 examples
                        print(f"\n=== Example {idx} ===")
                        print(json.dumps(converted, indent=2, ensure_ascii=False)[:600])
        print(f"\nProcessed {total} examples (dry run)")
    else:
        total, converted = convert_dataset(args.input, args.output, validate=args.validate)
        print(f"✓ Converted {converted}/{total} examples")
        print(f"✓ Output written to {args.output}")
        print("\nFormat details:")
        print("  - tool_calls: Separate field (not in content)")
        print("  - arguments: JSON string (not dict)")
        print("  - IDs: call_<random>")
        print("  - type: 'function'")


if __name__ == "__main__":
    main()
