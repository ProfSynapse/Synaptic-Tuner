#!/usr/bin/env python3
"""
Convert tool-calling dataset from text format to MCP/Anthropic Messages API format.

Original format:
    tool_call: toolName
    arguments: {...}

Target format:
    {
      "type": "tool_use",
      "id": "toolu_<unique_id>",
      "name": "toolName",
      "input": {...}
    }

Usage:
    python tools/convert_to_mcp_format.py input.jsonl output.jsonl
    python tools/convert_to_mcp_format.py input.jsonl output.jsonl --validate
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


def generate_tool_id() -> str:
    """Generate a unique tool call ID in Anthropic format: toolu_<24 random chars>"""
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    random_part = ''.join(random.choices(chars, k=24))
    return f"toolu_{random_part}"


def convert_tool_call_to_mcp(tool_name: str, arguments: dict) -> dict:
    """Convert a single tool call from text format to MCP format."""
    return {
        "type": "tool_use",
        "id": generate_tool_id(),
        "name": tool_name,
        "input": arguments
    }


def convert_assistant_content(content: str) -> List[Dict[str, Any]]:
    """
    Convert assistant content from text format to MCP format.

    Returns a list of content blocks (text and tool_use blocks).

    Simplified approach:
    - Extract text before first tool call (if any)
    - Convert all tool calls to tool_use blocks
    - Discard any text after/between tool calls (usually empty)
    """
    # Use the existing validator's extraction function
    try:
        tool_calls = validate_syngen.extract_tool_calls(content)
    except Exception as e:
        # If extraction fails, return content as-is in a text block
        print(f"Warning: Could not extract tool calls: {e}", file=sys.stderr)
        return [{"type": "text", "text": content}]

    if not tool_calls:
        # No tool calls, return as text block
        return [{"type": "text", "text": content}]

    blocks = []

    # Check if there's text before the first tool call
    first_tool_idx = content.find("tool_call:")
    if first_tool_idx > 0:
        text_before = content[:first_tool_idx].strip()
        if text_before:
            blocks.append({"type": "text", "text": text_before})

    # Add all tool calls as tool_use blocks
    for tool_name, arguments in tool_calls:
        blocks.append(convert_tool_call_to_mcp(tool_name, arguments))

    return blocks


def convert_example(example: dict) -> dict:
    """Convert a single training example from text format to MCP format."""
    conversations = example.get("conversations", [])
    converted_conversations = []

    for msg in conversations:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant" and "tool_call:" in content:
            # Convert assistant message with tool calls
            content_blocks = convert_assistant_content(content)
            converted_msg = {
                "role": role,
                "content": content_blocks
            }
        else:
            # Keep other messages as-is (but as text blocks for consistency)
            converted_msg = {
                "role": role,
                "content": [{"type": "text", "text": content}] if content else content
            }

        converted_conversations.append(converted_msg)

    # Preserve label if present
    result = {"conversations": converted_conversations}
    if "label" in example:
        result["label"] = example["label"]

    return result


def convert_dataset(input_path: Path, output_path: Path, validate: bool = False) -> Tuple[int, int]:
    """
    Convert entire dataset from text format to MCP format.

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
                    # Basic validation: check that tool calls were converted
                    for msg in converted_example["conversations"]:
                        if msg["role"] == "assistant":
                            content = msg["content"]
                            if isinstance(content, list):
                                for block in content:
                                    if block.get("type") == "tool_use":
                                        # Validate required fields
                                        assert "id" in block, "Missing 'id' field"
                                        assert "name" in block, "Missing 'name' field"
                                        assert "input" in block, "Missing 'input' field"
                                        assert block["type"] == "tool_use", "Wrong type"

                outfile.write(json.dumps(converted_example, ensure_ascii=False) + "\n")
                converted += 1

            except Exception as e:
                print(f"Error converting example {idx}: {e}", file=sys.stderr)
                continue

    return total, converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert tool-calling dataset from text format to MCP format"
    )
    parser.add_argument("input", type=Path, help="Input JSONL file (text format)")
    parser.add_argument("output", type=Path, help="Output JSONL file (MCP format)")
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
                        print(json.dumps(converted, indent=2)[:500])
        print(f"\nProcessed {total} examples (dry run)")
    else:
        total, converted = convert_dataset(args.input, args.output, validate=args.validate)
        print(f"✓ Converted {converted}/{total} examples")
        print(f"✓ Output written to {args.output}")


if __name__ == "__main__":
    main()
