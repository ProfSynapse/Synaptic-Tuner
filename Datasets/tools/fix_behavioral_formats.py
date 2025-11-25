#!/usr/bin/env python3
"""
Fix behavioral datasets to use proper OpenAI-compatible tool calling format.

Converts from old format:
    {"role": "assistant", "content": "tool_call: toolName\narguments: {...}"}

To new format:
    {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]}
"""

import json
import re
import random
import string
from pathlib import Path
from typing import Dict, Any, List


def generate_tool_call_id() -> str:
    """Generate a random 9-character tool call ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))


def extract_valid_json(json_str: str) -> str:
    """
    Extract valid JSON from a string that might have extra data after it.
    Uses a JSON decoder that stops at the first valid JSON object.
    """
    decoder = json.JSONDecoder()
    try:
        obj, idx = decoder.raw_decode(json_str)
        # Return the valid JSON portion as a string
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        # If that fails, return the original string
        return json_str.strip()


def parse_old_tool_call_format(content: str) -> Dict[str, Any]:
    """
    Parse old text-based tool call format into structured data.

    Old format:
        tool_call: toolName
        arguments: {"key": "value"}

    Or with multiple tool calls:
        tool_call: tool1
        arguments: {...}

        [Previous: ...]

        tool_call: tool2
        arguments: {...}
    """
    # Split by tool_call markers - capture everything after "arguments: " until double newline or end
    tool_call_pattern = r'tool_call:\s*(\S+)\s*\narguments:\s*(.+?)(?=\n\n|$)'
    matches = re.findall(tool_call_pattern, content, re.DOTALL)

    if not matches:
        # No tool calls found - this shouldn't happen, but handle gracefully
        print(f"WARNING: Could not parse tool call from content: {content[:100]}...")
        return None

    tool_calls = []
    for tool_name, arguments_str in matches:
        # Clean up the arguments string
        arguments_str = arguments_str.strip()

        # Try to extract valid JSON if there's extra data
        clean_arguments = extract_valid_json(arguments_str)

        # Validate it's valid JSON
        try:
            json.loads(clean_arguments)
        except json.JSONDecodeError as e:
            print(f"WARNING: Invalid JSON in arguments for {tool_name}: {arguments_str[:100]}...")
            print(f"Error: {e}")
            continue

        tool_calls.append({
            "id": generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": clean_arguments
            }
        })

    return tool_calls if tool_calls else None


def fix_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix a single conversation's format."""
    fixed = []

    for message in conversation:
        if message["role"] == "assistant" and "content" in message and message["content"]:
            # Check if this is the old tool_call format
            content = message["content"]
            if isinstance(content, str) and "tool_call:" in content:
                tool_calls = parse_old_tool_call_format(content)
                if tool_calls:
                    # Convert to new format
                    fixed.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                else:
                    # Couldn't parse - keep original
                    fixed.append(message)
            else:
                # Not a tool call, keep as-is
                fixed.append(message)
        else:
            # Keep as-is (user messages, etc.)
            fixed.append(message)

    return fixed


def fix_jsonl_file(input_path: Path, output_path: Path):
    """Fix a single JSONL file."""
    print(f"Processing {input_path}...")

    fixed_count = 0
    total_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                total_count += 1

                # Fix the conversations
                if "conversations" in entry:
                    original_convs = entry["conversations"]
                    fixed_convs = fix_conversation(original_convs)

                    # Check if anything changed
                    if fixed_convs != original_convs:
                        fixed_count += 1
                        entry["conversations"] = fixed_convs

                # Write the (potentially fixed) entry
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"ERROR on line {line_num}: {e}")
                print(f"Line content: {line[:100]}...")
                continue

    print(f"  Fixed {fixed_count}/{total_count} entries")
    return fixed_count, total_count


def main():
    """Fix all behavioral dataset files."""
    base_dir = Path(__file__).parent
    behavior_dir = base_dir / "behavior_datasets"

    print("=" * 80)
    print("Fixing Behavioral Datasets to OpenAI-Compatible Format")
    print("=" * 80)
    print()

    total_fixed = 0
    total_entries = 0

    # Fix individual behavioral dataset files
    behavior_types = [
        "verification_before_action",
        "workspace_awareness",
        "context_continuity",
        "context_efficiency",
        "strategic_tool_selection",
        "error_recovery",
        "execute_prompt_usage",
        "intellectual_humility"
    ]

    for behavior_type in behavior_types:
        input_file = behavior_dir / behavior_type / "pairs_v1.0.jsonl"
        output_file = behavior_dir / behavior_type / "pairs_v1.1.jsonl"

        if input_file.exists():
            fixed, total = fix_jsonl_file(input_file, output_file)
            total_fixed += fixed
            total_entries += total
            print(f"  → Saved to {output_file.relative_to(base_dir)}")
            print()
        else:
            print(f"  ⚠️  {input_file} not found, skipping")
            print()

    # Summary
    print("=" * 80)
    print(f"SUMMARY: Fixed {total_fixed}/{total_entries} total entries")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the v1.1 files to verify correctness")
    print("2. Update merge_behavior_datasets.py to include new behaviors")
    print("3. Run merge script to create updated merged dataset")
    print()


if __name__ == "__main__":
    main()
