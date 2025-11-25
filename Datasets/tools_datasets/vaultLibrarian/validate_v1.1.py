#!/usr/bin/env python3
"""
Validate that v1.1 dataset has correct system prompts with matching IDs.
"""

import json
import re
from pathlib import Path


def extract_ids_from_system_prompt(content: str) -> dict:
    """Extract sessionId and workspaceId from system prompt."""
    session_match = re.search(r'sessionId: "([^"]+)"', content)
    workspace_match = re.search(r'workspaceId: "([^"]+)"', content)

    return {
        'sessionId': session_match.group(1) if session_match else None,
        'workspaceId': workspace_match.group(1) if workspace_match else None
    }


def extract_ids_from_tool_call(conversations: list) -> dict:
    """Extract sessionId and workspaceId from tool call arguments."""
    for msg in conversations:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_call = tool_calls[0]
                function_data = tool_call.get("function", {})
                arguments_str = function_data.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str)
                    context = arguments.get("context", {})
                    return {
                        'sessionId': context.get("sessionId"),
                        'workspaceId': context.get("workspaceId")
                    }
                except json.JSONDecodeError:
                    return {'sessionId': None, 'workspaceId': None}

    return {'sessionId': None, 'workspaceId': None}


def validate_file(filepath: Path):
    """Validate the processed file."""
    print(f"Validating: {filepath}")
    print("="*60)

    total = 0
    has_system_prompt = 0
    session_id_match = 0
    workspace_id_match = 0
    errors = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                example = json.loads(line)
                conversations = example.get("conversations", [])

                # Check if system prompt exists
                if not conversations or conversations[0].get("role") != "system":
                    errors.append(f"Line {line_num}: No system prompt found")
                    continue

                has_system_prompt += 1

                # Extract IDs from system prompt
                system_content = conversations[0].get("content", "")
                system_ids = extract_ids_from_system_prompt(system_content)

                # Extract IDs from tool call
                tool_ids = extract_ids_from_tool_call(conversations)

                # Validate sessionId match
                if system_ids['sessionId'] == tool_ids['sessionId']:
                    session_id_match += 1
                else:
                    errors.append(
                        f"Line {line_num}: sessionId mismatch\n"
                        f"  System: {system_ids['sessionId']}\n"
                        f"  Tool:   {tool_ids['sessionId']}"
                    )

                # Validate workspaceId match
                if system_ids['workspaceId'] == tool_ids['workspaceId']:
                    workspace_id_match += 1
                else:
                    errors.append(
                        f"Line {line_num}: workspaceId mismatch\n"
                        f"  System: {system_ids['workspaceId']}\n"
                        f"  Tool:   {tool_ids['workspaceId']}"
                    )

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON parse error: {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error: {e}")

    # Print results
    print(f"\nTotal examples: {total}")
    print(f"With system prompts: {has_system_prompt} ({has_system_prompt/total*100:.1f}%)")
    print(f"SessionId matches: {session_id_match} ({session_id_match/total*100:.1f}%)")
    print(f"WorkspaceId matches: {workspace_id_match} ({workspace_id_match/total*100:.1f}%)")

    if errors:
        print(f"\n⚠️  Found {len(errors)} issues:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("\n✅ All validations passed!")

    return total, has_system_prompt, session_id_match, workspace_id_match, len(errors)


def main():
    script_dir = Path(__file__).parent
    output_file = script_dir / "tools_v1.1.jsonl"

    if not output_file.exists():
        print(f"Error: File not found: {output_file}")
        return

    validate_file(output_file)


if __name__ == "__main__":
    main()
