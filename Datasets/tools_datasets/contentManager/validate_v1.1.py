#!/usr/bin/env python3
"""
Validate tools_v1.1.jsonl according to CONTEXT_INJECTION_SPEC.md
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def validate_example(example: Dict, line_num: int) -> Tuple[bool, List[str]]:
    """
    Validate a single example.

    Returns: (is_valid, list_of_errors)
    """
    errors = []
    conversations = example.get("conversations", [])

    if not conversations:
        errors.append(f"Line {line_num}: No conversations found")
        return False, errors

    # Rule 1: System prompt must exist as first message
    if conversations[0].get("role") != "system":
        errors.append(f"Line {line_num}: First message is not a system message")
        return False, errors

    system_content = conversations[0].get("content", "")

    # Find assistant message with tool calls
    tool_call_msg = None
    for msg in conversations:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call_msg = msg
            break

    if not tool_call_msg:
        # No tool call, can't validate IDs
        return True, []

    # Extract tool call arguments
    tool_call = tool_call_msg["tool_calls"][0]
    function = tool_call.get("function", {})
    args_str = function.get("arguments", "{}")

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        errors.append(f"Line {line_num}: Could not parse tool call arguments")
        return False, errors

    context = args.get("context", {})
    session_id = context.get("sessionId", "")
    workspace_id = context.get("workspaceId", "default")

    # Rule 2: Session ID must match
    if session_id and session_id not in system_content:
        errors.append(f"Line {line_num}: Session ID '{session_id}' not found in system prompt")

    # Rule 3: Workspace ID must match
    if workspace_id and workspace_id not in system_content:
        errors.append(f"Line {line_num}: Workspace ID '{workspace_id}' not found in system prompt")

    # Check for session_context section
    if "<session_context>" not in system_content:
        errors.append(f"Line {line_num}: Missing <session_context> section")

    # Check for available_workspaces section (should be present for non-default workspaces)
    if workspace_id != "default":
        if "<available_workspaces>" not in system_content:
            errors.append(f"Line {line_num}: Missing <available_workspaces> section for non-default workspace")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    """Main validation function."""
    input_path = Path("/home/user/Toolset-Training/Datasets/tools_datasets/contentManager/tools_v1.1.jsonl")

    print(f"Validating {input_path}")
    print("=" * 60)

    total_count = 0
    valid_count = 0
    invalid_count = 0
    system_message_count = 0
    default_workspace_count = 0
    specific_workspace_count = 0
    all_errors = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1

            try:
                example = json.loads(line.strip())
                conversations = example.get("conversations", [])

                # Count system messages
                if conversations and conversations[0].get("role") == "system":
                    system_message_count += 1
                    system_content = conversations[0].get("content", "")

                    # Count workspace types
                    if 'workspaceId: "default"' in system_content:
                        default_workspace_count += 1
                    else:
                        specific_workspace_count += 1

                # Validate
                is_valid, errors = validate_example(example, line_num)

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    all_errors.extend(errors)

            except json.JSONDecodeError as e:
                invalid_count += 1
                all_errors.append(f"Line {line_num}: JSON parsing error: {str(e)}")
            except Exception as e:
                invalid_count += 1
                all_errors.append(f"Line {line_num}: Unexpected error: {str(e)}")

    # Print results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total examples: {total_count}")
    print(f"Valid examples: {valid_count}")
    print(f"Invalid examples: {invalid_count}")
    print(f"\nSystem message statistics:")
    print(f"  Examples with system messages: {system_message_count}")
    print(f"  Default workspace examples: {default_workspace_count}")
    print(f"  Specific workspace examples: {specific_workspace_count}")

    if all_errors:
        print(f"\n{'='*60}")
        print(f"ERRORS FOUND: {len(all_errors)}")
        print(f"{'='*60}")
        for error in all_errors[:20]:  # Show first 20 errors
            print(error)
        if len(all_errors) > 20:
            print(f"\n... and {len(all_errors) - 20} more errors")
    else:
        print(f"\n✓ All examples passed validation!")

    # Summary
    print(f"\n{'='*60}")
    if invalid_count == 0:
        print("✓ VALIDATION PASSED")
    else:
        print(f"✗ VALIDATION FAILED ({invalid_count} invalid examples)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
