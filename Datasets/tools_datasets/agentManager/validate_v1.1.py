#!/usr/bin/env python3
"""
Validate that system prompts in tools_v1.1.jsonl have matching IDs.
"""

import json
import re


def validate_example(example, line_num):
    """Validate a single example."""
    errors = []
    conversations = example['conversations']

    # Check first message is system
    if conversations[0].get('role') != 'system':
        return [f"Line {line_num}: First message is not system role"]

    system_content = conversations[0]['content']

    # Find tool call
    tool_call = None
    for msg in conversations:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            tool_call = msg['tool_calls'][0]
            break

    if not tool_call:
        return [f"Line {line_num}: No tool call found"]

    # Extract IDs from tool call
    args = json.loads(tool_call['function']['arguments'])
    context = args.get('context', {})
    session_id = context.get('sessionId', '')
    workspace_id = context.get('workspaceId', '')

    # Validate sessionId in system prompt
    if session_id and session_id not in system_content:
        errors.append(f"Line {line_num}: sessionId '{session_id}' not in system prompt")

    # Validate workspaceId in system prompt
    if workspace_id and workspace_id not in system_content:
        errors.append(f"Line {line_num}: workspaceId '{workspace_id}' not in system prompt")

    # Validate agent IDs if present
    function_name = tool_call['function']['name']

    if 'updateAgent' in function_name or 'deleteAgent' in function_name:
        agent_id = args.get('id', '')
        if agent_id and '<available_agents>' in system_content:
            if agent_id not in system_content:
                errors.append(f"Line {line_num}: agent ID '{agent_id}' not in system prompt")

    elif 'createAgent' in function_name:
        agent_name = args.get('name', '')
        if agent_name and '<available_agents>' in system_content:
            if agent_name not in system_content:
                errors.append(f"Line {line_num}: agent name '{agent_name}' not in system prompt")

    elif 'executePrompt' in function_name:
        agent_name = args.get('agent', '')
        if agent_name and '<available_agents>' in system_content:
            if agent_name not in system_content:
                errors.append(f"Line {line_num}: agent name '{agent_name}' not in system prompt")

    elif 'listAgents' in function_name:
        # listAgents should NOT have available_agents section
        if '<available_agents>' in system_content:
            errors.append(f"Line {line_num}: listAgents should not have available_agents section")

    return errors


def main():
    """Main validation function."""
    input_file = '/home/user/Toolset-Training/Datasets/tools_datasets/agentManager/tools_v1.1.jsonl'

    print("Validating tools_v1.1.jsonl...")
    print()

    all_errors = []
    examples_checked = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                errors = validate_example(example, line_num)
                all_errors.extend(errors)
                examples_checked += 1
            except Exception as e:
                all_errors.append(f"Line {line_num}: Parse error - {e}")

    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Examples checked: {examples_checked}")
    print(f"Errors found: {len(all_errors)}")
    print()

    if all_errors:
        print("ERRORS:")
        for error in all_errors[:20]:  # Show first 20 errors
            print(f"  ❌ {error}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors")
    else:
        print("✓ All examples passed validation!")
        print()
        print("Validation checks:")
        print("  ✓ All examples have system prompts")
        print("  ✓ All sessionIds match between system prompt and tool call")
        print("  ✓ All workspaceIds match between system prompt and tool call")
        print("  ✓ All agent IDs match when present")
        print("  ✓ listAgents examples don't have available_agents section")

    print()


if __name__ == '__main__':
    main()
