#!/usr/bin/env python3
"""
Fix broken examples where multi-tool-call conversations have mismatched IDs.

The issue: System prompts were built from the first tool call only, but some
examples have multiple tool calls with different workspaceIds.

Solution: Update system prompts to include ALL workspaceIds from ALL tool calls.
"""

import json
import re
import sys
from pathlib import Path

# Broken examples by file and line number
BROKEN_EXAMPLES = {
    "agentManager/tools_v1.1.jsonl": [],  # Need to find line numbers
    "contentManager/tools_v1.1.jsonl": [259, 379, 440, 698, 1007],
    "memoryManager/tools_v1.1.jsonl": [938, 1025, 1279],
    "vaultManager/tools_v1.1.jsonl": [],  # Need to find line number
}


def extract_all_workspace_ids(conversations):
    """Extract all unique workspaceIds from all tool calls."""
    workspace_ids = set()

    for msg in conversations:
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
                context = args.get("context", {})
                ws_id = context.get("workspaceId")
                if ws_id and ws_id != "default":
                    workspace_ids.add(ws_id)

                # Also check 'id' param for workspace operations
                target_id = args.get("id", "")
                if target_id.startswith("ws_"):
                    workspace_ids.add(target_id)
            except:
                pass

    return list(workspace_ids)


def extract_session_id(conversations):
    """Extract sessionId from first tool call."""
    for msg in conversations:
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str)
                context = args.get("context", {})
                return context.get("sessionId", "")
            except:
                pass
    return ""


def generate_workspace_name(ws_id):
    """Generate a workspace name from ID."""
    # Use last part of ID for uniqueness
    suffix = ws_id.split("_")[-1][:4] if "_" in ws_id else ws_id[:4]
    return f"Workspace {suffix.upper()}"


def build_fixed_system_prompt(session_id, workspace_ids, primary_workspace_id=None):
    """Build a fixed system prompt with all workspace IDs."""

    # Determine the "current" workspace (first non-default, or default)
    current_ws = primary_workspace_id or (workspace_ids[0] if workspace_ids else "default")

    prompt = "<session_context>\n"
    prompt += "IMPORTANT: When using tools, include these values in your tool call parameters:\n\n"
    prompt += f'- sessionId: "{session_id}"\n'

    if current_ws == "default":
        prompt += '- workspaceId: "default" (no specific workspace selected)\n'
        prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"
        prompt += "NOTE: Use \"default\" as the workspaceId when no specific workspace context is needed.\n"
    else:
        prompt += f'- workspaceId: "{current_ws}" (current workspace)\n'
        prompt += "\nInclude these in the \"context\" parameter of your tool calls.\n"

    prompt += "</session_context>"

    # Add available_workspaces section if we have non-default workspaces
    if workspace_ids:
        prompt += "\n<available_workspaces>\n"
        prompt += "The following workspaces are available in this vault:\n\n"

        for ws_id in workspace_ids:
            name = generate_workspace_name(ws_id)
            prompt += f'- {name} (id: "{ws_id}")\n'
            prompt += f"  Description: Workspace for {name.lower()} operations\n"
            prompt += f"  Root folder: {name.replace(' ', '')}/"
            prompt += "\n\n"

        prompt += "Use memoryManager with loadWorkspace mode to get full workspace context.\n"
        prompt += "</available_workspaces>"

    return prompt


def fix_example(example):
    """Fix a single example by updating its system prompt."""
    conversations = example.get("conversations", [])

    # Extract all workspace IDs from all tool calls
    all_workspace_ids = extract_all_workspace_ids(conversations)
    session_id = extract_session_id(conversations)

    if not session_id:
        return example  # Can't fix without session ID

    # Get primary workspace from first tool call's context
    primary_ws = None
    for msg in conversations:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                try:
                    args = json.loads(tool_calls[0]["function"]["arguments"])
                    primary_ws = args.get("context", {}).get("workspaceId")
                    break
                except:
                    pass

    # Build fixed system prompt
    fixed_prompt = build_fixed_system_prompt(session_id, all_workspace_ids, primary_ws)

    # Update or insert system message
    if conversations and conversations[0].get("role") == "system":
        conversations[0]["content"] = fixed_prompt
    else:
        conversations.insert(0, {"role": "system", "content": fixed_prompt})

    example["conversations"] = conversations
    return example


def process_file(filepath, broken_lines=None):
    """Process a file and fix broken examples."""
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    lines = path.read_text().strip().split("\n")
    fixed_count = 0

    # If no specific lines given, check all lines
    if broken_lines is None:
        broken_lines = range(1, len(lines) + 1)

    new_lines = []
    for i, line in enumerate(lines, 1):
        if i in broken_lines:
            try:
                example = json.loads(line)
                fixed_example = fix_example(example)
                new_lines.append(json.dumps(fixed_example, ensure_ascii=False))
                fixed_count += 1
            except Exception as e:
                print(f"Error fixing line {i}: {e}")
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write back
    path.write_text("\n".join(new_lines) + "\n")
    print(f"Fixed {fixed_count} examples in {filepath}")


def find_all_broken_lines(filepath):
    """Find all lines with ID mismatch errors."""
    import subprocess
    result = subprocess.run(
        ["python", "tools/validate_syngen.py", filepath],
        capture_output=True, text=True, cwd="/home/user/Toolset-Training"
    )

    broken = []
    for line in result.stdout.split("\n"):
        if "Example line" in line:
            try:
                num = int(line.split("Example line")[1].split(":")[0].strip())
                broken.append(num)
            except:
                pass

    # Also check stderr
    for line in result.stderr.split("\n"):
        if "does not match system prompt" in line:
            # The line number should be in the previous output
            pass

    return list(set(broken))


def main():
    base_dir = Path("/home/user/Toolset-Training/Datasets/tools_datasets")

    # Files to fix
    files_to_fix = [
        "agentManager/tools_v1.1.jsonl",
        "contentManager/tools_v1.1.jsonl",
        "memoryManager/tools_v1.1.jsonl",
        "vaultManager/tools_v1.1.jsonl",
    ]

    for rel_path in files_to_fix:
        filepath = base_dir / rel_path
        print(f"\nProcessing {rel_path}...")

        # Find broken lines by running validation
        broken = find_all_broken_lines(str(filepath))

        if broken:
            print(f"  Found {len(broken)} broken examples: {broken}")
            process_file(filepath, set(broken))
        else:
            print(f"  No broken examples found")


if __name__ == "__main__":
    main()
