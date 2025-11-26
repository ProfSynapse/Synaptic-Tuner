#!/usr/bin/env python3
"""
Add system context messages to behavioral datasets.

Processes datasets to add system messages with sessionId and workspaceId
at the start of each conversation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def determine_workspace_info(user_content: str) -> Tuple[str, str, str]:
    """
    Determine workspace name, description, and root folder based on user content.

    Returns:
        (workspace_name, description, root_folder)
    """
    content_lower = user_content.lower()

    # Pattern matching for workspace determination
    if any(word in content_lower for word in ['meeting', 'collaboration', 'team', 'calendar']):
        return (
            "Meetings & Collaboration",
            "Meeting notes, team discussions, and collaborative documents",
            "Meetings/"
        )

    if any(word in content_lower for word in ['research', 'paper', 'study', 'article', 'academic']):
        return (
            "Research & Papers",
            "Research notes, academic papers, and study materials",
            "Research/"
        )

    if any(word in content_lower for word in ['config', 'settings', 'api', 'endpoint', 'database']):
        return (
            "Configuration",
            "Configuration files, settings, and system parameters",
            "Config/"
        )

    if any(word in content_lower for word in ['note', 'journal', 'daily', 'diary']):
        return (
            "Daily Notes",
            "Personal notes, journal entries, and daily reflections",
            "Notes/"
        )

    if any(word in content_lower for word in ['project', 'development', 'code', 'build']):
        return (
            "Project Management",
            "Project files, development notes, and planning documents",
            "Projects/"
        )

    # Default workspace
    return (
        "Main Workspace",
        "General workspace for all vault operations",
        "/"
    )


def generate_session_workspace_ids(example_index: int) -> Tuple[str, str]:
    """
    Generate consistent sessionId and workspaceId for examples without tool calls.

    Args:
        example_index: Index of the example (for consistency)

    Returns:
        (sessionId, workspaceId)
    """
    import random
    import string

    # Use example index as seed for consistency
    random.seed(example_index)

    # Generate timestamp-based IDs (use a fixed base timestamp + index offset)
    base_timestamp = 1732300800000  # November 22, 2024
    timestamp = base_timestamp + (example_index * 1000)

    # Generate random suffix (9 lowercase alphanumeric chars)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))

    session_id = f"session_{timestamp}_{suffix}"
    workspace_id = f"ws_{timestamp}_{suffix}"

    return session_id, workspace_id


def extract_session_workspace_ids(example: Dict, example_index: int = 0) -> Tuple[str, str]:
    """
    Extract sessionId and workspaceId from the first tool call in the example.
    If no tool calls found, generate consistent IDs.

    Args:
        example: The example dict
        example_index: Index of the example (for generating IDs if needed)

    Returns:
        (sessionId, workspaceId)
    """
    conversations = example.get('conversations', [])

    # Find the assistant message with tool_calls
    for msg in conversations:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            tool_calls = msg['tool_calls']
            if tool_calls and len(tool_calls) > 0:
                # Parse the arguments JSON
                arguments_str = tool_calls[0]['function']['arguments']
                arguments = json.loads(arguments_str)

                # Extract from context object
                context = arguments.get('context', {})
                session_id = context.get('sessionId', '')
                workspace_id = context.get('workspaceId', '')

                if session_id and workspace_id:
                    return session_id, workspace_id

    # No tool calls found or missing IDs - generate them
    return generate_session_workspace_ids(example_index)


def create_system_message(session_id: str, workspace_id: str, workspace_name: str,
                         workspace_desc: str, root_folder: str) -> Dict:
    """
    Create a system message with session and workspace context.

    Returns:
        System message dict with role and content
    """
    system_content = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""

    return {
        "role": "system",
        "content": system_content
    }


def process_example(example: Dict, example_index: int = 0) -> Dict:
    """
    Process a single example by adding system message at the start.

    Args:
        example: The example dict
        example_index: Index of the example (for generating IDs if needed)

    Returns:
        Modified example with system message
    """
    # Extract IDs from tool calls (or generate if not found)
    session_id, workspace_id = extract_session_workspace_ids(example, example_index)

    # Get user message to determine workspace context
    user_content = ""
    for msg in example['conversations']:
        if msg.get('role') == 'user':
            user_content = msg.get('content', '')
            break

    # Determine workspace info
    workspace_name, workspace_desc, root_folder = determine_workspace_info(user_content)

    # Create system message
    system_msg = create_system_message(
        session_id, workspace_id, workspace_name, workspace_desc, root_folder
    )

    # Insert at start of conversations
    modified_example = example.copy()
    modified_example['conversations'] = [system_msg] + example['conversations']

    return modified_example


def process_dataset(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Process a dataset file and write output.

    Returns:
        (examples_processed, errors)
    """
    examples_processed = 0
    errors = 0

    print(f"\nProcessing: {input_path}")
    print(f"Output to: {output_path}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                example = json.loads(line.strip())
                # Use line_num as example_index for consistent ID generation
                modified_example = process_example(example, example_index=line_num)

                # Write to output
                outfile.write(json.dumps(modified_example, ensure_ascii=False) + '\n')
                examples_processed += 1

            except Exception as e:
                print(f"  ERROR on line {line_num}: {e}")
                errors += 1

    print(f"  Processed: {examples_processed} examples")
    print(f"  Errors: {errors}")

    return examples_processed, errors


def main():
    """Process all behavioral datasets."""
    base_dir = Path(__file__).parent

    datasets = [
        "intellectual_humility/pairs_v1.1.jsonl",
        "verification_before_action/pairs_v1.1.jsonl",
        "execute_prompt_usage/pairs_v1.1.jsonl"
    ]

    total_processed = 0
    total_errors = 0

    print("=" * 60)
    print("Adding System Context to Behavioral Datasets")
    print("=" * 60)

    for dataset in datasets:
        input_path = base_dir / dataset
        output_path = input_path.parent / "pairs_v1.2.jsonl"

        if not input_path.exists():
            print(f"\nWARNING: {input_path} not found, skipping")
            continue

        processed, errors = process_dataset(input_path, output_path)
        total_processed += processed
        total_errors += errors

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_processed} examples processed, {total_errors} errors")
    print("=" * 60)

    if total_errors == 0:
        print("\n✓ All datasets processed successfully!")
        print("\nNext steps:")
        print("  1. Validate each output file:")
        for dataset in datasets:
            output_file = base_dir / dataset.replace("v1.1", "v1.2")
            if output_file.exists():
                print(f"     python tools/validate_syngen.py {output_file}")
    else:
        print(f"\n✗ {total_errors} errors occurred during processing")


if __name__ == "__main__":
    main()
