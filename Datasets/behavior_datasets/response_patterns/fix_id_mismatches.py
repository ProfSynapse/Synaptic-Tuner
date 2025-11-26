#!/usr/bin/env python3
"""
Fix ID mismatches in text_only_pairs dataset.

For label=false examples, updates tool call arguments to use sessionId and
workspaceId from the system message instead of hardcoded IDs.
"""

import json
import re
from pathlib import Path


def extract_ids_from_system_message(system_content: str) -> tuple[str, str]:
    """Extract sessionId and workspaceId from system message."""
    session_match = re.search(r'sessionId:\s*["\']([^"\']+)["\']', system_content)
    workspace_match = re.search(r'workspaceId:\s*["\']([^"\']+)["\']', system_content)

    if not session_match or not workspace_match:
        raise ValueError(f"Could not extract IDs from system message")

    return session_match.group(1), workspace_match.group(1)


def fix_tool_call_ids(example: dict, session_id: str, workspace_id: str) -> dict:
    """Update tool call arguments to use correct IDs."""
    conversations = example['conversations']

    # Find the assistant message with tool_calls
    for msg in conversations:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                # Parse arguments JSON string
                args_str = tool_call['function']['arguments']
                args = json.loads(args_str)

                # Update IDs in context
                if 'context' in args:
                    args['context']['sessionId'] = session_id
                    args['context']['workspaceId'] = workspace_id

                # Re-serialize to JSON string
                tool_call['function']['arguments'] = json.dumps(args)

    return example


def process_dataset(input_path: Path, output_path: Path) -> dict:
    """Process the dataset and fix ID mismatches."""
    stats = {
        'total': 0,
        'label_true': 0,
        'label_false': 0,
        'fixed': 0,
        'errors': 0
    }

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                stats['total'] += 1

                if example['label']:
                    # label=true: pass through unchanged (text-only responses)
                    stats['label_true'] += 1
                    outfile.write(json.dumps(example) + '\n')
                else:
                    # label=false: fix ID mismatches
                    stats['label_false'] += 1

                    # Extract IDs from system message
                    system_msg = example['conversations'][0]
                    if system_msg['role'] != 'system':
                        raise ValueError(f"Line {line_num}: First message is not system role")

                    session_id, workspace_id = extract_ids_from_system_message(
                        system_msg['content']
                    )

                    # Fix tool call IDs
                    fixed_example = fix_tool_call_ids(example, session_id, workspace_id)
                    stats['fixed'] += 1

                    outfile.write(json.dumps(fixed_example) + '\n')

            except Exception as e:
                stats['errors'] += 1
                print(f"Error processing line {line_num}: {e}")
                # Write original line on error to not lose data
                outfile.write(line + '\n')

    return stats


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent
    input_file = base_dir / 'text_only_pairs_v1.1.jsonl'
    output_file = base_dir / 'text_only_pairs_v1.2.jsonl'

    print(f"Processing: {input_file}")
    print(f"Output to: {output_file}")
    print()

    stats = process_dataset(input_file, output_file)

    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total examples processed: {stats['total']}")
    print(f"  - label=true (passed through): {stats['label_true']}")
    print(f"  - label=false (processed): {stats['label_false']}")
    print(f"  - ID mismatches fixed: {stats['fixed']}")
    print(f"  - Errors encountered: {stats['errors']}")
    print()
    print(f"Output file created: {output_file}")
    print()

    if stats['errors'] > 0:
        print("WARNING: Errors were encountered during processing.")
        print("Check the output above for details.")
    else:
        print("✓ No errors encountered")

    return stats['errors']


if __name__ == '__main__':
    exit(main())
