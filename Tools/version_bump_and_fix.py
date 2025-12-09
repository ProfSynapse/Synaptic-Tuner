#!/usr/bin/env python3
"""
Version bump individual datasets from v1.5 to v1.6 with fixes:
- DELETE operations → Text-only with varied confirmation questions
- TOGGLE operations → Boost confidence, set risky=false, keep tool calls
- CREATE/EDIT operations → Boost confidence, set risky=false, keep tool calls
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict

# Question variations for DELETE operations (25+ variations)
DELETE_QUESTIONS = [
    "I notice this is a destructive operation. Can you confirm you want to delete {target}?",
    "Before I delete {target}, can you verify this is the correct item?",
    "This will permanently remove {target}. Are you sure you want to proceed?",
    "I want to make sure - should I delete {target}? This action cannot be undone.",
    "Just confirming: you want me to delete {target}, correct?",
    "Are you certain you want to delete {target}? I'll need explicit confirmation.",
    "To be safe, can you confirm the deletion of {target}?",
    "I see you want to delete {target}. Can you verify this is intentional?",
    "Before proceeding with deletion, can you confirm {target} is the correct item?",
    "This will remove {target}. Do you want me to go ahead?",
    "Can you double-check: should I delete {target}?",
    "I need confirmation before deleting {target}. Proceed?",
    "Just to be clear, you're asking me to delete {target}. Is that right?",
    "Should I proceed with deleting {target}? Please confirm.",
    "I want to verify before deletion: is {target} the correct item?",
    "Deleting {target} is irreversible. Are you sure?",
    "Can you confirm deletion of {target}? I want to make sure I have this right.",
    "Before I remove {target}, can you verify this action?",
    "This will delete {target} permanently. Shall I proceed?",
    "I need explicit permission to delete {target}. Can you confirm?",
    "Are you ready for me to delete {target}? Please confirm first.",
    "To avoid mistakes, can you confirm you want {target} deleted?",
    "Should I go ahead and delete {target}? I need your confirmation.",
    "I'm about to delete {target}. Is this what you want?",
    "Can you verify that {target} should be deleted?",
]

def extract_target_from_tool_call(tool_name, arguments):
    """Extract the target item being acted upon from the tool call."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments

        # Try to find the most descriptive identifier
        for key in ['name', 'path', 'filepath', 'folderPath', 'targetPath', 'agentId', 'contentId', 'stateId', 'workspaceId']:
            if key in args and args[key]:
                value = args[key]
                # Make it more readable if it's a path
                if '/' in str(value):
                    return f'"{value}"'
                return f'"{value}"' if value else "the item"

        # Fallback to a generic description based on tool name
        if 'folder' in tool_name.lower():
            return "the folder"
        elif 'file' in tool_name.lower():
            return "the file"
        elif 'agent' in tool_name.lower():
            return "the agent"
        elif 'content' in tool_name.lower():
            return "the content"
        elif 'state' in tool_name.lower():
            return "the state"
        elif 'workspace' in tool_name.lower():
            return "the workspace"
        else:
            return "the item"
    except:
        return "the item"

def get_operation_type(tool_name):
    """Determine operation type from tool name."""
    tool_lower = tool_name.lower()

    # Check for delete operations (truly destructive)
    if any(op in tool_lower for op in ['delete', 'remove', 'archive', 'destroy', 'purge']):
        return 'delete'

    # Check for toggle operations (not destructive)
    elif any(op in tool_lower for op in ['toggle', 'enable', 'disable', 'activate', 'deactivate']):
        return 'toggle'

    # Check for create/edit operations (not destructive)
    elif any(op in tool_lower for op in ['create', 'update', 'edit', 'modify', 'add', 'insert', 'append', 'set', 'write']):
        return 'create_edit'

    else:
        return 'other'

def extract_thinking_block(content):
    """Extract thinking block from assistant message."""
    if not content:
        return None, None
    match = re.search(r'(<thinking>.*?</thinking>)', content, re.DOTALL)
    if match:
        thinking_xml = match.group(1)
        thinking_text = re.search(r'<thinking>(.*?)</thinking>', thinking_xml, re.DOTALL).group(1).strip()
        try:
            return json.loads(thinking_text), thinking_xml
        except json.JSONDecodeError:
            return None, None
    return None, None

def process_file(input_filepath, output_filepath):
    """Process a single file and create v1.6 with fixes."""
    stats = {
        'total': 0,
        'delete_to_text': 0,
        'toggle_boosted': 0,
        'create_edit_boosted': 0,
        'other_boosted': 0,
        'unchanged': 0
    }

    processed_examples = []

    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            stats['total'] += 1
            data = json.loads(line)

            # Find assistant message
            assistant_msg = None
            for msg in data['conversations']:
                if msg.get('role') == 'assistant':
                    assistant_msg = msg
                    break

            if not assistant_msg or not assistant_msg.get('tool_calls'):
                processed_examples.append(data)
                stats['unchanged'] += 1
                continue

            tool_call = assistant_msg['tool_calls'][0]
            tool_name = tool_call['function']['name']
            operation_type = get_operation_type(tool_name)

            # Check if this needs fixing (risky + low confidence)
            thinking_json, thinking_xml = extract_thinking_block(assistant_msg.get('content', ''))

            if thinking_json:
                is_risky = thinking_json.get('assessment', {}).get('risky', False)
                confidence = thinking_json.get('confidence', 1.0)

                if is_risky and confidence <= 0.7:
                    if operation_type == 'delete':
                        # DELETE: Convert to text-only with varied question
                        target = extract_target_from_tool_call(tool_name, tool_call['function']['arguments'])
                        question = random.choice(DELETE_QUESTIONS).format(target=target)

                        # Update thinking
                        thinking_json['plan'] = [
                            "Recognize this is a destructive operation",
                            "Ask user for explicit confirmation before proceeding",
                            "Wait for user response"
                        ]
                        thinking_json['confidence'] = 0.5
                        thinking_str = json.dumps(thinking_json, ensure_ascii=False)
                        new_content = f"<thinking>\n{thinking_str}\n</thinking>\n\n{question}"

                        assistant_msg['content'] = new_content
                        if 'tool_calls' in assistant_msg:
                            del assistant_msg['tool_calls']

                        stats['delete_to_text'] += 1

                    elif operation_type == 'toggle':
                        # TOGGLE: Not destructive - boost confidence, keep tool
                        thinking_json['assessment']['risky'] = False
                        thinking_json['confidence'] = min(0.85, confidence + 0.2)
                        thinking_str = json.dumps(thinking_json, ensure_ascii=False)

                        # Preserve any text after thinking block
                        content_parts = assistant_msg['content'].split('</thinking>')
                        if len(content_parts) > 1:
                            text_after = content_parts[1]
                        else:
                            text_after = ''

                        new_content = f"<thinking>\n{thinking_str}\n</thinking>{text_after}"
                        assistant_msg['content'] = new_content
                        # Keep tool_calls

                        stats['toggle_boosted'] += 1

                    elif operation_type == 'create_edit':
                        # CREATE/EDIT: Not destructive - boost confidence, keep tool
                        thinking_json['assessment']['risky'] = False
                        thinking_json['confidence'] = min(0.85, confidence + 0.2)
                        thinking_str = json.dumps(thinking_json, ensure_ascii=False)

                        # Preserve any text after thinking block
                        content_parts = assistant_msg['content'].split('</thinking>')
                        if len(content_parts) > 1:
                            text_after = content_parts[1]
                        else:
                            text_after = ''

                        new_content = f"<thinking>\n{thinking_str}\n</thinking>{text_after}"
                        assistant_msg['content'] = new_content
                        # Keep tool_calls

                        stats['create_edit_boosted'] += 1

                    else:
                        # OTHER: Unknown type - boost confidence to be safe
                        thinking_json['assessment']['risky'] = False
                        thinking_json['confidence'] = min(0.85, confidence + 0.2)
                        thinking_str = json.dumps(thinking_json, ensure_ascii=False)

                        content_parts = assistant_msg['content'].split('</thinking>')
                        if len(content_parts) > 1:
                            text_after = content_parts[1]
                        else:
                            text_after = ''

                        new_content = f"<thinking>\n{thinking_str}\n</thinking>{text_after}"
                        assistant_msg['content'] = new_content
                        # Keep tool_calls

                        stats['other_boosted'] += 1
                else:
                    stats['unchanged'] += 1
            else:
                stats['unchanged'] += 1

            processed_examples.append(data)

    # Write output
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for example in processed_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    return stats

def main():
    """Process all datasets with version bump v1.5 → v1.6."""
    datasets_dir = Path('Datasets/tools_datasets/thinking')

    files_to_process = [
        {
            'manager': 'agentManager',
            'input': datasets_dir / 'agentManager/tools_v1.5.jsonl',
            'output': datasets_dir / 'agentManager/tools_v1.6.jsonl'
        },
        {
            'manager': 'contentManager',
            'input': datasets_dir / 'contentManager/tools_v1.5.jsonl',
            'output': datasets_dir / 'contentManager/tools_v1.6.jsonl'
        },
        {
            'manager': 'memoryManager',
            'input': datasets_dir / 'memoryManager/tools_v1.5.1.jsonl',
            'output': datasets_dir / 'memoryManager/tools_v1.6.jsonl'
        },
        {
            'manager': 'vaultLibrarian',
            'input': datasets_dir / 'vaultLibrarian/tools_v1.5.jsonl',
            'output': datasets_dir / 'vaultLibrarian/tools_v1.6.jsonl'
        },
        {
            'manager': 'vaultManager',
            'input': datasets_dir / 'vaultManager/tools_v1.5.jsonl',
            'output': datasets_dir / 'vaultManager/tools_v1.6.jsonl'
        },
    ]

    print("=" * 80)
    print("VERSION BUMP: v1.5 → v1.6 WITH OPERATION-AWARE FIXES")
    print("=" * 80)
    print()
    print("Strategy:")
    print("  • DELETE operations → Text-only with 25 question variations")
    print("  • TOGGLE operations → Boost confidence, set risky=false, KEEP tool calls")
    print("  • CREATE/EDIT operations → Boost confidence, set risky=false, KEEP tool calls")
    print("  • Questions are tool-specific (extract targets from tool calls)")
    print("=" * 80)
    print()

    total_stats = defaultdict(int)

    for file_info in files_to_process:
        manager = file_info['manager']
        input_file = file_info['input']
        output_file = file_info['output']

        if not input_file.exists():
            print(f"⚠️  Skipping {manager} - input file not found")
            continue

        print(f"📂 Processing {manager}...")
        print(f"   Input:  {input_file.name}")
        print(f"   Output: {output_file.name}")

        stats = process_file(input_file, output_file)

        print(f"   Total examples: {stats['total']}")
        print(f"   Fixes applied:")
        print(f"      - Delete → Text (varied): {stats['delete_to_text']}")
        print(f"      - Toggle → Boosted: {stats['toggle_boosted']}")
        print(f"      - Create/Edit → Boosted: {stats['create_edit_boosted']}")
        print(f"      - Other → Boosted: {stats['other_boosted']}")
        print(f"      - Unchanged: {stats['unchanged']}")
        print(f"   ✅ Written to: {output_file}")
        print()

        for key in stats:
            total_stats[key] += stats[key]

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total examples processed: {total_stats['total']}")
    print()
    print("Fixes applied:")
    print(f"  Delete → Text (25 variations):  {total_stats['delete_to_text']:5d}")
    print(f"  Toggle → Confidence boosted:    {total_stats['toggle_boosted']:5d}")
    print(f"  Create/Edit → Confidence boosted: {total_stats['create_edit_boosted']:5d}")
    print(f"  Other → Confidence boosted:     {total_stats['other_boosted']:5d}")
    print(f"  Unchanged (already correct):    {total_stats['unchanged']:5d}")
    print()
    print("✅ All v1.6 files created in their respective directories")
    print("✅ Original v1.5 files remain untouched")
    print("=" * 80)

if __name__ == '__main__':
    main()
