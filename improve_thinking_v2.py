#!/usr/bin/env python3
"""
Improved thinking block enhancement for vaultManager dataset.
More sophisticated improvements with better context understanding.
"""

import json
import re
from pathlib import Path

def extract_path_from_content(content):
    """Extract file/folder path from user content or tool call."""
    # Look for common path patterns
    path_match = re.search(r'(?:path|move|create|delete|open).*?([A-Z][A-Za-z0-9_/-]+(?:\.md)?)', content, re.IGNORECASE)
    if path_match:
        return path_match.group(1)
    return None

def improve_goal(goal, tool_name, user_content, tool_args):
    """Generate more specific, actionable goal."""
    # If goal is already specific and long enough, keep it
    if len(goal) > 40 and '/' in goal:
        return goal

    # Extract paths from tool arguments
    source_path = tool_args.get('path', tool_args.get('sourcePath', ''))
    dest_path = tool_args.get('newPath', tool_args.get('targetPath', ''))

    if 'delete' in tool_name.lower():
        if source_path:
            return f"Delete {source_path} from vault"
        return "Remove unwanted file from vault to reduce clutter"

    elif 'moveNote' in tool_name:
        if source_path and dest_path:
            return f"Move {source_path} to {dest_path}"
        return "Relocate note to better organizational location"

    elif 'moveFolder' in tool_name:
        if source_path and dest_path:
            return f"Relocate folder from {source_path} to {dest_path}"
        return "Reorganize folder structure for better workspace organization"

    elif 'createFolder' in tool_name:
        if source_path:
            return f"Create new folder at {source_path}"
        return "Establish new folder for organizational structure"

    elif 'duplicate' in tool_name.lower():
        if source_path and dest_path:
            return f"Duplicate {source_path} to {dest_path}"
        return "Create copy of file for backup or new usage"

    elif 'open' in tool_name.lower():
        if source_path:
            return f"Open and review {source_path}"
        return "Access file for reading and reference"

    elif 'list' in tool_name.lower():
        if source_path:
            return f"List contents of {source_path} directory"
        return "View folder contents to understand structure"

    # Fallback: keep original but trim if too long
    return goal[:80] if len(goal) > 80 else goal

def improve_memory(memory, tool_name, workspace_id):
    """Enhance memory with richer context."""
    # If memory is already rich (>100 chars with good context), keep it
    if len(memory) > 100 and any(word in memory.lower() for word in ['previously', 'working on', 'organizing', 'created']):
        return memory

    # Build enhanced memory
    enhanced_parts = []

    # Add what user is doing and why
    if 'delete' in tool_name.lower():
        enhanced_parts.append("User is cleaning up vault by removing unused or obsolete files.")
        enhanced_parts.append("This is part of ongoing vault maintenance to keep workspace organized.")
    elif 'move' in tool_name.lower():
        enhanced_parts.append("User is reorganizing vault structure by relocating files to more appropriate locations.")
        enhanced_parts.append("Previous organization work established folder hierarchy.")
    elif 'create' in tool_name.lower():
        enhanced_parts.append("User is building out organizational structure with new folders.")
        enhanced_parts.append("Part of systematic vault setup for improved workflow.")
    elif 'duplicate' in tool_name.lower():
        enhanced_parts.append("User is creating copy for backup or template purposes.")
        enhanced_parts.append("Duplication preserves original while enabling new usage.")
    elif 'list' in tool_name.lower():
        enhanced_parts.append("User is exploring vault structure to understand current organization.")
    elif 'open' in tool_name.lower():
        enhanced_parts.append("User needs to review or reference existing documentation.")

    # Add workspace context
    if workspace_id and workspace_id != 'default':
        enhanced_parts.append(f"Operating within specific workspace context.")
    else:
        enhanced_parts.append("Operating in default workspace without specific project context.")

    # Preserve original memory if it has unique info
    if memory and len(memory) > 20:
        if not any(part.lower() in memory.lower() for part in enhanced_parts):
            enhanced_parts.insert(0, memory)

    return ' '.join(enhanced_parts)

def generate_requirements(tool_name, tool_args):
    """Generate verification prerequisites distinct from execution plan."""
    source_path = tool_args.get('path', tool_args.get('sourcePath', ''))
    dest_path = tool_args.get('newPath', tool_args.get('targetPath', ''))

    if 'delete' in tool_name.lower():
        return [
            f"Verify {source_path} exists and is accessible",
            "Confirm file is truly unused with no critical dependencies",
            "Ensure user intent is to permanently remove"
        ]

    elif 'moveNote' in tool_name or 'moveFolder' in tool_name:
        reqs = [
            f"Verify source path {source_path} exists"
        ]
        if dest_path:
            parent = '/'.join(dest_path.split('/')[:-1])
            if parent:
                reqs.append(f"Confirm destination folder {parent} exists")
            reqs.append("Ensure no naming conflicts at destination")
        return reqs

    elif 'createFolder' in tool_name:
        if source_path:
            parent = '/'.join(source_path.split('/')[:-1])
            reqs = []
            if parent:
                reqs.append(f"Verify parent path {parent} exists")
            reqs.append(f"Ensure no existing folder named {source_path.split('/')[-1]}")
            return reqs
        return [
            "Verify parent folder exists",
            "Ensure no naming conflicts"
        ]

    elif 'duplicate' in tool_name.lower():
        reqs = []
        if source_path:
            reqs.append(f"Verify source {source_path} exists and is accessible")
        if dest_path:
            reqs.append(f"Confirm target {dest_path} doesn't already exist")
            parent = '/'.join(dest_path.split('/')[:-1])
            if parent:
                reqs.append(f"Ensure destination folder {parent} exists")
        return reqs if reqs else [
            "Verify source file exists",
            "Confirm destination is available",
            "Ensure no conflicts"
        ]

    elif 'list' in tool_name.lower():
        return [
            f"Verify folder path {source_path if source_path else 'target'} exists",
            "Confirm read permissions available"
        ]

    elif 'open' in tool_name.lower():
        return [
            f"Verify file {source_path if source_path else 'target'} exists",
            "Confirm file is accessible for reading"
        ]

    # Fallback
    return [
        "Verify operation prerequisites met",
        "Ensure no conflicts or errors"
    ]

def generate_plan(tool_name, tool_args):
    """Generate step-by-step execution plan."""
    is_risky = 'delete' in tool_name.lower() or 'move' in tool_name.lower()

    source_path = tool_args.get('path', tool_args.get('sourcePath', ''))
    dest_path = tool_args.get('newPath', tool_args.get('targetPath', ''))

    if 'delete' in tool_name.lower():
        return [
            f"Verify {source_path if source_path else 'target file'} exists and is deletable",
            "Execute deletion operation",
            "Confirm removal from vault structure"
        ]

    elif 'moveNote' in tool_name or 'moveFolder' in tool_name:
        plan = []
        if is_risky:
            plan.append(f"Verify source {source_path if source_path else 'path'} exists")
            plan.append(f"Check destination {dest_path if dest_path else 'path'} is available")
        plan.append("Execute move operation")
        plan.append("Confirm relocation successful")
        return plan

    elif 'createFolder' in tool_name:
        return [
            f"Create folder at {source_path if source_path else 'specified path'}",
            "Confirm folder structure ready for use"
        ]

    elif 'duplicate' in tool_name.lower():
        return [
            f"Duplicate file from {source_path if source_path else 'source'} to {dest_path if dest_path else 'destination'}",
            "Confirm copy created successfully",
            "Prepare new file for use"
        ]

    elif 'list' in tool_name.lower():
        return [
            f"Retrieve contents of {source_path if source_path else 'folder'}",
            "Present organized file/folder listing"
        ]

    elif 'open' in tool_name.lower():
        return [
            f"Access file {source_path if source_path else 'at specified path'}",
            "Load content for reading"
        ]

    # Fallback
    return [
        "Execute requested operation",
        "Confirm success"
    ]

def improve_thinking_block(thinking_str, tool_name, user_content, tool_args, workspace_id):
    """Comprehensive thinking block improvement."""
    try:
        thinking = json.loads(thinking_str)
    except:
        return thinking_str

    # Determine operation type
    is_delete = 'delete' in tool_name.lower()
    is_move = 'move' in tool_name.lower()
    is_create = 'create' in tool_name.lower()
    is_duplicate = 'duplicate' in tool_name.lower()
    is_list = 'list' in tool_name.lower()
    is_open = 'open' in tool_name.lower()

    # Fix risky assessment
    thinking['assessment']['risky'] = is_delete or is_move

    # Fix complex (single file/folder operations are not complex)
    thinking['assessment']['complex'] = False

    # Fix confidence based on risk
    if is_delete:
        # Deletes: 0.3-0.5 (low confidence, high risk)
        thinking['confidence'] = round(0.35 + (hash(user_content) % 15) / 100, 2)
    elif is_move:
        # Moves: 0.5-0.7 (medium confidence, medium risk)
        thinking['confidence'] = round(0.58 + (hash(user_content) % 12) / 100, 2)
    elif is_create:
        # Creates: 0.85-0.95 (high confidence, low risk)
        thinking['confidence'] = round(0.88 + (hash(user_content) % 7) / 100, 2)
    elif is_duplicate:
        # Duplicates: 0.85-0.95 (high confidence, low risk)
        thinking['confidence'] = round(0.89 + (hash(user_content) % 6) / 100, 2)
    elif is_list or is_open:
        # Read operations: 0.90-0.95 (very high confidence)
        thinking['confidence'] = round(0.91 + (hash(user_content) % 4) / 100, 2)
    else:
        # Default: 0.80-0.90
        thinking['confidence'] = round(0.83 + (hash(user_content) % 7) / 100, 2)

    # Improve goal
    thinking['goal'] = improve_goal(
        thinking.get('goal', ''),
        tool_name,
        user_content,
        tool_args
    )

    # Improve memory
    thinking['memory'] = improve_memory(
        thinking.get('memory', ''),
        tool_name,
        workspace_id
    )

    # Improve requirements (always replace to ensure quality)
    thinking['requirements'] = generate_requirements(tool_name, tool_args)

    # Improve plan (always replace to ensure quality)
    thinking['plan'] = generate_plan(tool_name, tool_args)

    return json.dumps(thinking, indent=2)


def process_jsonl_file(input_path, output_path):
    """Process JSONL file and improve thinking blocks."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    processed = 0
    improved = 0
    errors = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            try:
                if not line.strip():
                    continue

                data = json.loads(line)
                processed += 1

                # Get workspace ID
                workspace_id = 'default'
                for conv in data.get('conversations', []):
                    if conv.get('role') == 'system':
                        if 'workspaceId:' in conv.get('content', ''):
                            ws_match = re.search(r'workspaceId:\s*"([^"]+)"', conv['content'])
                            if ws_match:
                                workspace_id = ws_match.group(1)

                # Get user content
                user_content = ''
                for conv in data.get('conversations', []):
                    if conv.get('role') == 'user':
                        user_content = conv.get('content', '')
                        break

                # Process assistant messages
                for conv in data.get('conversations', []):
                    if conv.get('role') == 'assistant':
                        content = conv.get('content', '')

                        # Extract thinking block
                        thinking_match = re.search(r'<thinking>\s*(\{.*?\})\s*</thinking>',
                                                   content, re.DOTALL)
                        if thinking_match:
                            old_thinking = thinking_match.group(1)

                            # Get tool name and args
                            tool_name = ''
                            tool_args = {}
                            if 'tool_calls' in conv:
                                for tc in conv['tool_calls']:
                                    if 'function' in tc:
                                        tool_name = tc['function'].get('name', '')
                                        args_str = tc['function'].get('arguments', '{}')
                                        try:
                                            full_args = json.loads(args_str)
                                            tool_args = {k: v for k, v in full_args.items()
                                                         if k not in ['context']}
                                        except:
                                            pass

                            # Improve thinking
                            new_thinking = improve_thinking_block(
                                old_thinking,
                                tool_name,
                                user_content,
                                tool_args,
                                workspace_id
                            )

                            if new_thinking != old_thinking:
                                # Replace in content
                                conv['content'] = content.replace(
                                    thinking_match.group(0),
                                    f'<thinking>\n{new_thinking}\n</thinking>'
                                )
                                improved += 1

                # Write improved data
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

                if processed % 100 == 0:
                    print(f"Processed {processed} examples, improved {improved}...")

            except Exception as e:
                errors += 1
                print(f"Error on line {line_num}: {e}")
                # Write original line on error
                fout.write(line)

    print(f"\nComplete!")
    print(f"Total processed: {processed}")
    print(f"Total improved: {improved}")
    print(f"Total errors: {errors}")


if __name__ == '__main__':
    input_path = '/mnt/f/Code/Toolset-Training/Datasets/tools_datasets/thinking/vaultManager/tools_v1.3.jsonl'
    output_path = '/mnt/f/Code/Toolset-Training/Datasets/tools_datasets/thinking/vaultManager/tools_v1.4.jsonl'

    print("Starting comprehensive improvement process...")
    print("Improving: goal, memory, requirements, plan, confidence, risk assessment")
    print()
    process_jsonl_file(input_path, output_path)
