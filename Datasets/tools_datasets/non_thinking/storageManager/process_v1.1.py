#!/usr/bin/env python3
"""
Process vaultManager tools_v1.0.jsonl to add system prompts (v1.1)
According to CONTEXT_INJECTION_SPEC.md
"""

import json
import re
from typing import Dict, List, Tuple, Optional

# Workspace name mapping from spec
WORKSPACE_KEYWORDS = {
    r'budget|expense|finance': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    r'podcast|episode': ('Podcast Production', 'Podcast planning and production workflow', 'Podcast/'),
    r'research|paper|study': ('Research Hub', 'Academic research and study notes', 'Research/'),
    r'project|sprint|release': ('Project Management', 'Software project planning and tracking', 'Projects/'),
    r'recipe|cookbook|meal': ('Recipe Collection', 'Personal recipe library and meal planning', 'Recipes/'),
    r'workout|fitness|exercise': ('Fitness Tracker', 'Workout logs and fitness goals', 'Fitness/'),
    r'meeting|notes|agenda': ('Meeting Notes', 'Meeting notes and action items', 'Meetings/'),
    r'blog|content|post': ('Content Hub', 'Blog posts and content creation', 'Content/'),
    r'code|dev|programming': ('Development', 'Software development workspace', 'Dev/'),
    r'client|presentation': ('Client Work', 'Client projects and deliverables', 'Clients/'),
    r'learning|course|module': ('Learning Center', 'Educational content and courses', 'Courses/'),
    r'pet|health|vet': ('Pet Care', 'Pet health and care records', 'Pets/'),
    r'car|maintenance|vehicle': ('Vehicle Tracker', 'Vehicle maintenance logs', 'Vehicles/'),
    r'wellness|meditation': ('Wellness Journal', 'Mental health and wellness tracking', 'Wellness/'),
    r'agent|automation': ('Agent Workspace', 'AI agent configurations and workflows', 'Agents/'),
    r'ai|artificial intelligence': ('AI Research', 'AI research and experimentation', 'Research/AI/'),
    r'template|templates': ('Templates', 'Note templates and boilerplate', 'Templates/'),
    r'archive|old|completed': ('Archive', 'Archived and completed items', 'Archive/'),
    r'security|privacy|best practices': ('Security Docs', 'Security documentation and guidelines', 'Docs/Security/'),
    r'studio|boards|design': ('Creative Studio', 'Design and creative projects', 'Studios/'),
}


def extract_tool_call(conversations: List[Dict]) -> Optional[Dict]:
    """Extract tool_calls from assistant message"""
    for msg in conversations:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            return msg['tool_calls'][0]
    return None


def generate_workspace_info(context: Dict, user_message: str) -> Tuple[str, str, str]:
    """Generate workspace name, description, and root folder from context clues"""
    # Gather all text to search
    search_text = ' '.join([
        context.get('sessionDescription', ''),
        context.get('primaryGoal', ''),
        context.get('subgoal', ''),
        user_message
    ]).lower()

    # Try to match keywords
    for pattern, (name, desc, folder) in WORKSPACE_KEYWORDS.items():
        if re.search(pattern, search_text, re.IGNORECASE):
            return name, desc, folder

    # Default if no match
    return 'Personal Notes', 'General personal notes and organization', 'Notes/'


def build_system_prompt(session_id: str, workspace_id: str, workspace_name: str,
                        workspace_desc: str, root_folder: str) -> str:
    """Build system prompt according to spec"""

    # Session context section (always)
    if workspace_id == 'default':
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "default" (no specific workspace selected)

Include these in the "context" parameter of your tool calls.
NOTE: Use "default" as the workspaceId when no specific workspace context is needed.
</session_context>"""
    else:
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>"""

    # Available workspaces section (when not default)
    if workspace_id != 'default':
        workspaces_section = f"""<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""
        return f"{session_context}\n{workspaces_section}"

    return session_context


def process_example(example: Dict) -> Dict:
    """Process a single example to add system prompt"""
    conversations = example['conversations']

    # Extract tool call
    tool_call = extract_tool_call(conversations)
    if not tool_call:
        print(f"Warning: No tool call found in example")
        return example

    # Parse arguments
    try:
        args = json.loads(tool_call['function']['arguments'])
    except json.JSONDecodeError as e:
        print(f"Error parsing arguments: {e}")
        return example

    # Extract IDs from context
    context = args.get('context', {})
    session_id = context.get('sessionId', '')
    workspace_id = context.get('workspaceId', 'default')

    if not session_id:
        print(f"Warning: No sessionId found")
        return example

    # Get user message for context clues
    user_message = ''
    for msg in conversations:
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
            break

    # Generate workspace info
    workspace_name, workspace_desc, root_folder = generate_workspace_info(context, user_message)

    # Build system prompt
    system_prompt = build_system_prompt(
        session_id=session_id,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        workspace_desc=workspace_desc,
        root_folder=root_folder
    )

    # Insert system message at the beginning
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    # Create new conversations list with system message first
    new_conversations = [system_message] + conversations
    example['conversations'] = new_conversations

    return example


def main():
    input_file = '/home/user/Toolset-Training/Datasets/tools_datasets/vaultManager/tools_v1.0.jsonl'
    output_file = '/home/user/Toolset-Training/Datasets/tools_datasets/vaultManager/tools_v1.1.jsonl'

    processed_count = 0
    error_count = 0

    print(f"Processing {input_file}...")
    print(f"Output will be written to {output_file}")

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                example = json.loads(line.strip())
                processed_example = process_example(example)
                outfile.write(json.dumps(processed_example, ensure_ascii=False) + '\n')
                processed_count += 1

                if line_num % 100 == 0:
                    print(f"  Processed {line_num} examples...")

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
                # Write original example on error
                outfile.write(line)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total examples processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")

    # Validate output
    print("\nValidating output...")
    with open(output_file, 'r', encoding='utf-8') as f:
        system_prompt_count = 0
        for line in f:
            example = json.loads(line.strip())
            if example['conversations'][0].get('role') == 'system':
                system_prompt_count += 1

    print(f"Examples with system prompts: {system_prompt_count}/{processed_count}")
    if system_prompt_count == processed_count:
        print("✓ All examples have system prompts!")
    else:
        print(f"✗ Warning: {processed_count - system_prompt_count} examples missing system prompts")


if __name__ == '__main__':
    main()
