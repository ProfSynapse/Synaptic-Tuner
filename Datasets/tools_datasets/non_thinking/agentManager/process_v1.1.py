#!/usr/bin/env python3
"""
Process agentManager tools_v1.0.jsonl to add system prompts.
Creates tools_v1.1.jsonl with system context injection.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


# Workspace name mapping based on context keywords
WORKSPACE_MAPPINGS = {
    'budget': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'expense': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'finance': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'podcast': ('Podcast Production', 'Podcast episode planning and production', 'Podcast/'),
    'episode': ('Podcast Production', 'Podcast episode planning and production', 'Podcast/'),
    'research': ('Research Hub', 'Research notes and academic work', 'Research/'),
    'paper': ('Research Hub', 'Research notes and academic work', 'Research/'),
    'study': ('Research Hub', 'Research notes and academic work', 'Research/'),
    'project': ('Project Management', 'Project tracking and planning', 'Projects/'),
    'sprint': ('Project Management', 'Project tracking and planning', 'Projects/'),
    'release': ('Project Management', 'Project tracking and planning', 'Projects/'),
    'recipe': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'cookbook': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'meal': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'workout': ('Fitness Tracker', 'Workout logs and fitness goals', 'Fitness/'),
    'fitness': ('Fitness Tracker', 'Workout logs and fitness goals', 'Fitness/'),
    'exercise': ('Fitness Tracker', 'Workout logs and fitness goals', 'Fitness/'),
    'meeting': ('Meeting Notes', 'Meeting notes and agendas', 'Meetings/'),
    'agenda': ('Meeting Notes', 'Meeting notes and agendas', 'Meetings/'),
    'blog': ('Content Hub', 'Blog posts and content creation', 'Content/'),
    'content': ('Content Hub', 'Blog posts and content creation', 'Content/'),
    'post': ('Content Hub', 'Blog posts and content creation', 'Content/'),
    'code': ('Development', 'Software development projects', 'Dev/'),
    'dev': ('Development', 'Software development projects', 'Dev/'),
    'programming': ('Development', 'Software development projects', 'Dev/'),
    'client': ('Client Work', 'Client projects and deliverables', 'Clients/'),
    'presentation': ('Client Work', 'Client projects and deliverables', 'Clients/'),
    'learning': ('Learning Center', 'Course materials and learning resources', 'Courses/'),
    'course': ('Learning Center', 'Course materials and learning resources', 'Courses/'),
    'module': ('Learning Center', 'Course materials and learning resources', 'Courses/'),
    'pet': ('Pet Care', 'Pet health and care records', 'Pets/'),
    'health': ('Pet Care', 'Pet health and care records', 'Pets/'),
    'vet': ('Pet Care', 'Pet health and care records', 'Pets/'),
    'car': ('Vehicle Tracker', 'Vehicle maintenance and records', 'Vehicles/'),
    'maintenance': ('Vehicle Tracker', 'Vehicle maintenance and records', 'Vehicles/'),
    'vehicle': ('Vehicle Tracker', 'Vehicle maintenance and records', 'Vehicles/'),
    'wellness': ('Wellness Journal', 'Wellness and meditation tracking', 'Wellness/'),
    'meditation': ('Wellness Journal', 'Wellness and meditation tracking', 'Wellness/'),
    'agent': ('Agent Workspace', 'Custom agent configuration and automation', 'Agents/'),
    'automation': ('Agent Workspace', 'Custom agent configuration and automation', 'Agents/'),
    'playlist': ('Music Library', 'Music playlists and curation', 'Music/'),
    'music': ('Music Library', 'Music playlists and curation', 'Music/'),
    'review': ('Code Review', 'Code review and quality assurance', 'Dev/'),
    'solunox': ('Solunox Monitoring', 'Solunox system monitoring and diagnostics', 'Systems/'),
    'telemetry': ('System Monitoring', 'System telemetry and diagnostics', 'Systems/'),
    'studio': ('Creative Studio', 'Creative projects and workflows', 'Studio/'),
    'persona': ('Creative Studio', 'Creative projects and workflows', 'Studio/'),
}


def generate_workspace_info(context: Dict, user_message: str) -> Tuple[str, str, str]:
    """
    Generate workspace name, description, and root folder from context.

    Args:
        context: The context object from tool call arguments
        user_message: The user's message content

    Returns:
        Tuple of (workspace_name, description, root_folder)
    """
    # Gather text to search for keywords
    search_text = ' '.join([
        context.get('sessionDescription', ''),
        context.get('primaryGoal', ''),
        context.get('subgoal', ''),
        user_message
    ]).lower()

    # Find matching keyword
    for keyword, (name, desc, folder) in WORKSPACE_MAPPINGS.items():
        if keyword in search_text:
            return name, desc, folder

    # Default workspace
    return 'Personal Notes', 'General notes and personal content', 'Notes/'


def extract_agent_info(tool_call: Dict) -> Optional[Tuple[str, str]]:
    """
    Extract agent ID and name from tool call.

    Args:
        tool_call: The tool call object

    Returns:
        Tuple of (agent_id, agent_name) or None if no agent referenced
    """
    function_name = tool_call['function']['name']
    args = json.loads(tool_call['function']['arguments'])

    # listAgents doesn't reference a specific agent
    if 'listAgents' in function_name:
        return None

    # createAgent - uses name field (creating new agent)
    if 'createAgent' in function_name:
        agent_name = args.get('name', '')
        # Generate ID from name
        agent_id = f"agent_{agent_name.lower().replace(' ', '_').replace('-', '_')}"
        return agent_id, agent_name

    # updateAgent or deleteAgent - uses id field
    if 'updateAgent' in function_name or 'deleteAgent' in function_name:
        agent_id = args.get('id', '')
        # Derive name from ID (e.g., agent_code_reviewer -> Code Reviewer)
        agent_name = agent_id.replace('agent_', '').replace('_', ' ').title()
        return agent_id, agent_name

    # executePrompt - uses agent field directly as name
    if 'executePrompt' in function_name:
        agent_name = args.get('agent', '')
        # Generate ID from name
        agent_id = f"agent_{agent_name.lower().replace(' ', '_').replace('-', '_')}"
        return agent_id, agent_name

    return None


def build_system_prompt(
    session_id: str,
    workspace_id: str,
    workspace_name: str,
    workspace_desc: str,
    root_folder: str,
    agent_info: Optional[Tuple[str, str]] = None
) -> str:
    """
    Build the system prompt with appropriate sections.

    Args:
        session_id: The session ID
        workspace_id: The workspace ID
        workspace_name: The workspace name
        workspace_desc: The workspace description
        root_folder: The workspace root folder
        agent_info: Optional tuple of (agent_id, agent_name)

    Returns:
        The complete system prompt string
    """
    sections = []

    # Session context section (always included)
    if workspace_id == "default":
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

    sections.append(session_context)

    # Available workspaces section (skip for default sometimes)
    if workspace_id != "default":
        workspaces_section = f"""<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""
        sections.append(workspaces_section)

    # Available agents section (if agent is referenced)
    if agent_info:
        agent_id, agent_name = agent_info
        # Generate a description based on agent name
        agent_desc = f"Custom agent for {agent_name.lower()} operations"
        agents_section = f"""<available_agents>
The following custom agents are available:

- {agent_name} (id: "{agent_id}")
  {agent_desc}
</available_agents>"""
        sections.append(agents_section)

    return '\n'.join(sections)


def find_tool_call(conversations: List[Dict]) -> Optional[Dict]:
    """
    Find the first tool call in the conversation.

    Args:
        conversations: List of conversation messages

    Returns:
        The tool call object or None
    """
    for msg in conversations:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            return msg['tool_calls'][0]
    return None


def process_example(example: Dict) -> Dict:
    """
    Process a single example to add system prompt.

    Args:
        example: The input example

    Returns:
        The processed example with system prompt
    """
    conversations = example['conversations']

    # Find tool call
    tool_call = find_tool_call(conversations)
    if not tool_call:
        # No tool call found, return as-is
        return example

    # Extract context and IDs
    args = json.loads(tool_call['function']['arguments'])
    context = args.get('context', {})
    session_id = context.get('sessionId', '')
    workspace_id = context.get('workspaceId', 'default')

    # Get user message
    user_message = ''
    for msg in conversations:
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
            break

    # Generate workspace info
    workspace_name, workspace_desc, root_folder = generate_workspace_info(context, user_message)

    # Extract agent info if present
    agent_info = extract_agent_info(tool_call)

    # Build system prompt
    system_prompt = build_system_prompt(
        session_id=session_id,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        workspace_desc=workspace_desc,
        root_folder=root_folder,
        agent_info=agent_info
    )

    # Insert system message at beginning
    system_message = {'role': 'system', 'content': system_prompt}
    conversations.insert(0, system_message)

    return example


def main():
    """Main processing function."""
    input_file = '/home/user/Toolset-Training/Datasets/tools_datasets/agentManager/tools_v1.0.jsonl'
    output_file = '/home/user/Toolset-Training/Datasets/tools_datasets/agentManager/tools_v1.1.jsonl'

    processed_count = 0
    error_count = 0
    skipped_count = 0

    print(f"Processing {input_file}...")
    print(f"Output: {output_file}")
    print()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON
                example = json.loads(line.strip())

                # Check if already has system prompt
                if example['conversations'][0].get('role') == 'system':
                    skipped_count += 1
                    outfile.write(json.dumps(example) + '\n')
                    continue

                # Process example
                processed_example = process_example(example)

                # Write to output
                outfile.write(json.dumps(processed_example) + '\n')
                processed_count += 1

                # Progress update every 100 examples
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} examples...")

            except Exception as e:
                error_count += 1
                print(f"Error on line {line_num}: {e}")
                # Write original line to preserve data
                outfile.write(line)

    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total processed: {processed_count}")
    print(f"Skipped (already had system prompt): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Output file: {output_file}")
    print()

    # Validate a few examples
    print("Validating sample examples...")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i in range(min(3, processed_count)):
            line = f.readline()
            example = json.loads(line)

            # Check first message is system
            if example['conversations'][0]['role'] != 'system':
                print(f"  ⚠️  Example {i+1}: Missing system prompt!")
            else:
                print(f"  ✓  Example {i+1}: Has system prompt")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
