#!/usr/bin/env python3
"""
Process vaultLibrarian tools dataset to add system prompts.
Converts tools_v1.0.jsonl to tools_v1.1.jsonl with context injection.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Workspace name generation mapping from CONTEXT_INJECTION_SPEC.md
WORKSPACE_KEYWORDS = {
    'budget|expense|finance': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'podcast|episode': ('Podcast Production', 'Podcast episode planning and production', 'Podcast/'),
    'research|paper|study': ('Research Hub', 'Research notes and academic papers', 'Research/'),
    'project|sprint|release': ('Project Management', 'Project tracking and planning', 'Projects/'),
    'recipe|cookbook|meal': ('Recipe Collection', 'Cooking recipes and meal planning', 'Recipes/'),
    'workout|fitness|exercise': ('Fitness Tracker', 'Workout logs and fitness goals', 'Fitness/'),
    'meeting|notes|agenda': ('Meeting Notes', 'Meeting minutes and agendas', 'Meetings/'),
    'blog|content|post': ('Content Hub', 'Blog posts and content creation', 'Content/'),
    'code|dev|programming': ('Development', 'Software development projects', 'Dev/'),
    'client|presentation': ('Client Work', 'Client projects and presentations', 'Clients/'),
    'learning|course|module': ('Learning Center', 'Educational courses and learning materials', 'Courses/'),
    'pet|health|vet': ('Pet Care', 'Pet health records and care information', 'Pets/'),
    'car|maintenance|vehicle': ('Vehicle Tracker', 'Vehicle maintenance and records', 'Vehicles/'),
    'wellness|meditation': ('Wellness Journal', 'Wellness activities and meditation', 'Wellness/'),
    'agent|automation': ('Agent Workspace', 'AI agent configurations and automations', 'Agents/'),
    'archive|old|quarter': ('Archives', 'Archived documents and historical records', 'Archives/'),
    'migration|database': ('Database Projects', 'Database and data migration projects', 'Database/'),
    'interface|typescript|code': ('Development', 'Software development projects', 'Dev/'),
    'memory|discussion': ('Project Archives', 'Historical project discussions and context', 'Archives/'),
}


def extract_keywords(text: str) -> str:
    """Extract keywords from text for workspace name generation."""
    if not text:
        return ""
    return text.lower()


def generate_workspace_info(context: Dict, user_message: str) -> Tuple[str, str, str]:
    """
    Generate workspace name, description, and root folder from context clues.

    Args:
        context: Context object from tool call
        user_message: User message content

    Returns:
        Tuple of (workspace_name, description, root_folder)
    """
    # Combine all text sources for keyword matching
    text_sources = [
        context.get('sessionDescription', ''),
        context.get('primaryGoal', ''),
        context.get('subgoal', ''),
        user_message
    ]
    combined_text = ' '.join(text_sources).lower()

    # Try to match keywords to workspace types
    for pattern, (name, desc, root) in WORKSPACE_KEYWORDS.items():
        if re.search(pattern, combined_text):
            return name, desc, root

    # Default fallback
    return 'Personal Notes', 'General notes and personal documentation', 'Notes/'


def build_system_prompt(
    session_id: str,
    workspace_id: str,
    workspace_name: str,
    workspace_desc: str,
    root_folder: str
) -> str:
    """
    Build the system prompt with session context and available workspaces.

    Args:
        session_id: Session ID from context
        workspace_id: Workspace ID from context
        workspace_name: Generated workspace name
        workspace_desc: Generated workspace description
        root_folder: Generated root folder

    Returns:
        Complete system prompt string
    """
    # Always include session context
    if workspace_id == "default":
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "default" (no specific workspace selected)

Include these in the "context" parameter of your tool calls.
NOTE: Use "default" as the workspaceId when no specific workspace context is needed.
</session_context>"""
        return session_context
    else:
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>"""

        # Add available workspaces section
        workspaces_section = f"""<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""

        return session_context + "\n" + workspaces_section


def extract_tool_call_context(conversations: List[Dict]) -> Optional[Tuple[Dict, str]]:
    """
    Extract context from the first tool call in conversations.

    Args:
        conversations: List of conversation messages

    Returns:
        Tuple of (context_dict, user_message) or None if no tool call found
    """
    user_message = ""

    # Find user message
    for msg in conversations:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # Find assistant message with tool calls
    for msg in conversations:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Get first tool call
                tool_call = tool_calls[0]
                function_data = tool_call.get("function", {})
                arguments_str = function_data.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str)
                    context = arguments.get("context", {})
                    return context, user_message
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse arguments JSON: {arguments_str[:100]}")
                    return None

    return None


def process_example(example: Dict) -> Dict:
    """
    Process a single example to add system prompt.

    Args:
        example: Original example dictionary

    Returns:
        Updated example with system prompt inserted
    """
    conversations = example.get("conversations", [])

    # Check if system message already exists
    if conversations and conversations[0].get("role") == "system":
        print("Warning: System message already exists, skipping")
        return example

    # Extract context from tool call
    result = extract_tool_call_context(conversations)
    if not result:
        print("Warning: No tool call found or failed to extract context")
        return example

    context, user_message = result

    # Extract IDs
    session_id = context.get("sessionId", "")
    workspace_id = context.get("workspaceId", "default")

    if not session_id:
        print("Warning: No sessionId found in context")
        return example

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

    # Insert system message at beginning
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    conversations.insert(0, system_message)

    return example


def main():
    """Main processing function."""
    script_dir = Path(__file__).parent
    input_file = script_dir / "tools_v1.0.jsonl"
    output_file = script_dir / "tools_v1.1.jsonl"

    print(f"Processing: {input_file}")
    print(f"Output to: {output_file}")

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    processed_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                processed_example = process_example(example)

                # Write to output
                outfile.write(json.dumps(processed_example, ensure_ascii=False) + "\n")
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} examples...")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total examples processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output file: {output_file}")
    print(f"Output file size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify output by reading a few examples
    print("\n" + "="*60)
    print("Sample verification (first 2 examples):")
    print("="*60)
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            example = json.loads(line)
            conversations = example.get("conversations", [])
            if conversations and conversations[0].get("role") == "system":
                print(f"\nExample {i+1}: System prompt added ✓")
                print(f"  First 100 chars: {conversations[0]['content'][:100]}...")
            else:
                print(f"\nExample {i+1}: No system prompt found ✗")


if __name__ == "__main__":
    main()
