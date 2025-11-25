#!/usr/bin/env python3
"""
Process memoryManager tools_v1.0.jsonl to add system prompts.
Creates tools_v1.1.jsonl with session context and available workspaces.
"""

import json
import re
from pathlib import Path

# Workspace name generation heuristics (from spec)
WORKSPACE_KEYWORDS = {
    'budget': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'expense': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'finance': ('Budget Tracker', 'Monthly budget and expense tracking', 'Finance/'),
    'podcast': ('Podcast Production', 'Podcast episode planning and production', 'Podcast/'),
    'episode': ('Podcast Production', 'Podcast episode planning and production', 'Podcast/'),
    'research': ('Research Hub', 'Research notes and project tracking', 'Research/'),
    'paper': ('Research Hub', 'Research notes and project tracking', 'Research/'),
    'study': ('Research Hub', 'Research notes and project tracking', 'Research/'),
    'project': ('Project Management', 'Project planning and tracking', 'Projects/'),
    'sprint': ('Project Management', 'Project planning and tracking', 'Projects/'),
    'release': ('Project Management', 'Project planning and tracking', 'Projects/'),
    'recipe': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'cookbook': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'meal': ('Recipe Collection', 'Recipe management and meal planning', 'Recipes/'),
    'workout': ('Fitness Tracker', 'Workout and exercise tracking', 'Fitness/'),
    'fitness': ('Fitness Tracker', 'Workout and exercise tracking', 'Fitness/'),
    'exercise': ('Fitness Tracker', 'Workout and exercise tracking', 'Fitness/'),
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
    'pet': ('Pet Care', 'Pet health and care tracking', 'Pets/'),
    'health': ('Wellness Journal', 'Health and wellness tracking', 'Wellness/'),
    'vet': ('Pet Care', 'Pet health and care tracking', 'Pets/'),
    'car': ('Vehicle Tracker', 'Vehicle maintenance tracking', 'Vehicles/'),
    'maintenance': ('Vehicle Tracker', 'Vehicle maintenance tracking', 'Vehicles/'),
    'vehicle': ('Vehicle Tracker', 'Vehicle maintenance tracking', 'Vehicles/'),
    'wellness': ('Wellness Journal', 'Health and wellness tracking', 'Wellness/'),
    'meditation': ('Wellness Journal', 'Health and wellness tracking', 'Wellness/'),
    'habit': ('Habit Tracker', 'Daily habit tracking', 'Habits/'),
    'agent': ('Agent Workspace', 'Agent configuration and automation', 'Agents/'),
    'automation': ('Agent Workspace', 'Agent configuration and automation', 'Agents/'),
    'training': ('Training Materials', 'Training resources and documentation', 'Training/'),
    'portfolio': ('Portfolio', 'Portfolio projects and showcases', 'Portfolio/'),
    'interview': ('Career Development', 'Interview prep and job search', 'Career/'),
    'design': ('Design Projects', 'Design work and mockups', 'Design/'),
    'redesign': ('Design Projects', 'Design work and mockups', 'Design/'),
    'documentation': ('Documentation Hub', 'Technical documentation', 'Docs/'),
    'docs': ('Documentation Hub', 'Technical documentation', 'Docs/'),
    'guide': ('Documentation Hub', 'Technical documentation', 'Docs/'),
    'freelance': ('Freelance Business', 'Freelance projects and clients', 'Freelance/'),
    'game': ('Game Collection', 'Board games and game night planning', 'Games/'),
    'board': ('Game Collection', 'Board games and game night planning', 'Games/'),
    'campaign': ('RPG Campaign', 'D&D and RPG campaign management', 'RPG/'),
    'dnd': ('RPG Campaign', 'D&D and RPG campaign management', 'RPG/'),
    'd&d': ('RPG Campaign', 'D&D and RPG campaign management', 'RPG/'),
    'data': ('Data Analysis', 'Data analysis and reporting', 'Data/'),
    'analysis': ('Data Analysis', 'Data analysis and reporting', 'Data/'),
    'ml': ('Machine Learning', 'ML experiments and model development', 'ML/'),
    'machine learning': ('Machine Learning', 'ML experiments and model development', 'ML/'),
    'deployment': ('Production Deploy', 'Deployment and release management', 'Deploy/'),
    'production': ('Production Deploy', 'Deployment and release management', 'Deploy/'),
}


def generate_workspace_info(context, conversations):
    """Generate workspace name, description, and root folder from context clues."""

    # Gather all text to search
    search_text = ""
    if context:
        search_text += " " + context.get("sessionDescription", "")
        search_text += " " + context.get("primaryGoal", "")
        search_text += " " + context.get("subgoal", "")
        search_text += " " + context.get("toolContext", "")

    # Add user message
    for conv in conversations:
        if conv.get("role") == "user":
            search_text += " " + conv.get("content", "")

    search_text = search_text.lower()

    # Try to match keywords
    for keyword, (name, desc, folder) in WORKSPACE_KEYWORDS.items():
        if keyword in search_text:
            return name, desc, folder

    # Default fallback
    return "Personal Notes", "General notes and information", "Notes/"


def build_system_prompt(session_id, workspace_id, workspace_name, workspace_desc, root_folder):
    """Build the system prompt with session_context and available_workspaces sections."""

    # Session context section (always present)
    if workspace_id == "default":
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "default" (no specific workspace selected)

Include these in the "context" parameter of your tool calls.
NOTE: Use "default" as the workspaceId when no specific workspace context is needed.
</session_context>"""
        # For default, we may or may not include available_workspaces
        # Let's skip it for simplicity in default cases
        return session_context
    else:
        session_context = f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>"""

        # Available workspaces section
        available_workspaces = f"""<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""

        return session_context + "\n" + available_workspaces


def extract_tool_call_info(conversations):
    """Extract tool call arguments from assistant message."""
    for msg in conversations:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_call = tool_calls[0]  # Get first tool call
                function = tool_call.get("function", {})
                args_str = function.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    return args
                except json.JSONDecodeError:
                    return None
    return None


def process_example(example):
    """Process a single example to add system prompt."""
    conversations = example.get("conversations", [])

    # Extract tool call arguments
    args = extract_tool_call_info(conversations)
    if not args:
        # No tool call found, return as-is
        return example

    # Extract IDs from context
    context = args.get("context", {})
    session_id = context.get("sessionId", "")
    workspace_id = context.get("workspaceId", "default")

    # Generate workspace info
    workspace_name, workspace_desc, root_folder = generate_workspace_info(context, conversations)

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

    # Update example
    example["conversations"] = new_conversations

    return example


def main():
    """Main processing function."""
    input_file = Path(__file__).parent / "tools_v1.0.jsonl"
    output_file = Path(__file__).parent / "tools_v1.1.jsonl"

    print(f"Processing: {input_file}")
    print(f"Output to: {output_file}")

    processed_count = 0
    error_count = 0
    errors = []

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON
                example = json.loads(line.strip())

                # Process example
                processed_example = process_example(example)

                # Write to output
                outfile.write(json.dumps(processed_example, ensure_ascii=False) + "\n")

                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} examples...")

            except Exception as e:
                error_count += 1
                error_msg = f"Line {line_num}: {str(e)}"
                errors.append(error_msg)
                print(f"ERROR: {error_msg}")

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total examples processed: {processed_count}")
    print(f"Total errors: {error_count}")

    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    print(f"\nOutput file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
