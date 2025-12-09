#!/usr/bin/env python3
"""
Process contentManager tools_v1.0.jsonl to add system prompts.
Implements the Context Injection Specification.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Workspace name mapping from CONTEXT_INJECTION_SPEC.md
WORKSPACE_KEYWORDS = {
    "budget": ("Budget Tracker", "Monthly budget and expense tracking", "Finance/"),
    "expense": ("Budget Tracker", "Monthly budget and expense tracking", "Finance/"),
    "finance": ("Budget Tracker", "Monthly budget and expense tracking", "Finance/"),
    "podcast": ("Podcast Production", "Podcast episode planning and production", "Podcast/"),
    "episode": ("Podcast Production", "Podcast episode planning and production", "Podcast/"),
    "research": ("Research Hub", "Academic research and study notes", "Research/"),
    "paper": ("Research Hub", "Academic research and study notes", "Research/"),
    "study": ("Research Hub", "Academic research and study notes", "Research/"),
    "project": ("Project Management", "Project planning and tracking", "Projects/"),
    "sprint": ("Project Management", "Project planning and tracking", "Projects/"),
    "release": ("Project Management", "Project planning and tracking", "Projects/"),
    "recipe": ("Recipe Collection", "Cooking recipes and meal planning", "Recipes/"),
    "cookbook": ("Recipe Collection", "Cooking recipes and meal planning", "Recipes/"),
    "meal": ("Recipe Collection", "Cooking recipes and meal planning", "Recipes/"),
    "workout": ("Fitness Tracker", "Workout and fitness tracking", "Fitness/"),
    "fitness": ("Fitness Tracker", "Workout and fitness tracking", "Fitness/"),
    "exercise": ("Fitness Tracker", "Workout and fitness tracking", "Fitness/"),
    "meeting": ("Meeting Notes", "Meeting notes and agendas", "Meetings/"),
    "notes": ("Meeting Notes", "Meeting notes and agendas", "Meetings/"),
    "agenda": ("Meeting Notes", "Meeting notes and agendas", "Meetings/"),
    "blog": ("Content Hub", "Blog posts and content creation", "Content/"),
    "content": ("Content Hub", "Blog posts and content creation", "Content/"),
    "post": ("Content Hub", "Blog posts and content creation", "Content/"),
    "code": ("Development", "Software development and coding", "Dev/"),
    "dev": ("Development", "Software development and coding", "Dev/"),
    "programming": ("Development", "Software development and coding", "Dev/"),
    "client": ("Client Work", "Client projects and deliverables", "Clients/"),
    "presentation": ("Client Work", "Client projects and deliverables", "Clients/"),
    "learning": ("Learning Center", "Educational content and courses", "Courses/"),
    "course": ("Learning Center", "Educational content and courses", "Courses/"),
    "module": ("Learning Center", "Educational content and courses", "Courses/"),
    "pet": ("Pet Care", "Pet health and care tracking", "Pets/"),
    "health": ("Pet Care", "Pet health and care tracking", "Pets/"),
    "vet": ("Pet Care", "Pet health and care tracking", "Pets/"),
    "car": ("Vehicle Tracker", "Vehicle maintenance tracking", "Vehicles/"),
    "maintenance": ("Vehicle Tracker", "Vehicle maintenance tracking", "Vehicles/"),
    "vehicle": ("Vehicle Tracker", "Vehicle maintenance tracking", "Vehicles/"),
    "wellness": ("Wellness Journal", "Personal wellness and meditation", "Wellness/"),
    "meditation": ("Wellness Journal", "Personal wellness and meditation", "Wellness/"),
    "agent": ("Agent Workspace", "Agent automation and workflows", "Agents/"),
    "automation": ("Agent Workspace", "Agent automation and workflows", "Agents/"),
}

DEFAULT_WORKSPACE = ("Personal Notes", "General notes and documentation", "Notes/")


def extract_keywords(text: str) -> List[str]:
    """Extract lowercase keywords from text."""
    if not text:
        return []
    # Convert to lowercase and extract words
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def generate_workspace_info(context: Dict, user_message: str) -> Tuple[str, str, str]:
    """
    Generate workspace name, description, and root folder from context clues.

    Returns: (workspace_name, description, root_folder)
    """
    # Gather all text sources
    session_desc = context.get("sessionDescription", "")
    primary_goal = context.get("primaryGoal", "")
    session_memory = context.get("sessionMemory", "")
    tool_context = context.get("toolContext", "")

    # Combine all sources
    all_text = f"{session_desc} {primary_goal} {session_memory} {tool_context} {user_message}"
    keywords = extract_keywords(all_text)

    # Check for keyword matches
    for keyword in keywords:
        if keyword in WORKSPACE_KEYWORDS:
            return WORKSPACE_KEYWORDS[keyword]

    # Default if no match
    return DEFAULT_WORKSPACE


def build_system_prompt(
    session_id: str,
    workspace_id: str,
    workspace_name: str = None,
    workspace_desc: str = None,
    root_folder: str = None,
    include_workspaces: bool = True
) -> str:
    """Build the system prompt according to spec."""

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

    # Available workspaces section (optional for default, always for specific workspace)
    if include_workspaces and workspace_id != "default" and workspace_name:
        workspaces_section = f"""<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""
        sections.append(workspaces_section)

    return "\n".join(sections)


def extract_tool_call_info(conversations: List[Dict]) -> Optional[Dict]:
    """Extract tool call information from conversations."""
    for msg in conversations:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call = msg["tool_calls"][0]
            function = tool_call.get("function", {})
            args_str = function.get("arguments", "{}")

            try:
                args = json.loads(args_str)
                return {
                    "arguments": args,
                    "function_name": function.get("name", "")
                }
            except json.JSONDecodeError:
                return None
    return None


def process_example(example: Dict) -> Dict:
    """Process a single example to add system prompt."""
    conversations = example.get("conversations", [])

    # Skip if already has system message
    if conversations and conversations[0].get("role") == "system":
        return example

    # Extract tool call info
    tool_info = extract_tool_call_info(conversations)
    if not tool_info:
        # No tool call found, return as-is
        return example

    args = tool_info["arguments"]
    context = args.get("context", {})

    # Extract IDs
    session_id = context.get("sessionId", "")
    workspace_id = context.get("workspaceId", "default")

    if not session_id:
        # No session ID, can't create proper context
        return example

    # Get user message for context
    user_message = ""
    for msg in conversations:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # Generate workspace info
    workspace_name, workspace_desc, root_folder = generate_workspace_info(context, user_message)

    # Build system prompt
    system_prompt = build_system_prompt(
        session_id=session_id,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        workspace_desc=workspace_desc,
        root_folder=root_folder,
        include_workspaces=(workspace_id != "default")  # Don't include workspaces for default
    )

    # Insert system message at the beginning
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    conversations.insert(0, system_message)
    example["conversations"] = conversations

    return example


def main():
    """Main processing function."""
    input_path = Path("/home/user/Toolset-Training/Datasets/tools_datasets/contentManager/tools_v1.0.jsonl")
    output_path = Path("/home/user/Toolset-Training/Datasets/tools_datasets/contentManager/tools_v1.1.jsonl")

    print(f"Processing {input_path}")
    print(f"Output to {output_path}")

    processed_count = 0
    skipped_count = 0
    error_count = 0
    errors = []

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON
                example = json.loads(line.strip())

                # Process example
                processed_example = process_example(example)

                # Check if it was modified
                if len(processed_example["conversations"]) > len(example["conversations"]):
                    processed_count += 1
                else:
                    skipped_count += 1

                # Write to output
                outfile.write(json.dumps(processed_example, ensure_ascii=False) + "\n")

            except Exception as e:
                error_count += 1
                errors.append(f"Line {line_num}: {str(e)}")
                # Write original line on error
                outfile.write(line)

    # Report results
    print(f"\n=== Processing Complete ===")
    print(f"Total lines processed: {line_num}")
    print(f"Examples with system prompts added: {processed_count}")
    print(f"Examples skipped (already had system or no tool call): {skipped_count}")
    print(f"Errors encountered: {error_count}")

    if errors:
        print("\n=== Errors ===")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")

    print(f"\nOutput file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
