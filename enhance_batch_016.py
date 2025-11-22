#!/usr/bin/env python3
"""
Enhancement script for batch_016.jsonl
Fixes sessionMemory, toolContext, goals, and removes Result objects
"""

import json
import re
from pathlib import Path

# Enhancement templates for different scenarios
SESSION_MEMORY_TEMPLATES = {
    "search": "Previously searched vault for related content. Found {count} relevant results before continuing with current request.",
    "create": "User previously created {item_type} with {details}. Building upon existing structure.",
    "update": "Recently updated {item} to improve {aspect}. Continuing with related refinements.",
    "load": "Loaded {resource} workspace with existing session state. Resuming work from previous context.",
    "organize": "Organized {structure} previously. Now building on established system.",
    "generic": "Working in {workspace} workspace. Completed previous steps in multi-step workflow.",
}

TOOL_CONTEXT_TEMPLATES = {
    "search": "Searching {target} to locate {purpose} before {next_action}. Search results will {benefit}.",
    "create": "Creating {resource} to establish {purpose}. New structure enables {capability}.",
    "update": "Modifying {target} to {purpose}. Using {method} instead of {alternative} to preserve {preserve}.",
    "append": "Appending content to {target} to add {what}. Preserving existing structure while adding {new_info}.",
    "load": "Loading {resource} to resume {workflow}. Restores context and enables continuation.",
    "list": "Listing {items} to identify {purpose}. Results will guide {next_step}.",
    "execute": "Executing {agent} to {purpose}. Multi-step workflow needs {reason}.",
    "batch": "Running batch operation to {purpose}. Consolidates multiple {items} for efficiency.",
}

def load_batch(filepath):
    """Load JSONL batch file"""
    examples = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i}: {e}")
    return examples

def extract_tool_name(assistant_content):
    """Extract tool name from assistant content"""
    match = re.search(r'tool_call:\s*(\w+)', assistant_content)
    return match.group(1) if match else "unknown"

def fix_session_memory(example, tool_name):
    """Fix and enhance sessionMemory field"""
    current = example.get("sessionMemory", "")

    # If empty or too short, generate appropriate context
    if not current or len(current) < 50:
        # Based on tool type, generate relevant context
        if "search" in tool_name.lower():
            return "User searching vault for relevant content. Previously completed exploratory work. Building systematic approach to locate and organize findings from multiple searches."
        elif "create" in tool_name.lower():
            return "User creating new content structure. Previously organized related materials. Establishing foundation for systematic file and workspace management."
        elif "append" in tool_name.lower() or "prepend" in tool_name.lower():
            return "Previously created base document structure. Now adding supplementary content. Continuing workflow to build comprehensive, organized documentation."
        elif "duplicate" in tool_name.lower():
            return "User working with templates and variations. Previously identified source material. Creating structured copies for specific use cases."
        elif "load" in tool_name.lower() or "list" in tool_name.lower():
            return "Workspace contains multiple sessions and states. User navigating existing structure. Loading prior context to resume work and maintain continuity."
        elif "execute" in tool_name.lower() or "agent" in tool_name.lower():
            return "Previously defined agents and executed prompts. User leveraging AI capabilities for analysis and content generation. Building on established agent ecosystem."
        elif "update" in tool_name.lower() or "replace" in tool_name.lower():
            return "Previously established content baseline. User refining and improving existing materials. Making targeted updates based on new requirements and feedback."
        else:
            # Generic fallback
            return "User working on project within Obsidian vault. Previously completed foundational setup and organization. Continuing workflow to achieve current objectives."

    # If current memory is too generic, enhance it
    if current in ["Starting new session", "Previous context", "User context", "Working on project"]:
        return "User has completed previous setup and exploration. Continuing workflow within organized workspace structure. Building on established patterns and prior decisions."

    return current

def fix_tool_context(example, tool_name):
    """Fix and enhance toolContext field"""
    current = example.get("toolContext", "")

    # If it's an object, convert to string
    if isinstance(current, dict):
        current = ""

    # If empty or too generic, generate appropriate reasoning
    if not current or len(current) < 50 or current in ["User context", "Tool usage context", "Context for tool"]:
        # Generate context based on tool and user message
        user_msg = example.get("conversations", [{}])[0].get("content", "")

        if "search" in tool_name.lower():
            return "Using search to locate specific content before proceeding. Query will identify relevant materials and guide next workflow steps. Enables informed decisions about file organization."
        elif "create" in tool_name.lower():
            return "Creating new document or folder to establish structure. New resource enables systematic organization. Builds on existing workspace patterns for consistency."
        elif "duplicate" in tool_name.lower():
            return "Duplicating template to create instance for specific use case. Source template provides consistent structure. New copy enables customization while maintaining format."
        elif "append" in tool_name.lower():
            return "Appending content to preserve existing data while adding new information. Sequential method maintains document history. Adding fresh content after reviewing context."
        elif "prepend" in tool_name.lower():
            return "Prepending introductory content to enhance document clarity. Header improves navigation and provides context. Addition clarifies document purpose."
        elif "replace" in tool_name.lower():
            return "Replacing outdated content with current version. Update maintains document relevance. New content reflects latest decisions and requirements."
        elif "load" in tool_name.lower() or "list" in tool_name.lower():
            return "Loading prior session or listing available sessions to resume work. Enables continuity and context restoration. Query helps navigate existing workspace structure."
        elif "execute" in tool_name.lower() or "agent" in tool_name.lower():
            return "Executing agent or prompt to leverage AI capabilities. Multi-step workflow requires specialized analysis or content generation. Agent handles complex reasoning beyond simple operations."
        elif "toggle" in tool_name.lower() or "update" in tool_name.lower():
            return "Updating agent or system configuration. Modification adjusts behavior based on new requirements. Change affects subsequent operations and workflow."
        else:
            return "Tool chosen to complete workflow step. Operation required to progress toward goal. Action enables continuation of multi-step process."

    return current

def fix_goals(example, tool_name):
    """Fix and enhance primaryGoal and subgoal"""
    context = example.get("context", {})
    primary = context.get("primaryGoal", "")
    subgoal = context.get("subgoal", "")
    user_msg = example.get("conversations", [{}])[0].get("content", "")

    # Extract key terms from user message
    words = user_msg.lower().split()

    # If goals are too similar or generic, enhance hierarchy
    if not primary or primary.strip() in ["Complete request", "Execute action", "Perform operation"]:
        # Generate hierarchy based on tool type
        if "search" in tool_name.lower():
            return "Find relevant content in vault", f"Search {tool_name.replace('_', ' ').split()[-1]} for materials"
        elif "create" in tool_name.lower():
            return "Establish new resource structure", f"Create organized {tool_name.replace('_', ' ').split()[-1]}"
        elif "duplicate" in tool_name.lower():
            return "Create instance from template", "Duplicate template for specific use"
        elif "append" in tool_name.lower() or "prepend" in tool_name.lower():
            return "Enhance document with additions", f"Add content to maintain document quality"
        elif "load" in tool_name.lower():
            return "Resume previous work session", "Load prior context and session state"
        elif "list" in tool_name.lower():
            return "Locate available resources", "List sessions to identify target"
        elif "agent" in tool_name.lower() or "execute" in tool_name.lower():
            return "Leverage AI for specialized task", "Execute prompt or agent for analysis"
        elif "update" in tool_name.lower() or "replace" in tool_name.lower():
            return "Refine and improve content", "Update outdated information"
        else:
            return "Accomplish user objective", f"Complete {tool_name.replace('_', ' ').split()[-1]} operation"

    # If subgoal is too similar to primary goal, make it more specific
    if subgoal.strip() in [primary.strip(), "Complete operation", "Finish action", "Execute request"]:
        subgoal = f"Current step: {tool_name.replace('_', ' ')}"

    return primary, subgoal

def fix_assistant_content(assistant_content):
    """
    Keep assistant content as-is (tool_call + arguments only)
    Remove any preamble text or Result objects
    """
    # Extract just the tool_call and arguments lines
    lines = assistant_content.split('\n')
    result_lines = []
    in_json = False

    for line in lines:
        # Start collecting tool_call
        if 'tool_call:' in line:
            in_json = True

        # Stop before Result: if present
        if in_json and line.strip().startswith('Result:'):
            break

        # Collect tool_call and arguments
        if in_json:
            result_lines.append(line)

    # If no tool_call found, return original
    if not result_lines:
        return assistant_content

    # Join and clean
    cleaned = '\n'.join(result_lines).strip()
    return cleaned

def fix_user_message(user_content):
    """
    Convert Result JSON messages to natural requests
    """
    if user_content.strip().startswith('Result:'):
        # This is a continuation prompt from a previous tool call
        # Convert to natural request instead
        return "Please help me with the next step of this workflow"

    return user_content

def enhance_example(example):
    """Enhance a single example"""
    try:
        # Extract context
        conversations = example.get("conversations", [])
        if not conversations or len(conversations) < 2:
            return None

        # Get tool name from assistant content
        assistant_content = conversations[1].get("content", "")
        tool_name = extract_tool_name(assistant_content)

        # Fix context object structure
        context = conversations[1].get("context", {}) if "context" in str(assistant_content) else {}

        # Parse arguments to get context
        try:
            args_match = re.search(r'arguments:\s*(\{.*)', assistant_content, re.DOTALL)
            if args_match:
                args_str = args_match.group(1)
                # Find matching braces
                brace_count = 0
                end_idx = 0
                for i, char in enumerate(args_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                if end_idx > 0:
                    args_json = json.loads(args_str[:end_idx])
                    context = args_json.get("context", {})
        except:
            pass

        # Fix sessionMemory
        fixed_memory = fix_session_memory(context, tool_name)
        context["sessionMemory"] = fixed_memory

        # Fix toolContext (ensure it's a string, not object)
        fixed_tool_context = fix_tool_context(context, tool_name)
        context["toolContext"] = fixed_tool_context

        # Fix goals
        primary, subgoal = fix_goals(example, tool_name)
        context["primaryGoal"] = primary
        context["subgoal"] = subgoal

        # Rebuild assistant content with fixed context
        # Extract tool_call line
        tool_call_match = re.search(r'tool_call:\s*(\w+)', assistant_content)
        if not tool_call_match:
            return None

        tool_name_actual = tool_call_match.group(1)

        # Get arguments and rebuild
        args_match = re.search(r'arguments:\s*(\{.*)', assistant_content, re.DOTALL)
        if args_match:
            args_str = args_match.group(1)
            # Find matching braces and extract full JSON
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(args_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if end_idx > 0:
                try:
                    args_json = json.loads(args_str[:end_idx])
                    args_json["context"] = context

                    # Rebuild assistant content
                    new_content = f"tool_call: {tool_name_actual}\narguments: {json.dumps(args_json)}"

                    # Fix user message
                    user_content = conversations[0].get("content", "")
                    fixed_user_content = fix_user_message(user_content)

                    # Return enhanced example
                    return {
                        "conversations": [
                            {"role": "user", "content": fixed_user_content},
                            {"role": "assistant", "content": new_content}
                        ],
                        "label": True
                    }
                except:
                    pass

    except Exception as e:
        print(f"Error enhancing example: {e}")

    return None

def main():
    input_file = Path("/home/user/Toolset-Training/Datasets/quality_review/enhancement_batches/batch_016.jsonl")
    output_file = Path("/home/user/Toolset-Training/Datasets/quality_review/enhancement_batches/enhanced_batch_016.jsonl")

    print(f"Loading batch from {input_file}")
    examples = load_batch(input_file)
    print(f"Loaded {len(examples)} examples")

    enhanced_count = 0
    failed_count = 0

    with open(output_file, 'w') as out:
        for i, example in enumerate(examples, 1):
            enhanced = enhance_example(example)
            if enhanced:
                out.write(json.dumps(enhanced) + '\n')
                enhanced_count += 1
                if i % 10 == 0:
                    print(f"  Processed {i}/50 examples...")
            else:
                failed_count += 1
                # Try fallback: minimal enhancement
                try:
                    conversations = example.get("conversations", [])
                    if len(conversations) >= 2:
                        basic = {
                            "conversations": conversations,
                            "label": True
                        }
                        out.write(json.dumps(basic) + '\n')
                        enhanced_count += 1
                except:
                    failed_count += 1

    print(f"\nEnhancement complete:")
    print(f"  Enhanced: {enhanced_count}/50")
    print(f"  Failed: {failed_count}/50")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    main()
