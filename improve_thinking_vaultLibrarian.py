#!/usr/bin/env python3
"""
Improve thinking blocks in vaultLibrarian tools dataset.
Processes tools_v1.3.jsonl -> tools_v1.4.jsonl
"""

import json
import re
from pathlib import Path

def improve_thinking(thinking_str, context):
    """
    Improve a thinking block based on vaultLibrarian patterns.

    Args:
        thinking_str: The thinking JSON string
        context: Dict with user_content, tool_name, tool_args for context

    Returns:
        Improved thinking JSON string
    """
    try:
        thinking = json.loads(thinking_str)
    except:
        return thinking_str

    # Extract tool context
    tool_name = context.get('tool_name', '')
    user_msg = context.get('user_content', '')
    is_batch_op = 'batch' in tool_name.lower()
    is_memory_search = 'searchMemory' in tool_name
    is_content_search = 'searchContent' in tool_name
    is_dir_search = 'searchDirectory' in tool_name

    # Fix 1: Improve generic goals - make them specific and actionable
    goal = thinking.get('goal', '')
    if len(goal) < 40 or any(generic in goal.lower() for generic in ['find', 'search', 'locate', 'discover']) and goal.count(' ') < 6:
        # Extract key intent from user message
        user_lower = user_msg.lower()

        if is_memory_search:
            # For memory searches, focus on retrieving specific information
            if 'project' in user_lower:
                thinking['goal'] = "Retrieve session memory entries about project discussions and decisions for context reference"
            elif 'migration' in user_lower or 'database' in user_lower:
                thinking['goal'] = "Locate session memory containing technical discussion details for current implementation work"
            elif 'todo' in user_lower or 'task' in user_lower:
                thinking['goal'] = "Find memory entries about task assignments and action items to track work progress"
            else:
                thinking['goal'] = f"Retrieve relevant session memory to provide context for {goal.lower() if goal else 'current task'}"

        elif is_content_search:
            # For content searches, focus on discovery and analysis
            if 'interface' in user_lower or 'type' in user_lower:
                thinking['goal'] = "Identify all files containing type definitions to document codebase structure"
            elif 'todo' in user_lower or 'fixme' in user_lower:
                thinking['goal'] = "Catalog all pending tasks and code comments to prioritize development work"
            elif 'import' in user_lower or 'dependency' in user_lower:
                thinking['goal'] = "Map import statements and dependencies to assess refactoring impact"
            else:
                thinking['goal'] = f"Search vault content to {goal.lower() if goal and len(goal) > 10 else 'locate relevant files for analysis'}"

        elif is_dir_search:
            # For directory searches, focus on navigation and organization
            if 'archive' in user_lower:
                thinking['goal'] = "Locate archive directories to support content organization and cleanup workflow"
            elif 'config' in user_lower:
                thinking['goal'] = "Find configuration files for environment setup and validation"
            elif 'meeting' in user_lower or 'notes' in user_lower:
                thinking['goal'] = "Discover note files scattered across folders for consolidation and archival"
            else:
                thinking['goal'] = f"Identify directory structure matching '{goal.lower() if goal else 'pattern'}' for navigation"

        elif is_batch_op:
            # For batch operations, focus on efficiency and scope
            if 'css' in user_lower or 'unused' in user_lower:
                thinking['goal'] = "Analyze CSS usage patterns to identify optimization opportunities and reduce bundle size"
            elif 'search' in user_lower and ('react' in user_lower or 'vue' in user_lower):
                thinking['goal'] = "Execute parallel framework searches to assess technology usage across codebase"
            else:
                thinking['goal'] = f"Process multiple files efficiently to {goal.lower() if goal else 'complete batch operation'}"

    # Fix 2: Enrich memory with WHY + WHAT BEFORE + broader context
    memory = thinking.get('memory', '')
    if len(memory) < 80 or memory.count('.') < 2:
        # Build richer memory with specific context
        user_lower = user_msg.lower()

        # Determine what came before
        what_before = ""
        if is_memory_search:
            what_before = "previously discussed related topics during earlier session exchanges"
        elif is_batch_op:
            what_before = "completed preliminary searches and identified scope of work"
        elif is_content_search:
            what_before = "reviewed vault structure and identified search criteria"
        elif is_dir_search:
            what_before = "navigated workspace organization and determined search paths"

        # Determine why (broader situation)
        why_context = ""
        if 'archive' in user_lower or 'cleanup' in user_lower:
            why_context = "User maintaining vault organization through periodic archival workflow to improve content discoverability."
        elif 'todo' in user_lower or 'task' in user_lower:
            why_context = "User tracking development work and technical debt to prioritize upcoming sprint tasks."
        elif 'migration' in user_lower or 'database' in user_lower:
            why_context = "User referencing prior technical discussions to inform current implementation decisions."
        elif 'config' in user_lower or 'api' in user_lower:
            why_context = "User performing system configuration audit to ensure environment consistency."
        elif 'interface' in user_lower or 'type' in user_lower:
            why_context = "User documenting codebase architecture to understand type system and data contracts."
        elif 'css' in user_lower or 'bundle' in user_lower:
            why_context = "User optimizing frontend performance to reduce page load times and improve UX metrics."
        else:
            why_context = "User progressing through multi-step workflow requiring information discovery before next action."

        # Combine: WHAT BEFORE + current state + WHY
        if memory and len(memory) > 20:
            # Enhance existing memory
            thinking['memory'] = f"User {what_before.capitalize()}. {memory} {why_context}"
        else:
            # Create rich memory from scratch
            thinking['memory'] = f"User {what_before}. {why_context} This search provides necessary context for informed decision-making and workflow progression."

    # Fix 3: Make requirements distinct from plan (verification prerequisites)
    reqs = thinking.get('requirements', [])
    plan = thinking.get('plan', [])

    # If requirements == plan (duplicate content), fix it
    if reqs == plan or (len(reqs) > 0 and len(plan) > 0 and reqs[0] == plan[0]):
        # Requirements should be verification checks, not actions
        new_reqs = []
        if is_memory_search:
            new_reqs = [
                "Verify workspace ID is configured for memory search",
                "Confirm query terms will match stored memory entries",
                "Check limit parameter provides sufficient results"
            ]
        elif is_content_search:
            new_reqs = [
                "Verify search pattern will match target file content",
                "Confirm workspace context includes searchable paths",
                "Check includeContent setting matches user's needs"
            ]
        elif is_dir_search:
            new_reqs = [
                "Confirm search paths exist and are accessible",
                "Verify query pattern syntax is correct for file/folder matching",
                "Check searchType (files/folders) aligns with user intent"
            ]
        elif is_batch_op:
            new_reqs = [
                "Verify batch operation pattern covers all target files",
                "Confirm operation type is appropriate for task",
                "Check reference files are specified correctly if needed"
            ]
        else:
            new_reqs = [
                "Verify search parameters are correctly configured",
                "Confirm workspace access and permissions",
                "Check result limits are appropriate for user needs"
            ]
        thinking['requirements'] = new_reqs

    # Fix 4: Complex assessment for batch operations
    if is_batch_op and not thinking.get('assessment', {}).get('complex', False):
        thinking['assessment']['complex'] = True

    # Fix 5: Confidence - high for searches (0.8-0.95), not 1.0
    conf = thinking.get('confidence', 0.9)
    if conf >= 0.99:
        thinking['confidence'] = 0.88  # More realistic
    elif conf >= 0.97:
        thinking['confidence'] = 0.92
    elif conf < 0.75:
        thinking['confidence'] = 0.85  # Searches are generally reliable

    # Fix 6: Improve plan steps (should be execution steps, not duplicates of requirements)
    if len(plan) > 0 and (plan == reqs or any('verify' in step.lower() or 'confirm' in step.lower() or 'check' in step.lower() for step in plan)):
        # Plan should describe specific execution steps, not verification or generic actions
        new_plan = []
        user_lower = user_msg.lower()

        if is_memory_search:
            # Memory search plans focus on retrieval
            query_term = "database migration" if "migration" in user_lower else "project context" if "project" in user_lower else "relevant information"
            new_plan = [
                f"Execute memory search for '{query_term}' across session history",
                "Retrieve top-ranked memory entries ordered by semantic relevance",
                "Return results with timestamps and context for user review"
            ]
        elif is_content_search:
            # Content search plans focus on pattern matching
            if 'interface' in user_lower or 'type' in user_lower:
                new_plan = [
                    "Search all files for 'interface' keyword declarations",
                    "Filter to TypeScript/JavaScript files if applicable",
                    "Return file paths without content (as requested)",
                    "Organize results by directory for easy navigation"
                ]
            elif 'todo' in user_lower or 'fixme' in user_lower:
                new_plan = [
                    "Search codebase for TODO/FIXME comment patterns",
                    "Extract comment context and file locations",
                    "Return comprehensive list of pending tasks"
                ]
            elif 'import' in user_lower or 'logger' in user_lower:
                new_plan = [
                    "Search for import statements matching specified pattern",
                    "Include content snippets showing import usage",
                    "Return results for refactoring assessment"
                ]
            else:
                new_plan = [
                    "Execute content search with query across vault",
                    "Collect matching files and relevant snippets",
                    "Return organized results for analysis"
                ]
        elif is_dir_search:
            # Directory search plans focus on discovery
            if 'archive' in user_lower:
                new_plan = [
                    "Search directory tree for folders containing 'archive'",
                    "Scan from root or specified paths",
                    "Return folder paths for organization workflow"
                ]
            elif 'config' in user_lower:
                new_plan = [
                    "Search project root for configuration files",
                    "Match config* patterns at depth 0 (root level only)",
                    "Return up to 20 config files for review"
                ]
            elif 'meeting' in user_lower:
                new_plan = [
                    "Search Notes/, Meetings/, Projects/ for files matching 'meeting'",
                    "Filter to markdown files (.md extension)",
                    "Limit to 20 results for initial review",
                    "Return file paths for archival workflow"
                ]
            else:
                new_plan = [
                    "Search specified directories with pattern",
                    "Apply file type and search type filters",
                    "Return matching paths for navigation"
                ]
        elif is_batch_op:
            # Batch operations need detailed multi-step plans
            if 'css' in user_lower or 'unused' in user_lower:
                new_plan = [
                    "Analyze all CSS files matching **/*.css pattern",
                    "Cross-reference class names with usage in JSX/TSX/HTML files",
                    "Identify classes defined but never referenced",
                    "Generate report of unused classes for safe deletion"
                ]
            elif 'react' in user_lower and 'vue' in user_lower:
                new_plan = [
                    "Execute 3 parallel searches for React, Vue, Angular in web-dev folder",
                    "Limit each search to 10 results with content inclusion",
                    "Maintain separate result sets (mergeResults=false)",
                    "Return framework usage analysis across codebase"
                ]
            else:
                new_plan = [
                    "Process files matching batch pattern",
                    "Execute specified operation across file set",
                    "Generate results summary",
                    "Return completion status for next workflow step"
                ]
        else:
            new_plan = [
                "Execute search with specified parameters",
                "Process and organize results",
                "Return findings for user review"
            ]
        thinking['plan'] = new_plan

    # Ensure assessment has both keys
    if 'assessment' not in thinking:
        thinking['assessment'] = {}
    thinking['assessment']['complex'] = thinking['assessment'].get('complex', is_batch_op)
    thinking['assessment']['risky'] = False  # All search operations are safe

    return json.dumps(thinking, indent=2)


def process_file(input_path, output_path):
    """Process the entire dataset file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    processed = 0
    skipped = 0
    errors = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())

                    # Skip text-only uncertainty examples (no tool calls)
                    if 'pattern' in data and data.get('pattern') == 'text_only':
                        outfile.write(line)
                        skipped += 1
                        continue

                    # Find and improve thinking block in assistant message
                    improved = False
                    for conv in data.get('conversations', []):
                        if conv.get('role') == 'assistant' and 'content' in conv:
                            content = conv['content']

                            # Extract thinking block
                            match = re.search(r'<thinking>\n(\{.*?\})\n</thinking>', content, re.DOTALL)
                            if match:
                                old_thinking = match.group(1)

                                # Build context
                                user_content = ''
                                tool_name = ''
                                for c in data.get('conversations', []):
                                    if c.get('role') == 'user':
                                        user_content = c.get('content', '')

                                if 'tool_calls' in conv:
                                    tool_name = conv['tool_calls'][0].get('function', {}).get('name', '')

                                context = {
                                    'user_content': user_content,
                                    'tool_name': tool_name,
                                    'line_num': line_num
                                }

                                # Improve thinking
                                new_thinking = improve_thinking(old_thinking, context)

                                # Replace in content
                                new_content = content.replace(
                                    f'<thinking>\n{old_thinking}\n</thinking>',
                                    f'<thinking>\n{new_thinking}\n</thinking>'
                                )
                                conv['content'] = new_content
                                improved = True

                    # Write improved line
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed += 1

                    if line_num % 100 == 0:
                        print(f"Processed {line_num} examples...")

                except Exception as e:
                    print(f"Error on line {line_num}: {e}")
                    errors += 1
                    outfile.write(line)  # Write original on error

    print(f"\nProcessing complete!")
    print(f"Total processed: {processed}")
    print(f"Skipped (text-only): {skipped}")
    print(f"Errors: {errors}")
    return processed, skipped, errors


if __name__ == '__main__':
    input_file = '/mnt/f/Code/Toolset-Training/Datasets/tools_datasets/thinking/vaultLibrarian/tools_v1.3.jsonl'
    output_file = '/mnt/f/Code/Toolset-Training/Datasets/tools_datasets/thinking/vaultLibrarian/tools_v1.4.jsonl'

    print(f"Improving thinking blocks: {input_file} -> {output_file}")
    process_file(input_file, output_file)
