#!/usr/bin/env python3
"""
Dataset Enhancement Script for Batch 021 - Version 2
Fixes quality issues according to ENHANCEMENT_SPEC.md v1.1
Improved JSON parsing for multiline content
"""

import json
import re
from pathlib import Path

def extract_json_from_arguments(arguments_str):
    """Safely extract JSON from arguments string that may contain multiline content"""
    # Find first { and last }
    start_idx = arguments_str.find('{')
    if start_idx < 0:
        return None

    # Find the matching closing brace for the context object
    brace_count = 0
    for i in range(start_idx, len(arguments_str)):
        if arguments_str[i] == '{':
            brace_count += 1
        elif arguments_str[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return arguments_str[start_idx:i+1]

    return None

def extract_tool_and_args(assistant_content):
    """Extract tool name and arguments JSON from assistant content"""
    lines = assistant_content.split('\n')
    tool_name = None
    arguments_lines = []
    in_arguments = False

    for line in lines:
        if line.startswith('tool_call:'):
            tool_name = line.replace('tool_call:', '').strip()
        elif line.startswith('arguments:'):
            in_arguments = True
            # Get the rest of this line
            rest = line.replace('arguments:', '').strip()
            if rest:
                arguments_lines.append(rest)
        elif in_arguments:
            # Stop when we hit Result: or empty line followed by non-JSON
            if line.startswith('Result:') or (line.strip() == '' and len(arguments_lines) > 0):
                break
            arguments_lines.append(line)

    arguments_str = '\n'.join(arguments_lines) if arguments_lines else ''
    return tool_name, arguments_str

def extract_context_clues(user_content):
    """Extract meaningful context from user prompt"""
    lower_content = user_content.lower()

    if 'search' in lower_content or 'find' in lower_content:
        # Extract what we're searching for
        if 'typescript' in lower_content or 'any' in lower_content:
            return 'TypeScript type patterns'
        elif 'unused' in lower_content or 'export' in lower_content:
            return 'unused code exports'
        elif 'react' in lower_content or 'component' in lower_content:
            return 'React components'
        elif 'api' in lower_content or 'route' in lower_content:
            return 'API routes'
        elif 'async' in lower_content or 'function' in lower_content:
            return 'async functions'
        elif 'deprecated' in lower_content or 'package' in lower_content:
            return 'deprecated packages'
        return 'code patterns'

    elif 'create' in lower_content or 'folder' in lower_content:
        if 'performance' in lower_content:
            return 'performance benchmarks'
        elif 'webhook' in lower_content and 'documentation' in lower_content:
            return 'webhook API documentation'
        elif 'api' in lower_content and 'test' in lower_content:
            return 'API integration tests'
        elif 'contract' in lower_content:
            return 'contract test specifications'
        elif 'graphql' in lower_content or 'schema' in lower_content:
            return 'GraphQL schema definitions'
        elif 'component' in lower_content:
            return 'React component structure'
        elif 'test' in lower_content:
            return 'test framework'
        elif 'api' in lower_content:
            return 'API organization'
        return 'project resources'

    elif 'move' in lower_content or 'relocate' in lower_content:
        if 'utility' in lower_content or 'function' in lower_content:
            return 'utility functions'
        elif 'constant' in lower_content:
            return 'shared constants'
        elif 'api' in lower_content or 'doc' in lower_content:
            return 'API documentation'
        return 'files'

    elif 'delete' in lower_content or 'remove' in lower_content:
        if 'report' in lower_content or 'profile' in lower_content:
            return 'performance reports'
        return 'outdated files'

    elif 'append' in lower_content or 'add' in lower_content:
        if 'endpoint' in lower_content:
            return 'new API endpoints'
        elif 'documentation' in lower_content:
            return 'new documentation'
        return 'new content'

    elif 'update' in lower_content:
        if 'session' in lower_content or 'description' in lower_content:
            return 'session information'
        elif 'agent' in lower_content:
            return 'agent configuration'
        return 'configuration'

    elif 'read' in lower_content or 'open' in lower_content:
        if 'changelog' in lower_content:
            return 'project changelog'
        elif 'tsconfig' in lower_content or 'package' in lower_content:
            return 'configuration files'
        elif 'config' in lower_content or 'json' in lower_content:
            return 'configuration'
        return 'files'

    elif 'list' in lower_content or 'get' in lower_content:
        if 'session' in lower_content:
            return 'active sessions'
        elif 'workspace' in lower_content:
            return 'recent workspaces'
        return 'system resources'

    elif 'duplicate' in lower_content or 'copy' in lower_content:
        if 'template' in lower_content:
            return 'template files'
        return 'files'

    return 'resources'

def get_enhanced_session_memory(tool_name, context_clues):
    """Generate enhanced sessionMemory based on tool and context"""
    templates = {
        'vaultLibrarian_searchContent': f'Searching codebase for {context_clues}. Previous analysis identified patterns requiring systematic review of {context_clues} across multiple source files.',
        'vaultManager_createFolder': f'Setting up project structure. Creating {context_clues} folder to establish organized workflow and improve file management.',
        'vaultManager_moveNote': f'Organizing project files. Moving {context_clues} to improve structure and maintain proper file organization.',
        'contentManager_createContent': f'Planning implementation. Gathered requirements for {context_clues} and identified documentation needs.',
        'contentManager_appendContent': f'Updating documentation. Adding {context_clues} to existing records while preserving prior content.',
        'memoryManager_listSessions': f'Tracking workflow history. Previous sessions contain important context about {context_clues} and prior work.',
        'memoryManager_listWorkspaces': f'Reviewing workspace activity. Workspaces created previously contain context about {context_clues}.',
        'memoryManager_updateSession': f'Managing session state. Updating session metadata to reflect current {context_clues} work.',
        'memoryManager_createSession': f'Starting new workflow. Creating session to track {context_clues} with proper context and history.',
        'memoryManager_createState': f'Creating checkpoint for {context_clues}. State snapshot preserves current progress before making changes.',
        'memoryManager_updateState': f'Tagging snapshots for {context_clues}. Adding metadata to organize and track state backups.',
        'agentManager_toggleAgent': f'Managing agent lifecycle for {context_clues}. Agent state change enables operational continuity.',
        'agentManager_executePrompt': f'Running automation task. Agent execution for {context_clues} follows planned workflow steps.',
        'agentManager_updateAgent': f'Upgrading agent configuration. Improving agent capabilities for {context_clues}.',
        'agentManager_createAgent': f'Creating new automation agent for {context_clues}. Agent will handle {context_clues} operations.',
        'agentManager_getAgent': f'Reviewing agent configuration. Checking agent status for {context_clues} workflow.',
        'vaultManager_deleteNote': f'Cleaning up workspace. Removing {context_clues} files to reduce clutter and improve focus.',
        'vaultManager_duplicateNote': f'Reusing existing content. Duplicating {context_clues} template for new implementation.',
        'vaultManager_openNote': f'Reviewing configuration. Opening {context_clues} file to examine current settings and options.',
        'vaultLibrarian_searchDirectory': f'Mapping project structure. Searching {context_clues} directory to inventory all resources.',
    }
    return templates.get(tool_name, f'Working on task for {context_clues}. Prior work provides context for current activities.')

def get_enhanced_tool_context(tool_name, context_clues):
    """Generate enhanced toolContext based on tool and context"""
    templates = {
        'vaultLibrarian_searchContent': f'Using comprehensive content search with regex patterns to locate {context_clues} across codebase. Search results will guide organization and improvement decisions.',
        'vaultManager_createFolder': f'Creating {context_clues} folder to establish organized directory structure. Folder enables systematic organization of related resources.',
        'vaultManager_moveNote': f'Relocating {context_clues} to improve workspace organization. Moving preserves file history and relationships.',
        'contentManager_createContent': f'Creating comprehensive documentation for {context_clues}. Documentation provides clear guidelines for future work.',
        'contentManager_appendContent': f'Appending {context_clues} to existing documentation. Append method preserves prior entries and maintains history.',
        'memoryManager_listSessions': f'Listing sessions to review workflow history for {context_clues}. Session enumeration enables comprehensive activity tracking.',
        'memoryManager_listWorkspaces': f'Listing recent workspaces to understand {context_clues} activity. List provides overview of recent work.',
        'memoryManager_updateSession': f'Updating session metadata for {context_clues}. Accurate metadata maintains proper session context.',
        'memoryManager_createSession': f'Creating new session for {context_clues} workflow. Dedicated session enables focused work tracking.',
        'memoryManager_createState': f'Creating state backup for {context_clues}. Snapshot enables rollback if issues occur during changes.',
        'memoryManager_updateState': f'Updating state metadata for {context_clues}. Tags and descriptions improve state organization.',
        'agentManager_toggleAgent': f'Toggling agent state for {context_clues}. State change enables workflow continuation or maintenance.',
        'agentManager_executePrompt': f'Executing agent prompt for {context_clues}. Agent automation reduces manual effort and ensures consistency.',
        'agentManager_updateAgent': f'Updating agent configuration for {context_clues}. Configuration changes improve agent capabilities.',
        'agentManager_createAgent': f'Creating new agent for {context_clues} operations. Dedicated agent enables specialized task automation.',
        'agentManager_getAgent': f'Retrieving agent details for {context_clues}. Inspection enables configuration review and optimization.',
        'vaultManager_deleteNote': f'Deleting {context_clues} files permanently. Deletion improves workspace clarity and focus.',
        'vaultManager_duplicateNote': f'Duplicating {context_clues} template for new use. Duplication enables content reuse and consistency.',
        'vaultManager_openNote': f'Opening {context_clues} file for review. File access enables configuration examination and updates.',
        'vaultLibrarian_searchDirectory': f'Searching {context_clues} directory to enumerate resources. Directory search enables inventory and planning.',
    }
    return templates.get(tool_name, f'Using tool for {context_clues} workflow step. Tool application advances current task.')

def enhance_example(example, idx):
    """Enhance a single example"""
    try:
        # Get conversations
        if len(example['conversations']) < 2:
            return None

        user_content = example['conversations'][0]['content']
        assistant_content = example['conversations'][1]['content']

        # Extract tool and arguments
        tool_name, arguments_str = extract_tool_and_args(assistant_content)
        if not tool_name or not arguments_str:
            return None

        # Extract JSON from arguments
        json_str = extract_json_from_arguments(arguments_str)
        if not json_str:
            return None

        # Parse JSON
        try:
            args = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        # Extract context clues from user prompt
        context_clues = extract_context_clues(user_content)

        # Get context object
        context_obj = args.get('context', {})

        # Enhance sessionMemory
        enhanced_memory = get_enhanced_session_memory(tool_name, context_clues)
        if len(enhanced_memory) < 50:
            enhanced_memory += f' This maintains workflow continuity for {context_clues}.'
        context_obj['sessionMemory'] = enhanced_memory

        # Enhance toolContext - ensure it's a string
        enhanced_context = get_enhanced_tool_context(tool_name, context_clues)
        context_obj['toolContext'] = enhanced_context

        # Ensure all 7 context fields present
        if 'sessionId' not in context_obj:
            context_obj['sessionId'] = f'session_{1732115000000 + idx}_abc123def'
        if 'workspaceId' not in context_obj:
            context_obj['workspaceId'] = f'ws_{1732115000000 + idx}_xyz789klm'
        if 'sessionDescription' not in context_obj:
            context_obj['sessionDescription'] = f'Task execution {idx}'
        if 'primaryGoal' not in context_obj:
            context_obj['primaryGoal'] = 'Complete workflow task'
        if 'subgoal' not in context_obj:
            context_obj['subgoal'] = f'Execute step for {context_clues}'

        # Fix weak goal hierarchy
        if context_obj['primaryGoal'] == context_obj['subgoal']:
            context_obj['subgoal'] = f'Execute {context_clues} step'

        # Update context in args
        args['context'] = context_obj

        # Convert Result prompts to natural requests
        if user_content.strip().startswith('Result:'):
            lines = user_content.split('\n')
            natural_parts = []
            skip_until_content = False

            for line in lines:
                if line.startswith('Result:'):
                    skip_until_content = True
                    continue
                if skip_until_content and line.strip() and not line.startswith('{') and not line.startswith('}') and not line.startswith(','):
                    natural_parts.append(line.strip())

            if natural_parts:
                user_content = ' '.join(natural_parts)
            else:
                user_content = f'Please help me continue with the next step for {context_clues}.'

        # Rebuild clean assistant content (tool_call + arguments only, no Result)
        clean_content = f"tool_call: {tool_name}\narguments: {json.dumps(args)}"

        # Create enhanced example
        enhanced = {
            'conversations': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': clean_content}
            ],
            'label': True
        }

        return enhanced

    except Exception as e:
        return None

def main():
    input_file = Path('/home/user/Toolset-Training/Datasets/quality_review/enhancement_batches/batch_021.jsonl')
    output_file = Path('/home/user/Toolset-Training/Datasets/quality_review/enhancement_batches/enhanced_batch_021.jsonl')

    print("Starting enhancement of batch_021.jsonl...")
    print(f"Reading from: {input_file}")
    print()

    enhanced_count = 0
    skipped_count = 0
    enhanced_examples = []

    # Read and enhance examples
    with open(input_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
                enhanced = enhance_example(example, idx)

                if enhanced:
                    enhanced_examples.append(enhanced)
                    enhanced_count += 1
                    print(f"✓ Example {idx:2d}: Enhanced")
                else:
                    skipped_count += 1
                    print(f"✗ Example {idx:2d}: Skipped")

            except Exception as e:
                skipped_count += 1
                print(f"✗ Example {idx:2d}: Error - Skipped")

    # Write enhanced examples
    with open(output_file, 'w') as f:
        for example in enhanced_examples:
            f.write(json.dumps(example) + '\n')

    print()
    print("=" * 60)
    print(f"Enhancement Complete:")
    print(f"  Total examples processed: {enhanced_count + skipped_count}")
    print(f"  Successfully enhanced: {enhanced_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output file: {output_file}")
    print("=" * 60)

    return enhanced_count, skipped_count

if __name__ == '__main__':
    enhanced, skipped = main()
