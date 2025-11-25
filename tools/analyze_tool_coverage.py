#!/usr/bin/env python3
"""
Analyze tool usage coverage in synthetic Claudesidian dataset.

Provides detailed breakdown of:
- Tool usage by manager and mode
- Good (true) vs Bad (false) label distribution per tool
- Coverage statistics and missing tools
- Invalid tool calls in dataset
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Define all valid tools (47 total across 6 managers + 1 meta-tool)
VALID_TOOLS = {
    # Agent Manager (9 tools)
    "agentManager_createAgent",
    "agentManager_deleteAgent",
    "agentManager_executePrompt",
    "agentManager_generateImage",
    "agentManager_getAgent",
    "agentManager_listAgents",
    "agentManager_listModels",
    "agentManager_toggleAgent",
    "agentManager_updateAgent",

    # Command Manager (2 tools)
    "commandManager_executeCommand",
    "commandManager_listCommands",

    # Content Manager (9 tools)
    "contentManager_appendContent",
    "contentManager_batchContent",
    "contentManager_createContent",
    "contentManager_deleteContent",
    "contentManager_findReplaceContent",
    "contentManager_prependContent",
    "contentManager_readContent",
    "contentManager_replaceByLine",
    "contentManager_replaceContent",

    # Memory Manager (8 tools + 5 missing that need schema updates)
    "memoryManager_createSession",
    "memoryManager_createState",
    "memoryManager_listSessions",
    "memoryManager_listStates",
    "memoryManager_loadSession",
    "memoryManager_loadState",
    "memoryManager_updateSession",
    "memoryManager_updateWorkspace",
    # These 5 are called in dataset but missing from schema:
    "memoryManager_createWorkspace",
    "memoryManager_listWorkspaces",
    "memoryManager_loadWorkspace",
    "memoryManager_updateState",
    "agentManager_batchExecutePrompt",

    # Vault Manager (7 tools)
    "vaultManager_createFolder",
    "vaultManager_deleteFolder",
    "vaultManager_deleteNote",
    "vaultManager_duplicateNote",
    "vaultManager_editFolder",
    "vaultManager_listDirectory",
    "vaultManager_moveFolder",
    "vaultManager_moveNote",
    "vaultManager_openNote",

    # Vault Librarian (4 tools)
    "vaultLibrarian_batch",
    "vaultLibrarian_searchContent",
    "vaultLibrarian_searchDirectory",
    "vaultLibrarian_searchMemory",

    # Meta tool (1 tool)
    "get_tools",
}

# Organize tools by manager for reporting
TOOLS_BY_MANAGER = {
    "agentManager": [
        "agentManager_batchExecutePrompt",
        "agentManager_createAgent",
        "agentManager_deleteAgent",
        "agentManager_executePrompt",
        "agentManager_generateImage",
        "agentManager_getAgent",
        "agentManager_listAgents",
        "agentManager_listModels",
        "agentManager_toggleAgent",
        "agentManager_updateAgent",
    ],
    "commandManager": [
        "commandManager_executeCommand",
        "commandManager_listCommands",
    ],
    "contentManager": [
        "contentManager_appendContent",
        "contentManager_batchContent",
        "contentManager_createContent",
        "contentManager_deleteContent",
        "contentManager_findReplaceContent",
        "contentManager_prependContent",
        "contentManager_readContent",
        "contentManager_replaceByLine",
        "contentManager_replaceContent",
    ],
    "memoryManager": [
        "memoryManager_createSession",
        "memoryManager_createState",
        "memoryManager_createWorkspace",
        "memoryManager_listSessions",
        "memoryManager_listStates",
        "memoryManager_listWorkspaces",
        "memoryManager_loadSession",
        "memoryManager_loadState",
        "memoryManager_loadWorkspace",
        "memoryManager_updateSession",
        "memoryManager_updateState",
        "memoryManager_updateWorkspace",
    ],
    "vaultManager": [
        "vaultManager_createFolder",
        "vaultManager_deleteFolder",
        "vaultManager_deleteNote",
        "vaultManager_duplicateNote",
        "vaultManager_editFolder",
        "vaultManager_listDirectory",
        "vaultManager_moveFolder",
        "vaultManager_moveNote",
        "vaultManager_openNote",
    ],
    "vaultLibrarian": [
        "vaultLibrarian_batch",
        "vaultLibrarian_searchContent",
        "vaultLibrarian_searchDirectory",
        "vaultLibrarian_searchMemory",
    ],
    "meta": [
        "get_tools",
    ],
}


def extract_tool_calls(assistant_content):
    """Extract tool names from assistant content."""
    tools = []
    # Pattern to match "tool_call: toolName"
    pattern = r'tool_call:\s*([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, assistant_content)
    tools.extend(matches)
    return tools


def analyze_coverage(jsonl_file):
    """Analyze tool usage coverage in the dataset."""
    tool_counter = Counter()
    label_tool_counter = defaultdict(Counter)  # Track tools by label (good/bad)
    invalid_tool_counter = Counter()
    invalid_examples = []
    conversation_count = 0
    total_tool_calls = 0
    good_example_count = 0
    bad_example_count = 0

    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])
                label = data.get('label', None)

                conversation_count += 1
                if label is True:
                    good_example_count += 1
                elif label is False:
                    bad_example_count += 1

                # Extract tools from assistant messages
                for msg in conversations:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        tools = extract_tool_calls(content)

                        for tool in tools:
                            total_tool_calls += 1

                            if tool in VALID_TOOLS:
                                tool_counter[tool] += 1
                                if label is not None:
                                    label_tool_counter[label][tool] += 1
                            else:
                                invalid_tool_counter[tool] += 1
                                invalid_examples.append({
                                    'line': line_num,
                                    'tool': tool,
                                    'label': label
                                })

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue

    return {
        'tool_counter': tool_counter,
        'label_tool_counter': label_tool_counter,
        'invalid_tool_counter': invalid_tool_counter,
        'invalid_examples': invalid_examples,
        'conversation_count': conversation_count,
        'total_tool_calls': total_tool_calls,
        'good_example_count': good_example_count,
        'bad_example_count': bad_example_count,
    }


def print_report(results):
    """Print a formatted coverage report."""
    print("=" * 80)
    print("TOOL USAGE COVERAGE REPORT")
    print("=" * 80)
    print()

    # Summary stats
    print("DATASET SUMMARY")
    print("-" * 80)
    print(f"Total Examples:        {results['conversation_count']}")
    print(f"Good Examples (true):  {results['good_example_count']}")
    print(f"Bad Examples (false):  {results['bad_example_count']}")
    print(f"Total Tool Calls:      {results['total_tool_calls']}")
    print(f"Valid Tool Calls:      {sum(results['tool_counter'].values())}")
    print(f"Invalid Tool Calls:    {sum(results['invalid_tool_counter'].values())}")
    print(f"Unique Valid Tools:    {len(results['tool_counter'])}")
    print(f"Unique Invalid Tools:  {len(results['invalid_tool_counter'])}")

    if results['good_example_count'] > 0 and results['bad_example_count'] > 0:
        ratio = results['good_example_count'] / results['bad_example_count']
        print(f"Label Ratio:           {ratio:.2f}:1 (good:bad)")
    print()

    # Coverage by manager
    print("=" * 80)
    print("COVERAGE BY MANAGER")
    print("=" * 80)
    print()

    for manager in sorted(TOOLS_BY_MANAGER.keys()):
        manager_tools = TOOLS_BY_MANAGER[manager]
        coverage = sum(1 for tool in manager_tools if tool in results['tool_counter'])

        print(f"{manager.upper()} ({coverage}/{len(manager_tools)} tools with examples)")
        print("-" * 80)
        print(f"{'Tool':<45} {'Count':>10} {'Good':>10} {'Bad':>10}")
        print("-" * 80)

        for tool in sorted(manager_tools):
            count = results['tool_counter'].get(tool, 0)
            good = results['label_tool_counter'].get(True, Counter())[tool]
            bad = results['label_tool_counter'].get(False, Counter())[tool]

            if count > 0:
                print(f"{tool:<45} {count:>10} {good:>10} {bad:>10}")
            else:
                print(f"{tool:<45} {'(no examples)':>10}")
        print()

    # Invalid tools
    if results['invalid_tool_counter']:
        print("=" * 80)
        print("INVALID TOOL CALLS")
        print("=" * 80)
        print()
        print(f"{'Invalid Tool':<50} {'Count':>10}")
        print("-" * 80)

        for tool, count in results['invalid_tool_counter'].most_common():
            print(f"{tool:<50} {count:>10}")

        print()
        print(f"First 20 invalid tool call instances:")
        print("-" * 80)
        for example in results['invalid_examples'][:20]:
            print(f"  Line {example['line']:5d}: {example['tool']:<40} (label={example['label']})")

        if len(results['invalid_examples']) > 20:
            print(f"  ... and {len(results['invalid_examples']) - 20} more")
        print()

    print("=" * 80)


def main():
    # Analyze the main dataset file
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        dataset_path = Path('syngen_toolset_v1.0.0_claude.jsonl')

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {dataset_path}")
    print(f"Valid tools defined: {len(VALID_TOOLS)}")
    print()

    results = analyze_coverage(dataset_path)
    print_report(results)


if __name__ == '__main__':
    main()
