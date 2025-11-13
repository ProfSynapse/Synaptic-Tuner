#!/usr/bin/env python3
"""
Analyze tool usage coverage in claude.jsonl synthetic dataset.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

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
    label_tool_counter = defaultdict(Counter)  # Track tools by label (good/bad examples)
    conversation_count = 0
    total_tool_calls = 0

    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])
                label = data.get('label', None)

                conversation_count += 1

                # Extract tools from assistant messages
                for msg in conversations:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        tools = extract_tool_calls(content)

                        for tool in tools:
                            tool_counter[tool] += 1
                            total_tool_calls += 1
                            if label is not None:
                                label_tool_counter[label][tool] += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    return {
        'tool_counter': tool_counter,
        'label_tool_counter': label_tool_counter,
        'conversation_count': conversation_count,
        'total_tool_calls': total_tool_calls
    }

def print_report(results):
    """Print a formatted coverage report."""
    print("=" * 80)
    print("TOOL USAGE COVERAGE REPORT")
    print("=" * 80)
    print()

    print(f"Total Conversations: {results['conversation_count']}")
    print(f"Total Tool Calls: {results['total_tool_calls']}")
    print(f"Unique Tools Used: {len(results['tool_counter'])}")
    print()

    print("-" * 80)
    print("TOOL FREQUENCY (sorted by count)")
    print("-" * 80)
    print(f"{'Tool Name':<50} {'Count':>10} {'Percentage':>10}")
    print("-" * 80)

    for tool, count in results['tool_counter'].most_common():
        percentage = (count / results['total_tool_calls'] * 100) if results['total_tool_calls'] > 0 else 0
        print(f"{tool:<50} {count:>10} {percentage:>9.1f}%")

    print()
    print("-" * 80)
    print("TOOL USAGE BY LABEL (Good vs Bad Examples)")
    print("-" * 80)

    # Get all unique tools
    all_tools = set()
    for label_counter in results['label_tool_counter'].values():
        all_tools.update(label_counter.keys())

    print(f"{'Tool Name':<50} {'Good (True)':>12} {'Bad (False)':>12}")
    print("-" * 80)

    for tool in sorted(all_tools):
        good_count = results['label_tool_counter'].get(True, Counter())[tool]
        bad_count = results['label_tool_counter'].get(False, Counter())[tool]
        print(f"{tool:<50} {good_count:>12} {bad_count:>12}")

    print()
    print("-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)

    good_total = sum(results['label_tool_counter'].get(True, Counter()).values())
    bad_total = sum(results['label_tool_counter'].get(False, Counter()).values())

    print(f"Total Good Examples (label=true):  {good_total}")
    print(f"Total Bad Examples (label=false):  {bad_total}")

    if good_total > 0 and bad_total > 0:
        ratio = good_total / bad_total
        print(f"Good to Bad Ratio: {ratio:.2f}:1")

    print()
    print("=" * 80)

def main():
    # Analyze the main dataset file
    dataset_path = Path('/home/user/Toolset-Training/syngen_toolset_v1.0.0_claude.jsonl')

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print(f"Analyzing: {dataset_path}")
    print()

    results = analyze_coverage(dataset_path)
    print_report(results)

if __name__ == '__main__':
    main()
