#!/usr/bin/env python3
"""
Split tools SFT dataset into agent-based folders.

This script:
1. Reads the source JSONL file
2. Extracts agent name from tool_calls[0].function.name
3. Groups examples by agent (vaultManager, contentManager, etc.)
4. Creates agent folders under tools_datasets/
5. Writes tools_v1.0.jsonl for each agent
6. Reports statistics and any invalid entries
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple


# Valid agent prefixes
VALID_AGENTS = {
    "vaultManager",
    "contentManager",
    "memoryManager",
    "vaultLibrarian",
    "agentManager"
}


def extract_agent_name(example: Dict) -> Tuple[str, str]:
    """
    Extract agent name from tool call.

    Returns:
        Tuple of (agent_name, tool_name) or (None, error_message) if invalid
    """
    try:
        conversations = example.get('conversations', [])
        if len(conversations) < 2:
            return None, "Less than 2 conversation turns"

        assistant_msg = conversations[1]
        tool_calls = assistant_msg.get('tool_calls', [])

        if not tool_calls:
            return None, "No tool_calls in assistant message"

        tool_name = tool_calls[0].get('function', {}).get('name', '')

        if not tool_name:
            return None, "Empty tool name"

        # Extract agent prefix (e.g., "contentManager_appendContent" -> "contentManager")
        if '_' not in tool_name:
            return None, f"Invalid tool name format: {tool_name}"

        agent = tool_name.split('_')[0]

        if agent not in VALID_AGENTS:
            return None, f"Unknown agent: {agent} (from {tool_name})"

        return agent, tool_name

    except Exception as e:
        return None, f"Error extracting agent: {str(e)}"


def load_and_split_dataset(source_path: Path) -> Tuple[Dict[str, List[Dict]], List[Tuple[int, str, Dict]]]:
    """
    Load source dataset and split by agent.

    Returns:
        Tuple of (agent_examples dict, invalid_entries list)
    """
    agent_examples = defaultdict(list)
    invalid_entries = []

    with open(source_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
                agent, info = extract_agent_name(example)

                if agent:
                    agent_examples[agent].append(example)
                else:
                    invalid_entries.append((line_num, info, example))

            except json.JSONDecodeError as e:
                invalid_entries.append((line_num, f"JSON decode error: {e}", None))

    return dict(agent_examples), invalid_entries


def write_agent_dataset(agent: str, examples: List[Dict], output_dir: Path) -> Path:
    """Write examples to agent's tools_v1.0.jsonl file."""
    agent_dir = output_dir / agent
    agent_dir.mkdir(parents=True, exist_ok=True)

    output_file = agent_dir / "tools_v1.0.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    return output_file


def count_tools(examples: List[Dict]) -> Dict[str, int]:
    """Count occurrences of each tool in examples."""
    tool_counts = defaultdict(int)

    for example in examples:
        _, tool_name = extract_agent_name(example)
        if tool_name:
            tool_counts[tool_name] += 1

    return dict(sorted(tool_counts.items(), key=lambda x: -x[1]))


def main():
    # Configuration
    base_dir = Path(__file__).parent.parent
    source_file = base_dir / "syngen_tools_sft_11.24.25_cleaned.jsonl"
    output_dir = base_dir / "tools_datasets"

    print(f"Source: {source_file}")
    print(f"Output: {output_dir}")
    print()

    # Load and split
    print("Loading and splitting dataset...")
    agent_examples, invalid_entries = load_and_split_dataset(source_file)

    # Report invalid entries
    if invalid_entries:
        print(f"\n{len(invalid_entries)} invalid entries found:")
        for line_num, reason, example in invalid_entries[:10]:  # Show first 10
            print(f"  Line {line_num}: {reason}")
        if len(invalid_entries) > 10:
            print(f"  ... and {len(invalid_entries) - 10} more")

    # Write agent datasets
    print("\nWriting agent datasets:")
    total = 0
    stats = {}

    for agent in VALID_AGENTS:
        examples = agent_examples.get(agent, [])
        if examples:
            output_file = write_agent_dataset(agent, examples, output_dir)
            tool_counts = count_tools(examples)

            stats[agent] = {
                "total": len(examples),
                "tools": tool_counts
            }

            print(f"  {agent}: {len(examples)} examples -> {output_file.relative_to(base_dir)}")
            total += len(examples)

    # Write split metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "source": str(source_file.name),
        "total_examples": total,
        "invalid_entries": len(invalid_entries),
        "agent_stats": stats
    }

    metadata_file = output_dir / "split_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nSummary:")
    print(f"  Total valid examples: {total}")
    print(f"  Invalid entries: {len(invalid_entries)}")
    print(f"  Metadata: {metadata_file.relative_to(base_dir)}")

    print("\nTool distribution per agent:")
    for agent, data in sorted(stats.items()):
        print(f"\n  {agent} ({data['total']} examples):")
        for tool, count in list(data['tools'].items())[:5]:  # Top 5 tools
            print(f"    {tool}: {count}")
        if len(data['tools']) > 5:
            print(f"    ... and {len(data['tools']) - 5} more tools")

    print("\nDone!")


if __name__ == "__main__":
    main()
