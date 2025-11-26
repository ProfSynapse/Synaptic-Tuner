#!/usr/bin/env python3
"""
Merge all tools_datasets v1.2 files into a single SFT dataset.

v1.2 includes:
- Original tool-call examples from v1.1
- NEW: Text-only clarification examples (model asks questions)
- NEW: Text-only summary examples (model summarizes results)
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Input files (v1.2 with clarification and summary examples)
INPUT_FILES = [
    "agentManager/tools_v1.2.jsonl",
    "contentManager/tools_v1.2.jsonl",
    "memoryManager/tools_v1.2.jsonl",
    "vaultLibrarian/tools_v1.2.jsonl",
    "vaultManager/tools_v1.2.jsonl",
]

# Output file
OUTPUT_DIR = Path(__file__).parent.parent
OUTPUT_FILE = OUTPUT_DIR / "syngen_tools_sft_v1.2.jsonl"


def count_example_types(examples):
    """Count tool-call vs text-only examples."""
    tool_call_count = 0
    text_only_count = 0

    for ex in examples:
        conversations = ex.get("conversations", [])
        has_tool_call = False
        for msg in conversations:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                has_tool_call = True
                break
        if has_tool_call:
            tool_call_count += 1
        else:
            text_only_count += 1

    return tool_call_count, text_only_count


def main():
    base_dir = Path(__file__).parent

    all_examples = []
    stats = {}

    print("Loading v1.2 datasets with clarification and summary examples...\n")

    for input_file in INPUT_FILES:
        file_path = base_dir / input_file
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        manager = input_file.split("/")[0]
        examples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    # Only include positive examples (label=True or no label)
                    label = example.get("label")
                    if label is False:
                        continue
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {input_file}: {e}")

        tool_calls, text_only = count_example_types(examples)
        stats[manager] = {
            "total": len(examples),
            "tool_calls": tool_calls,
            "text_only": text_only
        }
        print(f"  {manager}: {len(examples)} examples ({tool_calls} tool-call, {text_only} text-only)")
        all_examples.extend(examples)

    # Shuffle for training diversity
    random.seed(42)
    random.shuffle(all_examples)

    total_tool_calls = sum(s["tool_calls"] for s in stats.values())
    total_text_only = sum(s["text_only"] for s in stats.values())

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"  Tool-call examples: {total_tool_calls}")
    print(f"  Text-only examples: {total_text_only} (clarification + summary)")
    print(f"  Text-only ratio: {total_text_only / len(all_examples) * 100:.1f}%")

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nOutput: {OUTPUT_FILE}")

    # Also create metadata file
    metadata = {
        "created": datetime.now().isoformat(),
        "version": "1.2",
        "source_files": INPUT_FILES,
        "total_examples": len(all_examples),
        "tool_call_examples": total_tool_calls,
        "text_only_examples": total_text_only,
        "text_only_ratio": f"{total_text_only / len(all_examples) * 100:.1f}%",
        "per_manager": stats,
        "features": [
            "System prompts with <session_context>",
            "System prompts with <available_workspaces>",
            "System prompts with <available_agents> (where applicable)",
            "IDs in tool calls match system prompt context",
            "NEW: Text-only clarification examples (ask before acting)",
            "NEW: Text-only summary examples (summarize results)"
        ],
        "format": "OpenAI tool_calls format with system message",
        "notes": "v1.2 adds ~365 handcrafted text-only examples for clarification and summary responses"
    }

    metadata_file = OUTPUT_FILE.with_suffix(".meta.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
