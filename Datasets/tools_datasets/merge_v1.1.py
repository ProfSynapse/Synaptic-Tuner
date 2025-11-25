#!/usr/bin/env python3
"""
Merge all tools_datasets v1.1 files into a single SFT dataset.

This merges the context-injected datasets (with system prompts) into a single
training file for SFT.
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Input files
INPUT_FILES = [
    "agentManager/tools_v1.1.jsonl",
    "contentManager/tools_v1.1.jsonl",
    "memoryManager/tools_v1.1.jsonl",
    "vaultLibrarian/tools_v1.1.jsonl",
    "vaultManager/tools_v1.1.jsonl",
]

# Output file
OUTPUT_DIR = Path(__file__).parent.parent
OUTPUT_FILE = OUTPUT_DIR / f"syngen_tools_sft_{datetime.now().strftime('%m.%d.%y')}_context.jsonl"

def main():
    base_dir = Path(__file__).parent

    all_examples = []
    stats = {}

    for input_file in INPUT_FILES:
        file_path = base_dir / input_file
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        manager = input_file.split("/")[0]
        count = 0

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
                    all_examples.append(example)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {input_file}: {e}")

        stats[manager] = count
        print(f"  {manager}: {count} examples")

    # Shuffle for training diversity
    random.seed(42)
    random.shuffle(all_examples)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"Output: {OUTPUT_FILE}")

    # Also create metadata file
    metadata = {
        "created": datetime.now().isoformat(),
        "source_files": INPUT_FILES,
        "total_examples": len(all_examples),
        "per_manager": stats,
        "features": [
            "System prompts with <session_context>",
            "System prompts with <available_workspaces>",
            "System prompts with <available_agents> (where applicable)",
            "IDs in tool calls match system prompt context"
        ],
        "format": "OpenAI tool_calls format with system message"
    }

    metadata_file = OUTPUT_FILE.with_suffix(".meta.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
