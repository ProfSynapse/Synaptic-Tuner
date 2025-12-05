#!/usr/bin/env python3
"""
Split SelfPlay generated dataset into individual category files.

This script:
1. Reads a combined SelfPlay JSONL file
2. Groups examples by category (tool name or behavior name)
3. Creates/appends to individual dataset files in proper folders:
   - Tools: Datasets/tools_datasets/{agent_name}/tools_vX.X_YYYYMMDD_HHMMSS.jsonl
   - Behaviors: Datasets/behavior_datasets/{behavior_name}/pairs_vX.X_YYYYMMDD_HHMMSS.jsonl
4. Creates versioned backups of existing datasets before appending
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[Dict], file_path: Path):
    """Save list of dicts to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def create_backup(file_path: Path) -> Path:
    """Create timestamped backup of existing file."""
    if not file_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    backup_path = file_path.parent / "backups" / backup_name

    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy existing file to backup
    existing = load_jsonl(file_path)
    save_jsonl(existing, backup_path)

    return backup_path


def get_agent_from_tool(tool_name: str) -> str:
    """
    Extract agent name from tool name.

    Tool names follow pattern: {agent}_{toolName}
    Example: vaultManager_createFolder -> vaultManager
    """
    if '_' in tool_name:
        return tool_name.split('_')[0]
    return "unknown"


def get_next_version(directory: Path, prefix: str) -> str:
    """
    Get next version number for files in directory.

    Args:
        directory: Directory to check
        prefix: File prefix (e.g., "tools" or "pairs")

    Returns:
        Version string like "v1.6"
    """
    if not directory.exists():
        return "v1.0"

    # Find all files matching pattern
    pattern = re.compile(rf"{prefix}_v(\d+)\.(\d+)")
    max_major = 1
    max_minor = -1

    for file_path in directory.glob(f"{prefix}_v*"):
        match = pattern.search(file_path.name)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            if major > max_major or (major == max_major and minor > max_minor):
                max_major = major
                max_minor = minor

    # Increment minor version
    return f"v{max_major}.{max_minor + 1}"


def determine_output_path(
    category: str,
    is_behavior: bool,
    base_dir: Path,
    timestamp: str
) -> Tuple[Path, str]:
    """
    Determine output path and filename for a category.

    Args:
        category: Tool name or behavior name
        is_behavior: Whether this is a behavior (vs tool)
        base_dir: Base Datasets directory
        timestamp: Timestamp string (YYYYMMDD_HHMMSS)

    Returns:
        Tuple of (output_path, filename)
    """
    if is_behavior:
        # Behaviors: Datasets/behavior_datasets/{behavior}/pairs_vX.X_timestamp.jsonl
        folder = base_dir / "behavior_datasets" / category
        folder.mkdir(parents=True, exist_ok=True)

        version = get_next_version(folder, "pairs")
        filename = f"pairs_{version}_{timestamp}.jsonl"

    else:
        # Tools: Datasets/tools_datasets/{agent}/tools_vX.X_timestamp.jsonl
        agent = get_agent_from_tool(category)
        folder = base_dir / "tools_datasets" / agent
        folder.mkdir(parents=True, exist_ok=True)

        version = get_next_version(folder, "tools")
        filename = f"tools_{version}_{timestamp}.jsonl"

    return folder / filename, filename


def group_by_category(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group examples by their category (tool or behavior)."""
    grouped = defaultdict(list)

    for example in examples:
        metadata = example.get("metadata", {})
        category = metadata.get("category")

        if category:
            grouped[category].append(example)
        else:
            # Fallback: try to determine category from metadata
            if "tool" in metadata:
                category = metadata["tool"]
            elif "behavior" in metadata:
                category = metadata["behavior"]
            else:
                category = "uncategorized"

            grouped[category].append(example)

    return dict(grouped)


def is_behavior(category: str) -> bool:
    """
    Determine if category is a behavior (vs tool).

    Behaviors don't have underscores in them (e.g., "intellectual_humility")
    Tools have format: {agent}_{toolName} (e.g., "vaultManager_createFolder")

    For edge cases, check against known behaviors list.
    """
    known_behaviors = [
        "intellectual_humility",
        "error_recovery",
        "verification_before_action",
        "workspace_awareness",
        "strategic_tool_selection",
        "ask_first",
        "context_continuity",
        "context_efficiency",
        "execute_prompt_usage",
        "response_patterns"
    ]

    if category in known_behaviors:
        return True

    # Heuristic: behaviors don't have underscore pattern of agent_toolName
    # They use snake_case for multi-word names
    return category in known_behaviors or not (
        '_' in category and
        category.split('_')[0] in [
            'vaultManager', 'contentManager', 'memoryManager',
            'vaultLibrarian', 'agentManager'
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split SelfPlay dataset into individual category files"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file containing all generated examples"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="Datasets",
        help="Base Datasets directory (default: Datasets)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating files"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    datasets_dir = Path(args.datasets_dir)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("SELFPLAY DATASET SPLITTER")
    print("=" * 70)
    print(f"\nInput file: {input_path}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")

    # Load examples
    print(f"\n📖 Loading examples from {input_path}...")
    examples = load_jsonl(input_path)
    print(f"   Loaded {len(examples)} examples")

    # Group by category
    print(f"\n📊 Grouping by category...")
    grouped = group_by_category(examples)

    print(f"   Found {len(grouped)} categories:")
    for category, cat_examples in sorted(grouped.items(), key=lambda x: -len(x[1])):
        cat_type = "behavior" if is_behavior(category) else "tool"
        print(f"     - {category} ({cat_type}): {len(cat_examples)} examples")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No files will be created")
        print("\nWould create the following files:")
        for category in sorted(grouped.keys()):
            cat_is_behavior = is_behavior(category)
            output_path, filename = determine_output_path(
                category, cat_is_behavior, datasets_dir, timestamp
            )
            print(f"  Create: {output_path}")
        return

    # Process each category
    print(f"\n💾 Writing category files...")

    stats = {
        "created": 0,
        "total_examples": 0,
        "tools": 0,
        "behaviors": 0
    }

    for category, cat_examples in sorted(grouped.items()):
        cat_is_behavior = is_behavior(category)

        # Determine output path
        output_path, filename = determine_output_path(
            category, cat_is_behavior, datasets_dir, timestamp
        )

        # Save examples
        save_jsonl(cat_examples, output_path)

        cat_type = "behavior" if cat_is_behavior else "tool"
        print(f"   ✨ Created {filename} ({cat_type}, {len(cat_examples)} examples)")

        stats["created"] += 1
        stats["total_examples"] += len(cat_examples)
        if cat_is_behavior:
            stats["behaviors"] += 1
        else:
            stats["tools"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("✅ SPLITTING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"   Files created: {stats['created']}")
    print(f"     - Tool datasets: {stats['tools']}")
    print(f"     - Behavior datasets: {stats['behaviors']}")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"\n📁 Base directory: {datasets_dir}")
    print(f"   Tools: {datasets_dir / 'tools_datasets'}")
    print(f"   Behaviors: {datasets_dir / 'behavior_datasets'}")


if __name__ == "__main__":
    main()
