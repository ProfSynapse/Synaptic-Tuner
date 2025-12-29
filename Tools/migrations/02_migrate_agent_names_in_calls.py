#!/usr/bin/env python3
"""
Migration 02: Change agent names in tool calls.

Changes in calls[].agent field:
- agentManager → promptManager
- vaultManager → storageManager
- vaultLibrarian → searchManager

Usage:
    python 02_migrate_agent_names_in_calls.py --dry-run
    python 02_migrate_agent_names_in_calls.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    find_latest_version,
    bump_version,
    read_jsonl,
    write_jsonl,
    validate_jsonl,
    find_all_dataset_folders,
    MigrationReport
)


# Agent name migrations
AGENT_MIGRATIONS = {
    "agentManager": "promptManager",
    "vaultManager": "storageManager",
    "vaultLibrarian": "searchManager"
}


def migrate_tool_calls(tool_calls: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
    """
    Migrate agent names in tool calls.

    Args:
        tool_calls: List of tool call objects

    Returns:
        (migrated_tool_calls, was_changed)
    """
    changed = False

    for tc in tool_calls:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")

        # Parse arguments (could be string or dict)
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                continue
        else:
            args = args_str

        # Check for calls array (useTools wrapper format)
        calls = args.get("calls", [])
        for call in calls:
            agent = call.get("agent", "")
            if agent in AGENT_MIGRATIONS:
                call["agent"] = AGENT_MIGRATIONS[agent]
                changed = True

        # Update arguments back to string if it was string
        if isinstance(args_str, str):
            func["arguments"] = json.dumps(args, ensure_ascii=False)
        else:
            func["arguments"] = args

    return tool_calls, changed


def migrate_item(item: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """
    Migrate agent names in a single conversation item.

    Args:
        item: Parsed JSONL item

    Returns:
        (migrated_item, was_changed)
    """
    changed = False
    conversations = item.get("conversations", [])

    for msg in conversations:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                _, tc_changed = migrate_tool_calls(tool_calls)
                if tc_changed:
                    changed = True

    return item, changed


def process_file(input_path: Path, output_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single JSONL file.

    Args:
        input_path: Source file
        output_path: Destination file (version bumped)
        dry_run: If True, don't write output

    Returns:
        (items_total, items_changed)
    """
    items = read_jsonl(input_path)
    items_changed = 0

    for item in items:
        _, changed = migrate_item(item)
        if changed:
            items_changed += 1

    if not dry_run and items_changed > 0:
        write_jsonl(output_path, items)

        # Validate output
        is_valid, error = validate_jsonl(output_path)
        if not is_valid:
            raise ValueError(f"Output validation failed: {error}")

    return len(items), items_changed


def main():
    parser = argparse.ArgumentParser(
        description="Migrate agent names in tool calls"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files"
    )
    parser.add_argument(
        "--datasets-dir",
        default="Datasets/tools_datasets",
        help="Path to datasets directory"
    )
    parser.add_argument(
        "--folder",
        help="Process only this specific folder (e.g., thinking/contentManager)"
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    datasets_dir = project_root / args.datasets_dir

    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        sys.exit(1)

    print("Agent migrations:")
    for old, new in AGENT_MIGRATIONS.items():
        print(f"  {old} → {new}")
    print()

    report = MigrationReport("02: Agent names in calls[].agent")

    # Find folders to process
    if args.folder:
        folders = [datasets_dir / args.folder]
    else:
        folders = find_all_dataset_folders(datasets_dir)

    print(f"Processing {len(folders)} dataset folders...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")

    for folder in folders:
        if not folder.exists():
            report.add_error(folder, "Folder not found")
            continue

        latest = find_latest_version(folder)
        if not latest:
            print(f"  {folder.name}: No version files found, skipping")
            continue

        # Determine output path
        old_version = latest.name
        new_version = bump_version(old_version)
        output_path = folder / new_version

        try:
            items_total, items_changed = process_file(
                latest,
                output_path,
                dry_run=args.dry_run
            )

            if items_changed > 0:
                action = "Would create" if args.dry_run else "Created"
                print(f"  {folder.name}: {action} {new_version} ({items_changed}/{items_total} items changed)")
                report.add_change(folder, old_version, new_version, items_changed, items_total)
            else:
                print(f"  {folder.name}: No changes needed ({items_total} items)")
                report.add_skip(folder, items_total)

        except Exception as e:
            print(f"  {folder.name}: ERROR - {e}")
            report.add_error(folder, str(e))

    report.print_report()

    if args.dry_run:
        print("\n(Dry run complete. Remove --dry-run to apply changes.)")


if __name__ == "__main__":
    main()
