#!/usr/bin/env python3
"""
Migration 03: Change tool/function names in tool_calls.

Uses migrations from tool-schemas.json to update function names like:
- vaultManager_moveFolder → storageManager_move
- agentManager_createAgent → promptManager_createPrompt
- vaultLibrarian_searchContent → searchManager_searchContent

This affects the function.name field in tool_calls.

Usage:
    python 03_migrate_tool_names.py --dry-run
    python 03_migrate_tool_names.py
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


def load_tool_migrations(schema_path: Path) -> Dict[str, str]:
    """Load tool name migrations from tool-schemas.json"""
    with open(schema_path) as f:
        schema = json.load(f)
    return schema.get("migrations", {}).get("tools", {})


def migrate_tool_calls(tool_calls: List[Dict[str, Any]], tool_migrations: Dict[str, str]) -> tuple[List[Dict[str, Any]], bool]:
    """
    Migrate tool/function names in tool calls.

    Args:
        tool_calls: List of tool call objects
        tool_migrations: Dict mapping old tool names to new names

    Returns:
        (migrated_tool_calls, was_changed)
    """
    changed = False

    for tc in tool_calls:
        func = tc.get("function", {})
        func_name = func.get("name", "")

        # Check if function name needs migration
        if func_name in tool_migrations:
            func["name"] = tool_migrations[func_name]
            changed = True

    return tool_calls, changed


def migrate_item(item: Dict[str, Any], tool_migrations: Dict[str, str]) -> tuple[Dict[str, Any], bool]:
    """
    Migrate tool names in a single conversation item.

    Args:
        item: Parsed JSONL item
        tool_migrations: Dict mapping old tool names to new names

    Returns:
        (migrated_item, was_changed)
    """
    changed = False
    conversations = item.get("conversations", [])

    for msg in conversations:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                _, tc_changed = migrate_tool_calls(tool_calls, tool_migrations)
                if tc_changed:
                    changed = True

    return item, changed


def process_file(input_path: Path, output_path: Path, tool_migrations: Dict[str, str], dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single JSONL file.

    Args:
        input_path: Source file
        output_path: Destination file (version bumped)
        tool_migrations: Dict mapping old tool names to new names
        dry_run: If True, don't write output

    Returns:
        (items_total, items_changed)
    """
    items = read_jsonl(input_path)
    items_changed = 0

    for item in items:
        _, changed = migrate_item(item, tool_migrations)
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
        description="Migrate tool/function names in tool_calls"
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
        "--schema",
        default="tool-schemas.json",
        help="Path to tool-schemas.json"
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
    schema_path = project_root / args.schema

    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        sys.exit(1)

    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}")
        sys.exit(1)

    # Load migrations
    tool_migrations = load_tool_migrations(schema_path)
    print(f"Loaded {len(tool_migrations)} tool migrations from {schema_path.name}")
    print("\nTool name migrations:")
    for old, new in sorted(tool_migrations.items()):
        print(f"  {old} → {new}")
    print()

    report = MigrationReport("03: Tool/function names")

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
                tool_migrations,
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
