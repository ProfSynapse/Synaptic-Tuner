#!/usr/bin/env python3
"""
Migration 04: Change old tool/agent names in all message content.

This catches references that the surgical migrations missed:
- Tool names in system prompts (e.g., "use agentManager_toggleAgent")
- Tool names in user messages (e.g., tool result listings)
- Agent names in descriptions

Usage:
    python 04_migrate_content_references.py --dry-run
    python 04_migrate_content_references.py
"""

import argparse
import json
import re
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


def load_migrations(schema_path: Path) -> dict:
    """Load all migrations from tool-schemas.json"""
    with open(schema_path) as f:
        schema = json.load(f)
    return schema.get("migrations", {})


def migrate_content(content: str, migrations: dict) -> tuple[str, bool]:
    """
    Apply all migrations to content string.

    Returns:
        (migrated_content, was_changed)
    """
    original = content

    # Tool name migrations (e.g., agentManager_createAgent -> promptManager_createPrompt)
    for old, new in migrations.get("tools", {}).items():
        content = content.replace(old, new)

    # Agent name migrations in various patterns
    for old, new in migrations.get("agents", {}).items():
        # Match "agent": "agentManager" patterns (in JSON)
        content = re.sub(
            rf'"agent"\s*:\s*"{old}"',
            f'"agent": "{new}"',
            content
        )
        # Match agent_xxx tool prefixes that weren't caught by tool migrations
        content = re.sub(
            rf'\b{old}_(\w+)',
            rf'{new}_\1',
            content
        )

    # XML tag migrations
    content = content.replace("<available_agents>", "<available_prompts>")
    content = content.replace("</available_agents>", "</available_prompts>")

    return content, content != original


def migrate_item(item: Dict[str, Any], migrations: dict) -> tuple[Dict[str, Any], bool]:
    """
    Migrate all content in a conversation item.

    Returns:
        (migrated_item, was_changed)
    """
    changed = False
    conversations = item.get("conversations", [])

    for msg in conversations:
        content = msg.get("content", "")
        if content:
            new_content, msg_changed = migrate_content(content, migrations)
            if msg_changed:
                msg["content"] = new_content
                changed = True

    return item, changed


def process_file(input_path: Path, output_path: Path, migrations: dict, dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single JSONL file.

    Returns:
        (items_total, items_changed)
    """
    items = read_jsonl(input_path)
    items_changed = 0

    for item in items:
        _, changed = migrate_item(item, migrations)
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
        description="Migrate old tool/agent names in all message content"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files"
    )
    parser.add_argument(
        "--datasets-dir",
        default="Datasets/tools_datasets/non_thinking",
        help="Path to datasets directory (default: non_thinking only)"
    )
    parser.add_argument(
        "--schema",
        default="tool-schemas.json",
        help="Path to tool-schemas.json"
    )
    parser.add_argument(
        "--folder",
        help="Process only this specific folder (e.g., agentManager)"
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
    migrations = load_migrations(schema_path)
    print(f"Loaded migrations from {schema_path.name}")
    print(f"  Agent migrations: {len(migrations.get('agents', {}))}")
    print(f"  Tool migrations: {len(migrations.get('tools', {}))}")
    print()

    report = MigrationReport("04: Content references (all message content)")

    # Find folders to process (non_thinking structure is flat)
    if args.folder:
        folders = [datasets_dir / args.folder]
    else:
        folders = [f for f in datasets_dir.iterdir() if f.is_dir()]

    print(f"Processing {len(folders)} dataset folders...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")

    for folder in sorted(folders):
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
                migrations,
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
