#!/usr/bin/env python3
"""
Migrate old tool/agent names to new names in JSONL datasets.

Uses migrations from tool-schemas.json:
- agentManager → promptManager
- vaultManager → storageManager
- vaultLibrarian → searchManager
- <available_agents> → <available_prompts>
"""

import json
import re
import sys
from pathlib import Path


def load_migrations(schema_path: Path) -> dict:
    """Load migrations from tool-schemas.json"""
    with open(schema_path) as f:
        schema = json.load(f)
    return schema.get("migrations", {})


def migrate_content(content: str, migrations: dict) -> str:
    """Apply migrations to content string."""

    # Agent name migrations (in "agent": "xxx" patterns)
    for old, new in migrations.get("agents", {}).items():
        # Match "agent": "agentManager" patterns
        content = re.sub(
            rf'"agent"\s*:\s*"{old}"',
            f'"agent": "{new}"',
            content
        )
        # Match agent in tool names like agentManager_createAgent
        content = content.replace(f'"{old}_', f'"{new}_')
        content = content.replace(f'{old}_', f'{new}_')

    # Tool name migrations
    for old, new in migrations.get("tools", {}).items():
        content = content.replace(f'"{old}"', f'"{new}"')
        content = content.replace(old, new)

    # XML tag migrations
    content = content.replace("<available_agents>", "<available_prompts>")
    content = content.replace("</available_agents>", "</available_prompts>")

    return content


def migrate_file(file_path: Path, migrations: dict, dry_run: bool = False) -> tuple[int, int]:
    """Migrate a single JSONL file. Returns (lines_processed, lines_changed)."""

    lines_processed = 0
    lines_changed = 0
    new_lines = []

    with open(file_path) as f:
        for line in f:
            lines_processed += 1
            original = line
            migrated = migrate_content(line, migrations)
            new_lines.append(migrated)
            if migrated != original:
                lines_changed += 1

    if not dry_run and lines_changed > 0:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return lines_processed, lines_changed


def rename_folder(old_path: Path, new_name: str, dry_run: bool = False) -> bool:
    """Rename a folder."""
    if not old_path.exists():
        return False

    new_path = old_path.parent / new_name
    if new_path.exists():
        print(f"  WARNING: {new_path} already exists, skipping rename")
        return False

    if not dry_run:
        old_path.rename(new_path)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate tool/agent names in datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying")
    parser.add_argument("--datasets-dir", default="Datasets/tools_datasets", help="Datasets directory")
    parser.add_argument("--schema", default="tool-schemas.json", help="Path to tool-schemas.json")
    parser.add_argument("--rename-folders", action="store_true", help="Also rename agent folders")
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    schema_path = project_root / args.schema
    datasets_dir = project_root / args.datasets_dir

    if not schema_path.exists():
        print(f"ERROR: Schema not found: {schema_path}")
        sys.exit(1)

    if not datasets_dir.exists():
        print(f"ERROR: Datasets dir not found: {datasets_dir}")
        sys.exit(1)

    migrations = load_migrations(schema_path)
    print(f"Loaded migrations from {schema_path}")
    print(f"  Agents: {list(migrations.get('agents', {}).keys())}")
    print(f"  Tools: {len(migrations.get('tools', {}))} mappings")
    print()

    # Folder renames
    folder_renames = {
        "agentManager": "promptManager",
        "vaultManager": "storageManager",
        "vaultLibrarian": "searchManager"
    }

    if args.rename_folders:
        print("=== Folder Renames ===")
        for old_name, new_name in folder_renames.items():
            for subdir in ["thinking", "non_thinking"]:
                old_path = datasets_dir / subdir / old_name
                if old_path.exists():
                    action = "Would rename" if args.dry_run else "Renaming"
                    print(f"  {action}: {old_path} → {new_name}")
                    rename_folder(old_path, new_name, args.dry_run)
        print()

    # Content migrations
    print("=== Content Migrations ===")
    total_files = 0
    total_lines = 0
    total_changed = 0

    for jsonl_file in datasets_dir.rglob("*.jsonl"):
        lines, changed = migrate_file(jsonl_file, migrations, args.dry_run)
        total_files += 1
        total_lines += lines
        total_changed += changed

        if changed > 0:
            action = "Would update" if args.dry_run else "Updated"
            print(f"  {action}: {jsonl_file.relative_to(project_root)} ({changed}/{lines} lines)")

    print()
    print(f"=== Summary ===")
    print(f"Files processed: {total_files}")
    print(f"Lines processed: {total_lines}")
    print(f"Lines changed: {total_changed}")

    if args.dry_run:
        print("\n(Dry run - no changes made. Remove --dry-run to apply.)")


if __name__ == "__main__":
    main()
