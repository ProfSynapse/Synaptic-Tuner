#!/usr/bin/env python3
"""
Migration 01: Change <available_agents> to <available_prompts> in system prompts.

This affects ALL datasets since every system prompt has this tag.

Changes:
- <available_agents> → <available_prompts>
- </available_agents> → </available_prompts>

Usage:
    python 01_migrate_available_agents_tag.py --dry-run
    python 01_migrate_available_agents_tag.py
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    find_latest_version,
    bump_version,
    read_jsonl,
    write_jsonl,
    validate_jsonl,
    get_system_prompt,
    set_system_prompt,
    find_all_dataset_folders,
    MigrationReport
)


def migrate_system_prompt(content: str) -> tuple[str, bool]:
    """
    Migrate <available_agents> to <available_prompts> in system prompt.

    Args:
        content: System prompt content

    Returns:
        (migrated_content, was_changed)
    """
    if not content:
        return content, False

    original = content

    # Replace opening tag
    content = content.replace('<available_agents>', '<available_prompts>')

    # Replace closing tag
    content = content.replace('</available_agents>', '</available_prompts>')

    return content, content != original


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
        system_prompt = get_system_prompt(item)
        if system_prompt:
            new_prompt, changed = migrate_system_prompt(system_prompt)
            if changed:
                set_system_prompt(item, new_prompt)
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
        description="Migrate <available_agents> to <available_prompts> in all datasets"
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

    report = MigrationReport("01: <available_agents> → <available_prompts>")

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
