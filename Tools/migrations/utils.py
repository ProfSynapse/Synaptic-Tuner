"""
Shared utilities for migration scripts.

Provides:
- Version detection and bumping
- JSONL read/write with validation
- Change reporting
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def find_latest_version(folder: Path, pattern: str = "tools_v*.jsonl") -> Optional[Path]:
    """
    Find the latest version file in a folder.

    Args:
        folder: Directory to search
        pattern: Glob pattern (default: tools_v*.jsonl)

    Returns:
        Path to latest version file, or None if no matches
    """
    files = list(folder.glob(pattern))

    # Filter out failed/passed_only variants
    files = [f for f in files if not any(x in f.name for x in ['.failed', '_failed', '_passed_only', '_automated', '_improved', '_seeds', '_generated', '_to_improve'])]

    if not files:
        return None

    # Extract version numbers and sort
    def version_key(p: Path) -> Tuple[int, int]:
        match = re.search(r'v(\d+)\.(\d+)', p.name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)

    files.sort(key=version_key, reverse=True)
    return files[0]


def bump_version(version_str: str) -> str:
    """
    Bump minor version: v1.8 → v1.9, v2.0 → v2.1

    Args:
        version_str: Version string like "v1.8" or "tools_v1.8.jsonl"

    Returns:
        Bumped version string in same format
    """
    match = re.search(r'v(\d+)\.(\d+)', version_str)
    if not match:
        raise ValueError(f"Cannot parse version from: {version_str}")

    major = int(match.group(1))
    minor = int(match.group(2))
    new_minor = minor + 1

    return re.sub(r'v\d+\.\d+', f'v{major}.{new_minor}', version_str)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read JSONL file, returning list of parsed JSON objects.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Raises:
        json.JSONDecodeError: If any line is invalid JSON
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Line {line_num}: {e.msg}",
                    e.doc,
                    e.pos
                )
    return items


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> int:
    """
    Write list of objects to JSONL file.

    Args:
        path: Output path
        items: List of JSON-serializable objects

    Returns:
        Number of lines written
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return len(items)


def validate_jsonl(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that a JSONL file contains valid JSON on each line.

    Args:
        path: Path to JSONL file

    Returns:
        (is_valid, error_message)
    """
    try:
        read_jsonl(path)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def get_system_prompt(item: Dict[str, Any]) -> Optional[str]:
    """
    Extract system prompt from a conversation item.

    Args:
        item: Parsed JSONL item with 'conversations' key

    Returns:
        System prompt content, or None if not found
    """
    conversations = item.get('conversations', [])
    for msg in conversations:
        if msg.get('role') == 'system':
            return msg.get('content', '')
    return None


def set_system_prompt(item: Dict[str, Any], new_content: str) -> bool:
    """
    Update system prompt in a conversation item.

    Args:
        item: Parsed JSONL item with 'conversations' key
        new_content: New system prompt content

    Returns:
        True if updated, False if no system message found
    """
    conversations = item.get('conversations', [])
    for msg in conversations:
        if msg.get('role') == 'system':
            msg['content'] = new_content
            return True
    return False


def find_all_dataset_folders(datasets_dir: Path) -> List[Path]:
    """
    Find all dataset folders (agent directories) in the datasets directory.

    Args:
        datasets_dir: Root datasets directory (e.g., Datasets/tools_datasets)

    Returns:
        List of paths to agent folders (e.g., .../thinking/contentManager)
    """
    folders = []

    for subdir in ['thinking', 'non_thinking']:
        subdir_path = datasets_dir / subdir
        if not subdir_path.exists():
            continue

        for agent_folder in subdir_path.iterdir():
            if agent_folder.is_dir():
                folders.append(agent_folder)

    return sorted(folders)


class MigrationReport:
    """Track and report migration changes."""

    def __init__(self, name: str):
        self.name = name
        self.files_processed = 0
        self.files_changed = 0
        self.items_processed = 0
        self.items_changed = 0
        self.errors = []
        self.changes = []  # List of (file, old_version, new_version, items_changed)

    def add_change(self, file_path: Path, old_version: str, new_version: str, items_changed: int, items_total: int):
        """Record a file that was changed."""
        self.files_processed += 1
        self.files_changed += 1
        self.items_processed += items_total
        self.items_changed += items_changed
        self.changes.append((file_path, old_version, new_version, items_changed, items_total))

    def add_skip(self, file_path: Path, items_total: int):
        """Record a file that was skipped (no changes needed)."""
        self.files_processed += 1
        self.items_processed += items_total

    def add_error(self, file_path: Path, error: str):
        """Record an error."""
        self.errors.append((file_path, error))

    def print_report(self):
        """Print summary report."""
        print(f"\n{'='*60}")
        print(f"Migration: {self.name}")
        print(f"{'='*60}")

        if self.changes:
            print(f"\nFiles changed ({self.files_changed}):")
            for file_path, old_v, new_v, items, total in self.changes:
                print(f"  {file_path.name}: {old_v} → {new_v} ({items}/{total} items)")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for file_path, error in self.errors:
                print(f"  {file_path}: {error}")

        print(f"\nSummary:")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Files changed: {self.files_changed}")
        print(f"  Items processed: {self.items_processed}")
        print(f"  Items changed: {self.items_changed}")
        print(f"  Errors: {len(self.errors)}")
