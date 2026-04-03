"""SynthChat Workspace Fixture Helpers - Convert environment fixtures to workspace data structures.

Location: SynthChat/workspace/fixture_helpers.py
Purpose: Transform EnvironmentFixture objects into workspace structures, vault trees,
         and note content lists for use in mocked workspace prompts.
Usage: Called by workspace/renderer.py during prompt construction.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..template_utils import _clean_path


def _merged_fixture_from_config(environment_config: Optional[Dict[str, Any]]):
    fixture_config = {}
    if isinstance(environment_config, dict):
        fixture_config = environment_config.get("fixture") or {}
    from shared.environments.fixture_parser import EnvironmentFixture, merge_environment_fixture

    return merge_environment_fixture(EnvironmentFixture(), fixture_config)


def _workspace_structure_from_fixture(fixture) -> List[Dict[str, Any]]:
    directory_set = {""}
    for directory in fixture.directories:
        cleaned = _clean_path(directory)
        if not cleaned:
            continue
        parts = [part for part in cleaned.split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            directory_set.add(current)

    for file_path in fixture.files:
        cleaned = _clean_path(file_path)
        parent = Path(cleaned).parent
        current = ""
        for part in parent.parts:
            if part in {"", "."}:
                continue
            current = f"{current}/{part}" if current else part
            directory_set.add(current)

    child_map: Dict[str, set[str]] = {directory: set() for directory in directory_set}
    for directory in directory_set:
        if not directory:
            continue
        parent = str(Path(directory).parent)
        parent_key = "" if parent == "." else parent.replace("\\", "/")
        child_map.setdefault(parent_key, set()).add(f"{Path(directory).name}/")

    for file_path in fixture.files:
        cleaned = _clean_path(file_path)
        parent = str(Path(cleaned).parent)
        parent_key = "" if parent == "." else parent.replace("\\", "/")
        child_map.setdefault(parent_key, set()).add(Path(cleaned).name)

    entries: List[Dict[str, Any]] = []
    for directory in sorted(directory_set):
        entries.append(
            {
                "path": f"{directory}/" if directory else "",
                "type": "folder",
                "children": sorted(child_map.get(directory, set())),
            }
        )
    return entries


def _vault_structure_text_from_fixture(fixture) -> str:
    folders = sorted(
        str(entry.get("path", "")).strip()
        for entry in _workspace_structure_from_fixture(fixture)
        if isinstance(entry, dict) and str(entry.get("path", "")).strip()
    )
    files = sorted(_clean_path(path) for path in fixture.files if _clean_path(path))
    lines = ["Folders:"]
    lines.extend(f" - {path}" for path in folders)
    lines.append("")
    lines.append("Files:")
    lines.extend(f" - {path}" for path in files)
    return "\n".join(lines).strip()


def _note_entries_from_fixture(fixture, note_paths: Optional[List[str]] = None) -> List[Dict[str, str]]:
    selected_paths = {str(path).strip() for path in (note_paths or []) if str(path).strip()}
    entries: List[Dict[str, str]] = []
    for path, content in sorted(fixture.files.items()):
        if selected_paths and path not in selected_paths:
            continue
        entries.append({"path": path, "content": content})
    return entries
