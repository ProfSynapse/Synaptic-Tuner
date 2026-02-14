"""Parse workspace fixture information from system prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class EnvironmentFixture:
    """Environment fixture derived from scenario/system context."""

    directories: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)

    def has_content(self) -> bool:
        return bool(self.directories or self.files)


def parse_environment_fixture(system_prompt: str) -> EnvironmentFixture:
    """Build an EnvironmentFixture from evaluator/synthchat system context."""
    collector = _FixtureCollector()
    prompt = system_prompt or ""

    selected_workspace = _extract_tag_content(prompt, "selected_workspace")
    if selected_workspace:
        _hydrate_from_selected_workspace(selected_workspace, collector)

    if not collector.has_content():
        vault_structure = _extract_tag_content(prompt, "vault_structure")
        if vault_structure:
            _hydrate_from_vault_structure(vault_structure, collector)

    if not collector.has_content():
        collector.add_dir(".")

    return EnvironmentFixture(
        directories=sorted(collector.directories),
        files=dict(sorted(collector.files.items())),
    )


class _FixtureCollector:
    """Mutable fixture collector with path normalization helpers."""

    def __init__(self):
        self.directories: Set[str] = set()
        self.files: Dict[str, str] = {}

    def has_content(self) -> bool:
        return bool(self.directories or self.files)

    def add_dir(self, path: str) -> None:
        normalized = _normalize_path(path)
        if normalized:
            self.directories.add(normalized)

    def add_file(self, path: str, content: str = "") -> None:
        normalized = _normalize_path(path)
        if not normalized:
            return
        parent = _parent_dir(normalized)
        if parent:
            self.directories.add(parent)
        self.files[normalized] = content


def _extract_tag_content(text: str, tag_name: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag_name}[^>]*>(.*?)</{tag_name}>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text or "")
    return match.group(1).strip() if match else None


def _hydrate_from_selected_workspace(section: str, collector: _FixtureCollector) -> None:
    json_blob = _extract_first_json_object(section)
    if not json_blob:
        return

    try:
        data = json.loads(json_blob)
    except json.JSONDecodeError:
        return

    workspace_structure = data.get("workspaceStructure", [])
    if not isinstance(workspace_structure, list):
        return

    for entry in workspace_structure:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        entry_type = str(entry.get("type", "folder")).lower()
        normalized_path = _normalize_path(path)

        if entry_type == "folder":
            collector.add_dir(normalized_path or ".")
        elif entry_type == "file":
            collector.add_file(normalized_path or "untitled.md")

        children = entry.get("children", [])
        if not isinstance(children, list):
            continue

        base = normalized_path
        for child in children:
            if not isinstance(child, str):
                continue
            child_path = _join_paths(base, child)
            if child.endswith("/"):
                collector.add_dir(child_path)
            else:
                collector.add_file(child_path)


def _hydrate_from_vault_structure(section: str, collector: _FixtureCollector) -> None:
    current_section = None
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("folders:"):
            current_section = "folders"
            continue
        if lower.startswith("files:"):
            current_section = "files"
            continue
        if not line.startswith("-"):
            continue

        item = line[1:].strip()
        if current_section == "folders":
            collector.add_dir(item)
        elif current_section == "files":
            collector.add_file(item)


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _normalize_path(path: str) -> str:
    p = str(path or "").strip().replace("\\", "/")
    p = p.lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    if p.endswith("/") and p != ".":
        p = p[:-1]
    return p


def _parent_dir(path: str) -> Optional[str]:
    if "/" not in path:
        return None
    return path.rsplit("/", 1)[0]


def _join_paths(parent: str, child: str) -> str:
    parent_norm = _normalize_path(parent)
    child_norm = _normalize_path(child)
    if not parent_norm:
        return child_norm
    if not child_norm:
        return parent_norm
    return f"{parent_norm}/{child_norm}"

