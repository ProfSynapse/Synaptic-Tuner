#!/usr/bin/env python3
"""Inventory SynthChat config references that need CLI-schema alignment."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent))

from cli_schema_utils import get_repo_root, load_target_catalog


SCAN_ROOTS = (
    "SynthChat/config",
    "SynthChat/scenarios",
    "SynthChat/rubrics",
)


SPECIAL_PATTERNS: Dict[str, str] = {
    "useTools_wrapper": r"\buseTools\b",
    "available_agents_tag": r"\bavailable_agents\b",
    "content_update_tool": r"\bcontentManager_update\b",
    "prompt_execute_prompts": r"\bpromptManager_executePrompts\b",
    "memory_update_workspace": r"\bmemoryManager_updateWorkspace\b",
    "search_memory_workspaceId": r"\bsearchMemory\b.*\bworkspaceId\b|\bworkspaceId\b.*\bsearchMemory\b",
}


def discover_files(repo_root: Path) -> List[Path]:
    files: List[Path] = []
    for root in SCAN_ROOTS:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix in {".yaml", ".yml", ".json", ".md", ".example"}:
                files.append(path)
    return sorted(files)


def extract_manager_tool_identifiers(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z]+Manager_[A-Za-z0-9]+\b", text)


def classify_identifier(identifier: str, current_ids: set[str]) -> str:
    if identifier in current_ids:
        return "current"
    return "stale_or_unknown"


def inventory_file(path: Path, current_ids: set[str]) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    identifiers = extract_manager_tool_identifiers(text)
    identifier_counts = Counter(identifiers)

    current_matches = {k: v for k, v in identifier_counts.items() if classify_identifier(k, current_ids) == "current"}
    stale_matches = {k: v for k, v in identifier_counts.items() if classify_identifier(k, current_ids) == "stale_or_unknown"}

    special_hits = {}
    for label, pattern in SPECIAL_PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            special_hits[label] = len(matches)

    if not current_matches and not stale_matches and not special_hits:
        return {}

    return {
        "path": str(path),
        "current_tool_refs": dict(sorted(current_matches.items())),
        "stale_tool_refs": dict(sorted(stale_matches.items())),
        "special_hits": dict(sorted(special_hits.items())),
    }


def build_report(repo_root: Path) -> Dict[str, object]:
    catalog = load_target_catalog(repo_root / "cli-first-tool-schemas.json")
    current_ids = {f"{agent}_{tool}" for agent, tool in catalog}

    files_report = []
    global_current = Counter()
    global_stale = Counter()
    global_special = Counter()
    files_by_area = defaultdict(list)

    for path in discover_files(repo_root):
        result = inventory_file(path, current_ids)
        if not result:
            continue

        rel = str(path.relative_to(repo_root))
        result["path"] = rel
        files_report.append(result)

        area = rel.split("/", 1)[0] + "/" + rel.split("/", 2)[1]
        files_by_area[area].append(rel)

        for key, value in result["current_tool_refs"].items():
            global_current[key] += value
        for key, value in result["stale_tool_refs"].items():
            global_stale[key] += value
        for key, value in result["special_hits"].items():
            global_special[key] += value

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_roots": list(SCAN_ROOTS),
        "files_with_hits": len(files_report),
        "files": files_report,
        "global": {
            "current_tool_refs": dict(sorted(global_current.items())),
            "stale_tool_refs": dict(sorted(global_stale.items())),
            "special_hits": dict(sorted(global_special.items())),
        },
        "areas": {key: sorted(value) for key, value in sorted(files_by_area.items())},
    }


def write_markdown(report: Dict[str, object], path: Path) -> None:
    lines = [
        "# SynthChat CLI-Schema Reference Inventory",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Files with hits: {report['files_with_hits']}",
        "",
        "## Special Pattern Counts",
        "",
    ]

    for label, count in report["global"]["special_hits"].items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Top Stale Tool References", ""])
    for identifier, count in report["global"]["stale_tool_refs"].items():
        lines.append(f"- `{identifier}`: {count}")

    lines.extend(["", "## Impacted Files", ""])
    for file_entry in report["files"]:
        lines.append(f"- `{file_entry['path']}`")
        if file_entry["stale_tool_refs"]:
            lines.append(f"  stale: {file_entry['stale_tool_refs']}")
        if file_entry["special_hits"]:
            lines.append(f"  patterns: {file_entry['special_hits']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory SynthChat references that need CLI-schema alignment")
    parser.add_argument(
        "--output-json",
        default="Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.json",
        help="Output JSON path relative to repo root",
    )
    parser.add_argument(
        "--output-md",
        default="Datasets/tools_datasets/reports/cli_schema/synthchat_config_inventory.md",
        help="Output markdown path relative to repo root",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    report = build_report(repo_root)

    json_path = repo_root / args.output_json
    md_path = repo_root / args.output_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print("SynthChat CLI-schema reference inventory")
    print(f"Files with hits: {report['files_with_hits']}")
    print(f"Special hits: {report['global']['special_hits']}")
    print(f"Stale refs: {report['global']['stale_tool_refs']}")
    print()
    print(f"Wrote {json_path.relative_to(repo_root)}")
    print(f"Wrote {md_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
