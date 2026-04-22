#!/usr/bin/env python3
"""Prepare per-tool seed JSONL files from the CLI-schema regeneration queue.

This script can either:
1. read an existing regen queue JSONL, or
2. reconstruct the queue directly from the canonical legacy source datasets.

It preserves the original example payloads and writes a manifest with counts
and suggested rubrics for downstream regeneration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from cli_schema_utils import (
    get_repo_root,
    load_jsonl_with_line_numbers,
    load_target_catalog,
)
from cli_schema_rules import IN_SCOPE_NONTHINKING_AGENTS
from utils import find_latest_version


def load_migration_module():
    module_path = Path(__file__).with_name("06_migrate_cli_schema_datasets.py")
    spec = importlib.util.spec_from_file_location("cli_migrate_06", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load migration module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REGEN_GROUPS = {
    ("contentManager", "replace_requires_old_content"): {
        "key": "contentManager_replace",
        "rubrics": ["tool_alignment", "contentManager_tools"],
    },
    ("memoryManager", "update_workspace_not_addressable_in_new_schema"): {
        "key": "memoryManager_updateWorkspace",
        "rubrics": ["tool_alignment", "memoryManager_tools"],
    },
    ("promptManager", "prompt_execute_prompts_marked_for_regeneration"): {
        "key": "promptManager_executePrompts",
        "rubrics": ["tool_alignment", "promptManager_tools"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        default="Datasets/tools_datasets/reports/cli_schema/regen_queue.jsonl",
        help="Path to regeneration queue JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="Datasets/tools_datasets/reports/cli_schema/regen_seeds",
        help="Directory for per-tool seed JSONL files and manifest",
    )
    parser.add_argument(
        "--rebuild-from-sources",
        action="store_true",
        help="Recompute the regeneration queue from source datasets instead of reading --queue",
    )
    parser.add_argument(
        "--agents",
        default="contentManager,memoryManager,promptManager,searchManager,storageManager",
        help="Comma-separated agents to consider when --rebuild-from-sources is set",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Override source dataset path using agent=relative/path.jsonl when rebuilding from sources",
    )
    return parser.parse_args()


def load_queue(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_agents(raw_agents: str) -> List[str]:
    values = [value.strip() for value in raw_agents.split(",") if value.strip()]
    return [value for value in values if value in IN_SCOPE_NONTHINKING_AGENTS]


def parse_source_overrides(raw_values: List[str], repo_root: Path) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for raw in raw_values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --source override: {raw}")
        agent, relative_path = raw.split("=", 1)
        overrides[agent.strip()] = repo_root / relative_path.strip()
    return overrides


def discover_default_sources(repo_root: Path, agents: List[str]) -> Dict[str, Path]:
    datasets_root = repo_root / "Datasets" / "tools_datasets" / "non_thinking"
    discovered: Dict[str, Path] = {}
    for agent in agents:
        latest = find_latest_version(datasets_root / agent)
        if latest is not None:
            discovered[agent] = latest
    return discovered


def rebuild_queue_from_sources(repo_root: Path, agents: List[str], source_overrides: Dict[str, Path]) -> List[Dict]:
    migrate_mod = load_migration_module()
    rewrite_example = migrate_mod.rewrite_example
    catalog = load_target_catalog(repo_root / "tool-schemas.json")
    default_sources = discover_default_sources(repo_root, agents)
    rows: List[Dict] = []

    for agent in agents:
        input_path = source_overrides.get(agent, default_sources.get(agent))
        if input_path is None:
            continue

        for line_number, example in load_jsonl_with_line_numbers(input_path):
            outcome, _, findings = rewrite_example(example, catalog)
            if outcome in {"migrated", "unchanged"}:
                continue

            reasons: List[str] = []
            for finding in findings:
                if finding["outcome"] == outcome:
                    reasons.extend(finding["reasons"])
            if not reasons:
                reasons = ["unspecified_transform_failure"]

            rows.append(
                {
                    "dataset": str(input_path.relative_to(repo_root)),
                    "line_number": line_number,
                    "agent": agent,
                    "outcome": outcome,
                    "reasons": reasons,
                    "findings": findings,
                    "example": example,
                }
            )
    return rows


def iter_group_assignments(rows: Iterable[Dict]) -> Iterable[Tuple[str, Dict, Dict]]:
    for row in rows:
        agent = row.get("agent")
        reasons = row.get("reasons") or []
        for reason in reasons:
            group = REGEN_GROUPS.get((agent, reason))
            if group:
                yield group["key"], group, row
                break


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    queue_path = repo_root / args.queue
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_from_sources:
        agents = parse_agents(args.agents)
        source_overrides = parse_source_overrides(args.source, repo_root)
        rows = rebuild_queue_from_sources(repo_root, agents, source_overrides)
    else:
        rows = load_queue(queue_path)
    grouped_examples: Dict[str, List[Dict]] = defaultdict(list)
    manifest: Dict[str, Dict] = {}
    totals = Counter()

    for key, group, row in iter_group_assignments(rows):
        seed_example = row.get("example")
        if not seed_example:
            continue
        grouped_examples[key].append(seed_example)
        totals[key] += 1
        manifest.setdefault(
            key,
            {
                "rubrics": list(group["rubrics"]),
                "source_reasons": [],
                "source_agents": [],
                "count": 0,
                "seed_file": f"{key}_seeds.jsonl",
            },
        )
        manifest[key]["count"] += 1
        if row.get("agent") not in manifest[key]["source_agents"]:
            manifest[key]["source_agents"].append(row.get("agent"))
        for reason in row.get("reasons") or []:
            if reason not in manifest[key]["source_reasons"]:
                manifest[key]["source_reasons"].append(reason)

    for key, examples in grouped_examples.items():
        seed_path = output_dir / f"{key}_seeds.jsonl"
        with open(seed_path, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "queue_path": str(queue_path),
                "groups": manifest,
                "total_examples": sum(totals.values()),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {sum(totals.values())} seed examples across {len(grouped_examples)} groups to {output_dir}")
    for key in sorted(grouped_examples):
        print(f"- {key}: {len(grouped_examples[key])}")


if __name__ == "__main__":
    main()
