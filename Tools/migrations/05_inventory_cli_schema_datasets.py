#!/usr/bin/env python3
"""Inventory latest non-thinking datasets for the CLI-schema migration."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).parent))

from cli_schema_rules import IN_SCOPE_NONTHINKING_AGENTS, classify_call
from cli_schema_utils import (
    classify_example_bucket,
    counter_to_sorted_dict,
    discover_latest_nonthinking_dataset_files,
    extract_normalized_calls,
    get_repo_root,
    load_jsonl_with_line_numbers,
    load_target_catalog,
)


def parse_agents(raw_agents: str | None) -> List[str]:
    if not raw_agents:
        return list(IN_SCOPE_NONTHINKING_AGENTS)
    values = [value.strip() for value in raw_agents.split(",") if value.strip()]
    return values or list(IN_SCOPE_NONTHINKING_AGENTS)


def build_inventory(repo_root: Path, agents: List[str]) -> Dict[str, Any]:
    schema_path = repo_root / "cli-first-tool-schemas.json"
    datasets_root = repo_root / "Datasets" / "tools_datasets"
    reports_root = datasets_root / "reports" / "cli_schema"

    catalog = load_target_catalog(schema_path)
    latest_files = discover_latest_nonthinking_dataset_files(datasets_root, agents)

    datasets_report: Dict[str, Any] = {}
    global_tool_counts: Counter[str] = Counter()
    global_bucket_counts: Counter[str] = Counter()
    global_reason_counts: Counter[str] = Counter()
    global_source_counts: Counter[str] = Counter()

    for agent in agents:
        dataset_path = latest_files.get(agent)
        if dataset_path is None:
            datasets_report[agent] = {"status": "missing"}
            continue

        example_bucket_counts: Counter[str] = Counter()
        call_bucket_counts: Counter[str] = Counter()
        call_reason_counts: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        unknown_calls: List[Dict[str, Any]] = []
        example_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        entries = load_jsonl_with_line_numbers(dataset_path)
        for line_number, example in entries:
            calls = extract_normalized_calls(example, catalog)
            call_buckets: List[str] = []

            for call in calls:
                normalized_agent = call.get("agent") or "unknown"
                normalized_tool = call.get("tool") or "unknown"
                params = call.get("params", {})
                source = call.get("source", "unknown")
                key = f"{normalized_agent}.{normalized_tool}"

                classification = classify_call(normalized_agent, normalized_tool, params)
                bucket = classification["bucket"]
                reasons = classification["reasons"]

                call_buckets.append(bucket)
                call_bucket_counts[bucket] += 1
                tool_counts[key] += 1
                source_counts[source] += 1
                global_tool_counts[key] += 1
                global_bucket_counts[bucket] += 1
                global_source_counts[source] += 1

                for reason in reasons:
                    call_reason_counts[reason] += 1
                    global_reason_counts[reason] += 1

                if (normalized_agent, normalized_tool) not in catalog:
                    unknown_calls.append(
                        {
                            "line": line_number,
                            "agent": normalized_agent,
                            "tool": normalized_tool,
                            "source": source,
                        }
                    )

            example_bucket = classify_example_bucket(call_buckets)
            example_bucket_counts[example_bucket] += 1

            if len(example_samples[example_bucket]) < 5:
                example_samples[example_bucket].append(
                    {
                        "line": line_number,
                        "call_count": len(calls),
                        "user": next(
                            (msg.get("content") for msg in example.get("conversations", []) if msg.get("role") == "user"),
                            None,
                        ),
                    }
                )

        datasets_report[agent] = {
            "status": "ok",
            "path": str(dataset_path.relative_to(repo_root)),
            "example_count": len(entries),
            "tool_counts": counter_to_sorted_dict(tool_counts),
            "call_bucket_counts": counter_to_sorted_dict(call_bucket_counts),
            "example_bucket_counts": counter_to_sorted_dict(example_bucket_counts),
            "reason_counts": counter_to_sorted_dict(call_reason_counts),
            "source_counts": counter_to_sorted_dict(source_counts),
            "unknown_calls": unknown_calls[:20],
            "sample_examples": dict(example_samples),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "schema_path": str(schema_path.relative_to(repo_root)),
        "reports_dir": str(reports_root.relative_to(repo_root)),
        "in_scope_agents": agents,
        "datasets": datasets_report,
        "global": {
            "tool_counts": counter_to_sorted_dict(global_tool_counts),
            "call_bucket_counts": counter_to_sorted_dict(global_bucket_counts),
            "reason_counts": counter_to_sorted_dict(global_reason_counts),
            "source_counts": counter_to_sorted_dict(global_source_counts),
        },
    }


def write_inventory_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def print_summary(report: Dict[str, Any]) -> None:
    print("CLI-schema inventory")
    print(f"Agents: {', '.join(report['in_scope_agents'])}")
    print()

    for agent in report["in_scope_agents"]:
        dataset = report["datasets"].get(agent, {})
        if dataset.get("status") != "ok":
            print(f"{agent}: missing")
            continue

        print(
            f"{agent}: {dataset['example_count']} examples | "
            f"example buckets={dataset['example_bucket_counts']} | "
            f"call buckets={dataset['call_bucket_counts']}"
        )

    print()
    print(f"Global call buckets: {report['global']['call_bucket_counts']}")
    print(f"Global reasons: {report['global']['reason_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory latest non-thinking CLI-schema datasets")
    parser.add_argument(
        "--agents",
        help="Comma-separated non-thinking agents to inventory "
        "(default: contentManager,memoryManager,promptManager,searchManager,storageManager)",
    )
    parser.add_argument(
        "--output",
        default="Datasets/tools_datasets/reports/cli_schema/inventory.json",
        help="Output JSON report path relative to repo root",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    agents = parse_agents(args.agents)
    report = build_inventory(repo_root, agents)

    output_path = repo_root / args.output
    write_inventory_report(report, output_path)
    print_summary(report)
    print()
    print(f"Wrote {output_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
