#!/usr/bin/env python3
"""Migrate latest non-thinking datasets toward the CLI-oriented schema."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent))

from cli_schema_rules import IN_SCOPE_NONTHINKING_AGENTS
from cli_schema_utils import (
    discover_latest_nonthinking_dataset_files,
    extract_normalized_calls,
    get_repo_root,
    load_jsonl_with_line_numbers,
    load_target_catalog,
    next_version_path,
    render_cli_command,
    serialize_arguments_like,
    validate_call_shape,
)
from utils import write_jsonl


SAFE_RENAMES: Dict[Tuple[str, str], Dict[str, str]] = {
    ("memoryManager", "loadState"): {"stateId": "name"},
    ("memoryManager", "loadWorkspace"): {"workspaceId": "id"},
    ("memoryManager", "listStates"): {"limit": "pageSize"},
    ("promptManager", "archivePrompt"): {"id": "name"},
    ("promptManager", "createPrompt"): {"enabled": "isEnabled"},
    ("promptManager", "updatePrompt"): {"enabled": "isEnabled"},
    ("storageManager", "copy"): {"sourcePath": "path", "targetPath": "newPath"},
    ("storageManager", "move"): {"sourcePath": "path", "targetPath": "newPath"},
}


SAFE_DROPS: Dict[Tuple[str, str], set[str]] = {
    ("memoryManager", "createState"): {"description"},
    ("searchManager", "searchMemory"): {"workspaceId"},
    ("storageManager", "archive"): {"recursive"},
    ("storageManager", "copy"): {"autoIncrement"},
    ("storageManager", "list"): {"includeFiles", "depth"},
    ("storageManager", "move"): {"autoIncrement"},
}


def parse_agents(raw_agents: str | None) -> List[str]:
    if not raw_agents:
        return list(IN_SCOPE_NONTHINKING_AGENTS)
    values = [value.strip() for value in raw_agents.split(",") if value.strip()]
    return values or list(IN_SCOPE_NONTHINKING_AGENTS)


def parse_source_overrides(raw_values: List[str] | None, repo_root: Path) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for raw in raw_values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --source override: {raw}")
        agent, relative_path = raw.split("=", 1)
        overrides[agent.strip()] = repo_root / relative_path.strip()
    return overrides


def is_rewrite_candidate(message: Dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    for tool_call in message.get("tool_calls", []) or []:
        if tool_call.get("function", {}).get("name") == "useTools":
            return True
    return False


def apply_safe_renames(agent: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    rewritten = dict(params)
    for old_name, new_name in SAFE_RENAMES.get((agent, tool), {}).items():
        if old_name in rewritten and new_name not in rewritten:
            rewritten[new_name] = rewritten.pop(old_name)
    return rewritten


def apply_safe_drops(agent: str, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
    rewritten = dict(params)
    for name in SAFE_DROPS.get((agent, tool), set()):
        rewritten.pop(name, None)
    return rewritten


def extract_workspace_lookup(example: Dict[str, Any]) -> Dict[str, str]:
    """Parse workspace id -> name mappings from the system prompt."""
    lookup: Dict[str, str] = {}
    for message in example.get("conversations", []):
        if message.get("role") != "system":
            continue
        content = message.get("content") or ""
        for name, workspace_id in re.findall(r'- ([^\n]+?) \(id: "([^"]+)"\)', content):
            lookup[workspace_id] = name.strip()
    return lookup


def extract_legacy_wrapper_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    context = arguments.get("context")
    if isinstance(context, dict):
        return dict(context)
    result: Dict[str, Any] = {}
    for field in ("workspaceId", "sessionId", "memory", "goal", "constraints", "strategy", "tool"):
        if field in arguments:
            result[field] = arguments[field]
    return result


def infer_constraints(calls: List[Dict[str, Any]]) -> str:
    if not calls:
        return "Do not touch unrelated files."

    tools = {(call["agent"], call["tool"]) for call in calls}

    if any(agent == "storageManager" and tool == "archive" for agent, tool in tools):
        return "Only archive the requested path. Do not delete or modify unrelated files."
    if any(agent == "contentManager" and tool in {"insert", "replace", "setProperty", "write"} for agent, tool in tools):
        return "Preserve surrounding content and do not modify unrelated files."
    if any(agent == "promptManager" and tool in {"createPrompt", "updatePrompt", "archivePrompt"} for agent, tool in tools):
        return "Only change the requested prompt configuration. Do not modify unrelated prompts."
    if any(agent == "searchManager" for agent, _ in tools):
        return "Do not modify any files while searching."
    if any(agent == "memoryManager" for agent, _ in tools):
        return "Only access or update the requested workspace or state. Do not change unrelated workspace data."
    return "Do not touch unrelated files."


def transform_auto_call(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    agent: str,
    tool: str,
    params: Dict[str, Any],
    workspace_lookup: Dict[str, str],
) -> Tuple[str, Dict[str, Any], List[str]]:
    rewritten = apply_safe_renames(agent, tool, params)
    rewritten = apply_safe_drops(agent, tool, rewritten)

    if agent == "memoryManager" and tool == "archiveWorkspace" and "workspaceId" in rewritten and "name" not in rewritten:
        workspace_id = rewritten.pop("workspaceId")
        if workspace_id in workspace_lookup:
            rewritten["name"] = workspace_lookup[workspace_id]
        else:
            return "regenerate", {}, ["archive_workspace_workspaceId_not_resolvable"]

    valid, errors = validate_call_shape(catalog, agent, tool, rewritten)
    if not valid:
        return "regenerate", {}, errors

    return "migrated", {"agent": agent, "tool": tool, "params": rewritten}, []


def transform_create_workspace(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[str]]:
    rewritten = dict(params)
    reasons: List[str] = []

    description = rewritten.get("description")
    purpose = rewritten.get("purpose")

    if not description and purpose:
        rewritten["description"] = purpose
        reasons.append("filled_description_from_purpose")
    elif description and not purpose:
        rewritten["purpose"] = description
        reasons.append("filled_purpose_from_description")

    valid, errors = validate_call_shape(catalog, "memoryManager", "createWorkspace", rewritten)
    if not valid:
        return "regenerate", {}, reasons + errors

    return "migrated", {"agent": "memoryManager", "tool": "createWorkspace", "params": rewritten}, reasons


def transform_content_update(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[str]]:
    if "endLine" in params:
        return "regenerate", {}, ["replace_requires_old_content"]
    if "oldText" in params or "findText" in params:
        return "regenerate", {}, ["text_find_replace_not_safe_to_infer"]

    rewritten = {
        "path": params.get("path"),
        "content": params.get("content"),
        "startLine": params.get("startLine"),
    }

    valid, errors = validate_call_shape(catalog, "contentManager", "insert", rewritten)
    if not valid:
        return "regenerate", {}, errors

    return "migrated", {"agent": "contentManager", "tool": "insert", "params": rewritten}, ["mapped_update_to_insert"]


def transform_call(
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
    agent: str,
    tool: str,
    params: Dict[str, Any],
    workspace_lookup: Dict[str, str],
) -> Tuple[str, Dict[str, Any], List[str]]:
    if agent == "promptManager" and tool == "executePrompts":
        return "regenerate", {}, ["prompt_execute_prompts_marked_for_regeneration"]

    if agent == "memoryManager" and tool == "updateWorkspace":
        return "regenerate", {}, ["update_workspace_not_addressable_in_new_schema"]

    if agent == "contentManager" and tool == "update":
        return transform_content_update(catalog, params)

    if agent == "storageManager" and tool == "open" and params.get("mode") in {"create", "splitview"}:
        return "regenerate", {}, [f"unsupported_open_mode:{params.get('mode')}"]

    if agent == "memoryManager" and tool == "createWorkspace":
        return transform_create_workspace(catalog, params)

    return transform_auto_call(catalog, agent, tool, params, workspace_lookup)


def rewrite_example(
    example: Dict[str, Any],
    catalog: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    rewritten = copy.deepcopy(example)
    findings: List[Dict[str, Any]] = []
    touched = False
    workspace_lookup = extract_workspace_lookup(example)

    for message in rewritten.get("conversations", []):
        if not is_rewrite_candidate(message):
            continue

        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            if function.get("name") != "useTools":
                continue

            original_arguments = function.get("arguments", {})
            arguments = original_arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            else:
                arguments = copy.deepcopy(arguments)

            legacy_context = extract_legacy_wrapper_context(arguments)
            calls = arguments.get("calls", [])
            if not calls:
                continue

            transformed_calls = []
            for call in calls:
                agent = call.get("agent")
                tool = call.get("tool")
                params = call.get("params", {}) or {}

                outcome, rewritten_call, reasons = transform_call(catalog, agent, tool, params, workspace_lookup)
                findings.append(
                    {
                        "agent": agent,
                        "tool": tool,
                        "outcome": outcome,
                        "reasons": reasons,
                    }
                )

                if outcome != "migrated":
                    return outcome, example, findings

                transformed_calls.append(rewritten_call)

            command_segments = [
                render_cli_command(item["agent"], item["tool"], item["params"], catalog)
                for item in transformed_calls
            ]
            new_arguments = {
                "workspaceId": legacy_context.get("workspaceId", "default"),
                "sessionId": legacy_context.get("sessionId", "session_migrated"),
                "memory": legacy_context.get("memory", "Continue the requested workspace operation."),
                "goal": legacy_context.get("goal", "Complete the requested tool operation."),
                "constraints": legacy_context.get("constraints") or infer_constraints(transformed_calls),
                "tool": ", ".join(command_segments),
            }
            if len(command_segments) > 1:
                new_arguments["strategy"] = legacy_context.get("strategy") or "serial"

            function["arguments"] = serialize_arguments_like(original_arguments, new_arguments)
            touched = True

    if touched:
        return "migrated", rewritten, findings
    return "unchanged", rewritten, findings


def write_queue(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# CLI Schema Migration Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Input | Output | Migrated | Unchanged | Regenerate | Manual |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for agent, data in report["datasets"].items():
        if data.get("status") != "ok":
            lines.append(f"| {agent} | - | - | 0 | 0 | 0 | 0 |")
            continue
        lines.append(
            f"| {agent} | `{data['input_path']}` | `{data['output_path']}` | "
            f"{data['migrated_examples']} | {data['unchanged_examples']} | "
            f"{data['regenerate_examples']} | {data['manual_examples']} |"
        )

    lines.extend(
        [
            "",
            "## Global Counts",
            "",
            f"- Migrated examples: {report['global']['migrated_examples']}",
            f"- Unchanged examples: {report['global']['unchanged_examples']}",
            f"- Regenerate examples: {report['global']['regenerate_examples']}",
            f"- Manual examples: {report['global']['manual_examples']}",
            "",
            "## Top Regeneration Reasons",
            "",
        ]
    )

    for reason, count in report["global"]["regenerate_reasons"].items():
        lines.append(f"- `{reason}`: {count}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate latest non-thinking datasets toward the CLI schema")
    parser.add_argument(
        "--agents",
        help="Comma-separated non-thinking agents to migrate "
        "(default: contentManager,memoryManager,promptManager,searchManager,storageManager)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write migrated datasets or queue files")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Write queue/report artifacts without writing versioned dataset files",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Override input source for an agent using agent=relative/path.jsonl. "
        "Can be passed multiple times.",
    )
    parser.add_argument(
        "--reports-dir",
        default="Datasets/tools_datasets/reports/cli_schema",
        help="Report directory relative to repo root",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Only process the first N examples from each dataset (useful for smoke tests)",
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    reports_dir = repo_root / args.reports_dir
    catalog = load_target_catalog(repo_root / "tool-schemas.json")
    agents = parse_agents(args.agents)
    latest_files = discover_latest_nonthinking_dataset_files(repo_root / "Datasets" / "tools_datasets", agents)
    source_overrides = parse_source_overrides(args.source, repo_root)

    regenerate_queue: List[Dict[str, Any]] = []
    manual_queue: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "datasets": {},
        "global": {
            "migrated_examples": 0,
            "unchanged_examples": 0,
            "regenerate_examples": 0,
            "manual_examples": 0,
            "regenerate_reasons": {},
        },
    }

    global_regen_reasons: Counter[str] = Counter()

    for agent in agents:
        input_path = source_overrides.get(agent, latest_files.get(agent))
        if input_path is None:
            report["datasets"][agent] = {"status": "missing"}
            continue

        output_path = next_version_path(input_path)
        migrated_examples: List[Dict[str, Any]] = []
        dataset_regen_reasons: Counter[str] = Counter()
        counts = Counter()

        entries = load_jsonl_with_line_numbers(input_path)
        if args.max_examples is not None:
            entries = entries[: max(0, args.max_examples)]

        for line_number, example in entries:
            outcome, rewritten, findings = rewrite_example(example, catalog)

            if outcome in {"migrated", "unchanged"}:
                migrated_examples.append(rewritten)
                counts[f"{outcome}_examples"] += 1
                continue

            reasons = []
            for finding in findings:
                if finding["outcome"] == outcome:
                    reasons.extend(finding["reasons"])
            if not reasons:
                reasons = ["unspecified_transform_failure"]

            queue_entry = {
                "dataset": str(input_path.relative_to(repo_root)),
                "line_number": line_number,
                "agent": agent,
                "outcome": outcome,
                "reasons": reasons,
                "findings": findings,
                "example": example,
            }

            if outcome == "manual":
                manual_queue.append(queue_entry)
                counts["manual_examples"] += 1
            else:
                regenerate_queue.append(queue_entry)
                counts["regenerate_examples"] += 1
                for reason in reasons:
                    dataset_regen_reasons[reason] += 1
                    global_regen_reasons[reason] += 1

        report["datasets"][agent] = {
            "status": "ok",
            "input_path": str(input_path.relative_to(repo_root)),
            "output_path": str(output_path.relative_to(repo_root)),
            "input_examples": sum(counts.values()),
            "migrated_examples": counts["migrated_examples"],
            "unchanged_examples": counts["unchanged_examples"],
            "regenerate_examples": counts["regenerate_examples"],
            "manual_examples": counts["manual_examples"],
            "regenerate_reasons": {key: dataset_regen_reasons[key] for key in sorted(dataset_regen_reasons)},
        }

        report["global"]["migrated_examples"] += counts["migrated_examples"]
        report["global"]["unchanged_examples"] += counts["unchanged_examples"]
        report["global"]["regenerate_examples"] += counts["regenerate_examples"]
        report["global"]["manual_examples"] += counts["manual_examples"]

        if not args.dry_run and not args.reports_only:
            write_jsonl(output_path, migrated_examples)

    report["global"]["regenerate_reasons"] = {
        key: global_regen_reasons[key] for key in sorted(global_regen_reasons)
    }

    regen_path = reports_dir / "regen_queue.jsonl"
    manual_path = reports_dir / "manual_review.jsonl"
    json_report_path = reports_dir / "migration_report.json"
    md_report_path = reports_dir / "migration_report.md"

    if not args.dry_run:
        write_queue(regen_path, regenerate_queue)
        write_queue(manual_path, manual_queue)
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_markdown_report(md_report_path, report)

    print("CLI-schema migration")
    print(f"Agents: {', '.join(agents)}")
    print()
    for agent in agents:
        data = report["datasets"].get(agent, {})
        if data.get("status") != "ok":
            print(f"{agent}: missing")
            continue
        print(
            f"{agent}: migrated={data['migrated_examples']} unchanged={data['unchanged_examples']} "
            f"regenerate={data['regenerate_examples']} manual={data['manual_examples']}"
        )
    print()
    print(
        f"Global: migrated={report['global']['migrated_examples']} "
        f"unchanged={report['global']['unchanged_examples']} "
        f"regenerate={report['global']['regenerate_examples']} "
        f"manual={report['global']['manual_examples']}"
    )
    print(f"Regeneration reasons: {report['global']['regenerate_reasons']}")
    if args.dry_run:
        print("\nDry run only; no files written.")
    else:
        print(f"\nWrote reports to {reports_dir.relative_to(repo_root)}")
        if args.reports_only:
            print("Dataset files were not written (--reports-only).")


if __name__ == "__main__":
    main()
