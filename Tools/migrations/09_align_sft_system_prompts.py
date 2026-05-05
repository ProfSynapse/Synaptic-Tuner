#!/usr/bin/env python3
"""Apply a configured system-prompt profile to SFT JSONL datasets."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import bump_version, read_jsonl, write_jsonl


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def version_key(path: Path) -> Tuple[int, int]:
    match = re.search(r"_v(\d+)\.(\d+)\.jsonl$", path.name)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def find_latest_dataset(agent_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for prefix in ("tools", "text_only"):
        for path in agent_dir.glob(f"{prefix}_v*.jsonl"):
            if any(skip in path.name for skip in ("failed", "review", "test", "_full", "smoke")):
                continue
            if "_" in path.stem.removeprefix(f"{prefix}_v"):
                continue
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=version_key, reverse=True)[0]


def discover_sources(root: Path, agents: Iterable[str]) -> Dict[str, Path]:
    base = root / "Datasets" / "tools_datasets" / "non_thinking"
    result: Dict[str, Path] = {}
    for agent in agents:
        latest = find_latest_dataset(base / agent)
        if latest:
            result[agent] = latest
    return result


def parse_source_overrides(raw_values: Optional[List[str]], repo_root: Path) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for raw in raw_values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --source override: {raw}")
        agent, relative_path = raw.split("=", 1)
        overrides[agent.strip()] = repo_root / relative_path.strip()
    return overrides


def load_profile(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile.get("template"), str):
        raise ValueError("Prompt profile must include a string 'template'.")
    if not isinstance(profile.get("variables", {}), dict):
        raise ValueError("Prompt profile 'variables' must be an object.")
    return profile


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def collect_tool_arguments(example: Dict[str, Any]) -> Dict[str, Any]:
    for message in example.get("conversations", []) or []:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function") or {}
            arguments = parse_tool_arguments(function.get("arguments", tool_call.get("arguments")))
            if arguments:
                return arguments
    return {}


def get_system_prompt(example: Dict[str, Any]) -> str:
    for message in example.get("conversations", []) or []:
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""


def set_system_prompt(example: Dict[str, Any], content: str) -> bool:
    conversations = example.setdefault("conversations", [])
    for message in conversations:
        if message.get("role") == "system":
            message["content"] = content
            return True
    conversations.insert(0, {"role": "system", "content": content})
    return True


def scrub_preserved_section(section: str, profile: Dict[str, Any]) -> str:
    raw_patterns = profile.get("drop_preserved_line_regexes", [])
    if not raw_patterns:
        return section
    patterns = [re.compile(pattern) for pattern in raw_patterns]
    kept_lines: List[str] = []
    for line in section.splitlines():
        if any(pattern.search(line) for pattern in patterns):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def extract_sections(system_prompt: str, section_names: Iterable[str], profile: Dict[str, Any]) -> List[str]:
    sections: List[str] = []
    for name in section_names:
        pattern = re.compile(rf"<{re.escape(name)}\b[^>]*>.*?</{re.escape(name)}>", re.DOTALL)
        for match in pattern.finditer(system_prompt):
            section = scrub_preserved_section(match.group(0).strip(), profile)
            if section:
                sections.append(section)
    return sections


def resolve_variables(profile: Dict[str, Any], example: Dict[str, Any]) -> Dict[str, str]:
    system_prompt = get_system_prompt(example)
    tool_arguments = collect_tool_arguments(example)
    values: Dict[str, str] = {}

    for name, spec in profile.get("variables", {}).items():
        value: Any = None
        tool_argument = spec.get("tool_argument")
        if tool_argument and tool_argument in tool_arguments:
            value = tool_arguments.get(tool_argument)
        if value in (None, "") and spec.get("system_regex"):
            match = re.search(spec["system_regex"], system_prompt)
            if match:
                value = match.group(1)
        if value in (None, ""):
            value = spec.get("default", "")
        values[name] = str(value)

    return values


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def build_system_prompt(profile: Dict[str, Any], example: Dict[str, Any]) -> str:
    variables = SafeFormatDict(resolve_variables(profile, example))
    rendered = profile["template"].format_map(variables).strip()
    preserved = extract_sections(get_system_prompt(example), profile.get("preserve_sections", []), profile)
    if preserved:
        return rendered + "\n\n" + "\n\n".join(preserved)
    return rendered


def align_dataset(items: List[Dict[str, Any]], profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    aligned: List[Dict[str, Any]] = []
    for item in items:
        rewritten = copy.deepcopy(item)
        old_prompt = get_system_prompt(rewritten)
        new_prompt = build_system_prompt(profile, rewritten)
        set_system_prompt(rewritten, new_prompt)
        counts["examples"] += 1
        if old_prompt != new_prompt:
            counts["changed"] += 1
        else:
            counts["unchanged"] += 1
        if "getTools" in new_prompt:
            counts["new_prompt_getTools_refs"] += 1
        if '"context" parameter' in new_prompt:
            counts["new_prompt_context_parameter_refs"] += 1
        aligned.append(rewritten)
    return aligned, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a configured SFT system prompt profile and bump dataset versions.")
    parser.add_argument(
        "--profile",
        default="Datasets/tools_datasets/system_prompt_profiles/lean_use_tools_sft.json",
        help="Prompt profile JSON path relative to repo root.",
    )
    parser.add_argument(
        "--agents",
        default="contentManager,memoryManager,promptManager,searchManager,storageManager,text_only",
        help="Comma-separated non-thinking dataset folders to process.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Override input source using agent=relative/path.jsonl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reports-dir",
        default="Datasets/tools_datasets/reports/system_prompt_alignment",
        help="Report directory relative to repo root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write datasets or reports.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    profile = load_profile(repo_root / args.profile)
    agents = [agent.strip() for agent in args.agents.split(",") if agent.strip()]
    discovered = discover_sources(repo_root, agents)
    overrides = parse_source_overrides(args.source, repo_root)
    sources = {agent: overrides.get(agent, discovered.get(agent)) for agent in agents}

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_name": profile.get("name"),
        "datasets": {},
        "global": Counter(),
    }

    for agent, input_path in sources.items():
        if input_path is None:
            report["datasets"][agent] = {"status": "missing"}
            continue
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        output_path = input_path.with_name(bump_version(input_path.name))
        items = read_jsonl(input_path)
        aligned, counts = align_dataset(items, profile)
        for key, value in counts.items():
            report["global"][key] += value
        report["datasets"][agent] = {
            "status": "ok",
            "input_path": str(input_path.relative_to(repo_root)),
            "output_path": str(output_path.relative_to(repo_root)),
            **dict(counts),
        }

        if not args.dry_run:
            write_jsonl(output_path, aligned)

    report["global"] = dict(report["global"])

    print("SFT system prompt alignment")
    print(f"Profile: {profile.get('name') or args.profile}")
    for agent in agents:
        data = report["datasets"].get(agent, {})
        if data.get("status") != "ok":
            print(f"{agent}: missing")
            continue
        print(
            f"{agent}: {data['input_path']} -> {data['output_path']} | "
            f"changed={data.get('changed', 0)} unchanged={data.get('unchanged', 0)}"
        )
    print(f"Global: {report['global']}")

    if args.dry_run:
        print("Dry run only; no files written.")
        return

    reports_dir = repo_root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
