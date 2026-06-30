#!/usr/bin/env python3
"""Project transcript-distillation rows into trainer-specific JSONL targets.

The distiller emits a shared flat/native row shape. This script keeps the
method-specific DPO and static-GRPO projections outside trainer loaders.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


TOOL_CALLS_RE = re.compile(r"Tool calls:\s*(\[[\s\S]*\])\s*$")


@dataclass
class ProjectConfig:
    target: str = "dpo"
    input_path: Path | None = None
    output_path: Path | None = None
    report_path: Path | None = None
    prompt_mode: str = "prior_messages"
    include_tool_messages: bool = False
    prompt_key_field: str | None = None


@dataclass
class ProjectionResult:
    rows: list[dict[str, Any]]
    counters: Counter = field(default_factory=Counter)


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"projection config must be a mapping: {path}")
    return loaded


def resolve_project_config(args: argparse.Namespace) -> ProjectConfig:
    cfg = load_config(args.config)
    projection = cfg.get("projection") or cfg
    if not isinstance(projection, dict):
        raise ValueError("projection config must be a mapping")

    target_cfg = projection.get("targets") or {}
    dpo_cfg = projection.get("dpo") or target_cfg.get("dpo") or {}

    target = args.target or projection.get("target") or "dpo"
    prompt_mode = args.prompt_mode or projection.get("prompt_mode") or "prior_messages"
    include_tool_messages = (
        args.include_tool_messages
        if args.include_tool_messages is not None
        else bool(projection.get("include_tool_messages", False))
    )
    prompt_key_field = args.prompt_key_field or dpo_cfg.get("prompt_key_field")

    return ProjectConfig(
        target=target,
        input_path=args.input or _optional_path(projection.get("input")),
        output_path=args.output or _optional_path(projection.get("output")),
        report_path=args.report or _optional_path(projection.get("report")),
        prompt_mode=prompt_mode,
        include_tool_messages=include_tool_messages,
        prompt_key_field=prompt_key_field,
    )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def write_report(path: Path, counters: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"counts": {key: counters[key] for key in sorted(counters)}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_dpo(rows: Iterable[dict[str, Any]], cfg: ProjectConfig) -> ProjectionResult:
    counters: Counter = Counter()
    buckets: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(lambda: {True: [], False: []})

    for row in rows:
        counters["input_rows"] += 1
        label = row.get("label")
        if label is None:
            counters["dropped_null_label"] += 1
            continue
        if label not in (True, False):
            counters["dropped_non_boolean_label"] += 1
            continue

        target = target_assistant_message(row)
        if target is None:
            counters["dropped_missing_target_assistant"] += 1
            continue

        prompt = prompt_messages(row, cfg, target_index=target[0])
        if not prompt:
            counters["dropped_empty_prompt"] += 1
            continue

        key = configured_prompt_key(row, cfg.prompt_key_field)
        if key is None:
            key = prompt_hash(prompt)
        elif key == "":
            counters["dropped_missing_prompt_key"] += 1
            continue

        buckets[str(key)][label].append({
            "prompt": prompt,
            "target": serialize_assistant_for_dpo(target[1]),
            "source_example_id": row.get("source_example_id"),
        })
        counters["eligible_rows"] += 1

    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        chosen_rows = buckets[key][True]
        rejected_rows = buckets[key][False]
        if not chosen_rows:
            counters["unpaired_rejected"] += len(rejected_rows)
            continue
        if not rejected_rows:
            counters["unpaired_chosen"] += len(chosen_rows)
            continue
        for chosen in chosen_rows:
            for rejected in rejected_rows:
                out.append({
                    "prompt": chosen["prompt"],
                    "chosen": [chosen["target"]],
                    "rejected": [rejected["target"]],
                    "provenance": {
                        "prompt_key": key,
                        "chosen_source_example_id": chosen["source_example_id"],
                        "rejected_source_example_id": rejected["source_example_id"],
                    },
                })
                counters["output_rows"] += 1

    return ProjectionResult(out, counters)


def project_static_grpo(rows: Iterable[dict[str, Any]], cfg: ProjectConfig) -> ProjectionResult:
    counters: Counter = Counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        counters["input_rows"] += 1
        target = target_assistant_message(row)
        if target is None:
            counters["dropped_missing_target_assistant"] += 1
            continue

        tool_call = first_target_tool_call(row, target[1])
        if tool_call is None:
            counters["dropped_missing_tool_call"] += 1
            continue

        prompt = prompt_messages(row, cfg, target_index=target[0])
        if not prompt:
            counters["dropped_empty_prompt"] += 1
            continue

        name, args_json = normalize_tool_call(tool_call)
        if not name:
            counters["dropped_missing_tool_name"] += 1
            continue

        out.append({
            "prompt": prompt,
            "ground_truth_tool": name,
            "ground_truth_args_json": args_json,
            "provenance": {
                "source_example_id": row.get("source_example_id"),
                "prompt_hash": prompt_hash(prompt),
            },
        })
        counters["output_rows"] += 1

    return ProjectionResult(out, counters)


def prompt_messages(row: dict[str, Any], cfg: ProjectConfig, *, target_index: int) -> list[dict[str, str]]:
    prior = row.get("conversations") or []
    if not isinstance(prior, list):
        return []
    prior = prior[:target_index]
    if not cfg.include_tool_messages:
        prior = [msg for msg in prior if msg.get("role") != "tool"]

    if cfg.prompt_mode == "last_user_only":
        for msg in reversed(prior):
            if msg.get("role") == "user":
                return [role_content_message(msg)]
        return []
    if cfg.prompt_mode != "prior_messages":
        raise ValueError("prompt_mode must be 'prior_messages' or 'last_user_only'")
    roles = {"system", "user", "assistant"}
    if cfg.include_tool_messages:
        roles.add("tool")
    return [role_content_message(msg) for msg in prior if msg.get("role") in roles]


def role_content_message(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(message.get("role") or ""),
        "content": "" if message.get("content") is None else str(message.get("content")),
    }


def target_assistant_message(row: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    conversations = row.get("conversations") or []
    if isinstance(conversations, list):
        for idx in range(len(conversations) - 1, -1, -1):
            msg = conversations[idx]
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return idx, msg

    completion = row.get("completion")
    if completion is not None:
        return len(conversations) if isinstance(conversations, list) else 0, {
            "role": "assistant",
            "content": str(completion),
        }
    return None


def serialize_assistant_for_dpo(message: dict[str, Any]) -> dict[str, str]:
    content = "" if message.get("content") is None else str(message.get("content")).strip()
    serialized_calls = serialize_tool_calls_text(message.get("tool_calls") or [])
    if serialized_calls:
        content = "\n".join(part for part in [content, serialized_calls] if part)
    return {"role": "assistant", "content": content}


def serialize_tool_calls_text(tool_calls: list[Any]) -> str:
    if not tool_calls:
        return ""
    normalized = [normalize_tool_call_payload(call) for call in tool_calls]
    return "Tool calls: " + json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_tool_call_payload(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {"name": "", "arguments": "{}"}
    if isinstance(call.get("function"), dict):
        fn = call["function"]
        return {
            "name": str(fn.get("name") or ""),
            "arguments": canonical_args_json(fn.get("arguments")),
        }
    return {
        "name": str(call.get("name") or ""),
        "arguments": canonical_args_json(call.get("input")),
    }


def first_target_tool_call(row: dict[str, Any], target_message: dict[str, Any]) -> Any | None:
    native_calls = target_message.get("tool_calls")
    if isinstance(native_calls, list) and native_calls:
        return native_calls[0]

    completion = row.get("completion")
    if completion is None:
        completion = target_message.get("content")
    calls = parse_flat_tool_calls(str(completion or ""))
    return calls[0] if calls else None


def parse_flat_tool_calls(completion: str) -> list[Any]:
    match = TOOL_CALLS_RE.search(completion)
    if not match:
        return []
    payload = match.group(1)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def normalize_tool_call(call: Any) -> tuple[str, str]:
    payload = normalize_tool_call_payload(call)
    return payload["name"], payload["arguments"]


def canonical_args_json(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return json.dumps(stripped, ensure_ascii=False)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def configured_prompt_key(row: dict[str, Any], field: str | None) -> Any | None:
    if not field:
        return None
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return ""
        current = current[part]
    return current


def prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_rows(rows: Iterable[dict[str, Any]], cfg: ProjectConfig) -> ProjectionResult:
    if cfg.target == "dpo":
        return project_dpo(rows, cfg)
    if cfg.target in {"static-grpo", "static_grpo"}:
        return project_static_grpo(rows, cfg)
    raise ValueError("target must be 'dpo' or 'static-grpo'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML projection config")
    parser.add_argument("--input", type=Path, help="input rows.jsonl path")
    parser.add_argument("--output", type=Path, help="output JSONL path")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    parser.add_argument("--target", choices=["dpo", "static-grpo", "static_grpo"])
    parser.add_argument("--prompt-mode", choices=["prior_messages", "last_user_only"])
    parser.add_argument("--prompt-key-field", help="dot path used to pair DPO rows")
    parser.add_argument(
        "--include-tool-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include role=tool messages in projected prompts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_project_config(args)
    if cfg.input_path is None or cfg.output_path is None:
        raise SystemExit("--input and --output are required, either as CLI flags or config values")

    result = project_rows(read_jsonl(cfg.input_path), cfg)
    write_jsonl(cfg.output_path, result.rows)
    if cfg.report_path:
        write_report(cfg.report_path, result.counters)
    print(json.dumps({"output_rows": result.counters["output_rows"], "counts": dict(result.counters)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
