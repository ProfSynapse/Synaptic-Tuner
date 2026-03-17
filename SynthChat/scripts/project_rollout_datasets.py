#!/usr/bin/env python3
"""Aggregate SynthChat rollout artifacts and project KTO/GRPO datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_records(paths: Iterable[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                row = json.loads(line)
                if line_index == 0 and "_meta" in row:
                    continue
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                yield path, line_index, row


def _first_message(conversations: List[Dict[str, Any]], role: str) -> Optional[str]:
    for message in conversations:
        if message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _stage_failures(metadata: Dict[str, Any]) -> List[str]:
    reviews = metadata.get("stage_reviews") or {}
    failures = []
    for stage, review in reviews.items():
        if isinstance(review, dict) and review.get("passed", True) is False:
            failures.append(stage)
    return sorted(failures)


def _issue_labels(metadata: Dict[str, Any]) -> List[str]:
    labels = (((metadata.get("labels") or {}).get("filter") or {}).get("issue_labels")) or []
    return [str(label) for label in labels if str(label).strip()]


def _kto_label(metadata: Dict[str, Any]) -> Optional[bool]:
    return (((metadata.get("labels") or {}).get("filter") or {}).get("kto_candidate_label"))


def _scenario_family(metadata: Dict[str, Any]) -> str:
    derivation = metadata.get("derivation_summary") or {}
    family = derivation.get("task_family_kind")
    if isinstance(family, str) and family.strip():
        return family.strip()
    return str(metadata.get("scenario") or "unknown")


def _build_canonical_row(path: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    canonical = dict(row)
    metadata = dict(canonical.get("metadata") or {})
    metadata["aggregate_source_artifact"] = path.name
    canonical["metadata"] = metadata
    return canonical


def _is_quality_positive(metadata: Dict[str, Any]) -> bool:
    environment = metadata.get("environment") or {}
    return bool(environment.get("passed")) and not _stage_failures(metadata)


def _is_meaningful_negative(metadata: Dict[str, Any]) -> bool:
    environment = metadata.get("environment") or {}
    if environment.get("passed") is not False:
        return False
    stop_reason = ((environment.get("episode_trace") or {}).get("stop_reason")) or ""
    if stop_reason not in {"max_tool_steps_exceeded", "text_response_before_completion"}:
        return False
    blocked_stages = {"environment_generation", "system_generation", "user_generation", "assistant_generation"}
    if blocked_stages.intersection(_stage_failures(metadata)):
        return False
    return True


def _build_kto_row(path: Path, line_index: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata") or {}
    label = _kto_label(metadata)
    if label is None:
        return None

    conversations = row.get("conversations") or []
    if not isinstance(conversations, list):
        return None

    prompt = _first_message(conversations, "user")
    completion = _first_message(conversations, "assistant")
    if not prompt or not completion:
        return None

    if label is True and not _is_quality_positive(metadata):
        return None
    if label is False and not _is_meaningful_negative(metadata):
        return None

    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "label": bool(label),
        "prompt": prompt,
        "completion": completion,
        "scenario_id": str(metadata.get("scenario") or "unknown"),
        "scenario_family": _scenario_family(metadata),
        "source_example_id": f"{path.name}:{line_index}",
        "score_tier": "acceptable" if label else "partial",
        "failure_labels": _issue_labels(metadata),
        "metadata": {
            "source_artifact": path.name,
            "seed_id": ((metadata.get("environment_seed") or {}).get("seed_id")),
            "stop_reason": (((metadata.get("environment") or {}).get("episode_trace") or {}).get("stop_reason")),
            "stage_failures": _stage_failures(metadata),
        },
    }


def _build_grpo_row(path: Path, line_index: int, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata") or {}
    if not _is_quality_positive(metadata):
        return None

    conversations = row.get("conversations") or []
    if not isinstance(conversations, list):
        return None

    prompt = _first_message(conversations, "user")
    if not prompt:
        return None

    environment = metadata.get("environment") or {}
    executed_tools = environment.get("executed_tools") or []
    first_tool = executed_tools[0] if executed_tools else {}
    tool_name = first_tool.get("name")
    tool_args = first_tool.get("arguments")
    if not tool_name or tool_args is None:
        return None

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "scenario_id": str(metadata.get("scenario") or "unknown"),
        "scenario_family": _scenario_family(metadata),
        "source_example_id": f"{path.name}:{line_index}",
        "allowed_tools": sorted({tool.get("name") for tool in executed_tools if tool.get("name")}),
        "environment_passed": True,
        "schema_passed": True,
        "score_value": 1.0,
        "score_tier": "acceptable",
        "stop_reason": ((environment.get("episode_trace") or {}).get("stop_reason")),
        "tool_call_count": int((environment.get("episode_trace") or {}).get("total_tool_calls") or 0),
        "failure_labels": _issue_labels(metadata),
        "ground_truth_tool": tool_name,
        "ground_truth_args_json": json.dumps(tool_args, sort_keys=True) if tool_args is not None else None,
        "metadata": {
            "source_artifact": path.name,
            "seed_id": ((metadata.get("environment_seed") or {}).get("seed_id")),
            "scenario": metadata.get("scenario"),
        },
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Project SynthChat rollout artifacts into canonical/KTO/GRPO datasets.")
    parser.add_argument("--input", action="append", required=True, help="Input rollout JSONL artifact. Repeat for multiple files.")
    parser.add_argument("--canonical-output", required=True, help="Output JSONL path for aggregated canonical records.")
    parser.add_argument("--kto-output", required=True, help="Output JSONL path for projected KTO records.")
    parser.add_argument("--grpo-output", required=True, help="Output JSONL path for projected GRPO records.")
    args = parser.parse_args()

    input_paths = [Path(item).expanduser().resolve() for item in args.input]
    canonical_rows: List[Dict[str, Any]] = []
    kto_rows: List[Dict[str, Any]] = []
    grpo_rows: List[Dict[str, Any]] = []

    for path, line_index, row in _iter_records(input_paths):
        canonical_rows.append(_build_canonical_row(path, row))
        kto_row = _build_kto_row(path, line_index, row)
        if kto_row is not None:
            kto_rows.append(kto_row)
        grpo_row = _build_grpo_row(path, line_index, row)
        if grpo_row is not None:
            grpo_rows.append(grpo_row)

    canonical_count = _write_jsonl(Path(args.canonical_output), canonical_rows)
    kto_count = _write_jsonl(Path(args.kto_output), kto_rows)
    grpo_count = _write_jsonl(Path(args.grpo_output), grpo_rows)

    print(json.dumps({
        "canonical_output": str(Path(args.canonical_output)),
        "canonical_count": canonical_count,
        "kto_output": str(Path(args.kto_output)),
        "kto_count": kto_count,
        "grpo_output": str(Path(args.grpo_output)),
        "grpo_count": grpo_count,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
