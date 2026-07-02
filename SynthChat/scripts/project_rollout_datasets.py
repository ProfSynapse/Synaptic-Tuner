#!/usr/bin/env python3
"""Aggregate SynthChat rollout artifacts and project KTO/GRPO datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure the repo root is importable so `shared.*` resolves when this script is
# run directly by path (the README-documented form), not just via `python -m`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.utilities import load_yaml
from shared.validation import FilterStats, RolloutFilterSet
from shared.flywheel.judge import judge_metadata_from_row

# Canonical projection-target names usable in a filter spec's ``applies_to``.
PROJECTION_TARGETS = ("sft", "grpo", "kto_positive", "kto_negative")

# Default target set for a filter that omits ``applies_to``. Quality filters
# should apply to positive/GRPO/SFT projections but MUST NOT silently drop hard
# negatives — a researcher must explicitly list ``kto_negative`` to filter them.
DEFAULT_FILTER_TARGETS = ("sft", "grpo", "kto_positive")


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


def _trace_message_content(turn: Dict[str, Any]) -> Optional[str]:
    content = turn.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _trace_prompt_messages(
    trace: List[Dict[str, Any]],
    stop_index: int,
    prompt_context: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for turn in trace[:stop_index]:
        role = turn.get("role")
        kind = turn.get("kind")
        content = _trace_message_content(turn)
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "system":
            continue
        if kind in {"judge_feedback", "validation_feedback", "final_text_request"}:
            continue
        if prompt_context == "user_only" and kind != "prompt_message":
            continue
        messages.append({"role": role, "content": content})
    return messages


def _assistant_tool_calls(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = turn.get("raw") or {}
    tool_calls = raw.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def _tool_call_name_and_args(tool_call: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    function = tool_call.get("function") or {}
    name = function.get("name")
    args_raw = function.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return None, None
    if isinstance(args_raw, dict):
        return name.strip(), args_raw
    if isinstance(args_raw, str):
        try:
            parsed = json.loads(args_raw)
        except json.JSONDecodeError:
            return None, None
        if isinstance(parsed, dict):
            return name.strip(), parsed
    return None, None


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


def _attach_judge_metadata(projected: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    judge = judge_metadata_from_row(source)
    if not judge:
        return projected
    metadata = dict(projected.get("metadata") or {})
    metadata["judge"] = judge
    projected["metadata"] = metadata
    return projected


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


def _passes_filters(
    row: Dict[str, Any],
    target: str,
    filter_set: Optional[RolloutFilterSet],
    stats: Optional[FilterStats],
) -> bool:
    """Apply the configured filter set to ``row`` for one projection target.

    Returns True when the record passes (or when no filters are configured).
    Records the decision into ``stats`` for the per-target/per-filter breakdown.
    The record is evaluated as-is so filters can address any field via dot-path
    (e.g. ``metadata.environment.episode_trace.total_turns``).
    """
    if filter_set is None or filter_set.is_empty:
        return True
    decision = filter_set.apply(row, target)
    if stats is not None:
        stats.record(target, filter_set, decision)
    return decision.passed


# Trace turn kinds that are NOT part of the trainable transcript. Mirrors the
# exclusions in _trace_prompt_messages: judge/validation feedback and the
# final-text request are environment scaffolding, not model training signal.
_SFT_DROP_KINDS = {"judge_feedback", "validation_feedback", "final_text_request"}


def _trace_assistant_content(turn: Dict[str, Any]) -> Optional[str]:
    """Authoritative assistant text for an assistant_response turn.

    Reads ``turn["raw"]["content"]`` (the real model content: a string for text
    turns, ``None`` for pure tool-call turns). The top-level ``turn["content"]``
    is a human-readable display repr (e.g. "Tool calls: [...]") and MUST NOT be
    used as training content. If the trace carries separate reasoning/thinking
    content, it is wrapped in <thinking>...</thinking> to match the thinking
    datasets — only when such a field actually exists in the trace.
    """
    raw = turn.get("raw") or {}
    content = raw.get("content")
    text = content.strip() if isinstance(content, str) and content.strip() else None

    reasoning = raw.get("reasoning") or raw.get("thinking")
    if isinstance(reasoning, str) and reasoning.strip():
        thinking_block = f"<thinking>{reasoning.strip()}</thinking>"
        return f"{thinking_block}\n\n{text}" if text else thinking_block
    return text


def _build_sft_row(
    path: Path,
    line_index: int,
    row: Dict[str, Any],
    filter_set: Optional[RolloutFilterSet] = None,
    stats: Optional[FilterStats] = None,
) -> Optional[Dict[str, Any]]:
    """Project a successful rollout into a full multi-turn SFT row.

    Reconstructs the entire conversation from ``conversation_trace`` in order,
    preserving OpenAI-style assistant ``tool_calls`` and emitting tool-result
    turns as ``tool``-role messages. Only quality positives are projected, and
    the generic filter engine is applied at the ``sft`` target seam.
    """
    metadata = row.get("metadata") or {}
    if not _is_quality_positive(metadata):
        return None

    if not _passes_filters(row, "sft", filter_set, stats):
        return None

    trace = row.get("conversation_trace") or []
    if not isinstance(trace, list):
        return None

    conversations: List[Dict[str, Any]] = []
    has_user = False
    has_assistant = False

    for turn in trace:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        kind = turn.get("kind")

        # Drop non-training scaffolding turns (judge/validation feedback, the
        # final-text request). Mirrors _trace_prompt_messages exclusions.
        if kind in _SFT_DROP_KINDS:
            continue

        if kind == "assistant_response":
            content = _trace_assistant_content(turn)
            tool_calls = _assistant_tool_calls(turn)
            if content is None and not tool_calls:
                # Degenerate empty assistant turn — skip it.
                continue
            message: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            conversations.append(message)
            has_assistant = True
            continue

        # Tool results surface in the trace as user turns tagged
        # kind == "tool_feedback"; re-tag them as canonical tool-role messages.
        if kind == "tool_feedback":
            content = _trace_message_content(turn)
            if content is None:
                continue
            conversations.append({"role": "tool", "content": content})
            continue

        if role == "system":
            content = _trace_message_content(turn)
            if content is None:
                continue
            conversations.append({"role": "system", "content": content})
            continue

        if role == "user":
            content = _trace_message_content(turn)
            if content is None:
                continue
            conversations.append({"role": "user", "content": content})
            has_user = True
            continue

        # Unknown role/kind — drop conservatively (no silent inclusion).

    if not has_assistant or not has_user:
        return None

    environment = metadata.get("environment") or {}
    episode_trace = environment.get("episode_trace") or {}

    return _attach_judge_metadata({
        "conversations": conversations,
        "label": True,
        "scenario_id": str(metadata.get("scenario") or "unknown"),
        "scenario_family": _scenario_family(metadata),
        "source_example_id": f"{path.name}:{line_index}",
        "total_turns": int(episode_trace.get("total_turns") or 0),
        "metadata": {
            "source_artifact": path.name,
            "seed_id": ((metadata.get("environment_seed") or {}).get("seed_id")),
            "scenario": metadata.get("scenario"),
            "stop_reason": episode_trace.get("stop_reason"),
        },
    }, row)


def _build_kto_row(
    path: Path,
    line_index: int,
    row: Dict[str, Any],
    filter_set: Optional[RolloutFilterSet] = None,
    stats: Optional[FilterStats] = None,
) -> Optional[Dict[str, Any]]:
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

    # Apply config filters at the per-target seam. kto_negative is only filtered
    # when a researcher explicitly lists it in a filter's ``applies_to`` (it is
    # excluded from DEFAULT_FILTER_TARGETS).
    kto_target = "kto_positive" if label is True else "kto_negative"
    if not _passes_filters(row, kto_target, filter_set, stats):
        return None

    return _attach_judge_metadata({
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
    }, row)


def _build_grpo_row(
    path: Path,
    line_index: int,
    row: Dict[str, Any],
    filter_set: Optional[RolloutFilterSet] = None,
    stats: Optional[FilterStats] = None,
) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata") or {}
    if not _is_quality_positive(metadata):
        return None

    if not _passes_filters(row, "grpo", filter_set, stats):
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

    return _attach_judge_metadata({
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
        "total_turns": int((environment.get("episode_trace") or {}).get("total_turns") or 0),
        "failure_labels": _issue_labels(metadata),
        "ground_truth_tool": tool_name,
        "ground_truth_args_json": json.dumps(tool_args, sort_keys=True) if tool_args is not None else None,
        "metadata": {
            "source_artifact": path.name,
            "seed_id": ((metadata.get("environment_seed") or {}).get("seed_id")),
            "scenario": metadata.get("scenario"),
        },
    }, row)


def _build_grpo_tool_turn_rows(
    path: Path,
    line_index: int,
    row: Dict[str, Any],
    prompt_context: str,
    filter_set: Optional[RolloutFilterSet] = None,
    stats: Optional[FilterStats] = None,
) -> List[Dict[str, Any]]:
    metadata = row.get("metadata") or {}
    if not _is_quality_positive(metadata):
        return []

    if not _passes_filters(row, "grpo", filter_set, stats):
        return []

    trace = row.get("conversation_trace") or []
    if not isinstance(trace, list):
        return []

    rows: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(trace):
        if turn.get("kind") != "assistant_response":
            continue
        tool_calls = _assistant_tool_calls(turn)
        if not tool_calls:
            continue
        tool_name, tool_args = _tool_call_name_and_args(tool_calls[0])
        if not tool_name or tool_args is None:
            continue
        prompt = _trace_prompt_messages(trace, turn_index, prompt_context)
        if not prompt:
            continue
        rows.append(_attach_judge_metadata({
            "prompt": prompt,
            "scenario_id": str(metadata.get("scenario") or "unknown"),
            "scenario_family": _scenario_family(metadata),
            "source_example_id": f"{path.name}:{line_index}:turn:{turn_index}",
            "allowed_tools": [tool_name],
            "environment_passed": True,
            "schema_passed": True,
            "score_value": 1.0,
            "score_tier": "acceptable",
            "stop_reason": ((metadata.get("environment") or {}).get("episode_trace") or {}).get("stop_reason"),
            "tool_call_count": int(((metadata.get("environment") or {}).get("episode_trace") or {}).get("total_tool_calls") or 0),
            "total_turns": int(((metadata.get("environment") or {}).get("episode_trace") or {}).get("total_turns") or 0),
            "failure_labels": _issue_labels(metadata),
            "ground_truth_tool": tool_name,
            "ground_truth_args_json": json.dumps(tool_args, sort_keys=True) if tool_args is not None else None,
            "metadata": {
                "source_artifact": path.name,
                "seed_id": ((metadata.get("environment_seed") or {}).get("seed_id")),
                "scenario": metadata.get("scenario"),
                "projection": "tool_turns",
                "source_turn_index": turn_index,
                "prompt_context": prompt_context,
            },
        }, row))
    return rows


def _load_filter_set(filter_config: Optional[str]) -> Optional[RolloutFilterSet]:
    """Build a RolloutFilterSet from a YAML config file's projection.filters list.

    The config channel mirrors this file's existing YAML/dict-of-config idiom:
    a single YAML document whose ``projection.filters`` key holds the list of
    declarative filter specs. Returns None when no config is supplied (no-op).
    Invalid specs raise ValueError eagerly (fail-closed) at load time.
    """
    if not filter_config:
        return None
    config = load_yaml(filter_config)
    projection = config.get("projection") if isinstance(config, dict) else None
    filters = (projection or {}).get("filters") if isinstance(projection, dict) else None
    return RolloutFilterSet(filters=filters or [], default_targets=DEFAULT_FILTER_TARGETS)


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
    parser.add_argument(
        "--sft-output",
        default=None,
        help=(
            "Optional output JSONL path for full multi-turn SFT records. When "
            "provided, each quality-positive rollout is projected into a single "
            "multi-turn conversation (system/user/assistant/tool turns). Omit to "
            "skip SFT projection (keeps the existing canonical/KTO/GRPO invocation)."
        ),
    )
    parser.add_argument(
        "--grpo-projection",
        choices=["first_tool", "tool_turns"],
        default="first_tool",
        help="How to project successful rollouts into GRPO rows.",
    )
    parser.add_argument(
        "--grpo-prompt-context",
        choices=["user_only", "prior_real_messages"],
        default="user_only",
        help="Prompt context for tool_turns projection. prior_real_messages keeps user/assistant/tool-feedback turns and drops system/judge/validation turns.",
    )
    parser.add_argument(
        "--filter-config",
        default=None,
        help=(
            "Path to a YAML file with a 'projection.filters' list of declarative "
            "rollout filters applied per projection target "
            f"({', '.join(PROJECTION_TARGETS)}). See shared/validation/rollout_filters.py "
            "for the spec and a YAML example."
        ),
    )
    args = parser.parse_args()

    filter_set = _load_filter_set(args.filter_config)
    filter_stats = FilterStats()

    input_paths = [Path(item).expanduser().resolve() for item in args.input]
    canonical_rows: List[Dict[str, Any]] = []
    kto_rows: List[Dict[str, Any]] = []
    grpo_rows: List[Dict[str, Any]] = []
    sft_rows: List[Dict[str, Any]] = []

    for path, line_index, row in _iter_records(input_paths):
        canonical_rows.append(_build_canonical_row(path, row))
        kto_row = _build_kto_row(path, line_index, row, filter_set, filter_stats)
        if kto_row is not None:
            kto_rows.append(kto_row)
        if args.sft_output:
            sft_row = _build_sft_row(path, line_index, row, filter_set, filter_stats)
            if sft_row is not None:
                sft_rows.append(sft_row)
        if args.grpo_projection == "tool_turns":
            grpo_rows.extend(
                _build_grpo_tool_turn_rows(
                    path, line_index, row, args.grpo_prompt_context, filter_set, filter_stats
                )
            )
        else:
            grpo_row = _build_grpo_row(path, line_index, row, filter_set, filter_stats)
            if grpo_row is not None:
                grpo_rows.append(grpo_row)

    canonical_count = _write_jsonl(Path(args.canonical_output), canonical_rows)
    kto_count = _write_jsonl(Path(args.kto_output), kto_rows)
    grpo_count = _write_jsonl(Path(args.grpo_output), grpo_rows)

    summary: Dict[str, Any] = {
        "canonical_output": str(Path(args.canonical_output)),
        "canonical_count": canonical_count,
        "kto_output": str(Path(args.kto_output)),
        "kto_count": kto_count,
        "grpo_output": str(Path(args.grpo_output)),
        "grpo_count": grpo_count,
        "grpo_projection": args.grpo_projection,
        "grpo_prompt_context": args.grpo_prompt_context,
    }
    if args.sft_output:
        sft_count = _write_jsonl(Path(args.sft_output), sft_rows)
        summary["sft_output"] = str(Path(args.sft_output))
        summary["sft_count"] = sft_count
    if filter_set is not None and not filter_set.is_empty:
        summary["filter_config"] = str(Path(args.filter_config))
        summary["filter_stats"] = filter_stats.as_dict()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
