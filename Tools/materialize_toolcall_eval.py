#!/usr/bin/env python3
"""Materialize the private holdout dataset into gitignored tool-call eval cases.

Location: Tools/materialize_toolcall_eval.py

WHAT THIS DOES
    Downloads a private HuggingFace dataset parquet shard (the personal-transcripts
    holdout), walks every conversation, and emits one eval case per assistant turn
    that issues tool calls. Each case feeds the tool-call-accuracy eval driven by
    the ``args_match`` verifier (shared/verifiers/builtins/args_match.py). The
    emitted JSONL is consumed by Evaluator/prompt_sets.py::_build_case, which keeps
    arbitrary top-level keys into ``PromptCase.metadata`` (so ``system``, ``messages``,
    ``ground_truth`` and ``verifiers`` all survive as metadata for the runner/verifier).

HOW IT IS USED
    python Tools/materialize_toolcall_eval.py \
        --dataset professorsynapse/personal-transcripts-sft \
        --file data/test-00000-of-00001.parquet \
        --out personal_finetune/eval/toolcall_cases.jsonl \
        --drop-report personal_finetune/eval/materialize_report.json

PRIVACY (SACROSANCT)
    The holdout is PRIVATE personal data. Both --out and --drop-report MUST resolve
    under a gitignored root (/personal_finetune/ per .gitignore:170, or a scratch/
    directory). The script asserts this BEFORE any write and refuses otherwise.
    The HF token is loaded from the repo-root .env (never logged).

This is a Tools/ utility (no application-state mutation); it is run-phase only —
do NOT run it against the real dataset during the Code phase.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Repo root = parent of Tools/ ; used for .env discovery and path-guard roots.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Char/token proxy: ~3.6 chars per token is a reasonable English-text estimate.
CHARS_PER_TOKEN = 3.6
DEFAULT_MAX_PROMPT_TOKENS = 16384


# ---------------------------------------------------------------------------
# Environment / privacy guard
# ---------------------------------------------------------------------------

def load_env_from_dotenv() -> None:
    """Load repo-root .env into os.environ (same pattern as scratch/health_check.py).

    Existing environment values win (setdefault), so an externally-exported HF
    token is never clobbered. Quotes around values are stripped.
    """
    envp = REPO_ROOT / ".env"
    if not envp.exists():
        return
    for line in envp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def assert_gitignored_path(path: pathlib.Path, *, arg_name: str) -> pathlib.Path:
    """Fail loud unless ``path`` resolves under a gitignored root.

    Allowed roots: /personal_finetune/ (gitignored at .gitignore:170) or any
    ``scratch/`` directory. The holdout is private personal data; writing it
    anywhere git-tracked would leak it. Returns the resolved absolute path.
    """
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    # Accept if any path segment is the personal_finetune workspace or a scratch dir.
    allowed = ("personal_finetune" in parts) or ("scratch" in parts)
    if not allowed:
        raise SystemExit(
            f"PRIVACY GUARD: refusing to write {arg_name}={resolved}\n"
            f"  Output must live under a gitignored root (/personal_finetune/ "
            f"or a scratch/ dir). The holdout is PRIVATE personal data."
        )
    return resolved


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_parquet(dataset: str, file: str) -> pathlib.Path:
    """Download the dataset parquet shard via huggingface_hub and return its path."""
    from huggingface_hub import hf_hub_download

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    local = hf_hub_download(
        repo_id=dataset,
        filename=file,
        repo_type="dataset",
        token=token,
    )
    return pathlib.Path(local)


def read_conversations(parquet_path: pathlib.Path) -> List[List[Dict[str, Any]]]:
    """Read the ``conversations`` column from the parquet into a list of turn-lists.

    Each row carries a single ``conversations`` column: a list of structs
    ``{role, content, reasoning_content, tool_calls}``. Uses pyarrow directly to
    avoid pulling the heavy datasets library.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["conversations"])
    column = table.column("conversations").to_pylist()
    conversations: List[List[Dict[str, Any]]] = []
    for row in column:
        if isinstance(row, list):
            conversations.append([_to_plain_turn(t) for t in row])
    return conversations


def _to_plain_turn(turn: Any) -> Dict[str, Any]:
    """Coerce a turn struct (dict-like) into a plain dict with the expected keys."""
    if not isinstance(turn, dict):
        return {"role": "", "content": "", "reasoning_content": None, "tool_calls": None}
    return {
        "role": turn.get("role", ""),
        "content": turn.get("content", ""),
        "reasoning_content": turn.get("reasoning_content"),
        "tool_calls": turn.get("tool_calls"),
    }


# ---------------------------------------------------------------------------
# Argument normalization
# ---------------------------------------------------------------------------

def parse_arguments(raw: Any, stats: Dict[str, int]) -> Any:
    """Normalize a tool-call ``arguments`` field (a JSON STRING in the source).

    Calls ``json.loads`` ONCE. On success returns the parsed value (dict/list/
    scalar); on failure keeps the raw string and increments
    ``stats['unparseable_arguments']``. None stays None.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            stats["unparseable_arguments"] += 1
            return raw
    # Already a structured object (defensive: source is a string, but tolerate dicts).
    return raw


def normalize_tool_calls(tool_calls: Any, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    """Map a turn's ``tool_calls`` list into [{tool_name, arguments(dict)}].

    ``tool_calls`` entries look like ``{type, function:{name, arguments(JSON STRING)}}``.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return out
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        arguments = parse_arguments(function.get("arguments"), stats)
        out.append({"tool_name": name, "arguments": arguments})
    return out


def _has_tool_calls(turn: Dict[str, Any]) -> bool:
    tc = turn.get("tool_calls")
    return isinstance(tc, list) and len(tc) > 0


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate prompt token length via a char/3.6 proxy across all messages."""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(chars / CHARS_PER_TOKEN)


def build_candidates(
    conversations: List[List[Dict[str, Any]]],
    stats: Dict[str, int],
    max_prompt_tokens: int,
) -> List[Dict[str, Any]]:
    """Walk all conversations and emit one candidate case per tool-calling turn.

    For each assistant turn whose ``tool_calls`` is non-empty:
      - ``messages``  = all turns UP TO BUT EXCLUDING that turn, mapped to
                        {role, content} only (reasoning_content + tool_calls dropped).
      - ``system``    = the conversation's system turn content verbatim (or "").
      - ``question``  = the text of the last user turn before the target turn.
      - ``ground_truth`` = that turn's tool_calls (first-call tool_name + arguments
                        dict + all_calls list).

    Oversize candidates (estimated prompt tokens > max_prompt_tokens) are dropped
    and counted under ``stats['oversize_dropped']``.
    """
    candidates: List[Dict[str, Any]] = []

    for conversation in conversations:
        # System turn content (verbatim). Count conversations that lack one.
        system_content = ""
        has_system = False
        for turn in conversation:
            if turn.get("role") == "system":
                system_content = str(turn.get("content") or "")
                has_system = True
                break
        if not has_system:
            stats["no_system_turn"] += 1

        for idx, turn in enumerate(conversation):
            if turn.get("role") != "assistant" or not _has_tool_calls(turn):
                continue

            all_calls = normalize_tool_calls(turn.get("tool_calls"), stats)
            if not all_calls:
                continue
            first = all_calls[0]
            first_tool = first["tool_name"]
            if not first_tool:
                stats["empty_first_tool_name"] += 1
                continue

            # Prompt-side history: all prior turns, {role, content} only.
            history = conversation[:idx]
            messages = [
                {"role": str(t.get("role", "")), "content": str(t.get("content") or "")}
                for t in history
                if str(t.get("role", "")).strip()
            ]

            # Last user turn text (informational fallback; messages is source of truth).
            question = ""
            for t in reversed(history):
                if t.get("role") == "user":
                    question = str(t.get("content") or "")
                    break

            # Token-length guard: a truncated toolbox makes the case unanswerable.
            if estimate_prompt_tokens(messages) > max_prompt_tokens:
                stats["oversize_dropped"] += 1
                continue

            candidates.append(
                {
                    "first_tool": first_tool,
                    "system": system_content,
                    "question": question,
                    "messages": messages,
                    "ground_truth": {
                        "tool_name": first_tool,
                        "arguments": first["arguments"] if first["arguments"] is not None else {},
                        "all_calls": all_calls,
                    },
                }
            )

    return candidates


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    candidates: List[Dict[str, Any]],
    sample_budget: int,
    per_tool_cap: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Stratify candidates by first-call tool name, cap per group, fill round-robin.

    1. Group by ``first_tool``.
    2. Shuffle each group deterministically (seeded) and keep at most ``per_tool_cap``.
    3. Fill round-robin across groups toward ``sample_budget`` so the long tail of
       rare tools gets coverage instead of being crowded out by frequent tools.

    Returns (selected, per_tool_report) where per_tool_report[tool] =
    {n_available, n_kept, n_capped}.
    """
    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        groups[cand["first_tool"]].append(cand)

    per_tool_report: Dict[str, Dict[str, int]] = {}
    capped_groups: Dict[str, List[Dict[str, Any]]] = {}
    for tool, items in groups.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        kept_after_cap = shuffled[:per_tool_cap]
        capped_groups[tool] = kept_after_cap
        per_tool_report[tool] = {
            "n_available": len(items),
            "n_capped": max(0, len(items) - len(kept_after_cap)),
            "n_kept": 0,  # filled after the round-robin fill below
        }

    # Round-robin fill toward the budget across capped groups (deterministic order:
    # sorted by tool name so the result is reproducible for a given seed).
    selected: List[Dict[str, Any]] = []
    cursors = {tool: 0 for tool in capped_groups}
    tool_order = sorted(capped_groups.keys())
    while len(selected) < sample_budget:
        progressed = False
        for tool in tool_order:
            if len(selected) >= sample_budget:
                break
            cursor = cursors[tool]
            bucket = capped_groups[tool]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                cursors[tool] = cursor + 1
                per_tool_report[tool]["n_kept"] += 1
                progressed = True
        if not progressed:
            # All capped groups exhausted before reaching the budget.
            break

    return selected, per_tool_report


# ---------------------------------------------------------------------------
# Case emission
# ---------------------------------------------------------------------------

def build_case(case_index: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Render a selected candidate into the exact emitted JSONL case schema.

    Field names are load-bearing: Evaluator/prompt_sets.py::_build_case keeps every
    top-level key except {id, question, prompt, tags} into PromptCase.metadata, so
    ``system``/``messages``/``ground_truth``/``verifiers`` all become metadata.
    """
    first_tool = candidate["first_tool"]
    return {
        "id": f"tc_{case_index:06d}",
        "question": candidate["question"],
        "tags": ["toolcall_holdout", f"tool:{first_tool}"],
        "system": candidate["system"],
        "messages": candidate["messages"],
        "ground_truth": candidate["ground_truth"],
        "verifiers": [
            {
                "type": "args_match",
                "params": {
                    "scheme": "overlap",
                    "gt_tool_field": "tool_name",
                    "gt_args_field": "arguments",
                    "pass_threshold": 0.5,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    """End-to-end materialization; returns the drop report dict."""
    out_path = assert_gitignored_path(pathlib.Path(args.out), arg_name="--out")
    report_path = assert_gitignored_path(pathlib.Path(args.drop_report), arg_name="--drop-report")

    load_env_from_dotenv()
    parquet_path = download_parquet(args.dataset, args.file)
    conversations = read_conversations(parquet_path)

    stats = defaultdict(int)
    candidates = build_candidates(conversations, stats, args.max_prompt_tokens)
    selected, per_tool_report = stratified_sample(
        candidates, args.sample_budget, args.per_tool_cap, args.seed
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for i, candidate in enumerate(selected):
            handle.write(json.dumps(build_case(i, candidate), ensure_ascii=False) + "\n")

    report = {
        "dataset": args.dataset,
        "file": args.file,
        "seed": args.seed,
        "sample_budget": args.sample_budget,
        "per_tool_cap": args.per_tool_cap,
        "max_prompt_tokens": args.max_prompt_tokens,
        "totals": {
            "conversations": len(conversations),
            "candidates_available": len(candidates),
            "cases_emitted": len(selected),
            "distinct_tools_available": len(per_tool_report),
            "distinct_tools_emitted": sum(1 for r in per_tool_report.values() if r["n_kept"] > 0),
            "oversize_dropped": stats.get("oversize_dropped", 0),
            "no_system_turn": stats.get("no_system_turn", 0),
            "empty_first_tool_name": stats.get("empty_first_tool_name", 0),
            "unparseable_arguments": stats.get("unparseable_arguments", 0),
        },
        "per_tool": {
            tool: per_tool_report[tool] for tool in sorted(per_tool_report)
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the private holdout into gitignored tool-call eval cases."
    )
    parser.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. professorsynapse/personal-transcripts-sft")
    parser.add_argument("--file", required=True, help="Parquet path within the dataset, e.g. data/test-00000-of-00001.parquet")
    parser.add_argument("--sample-budget", type=int, default=500, help="Target number of emitted cases (default 500)")
    parser.add_argument("--per-tool-cap", type=int, default=12, help="Max cases per first-call tool name (default 12)")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic sampling seed (default 1234)")
    parser.add_argument("--out", required=True, help="Output JSONL path (must be under a gitignored root)")
    parser.add_argument("--drop-report", required=True, help="Drop/cap report JSON path (must be under a gitignored root)")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"Drop cases whose estimated prompt exceeds this many tokens (default {DEFAULT_MAX_PROMPT_TOKENS})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = materialize(args)
    totals = report["totals"]
    print(
        f"Emitted {totals['cases_emitted']} cases "
        f"({totals['distinct_tools_emitted']}/{totals['distinct_tools_available']} tools) "
        f"from {totals['conversations']} conversations; "
        f"oversize_dropped={totals['oversize_dropped']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
