#!/usr/bin/env python3
"""Aggregate an Evaluator tool-call run into a per-tool accuracy report.

Location: Tools/aggregate_toolcall_report.py

WHAT THIS DOES
    Reads an Evaluator run JSON (the file written via --output, shape produced by
    Evaluator/reporting.py::build_run_payload -> {metadata, summary, records:[...]}).
    For every record that carries a tool-call ``verifier`` result it groups by the
    reference tool (the ground-truth tool name) and computes per-tool and overall:
      - n                : number of verifier-bearing records for that tool
      - name_match_rate  : fraction with verifier.score > 0 (the overlap scheme
                           returns 0.0 on a tool-name mismatch, so score>0 implies
                           the predicted tool name matched)
      - mean_arg_overlap : mean of verifier.score

HOW IT IS USED
    python Tools/aggregate_toolcall_report.py \
        --run-json personal_finetune/eval/runs/holdout_run.json \
        --out personal_finetune/eval/reports/toolcall_accuracy.json

PRIVACY (SACROSANCT)
    The run JSON is derived from PRIVATE personal data. This report emits ONLY tool
    names (schema identifiers), counts, and rates — NEVER message/content/argument
    text. The --out path must resolve under a gitignored root, and a guard asserts
    every emitted per_tool key looks like a tool identifier (no free-text leakage).

DEVIATION / ASSUMPTION (read Evaluator/reporting.py)
    The current Evaluator/reporting.py::record_to_dict does NOT emit a top-level
    ``verifier`` key or a ``case.metadata`` block — that wiring is owned by the
    coder working on Evaluator/. Per the build contract, records are expected to
    carry verifier.{name,score,passed,detail} and case.metadata.ground_truth.tool_name.
    To stay robust to the exact nesting that wiring lands on, this script resolves
    both the verifier block and the reference tool name through a small set of
    fallback paths (documented in _extract_verifier / _extract_reference_tool).
    Records WITHOUT a resolvable verifier block are skipped (per the contract).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# A tool identifier looks like a schema name: word chars, dots, hyphens, slashes,
# colons (e.g. "agent_tool", "custom_objects.touchpoints"). It must NOT look like
# free-text prose (no spaces, no sentence punctuation). Used as the privacy guard.
_TOOL_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
# Sentinel for records whose reference tool name could not be resolved.
_UNKNOWN_TOOL = "__unknown__"


def assert_gitignored_path(path: pathlib.Path, *, arg_name: str) -> pathlib.Path:
    """Fail loud unless ``path`` resolves under a gitignored root.

    Mirrors the guard in materialize_toolcall_eval.py: allowed roots are
    /personal_finetune/ (gitignored at .gitignore:170) or any scratch/ dir.
    """
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    allowed = ("personal_finetune" in parts) or ("scratch" in parts)
    if not allowed:
        raise SystemExit(
            f"PRIVACY GUARD: refusing to write {arg_name}={resolved}\n"
            f"  Output must live under a gitignored root (/personal_finetune/ "
            f"or a scratch/ dir). This report derives from PRIVATE data."
        )
    return resolved


def load_records(run_json_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load the records list from an Evaluator run JSON.

    build_run_payload (Evaluator/reporting.py) writes {metadata, summary, records:[...]}.
    Tolerate a bare top-level list as a fallback (defensive; not the canonical shape).
    """
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
        raise SystemExit(
            f"Run JSON {run_json_path} has no 'records' list "
            f"(expected build_run_payload shape: {{metadata, summary, records}})."
        )
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    raise SystemExit(f"Run JSON {run_json_path} is neither an object nor a list.")


def _extract_verifier(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the tool-call verifier result block from a record, or None.

    Contract shape: record['verifier'] = {name, score, passed, detail}. Because the
    Evaluator verifier-wiring is owned by another coder and the exact key may land
    as a single dict or a list of verifier results, resolve through fallbacks:
      1. record['verifier']            -> dict
      2. record['verifiers']           -> list; take the args_match one (or the first)
    Returns a dict exposing at least 'score'; None if no verifier result is present.
    """
    block = record.get("verifier")
    if isinstance(block, dict) and "score" in block:
        return block

    plural = record.get("verifiers")
    if isinstance(plural, list):
        scored = [v for v in plural if isinstance(v, dict) and "score" in v]
        if scored:
            # Prefer an args_match result if the name is present; else first scored.
            for v in scored:
                if str(v.get("name", "")).strip() == "args_match":
                    return v
            return scored[0]
    return None


def _extract_reference_tool(record: Dict[str, Any]) -> str:
    """Resolve the ground-truth (reference) tool name from a record.

    Contract path: record['case']['metadata']['ground_truth']['tool_name']. The
    materializer also exposes ground_truth at the case top level (since
    prompt_sets._build_case keeps unknown keys into PromptCase.metadata, the runner
    may surface it under case.metadata OR flattened). Resolve through fallbacks and
    return _UNKNOWN_TOOL if none is found.
    """
    candidates: List[Any] = []

    case = record.get("case")
    if isinstance(case, dict):
        meta = case.get("metadata")
        if isinstance(meta, dict):
            candidates.append(_dig(meta, ("ground_truth", "tool_name")))
            candidates.append(_dig(meta, ("ground_truth", "tool")))

    # Some run shapes surface metadata at the record top level.
    candidates.append(_dig(record, ("metadata", "ground_truth", "tool_name")))
    candidates.append(_dig(record, ("ground_truth", "tool_name")))
    # Last resort: the verifier detail records the gt_tool it scored against.
    verifier = _extract_verifier(record) or {}
    detail = verifier.get("detail")
    if isinstance(detail, dict):
        candidates.append(detail.get("gt_tool"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _UNKNOWN_TOOL


def _dig(obj: Any, path: tuple) -> Any:
    """Walk a nested dict by key path; return None on any miss."""
    current = obj
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _detect_multi_call(records: List[Dict[str, Any]]) -> bool:
    """Contract: multi-call scoring is NEVER in play, so this is ALWAYS False.

    The args_match overlap scheme scores ONLY the first call of a turn. Whether or
    not the underlying ground truth happened to carry multiple calls is irrelevant
    to what was *scored* — the report must not imply multi-call scoring occurred.
    Per the build contract this flag is therefore unconditionally ``False``.

    ``records`` is accepted for signature stability with the caller but is not
    inspected: there is no record shape that can flip this to True.
    """
    return False


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall + per-tool {n, name_match_rate, mean_arg_overlap}.

    Only records with a resolvable verifier block are considered.
    """
    per_tool_scores: Dict[str, List[float]] = defaultdict(list)
    overall_scores: List[float] = []

    for record in records:
        verifier = _extract_verifier(record)
        if verifier is None:
            continue
        try:
            score = float(verifier.get("score"))
        except (TypeError, ValueError):
            continue
        tool = _extract_reference_tool(record)
        per_tool_scores[tool].append(score)
        overall_scores.append(score)

    def _bucket(scores: List[float]) -> Dict[str, Any]:
        n = len(scores)
        if n == 0:
            return {"n": 0, "name_match_rate": 0.0, "mean_arg_overlap": 0.0}
        name_matches = sum(1 for s in scores if s > 0)
        return {
            "n": n,
            "name_match_rate": name_matches / n,
            "mean_arg_overlap": sum(scores) / n,
        }

    per_tool = {tool: _bucket(scores) for tool, scores in sorted(per_tool_scores.items())}
    return {"overall": _bucket(overall_scores), "per_tool": per_tool}


def assert_no_freetext_keys(per_tool: Dict[str, Any]) -> None:
    """Privacy guard: every emitted per_tool key must look like a tool identifier.

    Catches the failure mode where a free-text fragment (message/content) leaks into
    a grouping key. The _UNKNOWN_TOOL sentinel is allowed (it is not user content).
    """
    for key in per_tool:
        if key == _UNKNOWN_TOOL:
            continue
        if not _TOOL_ID_RE.match(key):
            raise SystemExit(
                f"PRIVACY GUARD: per_tool key {key!r} does not look like a tool "
                f"identifier — refusing to write (possible free-text leakage)."
            )


def build_report(run_json_path: pathlib.Path) -> Dict[str, Any]:
    records = load_records(run_json_path)
    aggregates = aggregate(records)
    assert_no_freetext_keys(aggregates["per_tool"])
    return {
        "overall": aggregates["overall"],
        "per_tool": aggregates["per_tool"],
        "meta": {
            "run_json": str(run_json_path),
            "multi_call_scored": _detect_multi_call(records),
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate an Evaluator tool-call run JSON into a per-tool accuracy report."
    )
    parser.add_argument("--run-json", required=True, help="Evaluator run JSON (the --output file)")
    parser.add_argument("--out", required=True, help="Report JSON path (must be under a gitignored root)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_json_path = pathlib.Path(args.run_json).expanduser().resolve()
    if not run_json_path.exists():
        raise SystemExit(f"Run JSON not found: {run_json_path}")
    out_path = assert_gitignored_path(pathlib.Path(args.out), arg_name="--out")

    report = build_report(run_json_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = report["overall"]
    print(
        f"Wrote {out_path} | overall n={overall['n']} "
        f"name_match_rate={overall['name_match_rate']:.3f} "
        f"mean_arg_overlap={overall['mean_arg_overlap']:.3f} "
        f"({len(report['per_tool'])} tools)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
