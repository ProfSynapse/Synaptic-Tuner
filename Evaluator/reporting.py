"""Reporting helpers for evaluation runs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

from .runner import EvaluationRecord


def aggregate_stats(records: Sequence[EvaluationRecord]) -> Dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record.passed)
    errors = sum(1 for record in records if record.error)

    by_tag = defaultdict(lambda: {"total": 0, "passed": 0})
    for record in records:
        tags = record.case.tags or ["__untagged__"]
        for tag in tags:
            bucket = by_tag[tag]
            bucket["total"] += 1
            if record.passed:
                bucket["passed"] += 1

    failure_reasons = Counter()
    for record in records:
        if record.error:
            failure_reasons[record.error] += 1
        elif record.validator and not record.validator.passed:
            for issue in record.validator.issues:
                if issue.level.upper() == "ERROR":
                    failure_reasons[issue.message] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "request_errors": errors,
        "pass_rate": (passed / total) if total else 0,
        "by_tag": {
            tag: {
                "total": bucket["total"],
                "passed": bucket["passed"],
                "pass_rate": (bucket["passed"] / bucket["total"]) if bucket["total"] else 0,
            }
            for tag, bucket in sorted(by_tag.items())
        },
        "top_failure_reasons": failure_reasons.most_common(10),
    }


def console_summary(records: Sequence[EvaluationRecord]) -> str:
    stats = aggregate_stats(records)
    lines = [
        f"Evaluated {stats['total']} prompt(s): {stats['passed']} passed, {stats['failed']} failed.",
        f"Request errors: {stats['request_errors']}",
        "Pass rate by tag:",
    ]
    for tag, bucket in stats["by_tag"].items():
        percent = bucket["pass_rate"] * 100
        lines.append(f"  - {tag}: {bucket['passed']}/{bucket['total']} ({percent:.1f}%)")
    if stats["top_failure_reasons"]:
        lines.append("Top failure reasons:")
        for reason, count in stats["top_failure_reasons"]:
            lines.append(f"  - {count}× {reason}")
    return "\n".join(lines)


def build_run_payload(
    records: Sequence[EvaluationRecord],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": {
            **metadata,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": aggregate_stats(records),
        "records": [record_to_dict(record) for record in records],
    }


def record_to_dict(record: EvaluationRecord) -> Dict[str, Any]:
    validator = record.validator.to_dict() if record.validator else None
    return {
        "case_id": record.case.case_id,
        "question": record.case.question,
        "tags": record.case.tags,
        "expected_tools": record.case.expected_tools,
        "response_text": record.response_text,
        "latency_s": record.latency_s,
        "passed": record.passed,
        "error": record.error,
        "validator": validator,
        "raw_response": record.raw_response,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(records: Sequence[EvaluationRecord]) -> str:
    stats = aggregate_stats(records)
    lines = [
        f"# Evaluator Run ({stats['total']} prompts)",
        "",
        f"- Passed: **{stats['passed']}**",
        f"- Failed: **{stats['failed']}**",
        f"- Request errors: **{stats['request_errors']}**",
        "",
        "## By Tag",
    ]
    for tag, bucket in stats["by_tag"].items():
        lines.append(f"- `{tag}`: {bucket['passed']}/{bucket['total']} ({bucket['pass_rate']*100:.1f}%)")
    if stats["top_failure_reasons"]:
        lines.append("")
        lines.append("## Top Failure Reasons")
        for reason, count in stats["top_failure_reasons"]:
            lines.append(f"- {count}× {reason}")
    return "\n".join(lines)
