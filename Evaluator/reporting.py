"""Reporting helpers for evaluation runs."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .runner import EvaluationRecord


def aggregate_stats(records: Sequence[EvaluationRecord]) -> Dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record.passed)
    errors = sum(1 for record in records if record.error)

    # Track schema vs behavior pass rates separately
    schema_passed = sum(1 for record in records if record.schema_passed)
    behavior_tested = sum(1 for record in records if record.behavior is not None)
    behavior_passed = sum(1 for record in records if record.behavior_passed and record.behavior is not None)

    by_tag = defaultdict(lambda: {"total": 0, "passed": 0, "schema_passed": 0, "behavior_passed": 0, "behavior_tested": 0})
    for record in records:
        tags = record.case.tags or ["__untagged__"]
        for tag in tags:
            bucket = by_tag[tag]
            bucket["total"] += 1
            if record.passed:
                bucket["passed"] += 1
            if record.schema_passed:
                bucket["schema_passed"] += 1
            if record.behavior is not None:
                bucket["behavior_tested"] += 1
                if record.behavior_passed:
                    bucket["behavior_passed"] += 1

    failure_reasons = Counter()
    behavior_failures = Counter()
    for record in records:
        if record.error:
            failure_reasons[record.error] += 1
        elif record.validator and not record.validator.passed:
            for issue in record.validator.issues:
                if issue.level.upper() == "ERROR":
                    failure_reasons[issue.message] += 1
        # Track behavior failures separately
        if record.behavior and not record.behavior.passed:
            for issue in record.behavior.issues:
                if not issue.passed:
                    behavior_failures[f"{issue.check}: {issue.message}"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "request_errors": errors,
        "pass_rate": (passed / total) if total else 0,
        # Detailed breakdown
        "schema_passed": schema_passed,
        "schema_pass_rate": (schema_passed / total) if total else 0,
        "behavior_tested": behavior_tested,
        "behavior_passed": behavior_passed,
        "behavior_pass_rate": (behavior_passed / behavior_tested) if behavior_tested else 0,
        "by_tag": {
            tag: {
                "total": bucket["total"],
                "passed": bucket["passed"],
                "pass_rate": (bucket["passed"] / bucket["total"]) if bucket["total"] else 0,
                "schema_passed": bucket["schema_passed"],
                "behavior_tested": bucket["behavior_tested"],
                "behavior_passed": bucket["behavior_passed"],
                "behavior_pass_rate": (bucket["behavior_passed"] / bucket["behavior_tested"]) if bucket["behavior_tested"] else 0,
            }
            for tag, bucket in sorted(by_tag.items())
        },
        "top_failure_reasons": failure_reasons.most_common(10),
        "top_behavior_failures": behavior_failures.most_common(10),
    }


def console_summary(records: Sequence[EvaluationRecord]) -> str:
    stats = aggregate_stats(records)
    lines = [
        f"Evaluated {stats['total']} prompt(s): {stats['passed']} passed, {stats['failed']} failed.",
        f"  Schema validation: {stats['schema_passed']}/{stats['total']} ({stats['schema_pass_rate']*100:.1f}%)",
    ]
    if stats['behavior_tested'] > 0:
        lines.append(f"  Behavior validation: {stats['behavior_passed']}/{stats['behavior_tested']} ({stats['behavior_pass_rate']*100:.1f}%)")
    lines.append(f"Request errors: {stats['request_errors']}")
    lines.append("Pass rate by tag:")
    for tag, bucket in stats["by_tag"].items():
        percent = bucket["pass_rate"] * 100
        line = f"  - {tag}: {bucket['passed']}/{bucket['total']} ({percent:.1f}%)"
        if bucket["behavior_tested"] > 0:
            beh_pct = bucket["behavior_pass_rate"] * 100
            line += f" [behavior: {bucket['behavior_passed']}/{bucket['behavior_tested']} ({beh_pct:.1f}%)]"
        lines.append(line)
    if stats["top_failure_reasons"]:
        lines.append("Top schema failure reasons:")
        for reason, count in stats["top_failure_reasons"]:
            lines.append(f"  - {count}× {reason}")
    if stats["top_behavior_failures"]:
        lines.append("Top behavior failure reasons:")
        for reason, count in stats["top_behavior_failures"]:
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
    behavior = record.behavior.to_dict() if record.behavior else None
    return {
        "case_id": record.case.case_id,
        "question": record.case.question,
        "tags": record.case.tags,
        "expected_tools": record.case.expected_tools,
        "response_text": record.response_text,
        "latency_s": record.latency_s,
        "passed": record.passed,
        "schema_passed": record.schema_passed,
        "behavior_passed": record.behavior_passed,
        "error": record.error,
        "validator": validator,
        "behavior": behavior,
        "raw_response": record.raw_response,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(records: Sequence[EvaluationRecord], model_name: str = None, test_suite: str = None) -> str:
    """Render evaluation results as a markdown report.

    Args:
        records: Evaluation records
        model_name: Optional model name for header
        test_suite: Optional test suite name for header
    """
    stats = aggregate_stats(records)

    # Build header
    header = "# Evaluator Run"
    if model_name:
        header = f"# Evaluation: {model_name}"
    if test_suite:
        header += f" ({test_suite})"
    elif stats['total']:
        header += f" ({stats['total']} prompts)"

    lines = [
        header,
        "",
        f"- **Passed:** {stats['passed']}/{stats['total']} ({stats['pass_rate']*100:.1f}%)",
        f"- **Failed:** {stats['failed']}",
        f"- **Request errors:** {stats['request_errors']}",
    ]

    # Add behavior stats if tested
    if stats['behavior_tested'] > 0:
        lines.append(f"- **Behavior tests:** {stats['behavior_passed']}/{stats['behavior_tested']} ({stats['behavior_pass_rate']*100:.1f}%)")

    lines.extend(["", "## Results by Category", ""])

    # Category table
    lines.append("| Category | Passed | Total | Pass Rate |")
    lines.append("|----------|--------|-------|-----------|")
    for tag, bucket in stats["by_tag"].items():
        lines.append(f"| {tag} | {bucket['passed']} | {bucket['total']} | {bucket['pass_rate']*100:.1f}% |")

    if stats["top_failure_reasons"]:
        lines.extend(["", "## Top Failure Reasons", ""])
        for reason, count in stats["top_failure_reasons"]:
            lines.append(f"- {count}× {reason}")

    if stats["top_behavior_failures"]:
        lines.extend(["", "## Top Behavior Failures", ""])
        for reason, count in stats["top_behavior_failures"]:
            lines.append(f"- {count}× {reason}")

    return "\n".join(lines)


def build_evaluation_lineage(
    records: Sequence[EvaluationRecord],
    model_name: str,
    test_suites: List[str],
    eval_config: Dict[str, Any],
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build comprehensive evaluation lineage for model cards.

    This creates a structured record of evaluation results that can be:
    - Embedded in model cards (README.md)
    - Saved as JSON for programmatic access
    - Uploaded to HuggingFace alongside models

    Args:
        records: Evaluation records from runner
        model_name: Name of the model being evaluated
        test_suites: List of test suite files used
        eval_config: Evaluation configuration (temperature, max_tokens, etc.)
        hardware_info: Optional hardware information (GPU, VRAM, etc.)

    Returns:
        Complete evaluation lineage dictionary
    """
    stats = aggregate_stats(records)
    timestamp = datetime.now(timezone.utc)

    # Calculate latency stats
    latencies = [r.latency_s for r in records if r.latency_s is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_time = sum(latencies) if latencies else 0

    # Build detailed results by category
    category_results = {}
    for tag, bucket in stats["by_tag"].items():
        category_results[tag] = {
            "passed": bucket["passed"],
            "total": bucket["total"],
            "pass_rate": round(bucket["pass_rate"] * 100, 1),
        }
        if bucket["behavior_tested"] > 0:
            category_results[tag]["behavior_tested"] = bucket["behavior_tested"]
            category_results[tag]["behavior_passed"] = bucket["behavior_passed"]
            category_results[tag]["behavior_pass_rate"] = round(bucket["behavior_pass_rate"] * 100, 1)

    # Build failure analysis
    failed_cases = []
    for record in records:
        if not record.passed:
            failure_info = {
                "case_id": record.case.case_id,
                "question": record.case.question[:200] + "..." if len(record.case.question) > 200 else record.case.question,
                "tags": record.case.tags,
            }
            if record.error:
                failure_info["error"] = record.error
            elif record.validator and record.validator.issues:
                failure_info["issues"] = [
                    {"level": i.level, "message": i.message}
                    for i in record.validator.issues[:3]  # Limit to 3 issues
                ]
            if record.behavior and not record.behavior.passed:
                failure_info["behavior_issues"] = [
                    {"check": i.check, "message": i.message}
                    for i in record.behavior.issues if not i.passed
                ][:3]
            failed_cases.append(failure_info)

    lineage = {
        # Identification
        "evaluation_timestamp": timestamp.isoformat(),
        "evaluation_date": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_evaluated": model_name,

        # Test configuration
        "test_config": {
            "test_suites": test_suites,
            "total_prompts": stats["total"],
            **eval_config,
        },

        # Overall results
        "results_summary": {
            "overall_pass_rate": round(stats["pass_rate"] * 100, 1),
            "passed": stats["passed"],
            "failed": stats["failed"],
            "request_errors": stats["request_errors"],
            "schema_pass_rate": round(stats["schema_pass_rate"] * 100, 1),
        },

        # Add behavior results if tested
        **({"behavior_results": {
            "tested": stats["behavior_tested"],
            "passed": stats["behavior_passed"],
            "pass_rate": round(stats["behavior_pass_rate"] * 100, 1),
        }} if stats["behavior_tested"] > 0 else {}),

        # Performance metrics
        "performance": {
            "avg_latency_s": round(avg_latency, 3),
            "total_time_s": round(total_time, 2),
            "tests_per_second": round(stats["total"] / total_time, 2) if total_time > 0 else 0,
        },

        # Detailed breakdown
        "results_by_category": category_results,

        # Failure analysis (limited to first 20)
        "failed_cases": failed_cases[:20],
        "top_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in stats["top_failure_reasons"]
        ],
    }

    # Add hardware info if provided
    if hardware_info:
        lineage["hardware"] = hardware_info

    return lineage


def generate_evaluation_model_card_section(lineage: Dict[str, Any]) -> str:
    """Generate a model card section from evaluation lineage.

    This creates markdown that can be appended to an existing model card
    or used standalone.

    Args:
        lineage: Evaluation lineage from build_evaluation_lineage()

    Returns:
        Markdown string for model card
    """
    results = lineage["results_summary"]
    perf = lineage["performance"]

    lines = [
        "## Evaluation Results",
        "",
        f"**Evaluation Date:** {lineage['evaluation_date']}",
        "",
        "### Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall Pass Rate | **{results['overall_pass_rate']}%** |",
        f"| Tests Passed | {results['passed']}/{results['passed'] + results['failed']} |",
        f"| Schema Validation | {results['schema_pass_rate']}% |",
    ]

    # Add behavior results if present
    if "behavior_results" in lineage:
        beh = lineage["behavior_results"]
        lines.append(f"| Behavior Validation | {beh['pass_rate']}% ({beh['passed']}/{beh['tested']}) |")

    lines.extend([
        f"| Avg Response Time | {perf['avg_latency_s']}s |",
        f"| Total Eval Time | {perf['total_time_s']}s |",
    ])

    # Results by category
    lines.extend(["", "### Results by Category", ""])
    lines.append("| Category | Pass Rate | Passed | Total |")
    lines.append("|----------|-----------|--------|-------|")

    for category, data in lineage["results_by_category"].items():
        lines.append(f"| {category} | {data['pass_rate']}% | {data['passed']} | {data['total']} |")

    # Test configuration
    config = lineage["test_config"]
    lines.extend(["", "### Test Configuration", ""])
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Test Suites | {', '.join(config['test_suites'])} |")
    lines.append(f"| Total Prompts | {config['total_prompts']} |")

    if "temperature" in config:
        lines.append(f"| Temperature | {config['temperature']} |")
    if "max_tokens" in config:
        lines.append(f"| Max Tokens | {config['max_tokens']} |")
    if "seed" in config:
        lines.append(f"| Seed | {config['seed']} |")

    # Top failures (if any)
    if lineage.get("top_failure_reasons"):
        lines.extend(["", "### Top Failure Reasons", ""])
        for item in lineage["top_failure_reasons"][:5]:
            lines.append(f"- {item['count']}× {item['reason']}")

    # Collapsible full lineage
    lines.extend([
        "",
        "<details>",
        "<summary>Full Evaluation Lineage (JSON)</summary>",
        "",
        "```json",
        json.dumps(lineage, indent=2),
        "```",
        "",
        "</details>",
    ])

    return "\n".join(lines)
