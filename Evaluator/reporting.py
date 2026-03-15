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
    errors = sum(1 for record in records if record.error)

    # Count by status: pass, warn, fail
    status_counts = Counter(record.status for record in records)
    passed = status_counts.get("pass", 0)
    warned = status_counts.get("warn", 0)
    failed = status_counts.get("fail", 0)

    # Track schema vs behavior pass rates separately
    schema_passed = sum(1 for record in records if record.schema_passed)
    behavior_tested = sum(1 for record in records if record.behavior is not None)
    behavior_passed = sum(1 for record in records if record.behavior_passed and record.behavior is not None)
    environment_tested = sum(1 for record in records if record.environment is not None)
    environment_passed = sum(1 for record in records if record.environment is not None and record.environment.passed)
    judge_tested = sum(1 for record in records if record.judge is not None)
    judge_passed = sum(1 for record in records if record.judge is not None and record.judge.passed)
    scoring_tested = sum(1 for record in records if record.scoring is not None)
    score_total = sum(record.scoring.awarded_score for record in records if record.scoring is not None)
    score_max_total = sum(record.scoring.max_score for record in records if record.scoring is not None)
    episode_recoveries = sum(
        1
        for record in records
        if record.environment
        and record.environment.episode_trace is not None
        and record.environment.episode_trace.recovered_after_error
    )

    by_tag = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
            "warned": 0,
            "failed": 0,
            "schema_passed": 0,
            "behavior_passed": 0,
            "behavior_tested": 0,
            "environment_passed": 0,
            "environment_tested": 0,
            "judge_passed": 0,
            "judge_tested": 0,
            "scoring_tested": 0,
            "score_total": 0.0,
            "score_max_total": 0.0,
        }
    )
    for record in records:
        tags = record.case.tags or ["__untagged__"]
        for tag in tags:
            bucket = by_tag[tag]
            bucket["total"] += 1
            status = record.status
            if status == "pass":
                bucket["passed"] += 1
            elif status == "warn":
                bucket["warned"] += 1
            else:
                bucket["failed"] += 1
            if record.schema_passed:
                bucket["schema_passed"] += 1
            if record.behavior is not None:
                bucket["behavior_tested"] += 1
                if record.behavior_passed:
                    bucket["behavior_passed"] += 1
            if record.environment is not None:
                bucket["environment_tested"] += 1
                if record.environment.passed:
                    bucket["environment_passed"] += 1
            if record.judge is not None:
                bucket["judge_tested"] += 1
                if record.judge.passed:
                    bucket["judge_passed"] += 1
            if record.scoring is not None:
                bucket["scoring_tested"] += 1
                bucket["score_total"] += record.scoring.awarded_score
                bucket["score_max_total"] += record.scoring.max_score

    failure_reasons = Counter()
    behavior_failures = Counter()
    environment_failures = Counter()
    for record in records:
        if record.error:
            # Truncate long error messages to avoid dumping prompts
            error_msg = record.error
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            failure_reasons[error_msg] += 1
        elif record.validator and not record.validator.passed:
            for issue in record.validator.issues:
                if issue.level.upper() == "ERROR":
                    failure_reasons[issue.message] += 1
        # Track behavior failures separately
        if record.behavior and not record.behavior.passed:
            for issue in record.behavior.issues:
                if not issue.passed:
                    behavior_failures[f"{issue.check}: {issue.message}"] += 1
        # Track environment failures separately
        if record.environment and not record.environment.passed:
            for issue in record.environment.issues:
                if issue.level.lower() == "error":
                    environment_failures[issue.message] += 1

    # Track judge failures
    judge_failures = Counter()
    for record in records:
        if record.judge and not record.judge.passed:
            for score in record.judge.judge_result.scores:
                if not score.passed:
                    judge_failures[f"{score.rubric_key}: {score.score:.2f} < {score.pass_threshold}"] += 1

    return {
        "total": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "request_errors": errors,
        "pass_rate": (passed / total) if total else 0,
        # Detailed breakdown
        "schema_passed": schema_passed,
        "schema_pass_rate": (schema_passed / total) if total else 0,
        "behavior_tested": behavior_tested,
        "behavior_passed": behavior_passed,
        "behavior_pass_rate": (behavior_passed / behavior_tested) if behavior_tested else 0,
        "environment_tested": environment_tested,
        "environment_passed": environment_passed,
        "environment_pass_rate": (environment_passed / environment_tested) if environment_tested else 0,
        "judge_tested": judge_tested,
        "judge_passed": judge_passed,
        "judge_pass_rate": (judge_passed / judge_tested) if judge_tested else 0,
        "scoring_tested": scoring_tested,
        "score_total": score_total,
        "score_max_total": score_max_total,
        "average_score": (score_total / scoring_tested) if scoring_tested else 0,
        "normalized_score": (score_total / score_max_total) if score_max_total else 0,
        "episode_recoveries": episode_recoveries,
        "by_tag": {
            tag: {
                "total": bucket["total"],
                "passed": bucket["passed"],
                "warned": bucket["warned"],
                "failed": bucket["failed"],
                "pass_rate": (bucket["passed"] / bucket["total"]) if bucket["total"] else 0,
                "schema_passed": bucket["schema_passed"],
                "behavior_tested": bucket["behavior_tested"],
                "behavior_passed": bucket["behavior_passed"],
                "behavior_pass_rate": (bucket["behavior_passed"] / bucket["behavior_tested"]) if bucket["behavior_tested"] else 0,
                "environment_tested": bucket["environment_tested"],
                "environment_passed": bucket["environment_passed"],
                "environment_pass_rate": (
                    (bucket["environment_passed"] / bucket["environment_tested"])
                    if bucket["environment_tested"]
                    else 0
                ),
                "judge_tested": bucket["judge_tested"],
                "judge_passed": bucket["judge_passed"],
                "judge_pass_rate": (
                    (bucket["judge_passed"] / bucket["judge_tested"])
                    if bucket["judge_tested"]
                    else 0
                ),
                "scoring_tested": bucket["scoring_tested"],
                "score_total": bucket["score_total"],
                "score_max_total": bucket["score_max_total"],
                "average_score": (
                    (bucket["score_total"] / bucket["scoring_tested"])
                    if bucket["scoring_tested"]
                    else 0
                ),
                "normalized_score": (
                    (bucket["score_total"] / bucket["score_max_total"])
                    if bucket["score_max_total"]
                    else 0
                ),
            }
            for tag, bucket in sorted(by_tag.items())
        },
        "top_failure_reasons": failure_reasons.most_common(10),
        "top_behavior_failures": behavior_failures.most_common(10),
        "top_environment_failures": environment_failures.most_common(10),
        "top_judge_failures": judge_failures.most_common(10),
    }


def console_summary(records: Sequence[EvaluationRecord]) -> str:
    stats = aggregate_stats(records)
    # Build summary line with pass/warn/fail
    summary_parts = [f"{stats['passed']} passed"]
    if stats['warned'] > 0:
        summary_parts.append(f"{stats['warned']} warned")
    summary_parts.append(f"{stats['failed']} failed")

    lines = [
        f"Evaluated {stats['total']} prompt(s): {', '.join(summary_parts)}.",
        f"  Schema validation: {stats['schema_passed']}/{stats['total']} ({stats['schema_pass_rate']*100:.1f}%)",
    ]
    if stats['behavior_tested'] > 0:
        lines.append(f"  Behavior validation: {stats['behavior_passed']}/{stats['behavior_tested']} ({stats['behavior_pass_rate']*100:.1f}%)")
    if stats['environment_tested'] > 0:
        lines.append(
            f"  Environment validation: {stats['environment_passed']}/{stats['environment_tested']} ({stats['environment_pass_rate']*100:.1f}%)"
        )
    if stats['judge_tested'] > 0:
        lines.append(
            f"  Judge validation: {stats['judge_passed']}/{stats['judge_tested']} ({stats['judge_pass_rate']*100:.1f}%)"
        )
    if stats['scoring_tested'] > 0:
        lines.append(
            f"  Path scoring: avg {stats['average_score']:.2f}, normalized {stats['normalized_score']*100:.1f}%"
        )
    if stats["episode_recoveries"] > 0:
        lines.append(f"  Recovered episodes: {stats['episode_recoveries']}")
    lines.append(f"Request errors: {stats['request_errors']}")
    lines.append("Results by tag:")
    for tag, bucket in stats["by_tag"].items():
        # Show pass/warn/fail counts
        parts = [f"{bucket['passed']}P"]
        if bucket["warned"] > 0:
            parts.append(f"{bucket['warned']}W")
        parts.append(f"{bucket['failed']}F")
        line = f"  - {tag}: {'/'.join(parts)} ({bucket['total']} total)"
        if bucket["behavior_tested"] > 0:
            beh_pct = bucket["behavior_pass_rate"] * 100
            line += f" [behavior: {bucket['behavior_passed']}/{bucket['behavior_tested']} ({beh_pct:.1f}%)]"
        if bucket["environment_tested"] > 0:
            env_pct = bucket["environment_pass_rate"] * 100
            line += f" [environment: {bucket['environment_passed']}/{bucket['environment_tested']} ({env_pct:.1f}%)]"
        if bucket["judge_tested"] > 0:
            judge_pct = bucket["judge_pass_rate"] * 100
            line += f" [judge: {bucket['judge_passed']}/{bucket['judge_tested']} ({judge_pct:.1f}%)]"
        if bucket["scoring_tested"] > 0:
            score_pct = bucket["normalized_score"] * 100
            line += f" [score: {bucket['average_score']:.2f}, {score_pct:.1f}%]"
        lines.append(line)
    if stats["top_failure_reasons"]:
        lines.append("Top schema failure reasons:")
        for reason, count in stats["top_failure_reasons"]:
            lines.append(f"  - {count}× {reason}")
    if stats["top_behavior_failures"]:
        lines.append("Top behavior failure reasons:")
        for reason, count in stats["top_behavior_failures"]:
            lines.append(f"  - {count}× {reason}")
    if stats["top_environment_failures"]:
        lines.append("Top environment failure reasons:")
        for reason, count in stats["top_environment_failures"]:
            lines.append(f"  - {count}× {reason}")
    if stats["top_judge_failures"]:
        lines.append("Top judge failure reasons:")
        for reason, count in stats["top_judge_failures"]:
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
    environment = record.environment.to_dict() if record.environment else None
    judge = record.judge.to_dict() if record.judge else None
    return {
        "case_id": record.case.case_id,
        "question": record.case.question,
        "tags": record.case.tags,
        "expected_tools": record.case.expected_tools,
        "acceptable_tools": record.case.acceptable_tools,
        "response_text": record.response_text,
        "latency_s": record.latency_s,
        "passed": record.passed,
        "schema_passed": record.schema_passed,
        "behavior_passed": record.behavior_passed,
        "environment_passed": record.environment.passed if record.environment else None,
        "judge_passed": record.judge.passed if record.judge else None,
        "score": record.score,
        "error": record.error,
        "validator": validator,
        "behavior": behavior,
        "environment": environment,
        "judge": judge,
        "scoring": record.scoring.to_dict() if record.scoring else None,
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
    if stats['environment_tested'] > 0:
        lines.append(f"- **Environment tests:** {stats['environment_passed']}/{stats['environment_tested']} ({stats['environment_pass_rate']*100:.1f}%)")
    if stats['judge_tested'] > 0:
        lines.append(f"- **Judge tests:** {stats['judge_passed']}/{stats['judge_tested']} ({stats['judge_pass_rate']*100:.1f}%)")

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
    if stats["top_environment_failures"]:
        lines.extend(["", "## Top Environment Failures", ""])
        for reason, count in stats["top_environment_failures"]:
            lines.append(f"- {count}× {reason}")
    if stats["top_judge_failures"]:
        lines.extend(["", "## Top Judge Failures", ""])
        for reason, count in stats["top_judge_failures"]:
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
        **({"environment_results": {
            "tested": stats["environment_tested"],
            "passed": stats["environment_passed"],
            "pass_rate": round(stats["environment_pass_rate"] * 100, 1),
        }} if stats["environment_tested"] > 0 else {}),

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
