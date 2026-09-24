"""Compare Evaluator result files and gate on regressions.

Location: ``Evaluator/compare.py``.

Reads two or more results JSON files written by ``python -m Evaluator.cli``
(``Evaluator/reporting.py`` ``build_run_payload``): one reference and one or
more candidates. The typical use is measuring what quantization cost: the
reference is the full-precision model, each candidate one quantized artifact
(Q8_0, Q5_K_M, Q4_K_M, ...) evaluated on the same scenarios and settings.

Per candidate it reports pass-rate and correctness deltas (percentage points),
per-tag deltas, per-case flips joined on case id, request errors, latency ratio
and an exact McNemar p-value over the discordant cases so a one-case swing on a
small suite is not read as a real regression. It knows nothing about tool
formats; it only reads the results JSON schema.

Usage::

    python -m Evaluator.compare \\
      --reference Evaluator/results/f16.json \\
      --candidate Q4_K_M=Evaluator/results/q4.json \\
      --candidate Evaluator/results/model-Q8_0.json \\
      --max-pass-rate-drop 3 --max-regressions 2 \\
      --output Evaluator/results/quant_compare.json \\
      --markdown Evaluator/results/quant_compare.md

Exit codes: 0 all candidates pass the gate (or no thresholds were given, which
is report-only), 1 a gate was breached, 2 input error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import expand_path
from .model_artifact import detect_quantization, is_full_precision
from .reporting import write_json


COMPARISON_SCHEMA_VERSION = "synaptic-evaluation-comparison/v1"

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_INPUT_ERROR = 2

# Results metadata keys that must match for a like-for-like comparison. A key
# is only compared when both files record it (older results lack some keys).
COMPARABLE_SETTINGS = (
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "scenarios",
    "preset",
    "tags_filter",
    "limit",
    "dry_run",
    "environment",
)

# Float tolerance for threshold comparisons on percentage points.
_EPSILON = 1e-9


class CompareInputError(Exception):
    """A results file or argument cannot be used for comparison."""


@dataclass
class ResultsFile:
    label: str
    path: Path
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    cases: Dict[str, Dict[str, Any]]
    duplicate_case_ids: List[str] = field(default_factory=list)


@dataclass
class GateThresholds:
    max_pass_rate_drop: Optional[float] = None
    max_correctness_drop: Optional[float] = None
    max_tag_drop: Optional[float] = None
    max_regressions: Optional[int] = None
    min_tag_cases: int = 1
    alpha: Optional[float] = None
    allow_mismatch: bool = False

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_pass_rate_drop,
                self.max_correctness_drop,
                self.max_tag_drop,
                self.max_regressions,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_pass_rate_drop_pp": self.max_pass_rate_drop,
            "max_correctness_drop_pp": self.max_correctness_drop,
            "max_tag_drop_pp": self.max_tag_drop,
            "max_regressions": self.max_regressions,
            "min_tag_cases": self.min_tag_cases,
            "alpha": self.alpha,
            "allow_mismatch": self.allow_mismatch,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def derive_label(path: Path, metadata: Mapping[str, Any]) -> str:
    """Label a results file by its recorded quantization, model name or filename."""
    artifact = metadata.get("model_artifact") or {}
    return (
        artifact.get("quantization")
        or detect_quantization(metadata.get("model"))
        or detect_quantization(path.stem)
        or path.stem
    )


def load_results(path: Path, label: Optional[str] = None) -> ResultsFile:
    """Load an Evaluator results JSON and key its records by case id.

    A repeated case id (the same id in two scenario files) is keyed as
    ``<id>#2``, ``<id>#3``, ... in file order; records without an id are keyed
    ``#<index>``.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CompareInputError(f"results file not found: {path}")
    except (OSError, ValueError) as exc:
        raise CompareInputError(f"cannot read results file {path}: {exc}")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise CompareInputError(f"{path} is not an Evaluator results file (no 'records' list)")

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    cases: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    seen: Dict[str, int] = {}
    for index, record in enumerate(payload["records"]):
        if not isinstance(record, dict):
            raise CompareInputError(f"{path}: record {index} is not an object")
        case_id = record.get("case_id")
        key = str(case_id) if case_id not in (None, "") else f"#{index}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            duplicates.append(key)
            key = f"{key}#{seen[key]}"
        cases[key] = record
    return ResultsFile(
        label=label or derive_label(Path(path), metadata),
        path=Path(path),
        metadata=metadata,
        summary=summary,
        cases=cases,
        duplicate_case_ids=sorted(set(duplicates)),
    )


def parse_candidate_arg(value: str) -> Tuple[Optional[str], str]:
    """Split ``LABEL=PATH``; a bare path (or one whose prefix has a separator) has no label."""
    label, sep, path = value.partition("=")
    if sep and label and path and not any(ch in label for ch in "/\\"):
        return label, path
    return None, value


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact_p(regressions: int, improvements: int) -> float:
    """Exact two-sided McNemar p-value over the discordant pairs.

    Under the null the discordant cases split 50/50, so the smaller count is
    Binomial(n, 0.5): p = min(1, 2 * P(X <= min(b, c))). No discordant cases
    gives p = 1.0.
    """
    b, c = int(regressions), int(improvements)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _rate(passed: int, total: int) -> Optional[float]:
    return (passed / total) if total else None


def _delta_pp(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    return (candidate - reference) * 100.0


def _case_tags(record: Mapping[str, Any]) -> List[str]:
    return list(record.get("tags") or ["__untagged__"])


def side_stats(results: ResultsFile, case_ids: Sequence[str]) -> Dict[str, Any]:
    """Pass/correctness/error/latency stats over ``case_ids`` for one file.

    ``passed`` is the record's overall pass flag (pass or warn), the same flag
    the per-case flips use.
    """
    records = [results.cases[case_id] for case_id in case_ids]
    total = len(records)
    passed = sum(1 for record in records if record.get("passed"))
    correctness = [record.get("correctness_passed") for record in records]
    correctness_tested = sum(1 for value in correctness if value is not None)
    correctness_passed = sum(1 for value in correctness if value)
    latencies = [
        float(record["latency_s"])
        for record in records
        if isinstance(record.get("latency_s"), (int, float))
    ]
    by_tag: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for tag in _case_tags(record):
            bucket = by_tag.setdefault(tag, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if record.get("passed"):
                bucket["passed"] += 1
    for bucket in by_tag.values():
        bucket["pass_rate"] = _rate(bucket["passed"], bucket["total"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": _rate(passed, total),
        "correctness_tested": correctness_tested,
        "correctness_passed": correctness_passed,
        "correctness_pass_rate": _rate(correctness_passed, correctness_tested),
        "request_errors": sum(1 for record in records if record.get("error")),
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "by_tag": dict(sorted(by_tag.items())),
    }


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------

def _normalized_setting(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def settings_mismatches(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> List[str]:
    """Differences in eval settings recorded by both results files."""
    mismatches = []
    for key in COMPARABLE_SETTINGS:
        if key not in reference or key not in candidate:
            continue
        ref_value, cand_value = reference[key], candidate[key]
        if _normalized_setting(ref_value) != _normalized_setting(cand_value):
            mismatches.append(f"setting '{key}' differs: reference={ref_value!r} candidate={cand_value!r}")
    return mismatches


def _status_mismatches(results: ResultsFile, role: str) -> List[str]:
    problems = []
    status = results.metadata.get("artifact_status")
    if status in ("partial", "failed"):
        problems.append(f"{role} results are {status} (artifact_status={status})")
    if results.metadata.get("dry_run"):
        problems.append(f"{role} results come from a dry run")
    return problems


def reference_precision_warnings(reference: ResultsFile) -> List[str]:
    """Warn when the reference itself was quantized, which understates the cost."""
    metadata = reference.metadata
    artifact = metadata.get("model_artifact")
    warnings = []
    if isinstance(artifact, Mapping):
        load_settings = artifact.get("load_settings") or {}
        if load_settings.get("load_in_4bit"):
            warnings.append(
                "reference was loaded in 4-bit (load_settings.load_in_4bit=true); "
                "the measured quantization cost is understated"
            )
        quantization = artifact.get("quantization")
        if quantization and not is_full_precision(quantization):
            warnings.append(f"reference artifact is quantized ({quantization})")
        return warnings

    quantization = detect_quantization(metadata.get("model"))
    if quantization and not is_full_precision(quantization):
        warnings.append(f"reference model name looks quantized ({quantization})")
    if metadata.get("backend") == "unsloth":
        warnings.append(
            "reference used the unsloth backend, which loads 4-bit by default, "
            "and its load precision was not recorded"
        )
    return warnings


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _flip_counts(
    reference: ResultsFile,
    candidate: ResultsFile,
    case_ids: Sequence[str],
) -> Tuple[List[str], List[str]]:
    regressions, improvements = [], []
    for case_id in case_ids:
        ref_passed = bool(reference.cases[case_id].get("passed"))
        cand_passed = bool(candidate.cases[case_id].get("passed"))
        if ref_passed and not cand_passed:
            regressions.append(case_id)
        elif cand_passed and not ref_passed:
            improvements.append(case_id)
    return regressions, improvements


def _significant(p_value: float, alpha: Optional[float]) -> bool:
    return alpha is None or p_value < alpha


def evaluate_gate(comparison: Dict[str, Any], thresholds: GateThresholds) -> Dict[str, Any]:
    """Apply thresholds to one candidate comparison."""
    breaches: List[str] = []
    if not thresholds.active:
        return {"active": False, "passed": True, "breaches": breaches}

    if not comparison["comparable"] and not thresholds.allow_mismatch:
        breaches.append("not comparable: " + "; ".join(comparison["mismatches"]))

    flips = comparison["flips"]
    p_value = flips["mcnemar_p"]
    deltas = comparison["deltas"]

    def check_drop(name: str, delta: Optional[float], limit: Optional[float], p: float) -> None:
        if limit is None or delta is None:
            return
        drop = -delta
        if drop > limit + _EPSILON and _significant(p, thresholds.alpha):
            suffix = f", McNemar p={p:.3g}" if thresholds.alpha is not None else ""
            breaches.append(f"{name} dropped {drop:.1f}pp (limit {limit:g}pp{suffix})")

    check_drop("pass rate", deltas["pass_rate_pp"], thresholds.max_pass_rate_drop, p_value)
    check_drop(
        "correctness pass rate",
        deltas["correctness_pass_rate_pp"],
        thresholds.max_correctness_drop,
        p_value,
    )
    if thresholds.max_tag_drop is not None:
        for tag, bucket in comparison["by_tag"].items():
            if min(bucket["reference_total"], bucket["candidate_total"]) < thresholds.min_tag_cases:
                continue
            check_drop(f"tag '{tag}'", bucket["delta_pp"], thresholds.max_tag_drop, bucket["mcnemar_p"])
    if thresholds.max_regressions is not None and flips["regression_count"] > thresholds.max_regressions:
        breaches.append(
            f"{flips['regression_count']} regressions (limit {thresholds.max_regressions})"
        )
    return {"active": True, "passed": not breaches, "breaches": breaches}


def compare_candidate(
    reference: ResultsFile,
    candidate: ResultsFile,
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    """Compare one candidate against the reference over their shared case ids."""
    shared = [case_id for case_id in reference.cases if case_id in candidate.cases]
    only_reference = [case_id for case_id in reference.cases if case_id not in candidate.cases]
    only_candidate = [case_id for case_id in candidate.cases if case_id not in reference.cases]

    mismatches: List[str] = []
    if only_reference or only_candidate:
        mismatches.append(
            f"case ids differ: {len(only_reference)} only in reference, "
            f"{len(only_candidate)} only in candidate"
        )
    if not shared:
        mismatches.append("no shared case ids")
    mismatches.extend(settings_mismatches(reference.metadata, candidate.metadata))
    mismatches.extend(_status_mismatches(reference, "reference"))
    mismatches.extend(_status_mismatches(candidate, "candidate"))

    warnings: List[str] = []
    for results, role in ((reference, "reference"), (candidate, "candidate")):
        if results.duplicate_case_ids:
            warnings.append(
                f"{role} repeats case ids {results.duplicate_case_ids}; repeats are joined by occurrence order"
            )
    artifact = candidate.metadata.get("model_artifact")
    if isinstance(artifact, Mapping) and artifact.get("warnings"):
        warnings.extend(f"candidate artifact: {warning}" for warning in artifact["warnings"])

    ref_stats = side_stats(reference, shared)
    cand_stats = side_stats(candidate, shared)
    regressions, improvements = _flip_counts(reference, candidate, shared)

    by_tag: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(set(ref_stats["by_tag"]) | set(cand_stats["by_tag"])):
        ref_bucket = ref_stats["by_tag"].get(tag, {"total": 0, "passed": 0, "pass_rate": None})
        cand_bucket = cand_stats["by_tag"].get(tag, {"total": 0, "passed": 0, "pass_rate": None})
        tag_regressions = [c for c in regressions if tag in _case_tags(reference.cases[c])]
        tag_improvements = [c for c in improvements if tag in _case_tags(reference.cases[c])]
        by_tag[tag] = {
            "reference_total": ref_bucket["total"],
            "reference_passed": ref_bucket["passed"],
            "reference_pass_rate": ref_bucket["pass_rate"],
            "candidate_total": cand_bucket["total"],
            "candidate_passed": cand_bucket["passed"],
            "candidate_pass_rate": cand_bucket["pass_rate"],
            "delta_pp": _delta_pp(ref_bucket["pass_rate"], cand_bucket["pass_rate"]),
            "regressions": len(tag_regressions),
            "improvements": len(tag_improvements),
            "mcnemar_p": mcnemar_exact_p(len(tag_regressions), len(tag_improvements)),
        }

    ref_latency, cand_latency = ref_stats["mean_latency_s"], cand_stats["mean_latency_s"]
    comparison: Dict[str, Any] = {
        "label": candidate.label,
        "path": str(candidate.path),
        "model": candidate.metadata.get("model"),
        "backend": candidate.metadata.get("backend"),
        "model_artifact": candidate.metadata.get("model_artifact"),
        "comparable": not mismatches,
        "mismatches": mismatches,
        "warnings": warnings,
        "case_counts": {
            "shared": len(shared),
            "only_reference": len(only_reference),
            "only_candidate": len(only_candidate),
        },
        "only_reference_case_ids": only_reference,
        "only_candidate_case_ids": only_candidate,
        "reference": {key: value for key, value in ref_stats.items() if key != "by_tag"},
        "candidate": {key: value for key, value in cand_stats.items() if key != "by_tag"},
        "deltas": {
            "pass_rate_pp": _delta_pp(ref_stats["pass_rate"], cand_stats["pass_rate"]),
            "correctness_pass_rate_pp": _delta_pp(
                ref_stats["correctness_pass_rate"], cand_stats["correctness_pass_rate"]
            ),
        },
        "flips": {
            "regressions": regressions,
            "improvements": improvements,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "discordant": len(regressions) + len(improvements),
            "mcnemar_p": mcnemar_exact_p(len(regressions), len(improvements)),
        },
        "request_errors": {
            "reference": ref_stats["request_errors"],
            "candidate": cand_stats["request_errors"],
        },
        "latency": {
            "reference_mean_s": ref_latency,
            "candidate_mean_s": cand_latency,
            "ratio": (cand_latency / ref_latency) if ref_latency and cand_latency is not None else None,
        },
        "by_tag": by_tag,
    }
    comparison["gate"] = evaluate_gate(comparison, thresholds)
    return comparison


def build_comparison(
    reference: ResultsFile,
    candidates: Sequence[ResultsFile],
    thresholds: GateThresholds,
) -> Dict[str, Any]:
    """Build the full comparison report (the ``--output`` JSON document)."""
    comparisons = [compare_candidate(reference, candidate, thresholds) for candidate in candidates]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "label": reference.label,
            "path": str(reference.path),
            "model": reference.metadata.get("model"),
            "backend": reference.metadata.get("backend"),
            "model_artifact": reference.metadata.get("model_artifact"),
            "stats": side_stats(reference, list(reference.cases)),
        },
        "warnings": reference_precision_warnings(reference),
        "thresholds": thresholds.to_dict(),
        "candidates": comparisons,
        "passed": all(item["gate"]["passed"] for item in comparisons),
    }


def exit_code_for(report: Mapping[str, Any]) -> int:
    return EXIT_OK if report["passed"] else EXIT_GATE_FAILED


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(rate: Optional[float]) -> str:
    return "-" if rate is None else f"{rate * 100:.1f}%"


def _pp(delta: Optional[float]) -> str:
    return "-" if delta is None else f"{delta:+.1f}pp"


def _pvalue(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3g}"


def _ratio(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _gate_label(gate: Mapping[str, Any]) -> str:
    if not gate["active"]:
        return "report"
    return "PASS" if gate["passed"] else "FAIL"


def _summary_rows(report: Mapping[str, Any]) -> List[List[str]]:
    reference = report["reference"]["stats"]
    rows = [[
        f"{report['reference']['label']} (ref)",
        str(reference["total"]),
        _pct(reference["pass_rate"]),
        "",
        _pct(reference["correctness_pass_rate"]),
        "",
        "",
        "",
        "",
        str(reference["request_errors"]),
        "",
        "",
    ]]
    for item in report["candidates"]:
        flips = item["flips"]
        rows.append([
            item["label"] + ("" if item["comparable"] else " (!)"),
            str(item["case_counts"]["shared"]),
            _pct(item["candidate"]["pass_rate"]),
            _pp(item["deltas"]["pass_rate_pp"]),
            _pct(item["candidate"]["correctness_pass_rate"]),
            _pp(item["deltas"]["correctness_pass_rate_pp"]),
            str(flips["regression_count"]),
            str(flips["improvement_count"]),
            _pvalue(flips["mcnemar_p"]),
            str(item["request_errors"]["candidate"]),
            _ratio(item["latency"]["ratio"]),
            _gate_label(item["gate"]),
        ])
    return rows


_SUMMARY_HEADERS = [
    "Label", "Cases", "Pass", "dPass", "Correct", "dCorrect",
    "Reg", "Imp", "McNemar p", "Errors", "Latency", "Gate",
]


def render_console(report: Mapping[str, Any]) -> str:
    """Compact plain-text table plus warnings and gate breaches."""
    rows = [_SUMMARY_HEADERS] + _summary_rows(report)
    widths = [max(len(row[col]) for row in rows) for col in range(len(_SUMMARY_HEADERS))]
    lines = [f"Reference: {report['reference']['path']}"]
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    for item in report["candidates"]:
        for mismatch in item["mismatches"]:
            lines.append(f"MISMATCH [{item['label']}]: {mismatch}")
        for warning in item["warnings"]:
            lines.append(f"WARNING [{item['label']}]: {warning}")
        for breach in item["gate"]["breaches"]:
            lines.append(f"GATE FAIL [{item['label']}]: {breach}")
    if not report["thresholds"]["active"]:
        lines.append("No thresholds given: report only.")
    else:
        lines.append(f"Gate: {'PASS' if report['passed'] else 'FAIL'}")
    return "\n".join(lines)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return lines


def _artifact_description(artifact: Any) -> str:
    if not isinstance(artifact, Mapping):
        return "not recorded"
    parts = [artifact.get("quantization") or "none detected"]
    if artifact.get("quantization_source"):
        parts.append(f"source: {artifact['quantization_source']}")
    load_settings = artifact.get("load_settings") or {}
    if "load_in_4bit" in load_settings:
        parts.append(f"load_in_4bit: {load_settings['load_in_4bit']}")
    return parts[0] + (f" ({', '.join(parts[1:])})" if len(parts) > 1 else "")


def render_markdown(report: Mapping[str, Any], *, max_listed_cases: int = 50) -> str:
    """Markdown comparison report."""
    reference = report["reference"]
    lines = [
        "# Evaluation Comparison",
        "",
        f"- **Reference:** `{reference['path']}` ({reference['label']})",
        f"- **Model:** {reference.get('model') or '-'} via {reference.get('backend') or '-'}",
        f"- **Reference quantization:** {_artifact_description(reference.get('model_artifact'))}",
        f"- **Gate:** {'report only' if not report['thresholds']['active'] else ('PASS' if report['passed'] else 'FAIL')}",
    ]
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    lines.extend(["", "## Summary", ""])
    lines.extend(_md_table(_SUMMARY_HEADERS, _summary_rows(report)))
    lines.extend([
        "",
        "Deltas are candidate minus reference in percentage points over shared case ids. "
        "McNemar p is the exact two-sided test over discordant cases (Reg + Imp); "
        "a large p means the difference is consistent with noise.",
    ])

    thresholds = report["thresholds"]
    if thresholds["active"]:
        lines.extend(["", "## Thresholds", ""])
        for key, value in thresholds.items():
            if key != "active" and value is not None:
                lines.append(f"- `{key}`: {value}")

    for item in report["candidates"]:
        lines.extend(["", f"## {item['label']}", "", f"- **Results:** `{item['path']}`"])
        lines.append(f"- **Quantization:** {_artifact_description(item.get('model_artifact'))}")
        lines.append(f"- **Comparable:** {'yes' if item['comparable'] else 'no'}")
        lines.extend(f"- Mismatch: {mismatch}" for mismatch in item["mismatches"])
        lines.extend(f"- Warning: {warning}" for warning in item["warnings"])
        lines.extend(f"- **Gate breach:** {breach}" for breach in item["gate"]["breaches"])
        if item["by_tag"]:
            lines.extend(["", "### By tag", ""])
            tag_rows = [
                [
                    tag,
                    f"{bucket['reference_passed']}/{bucket['reference_total']}",
                    f"{bucket['candidate_passed']}/{bucket['candidate_total']}",
                    _pp(bucket["delta_pp"]),
                    str(bucket["regressions"]),
                    str(bucket["improvements"]),
                    _pvalue(bucket["mcnemar_p"]),
                ]
                for tag, bucket in item["by_tag"].items()
            ]
            lines.extend(_md_table(["Tag", "Reference", "Candidate", "Delta", "Reg", "Imp", "McNemar p"], tag_rows))
        for title, case_ids in (
            ("Regressions (pass to fail)", item["flips"]["regressions"]),
            ("Improvements (fail to pass)", item["flips"]["improvements"]),
        ):
            if not case_ids:
                continue
            lines.extend(["", f"### {title}", ""])
            lines.extend(f"- `{case_id}`" for case_id in case_ids[:max_listed_cases])
            if len(case_ids) > max_listed_cases:
                lines.append(f"- ... {len(case_ids) - max_listed_cases} more")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m Evaluator.compare",
        description=(
            "Compare Evaluator results files (e.g. quantized GGUF candidates against the "
            "full-precision reference) and optionally gate on regressions."
        ),
        epilog="Exit codes: 0 pass or report-only, 1 gate breached, 2 input error.",
    )
    parser.add_argument("--reference", required=True, help="Reference results JSON (full-precision model)")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help="Candidate results JSON; repeatable. Label defaults to the recorded quantization, "
        "a quant name in the model/file name, or the file stem.",
    )
    parser.add_argument("--max-pass-rate-drop", type=float, metavar="PP", help="Max overall pass-rate drop (percentage points)")
    parser.add_argument("--max-correctness-drop", type=float, metavar="PP", help="Max correctness pass-rate drop (percentage points)")
    parser.add_argument("--max-tag-drop", type=float, metavar="PP", help="Max per-tag pass-rate drop (percentage points)")
    parser.add_argument("--max-regressions", type=int, metavar="N", help="Max cases that flip pass to fail")
    parser.add_argument(
        "--min-tag-cases",
        type=int,
        default=1,
        metavar="N",
        help="Only apply --max-tag-drop to tags with at least N shared cases (default: 1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Only count pass-rate/tag drops as breaches when the McNemar p-value is below ALPHA "
        "(default: absolute thresholds, no significance requirement)",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Do not fail the gate when case ids or eval settings differ (still reported)",
    )
    parser.add_argument("--output", help="Write the comparison JSON here")
    parser.add_argument("--markdown", help="Write the comparison Markdown report here")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_INPUT_ERROR

    thresholds = GateThresholds(
        max_pass_rate_drop=args.max_pass_rate_drop,
        max_correctness_drop=args.max_correctness_drop,
        max_tag_drop=args.max_tag_drop,
        max_regressions=args.max_regressions,
        min_tag_cases=max(1, args.min_tag_cases),
        alpha=args.alpha,
        allow_mismatch=args.allow_mismatch,
    )
    try:
        reference = load_results(expand_path(args.reference))
        candidates = []
        for value in args.candidate:
            label, path = parse_candidate_arg(value)
            candidates.append(load_results(expand_path(path), label=label))
        labels = [candidate.label for candidate in candidates]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise CompareInputError(
                f"duplicate candidate labels {repeated}; pass explicit LABEL=PATH values"
            )
        output_path = expand_path(args.output) if args.output else None
        markdown_path = expand_path(args.markdown) if args.markdown else None
    except CompareInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    report = build_comparison(reference, candidates, thresholds)
    print(render_console(report))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, report)
        print(f"Comparison JSON saved to {output_path}")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Comparison Markdown saved to {markdown_path}")
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
