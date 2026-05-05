#!/usr/bin/env python3
"""
Generic structural validator for SynthChat-style JSONL datasets.

This helper intentionally does not know about any specific tool wrapper,
context field set, command syntax, or project schema. Use scenario rubrics,
environment execution config, and Evaluator assertions for format-specific
validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class Issue:
    level: str
    message: str


@dataclass
class RowReport:
    line_number: int
    issues: list[Issue] = field(default_factory=list)

    def add(self, level: str, message: str) -> None:
        self.issues.append(Issue(level, message))

    @property
    def has_errors(self) -> bool:
        return any(issue.level == "ERROR" for issue in self.issues)


def _message_list(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = row.get("messages")
    if messages is None:
        messages = row.get("conversations")
    if isinstance(messages, list):
        return messages
    return None


def validate_row(line_number: int, row: Any) -> RowReport:
    report = RowReport(line_number=line_number)
    if not isinstance(row, dict):
        report.add("ERROR", "Row is not a JSON object")
        return report

    messages = _message_list(row)
    if messages is None:
        report.add("ERROR", "Row must contain a messages or conversations list")
        return report
    if not messages:
        report.add("ERROR", "Message list is empty")
        return report

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            report.add("ERROR", f"Message #{index + 1} is not an object")
            continue

        role = message.get("role")
        if role not in ALLOWED_ROLES:
            report.add("ERROR", f"Message #{index + 1} has invalid role {role!r}")

        has_content = "content" in message
        has_tool_calls = "tool_calls" in message
        if not has_content and not has_tool_calls:
            report.add("ERROR", f"Message #{index + 1} has neither content nor tool_calls")

        if has_content and message.get("content") is not None and not isinstance(message.get("content"), str):
            report.add("ERROR", f"Message #{index + 1} content must be a string or null")

        if has_tool_calls and not isinstance(message.get("tool_calls"), list):
            report.add("ERROR", f"Message #{index + 1} tool_calls must be a list")

    if "label" in row and not isinstance(row["label"], bool):
        report.add("WARN", "label is present but is not boolean")

    return report


def iter_jsonl(path: Path) -> list[RowReport]:
    reports: list[RowReport] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report = RowReport(line_number=line_number)
                report.add("ERROR", f"Invalid JSON: {exc}")
                reports.append(report)
                continue
            reports.append(validate_row(line_number, row))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="JSONL dataset to validate")
    parser.add_argument("--max-issues", type=int, default=50, help="Maximum issues to print")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    reports = iter_jsonl(path)
    issue_count = 0
    error_count = 0
    warning_count = 0

    for report in reports:
        for issue in report.issues:
            issue_count += 1
            if issue.level == "ERROR":
                error_count += 1
            else:
                warning_count += 1
            if issue_count <= args.max_issues:
                print(f"line {report.line_number}: {issue.level}: {issue.message}")

    if issue_count > args.max_issues:
        print(f"... {issue_count - args.max_issues} more issues omitted")

    print(f"Rows checked: {len(reports)}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
