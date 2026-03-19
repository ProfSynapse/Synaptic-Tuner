"""
Shared progress event helpers for local and cloud-backed evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


EVAL_PROGRESS_LOG_FILENAME = "eval_progress.jsonl"


def _truncate_message(message: Optional[str], *, limit: int = 50) -> Optional[str]:
    if not message:
        return None
    if len(message) <= limit:
        return message
    return message[:limit] + "..."


def extract_record_progress(
    record: Any,
    *,
    issue_formatter: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Convert an evaluation record into a dashboard-friendly progress payload."""

    def format_issue(message: str) -> Optional[str]:
        if issue_formatter is None:
            return message
        return issue_formatter(message)

    reason = None
    if getattr(record, "status", None) in ("fail", "warn"):
        error = getattr(record, "error", None)
        if error:
            reason = _truncate_message(f"Error: {error}", limit=40)
        else:
            validator = getattr(record, "validator", None)
            environment = getattr(record, "environment", None)
            behavior = getattr(record, "behavior", None)

            issue_collections = []
            if validator and getattr(validator, "issues", None):
                issue_collections.append(validator.issues)
            if environment and getattr(environment, "issues", None):
                issue_collections.append(environment.issues)
            if behavior and not getattr(behavior, "passed", True) and getattr(behavior, "issues", None):
                issue_collections.append(behavior.issues)

            for issues in issue_collections:
                for issue in issues:
                    message = getattr(issue, "message", None)
                    if not message:
                        continue
                    simplified = format_issue(message)
                    if simplified:
                        reason = _truncate_message(simplified)
                        break
                if reason:
                    break

    behavior = getattr(record, "behavior", None)
    behavior_tested = behavior is not None
    behavior_passed = bool(behavior_tested and getattr(behavior, "passed", False))

    case = getattr(record, "case", None)
    case_id = getattr(case, "case_id", None) if case is not None else None

    return {
        "event": "result",
        "status": getattr(record, "status", "fail"),
        "name": case_id or "unnamed",
        "latency": float(getattr(record, "latency_s", 0.0) or 0.0),
        "reason": reason,
        "behavior_tested": behavior_tested,
        "behavior_passed": behavior_passed,
    }


def append_progress_event(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSONL progress event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class CloudEvaluationProgressWriter:
    """Write structured progress events and optionally trigger incremental sync."""

    def __init__(
        self,
        progress_log_path: Path,
        *,
        sync_callback: Optional[Callable[[Path], None]] = None,
        sync_every_events: int = 5,
    ) -> None:
        self.progress_log_path = Path(progress_log_path)
        self.sync_callback = sync_callback
        self.sync_every_events = max(1, int(sync_every_events))
        self._event_count = 0

    @property
    def progress_dir(self) -> Path:
        return self.progress_log_path.parent

    def write_metadata(
        self,
        *,
        total_tests: int,
        title: str = "Cloud Evaluation",
        backend: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        append_progress_event(
            self.progress_log_path,
            {
                "event": "meta",
                "title": title,
                "total_tests": int(total_tests),
                "backend": backend,
                "model": model,
            },
        )
        self.sync(force=True)

    def write_record(
        self,
        record: Any,
        *,
        issue_formatter: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        append_progress_event(
            self.progress_log_path,
            extract_record_progress(record, issue_formatter=issue_formatter),
        )
        self._event_count += 1
        if self._event_count % self.sync_every_events == 0:
            self.sync(force=True)

    def write_complete(self) -> None:
        append_progress_event(self.progress_log_path, {"event": "complete"})
        self.sync(force=True)

    def sync(self, *, force: bool = False) -> None:
        if not force or self.sync_callback is None:
            return
        self.sync_callback(self.progress_dir)


class EvaluationDashboardReplayer:
    """Replay structured progress events into the existing evaluation dashboard."""

    def __init__(self, dashboard: Any) -> None:
        self.dashboard = dashboard

    def apply_event(self, payload: Dict[str, Any]) -> None:
        event_type = payload.get("event")
        if event_type == "meta":
            total_tests = payload.get("total_tests")
            if total_tests is not None:
                self.dashboard.metrics.total_tests = int(total_tests)
            title = payload.get("title")
            if title:
                self.dashboard.title = str(title)
            return

        if event_type != "result":
            return

        self.dashboard.update(
            status=payload.get("status"),
            name=payload.get("name"),
            latency=float(payload.get("latency", 0.0) or 0.0),
            reason=payload.get("reason"),
            behavior_tested=bool(payload.get("behavior_tested", False)),
            behavior_passed=bool(payload.get("behavior_passed", False)),
        )

    def replay_file(self, path: Path, processed_lines: int = 0) -> int:
        path = Path(path)
        if not path.exists():
            return processed_lines

        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        for line in lines[processed_lines:]:
            self.apply_event(json.loads(line))

        return len(lines)
