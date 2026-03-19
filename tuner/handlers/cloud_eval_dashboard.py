"""
Provider-neutral local dashboard replay for cloud evaluation runs.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from shared.cloud_eval_progress import EVAL_PROGRESS_LOG_FILENAME, EvaluationDashboardReplayer
from shared.ui import LiveEvaluationDashboard, RICH_AVAILABLE


FAILED_JOB_STATUSES = {"error", "failed", "cancelled", "ERROR", "FAILED", "CANCELLED"}
COMPLETED_JOB_STATUSES = {"completed", "COMPLETED"}


class CloudEvalProviderAdapter(Protocol):
    """Provider-specific job status and artifact access methods."""

    def fetch_status(self) -> str:
        ...

    def sync_progress(self, local_dir: Path) -> None:
        ...

    def sync_results(self, local_dir: Path) -> None:
        ...


@dataclass
class CloudEvalWatchResult:
    exit_code: int
    status: str
    local_root: Optional[Path] = None
    results_dir: Optional[Path] = None


class CloudEvalDashboardWatcher:
    """Replay cloud evaluation progress into the local rich dashboard."""

    def __init__(
        self,
        provider: CloudEvalProviderAdapter,
        *,
        title: str = "Cloud Evaluation",
        poll_interval: int = 15,
    ) -> None:
        self.provider = provider
        self.title = title
        self.poll_interval = max(1, int(poll_interval))

    def watch(self, *, timeout_seconds: int) -> CloudEvalWatchResult:
        local_root = Path(tempfile.mkdtemp(prefix="cloud-eval-"))
        progress_dir = local_root / "progress"
        results_dir = local_root / "results"
        dashboard = LiveEvaluationDashboard(title=self.title, total_tests=0, log_lines=5)
        replayer = EvaluationDashboardReplayer(dashboard)
        processed_lines = 0
        elapsed = 0

        with dashboard:
            while elapsed < timeout_seconds:
                try:
                    status = self.provider.fetch_status()
                except Exception:
                    status = "UNKNOWN"

                try:
                    self.provider.sync_progress(progress_dir)
                    processed_lines = replayer.replay_file(
                        progress_dir / EVAL_PROGRESS_LOG_FILENAME,
                        processed_lines,
                    )
                except Exception:
                    pass

                if status in COMPLETED_JOB_STATUSES:
                    try:
                        self.provider.sync_results(results_dir)
                    except Exception:
                        pass
                    return CloudEvalWatchResult(
                        exit_code=0,
                        status=status,
                        local_root=local_root,
                        results_dir=results_dir,
                    )

                if status in FAILED_JOB_STATUSES:
                    return CloudEvalWatchResult(
                        exit_code=1,
                        status=status,
                        local_root=local_root,
                        results_dir=results_dir,
                    )

                time.sleep(self.poll_interval)
                elapsed += self.poll_interval

        try:
            self.provider.sync_results(results_dir)
        except Exception:
            pass

        if _has_completed_results(results_dir):
            return CloudEvalWatchResult(
                exit_code=0,
                status="completed",
                local_root=local_root,
                results_dir=results_dir,
            )

        return CloudEvalWatchResult(
            exit_code=1,
            status="timeout",
            local_root=local_root,
            results_dir=results_dir,
        )


def _has_completed_results(results_dir: Optional[Path]) -> bool:
    if results_dir is None:
        return False
    return (Path(results_dir) / "evaluation_results.json").exists()


def can_render_cloud_eval_dashboard() -> bool:
    return RICH_AVAILABLE
