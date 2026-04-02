"""
HF Jobs job watching mixin.

Location: tuner/backends/training/cloud/_hf_job_watcher.py
Purpose: Monitor running HF Jobs via dashboard and log streaming
Used by: HFJobsBackend (via mixin inheritance) in hf_jobs_backend.py

Handles the live dashboard rendering loop that polls job status,
streams job logs, and syncs remote bucket logs into a local temp
directory for the LiveDashboard widget.
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

from shared.utilities.env import get_hf_token
from tuner.core.config import CloudTrainingConfig
from tuner.ui import RICH_AVAILABLE


class HFJobWatcherMixin:
    """Methods for monitoring and displaying running HF Jobs."""

    def _should_use_remote_dashboard(self, config: CloudTrainingConfig) -> bool:
        """Only use the live remote dashboard for interactive HF bucket runs."""
        return (
            config.artifact_backend == "hf_bucket"
            and RICH_AVAILABLE
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )

    def _watch_job_with_remote_dashboard(
        self,
        *,
        config: CloudTrainingConfig,
        huggingface_hub,
        job_id: str,
        artifact_prefix: str,
    ) -> int:
        """Render the existing training dashboard using metrics mirrored to the HF bucket."""
        from shared.ui import LiveDashboard
        from huggingface_hub import sync_bucket

        hf_token = get_hf_token()
        timeout_seconds = int(config.timeout_hours * 3600)
        poll_interval = 15
        elapsed = 0
        last_status = None
        last_job_log_offset = 0
        processed_remote_lines = 0
        local_logs_dir = Path(tempfile.mkdtemp(prefix="hf-job-logs-"))
        dashboard = LiveDashboard(
            title=f"HF Jobs {config.method.upper()}",
            training_type=config.method,
            log_lines=5,
        )

        try:
            with dashboard:
                while elapsed < timeout_seconds:
                    try:
                        job_info = huggingface_hub.inspect_job(job_id=job_id)
                        status_obj = getattr(job_info, "status", None)
                        status = status_obj.stage if status_obj and hasattr(status_obj, "stage") else str(status_obj or "UNKNOWN")
                    except Exception as e:
                        dashboard.update(log_message=f"Status check failed: {e}")
                        status = last_status or "UNKNOWN"

                    if status != last_status:
                        dashboard.update(log_message=f"Job status: {status}")
                        last_status = status

                    try:
                        logs = huggingface_hub.fetch_job_logs(job_id=job_id) or ""
                        if len(logs) > last_job_log_offset:
                            new_lines = [line for line in logs[last_job_log_offset:].splitlines() if line.strip()]
                            for line in new_lines[-2:]:
                                dashboard.update(log_message=line)
                            last_job_log_offset = len(logs)
                    except Exception:
                        pass

                    try:
                        sync_bucket(
                            f"hf://buckets/{config.artifact_identifier}/{artifact_prefix}/logs",
                            str(local_logs_dir),
                            token=hf_token,
                        )
                        processed_remote_lines = self._update_dashboard_from_local_log(
                            local_logs_dir=local_logs_dir,
                            dashboard=dashboard,
                            processed_remote_lines=processed_remote_lines,
                        )
                    except Exception as e:
                        dashboard.update(log_message=f"Remote log sync unavailable: {e}")

                    if status in ("completed", "COMPLETED"):
                        dashboard.update(log_message="Training completed successfully.")
                        return self._finalize_completed_job(config=config, artifact_prefix=artifact_prefix)
                    if status in ("error", "ERROR", "failed", "FAILED"):
                        dashboard.update(log_message=f"Job failed with status: {status}")
                        return 1
                    if status in ("cancelled", "CANCELLED"):
                        dashboard.update(log_message="Job was cancelled.")
                        return 1

                    time.sleep(poll_interval)
                    elapsed += poll_interval
        finally:
            shutil.rmtree(local_logs_dir, ignore_errors=True)

        recovered = self._recover_completed_run_from_bucket(
            config=config,
            artifact_prefix=artifact_prefix,
        )
        if recovered:
            print()
            print(f"  Job {job_id} appears complete based on synced artifacts.")
            return self._finalize_completed_job(config=config, artifact_prefix=artifact_prefix)

        print()
        print(f"  Job {job_id} failed: Timeout exceeded")
        return 1

    def _update_dashboard_from_local_log(self, *, local_logs_dir: Path, dashboard, processed_remote_lines: int) -> int:
        """Read mirrored bucket logs from a local temp directory and feed them into the dashboard."""
        candidates = sorted(local_logs_dir.glob("training_*.jsonl"))
        if not candidates:
            return processed_remote_lines

        with candidates[-1].open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]

        for line in lines[processed_remote_lines:]:
            dashboard._process_log_line(line)

        return len(lines)
