"""
shared/experiment_tracking/local_tracker.py

Zero-dependency JSON file tracker. Writes all run data (params, metrics,
artifacts) to a single tracking.json in the output directory.

On successful run completion, auto-registers a RunRecord in the unified
JSONL registry (best-effort — registration failure never blocks the run).

Used by: create_tracker() as the default/fallback backend.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from .tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class LocalTracker(ExperimentTracker):
    """Writes tracking data to a JSON file in the output directory.

    This is the zero-dependency fallback. Always available regardless of
    which optional packages are installed. Produces a human-readable
    tracking.json with experiment metadata, params, and metrics.

    On successful run exit (no exception), auto-registers the run in the
    unified JSONL registry at {repo}/.tracking/registry.jsonl.

    Args:
        output_dir: Directory where tracking.json will be written.
                    Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._experiment_name: str = "default"
        self._run_data: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def set_experiment(self, experiment_name: str) -> None:
        """Create or select an experiment by name.

        Args:
            experiment_name: Human-readable experiment identifier.
        """
        self._experiment_name = experiment_name

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        """Context manager for a tracking run.

        Initializes run data on entry, writes tracking.json on exit
        (even if an exception occurs within the run). On successful
        completion (no exception), auto-registers a RunRecord in the
        unified registry.

        Args:
            run_name: Human-readable name for this run.
        """
        self._run_data = {
            "experiment": self._experiment_name,
            "run_name": run_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "params": {},
            "metrics": {},
        }
        self._metadata = {}
        exception_occurred = False
        try:
            yield
        except BaseException:
            exception_occurred = True
            raise
        finally:
            self._run_data["ended_at"] = datetime.now(timezone.utc).isoformat()
            self._flush()
            if not exception_occurred:
                self._auto_register()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run.

        Args:
            params: Key-value pairs of hyperparameters.
        """
        self._run_data.setdefault("params", {}).update(params)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log numeric metrics for the current run.

        Args:
            metrics: Key-value pairs of metric name to value.
            step: Optional step number (ignored by LocalTracker; latest
                  value wins).
        """
        self._run_data.setdefault("metrics", {}).update(metrics)

    def log_artifact(self, local_path: str) -> None:
        """Record an artifact path for the current run.

        Note: LocalTracker only stores the path reference; it does not
        copy the artifact file.

        Args:
            local_path: Path to the file to log.
        """
        self._run_data.setdefault("artifacts", []).append(local_path)

    def log_metadata(self, key: str, value: Any) -> None:
        """Store metadata that will be included in the auto-registered RunRecord.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    def _flush(self) -> None:
        """Write accumulated run data to tracking.json."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "tracking.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._run_data, f, indent=2)

    def _auto_register(self) -> None:
        """Best-effort registration of the completed run in the unified registry.

        Converts the accumulated tracking data to a RunRecord via the ML adapter
        and appends it to the JSONL registry. Failures are logged but never raised.
        """
        try:
            from .adapters import ml_tracking_to_run_record
            from .registry import RunRegistry

            record = ml_tracking_to_run_record(
                self._run_data,
                str(self._output_dir.resolve()),
            )
            registry = RunRegistry()
            registry.register_run(record)
        except Exception as exc:
            logger.warning("Auto-registration failed (non-fatal): %s", exc)
