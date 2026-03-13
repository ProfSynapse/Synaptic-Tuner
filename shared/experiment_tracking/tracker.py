"""
shared/experiment_tracking/tracker.py

Abstract base class for experiment tracking backends. Defines the contract
that all trackers (MLflow, local JSON, etc.) must implement.

Used by: Trainers/ml/train.py via create_tracker() factory.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking.

    Implementations:
        - MLflowTracker: Full MLflow integration (requires mlflow package).
        - LocalTracker: Zero-dependency JSON file fallback.

    Usage:
        tracker = create_tracker("local", output_dir="./output")
        tracker.set_experiment("my_experiment")
        with tracker.start_run("run_001"):
            tracker.log_params({"lr": 0.05})
            tracker.log_metrics({"accuracy": 0.91})
    """

    @abstractmethod
    def set_experiment(self, experiment_name: str) -> None:
        """Create or select an experiment by name.

        Args:
            experiment_name: Human-readable experiment identifier.
        """
        ...

    @abstractmethod
    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        """Context manager that brackets a single tracking run.

        All log_params / log_metrics / log_artifact calls should happen
        inside this context. On exit, the tracker finalizes and persists
        the run data.

        Args:
            run_name: Human-readable name for this run.
        """
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters for the current run.

        Args:
            params: Key-value pairs of hyperparameters.
        """
        ...

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log numeric metrics for the current run.

        Args:
            metrics: Key-value pairs of metric name to value.
            step: Optional step number (for iterative logging).
        """
        ...

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """Log a file artifact for the current run.

        Args:
            local_path: Path to the file to log.
        """
        ...
