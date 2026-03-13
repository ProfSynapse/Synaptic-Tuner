"""
shared/experiment_tracking/mlflow_tracker.py

MLflow-backed experiment tracker. Requires the mlflow package to be installed.
If mlflow is not available, the create_tracker() factory in __init__.py falls
back to LocalTracker automatically.

Used by: create_tracker(backend="mlflow") when mlflow is installed.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from .tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """MLflow-backed experiment tracker.

    Wraps the mlflow Python API behind the ExperimentTracker interface.
    Import of mlflow happens at construction time so callers get an
    immediate ImportError if the package is missing.

    Raises:
        ImportError: If mlflow is not installed.
    """

    def __init__(self) -> None:
        import mlflow  # Fail fast if not installed

        self._mlflow = mlflow

    def set_experiment(self, experiment_name: str) -> None:
        """Create or select an MLflow experiment.

        Args:
            experiment_name: MLflow experiment name.
        """
        self._mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        """Context manager wrapping mlflow.start_run().

        Args:
            run_name: Display name for the MLflow run.
        """
        with self._mlflow.start_run(run_name=run_name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the active MLflow run.

        Args:
            params: Key-value pairs of hyperparameters.
        """
        self._mlflow.log_params(params)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log metrics to the active MLflow run.

        Args:
            metrics: Key-value pairs of metric name to value.
            step: Optional step number for iterative logging.
        """
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Log a file artifact to the active MLflow run.

        Args:
            local_path: Path to the file to log.
        """
        self._mlflow.log_artifact(local_path)
