"""
shared/experiment_tracking/

Experiment tracking abstraction layer. Provides a unified interface for
logging hyperparameters, metrics, and artifacts across different backends.

Backends:
    - "local": Zero-dependency JSON file tracker (always available).
    - "mlflow": Full MLflow integration (requires pip install mlflow).

Usage:
    from shared.experiment_tracking import create_tracker

    tracker = create_tracker(backend="local", output_dir="./output")
    tracker.set_experiment("my_experiment")
    with tracker.start_run("run_001"):
        tracker.log_params({"learning_rate": 0.05})
        tracker.log_metrics({"accuracy": 0.91})
"""
from .local_tracker import LocalTracker
from .tracker import ExperimentTracker


def create_tracker(
    backend: str = "local", output_dir: str = "."
) -> ExperimentTracker:
    """Create an experiment tracker by backend name.

    Args:
        backend: Tracker backend — "mlflow" or "local".
        output_dir: Output directory for LocalTracker's JSON file.
                    Ignored when backend is "mlflow" and mlflow is installed.

    Returns:
        An ExperimentTracker instance ready to use.

    Notes:
        If backend is "mlflow" but the mlflow package is not installed,
        a warning is emitted and LocalTracker is returned as a fallback.
    """
    if backend == "mlflow":
        try:
            from .mlflow_tracker import MLflowTracker

            return MLflowTracker()
        except ImportError:
            import warnings

            warnings.warn(
                "mlflow not installed, falling back to local JSON tracker. "
                "Install with: pip install mlflow",
                stacklevel=2,
            )
            return LocalTracker(output_dir)
    return LocalTracker(output_dir)


__all__ = [
    "ExperimentTracker",
    "LocalTracker",
    "create_tracker",
]
