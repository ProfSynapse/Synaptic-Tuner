"""
shared/experiment_tracking/

Experiment tracking abstraction layer. Provides a unified interface for
logging hyperparameters, metrics, and artifacts across different backends,
plus a centralized JSONL registry for cross-run discovery and linkage.

Backends:
    - "local": Zero-dependency JSON file tracker (always available).
    - "mlflow": Full MLflow integration (requires pip install mlflow).

Registry:
    - RunRecord: Common schema for all run types.
    - RunRegistry: Append-only JSONL index at {repo}/.tracking/registry.jsonl.
    - RunFilter: Query filter for run discovery.
    - Adapters: Bridge existing lineage/manifest formats to RunRecord.

Usage:
    from shared.experiment_tracking import create_tracker

    tracker = create_tracker(backend="local", output_dir="./output")
    tracker.set_experiment("my_experiment")
    with tracker.start_run("run_001"):
        tracker.log_params({"learning_rate": 0.05})
        tracker.log_metrics({"accuracy": 0.91})

    # Query runs
    from shared.experiment_tracking import RunRegistry, RunFilter

    registry = RunRegistry()
    recent_sft = registry.find_runs(RunFilter(run_type="sft"))
"""
from .local_tracker import LocalTracker
from .registry import RunRegistry
from .schema import RunFilter, RunRecord
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
    "RunFilter",
    "RunRecord",
    "RunRegistry",
    "create_tracker",
]
