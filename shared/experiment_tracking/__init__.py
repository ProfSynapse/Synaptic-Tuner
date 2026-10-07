"""Import-light facade for experiment tracking and orchestration."""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "LocalTracker": ".local_tracker",
    "write_analysis_bundle": ".analysis_bundle",
    "ExperimentOrchestrator": ".experiment_orchestrator",
    "StageResult": ".experiment_orchestrator",
    "ExecutionStageSpec": ".experiment_spec", "ExperimentSpec": ".experiment_spec",
    "load_experiment_spec": ".experiment_spec", "RunRegistry": ".registry",
    "RunFilter": ".schema", "RunRecord": ".schema", "LossResult": ".schema",
    "ExperimentTracker": ".tracker",
    "compute_per_example_losses": ".per_example_loss",
    "save_losses": ".per_example_loss", "load_losses": ".per_example_loss",
    "Experiment": ".experiment", "create_experiment": ".experiment",
    "load_experiment": ".experiment", "TrackingService": ".service",
}

__all__ = [
    "ExperimentTracker", "ExperimentOrchestrator", "ExecutionStageSpec",
    "ExperimentSpec", "Experiment", "LossResult", "LocalTracker", "RunFilter",
    "RunRecord", "RunRegistry", "StageResult", "TrackingService",
    "compute_per_example_losses", "create_tracker", "create_experiment",
    "load_experiment", "load_experiment_spec", "write_analysis_bundle",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def create_tracker(backend: str = "local", output_dir: str = "."):
    """Create the requested tracker, preserving the local fallback contract."""

    if backend == "mlflow":
        try:
            tracker_type = getattr(import_module(".mlflow_tracker", __name__), "MLflowTracker")
            return tracker_type()
        except ImportError:
            import warnings

            warnings.warn(
                "mlflow not installed, falling back to local JSON tracker. "
                "Install with: pip install mlflow",
                stacklevel=2,
            )
    return __getattr__("LocalTracker")(output_dir)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES) | {"create_tracker"})
