"""
shared/ml/

Machine learning utilities for the ML training pipeline. Provides reusable
data loading and metric computation used across trainers.

Modules:
    data_loaders - Multi-format dataset loading (CSV, Parquet, JSONL, Excel).
    metrics      - Classification and regression metric computation.

Usage:
    from shared.ml import load_dataset, compute_metrics

    df = load_dataset("data.csv", target_column="label")
    results = compute_metrics(y_true, y_pred, y_proba, "classification", ["accuracy"])
"""
from importlib import import_module


_EXPORTS = {
    "load_dataset": ".data_loaders",
    "compute_metrics": ".metrics",
}


def __getattr__(name):
    """Load only the optional feature requested by the caller.

    Missing feature dependencies retain their original import errors; importing
    dependency-free retrieval metrics must not require dataframe/ML packages.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))

__all__ = [
    "load_dataset",
    "compute_metrics",
]
