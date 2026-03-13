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
from .data_loaders import load_dataset
from .metrics import compute_metrics

__all__ = [
    "load_dataset",
    "compute_metrics",
]
