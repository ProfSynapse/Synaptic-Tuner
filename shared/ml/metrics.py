"""
shared/ml/metrics.py

Evaluation metric computation for classification and regression tasks.
Wraps scikit-learn metric functions behind a unified interface driven
by metric names from the training config.

Used by: Trainers/ml/train.py after model prediction to evaluate performance.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric registries — map metric name to a callable
# ---------------------------------------------------------------------------

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


def _f1_weighted(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _precision(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import precision_score
    return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))


def _recall(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import recall_score
    return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))


def _roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Union[np.ndarray, None] = None,
    **_: object,
) -> float:
    from sklearn.metrics import roc_auc_score

    if y_proba is None:
        logger.warning("roc_auc requested but y_proba is None; returning NaN")
        return float("nan")

    # Handle binary vs multiclass
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        # Binary: use probability of the positive class
        scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        return float(roc_auc_score(y_true, scores))

    # Multiclass: use one-vs-rest with probabilities
    return float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    )


def _log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Union[np.ndarray, None] = None,
    **_: object,
) -> float:
    from sklearn.metrics import log_loss as sklearn_log_loss

    if y_proba is None:
        logger.warning("log_loss requested but y_proba is None; returning NaN")
        return float("nan")
    return float(sklearn_log_loss(y_true, y_proba))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import root_mean_squared_error
    return float(root_mean_squared_error(y_true, y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import mean_absolute_error
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(y_true, y_pred))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, **_: object) -> float:
    from sklearn.metrics import mean_absolute_percentage_error
    return float(mean_absolute_percentage_error(y_true, y_pred))


_CLASSIFICATION_METRICS: dict[str, callable] = {
    "accuracy": _accuracy,
    "f1_weighted": _f1_weighted,
    "f1_macro": _f1_macro,
    "precision": _precision,
    "recall": _recall,
    "roc_auc": _roc_auc,
    "log_loss": _log_loss,
}

_REGRESSION_METRICS: dict[str, callable] = {
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
    "mape": _mape,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Union[np.ndarray, None],
    task_type: str,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested metrics for a given task type.

    Supported classification metrics:
        accuracy, f1_weighted, f1_macro, roc_auc, precision, recall, log_loss.

    Supported regression metrics:
        rmse, mae, r2, mape.

    Args:
        y_true: Ground truth labels (classification) or values (regression).
        y_pred: Predicted labels or values.
        y_proba: Predicted probabilities (classification only, optional).
                 Shape: (n_samples, n_classes) or (n_samples,) for binary.
                 Required for roc_auc and log_loss metrics.
        task_type: Either "classification" or "regression".
        metric_names: List of metric names to compute.

    Returns:
        Dict mapping each metric name to its computed float value.

    Raises:
        ValueError: If task_type is invalid or a metric name is unknown
                    for the given task type.
    """
    if task_type == "classification":
        registry = _CLASSIFICATION_METRICS
    elif task_type == "regression":
        registry = _REGRESSION_METRICS
    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Expected 'classification' or 'regression'."
        )

    # Validate all metric names before computing any
    unknown = [name for name in metric_names if name not in registry]
    if unknown:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown {task_type} metric(s): {unknown}. Available: {available}"
        )

    results: dict[str, float] = {}
    for name in metric_names:
        metric_fn = registry[name]
        value = metric_fn(y_true, y_pred, y_proba=y_proba)
        results[name] = round(value, 6) if not np.isnan(value) else value
        logger.debug("  %s = %.6f", name, value)

    return results
