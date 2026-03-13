# Trainers/ml/evaluation/metrics.py
# Evaluates a fitted pipeline on test data, delegating metric computation
# to shared/ml/metrics. Used by train.py after pipeline.fit().

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    metric_names: list[str],
) -> dict[str, float]:
    """Evaluate a fitted pipeline on test data.

    Generates predictions and probability estimates (for classification),
    then delegates to shared/ml/metrics.compute_metrics().

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Test labels/values.
        task_type: "classification" or "regression".
        metric_names: List of metric names to compute.

    Returns:
        Dict mapping metric name to computed value.
    """
    from shared.ml.metrics import compute_metrics

    y_pred = pipeline.predict(X_test)

    # Get probability estimates for classification metrics (roc_auc, log_loss)
    y_proba = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception as exc:
            logger.warning("Could not generate probability predictions: %s", exc)

    metrics = compute_metrics(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        task_type=task_type,
        metric_names=metric_names,
    )

    logger.info("Test metrics: %s", metrics)
    return metrics
