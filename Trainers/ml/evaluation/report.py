# Trainers/ml/evaluation/report.py
# Builds the metrics.json report structure for a training run.
# Produces the standardized report format defined in the architecture spec.
# Used by train.py to create the metrics dict before saving artifacts.

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from Trainers.ml.config import TrainingConfig


def build_metrics_report(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_metrics: dict[str, float],
) -> dict:
    """Build the metrics.json report content.

    Args:
        config: Validated TrainingConfig.
        X_train: Training features (for sample/feature counts).
        X_test: Test features (for sample count).
        test_metrics: Computed test metrics from evaluate_model().

    Returns:
        Dict matching the metrics.json schema from the architecture spec.
    """
    return {
        "task": {
            "type": config.task.type.value,
            "name": config.task.name,
            "target_column": config.task.target_column,
            "eval_metric": config.task.eval_metric,
        },
        "algorithm": config.algorithm.name,
        "dataset": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train.shape[1],
        },
        "test_metrics": test_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
