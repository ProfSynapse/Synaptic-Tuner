# Trainers/ml/evaluation/plots.py
# Optional visualization generation (confusion matrix, feature importance).
# Only runs when config.evaluation.generate_plots is True.
# Saves plots to run_dir/plots/. Used by train.py after evaluation.

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_plots(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    task_type: str,
    run_dir: Path,
) -> list[str]:
    """Generate evaluation plots and save to run_dir/plots/.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: Test features.
        y_test: Test labels/values.
        y_pred: Model predictions.
        task_type: "classification" or "regression".
        run_dir: Output directory for this run.

    Returns:
        List of saved plot file paths (relative to run_dir).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot generation")
        return []

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    saved = []

    if task_type == "classification":
        saved.extend(_plot_confusion_matrix(y_test, y_pred, plots_dir, plt))

    saved.extend(_plot_feature_importance(pipeline, X_test, plots_dir, plt))

    return saved


def _plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    plots_dir: Path,
    plt,
) -> list[str]:
    """Generate and save a confusion matrix heatmap."""
    try:
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Confusion Matrix")

        path = plots_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix: %s", path)
        return [str(path)]
    except Exception as exc:
        logger.warning("Failed to generate confusion matrix: %s", exc)
        return []


def _plot_feature_importance(
    pipeline,
    X_test: pd.DataFrame,
    plots_dir: Path,
    plt,
) -> list[str]:
    """Generate and save a feature importance bar chart (if estimator supports it)."""
    try:
        estimator = pipeline.named_steps.get("estimator")
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return []

        importances = estimator.feature_importances_

        # Try to get feature names from the preprocessor
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(sorted_names)), sorted_importances[::-1])
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (Top 20)")

        path = plots_dir / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance: %s", path)
        return [str(path)]
    except Exception as exc:
        logger.warning("Failed to generate feature importance plot: %s", exc)
        return []
