"""Tests for shared/ml/metrics.py — classification and regression metrics."""
from __future__ import annotations

import numpy as np
import pytest

from shared.ml.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    @pytest.fixture
    def binary_data(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7],
            [0.6, 0.4], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6],
        ])
        return y_true, y_pred, y_proba

    def test_accuracy(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_metrics(y_true, y_pred, y_proba, "classification", ["accuracy"])
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_f1_weighted(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_metrics(y_true, y_pred, y_proba, "classification", ["f1_weighted"])
        assert 0.0 <= result["f1_weighted"] <= 1.0

    def test_roc_auc(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_metrics(y_true, y_pred, y_proba, "classification", ["roc_auc"])
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_roc_auc_without_proba_returns_nan(self, binary_data):
        y_true, y_pred, _ = binary_data
        result = compute_metrics(y_true, y_pred, None, "classification", ["roc_auc"])
        assert np.isnan(result["roc_auc"])

    def test_log_loss_without_proba_returns_nan(self, binary_data):
        y_true, y_pred, _ = binary_data
        result = compute_metrics(y_true, y_pred, None, "classification", ["log_loss"])
        assert np.isnan(result["log_loss"])

    def test_multiple_metrics(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        result = compute_metrics(
            y_true, y_pred, y_proba, "classification",
            ["accuracy", "f1_weighted", "precision", "recall"],
        )
        assert len(result) == 4
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1])
        result = compute_metrics(y, y, None, "classification", ["accuracy", "f1_weighted"])
        assert result["accuracy"] == 1.0
        assert result["f1_weighted"] == 1.0


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    @pytest.fixture
    def regression_data(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        return y_true, y_pred

    def test_rmse(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_metrics(y_true, y_pred, None, "regression", ["rmse"])
        assert result["rmse"] > 0

    def test_mae(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_metrics(y_true, y_pred, None, "regression", ["mae"])
        assert result["mae"] > 0

    def test_r2(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_metrics(y_true, y_pred, None, "regression", ["r2"])
        assert result["r2"] <= 1.0

    def test_mape(self, regression_data):
        y_true, y_pred = regression_data
        result = compute_metrics(y_true, y_pred, None, "regression", ["mape"])
        assert result["mape"] >= 0

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y, y, None, "regression", ["rmse", "mae"])
        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestMetricErrors:
    def test_unknown_task_type(self):
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="Unknown task_type"):
            compute_metrics(y, y, None, "clustering", ["accuracy"])

    def test_unknown_metric_name(self):
        y = np.array([1, 0])
        with pytest.raises(ValueError, match="Unknown classification metric"):
            compute_metrics(y, y, None, "classification", ["nonexistent_metric"])

    def test_regression_metric_in_classification(self):
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Unknown classification metric"):
            compute_metrics(y, y, None, "classification", ["rmse"])

    def test_classification_metric_in_regression(self):
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown regression metric"):
            compute_metrics(y, y, None, "regression", ["accuracy"])
