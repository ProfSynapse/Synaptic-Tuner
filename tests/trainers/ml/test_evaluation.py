"""Tests for Trainers/ml/evaluation/ — metrics computation and report building."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Trainers.ml.config import TrainingConfig
from Trainers.ml.evaluation.metrics import evaluate_model
from Trainers.ml.evaluation.report import build_metrics_report


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    @pytest.fixture
    def fitted_classifier(self, classification_df: pd.DataFrame):
        """A simple fitted classification pipeline for testing."""
        X = classification_df[["age", "income", "tenure"]]
        y = classification_df["target"]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression(random_state=42)),
        ])
        pipe.fit(X, y)
        return pipe, X, y

    def test_returns_metrics_dict(self, fitted_classifier):
        pipe, X, y = fitted_classifier
        result = evaluate_model(pipe, X, y, "classification", ["accuracy"])
        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_multiple_metrics(self, fitted_classifier):
        pipe, X, y = fitted_classifier
        result = evaluate_model(
            pipe, X, y, "classification",
            ["accuracy", "f1_weighted", "precision"],
        )
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# build_metrics_report
# ---------------------------------------------------------------------------

class TestBuildMetricsReport:
    def test_report_structure(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_test = pd.DataFrame({"a": [7], "b": [8]})
        metrics = {"accuracy": 0.9}

        report = build_metrics_report(config, X_train, X_test, metrics)

        assert report["task"]["type"] == "classification"
        assert report["task"]["name"] == "test_classifier"
        assert report["algorithm"] == "lightgbm"
        assert report["dataset"]["train_samples"] == 3
        assert report["dataset"]["test_samples"] == 1
        assert report["dataset"]["features"] == 2
        assert report["test_metrics"] == {"accuracy": 0.9}
        assert "timestamp" in report
