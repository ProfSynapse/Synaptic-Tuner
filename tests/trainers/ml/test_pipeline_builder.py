"""Tests for Trainers/ml/pipeline_builder.py — Pipeline assembly."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from Trainers.ml.config import TrainingConfig
from Trainers.ml.pipeline_builder import build_pipeline


class TestBuildPipeline:
    def test_returns_pipeline(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        pipeline = build_pipeline(config)
        assert isinstance(pipeline, Pipeline)

    def test_has_preprocessor_and_estimator(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        pipeline = build_pipeline(config)
        assert "preprocessor" in pipeline.named_steps
        assert "estimator" in pipeline.named_steps

    def test_classification_pipeline_fits(
        self, classification_config_dict: dict, classification_df: pd.DataFrame,
    ):
        config = TrainingConfig(**classification_config_dict)
        pipeline = build_pipeline(config)
        X = classification_df.drop(columns=["target"])
        y = classification_df["target"]
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_regression_pipeline_fits(
        self, regression_config_dict: dict, regression_df: pd.DataFrame,
    ):
        config = TrainingConfig(**regression_config_dict)
        pipeline = build_pipeline(config)
        X = regression_df.drop(columns=["target"])
        y = regression_df["target"]
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_unknown_algorithm_raises(self, classification_config_dict: dict):
        classification_config_dict["algorithm"]["name"] = "xgboost"
        config = TrainingConfig(**classification_config_dict)
        with pytest.raises(KeyError, match="Unknown algorithm"):
            build_pipeline(config)
