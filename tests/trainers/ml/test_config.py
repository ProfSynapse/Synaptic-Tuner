"""Tests for Trainers/ml/config.py — Pydantic v2 config validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from Trainers.ml.config import (
    AlgorithmConfig,
    CategoricalFeaturesConfig,
    DataConfig,
    EvalConfig,
    FeaturesConfig,
    NumericFeaturesConfig,
    OutputConfig,
    TaskConfig,
    TaskType,
    TrackingConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

class TestTaskConfig:
    def test_valid_classification(self):
        cfg = TaskConfig(type="classification", name="test", target_column="y")
        assert cfg.type == TaskType.CLASSIFICATION
        assert cfg.eval_metric == "f1_weighted"
        assert cfg.random_state == 42

    def test_valid_regression(self):
        cfg = TaskConfig(type="regression", name="test", target_column="y")
        assert cfg.type == TaskType.REGRESSION

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError, match="string_too_short"):
            TaskConfig(type="classification", name="", target_column="y")

    def test_rejects_empty_target_column(self):
        with pytest.raises(ValidationError, match="string_too_short"):
            TaskConfig(type="classification", name="test", target_column="")

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            TaskConfig(type="clustering", name="test", target_column="y")


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_valid_config(self, classification_csv: Path):
        cfg = DataConfig(train_path=str(classification_csv))
        assert cfg.test_size == 0.2
        assert cfg.stratify is True

    def test_rejects_nonexistent_path(self):
        with pytest.raises(ValidationError, match="Training data not found"):
            DataConfig(train_path="/nonexistent/path.csv")

    @pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_test_size(self, classification_csv: Path, test_size: float):
        with pytest.raises(ValidationError):
            DataConfig(train_path=str(classification_csv), test_size=test_size)


# ---------------------------------------------------------------------------
# FeaturesConfig
# ---------------------------------------------------------------------------

class TestFeaturesConfig:
    def test_numeric_only(self):
        cfg = FeaturesConfig(
            numeric=NumericFeaturesConfig(columns=["a", "b"])
        )
        assert cfg.numeric is not None
        assert cfg.categorical is None

    def test_categorical_only(self):
        cfg = FeaturesConfig(
            categorical=CategoricalFeaturesConfig(columns=["c"])
        )
        assert cfg.categorical is not None

    def test_both_types(self):
        cfg = FeaturesConfig(
            numeric=NumericFeaturesConfig(columns=["a"]),
            categorical=CategoricalFeaturesConfig(columns=["b"]),
        )
        assert cfg.numeric is not None
        assert cfg.categorical is not None

    def test_rejects_no_feature_types(self):
        with pytest.raises(ValidationError, match="At least one feature type"):
            FeaturesConfig()

    def test_drop_columns_default(self):
        cfg = FeaturesConfig(numeric=NumericFeaturesConfig(columns=["a"]))
        assert cfg.drop_columns == []

    @pytest.mark.parametrize("imputer", ["mean", "median", "none"])
    def test_valid_imputers(self, imputer: str):
        cfg = NumericFeaturesConfig(columns=["a"], imputer=imputer)
        assert cfg.imputer == imputer

    @pytest.mark.parametrize("scaler", ["standard", "minmax", "robust", "none"])
    def test_valid_scalers(self, scaler: str):
        cfg = NumericFeaturesConfig(columns=["a"], scaler=scaler)
        assert cfg.scaler == scaler

    @pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
    def test_valid_encoders(self, encoder: str):
        cfg = CategoricalFeaturesConfig(columns=["a"], encoder=encoder)
        assert cfg.encoder == encoder


# ---------------------------------------------------------------------------
# TrainingConfig (root)
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_full_config_from_dict(self, classification_config_dict: dict):
        cfg = TrainingConfig(**classification_config_dict)
        assert cfg.task.name == "test_classifier"
        assert cfg.algorithm.name == "lightgbm"

    def test_minimal_config(self, classification_csv: Path):
        cfg = TrainingConfig(
            task={"type": "classification", "name": "t", "target_column": "target"},
            data={"train_path": str(classification_csv)},
            features={"numeric": {"columns": ["age"]}},
        )
        assert cfg.evaluation.metrics == ["accuracy", "f1_weighted"]
        assert cfg.output.dir == "./ml_output"
        assert cfg.tracking.enabled is False

    def test_defaults_applied(self, classification_csv: Path):
        cfg = TrainingConfig(
            task={"type": "classification", "name": "t", "target_column": "target"},
            data={"train_path": str(classification_csv)},
            features={"numeric": {"columns": ["age"]}},
        )
        assert isinstance(cfg.algorithm, AlgorithmConfig)
        assert isinstance(cfg.evaluation, EvalConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.tracking, TrackingConfig)
