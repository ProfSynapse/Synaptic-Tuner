"""Tests for Trainers/ml/data/splitter.py — data loading and splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from Trainers.ml.config import TrainingConfig
from Trainers.ml.data.splitter import load_and_split


class TestLoadAndSplit:
    def test_basic_split(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        X_train, X_test, y_train, y_test = load_and_split(config)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) + len(X_test) == 200
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_target_not_in_features(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        X_train, X_test, _, _ = load_and_split(config)
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_split_ratio(self, classification_config_dict: dict):
        classification_config_dict["data"]["test_size"] = 0.3
        config = TrainingConfig(**classification_config_dict)
        X_train, X_test, _, _ = load_and_split(config)
        # Approximate: 200 * 0.3 = 60 test, 140 train
        assert 50 <= len(X_test) <= 70

    def test_stratified_split(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        _, _, y_train, y_test = load_and_split(config)
        # Both splits should have both classes
        assert len(y_train.unique()) == 2
        assert len(y_test.unique()) == 2

    def test_non_stratified_regression(self, regression_config_dict: dict):
        config = TrainingConfig(**regression_config_dict)
        X_train, X_test, y_train, y_test = load_and_split(config)
        assert len(X_train) + len(X_test) == 200

    def test_drop_columns(self, classification_config_dict: dict):
        classification_config_dict["features"]["drop_columns"] = ["income"]
        config = TrainingConfig(**classification_config_dict)
        X_train, X_test, _, _ = load_and_split(config)
        assert "income" not in X_train.columns
        assert "income" not in X_test.columns

    def test_separate_test_file(
        self, tmp_path: Path, classification_df: pd.DataFrame,
        classification_config_dict: dict,
    ):
        # Write separate train/test files
        train_df = classification_df.iloc[:150]
        test_df = classification_df.iloc[150:]
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        classification_config_dict["data"]["train_path"] = str(train_path)
        classification_config_dict["data"]["test_path"] = str(test_path)
        config = TrainingConfig(**classification_config_dict)

        X_train, X_test, y_train, y_test = load_and_split(config)
        assert len(X_train) == 150
        assert len(X_test) == 50

    def test_deterministic_with_random_state(self, classification_config_dict: dict):
        config = TrainingConfig(**classification_config_dict)
        _, _, y1, _ = load_and_split(config)
        _, _, y2, _ = load_and_split(config)
        pd.testing.assert_series_equal(y1.reset_index(drop=True), y2.reset_index(drop=True))
