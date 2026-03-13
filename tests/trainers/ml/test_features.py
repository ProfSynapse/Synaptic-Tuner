"""Tests for Trainers/ml/features/builder.py — ColumnTransformer construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from Trainers.ml.config import (
    CategoricalFeaturesConfig,
    FeaturesConfig,
    NumericFeaturesConfig,
)
from Trainers.ml.features.builder import build_preprocessor


class TestBuildPreprocessor:
    def test_numeric_only(self):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(
                columns=["a", "b"], imputer="median", scaler="standard"
            )
        )
        ct = build_preprocessor(config)
        assert isinstance(ct, ColumnTransformer)
        assert len(ct.transformers) == 1
        assert ct.transformers[0][0] == "numeric"

    def test_categorical_only(self):
        config = FeaturesConfig(
            categorical=CategoricalFeaturesConfig(
                columns=["c"], encoder="onehot"
            )
        )
        ct = build_preprocessor(config)
        assert len(ct.transformers) == 1
        assert ct.transformers[0][0] == "categorical"

    def test_both_numeric_and_categorical(self):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(columns=["a"]),
            categorical=CategoricalFeaturesConfig(columns=["b"]),
        )
        ct = build_preprocessor(config)
        names = [t[0] for t in ct.transformers]
        assert "numeric" in names
        assert "categorical" in names

    def test_passthrough_when_no_steps(self):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(
                columns=["a"], imputer="none", scaler="none"
            )
        )
        ct = build_preprocessor(config)
        assert ct.transformers[0][1] == "passthrough"

    def test_fit_transform_numeric(self, classification_df: pd.DataFrame):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(
                columns=["age", "income", "tenure"],
                imputer="median",
                scaler="standard",
            )
        )
        ct = build_preprocessor(config)
        X = classification_df[["age", "income", "tenure"]]
        result = ct.fit_transform(X)
        assert result.shape[0] == len(X)
        assert result.shape[1] == 3
        # StandardScaler: mean ~0, std ~1
        assert np.abs(result.mean(axis=0)).max() < 0.1

    def test_fit_transform_categorical_onehot(self, classification_df: pd.DataFrame):
        config = FeaturesConfig(
            categorical=CategoricalFeaturesConfig(
                columns=["category_a", "category_b"],
                encoder="onehot",
            )
        )
        ct = build_preprocessor(config)
        X = classification_df[["category_a", "category_b"]]
        result = ct.fit_transform(X)
        assert result.shape[0] == len(X)
        # 3 categories for each column = 6 one-hot columns
        assert result.shape[1] == 6

    def test_fit_transform_categorical_ordinal(self, classification_df: pd.DataFrame):
        config = FeaturesConfig(
            categorical=CategoricalFeaturesConfig(
                columns=["category_a"],
                encoder="ordinal",
            )
        )
        ct = build_preprocessor(config)
        X = classification_df[["category_a"]]
        result = ct.fit_transform(X)
        assert result.shape[1] == 1

    @pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
    def test_scaler_variants(self, scaler: str):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(
                columns=["a"], imputer="median", scaler=scaler,
            )
        )
        ct = build_preprocessor(config)
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = ct.fit_transform(X)
        assert result.shape == (5, 1)

    def test_remainder_drop(self, classification_df: pd.DataFrame):
        config = FeaturesConfig(
            numeric=NumericFeaturesConfig(columns=["age"])
        )
        ct = build_preprocessor(config)
        # Pass DataFrame with extra columns — they should be dropped
        X = classification_df[["age", "income"]]
        result = ct.fit_transform(X)
        assert result.shape[1] == 1  # Only "age" kept
