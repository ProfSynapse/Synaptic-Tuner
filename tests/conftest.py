"""Root conftest.py — shared fixtures for all ML pipeline tests.

Provides synthetic classification and regression DataFrames, sample YAML
configs, and temporary directory helpers used across test modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Synthetic classification dataset with numeric + categorical columns."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(18, 80, size=n).astype(float),
        "income": rng.normal(50000, 15000, size=n),
        "tenure": rng.integers(0, 20, size=n).astype(float),
        "category_a": rng.choice(["cat", "dog", "bird"], size=n),
        "category_b": rng.choice(["red", "blue", "green"], size=n),
        "target": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Synthetic regression dataset with numeric columns only."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.normal(0, 1, size=n)
    y = 3 * x1 + 2 * x2 - x3 + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": x3,
        "target": y,
    })


# ---------------------------------------------------------------------------
# CSV/JSONL file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_csv(tmp_path: Path, classification_df: pd.DataFrame) -> Path:
    """Write classification DataFrame to a CSV file, return path."""
    p = tmp_path / "classification.csv"
    classification_df.to_csv(p, index=False)
    return p


@pytest.fixture
def regression_csv(tmp_path: Path, regression_df: pd.DataFrame) -> Path:
    """Write regression DataFrame to a CSV file, return path."""
    p = tmp_path / "regression.csv"
    regression_df.to_csv(p, index=False)
    return p


@pytest.fixture
def classification_jsonl(tmp_path: Path, classification_df: pd.DataFrame) -> Path:
    """Write classification DataFrame to a JSONL file, return path."""
    p = tmp_path / "classification.jsonl"
    with open(p, "w") as f:
        for _, row in classification_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    return p


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------

def _base_classification_config(train_path: str) -> dict[str, Any]:
    """Minimal classification config dict."""
    return {
        "task": {
            "type": "classification",
            "name": "test_classifier",
            "target_column": "target",
            "eval_metric": "f1_weighted",
            "random_state": 42,
        },
        "data": {
            "train_path": train_path,
            "test_size": 0.2,
            "stratify": True,
        },
        "features": {
            "numeric": {
                "columns": ["age", "income", "tenure"],
                "imputer": "median",
                "scaler": "standard",
            },
            "categorical": {
                "columns": ["category_a", "category_b"],
                "encoder": "onehot",
            },
        },
        "algorithm": {"name": "lightgbm", "params": {"n_estimators": 10, "verbosity": -1}},
        "evaluation": {"metrics": ["accuracy", "f1_weighted"]},
        "output": {"dir": "", "save_model": "joblib", "save_pipeline": True},
        "tracking": {"enabled": False, "backend": "local", "experiment_name": "test"},
    }


def _base_regression_config(train_path: str) -> dict[str, Any]:
    """Minimal regression config dict."""
    return {
        "task": {
            "type": "regression",
            "name": "test_regressor",
            "target_column": "target",
            "eval_metric": "rmse",
            "random_state": 42,
        },
        "data": {
            "train_path": train_path,
            "test_size": 0.2,
            "stratify": False,
        },
        "features": {
            "numeric": {
                "columns": ["feature_1", "feature_2", "feature_3"],
                "imputer": "median",
                "scaler": "standard",
            },
        },
        "algorithm": {"name": "lightgbm", "params": {"n_estimators": 10, "verbosity": -1}},
        "evaluation": {"metrics": ["rmse", "mae", "r2"]},
        "output": {"dir": "", "save_model": "joblib", "save_pipeline": True},
        "tracking": {"enabled": False, "backend": "local", "experiment_name": "test"},
    }


@pytest.fixture
def classification_config_dict(classification_csv: Path, tmp_path: Path) -> dict[str, Any]:
    """Classification config dict with valid train_path and output dir."""
    cfg = _base_classification_config(str(classification_csv))
    cfg["output"]["dir"] = str(tmp_path / "output")
    return cfg


@pytest.fixture
def regression_config_dict(regression_csv: Path, tmp_path: Path) -> dict[str, Any]:
    """Regression config dict with valid train_path and output dir."""
    cfg = _base_regression_config(str(regression_csv))
    cfg["output"]["dir"] = str(tmp_path / "output")
    return cfg


@pytest.fixture
def classification_yaml(tmp_path: Path, classification_config_dict: dict) -> Path:
    """Write classification config to a YAML file, return path."""
    import yaml
    p = tmp_path / "classification_config.yaml"
    with open(p, "w") as f:
        yaml.dump(classification_config_dict, f)
    return p


@pytest.fixture
def regression_yaml(tmp_path: Path, regression_config_dict: dict) -> Path:
    """Write regression config to a YAML file, return path."""
    import yaml
    p = tmp_path / "regression_config.yaml"
    with open(p, "w") as f:
        yaml.dump(regression_config_dict, f)
    return p
