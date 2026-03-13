"""Tests for Trainers/ml/output.py — run dir creation and artifact saving."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Trainers.ml.output import create_run_dir, save_run_artifacts


@pytest.fixture
def simple_pipeline():
    """A tiny fitted pipeline that can be serialized by joblib."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("estimator", LinearRegression()),
    ])
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1.0, 2.0, 3.0])
    pipe.fit(X, y)
    return pipe


class TestCreateRunDir:
    def test_creates_timestamped_dir(self, tmp_path: Path):
        run_dir = create_run_dir(str(tmp_path))
        assert run_dir.exists()
        assert run_dir.parent == tmp_path
        # Format: YYYYMMDD_HHMMSS
        assert len(run_dir.name) == 15
        assert run_dir.name[8] == "_"

    def test_creates_logs_subdir(self, tmp_path: Path):
        run_dir = create_run_dir(str(tmp_path))
        assert (run_dir / "logs").exists()

    def test_creates_nested_base_dir(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested"
        run_dir = create_run_dir(str(nested))
        assert run_dir.exists()


class TestSaveRunArtifacts:
    def test_saves_all_artifacts(self, tmp_path: Path, simple_pipeline):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        config = {"task": {"name": "test"}}
        metrics = {"test_metrics": {"accuracy": 0.9}}
        schema = {"columns": {"numeric": ["a"]}, "target": "y"}

        save_run_artifacts(run_dir, config, simple_pipeline, metrics, schema)

        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "model.joblib").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "schema.json").exists()

    def test_config_yaml_content(self, tmp_path: Path, simple_pipeline):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        config = {"task": {"type": "classification", "name": "test"}}
        save_run_artifacts(run_dir, config, simple_pipeline, {}, {})

        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert saved["task"]["name"] == "test"

    def test_metrics_json_content(self, tmp_path: Path, simple_pipeline):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        metrics = {"test_metrics": {"accuracy": 0.95, "f1": 0.92}}
        save_run_artifacts(run_dir, {}, simple_pipeline, metrics, {})

        saved = json.loads((run_dir / "metrics.json").read_text())
        assert saved["test_metrics"]["accuracy"] == 0.95

    def test_schema_json_content(self, tmp_path: Path, simple_pipeline):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        schema = {"columns": {"numeric": ["a", "b"]}, "target": "y"}
        save_run_artifacts(run_dir, {}, simple_pipeline, {}, schema)

        saved = json.loads((run_dir / "schema.json").read_text())
        assert saved["target"] == "y"
        assert saved["columns"]["numeric"] == ["a", "b"]

    def test_model_loadable(self, tmp_path: Path, simple_pipeline):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_run_artifacts(run_dir, {}, simple_pipeline, {}, {})

        import joblib
        loaded = joblib.load(run_dir / "model.joblib")
        assert hasattr(loaded, "predict")
