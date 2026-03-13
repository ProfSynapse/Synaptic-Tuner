"""Real-world validation tests: Iris multiclass E2E + CLI subprocess entry point.

Uses sklearn's Iris dataset (no external downloads) to validate the full
ML pipeline beyond synthetic data. Verifies both the Python API and the
user-facing CLI entry point.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.datasets import load_iris

from Trainers.ml.train import main as train_main


def _write_iris_csv(path: Path) -> Path:
    """Export sklearn Iris dataset to CSV, return file path."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # type: ignore[union-attr]
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    csv_path = path / "iris.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _iris_config(train_path: str, output_dir: str) -> dict:
    """Build a minimal Iris classification config dict."""
    return {
        "task": {
            "type": "classification",
            "name": "iris_multiclass",
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
                "columns": [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ],
                "imputer": "median",
                "scaler": "standard",
            },
        },
        "algorithm": {
            "name": "lightgbm",
            "params": {"n_estimators": 50, "verbosity": -1},
        },
        "evaluation": {"metrics": ["accuracy", "f1_weighted"]},
        "output": {
            "dir": output_dir,
            "save_model": "joblib",
            "save_pipeline": True,
        },
        "tracking": {
            "enabled": False,
            "backend": "local",
            "experiment_name": "iris_test",
        },
    }


@pytest.mark.integration
class TestIrisEndToEnd:
    """Validate the full pipeline on real-world Iris data (3 classes)."""

    def test_iris_multiclass_pipeline(self, tmp_path: Path):
        """Train LightGBM on Iris via Python API, verify artifacts and quality."""
        # Arrange: write Iris CSV + YAML config
        csv_path = _write_iris_csv(tmp_path)
        output_dir = tmp_path / "output"
        config_dict = _iris_config(str(csv_path), str(output_dir))
        config_path = tmp_path / "iris_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Act: run the pipeline
        run_dir = train_main(str(config_path))

        # Assert: output artifacts exist
        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "model.joblib").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "schema.json").exists()

        # Assert: metrics are present and reasonable
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["task"]["type"] == "classification"
        assert metrics["task"]["name"] == "iris_multiclass"
        assert "accuracy" in metrics["test_metrics"]
        assert "f1_weighted" in metrics["test_metrics"]
        # Iris is easy — LightGBM should achieve > 0.8 accuracy
        assert metrics["test_metrics"]["accuracy"] > 0.8
        assert metrics["test_metrics"]["f1_weighted"] > 0.8

        # Assert: schema reflects 3-class problem
        schema = json.loads((run_dir / "schema.json").read_text())
        assert schema["target"] == "target"
        assert "target_classes" in schema
        assert len(schema["target_classes"]) == 3

        # Assert: model can be reloaded and predicts all 3 classes
        import joblib

        pipeline = joblib.load(run_dir / "model.joblib")
        iris_df = pd.read_csv(csv_path)
        X = iris_df.drop(columns=["target"])
        predictions = pipeline.predict(X)
        unique_preds = set(predictions)
        assert len(unique_preds) == 3, f"Expected 3 classes, got {unique_preds}"


@pytest.mark.integration
class TestCLIEntryPoint:
    """Validate the user-facing CLI: python -m Trainers.ml.train --config."""

    def test_cli_trains_and_produces_artifacts(self, tmp_path: Path):
        """Invoke the CLI as a subprocess, verify exit code and output."""
        # Arrange: write Iris CSV + YAML config
        csv_path = _write_iris_csv(tmp_path)
        output_dir = tmp_path / "cli_output"
        config_dict = _iris_config(str(csv_path), str(output_dir))
        config_path = tmp_path / "cli_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Act: run via subprocess (the actual user entry point)
        result = subprocess.run(
            [sys.executable, "-m", "Trainers.ml.train", "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Assert: process succeeded
        assert result.returncode == 0, (
            f"CLI failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

        # Assert: output directory was created with artifacts
        assert output_dir.exists(), "Output directory was not created"
        # Find the timestamped run directory inside output_dir
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
        run_dir = run_dirs[0]

        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "model.joblib").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "schema.json").exists()
