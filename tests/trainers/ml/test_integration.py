"""Integration test: end-to-end ML pipeline — YAML config to saved artifacts.

Validates the full pipeline: config loading -> data split -> preprocessing ->
training -> evaluation -> artifact saving -> experiment tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from Trainers.ml.train import main as train_main


@pytest.mark.integration
class TestEndToEndPipeline:
    def test_classification_pipeline(
        self, classification_config_dict: dict, tmp_path: Path,
    ):
        """Full classification pipeline: YAML -> train -> evaluate -> save."""
        # Write config to YAML
        config_path = tmp_path / "config.yaml"
        output_dir = tmp_path / "output"
        classification_config_dict["output"]["dir"] = str(output_dir)
        with open(config_path, "w") as f:
            yaml.dump(classification_config_dict, f)

        # Run the pipeline
        run_dir = train_main(str(config_path))

        # Verify output directory structure
        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "model.joblib").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "schema.json").exists()
        assert (run_dir / "logs").is_dir()

        # Verify metrics.json content
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["task"]["type"] == "classification"
        assert metrics["task"]["name"] == "test_classifier"
        assert metrics["algorithm"] == "lightgbm"
        assert metrics["dataset"]["train_samples"] > 0
        assert metrics["dataset"]["test_samples"] > 0
        assert "accuracy" in metrics["test_metrics"]
        assert "f1_weighted" in metrics["test_metrics"]
        assert 0.0 <= metrics["test_metrics"]["accuracy"] <= 1.0

        # Verify schema.json content
        schema = json.loads((run_dir / "schema.json").read_text())
        assert schema["target"] == "target"
        assert "numeric" in schema["columns"]
        assert "categorical" in schema["columns"]

        # Verify tracking.json from LocalTracker
        assert (run_dir / "tracking.json").exists()
        tracking = json.loads((run_dir / "tracking.json").read_text())
        assert tracking["experiment"] == "test"
        assert "params" in tracking
        assert "metrics" in tracking

        # Verify model can be loaded
        import joblib
        pipeline = joblib.load(run_dir / "model.joblib")
        assert hasattr(pipeline, "predict")

    def test_regression_pipeline(
        self, regression_config_dict: dict, tmp_path: Path,
    ):
        """Full regression pipeline: YAML -> train -> evaluate -> save."""
        config_path = tmp_path / "config.yaml"
        output_dir = tmp_path / "output"
        regression_config_dict["output"]["dir"] = str(output_dir)
        with open(config_path, "w") as f:
            yaml.dump(regression_config_dict, f)

        run_dir = train_main(str(config_path))

        assert run_dir.exists()
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["task"]["type"] == "regression"
        assert "rmse" in metrics["test_metrics"]
        assert "mae" in metrics["test_metrics"]
        assert "r2" in metrics["test_metrics"]
        assert metrics["test_metrics"]["rmse"] >= 0
        assert metrics["test_metrics"]["r2"] <= 1.0

        schema = json.loads((run_dir / "schema.json").read_text())
        assert "target_classes" not in schema  # Regression has no classes

    def test_custom_algorithm_params(
        self, classification_config_dict: dict, tmp_path: Path,
    ):
        """Verify user-provided algorithm params are used."""
        config_path = tmp_path / "config.yaml"
        output_dir = tmp_path / "output"
        classification_config_dict["output"]["dir"] = str(output_dir)
        classification_config_dict["algorithm"]["params"]["n_estimators"] = 5
        with open(config_path, "w") as f:
            yaml.dump(classification_config_dict, f)

        run_dir = train_main(str(config_path))

        # Verify tracking captured the custom param
        tracking = json.loads((run_dir / "tracking.json").read_text())
        assert tracking["params"]["algo_n_estimators"] == 5
