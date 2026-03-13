"""Tests for shared/experiment_tracking/local_tracker.py — JSON file tracker."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.experiment_tracking.local_tracker import LocalTracker


class TestLocalTracker:
    def test_start_run_writes_tracking_json(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        tracker.set_experiment("test_exp")
        with tracker.start_run("run_001"):
            tracker.log_params({"lr": 0.05})
            tracker.log_metrics({"accuracy": 0.91})

        tracking_file = tmp_path / "tracking.json"
        assert tracking_file.exists()

        data = json.loads(tracking_file.read_text())
        assert data["experiment"] == "test_exp"
        assert data["run_name"] == "run_001"
        assert data["params"]["lr"] == 0.05
        assert data["metrics"]["accuracy"] == 0.91
        assert "started_at" in data
        assert "ended_at" in data

    def test_log_params_accumulates(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with tracker.start_run("run"):
            tracker.log_params({"a": 1})
            tracker.log_params({"b": 2})

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["params"] == {"a": 1, "b": 2}

    def test_log_metrics_accumulates(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with tracker.start_run("run"):
            tracker.log_metrics({"loss": 0.5})
            tracker.log_metrics({"loss": 0.3, "acc": 0.9})

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["metrics"]["loss"] == 0.3  # Overwritten
        assert data["metrics"]["acc"] == 0.9

    def test_log_artifact(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with tracker.start_run("run"):
            tracker.log_artifact("/path/to/model.joblib")
            tracker.log_artifact("/path/to/metrics.json")

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["artifacts"] == [
            "/path/to/model.joblib",
            "/path/to/metrics.json",
        ]

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "dir"
        tracker = LocalTracker(nested)
        with tracker.start_run("run"):
            tracker.log_metrics({"x": 1.0})

        assert (nested / "tracking.json").exists()

    def test_default_experiment_name(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with tracker.start_run("run"):
            pass

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["experiment"] == "default"

    def test_writes_even_on_exception(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with pytest.raises(RuntimeError):
            with tracker.start_run("run"):
                tracker.log_metrics({"before_error": 1.0})
                raise RuntimeError("boom")

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["metrics"]["before_error"] == 1.0
        assert "ended_at" in data
