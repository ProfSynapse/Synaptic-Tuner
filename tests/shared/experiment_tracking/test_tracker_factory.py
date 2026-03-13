"""Tests for shared/experiment_tracking/__init__.py — create_tracker factory."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.experiment_tracking import create_tracker
from shared.experiment_tracking.local_tracker import LocalTracker
from shared.experiment_tracking.tracker import ExperimentTracker


class TestCreateTracker:
    def test_default_returns_local(self, tmp_path: Path):
        tracker = create_tracker(output_dir=str(tmp_path))
        assert isinstance(tracker, LocalTracker)

    def test_explicit_local_returns_local(self, tmp_path: Path):
        tracker = create_tracker(backend="local", output_dir=str(tmp_path))
        assert isinstance(tracker, LocalTracker)

    def test_mlflow_fallback_when_not_installed(self, tmp_path: Path):
        """When mlflow import fails, create_tracker should fall back to LocalTracker."""
        # Temporarily make mlflow_tracker's import of mlflow fail
        with patch.dict(sys.modules, {"mlflow": None}):
            # Need to also remove cached import of mlflow_tracker
            # since it may have already successfully imported mlflow
            cached_key = "shared.experiment_tracking.mlflow_tracker"
            saved = sys.modules.pop(cached_key, None)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    tracker = create_tracker(backend="mlflow", output_dir=str(tmp_path))
                    assert isinstance(tracker, LocalTracker)
                    assert any("mlflow not installed" in str(warning.message) for warning in w)
            finally:
                if saved is not None:
                    sys.modules[cached_key] = saved

    def test_all_trackers_implement_abc(self, tmp_path: Path):
        tracker = create_tracker(output_dir=str(tmp_path))
        assert isinstance(tracker, ExperimentTracker)
        assert hasattr(tracker, "set_experiment")
        assert hasattr(tracker, "start_run")
        assert hasattr(tracker, "log_params")
        assert hasattr(tracker, "log_metrics")
        assert hasattr(tracker, "log_artifact")
