"""Integration tests for auto-registration and LocalTracker + registry interaction."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.experiment_tracking.local_tracker import LocalTracker
from shared.experiment_tracking.registry import RunRegistry
from shared.experiment_tracking.schema import RunRecord


class TestAutoRegistration:
    """Verify that LocalTracker auto-registers completed runs in the registry."""

    def test_successful_run_auto_registers(self, tmp_path: Path):
        """P1: Auto-registration on start_run() exit."""
        registry_path = tmp_path / ".tracking" / "registry.jsonl"

        # Patch _default_registry_path to use our tmp_path
        with patch(
            "shared.experiment_tracking.registry._default_registry_path",
            return_value=registry_path,
        ):
            tracker = LocalTracker(tmp_path / "output")
            tracker.set_experiment("test_exp")
            with tracker.start_run("my_run"):
                tracker.log_params({"lr": 0.01})
                tracker.log_metrics({"accuracy": 0.95})

        # Verify run was registered
        assert registry_path.exists()
        registry = RunRegistry(registry_path)
        runs = registry.find_runs()
        assert len(runs) == 1
        assert runs[0].run_type == "ml"
        assert runs[0].name == "my_run"
        assert runs[0].status == "completed"

    def test_failed_run_not_auto_registered(self, tmp_path: Path):
        """P1: Failed run does NOT auto-register."""
        registry_path = tmp_path / ".tracking" / "registry.jsonl"

        with patch(
            "shared.experiment_tracking.registry._default_registry_path",
            return_value=registry_path,
        ):
            with pytest.raises(RuntimeError):
                tracker = LocalTracker(tmp_path / "output")
                with tracker.start_run("failed_run"):
                    tracker.log_metrics({"loss": 5.0})
                    raise RuntimeError("Training crashed")

        # tracking.json should still be written (for debugging)
        assert (tmp_path / "output" / "tracking.json").exists()

        # But registry should NOT have the run
        if registry_path.exists():
            registry = RunRegistry(registry_path)
            runs = registry.find_runs()
            assert len(runs) == 0
        # If file doesn't exist at all, that's also correct

    def test_auto_registration_failure_is_non_fatal(self, tmp_path: Path):
        """Registration failure should log a warning but never block the run."""
        with patch(
            "shared.experiment_tracking.registry.RunRegistry.register_run",
            side_effect=PermissionError("Can't write registry"),
        ):
            tracker = LocalTracker(tmp_path / "output")
            with tracker.start_run("my_run"):
                tracker.log_metrics({"accuracy": 0.9})

        # Run should complete successfully despite registration failure
        data = json.loads((tmp_path / "output" / "tracking.json").read_text())
        assert data["metrics"]["accuracy"] == 0.9

    def test_log_metadata_stored_on_tracker(self, tmp_path: Path):
        """log_metadata() stores key-value pairs accessible by the tracker."""
        tracker = LocalTracker(tmp_path / "output")
        with tracker.start_run("meta_run"):
            tracker.log_metadata("dataset_version", "v1.8")
            tracker.log_metadata("experiment_phase", "pilot")

        # Verify metadata was stored internally
        assert tracker._metadata["dataset_version"] == "v1.8"
        assert tracker._metadata["experiment_phase"] == "pilot"

    def test_multiple_runs_each_auto_register(self, tmp_path: Path):
        """Multiple successive runs should each get registered."""
        registry_path = tmp_path / ".tracking" / "registry.jsonl"

        with patch(
            "shared.experiment_tracking.registry._default_registry_path",
            return_value=registry_path,
        ):
            for i in range(3):
                tracker = LocalTracker(tmp_path / f"output_{i}")
                tracker.set_experiment("multi_exp")
                with tracker.start_run(f"run_{i}"):
                    tracker.log_metrics({"step": float(i)})

        registry = RunRegistry(registry_path)
        runs = registry.find_runs()
        assert len(runs) == 3


class TestBackwardCompatibility:
    """Verify existing LocalTracker behavior is preserved."""

    def test_tracking_json_still_written(self, tmp_path: Path):
        """Core behavior: tracking.json is always written."""
        tracker = LocalTracker(tmp_path)
        tracker.set_experiment("compat_test")
        with tracker.start_run("run_001"):
            tracker.log_params({"lr": 0.05})
            tracker.log_metrics({"accuracy": 0.91})

        tracking_file = tmp_path / "tracking.json"
        assert tracking_file.exists()
        data = json.loads(tracking_file.read_text())
        assert data["experiment"] == "compat_test"
        assert data["params"]["lr"] == 0.05

    def test_exception_still_writes_tracking_json(self, tmp_path: Path):
        tracker = LocalTracker(tmp_path)
        with pytest.raises(ValueError):
            with tracker.start_run("bad_run"):
                tracker.log_metrics({"before": 1.0})
                raise ValueError("boom")

        data = json.loads((tmp_path / "tracking.json").read_text())
        assert data["metrics"]["before"] == 1.0
        assert "ended_at" in data

    def test_log_metadata_is_callable(self, tmp_path: Path):
        """log_metadata exists and doesn't raise."""
        tracker = LocalTracker(tmp_path)
        with tracker.start_run("run"):
            tracker.log_metadata("key", "value")
