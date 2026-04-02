"""
Tests for tuner/handlers/stages/ subpackage.

Verifies:
- Each stage runner is independently importable
- Re-exports from __init__.py work correctly
- Stage runner constructors accept expected args
- _util helper functions
- Stage runner isolation (no circular imports)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# =========================================================
# Import Isolation
# =========================================================

class TestStageRunnerImports:
    def test_import_from_package_init(self):
        """All 3 runners importable via stages/__init__.py re-exports."""
        from tuner.handlers.stages import (
            HFEvalStageRunner,
            HFLossStageRunner,
            HFTrainingStageRunner,
        )
        assert HFTrainingStageRunner is not None
        assert HFEvalStageRunner is not None
        assert HFLossStageRunner is not None

    def test_import_directly_from_submodules(self):
        """Each runner importable directly from its own module."""
        from tuner.handlers.stages.hf_training_stage import HFTrainingStageRunner
        from tuner.handlers.stages.hf_eval_stage import HFEvalStageRunner
        from tuner.handlers.stages.hf_loss_stage import HFLossStageRunner
        assert HFTrainingStageRunner is not None
        assert HFEvalStageRunner is not None
        assert HFLossStageRunner is not None

    def test_import_via_init_matches_direct_import(self):
        """Re-exports should be the exact same class objects."""
        from tuner.handlers.stages import HFTrainingStageRunner as FromInit
        from tuner.handlers.stages.hf_training_stage import HFTrainingStageRunner as Direct
        assert FromInit is Direct

        from tuner.handlers.stages import HFEvalStageRunner as FromInit2
        from tuner.handlers.stages.hf_eval_stage import HFEvalStageRunner as Direct2
        assert FromInit2 is Direct2

        from tuner.handlers.stages import HFLossStageRunner as FromInit3
        from tuner.handlers.stages.hf_loss_stage import HFLossStageRunner as Direct3
        assert FromInit3 is Direct3

    def test_stages_all_export(self):
        """__all__ should list exactly the 3 runners."""
        import tuner.handlers.stages as stages
        assert set(stages.__all__) == {
            "HFEvalStageRunner",
            "HFLossStageRunner",
            "HFTrainingStageRunner",
        }


# =========================================================
# Utility Functions
# =========================================================

class TestStageUtil:
    def test_optional_backend_value_with_valid_string(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value("hello") == "hello"

    def test_optional_backend_value_strips_whitespace(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value("  hello  ") == "hello"

    def test_optional_backend_value_empty_string_returns_none(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value("") is None

    def test_optional_backend_value_whitespace_only_returns_none(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value("   ") is None

    def test_optional_backend_value_none_returns_none(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value(None) is None

    def test_optional_backend_value_non_string_returns_none(self):
        from tuner.handlers.stages._util import _optional_backend_value
        assert _optional_backend_value(42) is None
        assert _optional_backend_value([]) is None
        assert _optional_backend_value({}) is None


# =========================================================
# Constructor Contracts
# =========================================================

class TestStageRunnerConstructors:
    def test_training_runner_accepts_keyword_args(self, tmp_path):
        from tuner.handlers.stages import HFTrainingStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        assert runner.repo_root == tmp_path
        assert runner.tracking_service is service

    def test_eval_runner_accepts_keyword_args(self, tmp_path):
        from tuner.handlers.stages import HFEvalStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)
        assert runner.repo_root == tmp_path
        assert runner.tracking_service is service

    def test_loss_runner_accepts_keyword_args(self, tmp_path):
        from tuner.handlers.stages import HFLossStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        assert runner.repo_root == tmp_path
        assert runner.tracking_service is service

    def test_training_runner_has_run_method(self, tmp_path):
        from tuner.handlers.stages import HFTrainingStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFTrainingStageRunner(repo_root=tmp_path, tracking_service=service)
        assert callable(getattr(runner, "run", None))

    def test_eval_runner_has_run_method(self, tmp_path):
        from tuner.handlers.stages import HFEvalStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFEvalStageRunner(repo_root=tmp_path, tracking_service=service)
        assert callable(getattr(runner, "run", None))

    def test_loss_runner_has_run_method(self, tmp_path):
        from tuner.handlers.stages import HFLossStageRunner
        from shared.experiment_tracking import TrackingService

        service = TrackingService(tmp_path)
        runner = HFLossStageRunner(repo_root=tmp_path, tracking_service=service)
        assert callable(getattr(runner, "run", None))


# =========================================================
# ExperimentHandler Integration
# =========================================================

class TestExperimentHandlerImportsStages:
    def test_experiment_handler_still_importable(self):
        """ExperimentHandler should still import from updated module."""
        from tuner.handlers.experiment_handler import ExperimentHandler
        assert ExperimentHandler is not None

    def test_existing_test_imports_still_resolve(self):
        """The pattern used in test_experiment_handler.py should work."""
        from tuner.handlers.stages import (
            HFEvalStageRunner,
            HFLossStageRunner,
            HFTrainingStageRunner,
        )
        # These are the exact imports used by test_experiment_handler.py
        assert HFTrainingStageRunner is not None
        assert HFEvalStageRunner is not None
        assert HFLossStageRunner is not None
