"""
Tests for HFJobsBackend mixin decomposition.

Verifies:
- MRO resolution and mixin composition
- Cross-mixin method availability
- ITrainingBackend interface compliance
- Individual mixin isolation (each importable independently)
- Bucket operations edge cases
- Command builder edge cases
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend
from tuner.backends.training.cloud._hf_command_builder import HFCommandBuilderMixin
from tuner.backends.training.cloud._hf_job_watcher import HFJobWatcherMixin
from tuner.backends.training.cloud._hf_bucket_ops import HFBucketOpsMixin
from tuner.backends.training.cloud._hf_post_training import HFPostTrainingMixin
from tuner.backends.training.base import ITrainingBackend
from tuner.core.config import CloudTrainingConfig
from tuner.core.exceptions import CloudProviderError


# =========================================================
# MRO and Composition
# =========================================================

class TestMixinComposition:
    def test_hf_jobs_backend_mro_includes_all_mixins(self):
        """All 4 mixins and ITrainingBackend must appear in the MRO."""
        mro = HFJobsBackend.__mro__
        assert HFCommandBuilderMixin in mro
        assert HFJobWatcherMixin in mro
        assert HFBucketOpsMixin in mro
        assert HFPostTrainingMixin in mro
        assert ITrainingBackend in mro

    def test_hf_jobs_backend_mro_order(self):
        """Mixins should precede ITrainingBackend in MRO (left-to-right)."""
        mro = HFJobsBackend.__mro__
        mixin_indices = [
            mro.index(HFCommandBuilderMixin),
            mro.index(HFJobWatcherMixin),
            mro.index(HFBucketOpsMixin),
            mro.index(HFPostTrainingMixin),
        ]
        interface_index = mro.index(ITrainingBackend)
        for idx in mixin_indices:
            assert idx < interface_index, (
                f"Mixin at index {idx} should precede ITrainingBackend at {interface_index}"
            )

    def test_hf_jobs_backend_is_instance_of_all_bases(self, repo_root):
        """Instance should be recognized as all base types."""
        backend = HFJobsBackend(repo_root)
        assert isinstance(backend, HFCommandBuilderMixin)
        assert isinstance(backend, HFJobWatcherMixin)
        assert isinstance(backend, HFBucketOpsMixin)
        assert isinstance(backend, HFPostTrainingMixin)
        assert isinstance(backend, ITrainingBackend)


# =========================================================
# ITrainingBackend Interface Compliance
# =========================================================

class TestITrainingBackendCompliance:
    def test_has_name_property(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert hasattr(backend, "name")
        assert isinstance(backend.name, str)

    def test_has_get_available_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        methods = backend.get_available_methods()
        assert isinstance(methods, list)
        assert all(isinstance(m, str) for m in methods)

    def test_has_validate_environment(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(backend.validate_environment)

    def test_has_load_config(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(backend.load_config)

    def test_has_execute(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(backend.execute)


# =========================================================
# Individual Mixin Isolation
# =========================================================

class TestMixinIsolation:
    def test_command_builder_importable_independently(self):
        """HFCommandBuilderMixin should be importable on its own."""
        assert HFCommandBuilderMixin is not None
        assert hasattr(HFCommandBuilderMixin, "_build_training_command")

    def test_job_watcher_importable_independently(self):
        assert HFJobWatcherMixin is not None
        assert hasattr(HFJobWatcherMixin, "_watch_job_with_remote_dashboard")

    def test_bucket_ops_importable_independently(self):
        assert HFBucketOpsMixin is not None
        assert hasattr(HFBucketOpsMixin, "_ensure_hf_bucket")
        assert hasattr(HFBucketOpsMixin, "_build_remote_run_uri")
        assert hasattr(HFBucketOpsMixin, "_sync_bucket_path")
        assert hasattr(HFBucketOpsMixin, "_run_dir_has_completion_artifacts")
        assert hasattr(HFBucketOpsMixin, "_download_completed_run")
        assert hasattr(HFBucketOpsMixin, "_recover_completed_run_from_bucket")

    def test_post_training_importable_independently(self):
        assert HFPostTrainingMixin is not None
        assert hasattr(HFPostTrainingMixin, "_print_completion_summary")
        assert hasattr(HFPostTrainingMixin, "_handle_post_training_actions")


# =========================================================
# Cross-Mixin Method Access
# =========================================================

class TestCrossMixinAccess:
    """Verify that composed HFJobsBackend can access methods from all mixins."""

    def test_backend_has_command_builder_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(getattr(backend, "_build_training_command", None))
        assert callable(getattr(backend, "_build_artifact_prefix", None))

    def test_backend_has_watcher_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(getattr(backend, "_watch_job_with_remote_dashboard", None))
        assert callable(getattr(backend, "_should_use_remote_dashboard", None))

    def test_backend_has_bucket_ops_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(getattr(backend, "_ensure_hf_bucket", None))
        assert callable(getattr(backend, "_build_remote_run_uri", None))
        assert callable(getattr(backend, "_local_download_run_dir", None))
        assert callable(getattr(backend, "_sync_bucket_path", None))
        assert callable(getattr(backend, "_run_dir_has_completion_artifacts", None))
        assert callable(getattr(backend, "_download_completed_run", None))
        assert callable(getattr(backend, "_recover_completed_run_from_bucket", None))

    def test_backend_has_post_training_methods(self, repo_root):
        backend = HFJobsBackend(repo_root)
        assert callable(getattr(backend, "_print_completion_summary", None))
        assert callable(getattr(backend, "_handle_post_training_actions", None))


# =========================================================
# BucketOps Edge Cases
# =========================================================

class TestBucketOpsEdgeCases:
    def _cloud_config(self, **overrides):
        from tuner.backends.training.cloud.hf_jobs_backend import DEFAULT_IMAGE
        config = CloudTrainingConfig(
            method="sft",
            platform="hf_jobs",
            config_path=Path("/fake"),
            trainer_dir=Path("/fake"),
            model_name="test",
            dataset_file="test",
            epochs=1,
            batch_size=4,
            learning_rate=2e-4,
            provider="hf_jobs",
            gpu_type="a10g-small",
            timeout_hours=4.0,
            cloud_image=DEFAULT_IMAGE,
            hf_flavor="a10g-small",
            artifact_backend="hf_bucket",
            artifact_identifier="user/bucket",
            artifact_mount_path="/workspace/outputs",
            repo_url="https://github.com/test/repo.git",
            repo_branch="main",
            repo_commit="abc12345def67890",
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_build_remote_run_uri_strips_trailing_slashes(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._cloud_config()
        uri = backend._build_remote_run_uri(config, "runs/sft/20260321/")
        assert not uri.endswith("/")
        assert "runs/sft/20260321" in uri

    def test_build_remote_run_uri_strips_leading_slashes(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._cloud_config()
        uri = backend._build_remote_run_uri(config, "/runs/sft/20260321")
        assert "//runs" not in uri

    def test_build_remote_run_uri_raises_without_identifier(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._cloud_config(artifact_identifier="")
        with pytest.raises(CloudProviderError, match="artifact bucket"):
            backend._build_remote_run_uri(config, "runs/sft/20260321")

    def test_ensure_hf_bucket_raises_without_identifier(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._cloud_config(artifact_identifier="")
        mock_hub = MagicMock()
        with pytest.raises(CloudProviderError, match="artifact bucket"):
            backend._ensure_hf_bucket(config, mock_hub)

    def test_local_download_run_dir_extracts_slug(self, repo_root):
        backend = HFJobsBackend(repo_root)
        config = self._cloud_config()
        run_dir = backend._local_download_run_dir(config, "runs/hf_jobs/sft/20260321-deadbeef")
        assert run_dir.name == "20260321-deadbeef"

    def test_run_dir_has_completion_artifacts_requires_final_model(self, tmp_path):
        """Completion detection requires final_model directory with contents."""
        backend = HFJobsBackend(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Empty dir: not complete
        assert not backend._run_dir_has_completion_artifacts(run_dir)

        # final_model dir but empty: not complete
        (run_dir / "final_model").mkdir()
        assert not backend._run_dir_has_completion_artifacts(run_dir)

        # final_model with file + lineage: complete
        (run_dir / "final_model" / "model.safetensors").touch()
        (run_dir / "training_lineage.json").touch()
        assert backend._run_dir_has_completion_artifacts(run_dir)

    def test_run_dir_has_completion_artifacts_accepts_grpo_logs(self, tmp_path):
        """GRPO runs complete with logs/training_latest.jsonl instead of lineage."""
        backend = HFJobsBackend(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "final_model").mkdir()
        (run_dir / "final_model" / "weights.bin").touch()
        (run_dir / "logs").mkdir()
        (run_dir / "logs" / "training_latest.jsonl").touch()
        assert backend._run_dir_has_completion_artifacts(run_dir)


# =========================================================
# Backward-Compatible Imports
# =========================================================

class TestBackwardCompatImports:
    def test_hf_jobs_backend_importable_from_cloud_package(self):
        """HFJobsBackend should still be importable from __init__."""
        from tuner.backends.training.cloud import HFJobsBackend as Imported
        assert Imported is HFJobsBackend

    def test_module_level_constants_still_accessible(self):
        """DEFAULT_FLAVOR, DEFAULT_TIMEOUT, DEFAULT_IMAGE, _parse_timeout remain accessible."""
        from tuner.backends.training.cloud.hf_jobs_backend import (
            DEFAULT_FLAVOR,
            DEFAULT_IMAGE,
            DEFAULT_TIMEOUT,
            _parse_timeout,
        )
        assert isinstance(DEFAULT_FLAVOR, str)
        assert isinstance(DEFAULT_IMAGE, str)
        assert isinstance(DEFAULT_TIMEOUT, str)
        assert callable(_parse_timeout)
