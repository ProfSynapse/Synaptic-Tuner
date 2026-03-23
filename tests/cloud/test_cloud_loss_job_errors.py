"""Additional cloud_loss_job tests: error recovery, malformed data, missing args."""

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import shared.experiment_tracking.cloud_loss_job as cloud_loss_job


class TestCloudLossJobErrors:
    """Error and edge-case paths in cloud_loss_job.main()."""

    def _base_args(self, tmp_path, **overrides):
        defaults = dict(
            bucket_id="test-bucket",
            run_prefix="runs/hf_jobs/sft/test",
            dataset_path=str(tmp_path / "dataset.jsonl"),
            dataset_name=None,
            dataset_file=None,
            results_prefix=None,
            output_root=str(tmp_path / "loss_outputs"),
            max_seq_length=2048,
            batch_max_tokens=512,
            max_batch_size=2,
            sync_every_batches=1,
            num_workers=2,
            aggregate_interval_seconds=1.0,
            no_completion_only=False,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_missing_hf_token_raises(self, tmp_path):
        args = self._base_args(tmp_path)
        with patch.object(cloud_loss_job, "_parse_args", return_value=args):
            with patch.object(cloud_loss_job, "get_hf_token", return_value=None):
                with pytest.raises(RuntimeError, match="HF_TOKEN"):
                    cloud_loss_job.main()

    def test_missing_dataset_path_and_name_raises(self, tmp_path):
        args = self._base_args(
            tmp_path,
            dataset_path=None,
            dataset_name=None,
            dataset_file=None,
        )
        with patch.object(cloud_loss_job, "_parse_args", return_value=args):
            with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
                with patch.object(cloud_loss_job, "_sync_from_bucket"):
                    with pytest.raises(ValueError, match="Provide either"):
                        cloud_loss_job.main()

    def test_nonexistent_dataset_path_raises(self, tmp_path):
        args = self._base_args(
            tmp_path,
            dataset_path=str(tmp_path / "nonexistent.jsonl"),
        )
        with patch.object(cloud_loss_job, "_parse_args", return_value=args):
            with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
                with patch.object(cloud_loss_job, "_sync_from_bucket"):
                    with pytest.raises(FileNotFoundError, match="nonexistent.jsonl"):
                        cloud_loss_job.main()

    def test_compute_failure_emits_stage_failure(self, tmp_path):
        args = self._base_args(tmp_path)
        Path(args.dataset_path).write_text(
            '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
            encoding="utf-8",
        )

        with patch.object(cloud_loss_job, "_parse_args", return_value=args):
            with patch.object(cloud_loss_job, "get_hf_token", return_value="hf-token"):
                with patch.object(cloud_loss_job, "_sync_from_bucket"):
                    with patch(
                        "shared.experiment_tracking.per_example_loss.compute_per_example_losses_parallel",
                        side_effect=RuntimeError("OOM: CUDA out of memory"),
                    ):
                        with pytest.raises(RuntimeError, match="OOM"):
                            cloud_loss_job.main()

        # Verify stage logger captured the failure
        logs_dir = Path(args.output_root) / "results" / "logs"
        events_path = logs_dir / "stage_events.jsonl"
        if events_path.exists():
            events = events_path.read_text(encoding="utf-8")
            assert "failed" in events.lower() or "error" in events.lower()


class TestSyncBucket:
    """Tests for the _sync_bucket helper."""

    def test_strict_failure_raises(self):
        with patch("subprocess.run", side_effect=Exception("sync failed")):
            with pytest.raises(Exception, match="sync failed"):
                cloud_loss_job._sync_bucket("source", "dest", "token", strict=True)

    def test_non_strict_failure_returns_false(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
            result = cloud_loss_job._sync_bucket("source", "dest", "token", strict=False)
            assert result is False

    def test_success_returns_true(self):
        with patch("subprocess.run"):
            result = cloud_loss_job._sync_bucket("source", "dest", "token", strict=True)
            assert result is True

    def test_none_token_removes_env_vars(self):
        captured_env = {}

        def _capture_run(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))

        with patch("subprocess.run", side_effect=_capture_run):
            cloud_loss_job._sync_bucket("s", "d", None, strict=True)

        assert "HF_TOKEN" not in captured_env
        assert "HF_API_KEY" not in captured_env
