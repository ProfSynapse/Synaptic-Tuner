import json
import signal
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from Evaluator import cloud_hf_job
from Evaluator.cloud_hf_job import _PeriodicBucketSyncer, _finalize_cloud_exit_code, _install_termination_handler
from shared.cloud_stage_logging import StageLogger


def test_finalize_cloud_exit_code_keeps_success_when_eval_passes(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"
    output_json.write_text("{}", encoding="utf-8")

    assert _finalize_cloud_exit_code(0, output_json) == 0


def test_finalize_cloud_exit_code_converts_quality_failures_when_results_exist(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"
    output_json.write_text("{}", encoding="utf-8")

    assert _finalize_cloud_exit_code(2, output_json) == 0


def test_finalize_cloud_exit_code_converts_exception_failures_when_results_exist(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"
    output_json.write_text("{}", encoding="utf-8")

    assert _finalize_cloud_exit_code(1, output_json) == 0


def test_finalize_cloud_exit_code_preserves_runtime_failure_when_results_missing(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"

    assert _finalize_cloud_exit_code(2, output_json) == 2


def test_periodic_bucket_syncer_emits_stage_sync_event(tmp_path: Path):
    logger = StageLogger(tmp_path / "logs", stage="evaluation", provider="hf_jobs", run_prefix="runs/demo")
    syncer = _PeriodicBucketSyncer(
        tmp_path / "results",
        "bucket-id",
        "runs/demo/evaluations/vllm/ts",
        "hf-token",
        stage_logger=logger,
    )
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    with patch("Evaluator.cloud_hf_job._sync_bucket") as mock_sync:
        syncer.sync_once()

    summary = json.loads((tmp_path / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert mock_sync.call_count == 1
    assert summary["event"] == "artifacts_synced"
    assert summary["details"]["last_sync_path"] == "hf://buckets/bucket-id/runs/demo/evaluations/vllm/ts"


def test_termination_handler_without_progress_syncer_syncs_results(tmp_path: Path):
    logger = StageLogger(tmp_path / "logs", stage="evaluation", provider="hf_jobs", run_prefix="runs/demo")
    handlers = {}

    with patch("Evaluator.cloud_hf_job.signal.signal", side_effect=lambda sig, handler: handlers.setdefault(sig, handler)):
        _install_termination_handler(
            stage_logger=logger,
            results_dir=tmp_path / "results",
            bucket_id="bucket-id",
            eval_prefix="runs/demo/evaluations/unsloth/ts",
            token="hf-token",
        )

    with patch("Evaluator.cloud_hf_job._sync_bucket") as mock_sync:
        with pytest.raises(SystemExit) as exc_info:
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert exc_info.value.code == 128 + signal.SIGTERM
    mock_sync.assert_called_once_with(
        str(tmp_path / "results"),
        "hf://buckets/bucket-id/runs/demo/evaluations/unsloth/ts",
        token="hf-token",
    )
    summary = json.loads((tmp_path / "logs" / "stage_summary.json").read_text(encoding="utf-8"))
    assert summary["event"] == "terminated"


def test_main_installs_termination_handler_before_initial_download(tmp_path: Path):
    args = Namespace(
        bucket_id="bucket-id",
        run_prefix="runs/demo",
        eval_prefix="runs/demo/evaluations/unsloth/ts",
        config_dir="Evaluator/config",
        output_root=str(tmp_path / "eval_outputs"),
        preset=None,
        scenarios=None,
        tags=None,
        env_backend="none",
        env_template=None,
        env_tool_schema=None,
        env_exec_config=None,
        upload_to_hf=None,
        update_model_card=False,
        with_loss=False,
        loss_dataset_path=None,
        loss_dataset_name=None,
        loss_dataset_file=None,
        loss_max_seq_length=2048,
        loss_no_completion_only=False,
    )
    installed = []

    def _fake_install(**kwargs):
        installed.append(kwargs.get("progress_syncer"))

    def _fake_sync_from_bucket(*_args, **_kwargs):
        assert installed == [None]
        raise RuntimeError("stop after bootstrap ordering check")

    with patch.object(cloud_hf_job, "_parse_args", return_value=args), patch.object(
        cloud_hf_job, "get_hf_token", return_value="hf-token"
    ), patch.object(cloud_hf_job, "_log_runtime_versions"), patch.object(
        cloud_hf_job, "_install_termination_handler", side_effect=_fake_install
    ), patch.object(
        cloud_hf_job, "_sync_from_bucket", side_effect=_fake_sync_from_bucket
    ):
        with pytest.raises(RuntimeError, match="stop after bootstrap ordering check"):
            cloud_hf_job.main()
