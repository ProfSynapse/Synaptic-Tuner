import json
from pathlib import Path
from unittest.mock import patch

from Evaluator.cloud_hf_job import _PeriodicBucketSyncer, _finalize_cloud_exit_code
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
