from pathlib import Path

from Evaluator.cloud_hf_job import _finalize_cloud_exit_code


def test_finalize_cloud_exit_code_keeps_success_when_eval_passes(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"
    output_json.write_text("{}", encoding="utf-8")

    assert _finalize_cloud_exit_code(0, output_json) == 0


def test_finalize_cloud_exit_code_converts_quality_failures_when_results_exist(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"
    output_json.write_text("{}", encoding="utf-8")

    assert _finalize_cloud_exit_code(2, output_json) == 0


def test_finalize_cloud_exit_code_preserves_runtime_failure_when_results_missing(tmp_path: Path):
    output_json = tmp_path / "evaluation_results.json"

    assert _finalize_cloud_exit_code(2, output_json) == 2
