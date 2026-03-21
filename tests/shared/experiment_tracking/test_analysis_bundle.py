from __future__ import annotations

import json
from pathlib import Path

from shared.experiment_tracking.analysis_bundle import write_analysis_bundle
from shared.experiment_tracking.experiment import Experiment
from shared.experiment_tracking.schema import LossResult, RunRecord


def test_write_analysis_bundle_materializes_summary_and_features(tmp_path: Path):
    experiment = Experiment(
        experiment_id="exp_20260321_180000",
        name="smoke",
        created_at="2026-03-21T18:00:00+00:00",
        dataset_path="repo/dataset.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        training_run_id="exp-training",
        evaluation_run_id="exp-eval",
        loss_run_id="exp-loss",
        status="completed",
        stage_statuses={"training": "completed", "evaluation": "completed", "loss": "completed"},
    )
    runs = [
        RunRecord(
            run_id="exp-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-21T18:00:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/train",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/train",
            primary_metric=0.9,
            primary_metric_name="final_loss",
            stage="training",
        ),
        RunRecord(
            run_id="exp-eval",
            run_type="evaluation",
            name="eval",
            timestamp="2026-03-21T18:05:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/eval",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/eval",
            stage="evaluation",
        ),
    ]
    eval_payload = {
        "summary": {"passed": 2, "failed": 1, "warned": 0, "total": 3},
        "records": [
            {"case_id": "ok-1", "passed": True},
            {"case_id": "bad-1", "passed": False, "error": "tool failure"},
        ],
    }
    loss_results = [
        LossResult(index=0, loss=1.5, num_completion_tokens=8, num_total_tokens=16, jsonl_hash="aaaa1111"),
        LossResult(index=1, loss=0.8, num_completion_tokens=10, num_total_tokens=18, jsonl_hash="bbbb2222"),
    ]

    outputs = write_analysis_bundle(
        experiment=experiment,
        runs=runs,
        base_dir=tmp_path,
        eval_payload=eval_payload,
        loss_results=loss_results,
    )

    assert Path(outputs["experiment_summary_json"]).exists()
    assert Path(outputs["run_matrix_csv"]).exists()
    assert Path(outputs["feature_dataset_csv"]).exists()
    assert Path(outputs["hypothesis_context_json"]).exists()

    summary_payload = json.loads(Path(outputs["experiment_summary_json"]).read_text(encoding="utf-8"))
    assert summary_payload["status"] == "completed"
    assert summary_payload["eval_summary"]["failed"] == 1

    candidates_payload = json.loads(Path(outputs["next_run_candidates_json"]).read_text(encoding="utf-8"))
    assert candidates_payload["candidates"]


