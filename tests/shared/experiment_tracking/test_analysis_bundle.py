from __future__ import annotations

import json
from pathlib import Path

import yaml

import shared.experiment_tracking.analysis_bundle as analysis_bundle
import shared.experiment_tracking.benchmark_ledger as benchmark_ledger
from shared.experiment_tracking.experiment import Experiment
from shared.experiment_tracking.schema import LossResult, RunRecord


def test_write_analysis_bundle_materializes_summary_and_features(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])
    monkeypatch.setattr(
        analysis_bundle,
        "upsert_benchmark_ledger",
        lambda **_: str(tmp_path / "docs" / "benchmarks" / "model_hardware_benchmark_ledger.csv"),
    )

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
        stage_details={
            "training": {"status": "completed", "started_at": "2026-03-21T18:00:00+00:00", "finished_at": "2026-03-21T18:10:00+00:00"},
            "evaluation": {"status": "completed", "started_at": "2026-03-21T18:10:00+00:00", "finished_at": "2026-03-21T18:15:00+00:00"},
            "loss": {"status": "completed", "started_at": "2026-03-21T18:15:00+00:00", "finished_at": "2026-03-21T18:18:00+00:00"},
        },
    )
    training_dir = tmp_path / "hf_train"
    training_dir.mkdir()
    (training_dir / "training_lineage.json").write_text(
        json.dumps(
            {
                "training": {
                    "num_epochs": 2,
                    "max_seq_length": 2048,
                    "batch_size": 8,
                    "gradient_accumulation_steps": 4,
                    "effective_batch_size": 32,
                },
                "results": {"training_time_seconds": 600, "final_loss": 0.9},
            }
        ),
        encoding="utf-8",
    )
    runs = [
        RunRecord(
            run_id="exp-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-21T18:00:00+00:00",
            status="completed",
            output_dir=str(training_dir),
            provider="hf_jobs",
            artifact_root=str(training_dir),
            primary_metric=0.9,
            primary_metric_name="final_loss",
            stage="training",
            hardware="a100-large",
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
            hardware="a100-large",
        ),
        RunRecord(
            run_id="exp-loss",
            run_type="loss",
            name="loss",
            timestamp="2026-03-21T18:15:00+00:00",
            status="completed",
            output_dir="hf://buckets/test/loss",
            provider="hf_jobs",
            artifact_root="hf://buckets/test/loss",
            stage="loss",
            hardware="a100-large",
        ),
    ]
    eval_payload = {
        "summary": {"passed": 2, "failed": 1, "warned": 0, "total": 3, "pass_rate": 2 / 3, "schema_pass_rate": 1.0},
        "records": [
            {"case_id": "ok-1", "passed": True},
            {"case_id": "bad-1", "passed": False, "error": "tool failure"},
        ],
    }
    loss_results = [
        LossResult(index=0, loss=1.5, num_completion_tokens=8, num_total_tokens=16, jsonl_hash="aaaa1111"),
        LossResult(index=1, loss=0.8, num_completion_tokens=10, num_total_tokens=18, jsonl_hash="bbbb2222"),
    ]

    outputs = analysis_bundle.write_analysis_bundle(
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
    assert Path(outputs["draft_next_spec_yaml"]).exists()
    assert outputs["benchmark_ledger_csv"].endswith("model_hardware_benchmark_ledger.csv")

    summary_payload = json.loads(Path(outputs["experiment_summary_json"]).read_text(encoding="utf-8"))
    assert summary_payload["status"] == "completed"
    assert summary_payload["eval_summary"]["failed"] == 1
    assert summary_payload["stage_lineages"]["training_lineage"].endswith("training_lineage.json")

    candidates_payload = json.loads(Path(outputs["next_run_candidates_json"]).read_text(encoding="utf-8"))
    assert candidates_payload["candidates"]
    assert candidates_payload["summary"]["experiment_id"] == experiment.experiment_id
    assert candidates_payload["candidates"][0]["signal"] in {"training_final_loss", "evaluation_failure_rate", "loss_spread"}

    draft_spec_payload = yaml.safe_load(Path(outputs["draft_next_spec_yaml"]).read_text(encoding="utf-8"))
    assert draft_spec_payload["experiment"]["recommendation"]["selected_candidate_rank"] == 1
    assert draft_spec_payload["experiment"]["training"]["model_name"] == experiment.base_model_name
