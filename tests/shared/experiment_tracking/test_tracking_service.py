from __future__ import annotations

import json
from pathlib import Path

from shared.experiment_tracking import TrackingService
from shared.experiment_tracking.experiment import Experiment, save_experiment
from shared.experiment_tracking.schema import RunRecord


def test_tracking_service_creates_and_updates_experiment(tmp_path: Path):
    service = TrackingService(tmp_path)
    experiment = service.create_experiment(
        name="smoke",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
    )

    record = RunRecord(
        run_id="exp-training",
        run_type="sft",
        name="training",
        timestamp="2026-03-21T18:00:00+00:00",
        status="completed",
        output_dir="hf://buckets/test/runs/hf_jobs/sft/123",
        provider="hf_jobs",
        artifact_root="hf://buckets/test/runs/hf_jobs/sft/123",
        stage="training",
    )

    run_id = service.attach_run(experiment, record, role="training")
    service.mark_stage(experiment, "training", "completed")
    service.set_artifact_root(experiment, "training", record.artifact_root or "")
    service.set_derived_output(experiment, "feature_dataset_csv", "/tmp/features.csv")
    service.set_derived_output(experiment, "hypothesis_context_json", "/tmp/hypothesis.json")

    reloaded = service.load_experiment(experiment.experiment_id)
    assert run_id == "exp-training"
    assert reloaded.training_run_id == "exp-training"
    assert reloaded.stage_statuses["training"] == "completed"
    assert reloaded.artifact_roots["training"] == "hf://buckets/test/runs/hf_jobs/sft/123"
    assert reloaded.features_csv_path == "/tmp/features.csv"
    assert reloaded.hypothesis_context_path == "/tmp/hypothesis.json"

    registry_lines = (tmp_path / "registry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(registry_lines) == 1
    assert json.loads(registry_lines[0])["experiment_id"] == experiment.experiment_id


def test_tracking_service_finds_latest_recoverable_experiment(tmp_path: Path):
    service = TrackingService(tmp_path)

    older = Experiment(
        experiment_id="exp_older",
        name="smoke",
        created_at="2026-03-21T18:00:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
        status="partial",
    )
    save_experiment(older, tmp_path)

    newer = Experiment(
        experiment_id="exp_newer",
        name="smoke",
        created_at="2026-03-21T18:05:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/spec.yaml",
        status="partial",
    )
    save_experiment(newer, tmp_path)

    completed = Experiment(
        experiment_id="exp_done",
        name="done",
        created_at="2026-03-21T18:10:00+00:00",
        dataset_path="repo/data.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        spec_path="/tmp/other.yaml",
        status="completed",
    )
    save_experiment(completed, tmp_path)

    recovered = service.find_recoverable_experiment(
        spec_path="/tmp/spec.yaml",
        provider="hf_jobs",
        method="sft",
    )

    assert recovered is not None
    assert recovered.experiment_id == newer.experiment_id
