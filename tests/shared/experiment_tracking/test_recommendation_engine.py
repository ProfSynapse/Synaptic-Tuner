from __future__ import annotations

from shared.experiment_tracking.experiment import Experiment
from shared.experiment_tracking.recommendation_engine import build_recommendation_bundle, render_draft_next_spec
from shared.experiment_tracking.schema import LossResult, RunRecord


def test_build_recommendation_bundle_emits_richer_candidates_and_draft_spec():
    experiment = Experiment(
        experiment_id="exp_20260321_180000",
        name="smoke",
        created_at="2026-03-21T18:00:00+00:00",
        dataset_path="repo/dataset/sample.jsonl",
        dataset_hash="abc123",
        base_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        provider="hf_jobs",
        method="sft",
        objective="train_eval_loss_smoke",
        training_run_id="exp-training",
        evaluation_run_id="exp-eval",
        loss_run_id="exp-loss",
        status="completed",
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
            primary_metric=1.25,
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
        "summary": {"passed": 1, "failed": 2, "warned": 0, "total": 3},
        "records": [{"case_id": "bad-1", "passed": False}],
    }
    loss_results = [
        LossResult(index=0, loss=1.9, num_completion_tokens=8, num_total_tokens=16, jsonl_hash="aaaa1111"),
        LossResult(index=1, loss=0.9, num_completion_tokens=10, num_total_tokens=18, jsonl_hash="bbbb2222"),
    ]

    bundle = build_recommendation_bundle(
        experiment=experiment,
        runs=runs,
        eval_payload=eval_payload,
        loss_results=loss_results,
    )

    assert bundle["candidates"]
    assert len(bundle["candidates"]) >= 3
    assert bundle["candidates"][0]["signal"] == "training_final_loss"
    assert bundle["candidates"][0]["suggested_changes"]["train_learning_rate"]["operator"] == "multiply"
    assert bundle["draft_next_spec"]["experiment"]["recommendation"]["selected_candidate_rank"] == 1
    assert bundle["draft_next_spec"]["experiment"]["recommendation"]["candidates"][0]["rank"] == 1

    rendered = render_draft_next_spec(bundle["draft_next_spec"])
    assert "smoke-next" in rendered
