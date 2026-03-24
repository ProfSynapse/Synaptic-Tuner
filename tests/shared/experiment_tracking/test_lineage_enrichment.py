from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import shared.experiment_tracking.lineage_enrichment as lineage_enrichment
from shared.experiment_tracking.schema import LossResult


def test_build_loss_lineage_includes_runtime_pricing_and_distribution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        lineage_enrichment,
        "load_live_hf_hardware_rows",
        lambda: [SimpleNamespace(flavor="a100-large", price_hr=2.5)],
    )

    output_root = tmp_path / "loss"
    output_root.mkdir(parents=True)
    (output_root / "loss_summary.json").write_text(
        '{"rows_written": 2, "batch_count": 1, "worker_count": 2, "completion_tokens": 8, "total_tokens": 14}',
        encoding="utf-8",
    )

    payload = lineage_enrichment.build_loss_lineage(
        dataset_path="repo/dataset_variant.jsonl",
        output_root=output_root,
        loss_results=[
            LossResult(index=0, loss=1.0, num_completion_tokens=4, num_total_tokens=7, jsonl_hash="aaaa1111"),
            LossResult(index=1, loss=0.5, num_completion_tokens=4, num_total_tokens=7, jsonl_hash="bbbb2222"),
        ],
        completion_only=True,
        max_seq_length=2048,
        batch_max_tokens=4096,
        adaptive_batching=True,
        runtime_backend="transformers",
        hardware_flavor="a100-large",
        worker_count=2,
        started_at="2026-03-22T20:00:00+00:00",
        finished_at="2026-03-22T20:01:00+00:00",
    )

    assert payload["stage"] == "loss"
    assert payload["dataset"]["variant"] == "dataset_variant"
    assert payload["runtime"]["duration_seconds"] == 60.0
    assert payload["pricing"]["price_hour_usd"] == 2.5
    assert payload["pricing"]["estimated_cost_usd"] == 2.5 / 60.0
    assert payload["execution"]["worker_count"] == 2
    assert payload["results"]["row_count"] == 2
    assert payload["results"]["median_loss"] == 0.75
    assert payload["results"]["p95_loss"] == 1.0


def test_enrich_training_lineage_adds_variant_and_planner(tmp_path: Path) -> None:
    args = SimpleNamespace(auto_hardware=True, optimize_for="cost", hf_flavor="a100-large")
    payload = lineage_enrichment.enrich_training_lineage(
        {
            "timestamp": "2026-03-22T20:00:00+00:00",
            "dataset": {"source": "repo/my_dataset.jsonl"},
            "training": {"effective_batch_size": 32},
            "results": {"training_time_seconds": 120.0},
            "hardware": {},
        },
        args=args,
    )

    assert payload["stage"] == "training"
    assert payload["dataset"]["variant"] == "my_dataset"
    assert payload["planner"]["auto_hardware"] is True
    assert payload["planner"]["resolved_hardware_flavor"] == "a100-large"
    assert payload["runtime"]["duration_seconds"] == 120.0


def test_enrich_training_lineage_preserves_evolutionary_summary() -> None:
    payload = lineage_enrichment.enrich_training_lineage(
        {
            "timestamp": "2026-03-24T16:20:35+00:00",
            "dataset": {"source": "repo/my_dataset.jsonl"},
            "training": {"effective_batch_size": 48},
            "results": {"training_time_seconds": 195.1},
            "hardware": {},
            "evolutionary": {
                "enabled": True,
                "selection_events": 2,
                "baseline_kept_count": 1,
                "events_path": "/tmp/run/logs/evolutionary_events.jsonl",
            },
        },
        args=SimpleNamespace(auto_hardware=False, optimize_for=None, hf_flavor="a100-large"),
    )

    assert payload["evolutionary"]["enabled"] is True
    assert payload["evolutionary"]["selection_events"] == 2
    assert payload["evolutionary"]["baseline_kept_count"] == 1
    assert payload["evolutionary"]["events_path"].endswith("evolutionary_events.jsonl")


def test_enrich_evaluation_lineage_adds_execution_and_runtime() -> None:
    payload = lineage_enrichment.enrich_evaluation_lineage(
        {"results_summary": {"overall_pass_rate": 55.0}},
        backend="vllm",
        hardware_flavor="a100x4",
        started_at="2026-03-22T20:00:00+00:00",
        finished_at="2026-03-22T20:10:00+00:00",
        tensor_parallel_size=4,
        worker_count=4,
        fallback_reason="bnb-4bit forced single-gpu eval",
    )

    assert payload["stage"] == "evaluation"
    assert payload["runtime"]["duration_seconds"] == 600.0
    assert payload["execution"]["hardware_flavor"] == "a100x4"
    assert payload["execution"]["tensor_parallel_size"] == 4
    assert payload["execution"]["fallback_reason"]
