from __future__ import annotations

from argparse import Namespace

from shared.experiment_tracking.experiment_spec import (
    DatasetSpec,
    EvaluationStageSpec,
    ExperimentSpec,
    LossStageSpec,
    TrainingStageSpec,
)
from tuner.cloud.hardware_planner import (
    HardwareFlavor,
    StageEstimate,
    StagePlan,
    _rank_estimates,
    normalize_hardware_rows,
    parse_model_params_billions,
)
from tuner.handlers.experiment_handler import ExperimentHandler


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="hardware-plan-smoke",
        provider="hf_jobs",
        method="sft",
        dataset=DatasetSpec(source="repo/dataset", file="train.jsonl"),
        training=TrainingStageSpec(
            model_name="Qwen/Qwen3-4B",
            load_in_4bit=True,
            max_seq_length=2048,
        ),
        evaluation=EvaluationStageSpec(enabled=True, preset="quick", runtime="vllm"),
        loss=LossStageSpec(enabled=True, max_seq_length=2048),
    )


def test_normalize_hardware_rows_parses_nested_hf_payload() -> None:
    payload = [
        {
            "accelerator": {
                "manufacturer": "Nvidia",
                "model": "A10G",
                "quantity": "1",
                "type": "gpu",
                "vram": "24 GB",
            },
            "cpu": "4 vCPU",
            "name": "a10g-small",
            "prettyName": "Nvidia A10G - small",
            "ram": "15 GB",
            "unitCostUSD": 0.016667,
            "unitLabel": "minute",
        }
    ]

    rows = normalize_hardware_rows(payload)

    assert len(rows) == 1
    assert rows[0].flavor == "a10g-small"
    assert rows[0].gpu_model == "A10G"
    assert rows[0].gpus == 1
    assert rows[0].vram_gb == 24.0
    assert rows[0].ram_gb == 15.0
    assert rows[0].cpu == 4.0
    assert round(rows[0].price_hr, 2) == 1.00


def test_parse_model_params_billions_handles_common_model_names() -> None:
    assert parse_model_params_billions("Qwen/Qwen3-4B") == 4.0
    assert parse_model_params_billions("HuggingFaceTB/SmolLM2-1.7B-Instruct") == 1.7
    assert parse_model_params_billions("phi-410m") == 0.41


def test_rank_estimates_balanced_prefers_cheapest_when_speed_gain_is_not_large_enough() -> None:
    rows = [
        StageEstimate(
            stage="training",
            flavor="a10g-small",
            pretty_name="A10G",
            feasible=True,
            reason="fits",
            gpu_model="A10G",
            vram_gb=24.0,
            price_hr=1.0,
            recommended_batch_size=4,
            recommended_gradient_accumulation=8,
            estimated_memory_gb=20.0,
            estimated_headroom_gb=1.5,
            throughput_score=1.0,
            score_per_dollar=1.0,
            estimated_hours=1.0,
            estimated_cost=1.0,
        ),
        StageEstimate(
            stage="training",
            flavor="a100-large",
            pretty_name="A100",
            feasible=True,
            reason="fits",
            gpu_model="A100",
            vram_gb=80.0,
            price_hr=2.5,
            recommended_batch_size=8,
            recommended_gradient_accumulation=4,
            estimated_memory_gb=30.0,
            estimated_headroom_gb=40.0,
            throughput_score=2.6,
            score_per_dollar=1.04,
            estimated_hours=0.45,
            estimated_cost=1.13,
        ),
    ]

    ranked = _rank_estimates(rows, "balanced")
    assert ranked[0].flavor == "a10g-small"


def test_experiment_handler_auto_hardware_populates_missing_stage_gpus(monkeypatch) -> None:
    handler = ExperimentHandler(args=Namespace(optimize_for="balanced", max_hourly_price=None))
    spec = _spec()

    plans = {
        "training": StagePlan(
            stage="training",
            optimize_for="balanced",
            model_name=spec.training.model_name,
            rows=[
                StageEstimate(
                    stage="training",
                    flavor="a10g-small",
                    pretty_name="A10G",
                    feasible=True,
                    reason="fits",
                    gpu_model="A10G",
                    vram_gb=24.0,
                    price_hr=1.0,
                    recommended_batch_size=5,
                    recommended_gradient_accumulation=6,
                    estimated_memory_gb=22.0,
                    estimated_headroom_gb=0.5,
                    throughput_score=1.0,
                    score_per_dollar=1.0,
                    estimated_hours=None,
                    estimated_cost=None,
                )
            ],
        ),
        "evaluation": StagePlan(
            stage="evaluation",
            optimize_for="balanced",
            model_name=spec.training.model_name,
            rows=[
                StageEstimate(
                    stage="evaluation",
                    flavor="l4x1",
                    pretty_name="L4",
                    feasible=True,
                    reason="fits",
                    gpu_model="L4",
                    vram_gb=24.0,
                    price_hr=0.8,
                    recommended_batch_size=None,
                    recommended_gradient_accumulation=None,
                    estimated_memory_gb=9.0,
                    estimated_headroom_gb=11.0,
                    throughput_score=1.0,
                    score_per_dollar=1.0,
                    estimated_hours=None,
                    estimated_cost=None,
                )
            ],
        ),
        "loss": StagePlan(
            stage="loss",
            optimize_for="balanced",
            model_name=spec.training.model_name,
            rows=[
                StageEstimate(
                    stage="loss",
                    flavor="a10g-small",
                    pretty_name="A10G",
                    feasible=True,
                    reason="fits",
                    gpu_model="A10G",
                    vram_gb=24.0,
                    price_hr=1.0,
                    recommended_batch_size=None,
                    recommended_gradient_accumulation=None,
                    estimated_memory_gb=15.0,
                    estimated_headroom_gb=5.0,
                    throughput_score=1.0,
                    score_per_dollar=1.0,
                    estimated_hours=None,
                    estimated_cost=None,
                )
            ],
        ),
    }

    monkeypatch.setattr("tuner.handlers.experiment_handler.plan_experiment_hardware", lambda **kwargs: plans)

    updated_spec, returned_plans = handler._apply_auto_hardware(spec)

    assert updated_spec.training.gpu == "a10g-small"
    assert updated_spec.training.batch_size == 5
    assert updated_spec.training.gradient_accumulation == 6
    assert updated_spec.evaluation.gpu == "l4x1"
    assert updated_spec.loss.gpu == "a10g-small"
    assert returned_plans is plans
