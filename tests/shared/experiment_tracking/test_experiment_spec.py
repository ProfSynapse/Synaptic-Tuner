from __future__ import annotations

import textwrap

import pytest

from shared.experiment_tracking import load_experiment_spec


def test_load_experiment_spec_round_trip(tmp_path):
    spec_path = tmp_path / "experiment.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: smoke-smollm2
              provider: hf_jobs
              method: sft
              objective: train_eval_loss_smoke
              dataset:
                source: professorsynapse/claudesidian-synthetic-dataset
                file: sample.jsonl
              training:
                model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
                gpu: a10g-small
                max_steps: 20
              execution:
                from_stage: evaluation
              evaluation:
                preset: quick
                runtime: vllm
                image_profile: fast_vllm
              loss:
                enabled: true
            """
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)

    assert spec.name == "smoke-smollm2"
    assert spec.provider == "hf_jobs"
    assert spec.method == "sft"
    assert spec.dataset.identifier == "professorsynapse/claudesidian-synthetic-dataset/sample.jsonl"
    assert spec.training.max_steps == 20
    assert spec.execution.from_stage == "evaluation"
    assert spec.evaluation.preset == "quick"
    assert spec.evaluation.runtime == "vllm"
    assert spec.evaluation.image_profile == "fast_vllm"
    assert spec.loss.enabled is True
    assert spec.execution.selected_stages() == ["evaluation", "loss", "analysis", "recommendation"]


def test_load_experiment_spec_rejects_invalid_provider(tmp_path):
    spec_path = tmp_path / "bad_experiment.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: bad
              provider: unsupported
              method: sft
              dataset:
                source: repo/dataset
                file: sample.jsonl
              training:
                model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported provider"):
        load_experiment_spec(spec_path)


def test_load_experiment_spec_rejects_empty_execution_selection(tmp_path):
    spec_path = tmp_path / "empty_execution.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: bad
              provider: hf_jobs
              method: sft
              dataset:
                source: repo/dataset
                file: sample.jsonl
              training:
                model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
              execution:
                only_stage: loss
                skip_stages: [loss]
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution stage selection resolves to an empty set"):
        load_experiment_spec(spec_path)


def test_load_experiment_spec_supports_parallel_post_training_mode(tmp_path):
    spec_path = tmp_path / "parallel_post_training.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: parallel-post-training
              provider: hf_jobs
              method: sft
              dataset:
                source: repo/dataset
                file: sample.jsonl
              training:
                model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
              post_training:
                mode: parallel
            """
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)

    assert spec.post_training.mode == "parallel"


def test_load_experiment_spec_supports_nested_evolutionary_training(tmp_path):
    spec_path = tmp_path / "evolutionary.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: evolutionary-smoke
              provider: hf_jobs
              method: sft
              dataset:
                source: repo/dataset
                file: sample.jsonl
              training:
                model_name: Qwen/Qwen3-4B
                evolutionary:
                  enabled: true
                  candidates: 4
                  eval_batch_size: 2
                  validation_config: configs/fitness/tool_calling.yaml
                  strategy:
                    type: gradient_noise
                    params:
                      noise_scale: 0.03
                      max_grad_norm: 1.0
                      scale_factors: [0.5, 1.0, 1.5]
                  selection:
                    method: best
                    min_improvement: 0.01
                  eval_frequency: 5
                  warmup_steps: 200
                  cache_baseline: true
                  logging:
                    candidates: false
                    selected: true
            """
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)

    assert spec.training.evolutionary.enabled is True
    assert spec.training.evolutionary.candidates == 4
    assert spec.training.evolutionary.validation_config == "configs/fitness/tool_calling.yaml"
    assert spec.training.evolutionary.strategy.type == "gradient_noise"
    assert spec.training.evolutionary.strategy.params["noise_scale"] == 0.03
    assert spec.training.evolutionary.selection.method == "best"
    assert spec.training.evolutionary.logging.candidates is False


def test_load_experiment_spec_rejects_invalid_post_training_mode(tmp_path):
    spec_path = tmp_path / "invalid_post_training.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            experiment:
              name: invalid-post-training
              provider: hf_jobs
              method: sft
              dataset:
                source: repo/dataset
                file: sample.jsonl
              training:
                model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct
              post_training:
                mode: unsupported
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported post_training.mode"):
        load_experiment_spec(spec_path)
