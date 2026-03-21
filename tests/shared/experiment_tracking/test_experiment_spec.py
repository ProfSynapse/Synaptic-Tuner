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
              evaluation:
                preset: quick
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
    assert spec.evaluation.preset == "quick"
    assert spec.loss.enabled is True


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
