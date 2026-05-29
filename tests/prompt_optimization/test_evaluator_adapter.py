from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.prompt_optimization import PromptOptimizationService


def test_labkit_epistemic_humility_evaluator_smoke_config_runs_dry_run(tmp_path):
    config = Path("configs/prompt_optimization/labkit_epistemic_humility_evaluator_smoke.yaml")
    output_dir = tmp_path / "epistemic-smoke-out"

    result = PromptOptimizationService.from_config(
        config,
        overrides={"output_dir": output_dir.as_posix()},
    ).run()

    assert result.schema_version == 2
    assert result.strategy == "evolutionary"
    assert result.candidate_count == 3
    assert result.generation_count == 1
    assert result.stop_reason == "max_generations"
    assert result.best_score == 0.0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "evaluator"
    assert manifest["strategy"] == "evolutionary"
    assert manifest["schema_version"] == 2
    assert manifest["candidate_count"] == 3

    replay = json.loads((output_dir / "replay.json").read_text(encoding="utf-8"))
    evaluator_config = replay["config"]["evaluation"]["evaluator"]
    assert evaluator_config["dry_run"] is True
    assert evaluator_config["scenarios"] == ["labkit_epistemic_humility_smoke.yaml"]
    assert evaluator_config["objective"]["metric"] == "stats.pass_rate"

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 3
    assert {row["generation"] for row in history} == {0}
    assert {row["metrics"]["case_count"] for row in history} == {3}
    assert {row["metrics"]["objective_metric"] for row in history} == {"stats.pass_rate"}
    assert {row["metrics"]["objective_value"] for row in history} == {0.0}
    assert {row["score"] for row in history} == {0.0}
    assert all(row["metrics"]["evaluator_stats"]["by_tag"]["labkit"]["total"] == 3 for row in history)


def test_evolutionary_evaluator_adapter_injects_system_overlay_and_selects_objective(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-backed
  output_dir: {output_dir.as_posix()}
  population_size: 1
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      prompt_placement:
        mode: system_overlay
        template: "OPTIMIZED:\\n{{candidate_prompt}}\\nBASE:\\n{{system}}"
      objective:
        metric: stats.pass_rate
""".lstrip(),
        encoding="utf-8",
    )
    calls = {}

    def fake_evaluate_cases(cases, client, **kwargs):
        calls["case_system"] = cases[0].metadata["system"]
        calls["dry_run"] = kwargs["dry_run"]
        calls["parallel"] = kwargs["parallel"]
        return [_FakeRecord(cases[0])]

    def fake_aggregate_stats(records):
        calls["record_count"] = len(records)
        return {"pass_rate": 0.75, "normalized_score": 0.25}

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", fake_evaluate_cases)
    monkeypatch.setattr("Evaluator.reporting.aggregate_stats", fake_aggregate_stats)

    result = PromptOptimizationService.from_config(config).run()

    assert result.best_score == 0.75
    assert "OPTIMIZED:\nCandidate instruction." in calls["case_system"]
    assert "BASE:\nOriginal system." in calls["case_system"]
    assert calls["dry_run"] is True
    assert calls["parallel"] is False
    assert calls["record_count"] == 1

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history[0]["metrics"]["objective_metric"] == "stats.pass_rate"
    assert history[0]["metrics"]["objective_value"] == 0.75
    assert history[0]["metrics"]["normalized_score"] == 0.75


def test_evolutionary_evaluator_adapter_scores_floor_on_candidate_failure(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-floor
  output_dir: {output_dir.as_posix()}
  population_size: 1
  max_generations: 1
  score_floor: 0.2
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      prompt_placement:
        mode: system_overlay
        template: "{{candidate_prompt}}\\n{{system}}"
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    def fake_evaluate_cases(cases, client, **kwargs):
        raise RuntimeError("local evaluator failed")

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", fake_evaluate_cases)

    result = PromptOptimizationService.from_config(config).run()

    assert result.best_score == 0.2
    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history[0]["score"] == 0.2
    assert history[0]["diagnostics"][0]["code"] == "EVALUATOR_SCORING_FAILED"
    assert history[0]["diagnostics"][0]["severity"] == "error"


def test_evolutionary_evaluator_adapter_fail_policy_raise_propagates(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-raise
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      failure_policy: raise
      prompt_placement:
        mode: system_overlay
        template: "{{candidate_prompt}}\\n{{system}}"
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    def fake_evaluate_cases(cases, client, **kwargs):
        raise RuntimeError("hard failure")

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", fake_evaluate_cases)

    with pytest.raises(RuntimeError, match="hard failure"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_evaluator_adapter_requires_explicit_prompt_placement(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-bad-placement
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      prompt_placement:
        mode: template_var
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="system_overlay"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_evaluator_adapter_accepts_legacy_prompt_injection_alias(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-legacy-alias
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      prompt_injection:
        mode: system_overlay
        template: "LEGACY {{candidate_prompt}} {{system}}"
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    def fake_evaluate_cases(cases, client, **kwargs):
        assert "LEGACY Candidate instruction. Original system." == cases[0].metadata["system"]
        return [_FakeRecord(cases[0])]

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", fake_evaluate_cases)
    monkeypatch.setattr("Evaluator.reporting.aggregate_stats", lambda records: {"normalized_score": 0.6})

    result = PromptOptimizationService.from_config(config).run()

    assert result.best_score == 0.6


def test_evolutionary_evaluator_adapter_metric_typo_raises_instead_of_flooring(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-metric-typo
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  score_floor: 0.2
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: true
      prompt_placement:
        mode: system_overlay
        template: "{{candidate_prompt}}\\n{{system}}"
      objective:
        metric: stats.typo
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", lambda cases, client, **kwargs: [_FakeRecord(cases[0])])
    monkeypatch.setattr("Evaluator.reporting.aggregate_stats", lambda records: {"normalized_score": 0.8})

    with pytest.raises(Exception, match="objective metric not found"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_evaluator_adapter_scalar_scenarios_raise_instead_of_flooring(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-scalar-scenarios
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  score_floor: 0.2
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: smoke.yaml
      dry_run: true
      prompt_placement:
        mode: system_overlay
        template: "{{candidate_prompt}}\\n{{system}}"
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", lambda *args, **kwargs: pytest.fail("should not execute"))

    with pytest.raises(Exception, match="scenarios must be a list"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_evaluator_adapter_bad_model_shape_raises_instead_of_flooring(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Candidate instruction.\n", encoding="utf-8")
    evaluator_config_dir = _write_evaluator_config(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evaluator-bad-model
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 1
  max_generations: 1
  score_floor: 0.2
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["unused"]
  evaluation:
    mode: evaluator
    evaluator:
      config_dir: {evaluator_config_dir.as_posix()}
      scenarios: ["smoke.yaml"]
      dry_run: false
      model: configured-model
      prompt_placement:
        mode: system_overlay
        template: "{{candidate_prompt}}\\n{{system}}"
      objective:
        metric: stats.normalized_score
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("Evaluator.runner.evaluate_cases", lambda *args, **kwargs: pytest.fail("should not execute"))

    with pytest.raises(Exception, match="model must be a mapping"):
        PromptOptimizationService.from_config(config).run()


class _FakeRecord:
    def __init__(self, case):
        self.case = case
        self.error = None
        self.scoring = None

    @property
    def status(self):
        return "pass"

    @property
    def passed(self):
        return True

    @property
    def score(self):
        return None


def _write_evaluator_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "EvaluatorConfig"
    scenarios_dir = config_dir / "scenarios"
    scenarios_dir.mkdir(parents=True)
    (scenarios_dir / "smoke.yaml").write_text(
        """
name: Prompt Optimization Smoke
description: Local adapter test scenario.
tests:
  - id: local_case
    question: Answer the request.
    system: Original system.
""".lstrip(),
        encoding="utf-8",
    )
    (config_dir / "eval_run.yaml").write_text(
        """
run:
  scenarios: ["smoke.yaml"]
""".lstrip(),
        encoding="utf-8",
    )
    return config_dir
