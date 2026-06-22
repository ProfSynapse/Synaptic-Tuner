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


# ---------------------------------------------------------------------------
# _judge_metric_unresolved: the None-guard for judge-derived gradient metrics.
# Extended (Task #23) to ALSO cover stats.quality_gated_normalized_score so the
# all-reject / no-gate edges degrade to score_floor instead of crashing
# _resolve_metric on a None. Unit-tested directly (no filesystem / LLM).
# ---------------------------------------------------------------------------

def _make_adapter(*, metric: str, failure_policy: str = "score_floor"):
    from shared.prompt_optimization.evaluators import EvaluatorScoringAdapter

    evaluation_config = {
        "evaluator": {
            "failure_policy": failure_policy,
            "objective": {"metric": metric},
            "prompt_placement": {
                "mode": "system_overlay",
                "template": "{candidate_prompt}\n\n{system}",
            },
        }
    }
    return EvaluatorScoringAdapter(
        evaluation_config=evaluation_config,
        config_path=Path("unused.yaml"),
        repo_root=Path("."),
        score_floor=0.2,
    )


def test_judge_metric_unresolved_fires_for_judge_gradient_when_none():
    adapter = _make_adapter(metric="stats.judge_normalized_score")
    assert adapter._judge_metric_unresolved({"judge_normalized_score": None}) is True


def test_judge_metric_unresolved_fires_for_quality_gated_when_none():
    """The new gated metric joins the None-guard: a None gated gradient (no rubric
    carried a quality_gate / no case judged) degrades to floor, not a crash."""
    adapter = _make_adapter(metric="stats.quality_gated_normalized_score")
    assert adapter._judge_metric_unresolved({"quality_gated_normalized_score": None}) is True


def test_quality_gated_all_reject_is_numeric_zero_not_none():
    """ALL-REJECT edge: every case trips a floor -> each per-case gated value is 0.0
    -> the run-level mean is a numeric 0.0, NOT None. So the guard does NOT fire and
    _resolve_metric consumes 0.0 cleanly (worst candidate, no crash). This is the
    distinction the guard's None-path and the helper's numeric-path must both honour.
    """
    from Evaluator.reporting import _quality_gated_normalized_score

    class _Judge:
        def __init__(self, scores):
            self.judge_result = type("R", (), {"scores": scores})()

    class _Score:
        def __init__(self, gated):
            self.quality_gated_score = gated

    class _Record:
        def __init__(self, judge):
            self.judge = judge

    # Two cases, BOTH gate-rejected (gated == 0.0 each).
    records = [_Record(_Judge([_Score(0.0)])), _Record(_Judge([_Score(0.0)]))]
    result = _quality_gated_normalized_score(records)
    assert result == 0.0
    assert result is not None  # numeric zero, not the None default-off path

    # And the guard does NOT fire on a numeric 0.0 stat (only on None).
    adapter = _make_adapter(metric="stats.quality_gated_normalized_score")
    assert adapter._judge_metric_unresolved({"quality_gated_normalized_score": 0.0}) is False


def test_judge_metric_unresolved_does_not_fire_for_other_metrics():
    adapter = _make_adapter(metric="stats.pass_rate")
    # Even with a None gated gradient present, a non-gradient objective keeps the
    # strict numeric contract (guard returns False -> _resolve_metric handles it).
    assert adapter._judge_metric_unresolved({"quality_gated_normalized_score": None}) is False


def test_judge_metric_unresolved_respects_failure_policy_raise():
    adapter = _make_adapter(metric="stats.quality_gated_normalized_score", failure_policy="raise")
    # failure_policy='raise' must NOT short-circuit; caller proceeds to _resolve_metric
    # which raises on the None as before.
    assert adapter._judge_metric_unresolved({"quality_gated_normalized_score": None}) is False


def test_all_cases_structural_gate_rejected_yields_none_then_guard_fires():
    """The architect's named landmine path, end to end at the stats+guard layer:
    every case fails the structural gate (--judge-mode and) -> NO case is judged
    -> record.judge is None on every record -> _quality_gated_normalized_score
    returns None (NOT 0.0; nothing was scored). The extended guard must then fire
    so the candidate floors instead of crashing _resolve_metric on the None.

    This is DISTINCT from the all-floor-breach case (which IS judged -> per-case
    0.0 -> numeric 0.0 mean): here the rejection happens BEFORE the judge, so the
    gated gradient is genuinely absent (None), which is exactly the case the
    judge_normalized_score-only guard would have missed for the new metric key.
    """
    from Evaluator.reporting import _quality_gated_normalized_score

    class _Record:
        def __init__(self):
            self.judge = None  # structurally gate-rejected -> never judged

    records = [_Record(), _Record()]
    gradient = _quality_gated_normalized_score(records)
    assert gradient is None  # nothing judged -> genuinely absent, not 0.0

    adapter = _make_adapter(metric="stats.quality_gated_normalized_score")
    # The guard fires on the None gradient -> caller floors the candidate instead
    # of passing None into _resolve_metric (which would crash).
    assert adapter._judge_metric_unresolved({"quality_gated_normalized_score": gradient}) is True


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
