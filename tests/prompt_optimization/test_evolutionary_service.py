from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from shared.prompt_optimization import PromptOptimizationService
from tuner.handlers.prompt_optimize_handler import PromptOptimizeHandler


def test_evolutionary_prompt_optimization_reaches_target_and_writes_v2_artifacts(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text(
        """
prompts:
  main: Answer carefully.
  formatter: Return JSON.
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-target
  seed: 7
  output_dir: {output_dir.as_posix()}
  population_size: 4
  max_generations: 4
  elite_count: 1
  mutation_rate: 1.0
  crossover_rate: 0.0
  stopping:
    target_score: 1.0
    max_stagnation: 3
    min_delta: 0.0
  subjects:
    - id: main
      path: {source.as_posix()}
      dotted_path: prompts.main
    - id: formatter
      path: {source.as_posix()}
      dotted_path: prompts.formatter
  operators:
    - type: append
      values:
        - Use provided context only.
  evaluation:
    assertions:
      - id: context_bound
        type: contains
        text: Use provided context only.
      - id: keeps_json
        type: contains
        text: JSON
""".lstrip(),
        encoding="utf-8",
    )

    first = PromptOptimizationService.from_config(config).run()
    second = PromptOptimizationService.from_config(config).run()

    assert first.schema_version == 2
    assert first.strategy == "evolutionary"
    assert first.stop_reason == "target_score"
    assert first.best_score == 1.0
    assert first.best_candidate["id"] == second.best_candidate["id"]
    assert first.generation_count == 1
    for name in [
        "manifest.json",
        "candidate_history.jsonl",
        "best_candidate.json",
        "overlays.json",
        "replay.json",
        "generation_history.jsonl",
        "lineage.json",
    ]:
        assert (output_dir / name).exists()

    overlays = json.loads((output_dir / "overlays.json").read_text(encoding="utf-8"))
    assert overlays["version"] == 2
    assert overlays["strategy"] == "evolutionary"
    assert set(overlays["subjects"]) == {"main", "formatter"}
    assert overlays["subjects"]["main"]["optimized_prompt"]
    assert overlays["subjects"]["formatter"]["optimized_prompt"]

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["generation"] for row in history} == {0}
    assert any(row["selected"] for row in history)
    assert all(0.0 <= row["score"] <= 1.0 for row in history)
    history_ids = {row["id"] for row in history}
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["selected_candidate_ids"].values()) <= history_ids
    assert set(overlays["selected_candidate_ids"].values()) <= history_ids
    assert {
        subject["selected_candidate_id"]
        for subject in overlays["subjects"].values()
    } <= history_ids


def test_evolutionary_prompt_optimization_stops_on_stagnation_and_records_lineage(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Stable baseline.\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-stagnation
  seed: 2
  output_dir: {output_dir.as_posix()}
  population_size: 3
  max_generations: 5
  elite_count: 1
  mutation_rate: 1.0
  crossover_rate: 0.0
  stopping:
    target_score: 1.1
    max_stagnation: 1
    min_delta: 0.0
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values:
        - Stable baseline.
  evaluation:
    assertions:
      - type: contains
        text: Stable baseline.
""".lstrip(),
        encoding="utf-8",
    )

    result = PromptOptimizationService.from_config(config).run()

    assert result.stop_reason == "stagnation"
    assert result.generation_count == 2
    rows = [
        json.loads(line)
        for line in (output_dir / "generation_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["generation"] for row in rows] == [0, 1]
    lineage = json.loads((output_dir / "lineage.json").read_text(encoding="utf-8"))
    assert any(item["operator"] == "elite" for item in lineage["candidates"].values())
    assert any(item["parents"] for item in lineage["candidates"].values())


def test_evolutionary_prompt_optimization_uses_crossover_for_multi_subject_genomes(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text(
        """
prompts:
  a: Alpha
  b: Beta
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-crossover
  seed: 5
  output_dir: {output_dir.as_posix()}
  population_size: 4
  max_generations: 2
  elite_count: 1
  mutation_rate: 0.0
  crossover_rate: 1.0
  stopping:
    target_score: 1.1
    max_stagnation: 4
  subjects:
    - id: a
      path: {source.as_posix()}
      dotted_path: prompts.a
    - id: b
      path: {source.as_posix()}
      dotted_path: prompts.b
  operators:
    - type: append
      values:
        - improved
  evaluation:
    assertions:
      - type: contains
        text: improved
""".lstrip(),
        encoding="utf-8",
    )

    PromptOptimizationService.from_config(config).run()

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    crossover_rows = [row for row in history if row["operator"] == "crossover"]
    assert crossover_rows
    assert all(len(row["parents"]) == 2 for row in crossover_rows)
    assert all(set(row["genome"]) == {"a", "b"} for row in history)


def test_evolutionary_prompt_optimization_penalizes_missing_required_placeholders(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Answer using {context}.\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-placeholders
  seed: 1
  output_dir: {output_dir.as_posix()}
  population_size: 2
  max_generations: 1
  score_floor: 0.0
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
      required_placeholders:
        - "{{context}}"
  operators:
    - type: replace
      replacements:
        - old: "{{context}}"
          new: "the available details"
  evaluation:
    assertions:
      - type: contains
        text: available details
""".lstrip(),
        encoding="utf-8",
    )

    PromptOptimizationService.from_config(config).run()

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = [row for row in history if row["diagnostics"]]
    assert failed
    assert failed[0]["score"] == 0.0
    assert failed[0]["diagnostics"][0]["code"] == "MISSING_REQUIRED_PLACEHOLDER"


def test_schema_v1_prompt_optimization_result_defaults_remain_compatible(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: v1-compatible
  output_dir: {(tmp_path / "out").as_posix()}
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions:
      - type: contains
        text: schema
""".lstrip(),
        encoding="utf-8",
    )

    result = PromptOptimizationService.from_config(config).run()

    assert result.schema_version == 1
    assert result.strategy == "fixture"
    assert result.generation_count is None
    assert result.stop_reason is None


def test_evolutionary_prompt_optimization_requires_positive_weight_assertion(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-no-positive-assertions
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 2
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions:
      - type: contains
        text: schema
        weight: 0
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="strictly positive"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_prompt_optimization_rejects_negative_assertion_weight(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-negative-weight
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 2
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions:
      - type: contains
        text: schema
        weight: -1
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="strictly positive"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_prompt_optimization_requires_non_empty_assertions(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-empty-assertions
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 2
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions: []
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="positive-weight assertion"):
        PromptOptimizationService.from_config(config).run()


def test_evolutionary_prompt_optimization_requires_known_assertion_subject_id(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-bad-subject-id
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 2
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions:
      - type: contains
        subject_id: typo
        text: schema
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="unknown subject: typo"):
        PromptOptimizationService.from_config(config).run()


def test_prompt_optimize_handler_json_includes_v2_fields(tmp_path, capsys):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  schema_version: 2
  strategy: evolutionary
  run_id: evo-handler
  seed: 4
  output_dir: {(tmp_path / "out").as_posix()}
  population_size: 2
  max_generations: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  operators:
    - type: append
      values: ["Use schema."]
  evaluation:
    assertions:
      - type: contains
        text: schema
""".lstrip(),
        encoding="utf-8",
    )
    handler = PromptOptimizeHandler(
        Namespace(
            json=True,
            prompt_opt_config=str(config),
            prompt_opt_output_dir=None,
        )
    )

    assert handler.handle() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["success"] is True
    assert payload["data"]["schema_version"] == 2
    assert payload["data"]["strategy"] == "evolutionary"
    assert payload["data"]["generation_count"] == 1
    assert payload["data"]["stop_reason"] in {"target_score", "max_generations", "stagnation"}
