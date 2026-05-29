from __future__ import annotations

import json
import re
from pathlib import Path

from shared.prompt_optimization import PromptOptimizationService


def test_prompt_optimization_writes_deterministic_artifacts(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text(
        """
formats:
  neutral:
    instructions:
      - Return a JSON object.
      - Keep the configured format.
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: test-run
  seed: 3
  output_dir: {output_dir.as_posix()}
  candidates_per_subject: 3
  subjects:
    - id: subject
      path: {source.as_posix()}
      dotted_path: formats.neutral.instructions
      join_with: "\\n"
  operators:
    - type: append
      values:
        - "Verify JSON before returning."
    - type: replace
      replacements:
        - old: "Return a JSON object."
          new: "Return exactly one JSON object."
  evaluation:
    assertions:
      - id: exact_json
        type: contains
        text: "exactly one JSON object"
        weight: 2
      - id: format
        type: contains
        text: "format"
        weight: 1
""".lstrip(),
        encoding="utf-8",
    )

    first = PromptOptimizationService.from_config(config).run()
    second = PromptOptimizationService.from_config(config).run()

    assert first.best_candidate["id"] == second.best_candidate["id"]
    assert first.candidate_count == 4
    assert first.best_candidate["score"] == 3.0
    for name in [
        "manifest.json",
        "candidate_history.jsonl",
        "best_candidate.json",
        "overlays.json",
        "replay.json",
    ]:
        assert (output_dir / name).exists()

    overlays = json.loads((output_dir / "overlays.json").read_text(encoding="utf-8"))
    assert overlays["subjects"]["subject"]["optimized_prompt"] == first.best_candidate["prompt_text"]


def test_dotted_path_can_resolve_list_index(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("items:\n  - first\n  - second\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: list-index
  output_dir: {(tmp_path / "out").as_posix()}
  subjects:
    - id: item
      path: {source.as_posix()}
      dotted_path: items.1
  operators:
    - type: append
      values: ["third"]
  evaluation:
    assertions:
      - type: contains
        text: second
""".lstrip(),
        encoding="utf-8",
    )

    result = PromptOptimizationService.from_config(config).run()

    assert result.best_candidate["prompt_text"].startswith("second")


def test_prompt_optimization_writes_overlay_for_each_subject(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text(
        """
items:
  first: alpha
  second: beta
""".lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: multi-subject
  seed: 5
  output_dir: {output_dir.as_posix()}
  include_baseline: true
  candidates_per_subject: 1
  subjects:
    - id: first
      path: {source.as_posix()}
      dotted_path: items.first
    - id: second
      path: {source.as_posix()}
      dotted_path: items.second
  operators:
    - type: append
      values: ["gamma"]
  evaluation:
    assertions:
      - type: contains
        text: gamma
""".lstrip(),
        encoding="utf-8",
    )

    PromptOptimizationService.from_config(config).run()

    overlays = json.loads((output_dir / "overlays.json").read_text(encoding="utf-8"))
    assert set(overlays["subjects"]) == {"first", "second"}
    assert overlays["subjects"]["first"]["source_path_absolute"] == str(source.resolve())
    assert overlays["subjects"]["first"]["source_path_repo_relative"] == ""
    assert overlays["subjects"]["first"]["selected_candidate_id"]
    assert overlays["subjects"]["second"]["selected_candidate_id"]


def test_llm_rewrite_operator_uses_configured_shared_llm_client(tmp_path, monkeypatch):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: llm-rewrite
  output_dir: {output_dir.as_posix()}
  include_baseline: false
  candidates_per_subject: 1
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  llm:
    provider: lmstudio
    model: configured-model
    temperature: 0.1
    max_tokens: 128
    system_prompt: Rewrite prompts without changing their intent.
    prompt_template: |
      Subject: {{subject_id}}
      Current prompt:
      {{prompt}}
  operators:
    - type: llm_rewrite
  evaluation:
    assertions:
      - type: contains
        text: exactly one JSON object
        weight: 2
""".lstrip(),
        encoding="utf-8",
    )
    calls = []

    class FakeClient:
        def chat(self, messages, temperature=0.7, max_tokens=1024, **kwargs):
            calls.append(
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return "Return exactly one JSON object."

    def fake_create_client(**kwargs):
        calls.append({"create_client": kwargs})
        return FakeClient()

    monkeypatch.setattr("shared.llm.create_client", fake_create_client)

    result = PromptOptimizationService.from_config(config).run()

    assert result.best_candidate["prompt_text"] == "Return exactly one JSON object."
    assert result.best_candidate["score"] == 2.0
    assert calls[0]["create_client"]["provider"] == "lmstudio"
    assert calls[0]["create_client"]["model"] == "configured-model"
    assert calls[1]["temperature"] == 0.1
    assert calls[1]["max_tokens"] == 128
    assert "Return JSON." in calls[1]["messages"][1]["content"]

    history = (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8")
    assert '"source": "llm"' in history


def test_llm_rewrite_requires_explicit_prompting_config(tmp_path):
    source = tmp_path / "prompts.yaml"
    source.write_text("prompt: Return JSON.\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
prompt_optimization:
  run_id: missing-llm-template
  output_dir: {(tmp_path / "out").as_posix()}
  include_baseline: false
  subjects:
    - id: prompt
      path: {source.as_posix()}
      dotted_path: prompt
  llm:
    provider: lmstudio
    model: configured-model
  operators:
    - type: llm_rewrite
""".lstrip(),
        encoding="utf-8",
    )

    try:
        PromptOptimizationService.from_config(config).run()
    except Exception as exc:
        assert "llm.prompt_template is required" in str(exc)
    else:
        raise AssertionError("Expected missing LLM prompt config to fail")


def test_labkit_healthcare_safety_fixture_config_runs_with_expected_guardrails(tmp_path):
    config = Path("configs/prompt_optimization/labkit_healthcare_safety.yaml")
    output_dir = tmp_path / "healthcare-out"

    result = PromptOptimizationService.from_config(
        config,
        overrides={"output_dir": output_dir.as_posix()},
    ).run()

    assert result.candidate_count == 9
    assert result.best_candidate["score"] >= 7.0

    optimized = result.best_candidate["prompt_text"]
    assert re.search(r"\b(may|might|could|uncertain|cannot determine)\b", optimized)
    assert "professional" in optimized
    assert "you have" not in optimized
    assert "diagnosis is" not in optimized
    assert len(optimized) <= 900

    history = [
        json.loads(line)
        for line in (output_dir / "candidate_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    operators = {item["operator"] for item in history}
    assert {"baseline", "append", "prepend", "replace", "remove_line_contains"} <= operators


def test_labkit_rag_customer_support_fixture_config_preserves_placeholders(tmp_path):
    config = Path("configs/prompt_optimization/labkit_rag_customer_support.yaml")
    output_dir = tmp_path / "rag-out"

    result = PromptOptimizationService.from_config(
        config,
        overrides={"output_dir": output_dir.as_posix()},
    ).run()

    assert result.candidate_count == 30

    overlays = json.loads((output_dir / "overlays.json").read_text(encoding="utf-8"))
    subjects = overlays["subjects"]
    assert set(subjects) == {"responseGeneration", "contextFormatting", "queryEnhancement"}

    response_prompt = subjects["responseGeneration"]["optimized_prompt"]
    context_prompt = subjects["contextFormatting"]["optimized_prompt"]
    query_prompt = subjects["queryEnhancement"]["optimized_prompt"]

    assert "{context}" in response_prompt
    assert "{query}" in response_prompt
    assert "{chunks}" in context_prompt
    assert "{query}" in query_prompt

    for prompt in [response_prompt, context_prompt, query_prompt]:
        assert re.search(r"(?i)(insufficient|limitations|only|provided|retrieved)", prompt)
        assert len(prompt) <= 1400
