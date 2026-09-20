from __future__ import annotations

import json

import pytest
import yaml

from SynthChat.scripts.structured_document_bakeoff import judge_existing, load_config, run_bakeoff
from shared.llm.usage import LLMStructuredV1, measured_usage


class FakeClient:
    def __init__(self, model, calls, response):
        self.model = model
        self.calls = calls
        self.response = response

    def structured_output(self, messages, schema, temperature, max_tokens):
        self.calls.append(
            {
                "model": self.model,
                "messages": messages,
                "schema": schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _config(tmp_path, **extra):
    source = tmp_path / "chapter.md"
    source.write_text("---\ntitle: Example\n---\nThe actual document.", encoding="utf-8")
    config = {
        "source": {"path": "chapter.md", "strip_yaml_frontmatter": True},
        "prompt_template": "Outline this:\n{document}",
        "response_schema": {"type": "object"},
        "models": [
            {"id": "provider/a", "provider_routing": {"data_collection": "deny"}},
            "provider/b",
        ],
        "temperature": 0.2,
        "max_tokens": 321,
        "output_dir": "results",
    }
    config.update(extra)
    path = tmp_path / "bakeoff.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_records_each_candidate_and_fixed_judge_without_source_frontmatter(tmp_path):
    config_path = _config(
        tmp_path,
        judge={
            "model": "provider/judge",
            "prompt_template": "SOURCE={document}\nCANDIDATE={candidate}",
            "response_schema": {"type": "object"},
            "temperature": 0,
            "max_tokens": 99,
        },
    )
    calls = []
    responses = {
        "provider/a": LLMStructuredV1({"outline": "a"}, measured_usage(10, 3)),
        "provider/b": LLMStructuredV1({"outline": "b"}),
        "provider/judge": LLMStructuredV1({"score": 8}),
    }
    factory_defaults = []

    def factory(*, config_defaults):
        factory_defaults.append(config_defaults)
        return FakeClient(config_defaults["model"], calls, responses[config_defaults["model"]])

    ticks = iter((1.0, 1.5, 2.0, 2.25, 3.0, 3.5, 4.0, 4.25))
    manifest = run_bakeoff(config_path, client_factory=factory, clock=lambda: next(ticks))

    assert [item["model"] for item in manifest["models"]] == ["provider/a", "provider/b"]
    assert factory_defaults[0] == {"provider": "openrouter", "model": "provider/judge"}
    assert factory_defaults[1] == {
        "provider": "openrouter",
        "model": "provider/a",
        "provider_routing": {"data_collection": "deny"},
    }
    assert "title: Example" not in calls[0]["messages"][0]["content"]
    assert "The actual document." in calls[0]["messages"][0]["content"]
    assert calls[1]["model"] == "provider/judge"
    assert '"outline": "a"' in calls[1]["messages"][0]["content"]
    assert calls[1]["temperature"] == 0.0
    assert calls[1]["max_tokens"] == 99

    first = json.loads((tmp_path / "results" / "01-provider_a.json").read_text(encoding="utf-8"))
    assert first["generation"]["payload"] == {"outline": "a"}
    assert first["generation"]["usage"]["input_tokens"] == 10
    assert first["judge"]["payload"] == {"score": 8}
    assert json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))["kind"] == "structured_document_bakeoff/v1"


def test_generation_failure_is_recorded_and_does_not_call_judge(tmp_path):
    config_path = _config(
        tmp_path,
        models=["provider/failing"],
        judge={
            "model": "provider/judge",
            "prompt_template": "{document}\n{candidate}",
            "response_schema": {"type": "object"},
        },
    )
    calls = []

    def factory(*, config_defaults):
        response = RuntimeError("provider unavailable") if config_defaults["model"] == "provider/failing" else LLMStructuredV1({"score": 1})
        return FakeClient(config_defaults["model"], calls, response)

    run_bakeoff(config_path, client_factory=factory)
    record = json.loads((tmp_path / "results" / "01-provider_failing.json").read_text(encoding="utf-8"))
    assert record["generation"]["payload"] is None
    assert record["generation"]["error"]["type"] == "RuntimeError"
    assert "judge" not in record
    assert [call["model"] for call in calls] == ["provider/failing"]


def test_schema_invalid_generation_is_recorded_and_does_not_call_judge(tmp_path):
    config_path = _config(
        tmp_path,
        models=["provider/invalid"],
        response_schema={
            "type": "object",
            "properties": {"metadata": {"type": "object", "required": ["title"]}},
            "required": ["metadata"],
        },
        judge={
            "model": "provider/judge",
            "prompt_template": "{document}\n{candidate}",
            "response_schema": {"type": "object"},
        },
    )
    calls = []

    def factory(*, config_defaults):
        response = LLMStructuredV1({"metadata": {}}) if config_defaults["model"] == "provider/invalid" else LLMStructuredV1({"score": 1})
        return FakeClient(config_defaults["model"], calls, response)

    manifest = run_bakeoff(config_path, client_factory=factory)

    record = json.loads((tmp_path / "results" / "01-provider_invalid.json").read_text(encoding="utf-8"))
    assert record["generation"]["payload"] is None
    assert record["generation"]["error"]["type"] == "ValueError"
    assert "generation payload failed JSON Schema validation at $.metadata" in record["generation"]["error"]["message"]
    assert "'title' is a required property" in record["generation"]["error"]["message"]
    assert [call["model"] for call in calls] == ["provider/invalid"]
    assert manifest["models"][0]["success"] is False


def test_schema_invalid_judgment_is_recorded_as_judge_failure(tmp_path):
    config_path = _config(
        tmp_path,
        models=["provider/candidate"],
        judge={
            "model": "provider/judge",
            "prompt_template": "{document}\n{candidate}",
            "response_schema": {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
            },
        },
    )
    calls = []
    responses = {
        "provider/candidate": LLMStructuredV1({"outline": "valid"}),
        "provider/judge": LLMStructuredV1({"verdict": "missing score"}),
    }

    def factory(*, config_defaults):
        return FakeClient(config_defaults["model"], calls, responses[config_defaults["model"]])

    run_bakeoff(config_path, client_factory=factory)

    record = json.loads((tmp_path / "results" / "01-provider_candidate.json").read_text(encoding="utf-8"))
    assert record["generation"]["payload"] == {"outline": "valid"}
    assert record["judge"]["payload"] is None
    assert record["judge"]["error"]["type"] == "ValueError"
    assert "judge payload failed JSON Schema validation at $" in record["judge"]["error"]["message"]


def test_judge_existing_retries_only_successful_unjudged_candidates(tmp_path):
    config_path = _config(
        tmp_path,
        judge={
            "model": "provider/judge",
            "prompt_template": "SOURCE={document}\nCANDIDATE={candidate}",
            "response_schema": {"type": "object"},
        },
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "01-provider_a.json").write_text(
        json.dumps({"model": "provider/a", "generation": {"payload": {"outline": "retry"}, "error": None}}),
        encoding="utf-8",
    )
    (results_dir / "02-provider_b.json").write_text(
        json.dumps({"model": "provider/b", "generation": {"payload": None, "error": {"type": "RuntimeError"}}}),
        encoding="utf-8",
    )
    (results_dir / "03-provider_c.json").write_text(
        json.dumps(
            {
                "model": "provider/c",
                "generation": {"payload": {"outline": "already judged"}, "error": None},
                "judge": {"payload": {"score": 10}, "error": None},
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "manifest.json").write_text("{}", encoding="utf-8")
    calls = []

    def factory(*, config_defaults):
        assert config_defaults == {"provider": "openrouter", "model": "provider/judge"}
        return FakeClient("provider/judge", calls, LLMStructuredV1({"score": 7}))

    summary = judge_existing(config_path, results_dir, client_factory=factory)

    assert summary == {
        "results_dir": str(results_dir.resolve()),
        "files": 3,
        "attempted": 1,
        "judged": 1,
        "judge_failed": 0,
        "skipped_generation_failure": 1,
        "skipped_existing_judgment": 1,
        "invalid_records": 0,
    }
    retried = json.loads((results_dir / "01-provider_a.json").read_text(encoding="utf-8"))
    assert retried["judge"]["payload"] == {"score": 7}
    assert '"outline": "retry"' in calls[0]["messages"][0]["content"]
    assert [call["model"] for call in calls] == ["provider/judge"]


def test_judge_existing_retries_schema_invalid_judge_and_rejects_invalid_generation(tmp_path):
    config_path = _config(
        tmp_path,
        response_schema={
            "type": "object",
            "properties": {"outline": {"type": "string"}},
            "required": ["outline"],
        },
        judge={
            "model": "provider/judge",
            "prompt_template": "{document}\n{candidate}",
            "response_schema": {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
            },
        },
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    invalid_judge_path = results_dir / "01-provider_a.json"
    invalid_judge_path.write_text(
        json.dumps(
            {
                "model": "provider/a",
                "generation": {"payload": {"outline": "valid"}, "error": None},
                "judge": {"payload": {"verdict": "invalid"}, "error": None},
            }
        ),
        encoding="utf-8",
    )
    invalid_generation_path = results_dir / "02-provider_b.json"
    invalid_generation_path.write_text(
        json.dumps({"model": "provider/b", "generation": {"payload": {"outline": 42}, "error": None}}),
        encoding="utf-8",
    )
    calls = []

    def factory(*, config_defaults):
        return FakeClient(config_defaults["model"], calls, LLMStructuredV1({"score": 9}))

    summary = judge_existing(config_path, results_dir, client_factory=factory)

    assert summary["attempted"] == 1
    assert summary["judged"] == 1
    assert summary["invalid_records"] == 1
    assert summary["skipped_existing_judgment"] == 0
    assert [call["model"] for call in calls] == ["provider/judge"]
    retried = json.loads(invalid_judge_path.read_text(encoding="utf-8"))
    assert retried["judge"]["payload"] == {"score": 9}
    rejected = json.loads(invalid_generation_path.read_text(encoding="utf-8"))
    assert rejected["generation"]["payload"] == {"outline": 42}
    assert "generation payload failed JSON Schema validation at $.outline" in rejected["generation"]["error"]["message"]


def test_config_requires_the_document_token_and_a_judge_candidate_token(tmp_path):
    path = _config(tmp_path, prompt_template="No placeholder")
    with pytest.raises(ValueError, match=r"\{document\}"):
        load_config(path)

    path = _config(
        tmp_path,
        judge={"model": "provider/judge", "prompt_template": "{document}", "response_schema": {"type": "object"}},
    )
    with pytest.raises(ValueError, match=r"\{candidate\}"):
        load_config(path)


def test_config_rejects_invalid_response_schemas(tmp_path):
    path = _config(tmp_path, response_schema={"type": "not-a-json-schema-type"})
    with pytest.raises(ValueError, match="response_schema is not a valid JSON Schema"):
        load_config(path)

    path = _config(
        tmp_path,
        judge={
            "model": "provider/judge",
            "prompt_template": "{document}\n{candidate}",
            "response_schema": {"required": "not-an-array"},
        },
    )
    with pytest.raises(ValueError, match="judge.response_schema is not a valid JSON Schema"):
        load_config(path)
