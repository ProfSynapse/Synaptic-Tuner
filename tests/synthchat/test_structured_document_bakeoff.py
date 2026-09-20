from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import pytest
import yaml

import SynthChat.scripts.structured_document_bakeoff as batch_module
from SynthChat.scripts.structured_document_bakeoff import (
    collect_openrouter_batch,
    judge_existing,
    load_config,
    observe_openrouter_batch,
    run_bakeoff,
    submit_openrouter_batch,
)
from shared.llm.providers.openrouter import (
    OpenRouterBatchRejectedError,
    OpenRouterBatchSubmissionAmbiguousError,
)
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


class FakeBatchClient:
    def __init__(self, model, submit_calls, observations):
        self.model = model
        self.submit_calls = submit_calls
        self.observations = observations

    def submit_batch(self, items, *, endpoint):
        self.submit_calls.append({"model": self.model, "items": items, "endpoint": endpoint})
        return {"id": f"batch-{self.model.rsplit('/', 1)[-1]}", "status": "validating", "created_at": 123}

    def observe_batch(self, batch_id):
        return self.observations[batch_id]


def _batch_config(tmp_path, *, model_ids=None, document_count=2, schema=None, max_requests=20):
    documents = []
    for index in range(document_count):
        path = tmp_path / f"doc-{index}.md"
        path.write_text(f"---\ntitle: Hidden {index}\n---\nDocument {index} body.", encoding="utf-8")
        documents.append(
            {
                "id": f"doc-{index}",
                "source_path": path.name,
                "strip_yaml_frontmatter": True,
                "metadata": {"sequence": index},
            }
        )
    config = {
        "documents": documents,
        "prompt_template": "Metadata={metadata}\nText={document}",
        "response_schema": schema
        or {
            "type": "object",
            "properties": {"outline": {"type": "string"}},
            "required": ["outline"],
            "additionalProperties": False,
        },
        "models": model_ids or ["openai/luna"],
        "temperature": 0.1,
        "max_tokens": 444,
        "output_dir": "batch-results",
        "batch": {"max_requests": max_requests},
    }
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _fixed_now():
    return datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


def test_batch_submit_persists_rendered_request_mapping_and_resumes_without_resubmission(tmp_path):
    config_path = _batch_config(tmp_path, model_ids=["openai/luna", "openai/second"])
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert len(submit_calls) == 2
    assert all(call["endpoint"] == "/v1/chat/completions" for call in submit_calls)
    assert [len(call["items"]) for call in submit_calls] == [2, 2]
    first_body = submit_calls[0]["items"][0]["body"]
    assert first_body["model"] == "openai/luna"
    assert first_body["response_format"]["type"] == "json_schema"
    assert "Metadata={\"sequence\": 0}" in first_body["messages"][0]["content"]
    assert "title: Hidden" not in first_body["messages"][0]["content"]
    custom_ids = [item["custom_id"] for call in submit_calls for item in call["items"]]
    assert len(custom_ids) == len(set(custom_ids)) == 4
    assert state["spec_digest"]
    assert all(batch["provider_batch_id"] for batch in state["batches"])
    assert all(batch["custom_ids"] for batch in state["batches"])
    assert all(
        mapping["request_digest"]
        for batch in state["batches"]
        for mapping in batch["custom_ids"].values()
    )

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert len(submit_calls) == 2


def test_batch_submit_changed_rendered_input_rejects_existing_state(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, {})

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    (tmp_path / "doc-0.md").write_text("Changed body.", encoding="utf-8")

    with pytest.raises(ValueError, match="spec digest does not match"):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert len(submit_calls) == 1


def test_batch_plan_enforces_explicit_request_bound_and_rejects_batch_slug(tmp_path):
    bounded = _batch_config(tmp_path, document_count=2, max_requests=1)
    with pytest.raises(ValueError, match="exceeds configured max_requests"):
        submit_openrouter_batch(bounded, client_factory=lambda **_: None, now=_fixed_now)

    slugged_dir = tmp_path / "slugged"
    slugged_dir.mkdir()
    slugged = _batch_config(slugged_dir, document_count=1, model_ids=["openai/luna:batch"])
    with pytest.raises(ValueError, match="base model id"):
        submit_openrouter_batch(slugged, client_factory=lambda **_: None, now=_fixed_now)


def test_batch_submit_ambiguous_failure_leaves_marker_and_never_blindly_retries(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    calls = []

    class AmbiguousClient:
        def submit_batch(self, items, *, endpoint):
            calls.append(items)
            raise RuntimeError("connection vanished after POST")

    def factory(*, config_defaults):
        return AmbiguousClient()

    with pytest.raises(RuntimeError, match="connection vanished"):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["batches"][0]["submission_status"] == "submitting"

    with pytest.raises(RuntimeError, match="ambiguous prior outcome"):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert len(calls) == 1


def test_batch_submit_exclusive_claim_allows_only_one_concurrent_post(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    entered = threading.Event()
    release = threading.Event()
    post_calls = []
    factory_calls = []
    first_errors = []

    class BlockingClient:
        def submit_batch(self, items, *, endpoint):
            post_calls.append(items)
            entered.set()
            assert release.wait(timeout=5)
            return {"id": "batch-luna", "status": "validating"}

    def factory(*, config_defaults):
        factory_calls.append(config_defaults["model"])
        return BlockingClient()

    def first_submit():
        try:
            submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
        except Exception as exc:  # pragma: no cover - asserted below
            first_errors.append(exc)

    thread = threading.Thread(target=first_submit)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="exclusively claimed"):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_errors == []
    assert len(post_calls) == 1
    assert factory_calls == ["openai/luna"]


def test_batch_submit_stale_claim_fails_closed_without_takeover(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    state_dir = tmp_path / "batch-results"
    state_dir.mkdir()
    claim_path = state_dir / "openrouter-batch-state.json.lock"
    claim_path.write_text('{"owner":"crashed"}', encoding="utf-8")
    factory_calls = []

    def factory(*, config_defaults):
        factory_calls.append(config_defaults)
        return object()

    with pytest.raises(RuntimeError, match="exclusively claimed"):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert claim_path.read_text(encoding="utf-8") == '{"owner":"crashed"}'
    assert factory_calls == []


@pytest.mark.parametrize("mode", ["factory_failure", "unsupported_client"])
def test_batch_submit_local_client_preflight_failure_remains_pending_and_retryable(tmp_path, mode):
    config_path = _batch_config(tmp_path, document_count=1)

    if mode == "factory_failure":
        def factory(*, config_defaults):
            raise RuntimeError("local configuration failed")
        expected = RuntimeError
    else:
        def factory(*, config_defaults):
            return object()
        expected = TypeError

    with pytest.raises(expected):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["batches"][0]["submission_status"] == "pending"
    assert "submission_attempted_at" not in state["batches"][0]


def test_batch_submit_confirmed_4xx_persists_history_and_explicit_submit_retries(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    calls = []

    class RejectedClient:
        def submit_batch(self, items, *, endpoint):
            calls.append(items)
            raise OpenRouterBatchRejectedError(400, "unsupported")

    with pytest.raises(OpenRouterBatchRejectedError):
        submit_openrouter_batch(
            config_path,
            client_factory=lambda **_: RejectedClient(),
            now=_fixed_now,
        )

    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = state["batches"][0]
    assert batch["submission_status"] == "rejected"
    assert batch["rejection_history"] == [
        {
            "type": "OpenRouterBatchRejectedError",
            "status_code": 400,
            "code": "unsupported",
            "rejected_at": "2026-09-20T12:00:00Z",
        }
    ]

    with pytest.raises(OpenRouterBatchRejectedError):
        submit_openrouter_batch(
            config_path,
            client_factory=lambda **_: RejectedClient(),
            now=_fixed_now,
        )
    after_second_rejection = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(after_second_rejection["batches"][0]["rejection_history"]) == 2

    class SuccessfulClient:
        def submit_batch(self, items, *, endpoint):
            calls.append(items)
            return {"id": "batch-luna", "status": "validating"}

    retried = submit_openrouter_batch(
        config_path,
        client_factory=lambda **_: SuccessfulClient(),
        now=_fixed_now,
    )
    assert retried["batches"][0]["submission_status"] == "submitted"
    assert len(retried["batches"][0]["rejection_history"]) == 2
    assert len(calls) == 3


def test_batch_submit_mixed_models_retries_only_rejected_model(tmp_path):
    config_path = _batch_config(
        tmp_path,
        document_count=1,
        model_ids=["openai/model-a", "openai/model-b"],
    )
    posts = []
    b_attempts = 0

    class MixedClient:
        def __init__(self, model):
            self.model = model

        def submit_batch(self, items, *, endpoint):
            nonlocal b_attempts
            posts.append(self.model)
            if self.model == "openai/model-b":
                b_attempts += 1
                if b_attempts == 1:
                    raise OpenRouterBatchRejectedError(429, "rate_limit")
            return {"id": f"batch-{self.model[-1]}", "status": "validating"}

    def factory(*, config_defaults):
        return MixedClient(config_defaults["model"])

    with pytest.raises(OpenRouterBatchRejectedError):
        submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    after_first = json.loads(state_path.read_text(encoding="utf-8"))
    assert [batch["submission_status"] for batch in after_first["batches"]] == ["submitted", "rejected"]

    resumed = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert posts == ["openai/model-a", "openai/model-b", "openai/model-b"]
    assert [batch["submission_status"] for batch in resumed["batches"]] == ["submitted", "submitted"]
    assert resumed["batches"][0]["rejection_history"] == []
    assert resumed["batches"][1]["rejection_history"][0]["status_code"] == 429


@pytest.mark.parametrize("outcome", ["timeout", "invalid_success"])
def test_batch_submit_ambiguous_provider_outcomes_remain_submitting(tmp_path, outcome):
    config_path = _batch_config(tmp_path, document_count=1)

    class AmbiguousClient:
        def submit_batch(self, items, *, endpoint):
            if outcome == "timeout":
                raise OpenRouterBatchSubmissionAmbiguousError("timeout")
            return {"status": "validating"}

    with pytest.raises(OpenRouterBatchSubmissionAmbiguousError):
        submit_openrouter_batch(
            config_path,
            client_factory=lambda **_: AmbiguousClient(),
            now=_fixed_now,
        )

    state = json.loads(
        (tmp_path / "batch-results" / "openrouter-batch-state.json").read_text(encoding="utf-8")
    )
    assert state["batches"][0]["submission_status"] == "submitting"
    assert state["batches"][0]["ambiguity"]["type"] == "OpenRouterBatchSubmissionAmbiguousError"


def test_batch_submit_persist_failure_after_success_keeps_durable_ambiguous_marker(tmp_path, monkeypatch):
    config_path = _batch_config(tmp_path, document_count=1)
    original_write = batch_module._write_json
    writes = []
    posts = []

    def failing_third_write(path, payload):
        writes.append(path)
        if len(writes) == 3:
            raise OSError("simulated durable replace failure")
        return original_write(path, payload)

    class SuccessfulClient:
        def submit_batch(self, items, *, endpoint):
            posts.append(items)
            return {"id": "batch-luna", "status": "validating"}

    monkeypatch.setattr(batch_module, "_write_json", failing_third_write)
    with pytest.raises(OSError, match="durable replace failure"):
        submit_openrouter_batch(
            config_path,
            client_factory=lambda **_: SuccessfulClient(),
            now=_fixed_now,
        )
    monkeypatch.setattr(batch_module, "_write_json", original_write)

    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["batches"][0]["submission_status"] == "submitting"
    with pytest.raises(RuntimeError, match="ambiguous prior outcome"):
        submit_openrouter_batch(
            config_path,
            client_factory=lambda **_: SuccessfulClient(),
            now=_fixed_now,
        )
    assert len(posts) == 1


def test_batch_observe_persists_provider_status_counts_and_timestamps(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "in_progress",
        "in_progress_at": 456,
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "results": None,
    }

    state = observe_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert state["batches"][0]["provider_status"] == "in_progress"
    assert state["batches"][0]["provider_timestamps"] == {"in_progress_at": 456}
    assert state["batches"][0]["request_counts"]["total"] == 1


def test_batch_collect_reconciles_ids_and_locally_admits_schema_before_success(tmp_path):
    config_path = _batch_config(tmp_path, document_count=3)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    custom_ids = list(state["batches"][0]["custom_ids"])
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "completed_at": 789,
        "request_counts": {"total": 3, "completed": 2, "failed": 1},
        "results": [
            {
                "custom_id": custom_ids[0],
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": json.dumps({"outline": "valid"})}}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                    },
                },
                "error": None,
            },
            {
                "custom_id": custom_ids[1],
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": json.dumps({"outline": 42})}}]},
                },
                "error": None,
            },
            {
                "custom_id": custom_ids[2],
                "response": None,
                "error": {"code": "provider_error", "message": "upstream failed"},
            },
        ],
    }

    summary = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert summary["ready"] is True
    assert summary["written"] == 3
    assert summary["succeeded"] == 1
    assert summary["item_failed"] == 2
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (tmp_path / "batch-results").glob("doc-*.json")]
    assert len(records) == 3
    valid = next(record for record in records if record["generation"]["payload"] is not None)
    assert valid["generation"]["payload"] == {"outline": "valid"}
    assert valid["generation"]["usage"]["input_tokens"] == 12
    schema_invalid = next(
        record for record in records if record["generation"]["error"] and record["generation"]["error"]["type"] == "ValueError"
    )
    assert "JSON Schema validation" in schema_invalid["generation"]["error"]["message"]
    provider_failed = next(record for record in records if record["generation"].get("provider_error"))
    assert provider_failed["generation"]["provider_error"]["code"] == "provider_error"
    assert "message" not in provider_failed["generation"]["provider_error"]


def test_batch_collect_in_progress_reserves_no_final_artifacts_then_completed_succeeds(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    custom_id = next(iter(state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "in_progress",
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "results": None,
    }

    pending = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    output_dir = tmp_path / "batch-results"
    assert pending["ready"] is False
    assert pending["written"] == 0
    assert pending["pending_batches"][0]["status"] == "in_progress"
    assert "manifest_path" not in pending
    assert not (output_dir / "batch-manifest.json").exists()
    assert list(output_dir.glob("doc-*.json")) == []

    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "results": [
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": '{"outline":"done"}'}}]},
                },
                "error": None,
            }
        ],
    }
    completed = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert completed["ready"] is True
    assert completed["succeeded"] == 1
    assert (output_dir / "batch-manifest.json").exists()
    assert len(list(output_dir.glob("doc-*.json"))) == 1


def test_batch_collect_is_immutable_across_identical_and_changed_recollection(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    custom_id = next(iter(state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "results": [
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": '{"outline":"first"}'}}]},
                },
                "error": None,
            }
        ],
    }

    first = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    result_path = next((tmp_path / "batch-results").glob("doc-*.json"))
    manifest_path = tmp_path / "batch-results" / "batch-manifest.json"
    first_result_bytes = result_path.read_bytes()
    first_manifest_bytes = manifest_path.read_bytes()

    second = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert second == first
    assert result_path.read_bytes() == first_result_bytes
    assert manifest_path.read_bytes() == first_manifest_bytes

    observations["batch-luna"]["results"][0]["response"]["body"]["choices"][0]["message"]["content"] = (
        '{"outline":"changed"}'
    )
    with pytest.raises(ValueError, match="immutable artifact already exists"):
        collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert result_path.read_bytes() == first_result_bytes
    assert manifest_path.read_bytes() == first_manifest_bytes


def test_batch_collect_rejects_non_finite_structured_content_and_provider_errors(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    custom_id = next(iter(state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "results": [
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": '{"outline": NaN}'}}]},
                },
                "error": None,
            }
        ],
    }
    summary = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert summary["succeeded"] == 0
    record = json.loads(next((tmp_path / "batch-results").glob("doc-*.json")).read_text(encoding="utf-8"))
    assert record["generation"]["payload"] is None
    assert "non-finite JSON number" in record["generation"]["error"]["message"]

    other_dir = tmp_path / "provider-nan"
    other_dir.mkdir()
    other_config = _batch_config(other_dir, document_count=1)
    other_state = submit_openrouter_batch(other_config, client_factory=factory, now=_fixed_now)
    other_id = next(iter(other_state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "results": [
            {"custom_id": other_id, "response": None, "error": {"code": float("inf")}},
        ],
    }
    with pytest.raises(ValueError, match="finite JSON"):
        collect_openrouter_batch(other_config, client_factory=factory, now=_fixed_now)
    assert list((other_dir / "batch-results").glob("doc-*.json")) == []


def test_batch_item_error_omits_non_identifier_provider_code():
    private_code = "PRIVATE PROMPT / inject into artifact"

    closed = batch_module._closed_provider_error(
        {"type": "provider error with spaces", "code": private_code, "message": "private"}
    )

    assert closed == {"type": "provider_error", "code": None, "status_code": None}
    assert private_code not in json.dumps(closed)


def test_batch_collect_rejects_missing_duplicate_or_unexpected_custom_ids(tmp_path):
    config_path = _batch_config(tmp_path, document_count=2)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    first_id = next(iter(state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "results": [
            {"custom_id": first_id, "response": {}, "error": None},
            {"custom_id": first_id, "response": {}, "error": None},
        ],
    }

    with pytest.raises(ValueError, match="duplicate custom_id"):
        collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)


def test_batch_collect_rejects_missing_and_unexpected_custom_ids(tmp_path):
    config_path = _batch_config(tmp_path, document_count=2)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    state = submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    first_id = next(iter(state["batches"][0]["custom_ids"]))
    observations["batch-luna"] = {
        "id": "batch-luna",
        "status": "completed",
        "results": [{"custom_id": first_id, "response": {}, "error": None}],
    }
    with pytest.raises(ValueError, match=r"missing=\["):
        collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    observations["batch-luna"]["results"].append(
        {"custom_id": "unexpected-id", "response": {}, "error": None}
    )
    with pytest.raises(ValueError, match="unexpected=.*unexpected-id"):
        collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update({"unexpected": True}),
        lambda state: state["batches"][0].update({"provider_batch_id": None}),
        lambda state: state["batches"][0].update({"provider_status": "unknown"}),
        lambda state: next(iter(state["batches"][0]["custom_ids"].values())).update({"model": "other"}),
        lambda state: state["batches"][0]["rejection_history"].append(
            {"type": "bad", "code": "private code with spaces", "status_code": 429, "rejected_at": "now"}
        ),
    ],
)
def test_batch_state_corruption_fails_closed_before_client_construction(tmp_path, mutate):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}
    factory_calls = []

    def factory(*, config_defaults):
        factory_calls.append(config_defaults["model"])
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    factory_calls.clear()
    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mutate(state)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError):
        observe_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert factory_calls == []


def test_batch_state_non_finite_json_fails_closed_before_client_construction(tmp_path):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    factory_calls = []

    def factory(*, config_defaults):
        factory_calls.append(config_defaults["model"])
        return FakeBatchClient(config_defaults["model"], submit_calls, {})

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    factory_calls.clear()
    state_path = tmp_path / "batch-results" / "openrouter-batch-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["batches"][0]["request_counts"] = {"total": float("-inf")}
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        observe_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    assert factory_calls == []


@pytest.mark.parametrize("writer_name", ["_write_json", "_write_json_immutable"])
def test_json_artifact_writers_reject_non_finite_values_without_creating_target(tmp_path, writer_name):
    target = tmp_path / f"{writer_name}.json"
    writer = getattr(batch_module, writer_name)

    with pytest.raises(ValueError):
        writer(target, {"bad": float("nan")})
    assert not target.exists()


@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
def test_batch_collect_exposes_terminal_batch_failures_without_writing_results(tmp_path, status):
    config_path = _batch_config(tmp_path, document_count=1)
    submit_calls = []
    observations = {}

    def factory(*, config_defaults):
        return FakeBatchClient(config_defaults["model"], submit_calls, observations)

    submit_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)
    observations["batch-luna"] = {"id": "batch-luna", "status": status, "results": None}

    summary = collect_openrouter_batch(config_path, client_factory=factory, now=_fixed_now)

    assert summary["ready"] is False
    assert summary["terminal_failures"] == [
        {"model": "openai/luna", "provider_batch_id": "batch-luna", "status": status}
    ]
    assert summary["written"] == 0
