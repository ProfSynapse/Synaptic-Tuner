"""Actual example/public APIs/fixed child; model responses are CPU fixtures."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import SynthChat
import yaml

import test_environment_admission_v1 as admission
import test_environment_sessions_v1 as sessions
import test_reference_evaluation_v1 as evaluation
import test_reference_spend_v1 as spend
from shared.llm.usage import LLMCompletionV1, LLMStructuredV1, measured_usage
from synaptic_tuner.api.v1.data_facade import DataMode

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/nontraining/qualification.py"
spec = importlib.util.spec_from_file_location("nontraining_example", EXAMPLE)
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    spend._no_network.__wrapped__(monkeypatch)


class Measured(sessions.ScriptedClient):
    provider_name = "openrouter"

    def __init__(self, responses, scores, missing=False):
        super().__init__(responses, scores)
        self.calls = 0
        self.missing = missing

    def chat(self, *args, **kwargs):
        result = super().chat(*args, **kwargs)
        self.calls += 1
        return LLMCompletionV1(
            result.text, None if self.missing else measured_usage(3, 5)
        )

    def structured_output(self, *args, **kwargs):
        result = super().structured_output(*args, **kwargs)
        self.calls += 1
        return LLMStructuredV1(result.value, measured_usage(3, 5))


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(
        admission, "ROOT", Path(SynthChat.__file__).resolve().parent.parent
    )
    original = admission.harness.__wrapped__(tmp_path)
    chosen = sessions.selected.__wrapped__(
        original, tmp_path, monkeypatch, SimpleNamespace(param="worker")
    )
    _, request, ports, host, _, scenario, _, _, _, _ = chosen
    environment = {
        "fixture": {"files": {"declared.txt": "bound fixture"}},
        "assertions": [
            {"type": "file_contains", "path": "declared.txt", "text": "bound fixture"}
        ],
    }
    scenario.write_text(
        yaml.safe_dump(
            {
                "scenarios": {
                    "selected": {
                        "type": "behavioral",
                        "environment_execution": "required",
                        "environment": environment,
                        "prompts": {
                            "system": "system",
                            "user": "user",
                            "assistant": "reply",
                        },
                        "rubrics": {"response": ["repair"]},
                    }
                }
            }
        )
    )
    sessions.update_host(
        host,
        execution={
            "tool_action_hints": {},
            "candidate_projection": {
                "response": ["conversations", -1],
                "system": ["conversations", 0, "content"],
                "environment": ["metadata", "resolved_environment_config"],
            },
        },
    )
    files = {}
    for ref, raw in host.raw.items():
        path = tmp_path / (ref + ".json")
        path.write_bytes(raw)
        files[ref] = str(path)
    evaluation_root = tmp_path / "evaluation"
    evaluation_root.mkdir()
    evaluation._write_config(evaluation_root)
    root = tmp_path / "private"
    generate = replace(
        request,
        request_id="generate",
        backend="openrouter",
        model="data-model",
        output_ref=str(root / "generated.jsonl"),
    )
    improve = replace(
        generate,
        request_id="improve",
        mode=DataMode.IMPROVE,
        scenarios=(),
        input_dataset_ref=generate.output_ref,
        output_ref=str(root / "improved.jsonl"),
        rubric_refs=("repair",),
    )
    eval_requests = [
        replace(
            evaluation._request(
                "alpha.yaml", backend="openrouter", request_id="evaluation-" + str(i)
            ),
            case_limit=2,
            model=evaluation.EvaluationModelRef(
                "evaluation-" + str(i), "fixture-revision"
            ),
        ).to_dict()
        for i in range(2)
    ]
    paths = [*map(Path, files.values())]
    for directory in (
        ports.config_dir,
        ports.scenarios_dir,
        ports.rubrics_dir,
        evaluation_root / "config",
    ):
        paths.extend(
            p for p in directory.rglob("*") if p.suffix in {".yaml", ".yml", ".json"}
        )
    config = {
        "project_ref": request.project_ref,
        "state_root": str(root),
        "data": {
            "config_dir": str(ports.config_dir),
            "scenarios_dir": str(ports.scenarios_dir),
            "rubrics_dir": str(ports.rubrics_dir),
            "environment_binding": host.resolve(
                request.environment_selection, project_ref=request.project_ref
            ).to_dict(),
            "environment_files": files,
            "generate": generate.to_dict(),
            "improve": improve.to_dict(),
        },
        "evaluation": {
            "config_root": str(evaluation_root / "config"),
            "requests": eval_requests,
        },
        "provider": {
            "backend": "openrouter",
            "credential": {"provider": "env", "name": "EXAMPLE_KEY"},
            "account_ref": "fixture-account",
            "namespace_ref": "fixture-namespace",
        },
        "bounds": {
            "max_calls": 32,
            "max_output_tokens": 4096,
            "max_request_bytes": 1048576,
            "max_response_bytes": 1048576,
            "timeout_seconds": 30,
            "duration_seconds": 120,
        },
        "inputs": [
            {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(set(paths))
        ],
    }
    return consumer.validate_config(config), ports, scenario


def install_clients(monkeypatch, missing=False):
    clients = [
        Measured(
            [
                "bound system",
                "bound user",
                "initial",
                json.dumps({"role": "assistant", "content": "generated revised"}),
            ],
            [0.1, 0.95, 0.95],
            missing,
        ),
        Measured(
            [json.dumps({"role": "assistant", "content": "improved revised"})],
            [0.1, 0.95, 0.95],
        ),
    ]
    created, credentials = [], []

    class Secrets:
        def resolve(self, ref):
            credentials.append(ref.name)
            return "synthetic-key"

    def create(key, model, **kwargs):
        assert key == "synthetic-key" and kwargs["max_response_bytes"] == 1048576
        client = clients[len(created)]
        created.append(client)
        return client

    monkeypatch.setattr(consumer, "OpenRouterClient", create)
    backend = spend.MeteredScriptedBackend()
    return Secrets(), created, credentials, backend


@pytest.mark.parametrize("reverse_models", [False, True])
def test_actual_consumer_join_unchanged_bytes_and_cold_readback(
    configured, monkeypatch, reverse_models
):
    config, ports, scenario = configured
    if reverse_models:
        config["evaluation"]["requests"].reverse()
    secrets, created, credentials, backend = install_clients(monkeypatch)
    preflight = consumer.plan(config, backends={"openrouter": backend})
    assert (
        preflight["ready"]
        and preflight["improvement"] == "deferred_until_generated_bytes"
    )
    assert not credentials and not created and not backend.opened
    reads = []
    original_read = consumer.WorkerEnvironmentRuntime.read_text

    def read(runtime, path):
        result = original_read(runtime, path)
        reads.append((path, result))
        return result

    monkeypatch.setattr(consumer.WorkerEnvironmentRuntime, "read_text", read)
    result = consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    assert len(result["data"]) == len(result["evaluation"]) == 2
    assert all(
        item["state"] == "succeeded" for item in result["data"] + result["evaluation"]
    )
    generated = Path(config["data"]["generate"]["output_ref"])
    raw = generated.read_bytes()
    row = json.loads(raw)
    assert row["metadata"]["resolved_environment_config"]["fixture"]["files"] == {
        "declared.txt": "bound fixture"
    }
    assert reads and all(value == ("declared.txt", "bound fixture") for value in reads)
    assert len(created) == 2 and created[1].chat_calls == 1
    assert "generated revised" in json.dumps(created[1].prompts)
    assert result["comparison"] is not None
    assert result["results"][0]["model"] == config["evaluation"]["requests"][0]["model"]
    assert result["results"][1]["model"] == config["evaluation"]["requests"][1]["model"]
    assert result["comparison"]["reference_run"] == result["results"][0]["run"]
    assert result["comparison"]["candidate_run"] == result["results"][1]["run"]
    before = (len(credentials), len(backend.opened), tuple(c.calls for c in created))
    with pytest.raises(FileExistsError):
        consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    for directory in (
        ports.config_dir,
        ports.scenarios_dir,
        ports.rubrics_dir,
        Path(config["evaluation"]["config_root"]),
    ):
        shutil.rmtree(directory)

    def forbidden(*args, **kwargs):
        pytest.fail("retained read reopened client or secret resolver")

    monkeypatch.setattr(consumer.NamedSecrets, "resolve", forbidden)
    monkeypatch.setattr(consumer.DataClients, "create", forbidden)
    backend.open = backend.requirement = forbidden
    cold = consumer.readback(config["state_root"])
    cold["comparison"].pop("generated_at")
    result["comparison"].pop("generated_at")
    assert cold == result and generated.read_bytes() == raw
    assert before == (
        len(credentials),
        len(backend.opened),
        tuple(c.calls for c in created),
    )


def test_unknown_generation_stops_downstream_and_cannot_replay(configured, monkeypatch):
    config, _, _ = configured
    secrets, created, _, backend = install_clients(monkeypatch, missing=True)
    result = consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    assert len(result["data"]) == 1 and not result["evaluation"]
    assert result["data"][0]["diagnostic_code"] == "spend_indeterminate"
    assert created[0].calls == 1 and not backend.opened
    with pytest.raises(FileExistsError):
        consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    assert created[0].calls == 1


def test_committed_source_mutation_rejects_before_state_or_credentials(
    configured, monkeypatch
):
    config, _, scenario = configured
    scenario.write_text("changed: true")

    def forbidden(*args, **kwargs):
        pytest.fail("opened credential before admission")

    monkeypatch.setattr(consumer.NamedSecrets, "resolve", forbidden)
    with pytest.raises(ValueError, match="commitment mismatch"):
        consumer.execute(config)
    assert not Path(config["state_root"]).exists()


def test_between_stage_mutation_is_not_newly_admitted(configured, monkeypatch):
    config, _, scenario = configured
    secrets, created, _, backend = install_clients(monkeypatch)
    original = consumer.OpenRouterClient

    def create(*args, **kwargs):
        client = original(*args, **kwargs)
        chat = client.chat

        def mutate(*args, **kwargs):
            result = chat(*args, **kwargs)
            if client.chat_calls == 4:
                scenario.write_text("changed: during_generation")
            return result

        client.chat = mutate
        return client

    monkeypatch.setattr(consumer, "OpenRouterClient", create)
    with pytest.raises(ValueError, match="commitment mismatch"):
        consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    assert len(created) == 1 and not backend.opened
    result = consumer.readback(config["state_root"])
    assert result["data"][0]["state"] == "succeeded" and len(result["data"]) == 1


@pytest.mark.parametrize("max_calls", [32, 14])
def test_cli_plan_execute_and_json_readback_actual_adapters(
    configured, monkeypatch, capsys, tmp_path, max_calls
):
    config, _, _ = configured
    config["bounds"]["max_calls"] = max_calls
    rubric = tmp_path / "judge-rubrics"
    rubric.mkdir()
    (rubric / "quality.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "quality",
                "description": "CPU quality fixture",
                "scope": "response",
                "pass_threshold": 0.8,
                "judge_prompt": "Check {response}",
                "output_schema": {
                    "type": "object",
                    "properties": {"quality_score": {"type": "number"}},
                },
            }
        )
    )
    config["evaluation"]["judge"] = {
        "model": {"model_ref": "judge-model", "model_revision": "fixture-revision"},
        "rubrics_dir": str(rubric),
        "rubric_refs": ["quality"],
        "config": {"max_tokens": 64, "temperature": 0, "reasoning_effort": None},
    }
    raw = (rubric / "quality.yaml").read_bytes()
    config["inputs"].append(
        {
            "path": str(rubric / "quality.yaml"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    )
    data_clients = [
        Measured(
            [
                "bound system",
                "bound user",
                "initial",
                json.dumps({"role": "assistant", "content": "generated revised"}),
            ],
            [0.1, 0.95, 0.95],
        ),
        Measured(
            [json.dumps({"role": "assistant", "content": "improved revised"})],
            [0.1, 0.95, 0.95],
        ),
    ]
    all_calls, opens = [], []

    class RemoteFixture:
        provider_name = "openrouter"

        def __init__(self, model):
            self.model_name = model

        def chat(self, messages, **kwargs):
            all_calls.append((self.model_name, "chat"))
            return LLMCompletionV1(
                evaluation.RESPONSES[messages[-1]["content"]], measured_usage(3, 5)
            )

        def structured_output(self, *args, **kwargs):
            all_calls.append((self.model_name, "judge"))
            return LLMStructuredV1(
                {"quality_score": 0.95, "overall_feedback": "CPU fixture"},
                measured_usage(3, 5),
            )

    def create(key, model, **kwargs):
        assert key == "synthetic-key" and kwargs["max_response_bytes"] == 1048576
        opens.append(model)
        return data_clients.pop(0) if model == "data-model" else RemoteFixture(model)

    monkeypatch.setenv("EXAMPLE_KEY", "synthetic-key")
    monkeypatch.setattr(consumer, "OpenRouterClient", create)
    monkeypatch.setattr("shared.llm.providers.openrouter.OpenRouterClient", create)
    original_take = consumer.DataClients.take_call
    ticks = []

    def take(factory):
        original_take(factory)
        ticks.append(factory.calls)

    monkeypatch.setattr(consumer.DataClients, "take_call", take)
    config_path = tmp_path / "consumer.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert consumer.main(["plan", str(config_path)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["ready"] and not opens and not all_calls
    assert consumer.main(["execute", str(config_path)]) == 0
    executed = json.loads(capsys.readouterr().out)
    if max_calls == 32:
        assert len(executed["spend"]) == 4 and all(
            e["state"] == "found" for e in executed["spend"]
        )
        assert len(all_calls) == 8 and len(ticks) == 19
        assert executed["comparison"]["candidates"][0]["comparable"] is False
    else:
        assert len(executed["spend"]) == 3 and len(all_calls) == 3 and len(ticks) == 14
        assert executed["evaluation"][0]["diagnostic_code"] == "spend_indeterminate"
        assert executed["comparison"] is None and "evaluation-1" not in opens
    assert "synthetic-key" not in json.dumps(executed)

    def forbidden(*args, **kwargs):
        pytest.fail("CLI retained read opened model or secret")

    monkeypatch.setattr(consumer.NamedSecrets, "resolve", forbidden)
    monkeypatch.setattr(consumer, "OpenRouterClient", forbidden)
    monkeypatch.setattr("shared.llm.providers.openrouter.OpenRouterClient", forbidden)
    assert consumer.main(["readback", config["state_root"]]) == 0
    retained = json.loads(capsys.readouterr().out)
    assert (
        retained["data"] == executed["data"]
        and retained["results"] == executed["results"]
    )
    assert consumer.main(["execute", str(config_path)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "diagnostic_code": "nontraining_consumer_failed"
    }


def test_retained_data_tampering_refuses_actual_byte_readback(configured, monkeypatch):
    config, _, _ = configured
    secrets, _, _, backend = install_clients(monkeypatch)
    consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    target = Path(config["data"]["improve"]["output_ref"])
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="artifact commitment mismatch"):
        consumer.readback(config["state_root"])


def test_dispatch_budget_exhaustion_does_not_open_downstream_clients(
    configured, monkeypatch
):
    config, _, _ = configured
    config["bounds"]["max_calls"] = 1
    secrets, created, _, backend = install_clients(monkeypatch)
    result = consumer.execute(config, secrets=secrets, backends={"openrouter": backend})
    assert len(created) == 1 and created[0].calls == 1 and not backend.opened
    assert len(result["data"]) == 1 and not result["evaluation"]
    assert result["spend"][0]["state"] == "indeterminate"


@pytest.mark.parametrize("requested", [None, 64, 65, True, 0])
def test_configured_token_ceiling_is_explicit_before_inner_call(configured, requested):
    config, _, _ = configured
    config["bounds"]["max_output_tokens"] = 64
    factory = consumer.DataClients(config, object(), consumer.time.monotonic() + 30)
    calls = []

    class Client:
        def chat(self, messages, **kwargs):
            calls.append(kwargs)
            return LLMCompletionV1("CPU fixture", measured_usage(3, 5))

    client = consumer._BudgetClient(Client(), factory)
    if requested is None or type(requested) is int and requested == 64:
        client.chat([], max_tokens=requested)
        assert calls[0]["max_tokens"] == 64 and factory.calls == 1
    else:
        with pytest.raises(ValueError, match="token ceiling"):
            client.chat([], max_tokens=requested)
        assert not calls and factory.calls == 0
