"""Installed CPU joins; scripted usage is not a live account/model qualification."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import SynthChat

import test_environment_admission_v1 as admission
import test_environment_sessions_v1 as sessions
import test_reference_evaluation_v1 as evaluation
import test_reference_spend_v1 as spend
from shared.llm.usage import LLMCompletionV1, LLMStructuredV1, measured_usage
from synaptic_tuner.api.v1.data_facade import (
    DataAPI,
    DataMode,
    DataRunState,
    DatasetValidateRequest,
)
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationModelRef,
    EvaluationResultRequest,
    EvaluationRunState,
)
from synaptic_tuner.api.v1.reference.data import ReferenceDataOperationsV1
from synaptic_tuner.api.v1.reference.evaluation import DirectoryEvaluationArtifactSinkV1
from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
from synaptic_tuner.api.v1.reference.spend import SpendEffectState


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    spend._no_network.__wrapped__(monkeypatch)


class MeasuredRepairClient(sessions.ScriptedClient):
    provider_name = "openrouter"

    def __init__(self, responses, scores, *, missing=False):
        super().__init__(responses, scores)
        self.missing = missing
        self.calls = 0

    def chat(self, messages, **kwargs):
        result = super().chat(messages, **kwargs)
        self.calls += 1
        if self.missing and self.calls == 1:
            return result
        return LLMCompletionV1(result.text, measured_usage(*spend.PER_CALL))

    def structured_output(self, messages, schema, **kwargs):
        result = super().structured_output(messages, schema, **kwargs)
        self.calls += 1
        return LLMStructuredV1(result.value, measured_usage(*spend.PER_CALL))


@pytest.mark.parametrize("missing_usage", [False, True])
def test_measured_data_environment_generate_improve_and_cold_reads(
    tmp_path, monkeypatch, missing_usage
):
    monkeypatch.setattr(
        admission, "ROOT", Path(SynthChat.__file__).resolve().parent.parent
    )
    base = admission.harness.__wrapped__(tmp_path)
    selected = sessions.selected.__wrapped__(
        base, tmp_path, monkeypatch, SimpleNamespace(param="worker")
    )
    (
        operations,
        request,
        ports,
        host,
        _,
        scenario,
        environment,
        scratch,
        runtimes,
        prepared,
    ) = selected
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "workflow.sqlite3"
    clients = []
    current = [None]

    class Factory:
        def spend_scope(self, *, backend):
            assert backend == "openrouter"
            return spend.DATA_SCOPE

        def create(self, **kwargs):
            clients.append(kwargs)
            return current[0]

    ports = replace(ports, llm_clients=Factory())
    with SQLiteDurableStoreV1(database) as store:
        operations = ReferenceDataOperationsV1(
            records=store, streams=store, clock=operations._clock, ports=ports
        )
        api = DataAPI(operations)
        request = replace(request, backend="openrouter")
        current[0] = MeasuredRepairClient(
            [
                "bound system",
                "bound user",
                json.dumps(sessions.response("files_fetch", path="result.txt")),
                json.dumps(
                    sessions.response("files_store", path="result.txt", content="fixed")
                ),
            ],
            [0.1, 0.95, 0.95],
            missing=missing_usage,
        )
        generation_client = current[0]
        plan = api.plan(request)
        assert api.preflight(plan).ready
        generated = api.start(plan)
        outcome = api.show(generated.run)
        assert outcome.rows_written == 1
        assert outcome.state is (
            DataRunState.FAILED if missing_usage else DataRunState.SUCCEEDED
        )
        assert outcome.diagnostic_code == (
            "spend_indeterminate" if missing_usage else None
        )
        spend._assert_artifacts_match_disk(outcome, Path(request.output_ref))
        assert generation_client.chat_calls == 4 and generation_client.judge_calls > 0
        assert any("missing_path" in prompt for prompt in generation_client.prompts)
        assert prepared[-1].last_evaluation["passed"] and not list(scratch.iterdir())
        (effect,) = spend._spend_records(store)
        assert effect.state is (
            SpendEffectState.INDETERMINATE if missing_usage else SpendEffectState.FOUND
        )
        if not missing_usage:
            assert (
                effect.usage.input_tokens == generation_client.calls * spend.PER_CALL[0]
            )
            assert (
                effect.usage.output_tokens
                == generation_client.calls * spend.PER_CALL[1]
            )
        else:
            assert effect.usage is None

        policy = json.loads(host.raw["policy"])
        policy["execution"]["candidate_projection"]["environment"] = ["environment"]
        sessions.update_host(host, execution=policy["execution"])
        seed = tmp_path / "seed.jsonl"
        seed.write_text(
            json.dumps(
                {
                    "conversations": [
                        {"role": "system", "content": "bound"},
                        sessions.response("files_fetch", path="result.txt"),
                    ],
                    "environment": environment,
                }
            )
            + "\n"
        )
        improvement = replace(
            request,
            request_id="improve",
            mode=DataMode.IMPROVE,
            scenarios=(),
            input_dataset_ref=str(seed),
            rubric_refs=("repair",),
            output_ref=str(tmp_path / "improved.jsonl"),
        )
        current[0] = MeasuredRepairClient(
            [
                json.dumps(
                    sessions.response("files_store", path="result.txt", content="fixed")
                )
            ],
            [0.1, 0.95, 0.95],
        )
        improve_client = current[0]
        improved = api.start(api.plan(improvement))
        improved_outcome = api.show(improved.run)
        assert improved_outcome.state is DataRunState.SUCCEEDED
        spend._assert_artifacts_match_disk(
            improved_outcome, Path(improvement.output_ref)
        )
        assert improve_client.chat_calls == 1 and improve_client.judge_calls > 0
        validation = api.validate(
            DatasetValidateRequest(request.project_ref, improvement.output_ref)
        )
        assert (
            validation.rows_checked == 1
            and validation.rows_valid == 1
            and not validation.findings
        )
        effects = spend._spend_records(store)
        found = next(
            item for item in effects if item.entity.run_id == improved.run.run_id
        )
        assert len(effects) == 2 and found.state is SpendEffectState.FOUND
        assert found.usage.input_tokens == improve_client.calls * spend.PER_CALL[0]
        assert found.usage.output_tokens == improve_client.calls * spend.PER_CALL[1]
        snapshots = {
            Path(path): Path(path).read_bytes()
            for path in (request.output_ref, improvement.output_ref)
        }
        assert not operations._environment_owners and not list(scratch.iterdir())
    scenario.unlink()
    seed.unlink()
    shutil.rmtree(ports.config_dir)
    shutil.rmtree(ports.rubrics_dir)
    before = (
        len(clients),
        len(runtimes),
        generation_client.calls,
        improve_client.calls,
    )
    current[0] = None
    with SQLiteDurableStoreV1(database, create=False) as store:
        cold = DataAPI(
            ReferenceDataOperationsV1(
                records=store, streams=store, clock=operations._clock, ports=ports
            )
        )
        assert cold.show(generated.run) == outcome
        assert cold.show(improved.run) == improved_outcome
        assert len(spend._spend_records(store)) == 2
        for path, raw in snapshots.items():
            assert path.read_bytes() == raw
            assert hashlib.sha256(raw).hexdigest() in {
                artifact.sha256
                for result in (outcome, improved_outcome)
                for artifact in result.artifacts
            }
    assert before == (
        len(clients),
        len(runtimes),
        generation_client.calls,
        improve_client.calls,
    )


class ComparableMeasuredClient(spend.MeteredScriptedClient):
    comparison_configuration = (
        evaluation.ComparisonScriptedClient.comparison_configuration
    )


class ComparableMeasuredBackend(spend.MeteredScriptedBackend):
    def open(self, model):
        self.opened.append(model)
        return ComparableMeasuredClient(self, self._responses, None, self._unmeasured)


@pytest.mark.parametrize("missing_usage", [False, True])
def test_measured_evaluation_reports_compare_after_sqlite_reopen(
    tmp_path, missing_usage
):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "workflow.sqlite3"
    sink = DirectoryEvaluationArtifactSinkV1(tmp_path / "sink")
    backend = ComparableMeasuredBackend()
    runs, results = [], []
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, _, _, _ = evaluation._host(
            tmp_path,
            backends={"openrouter": backend},
            records=store,
            streams=store,
            artifact_reader=sink,
        )
        for index in range(2):
            backend._unmeasured = ("Say hello",) if missing_usage and index == 1 else ()
            request = replace(
                evaluation._request(
                    "alpha.yaml", backend="openrouter", request_id=f"request-{index}"
                ),
                model=EvaluationModelRef(f"fixture-{index}", "revision"),
            )
            plan = api.evaluation.plan(request)
            assert api.evaluation.preflight(plan).ready
            run = api.evaluation.start(plan).run
            runs.append(run)
            outcome = api.evaluation.show(run)
            assert outcome.state is (
                EvaluationRunState.FAILED
                if missing_usage and index == 1
                else EvaluationRunState.SUCCEEDED
            )
            results.append(api.evaluation.result(EvaluationResultRequest(run)))
        effects = spend._spend_records(store)
        assert len(effects) == 2 and len(backend.chats) == 4
        for effect in effects:
            missing = missing_usage and effect.entity.run_id == runs[1].run_id
            assert effect.state is (
                SpendEffectState.INDETERMINATE if missing else SpendEffectState.FOUND
            )
            if not missing:
                assert effect.usage.input_tokens == 2 * spend.PER_CALL[0]
                assert effect.usage.output_tokens == 2 * spend.PER_CALL[1]
            else:
                assert effect.usage is None
    snapshots = {
        path: path.read_bytes()
        for path in (tmp_path / "sink").rglob("*")
        if path.is_file()
    }
    before = (len(backend.chats), len(backend.opened))

    def forbidden(*args, **kwargs):
        pytest.fail("retained read/compare reopened backend or authority")

    backend.open = backend.requirement = forbidden
    with SQLiteDurableStoreV1(database, create=False) as store:
        _, cold, grants, _, _, _ = evaluation._host(
            tmp_path,
            backends={"openrouter": backend},
            records=store,
            streams=store,
            artifact_reader=sink,
        )
        for run, result in zip(runs, results):
            assert cold.evaluation.result(EvaluationResultRequest(run)) == result
        # Scoring can be comparable despite accounting uncertainty; comparing
        # retained reports must not rewrite the failed run or close its spend.
        report = cold.evaluation.compare(
            *(EvaluationResultRequest(run) for run in runs)
        )
        assert report["passed"] and report["reference_run"] == runs[0].to_dict()
        assert report["candidate_run"] == runs[1].to_dict()
        if missing_usage:
            assert cold.evaluation.show(runs[1]).state is EvaluationRunState.FAILED
            effect = next(
                item
                for item in spend._spend_records(store)
                if item.entity.run_id == runs[1].run_id
            )
            assert (
                effect.state is SpendEffectState.INDETERMINATE and effect.usage is None
            )
        assert not grants.authorized and not grants.bound
        assert len(spend._spend_records(store)) == 2
    assert before == (len(backend.chats), len(backend.opened))
    assert all(path.read_bytes() == raw for path, raw in snapshots.items())
