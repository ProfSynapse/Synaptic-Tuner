"""One ``spend`` effect per paid run (api-facade slice 11, architecture §4.4).

Drives the reference Evaluation and Data families over paid and local scripted
backends that never touch a socket and asserts the spend ledger contract: a
paid, metered run claims exactly one ``spend`` effect however many LLM calls
it makes and closes it ``found`` with the usage summed over every call; a paid
run with any unmeasured call finishes ``failed`` / ``spend_indeterminate`` and
claims no found spend; a local run never claims a spend effect and carries
``usage`` only when every call was measured; a paid backend without a spend
scope stays refused ``backend_unmetered``; the effect record is closed and
carries no provider text or secret. ``MeteredPaidBackendV1`` is admitted by
preflight and start with its API key resolved as a ``SecretRef`` only.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket

import pytest

from synaptic_tuner.api.v1.data_facade import DataOperationCode, DataOperationError, DataRunState
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationOperationCode, EvaluationOperationError, EvaluationResultRequest, EvaluationRunState,
)
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.reference.data import ReferenceDataPortsV1, build_data_operations, data_run_ref
from synaptic_tuner.api.v1.reference.evaluation import MeteredPaidBackendV1, UnmeteredPaidBackendV1
from synaptic_tuner.api.v1.reference.spend import (
    SPEND_EFFECT_KEY_PREFIX, SPEND_EFFECT_SCHEMA_VERSION, SpendEffectRecordV1, SpendEffectState, SpendScopeV1,
)
from synaptic_tuner.api.v1.reference.stores import InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1
from Evaluator.protocols import BackendResponse
from shared.llm.usage import LLMCompletionV1, LLMStructuredV1, measured_usage
from tuner.execution.foundation_v2.identities import EffectKind

from test_reference_data_v1 import (
    CHAT_CALLS_PER_ROW, CONFIG_DIR, RUBRICS_DIR, Clock as DataClock, FakeLocalClient, Factory,
    _assert_artifacts_match_disk,
)
from test_reference_evaluation_v1 import (
    MODEL, PROJECT, RESPONSES, SECRET_VALUE, ScriptedBackend, ScriptedClient, _host, _request, _stores,
)

EFFECTS = StoragePartition.EFFECTS.value
PER_CALL = (3, 5)
PAID_KEY = SecretRef("env", "OPENROUTER_API_KEY")
PAID_KEY_VALUE = "sk-or-LIVE-KEY-VALUE-MUST-NEVER-PERSIST-0000"


def _refuse_network(*args, **kwargs):
    raise AssertionError("network access is forbidden in the spend contract test")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_network)


# --- evaluation fakes ------------------------------------------------------------------


class MeteredScriptedClient(ScriptedClient):
    """A ``BackendClient`` whose responses carry measured usage except for ``unmeasured`` questions."""

    def __init__(self, backend, responses, on_chat, unmeasured):
        super().__init__(backend, responses, on_chat)
        self._unmeasured = frozenset(unmeasured)

    def chat(self, messages):
        response = super().chat(messages)
        if messages[-1]["content"] in self._unmeasured:
            return response
        return BackendResponse(
            message=response.message, raw=response.raw, latency_s=response.latency_s,
            usage=measured_usage(*PER_CALL),
        )


class MeteredScriptedBackend(ScriptedBackend):
    """A paid ``EvaluationBackendPort`` with a spend scope; every answer is scripted."""

    def __init__(self, responses=RESPONSES, *, unmeasured=(), scope=None, cap=(500, "USD"), open_error=None):
        super().__init__(responses)
        self._unmeasured = unmeasured
        self._scope = scope if scope is not None else SpendScopeV1("openrouter", "acct-7", "team-a")
        self._cap = cap
        self._open_error = open_error

    def requirement(self):
        return AuthorizationRequirement("evaluation.start", True, *self._cap)

    def spend_scope(self):
        return self._scope

    def open(self, model):
        self.opened.append(model)
        if self._open_error is not None:
            raise self._open_error
        return MeteredScriptedClient(self, self._responses, None, self._unmeasured)


class LocalMeteredBackend(MeteredScriptedBackend):
    """A local backend whose client happens to report usage: no spend effect, usage recorded."""

    def requirement(self):
        return AuthorizationRequirement("evaluation.start", False)

    def spend_scope(self):
        raise AssertionError("a local backend is never asked for a spend scope")


class KeySecrets:
    def __init__(self):
        self.resolved = []

    def resolve(self, reference):
        assert reference == PAID_KEY
        self.resolved.append(reference)
        return PAID_KEY_VALUE


def _spend_records(records) -> list[SpendEffectRecordV1]:
    page = records.list_page(partition=EFFECTS, prefix=SPEND_EFFECT_KEY_PREFIX, after_key=None, limit=100)
    assert page.truncated is False
    return [SpendEffectRecordV1.decode(item.canonical) for item in page.records]


def _raw_spend_documents(records) -> list[bytes]:
    page = records.list_page(partition=EFFECTS, prefix="", after_key=None, limit=100)
    return [item.canonical for item in page.records]


def _evaluate(tmp_path, backend, *, name="openrouter", judge=None):
    _, api, grants, records, _, _ = _host(tmp_path, backends={name: backend}, judge=judge)
    plan = api.evaluation.plan(_request("alpha.yaml", backend=name))
    ready = api.evaluation.preflight(plan)
    return api, grants, records, plan, ready


# --- evaluation --------------------------------------------------------------------------


def test_paid_evaluation_run_claims_one_spend_effect_and_publishes_summed_usage(tmp_path) -> None:
    backend = MeteredScriptedBackend()
    api, grants, records, plan, ready = _evaluate(tmp_path, backend)
    assert ready.ready is True and ready.diagnostic_codes == ()
    assert [(item.operation, item.paid_effect, item.maximum_cost_minor_units, item.currency) for item in ready.authorization] == [
        ("evaluation.start", True, 500, "USD"),
    ]

    started = api.evaluation.start(plan)
    run = started.run
    assert len(backend.chats) == 2  # two cases, two provider calls

    outcome = api.evaluation.show(run)
    assert outcome.state is EvaluationRunState.SUCCEEDED and outcome.diagnostic_code is None
    result = api.evaluation.result(EvaluationResultRequest(run))
    assert type(result.usage) is UsageRecordV1 and result.usage.availability is UsageAvailability.MEASURED
    assert (result.usage.input_tokens, result.usage.output_tokens) == (2 * PER_CALL[0], 2 * PER_CALL[1])
    assert (result.usage.cost_minor_units, result.usage.currency) == (None, None)  # no adapter prices a call
    document = result.to_dict()
    assert document["usage"]["availability"] == "measured"

    # Exactly one spend effect, regardless of the number of calls; closed found with the same usage.
    spend_effects = _spend_records(records)
    assert len(spend_effects) == 1
    (effect,) = spend_effects
    assert effect.kind is EffectKind.SPEND and effect.state is SpendEffectState.FOUND
    assert effect.entity.family == "evaluation" and effect.entity.run_id == run.run_id
    assert effect.scope == SpendScopeV1("openrouter", "acct-7", "team-a")
    assert effect.authorization.models == (MODEL.model_ref,)
    assert (effect.authorization.maximum_cost_minor_units, effect.authorization.currency) == (500, "USD")
    assert effect.provider_job_ref == effect.effect_id and effect.closed_at is not None
    assert effect.usage == result.usage
    assert result.usage.spend == SpendRef("openrouter", "acct-7", effect.effect_id)
    assert set(_stores(records)) == {StoragePartition.EVALUATION.value, EFFECTS}
    assert grants.authorized == [] and grants.bound == []

    # Idempotent restart claims nothing more.
    assert api.evaluation.start(plan) == started
    assert len(_spend_records(records)) == 1 and len(backend.chats) == 2


def test_paid_evaluation_run_with_an_unmeasured_call_fails_spend_indeterminate_and_claims_no_spend(tmp_path) -> None:
    backend = MeteredScriptedBackend(unmeasured={"Summarise the document"})
    api, _, records, plan, ready = _evaluate(tmp_path, backend)
    assert ready.ready is True

    run = api.evaluation.start(plan).run
    assert len(backend.chats) == 2  # the run is not cut short; its spend is what is indeterminate

    outcome = api.evaluation.show(run)
    assert outcome.state is EvaluationRunState.FAILED
    assert outcome.diagnostic_code == EvaluationOperationCode.SPEND_INDETERMINATE.value == "spend_indeterminate"
    result = api.evaluation.result(EvaluationResultRequest(run))
    assert result.usage is None and "usage" not in result.to_dict()

    (effect,) = _spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE
    assert effect.usage is None and effect.provider_job_ref is None and effect.closed_at is not None
    assert not any(item.state is SpendEffectState.FOUND for item in _spend_records(records))


def test_paid_evaluation_run_that_makes_no_call_closes_its_spend_definitely_absent(tmp_path) -> None:
    backend = MeteredScriptedBackend(open_error=RuntimeError("provider down"))
    api, _, records, plan, _ = _evaluate(tmp_path, backend)
    run = api.evaluation.start(plan).run
    outcome = api.evaluation.show(run)
    assert outcome.state is EvaluationRunState.FAILED
    assert outcome.diagnostic_code == EvaluationOperationCode.BACKEND_UNAVAILABLE.value
    assert api.evaluation.result(EvaluationResultRequest(run)).usage is None
    (effect,) = _spend_records(records)
    assert effect.state is SpendEffectState.DEFINITELY_ABSENT and effect.usage is None


def test_local_evaluation_run_records_measured_usage_but_claims_no_spend(tmp_path) -> None:
    backend = LocalMeteredBackend()
    api, grants, records, plan, ready = _evaluate(tmp_path, backend, name="local")
    assert ready.ready is True and ready.authorization == ()
    run = api.evaluation.start(plan).run
    result = api.evaluation.result(EvaluationResultRequest(run))
    assert result.state is EvaluationRunState.SUCCEEDED
    assert (result.usage.input_tokens, result.usage.output_tokens) == (2 * PER_CALL[0], 2 * PER_CALL[1])
    assert result.usage.spend is None
    assert set(_stores(records)) == {StoragePartition.EVALUATION.value}
    assert _spend_records(records) == [] and grants.authorized == []


def test_local_evaluation_run_without_usage_publishes_no_usage_field_and_no_spend(tmp_path) -> None:
    backend = ScriptedBackend()
    api, _, records, plan, _ = _evaluate(tmp_path, backend, name="local")
    run = api.evaluation.start(plan).run
    result = api.evaluation.result(EvaluationResultRequest(run))
    assert result.state is EvaluationRunState.SUCCEEDED and result.usage is None
    assert "usage" not in result.to_dict()
    assert set(_stores(records)) == {StoragePartition.EVALUATION.value}


def test_paid_backend_without_a_spend_scope_stays_refused_backend_unmetered(tmp_path) -> None:
    backend = UnmeteredPaidBackendV1("openrouter")
    api, _, records, plan, ready = _evaluate(tmp_path, backend)
    assert ready.ready is False and ready.diagnostic_codes == ("backend_unmetered",)
    assert [(item.operation, item.paid_effect) for item in ready.authorization] == [("evaluation.start", True)]
    with pytest.raises(EvaluationOperationError) as refused:
        api.evaluation.start(plan)
    assert refused.value.code is EvaluationOperationCode.BACKEND_UNMETERED
    assert _stores(records) == {}


def test_paid_judge_leaves_a_metered_backend_refused_backend_unmetered(tmp_path) -> None:
    class PaidJudge:
        def requirement(self):
            return AuthorizationRequirement("evaluation.judge", True, 10, "USD")

        def open(self):
            raise AssertionError("a paid judge must never be opened")

    backend = MeteredScriptedBackend()
    api, _, records, plan, ready = _evaluate(tmp_path, backend, judge=PaidJudge())
    assert ready.ready is False and ready.diagnostic_codes == ("backend_unmetered",)
    with pytest.raises(EvaluationOperationError) as refused:
        api.evaluation.start(plan)
    assert refused.value.code is EvaluationOperationCode.BACKEND_UNMETERED
    assert backend.opened == [] and _stores(records) == {}


def test_metered_paid_backend_is_admitted_and_resolves_its_key_only_at_open(tmp_path) -> None:
    secrets = KeySecrets()
    backend = MeteredPaidBackendV1(
        "openrouter", api_key=PAID_KEY, secrets=secrets, account_ref="acct-7", namespace_ref="team-a",
        maximum_cost_minor_units=500, currency="USD",
    )
    assert backend.spend_scope() == SpendScopeV1("openrouter", "acct-7", "team-a")
    api, _, records, plan, ready = _evaluate(tmp_path, backend)
    assert ready.ready is True and ready.diagnostic_codes == ()
    assert [(item.paid_effect, item.maximum_cost_minor_units) for item in ready.authorization] == [(True, 500)]
    assert secrets.resolved == []

    # start is admitted; the socket guard turns the first provider call into an unmeasured call.
    run = api.evaluation.start(plan).run
    assert secrets.resolved == [PAID_KEY]
    outcome = api.evaluation.show(run)
    assert outcome.state is EvaluationRunState.FAILED
    assert outcome.diagnostic_code == "spend_indeterminate"
    (effect,) = _spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE and effect.scope.account_ref == "acct-7"
    for document in _raw_spend_documents(records):
        assert PAID_KEY_VALUE.encode() not in document

    with pytest.raises(ValueError):
        MeteredPaidBackendV1("lmstudio", api_key=PAID_KEY, secrets=secrets, account_ref="a", namespace_ref="n")
    with pytest.raises(TypeError):
        MeteredPaidBackendV1("openrouter", api_key="plain-text", secrets=secrets, account_ref="a", namespace_ref="n")


def test_spend_effect_record_is_closed_and_carries_no_provider_text(tmp_path) -> None:
    backend = MeteredScriptedBackend()
    api, _, records, plan, _ = _evaluate(tmp_path, backend)
    api.evaluation.start(plan)
    (raw,) = _raw_spend_documents(records)
    document = json.loads(raw.decode("utf-8"))
    assert set(document) == {
        "schema_version", "effect_id", "kind", "entity", "scope", "authorization", "command_digest",
        "state", "provider_job_ref", "usage", "claimed_at", "closed_at",
    }
    assert document["schema_version"] == SPEND_EFFECT_SCHEMA_VERSION and document["kind"] == "spend"
    assert set(document["usage"]) == {
        "schema_version", "availability", "input_tokens", "output_tokens", "cost_minor_units", "currency", "spend",
    }
    text = raw.decode("utf-8")
    assert SECRET_VALUE not in text
    assert f'"{PROJECT}"' not in text  # the project appears only as a digest
    for answer in RESPONSES.values():
        assert answer not in text
    for question in RESPONSES:
        assert question not in text


# --- data --------------------------------------------------------------------------------


class MeteredFakeClient(FakeLocalClient):
    """A ``shared.llm`` style client whose completions carry measured usage except when scripted not to."""

    provider_name = "openrouter"

    def __init__(self, responses, *, unmeasured_calls=(), **kwargs):
        super().__init__(responses, **kwargs)
        self._unmeasured_calls = frozenset(unmeasured_calls)

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        index = self.chat_calls
        completion = super().chat(messages, temperature, max_tokens)
        if index in self._unmeasured_calls:
            return completion
        return LLMCompletionV1(completion.text, measured_usage(*PER_CALL))

    def structured_output(self, messages, schema, temperature=0.3, max_tokens=2048):
        return LLMStructuredV1(super().structured_output(messages, schema, temperature, max_tokens).value, measured_usage(*PER_CALL))


class MeteredFactory(Factory):
    def __init__(self, client, *, scopes):
        super().__init__(client)
        self.scopes = dict(scopes)
        self.scope_requests = []

    def spend_scope(self, *, backend):
        self.scope_requests.append(backend)
        return self.scopes.get(backend)


class DataHarness:
    def __init__(self, tmp_path: Path, factory) -> None:
        from test_reference_data_v1 import SCENARIO_YAML

        self.root = tmp_path
        scenarios_dir = tmp_path / "scenarios"
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        (scenarios_dir / "behavioral.yaml").write_text(SCENARIO_YAML, encoding="utf-8")
        self.records = InMemoryDurableRecordStoreV1()
        self.streams = InMemoryDurableStreamStoreV1()
        self.clock = DataClock()
        self.factory = factory
        self.operations = build_data_operations(
            records=self.records, streams=self.streams, clock=self.clock,
            ports=ReferenceDataPortsV1(CONFIG_DIR, scenarios_dir, RUBRICS_DIR, factory),
        )
        self.api = APIHost(HostPorts(
            training=object(), runs=object(), artifacts=None, evaluation=None,
            chat=None, data=self.operations, pipelines=None, clock=self.clock,
        )).data

    def request(self, name="run", *, backend="openrouter"):
        from synaptic_tuner.api.v1.data_facade import DataMode, DataRequest, DataScenarioTarget

        return DataRequest(
            request_id=f"request-{name}", project_ref="acme", mode=DataMode.GENERATE, backend=backend,
            model="fake-paid", output_ref=str(self.root / f"{name}.jsonl"),
            scenarios=(DataScenarioTarget("greeting", 2), DataScenarioTarget("farewell", 1)),
        )

    def head_result(self, run) -> dict:
        from synaptic_tuner.api.v1.reference.data import _run_key

        stored = self.records.read(partition=StoragePartition.DATA.value, key=_run_key(run))
        return json.loads(stored.canonical.decode("utf-8"))["result"]


DATA_SCOPE = SpendScopeV1("openrouter", "acct-9", "team-b")


def test_paid_data_run_claims_one_spend_effect_and_publishes_summed_usage(tmp_path) -> None:
    client = MeteredFakeClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)])
    harness = DataHarness(tmp_path, MeteredFactory(client, scopes={"openrouter": DATA_SCOPE}))
    api = harness.api
    plan = api.plan(harness.request())
    preflight = api.preflight(plan)
    assert preflight.ready and preflight.diagnostic_codes == ()
    assert [(item.operation, item.paid_effect) for item in preflight.authorization] == [("data.start", True)]

    start = api.start(plan)
    assert start.accepted and start.run == data_run_ref(plan)
    assert client.chat_calls == 3 * CHAT_CALLS_PER_ROW
    outcome = api.show(start.run)
    assert outcome.state is DataRunState.SUCCEEDED and outcome.rows_written == 3
    _assert_artifacts_match_disk(outcome, tmp_path / "run.jsonl")

    (effect,) = _spend_records(harness.records)
    assert effect.state is SpendEffectState.FOUND and effect.entity.family == "data"
    assert effect.entity.run_id == start.run.run_id and effect.scope == DATA_SCOPE
    assert effect.authorization.models == ("fake-paid",)
    calls = 3 * CHAT_CALLS_PER_ROW
    assert (effect.usage.input_tokens, effect.usage.output_tokens) == (calls * PER_CALL[0], calls * PER_CALL[1])
    result = harness.head_result(start.run)
    assert result["usage"] == effect.usage.to_dict()
    assert result["usage"]["spend"] == {"provider": "openrouter", "account_ref": "acct-9", "spend_id": effect.effect_id}
    assert api.start(plan) == start and len(_spend_records(harness.records)) == 1


def test_paid_data_run_with_an_unmeasured_call_fails_spend_indeterminate(tmp_path) -> None:
    client = MeteredFakeClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)], unmeasured_calls={4})
    harness = DataHarness(tmp_path, MeteredFactory(client, scopes={"openrouter": DATA_SCOPE}))
    api = harness.api
    plan = api.plan(harness.request())
    run = api.start(plan).run
    outcome = api.show(run)
    assert outcome.state is DataRunState.FAILED and outcome.diagnostic_code == "spend_indeterminate"
    assert outcome.rows_written == 3  # rows were written; only the spend is indeterminate
    assert "usage" not in harness.head_result(run)
    (effect,) = _spend_records(harness.records)
    assert effect.state is SpendEffectState.INDETERMINATE and effect.usage is None


def test_local_data_run_records_measured_usage_but_claims_no_spend(tmp_path) -> None:
    client = MeteredFakeClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)])
    factory = MeteredFactory(client, scopes={"openrouter": DATA_SCOPE})
    harness = DataHarness(tmp_path, factory)
    api = harness.api
    plan = api.plan(harness.request(backend="lmstudio"))
    assert [(item.paid_effect) for item in api.preflight(plan).authorization] == [False]
    run = api.start(plan).run
    assert api.show(run).state is DataRunState.SUCCEEDED
    result = harness.head_result(run)
    calls = 3 * CHAT_CALLS_PER_ROW
    assert (result["usage"]["input_tokens"], result["usage"]["output_tokens"]) == (calls * PER_CALL[0], calls * PER_CALL[1])
    assert result["usage"]["spend"] is None
    assert _spend_records(harness.records) == [] and factory.scope_requests == []


def test_local_data_run_without_usage_publishes_no_usage_and_no_spend(tmp_path) -> None:
    client = FakeLocalClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)])
    harness = DataHarness(tmp_path, Factory(client))
    api = harness.api
    plan = api.plan(harness.request(backend="lmstudio"))
    run = api.start(plan).run
    assert api.show(run).state is DataRunState.SUCCEEDED
    assert "usage" not in harness.head_result(run)
    assert not harness.records.list_page(partition=EFFECTS, prefix="", after_key=None, limit=10).records


@pytest.mark.parametrize("backend", ["openrouter", "openai", "openai_responses", "anthropic"])
def test_paid_data_backend_without_a_factory_spend_scope_stays_refused_backend_unmetered(tmp_path, backend) -> None:
    client = MeteredFakeClient(["never used"])
    for factory in (Factory(client), MeteredFactory(client, scopes={})):
        harness = DataHarness(tmp_path / factory.__class__.__name__, factory)
        api = harness.api
        plan = api.plan(harness.request(backend=backend))
        preflight = api.preflight(plan)
        assert not preflight.ready and preflight.diagnostic_codes == ("backend_unmetered",)
        assert [(item.operation, item.paid_effect) for item in preflight.authorization] == [("data.start", True)]
        with pytest.raises(DataOperationError) as refused:
            api.start(plan)
        assert refused.value.code is DataOperationCode.BACKEND_UNMETERED
        assert factory.requests == [] and client.chat_calls == 0
        assert not harness.records.list_page(partition=EFFECTS, prefix="", after_key=None, limit=10).records
