"""Real JudgeService accounting with CPU provider doubles; no live model evidence."""

from dataclasses import replace
import json

import pytest

import test_reference_evaluation_v1 as evaluation
import test_reference_spend_v1 as spend
from shared.judge import JudgeConfig, RubricDef
from shared.llm.metering import MeteredLLMClient
from shared.llm.usage import (
    LLMCompletionV1,
    LLMStructuredV1,
    UsageAccumulator,
    measured_usage,
)
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationModelRef,
    EvaluationRunState,
    EvaluationResultRequest,
)
from synaptic_tuner.api.v1.reference.evaluation import MeteredPaidJudgeV1
from synaptic_tuner.api.v1.reference.spend import SpendEffectState
from synaptic_tuner.api.v1.secrets import SecretRef

MODEL = EvaluationModelRef("judge-model", "revision")
KEY = SecretRef("env", "JUDGE_KEY")
TOKEN = "synthetic-judge-key-never-persist"


class Secrets:
    def __init__(self, value=TOKEN):
        self.value = value
        self.calls = []

    def resolve(self, reference):
        assert reference == KEY
        self.calls.append(reference)
        return self.value


class Client:
    provider_name = "openrouter"
    model_name = "judge-model"

    def __init__(self, mode="measured"):
        self.mode = mode
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append(("chat", kwargs))
        return (
            object()
            if self.mode == "wrong"
            else LLMCompletionV1("text", measured_usage(7, 11))
        )

    def structured_output(self, *args, **kwargs):
        self.calls.append(("structured", kwargs))
        if self.mode == "raise":
            raise RuntimeError("synthetic private judge failure")
        if self.mode == "wrong":
            return {"quality_score": 1}
        usage = None if self.mode == "missing" else measured_usage(7, 11)
        return LLMStructuredV1(
            {"quality_score": 0.95, "safety_score": 0.95, "overall_feedback": "ok"},
            usage,
        )


def judge(
    monkeypatch,
    *,
    client=None,
    backend="openrouter",
    account="acct-7",
    namespace="team-a",
    cap=(50, "USD"),
    secrets=None,
):
    secrets = secrets or Secrets()
    client = client or Client()
    constructed = []

    def create(key, model, **kwargs):
        assert key == TOKEN and model == MODEL.model_ref
        constructed.append(kwargs)
        return client

    module = (
        "shared.llm.providers.openrouter.OpenRouterClient"
        if backend == "openrouter"
        else "shared.llm.providers.openai_responses.OpenAIResponsesClient"
    )
    monkeypatch.setattr(module, create)
    rubrics = [
        RubricDef(
            key,
            key.title(),
            "declaration",
            "response",
            0.8,
            "Score {response}.",
            {"type": "object"},
        )
        for key in ("quality", "safety")
    ]
    config = JudgeConfig(max_tokens=123)
    port = MeteredPaidJudgeV1(
        backend,
        MODEL,
        api_key=KEY,
        secrets=secrets,
        account_ref=account,
        namespace_ref=namespace,
        rubrics=rubrics,
        judge_config=config,
        maximum_cost_minor_units=cap[0],
        currency=cap[1],
    )
    rubrics[0].judge_prompt = "changed after construction"
    config.max_tokens = 999
    return port, client, secrets, constructed


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    spend._no_network.__wrapped__(monkeypatch)


@pytest.mark.parametrize("response_paid", [False, True])
@pytest.mark.parametrize("mode", ["measured", "missing", "wrong", "raise"])
def test_real_judge_same_account_usage_and_swallowed_uncertainty(
    tmp_path, monkeypatch, response_paid, mode
):
    port, inner, secrets, constructed = judge(monkeypatch, client=Client(mode))
    backend = (
        spend.MeteredScriptedBackend() if response_paid else spend.LocalMeteredBackend()
    )
    name = "openrouter" if response_paid else "local"
    _, api, _, records, _, sink = evaluation._host(
        tmp_path, backends={name: backend}, judge=port
    )
    plan = api.evaluation.plan(evaluation._request("alpha.yaml", backend=name))
    assert api.evaluation.preflight(plan).ready
    assert not secrets.calls and not constructed
    run = api.evaluation.start(plan).run
    outcome = api.evaluation.show(run)
    result = api.evaluation.result(EvaluationResultRequest(run))
    (effect,) = spend._spend_records(records)
    assert len(inner.calls) == (2 if mode == "measured" else 1)
    assert len(backend.chats) == (1 if response_paid and mode != "measured" else 2)
    assert secrets.calls == [KEY] and constructed == [{"timeout_seconds": 60.0}]
    assert inner.calls[0][1]["max_tokens"] == 123
    assert "changed after construction" not in json.dumps(inner.calls)
    expected_models = (
        tuple(sorted((evaluation.MODEL.model_ref, MODEL.model_ref)))
        if response_paid
        else (MODEL.model_ref,)
    )
    assert effect.authorization.models == expected_models
    assert effect.authorization.maximum_cost_minor_units == (
        550 if response_paid else 50
    )
    if mode == "measured":
        assert effect.state is SpendEffectState.FOUND
        assert result.usage == effect.usage
        assert (effect.usage.input_tokens, effect.usage.output_tokens) == (
            14 + (6 if response_paid else 0),
            22 + (10 if response_paid else 0),
        )
        assert outcome.state is EvaluationRunState.SUCCEEDED
    else:
        assert effect.state is SpendEffectState.INDETERMINATE and effect.usage is None
        assert result.usage is None and outcome.state is EvaluationRunState.FAILED
    raw = b"".join(path.read_bytes() for path in sink.rglob("*") if path.is_file())
    assert TOKEN.encode() not in raw
    assert TOKEN not in json.dumps(spend._raw_spend_documents(records), default=str)


@pytest.mark.parametrize(
    "backend,account,namespace,cap",
    [
        ("openrouter", "wrong", "team-a", (50, "USD")),
        ("openrouter", "acct-7", "wrong", (50, "USD")),
        ("openai_responses", "acct-7", "team-a", (50, "USD")),
        ("openrouter", "acct-7", "team-a", (50, "EUR")),
    ],
)
def test_paid_scope_or_currency_mismatch_refuses_before_open(
    tmp_path, monkeypatch, backend, account, namespace, cap
):
    port, inner, secrets, constructed = judge(
        monkeypatch, backend=backend, account=account, namespace=namespace, cap=cap
    )
    backend = spend.MeteredScriptedBackend()
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"openrouter": backend}, judge=port
    )
    plan = api.evaluation.plan(evaluation._request("alpha.yaml", backend="openrouter"))
    assert api.evaluation.preflight(plan).diagnostic_codes == ("backend_unmetered",)
    with pytest.raises(Exception):
        api.evaluation.start(plan)
    assert (
        not secrets.calls and not constructed and not inner.calls and not backend.opened
    )
    assert not spend._spend_records(records)


@pytest.mark.parametrize("value", [None, "", "  ", 123])
def test_invalid_named_credential_construction_is_uncertain_not_absent(
    tmp_path, monkeypatch, value
):
    port, inner, secrets, constructed = judge(monkeypatch, secrets=Secrets(value))
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=port
    )
    run = api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml"))
    ).run
    assert api.evaluation.show(run).diagnostic_code == "judge_failed"
    (effect,) = spend._spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE
    assert not constructed and not inner.calls


@pytest.mark.parametrize("operation", ["chat", "structured_output"])
def test_wrong_completion_shape_marks_actual_request_unknown(operation):
    meter = UsageAccumulator()
    inner = Client("wrong")
    wrapped = MeteredLLMClient(inner, meter)
    with pytest.raises(TypeError):
        if operation == "chat":
            wrapped.chat([])
        else:
            wrapped.structured_output([], {})
    assert len(inner.calls) == 1 and meter.calls == 1 and meter.indeterminate


def test_cancel_before_paid_judge_request_closes_absent(tmp_path, monkeypatch):
    port, inner, _, _ = judge(monkeypatch)
    backend = spend.LocalMeteredBackend()
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": backend}, judge=port
    )
    plan = api.evaluation.plan(evaluation._request("alpha.yaml"))
    from synaptic_tuner.api.v1.reference.evaluation import _run_id
    from synaptic_tuner.api.v1.evaluation_facade import EvaluationRunRef

    run = EvaluationRunRef(_run_id(plan), plan.request.project_ref)
    original = port.open_with_client

    def cancel_then_construct(client):
        api.evaluation.cancel(run, "cancel before first paid call")
        return original(client)

    port.open_with_client = cancel_then_construct
    api.evaluation.start(plan)
    assert api.evaluation.show(run).state is EvaluationRunState.CANCELLED
    (effect,) = spend._spend_records(records)
    assert effect.state is SpendEffectState.DEFINITELY_ABSENT
    assert not inner.calls and not backend.chats


@pytest.mark.parametrize("provider", ["openrouter", "openai_responses"])
def test_concrete_judge_is_usable_without_ambient_factory(monkeypatch, provider):
    port, inner, secrets, constructed = judge(monkeypatch, backend=provider)

    def forbidden(*args, **kwargs):
        pytest.fail("ambient configuration/factory used")

    monkeypatch.setattr("shared.llm.config.LLMConfig.from_env", forbidden)
    assert not secrets.calls
    validator = port.open()
    assert validator.judge_service.llm_client is inner
    assert secrets.calls == [KEY] and len(constructed) == 1


def test_paid_response_unpaid_judge_preserves_response_only_accounting(
    tmp_path, monkeypatch
):
    paid, inner, _, _ = judge(monkeypatch)

    class UnpaidJudge:
        def requirement(self):
            from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement

            return AuthorizationRequirement("evaluation.judge", False)

        def open(self):
            return paid.open_with_client(inner)

    backend = spend.MeteredScriptedBackend()
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"openrouter": backend}, judge=UnpaidJudge()
    )
    run = api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml", backend="openrouter"))
    ).run
    assert api.evaluation.show(run).state is EvaluationRunState.SUCCEEDED
    (effect,) = spend._spend_records(records)
    assert effect.authorization.models == (evaluation.MODEL.model_ref,)
    assert (effect.usage.input_tokens, effect.usage.output_tokens) == (6, 10)
    assert len(inner.calls) == 2


def test_unknown_paid_response_constructor_with_paid_judge_is_not_absent(
    tmp_path, monkeypatch
):
    port, inner, _, constructed = judge(monkeypatch)
    backend = spend.MeteredScriptedBackend(
        open_error=RuntimeError("unknown construction")
    )
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"openrouter": backend}, judge=port
    )
    run = api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml", backend="openrouter"))
    ).run
    assert api.evaluation.show(run).diagnostic_code == "backend_unavailable"
    (effect,) = spend._spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE
    assert not inner.calls and not constructed


def test_missing_cap_stays_absent_for_combined_authorization(tmp_path, monkeypatch):
    port, _, _, _ = judge(monkeypatch, cap=(None, None))
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"openrouter": spend.MeteredScriptedBackend()}, judge=port
    )
    api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml", backend="openrouter"))
    )
    (effect,) = spend._spend_records(records)
    assert (
        effect.authorization.maximum_cost_minor_units is None
        and effect.authorization.currency is None
    )


@pytest.mark.parametrize("returned", [None, object()])
def test_wrong_paid_validator_constructor_is_uncertain(tmp_path, monkeypatch, returned):
    port, inner, _, _ = judge(monkeypatch)
    port.open_with_client = lambda client: returned
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=port
    )
    run = api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml"))
    ).run
    assert api.evaluation.show(run).diagnostic_code == "judge_failed"
    (effect,) = spend._spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE and not inner.calls


def test_paid_judge_retained_sqlite_results_do_not_resolve_credentials(
    tmp_path, monkeypatch
):
    from synaptic_tuner.api.v1.reference.sqlite_store import SQLiteDurableStoreV1
    from synaptic_tuner.api.v1.reference.evaluation import (
        DirectoryEvaluationArtifactSinkV1,
    )

    port, _, secrets, _ = judge(monkeypatch)
    backend = spend.LocalMeteredBackend()
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "workflow.sqlite3"
    sink = DirectoryEvaluationArtifactSinkV1(tmp_path / "sink")
    runs, results = [], []
    with SQLiteDurableStoreV1(database) as store:
        _, api, _, _, _, _ = evaluation._host(
            tmp_path,
            backends={"local": backend},
            judge=port,
            records=store,
            streams=store,
            artifact_reader=sink,
        )
        for index in range(2):
            request = replace(
                evaluation._request("alpha.yaml", request_id=f"request-{index}"),
                model=EvaluationModelRef(f"local-{index}", "revision"),
            )
            run = api.evaluation.start(api.evaluation.plan(request)).run
            runs.append(run)
            results.append(api.evaluation.result(EvaluationResultRequest(run)))
        assert len(spend._spend_records(store)) == 2
    before = len(secrets.calls)

    def forbidden(*args, **kwargs):
        pytest.fail("retained result/compare opened paid collaborators")

    secrets.resolve = port.create_client = port.requirement = backend.open = (
        backend.requirement
    ) = forbidden
    with SQLiteDurableStoreV1(database, create=False) as store:
        _, api, _, _, _, _ = evaluation._host(
            tmp_path,
            backends={"local": backend},
            judge=port,
            records=store,
            streams=store,
            artifact_reader=sink,
        )
        assert [
            api.evaluation.result(EvaluationResultRequest(run)) for run in runs
        ] == results
        report = api.evaluation.compare(*(EvaluationResultRequest(run) for run in runs))
        assert (
            report["reference_run"] == runs[0].to_dict()
            and report["candidate_run"] == runs[1].to_dict()
        )
        assert report["candidates"][0]["comparable"] is False
        assert len(spend._spend_records(store)) == 2
    assert len(secrets.calls) == before


@pytest.mark.parametrize(
    "model",
    ["", "  ", "x" * 4097, 123],
    ids=["empty", "whitespace", "oversize", "wrong-type"],
)
def test_invalid_judge_model_refused_before_claim(tmp_path, monkeypatch, model):
    port, inner, secrets, constructed = judge(monkeypatch)
    port.model_ref = lambda: model
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=port
    )
    plan = api.evaluation.plan(evaluation._request("alpha.yaml"))
    assert api.evaluation.preflight(plan).diagnostic_codes == ("backend_unmetered",)
    with pytest.raises(Exception):
        api.evaluation.start(plan)
    assert (
        not spend._spend_records(records)
        and not constructed
        and not secrets.calls
        and not inner.calls
    )


def test_paid_validator_ignoring_metered_client_refuses_before_judging(
    tmp_path, monkeypatch
):
    port, inner, _, _ = judge(monkeypatch)
    original = port.open_with_client
    port.open_with_client = lambda supplied: original(inner)
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=port
    )
    run = api.evaluation.start(
        api.evaluation.plan(evaluation._request("alpha.yaml"))
    ).run
    assert api.evaluation.show(run).diagnostic_code == "judge_failed"
    (effect,) = spend._spend_records(records)
    assert effect.state is SpendEffectState.INDETERMINATE and not inner.calls


@pytest.mark.parametrize(
    "model,admitted",
    [("x" * 4096, True), ("é" * 2048, True), ("x" * 4097, False), ("é" * 2049, False)],
    ids=["ascii-boundary", "utf8-boundary", "ascii-oversize", "utf8-oversize"],
)
def test_judge_model_bound_is_exact_utf8_not_characters(
    tmp_path, monkeypatch, model, admitted
):
    original, inner, secrets, constructed = judge(monkeypatch)
    kwargs = dict(
        api_key=KEY,
        secrets=secrets,
        account_ref="acct-7",
        namespace_ref="team-a",
        rubrics=list(original._rubrics),
        judge_config=original._config,
    )
    if admitted:
        port = MeteredPaidJudgeV1(
            "openrouter", EvaluationModelRef(model, "revision"), **kwargs
        )
        assert port.model_ref() == model
    else:
        with pytest.raises(ValueError):
            MeteredPaidJudgeV1(
                "openrouter", EvaluationModelRef(model, "revision"), **kwargs
            )
    original.model_ref = lambda: model
    _, api, _, records, _, _ = evaluation._host(
        tmp_path, backends={"local": spend.LocalMeteredBackend()}, judge=original
    )
    assert (
        api.evaluation.preflight(
            api.evaluation.plan(evaluation._request("alpha.yaml"))
        ).ready
        is admitted
    )
    assert (
        not secrets.calls
        and not constructed
        and not inner.calls
        and not spend._spend_records(records)
    )
