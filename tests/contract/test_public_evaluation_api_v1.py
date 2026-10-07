"""Contract tests for the EvaluationAPI facade (api-facade slice 5, contract only)."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import jsonschema
import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import evaluation_facade
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationAPI,
    EvaluationListRequest,
    EvaluationModelRef,
    EvaluationOperationCode,
    EvaluationOperationError,
    EvaluationOperations,
    EvaluationOutcome,
    EvaluationPage,
    EvaluationPlan,
    EvaluationPreflight,
    EvaluationRequest,
    EvaluationResult,
    EvaluationResultRequest,
    EvaluationRunRef,
    EvaluationRunState,
    EvaluationStart,
    EvaluationVerdictCounts,
    JudgeVerdict,
    ScoreV1,
)
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    EVALUATION_VERDICTS,
    EvaluationCaseScoredPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunsAPI
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1


ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = json.loads((ROOT / "schemas/synaptic-evaluation-plan-v1.schema.json").read_text(encoding="utf-8"))
RESULT_SCHEMA = json.loads((ROOT / "schemas/synaptic-evaluation-result-v1.schema.json").read_text(encoding="utf-8"))

VERBS = {"plan", "preflight", "start", "show", "cancel", "list", "result", "observations"}
RUN = EvaluationRunRef("ev-01J", "acme")
MODEL = EvaluationModelRef("acme/qwen-sft", "a1b2" * 10)


def _request(**changes: object) -> EvaluationRequest:
    values: dict[str, object] = {
        "request_id": "request-1", "project_ref": "acme", "model": MODEL,
        "backend": "openrouter", "scenario_refs": ("tool-choice", "grounding"),
        "preset": "quick", "tags": ("smoke",), "case_limit": 120,
    }
    values.update(changes)
    return EvaluationRequest(**values)  # type: ignore[arg-type]


def _plan(request: EvaluationRequest | None = None) -> EvaluationPlan:
    return EvaluationPlan("synaptic-evaluation-plan/v1", request or _request(), 120, "b" * 64)


def _preflight(plan: EvaluationPlan, **changes: object) -> EvaluationPreflight:
    values: dict[str, object] = {
        "plan_fingerprint": plan.plan_fingerprint, "ready": True,
        "checked_at": "2026-09-17T12:00:00Z", "expires_at": "2026-09-17T12:05:00Z",
        "authorization": (AuthorizationRequirement("evaluation.start", True, 2000, "USD"),),
    }
    values.update(changes)
    return EvaluationPreflight(**values)  # type: ignore[arg-type]


def _outcome(run: EvaluationRunRef = RUN) -> EvaluationOutcome:
    return EvaluationOutcome(run, EvaluationRunState.RUNNING, 120, 41)


def _measured_usage() -> UsageRecordV1:
    return UsageRecordV1(
        "synaptic-usage/v1", UsageAvailability.MEASURED, 412334, 88120, 1874, "USD",
        SpendRef("openrouter", "acct-main", "spend-01J"),
    )


def _result(run: EvaluationRunRef = RUN, **changes: object) -> EvaluationResult:
    values: dict[str, object] = {
        "schema_version": "synaptic-evaluation-result/v1", "run": run,
        "state": EvaluationRunState.SUCCEEDED, "backend": "openrouter", "model": MODEL,
        "cases_total": 120, "cases_scored": 120,
        "scores": (ScoreV1("composite", 0.812), ScoreV1("quality_gated", 0.744)),
        "verdicts": EvaluationVerdictCounts(97, 21, 2),
        "artifacts": (VerifiedArtifact("evaluation_results", "c" * 64, 220114),),
        "diagnostic_code": None, "usage": _measured_usage(),
    }
    values.update(changes)
    return EvaluationResult(**values)  # type: ignore[arg-type]


def _stream() -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.EVALUATION, "acme", "ev-01J")


def _observation(sequence: int = 41) -> ObservationRecordV1:
    return ObservationRecordV1(
        "synaptic-observation/v1", _stream(), sequence, "2026-09-17T12:00:04Z",
        ObservationKind.EVALUATION_CASE_SCORED,
        EvaluationCaseScoredPayloadV1("tool-choice#3", "passed", 0.82),
    )


# --- exports, verbs, closed vocabularies ---------------------------------------


def test_root_evaluation_exports_are_the_canonical_facade_identities() -> None:
    for name in evaluation_facade.__all__:
        assert getattr(v1, name) is getattr(evaluation_facade, name), name
        assert name in v1.__all__, name
    assert set(v1._LAZY_MODULE_ATTRIBUTES["evaluation_facade"]) == set(evaluation_facade.__all__)
    with pytest.raises(ModuleNotFoundError):
        __import__("synaptic_tuner.api.v1.evaluation")


def test_evaluation_api_and_operations_have_only_the_accepted_verbs() -> None:
    def verbs(owner: type) -> set[str]:
        return {
            name for name, member in owner.__dict__.items()
            if not name.startswith("_") and inspect.isfunction(member)
        }
    assert verbs(EvaluationAPI) == VERBS
    assert verbs(EvaluationOperations) == VERBS
    for name in VERBS:
        assert tuple(inspect.signature(getattr(EvaluationAPI, name)).parameters) == tuple(
            inspect.signature(getattr(EvaluationOperations, name)).parameters
        ), name
    assert "reconcile" not in verbs(EvaluationAPI) and "verify" not in verbs(EvaluationAPI)


def test_closed_vocabularies_match_the_architecture() -> None:
    assert tuple(item.value for item in EvaluationRunState) == (
        "planned", "running", "succeeded", "partially_succeeded", "failed",
        "cancel_requested", "cancelled",
    )
    assert "reconcile_required" not in {item.value for item in EvaluationRunState}
    assert {item.value for item in JudgeVerdict} == EVALUATION_VERDICTS
    assert tuple(item.value for item in EvaluationOperationCode) == (
        "run_missing", "cursor_invalid", "scenario_invalid", "backend_unavailable",
        "backend_unmetered", "judge_failed", "spend_indeterminate", "cancel_ineligible",
        "result_unavailable", "state_conflict", "integrity_error",
    )
    assert set(RESULT_SCHEMA["properties"]["state"]["enum"]) == {item.value for item in EvaluationRunState}
    assert set(RESULT_SCHEMA["properties"]["verdicts"]["required"]) == {item.value for item in JudgeVerdict}
    error = EvaluationOperationError(EvaluationOperationCode.BACKEND_UNMETERED)
    assert isinstance(error, ValueError) and str(error) == "backend_unmetered"
    with pytest.raises(TypeError):
        EvaluationOperationError("backend_unmetered")  # type: ignore[arg-type]


def test_evaluation_run_ref_is_its_own_exact_type() -> None:
    assert EvaluationRunRef is not TrainingRunRef
    assert EvaluationRunRef("r", "p") != TrainingRunRef("r", "p")
    assert EvaluationRunRef.from_dict(RUN.to_dict()) == RUN
    with pytest.raises(TypeError, match="exact TrainingRunRef"):
        RunsAPI(object()).show(RUN)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationRunRef"):
        EvaluationAPI(object()).show(TrainingRunRef("ev-01J", "acme"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationRunRef"):
        EvaluationStart(TrainingRunRef("ev-01J", "acme"), True)  # type: ignore[arg-type]


# --- canonical documents and schemas ---------------------------------------------


def test_result_matches_the_architecture_document_and_its_closed_schema() -> None:
    document = _result().to_dict()
    assert document == {
        "schema_version": "synaptic-evaluation-result/v1",
        "run": {"run_id": "ev-01J", "project_ref": "acme"},
        "state": "succeeded", "backend": "openrouter",
        "model": {"model_ref": "acme/qwen-sft", "model_revision": "a1b2" * 10},
        "cases_total": 120, "cases_scored": 120,
        "scores": [{"name": "composite", "value": 0.812}, {"name": "quality_gated", "value": 0.744}],
        "verdicts": {"passed": 97, "failed": 21, "inconclusive": 2},
        "artifacts": [{"role": "evaluation_results", "sha256": "c" * 64, "size_bytes": 220114}],
        "diagnostic_code": None,
        "usage": _measured_usage().to_dict(),
    }
    jsonschema.Draft202012Validator.check_schema(RESULT_SCHEMA)
    jsonschema.validate(document, RESULT_SCHEMA)
    assert EvaluationResult.from_dict(json.loads(json.dumps(document))) == _result()
    for forbidden in ("raw_response", "conversation_trace"):
        assert forbidden not in json.dumps(document)
        assert forbidden not in json.dumps(RESULT_SCHEMA)


def test_unavailable_usage_is_absent_from_the_result_not_null() -> None:
    local = _result(usage=None)
    document = local.to_dict()
    assert "usage" not in document
    jsonschema.validate(document, RESULT_SCHEMA)
    assert EvaluationResult.from_dict(document) == local
    with pytest.raises(ValueError, match="omitted, not published"):
        _result(usage=UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE))
    with pytest.raises(ValueError, match="omitted when unavailable"):
        EvaluationResult.from_dict(dict(document, usage=None))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, usage=None), RESULT_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, usage=UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE).to_dict()), RESULT_SCHEMA)


def test_plan_round_trips_through_its_closed_schema_and_derives_its_fingerprint() -> None:
    plan = _plan()
    document = plan.to_dict()
    assert document == {
        "schema_version": "synaptic-evaluation-plan/v1",
        "request": {
            "request_id": "request-1", "project_ref": "acme",
            "model": {"model_ref": "acme/qwen-sft", "model_revision": "a1b2" * 10},
            "backend": "openrouter", "scenario_refs": ["grounding", "tool-choice"],
            "preset": "quick", "tags": ["smoke"], "case_limit": 120,
        },
        "cases_total": 120, "scenario_digest": "b" * 64,
    }
    jsonschema.Draft202012Validator.check_schema(PLAN_SCHEMA)
    jsonschema.validate(document, PLAN_SCHEMA)
    assert EvaluationPlan.from_dict(json.loads(json.dumps(document))) == plan
    assert "plan_fingerprint" not in json.dumps(document)
    assert len(plan.plan_fingerprint) == 64
    assert _plan(_request(case_limit=None)).plan_fingerprint != plan.plan_fingerprint
    for field, bad in (("scenario_refs", []), ("case_limit", 0), ("unknown", 1)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(dict(document, request=dict(document["request"], **{field: bad})), PLAN_SCHEMA)
    minimal = _request(preset=None, tags=(), case_limit=None)
    jsonschema.validate(_plan(minimal).to_dict(), PLAN_SCHEMA)
    assert EvaluationRequest.from_dict(minimal.to_dict()) == minimal


def test_preflight_binds_the_exact_plan_and_round_trips() -> None:
    plan = _plan()
    preflight = _preflight(plan)
    assert preflight.binds(plan)
    assert not preflight.binds(_plan(_request(backend="vllm")))
    assert EvaluationPreflight.from_dict(preflight.to_dict()) == preflight
    assert preflight.to_dict()["authorization"] == {
        "evaluation.start": {"paid_effect": True, "maximum_cost_minor_units": 2000, "currency": "USD"},
    }
    assert preflight.is_expired("2026-09-17T12:05:00Z") and not preflight.is_expired("2026-09-17T12:04:59Z")
    with pytest.raises(ValueError, match="requires a diagnostic code"):
        _preflight(plan, ready=False)
    assert _preflight(plan, ready=False, diagnostic_codes=("scenario_invalid",)).diagnostic_codes == ("scenario_invalid",)
    with pytest.raises(ValueError, match="later than checked_at"):
        _preflight(plan, expires_at="2026-09-17T12:00:00Z")
    with pytest.raises(TypeError):
        _preflight(plan, authorization=[AuthorizationRequirement("evaluation.start", False)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationPlan"):
        preflight.binds(object())  # type: ignore[arg-type]


def test_result_and_outcome_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="sum to cases_scored"):
        _result(verdicts=EvaluationVerdictCounts(97, 21, 1))
    with pytest.raises(ValueError, match="must not exceed cases_total"):
        _result(cases_scored=121, verdicts=EvaluationVerdictCounts(98, 21, 2))
    with pytest.raises(ValueError, match="score names must be unique"):
        _result(scores=(ScoreV1("composite", 0.1), ScoreV1("composite", 0.2)))
    with pytest.raises(ValueError, match="artifact roles must be unique"):
        _result(artifacts=(VerifiedArtifact("r", "c" * 64, 1), VerifiedArtifact("r", "d" * 64, 2)))
    with pytest.raises(ValueError, match="finite"):
        ScoreV1("composite", float("nan"))
    with pytest.raises(TypeError, match="must be a number"):
        ScoreV1("composite", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported evaluation result schema version"):
        _result(schema_version="synaptic-evaluation-result/v2")
    with pytest.raises(ValueError, match="must not exceed cases_total"):
        EvaluationOutcome(RUN, EvaluationRunState.RUNNING, 1, 2)
    with pytest.raises(TypeError, match="exact integer"):
        EvaluationOutcome(RUN, EvaluationRunState.RUNNING, 1.0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown state"):
        EvaluationOutcome.from_dict(dict(_outcome().to_dict(), state="reconcile_required"))
    assert EvaluationOutcome.from_dict(_outcome().to_dict()) == _outcome()
    with pytest.raises(ValueError, match="requires at least 1"):
        _request(scenario_refs=())
    with pytest.raises(ValueError, match="must be unique"):
        _request(scenario_refs=("a", "a"))
    assert _request(scenario_refs=("z", "a")).scenario_refs == ("a", "z")
    with pytest.raises(ValueError, match="cases_total"):
        EvaluationPlan("synaptic-evaluation-plan/v1", _request(), 0, "b" * 64)


def test_paging_rule_matches_the_runs_facade() -> None:
    listing = EvaluationListRequest("acme", limit=1)
    assert EvaluationPage(listing, (_outcome(),), "c", True).truncated
    with pytest.raises(ValueError, match="matrix"):
        EvaluationPage(listing, (_outcome(),), None, True)
    with pytest.raises(ValueError, match="matrix"):
        EvaluationPage(listing, (_outcome(),), "c", False)
    with pytest.raises(ValueError, match="a truncated page"):
        EvaluationPage(listing, (), "c", True)
    with pytest.raises(ValueError, match="exceed requested limit"):
        EvaluationPage(listing, (_outcome(), _outcome(EvaluationRunRef("ev-02", "acme"))))
    with pytest.raises(ValueError, match="project does not match"):
        EvaluationPage(listing, (_outcome(EvaluationRunRef("ev-02", "other")),))
    with pytest.raises(ValueError, match="limit must be"):
        EvaluationListRequest("acme", limit=101)
    with pytest.raises(ValueError, match="ASCII"):
        EvaluationListRequest("acme", cursor="é")


# --- facade discipline -------------------------------------------------------------


def test_evaluation_facade_reconstructs_and_binds_every_callback_result() -> None:
    request = _request()
    plan = _plan(request)
    preflight = _preflight(plan)
    outcome = _outcome()
    listing = EvaluationListRequest("acme", limit=1)
    result_request = EvaluationResultRequest(RUN)
    observations = ObservationsRequest(_stream(), after_sequence=40, limit=1)
    record = _observation()

    class Operations:
        def plan(self, supplied):
            assert supplied is not request and supplied == request
            return plan

        def preflight(self, supplied):
            assert supplied is not plan and supplied == plan
            return preflight

        def start(self, supplied):
            assert supplied is not plan and supplied == plan
            return EvaluationStart(RUN, True)

        def show(self, supplied):
            assert supplied is not RUN and supplied == RUN
            return outcome

        def cancel(self, supplied, reason):
            assert supplied is not RUN and reason == "operator request"
            return outcome

        def list(self, supplied):
            assert supplied is not listing and supplied == listing
            return EvaluationPage(supplied, (outcome,))

        def result(self, supplied):
            assert supplied is not result_request and supplied == result_request
            return _result()

        def observations(self, supplied):
            assert supplied is not observations and supplied == observations
            return ObservationPage(supplied, (record,), 41, True)

    api = EvaluationAPI(Operations())
    assert api.plan(request) == plan
    assert api.preflight(plan) == preflight
    assert api.start(plan) == EvaluationStart(RUN, True)
    assert api.show(RUN) == outcome
    assert api.cancel(RUN, "operator request") == outcome
    assert api.list(listing) == EvaluationPage(listing, (outcome,))
    assert api.result(result_request) == _result()
    assert api.observations(observations) == ObservationPage(observations, (record,), 41, True)

    public = api.show(RUN)
    assert public is not outcome and public.run is not outcome.run
    object.__setattr__(outcome.run, "run_id", "mutated")
    assert public.run == RUN


@pytest.mark.parametrize(
    ("verb", "callback", "argument"),
    [
        ("plan", lambda supplied: _plan(_request(request_id="other")), _request()),
        ("preflight", lambda supplied: _preflight(_plan(_request(backend="vllm"))), _plan()),
        ("start", lambda supplied: EvaluationStart(EvaluationRunRef("ev-02", "other"), True), _plan()),
        ("show", lambda supplied: _outcome(EvaluationRunRef("ev-02", "acme")), RUN),
        ("list", lambda supplied: EvaluationPage(EvaluationListRequest("acme", limit=2), ()), EvaluationListRequest("acme", limit=1)),
        ("result", lambda supplied: _result(EvaluationRunRef("ev-02", "acme")), EvaluationResultRequest(RUN)),
        (
            "observations",
            lambda supplied: ObservationPage(ObservationsRequest(_stream(), limit=2), ()),
            ObservationsRequest(_stream(), limit=1),
        ),
    ],
)
def test_evaluation_facade_rejects_callback_identity_drift(verb, callback, argument) -> None:
    operations = type("Operations", (), {verb: staticmethod(callback)})()
    with pytest.raises(ValueError, match="bind"):
        getattr(EvaluationAPI(operations), verb)(argument)


@pytest.mark.parametrize("verb", sorted(VERBS))
@pytest.mark.parametrize("raises", [False, True])
def test_evaluation_facade_rejects_presented_input_mutation_on_return_and_raise(verb, raises) -> None:
    request = _request()
    plan = _plan(request)
    listing = EvaluationListRequest("acme", limit=1)
    result_request = EvaluationResultRequest(RUN)
    observations = ObservationsRequest(_stream(), limit=1)

    class Operations:
        def __getattr__(self, name):
            def callback(value, *extra):
                if name == "plan":
                    object.__setattr__(value, "request_id", "changed")
                elif name in {"preflight", "start"}:
                    object.__setattr__(value.request, "request_id", "changed")
                elif name == "list":
                    object.__setattr__(value, "project_ref", "changed")
                elif name == "result":
                    object.__setattr__(value.run, "run_id", "changed")
                elif name == "observations":
                    object.__setattr__(value.stream, "entity_id", "changed")
                else:
                    object.__setattr__(value, "run_id", "changed")
                if raises:
                    raise RuntimeError("collaborator detail")
                return object()
            return callback

    api = EvaluationAPI(Operations())
    invocation = {
        "plan": lambda: api.plan(request),
        "preflight": lambda: api.preflight(plan),
        "start": lambda: api.start(plan),
        "show": lambda: api.show(RUN),
        "cancel": lambda: api.cancel(RUN, "operator request"),
        "list": lambda: api.list(listing),
        "result": lambda: api.result(result_request),
        "observations": lambda: api.observations(observations),
    }[verb]
    with pytest.raises(ValueError, match="input changed") as captured:
        invocation()
    if raises:
        pending = [captured.value]
        seen = set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            assert type(error) is not RuntimeError
            assert "collaborator detail" not in str(error)
            pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)


def test_observations_verb_admits_only_evaluation_streams_before_any_callback() -> None:
    calls = []

    class Operations:
        def observations(self, supplied):
            calls.append(supplied)
            return ObservationPage(supplied, ())

    api = EvaluationAPI(Operations())
    training = ObservationsRequest(ObservationStreamRef(ObservationFamily.TRAINING, "acme", "run-1"))
    with pytest.raises(ValueError, match="evaluation stream"):
        api.observations(training)
    assert calls == []
    page = api.observations(ObservationsRequest(_stream()))
    assert page.records == () and len(calls) == 1


def test_facade_rejects_nonexact_results_and_wrong_types() -> None:
    class Operations:
        def show(self, run):
            return object()

        def plan(self, request):
            return _plan()

        def result(self, request):
            return _outcome()

    api = EvaluationAPI(Operations())
    with pytest.raises(TypeError, match="exact EvaluationOutcome"):
        api.show(RUN)
    with pytest.raises(TypeError, match="exact EvaluationRequest"):
        api.plan(_plan())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationResult"):
        api.result(EvaluationResultRequest(RUN))
    with pytest.raises(TypeError, match="exact EvaluationPlan"):
        api.preflight(_request())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationResultRequest"):
        api.result(RUN)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact ObservationsRequest"):
        api.observations(_stream())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reason must be an exact string"):
        api.cancel(RUN, None)  # type: ignore[arg-type]


# --- hostile inputs --------------------------------------------------------------------


def test_canonical_evaluation_contracts_reject_nonexact_inputs() -> None:
    class Text(str):
        pass

    class RunSubclass(EvaluationRunRef):
        pass

    class DictSubclass(dict):
        pass

    with pytest.raises(TypeError):
        EvaluationRunRef(Text("ev-01J"), "acme")
    with pytest.raises(TypeError, match="exact object"):
        EvaluationRunRef.from_dict(MappingProxyType({"run_id": "ev-01J", "project_ref": "acme"}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact object"):
        EvaluationPlan.from_dict(DictSubclass(_plan().to_dict()))
    with pytest.raises(TypeError, match="exact string"):
        EvaluationModelRef(Text("acme/qwen-sft"), "rev")
    with pytest.raises(TypeError, match="exact integer"):
        EvaluationVerdictCounts(1.0, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact EvaluationRunRef"):
        EvaluationAPI(object()).show(RunSubclass("ev-01J", "acme"))
    with pytest.raises(TypeError, match="exact array"):
        EvaluationRequest.from_dict(dict(_request().to_dict(), scenario_refs=("a",)))
    with pytest.raises(TypeError, match="exact tuple"):
        _request(scenario_refs=["a"])  # type: ignore[arg-type]


@pytest.mark.parametrize("container", [MappingProxyType, dict])
@pytest.mark.parametrize("parser", [EvaluationResult.from_dict, EvaluationPlan.from_dict, EvaluationOutcome.from_dict])
def test_evaluation_parsers_require_exact_builtin_objects(container, parser) -> None:
    document = {
        EvaluationResult.from_dict: _result().to_dict(),
        EvaluationPlan.from_dict: _plan().to_dict(),
        EvaluationOutcome.from_dict: _outcome().to_dict(),
    }[parser]
    hostile = container(document)
    if container is dict:
        class DictSubclass(dict):
            pass
        hostile = DictSubclass(document)
    with pytest.raises(TypeError, match="exact object"):
        parser(hostile)  # type: ignore[arg-type]


@pytest.mark.parametrize("parser", [EvaluationResult.from_dict, EvaluationPlan.from_dict, EvaluationOutcome.from_dict])
def test_evaluation_parsers_reject_hostile_field_name_subclasses_without_callbacks(parser) -> None:
    class Field(str):
        calls = 0

        def __hash__(self):
            type(self).calls += 1
            if type(self).calls > 1:
                raise RuntimeError("secret callback")
            return str.__hash__(self)

    document = {
        EvaluationResult.from_dict: _result().to_dict(),
        EvaluationPlan.from_dict: _plan().to_dict(),
        EvaluationOutcome.from_dict: _outcome().to_dict(),
    }[parser]
    value = dict(document)
    key = "state" if "state" in value else "cases_total"
    original = value.pop(key)
    dict.__setitem__(value, Field(key), original)
    with pytest.raises(TypeError, match="field names") as captured:
        parser(value)
    assert captured.value.__cause__ is None
    assert Field.calls == 1


# --- host composition and import closure --------------------------------------------


def test_api_host_composes_the_evaluation_facade() -> None:
    class Clock:
        def now(self):
            return "2026-09-17T12:00:00Z"

    class Operations:
        def show(self, run):
            return _outcome()

    ports = HostPorts(
        training=object(), runs=object(), artifacts=None, evaluation=Operations(),
        chat=None, data=None, pipelines=None, clock=Clock(),
    )
    host = APIHost(ports)
    assert type(host.evaluation) is EvaluationAPI
    assert host.evaluation.show(RUN) == _outcome()
    with pytest.raises(RuntimeError, match="did not compose the 'evaluation' family"):
        APIHost(HostPorts(
            training=object(), runs=object(), artifacts=None, evaluation=None,
            chat=None, data=None, pipelines=None, clock=Clock(),
        )).evaluation


def test_evaluation_facade_imports_no_engine_provider_or_database_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.evaluation_facade
print(json.dumps(sorted(n for n in sys.modules if n in ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod') or n.startswith(('tuner.', 'modal.', 'sqlite3.', 'huggingface_hub.', 'runpod.')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
