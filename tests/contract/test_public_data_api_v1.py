"""Contract tests for the DataAPI facade (api-facade slice 9, contract half).

Pins the closed vocabularies, the exact record fields, both closed schemas,
the protocol shape, the terminal-with-artifacts rulings for
``partially_succeeded`` and ``cancelled``, the paging rule for ``list``,
``datasets`` and ``validate``, and the ``RunsAPI._call`` detach-and-revalidate
discipline, mirroring ``test_public_evaluation_api_v1.py``.
"""

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
from synaptic_tuner.api.v1 import data_facade
from synaptic_tuner.api.v1.data_facade import (
    DataAPI,
    DataListRequest,
    DataMode,
    DataOperationCode,
    DataOperationError,
    DataOperations,
    DataOutcome,
    DataPage,
    DataPlan,
    DataPreflight,
    DataRequest,
    DataResult,
    DataRunRef,
    DataRunState,
    DataScenarioOutcome,
    DataScenarioTarget,
    DataStart,
    DatasetDescriptor,
    DatasetListRequest,
    DatasetPage,
    DatasetValidateRequest,
    ValidationFinding,
    ValidationFindingCode,
    ValidationReport,
)
from synaptic_tuner.api.v1.evaluation_facade import EvaluationRunRef
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    DataRowWrittenPayloadV1,
    ObservationFamily,
    ObservationKind,
    ObservationPage,
    ObservationRecordV1,
    ObservationStreamRef,
    ObservationsRequest,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.training_facade import AuthorizationRequirement
from synaptic_tuner.api.v1.usage import SpendRef, UsageAvailability, UsageRecordV1


ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = json.loads((ROOT / "schemas/synaptic-dataset-plan-v1.schema.json").read_text(encoding="utf-8"))
RESULT_SCHEMA = json.loads((ROOT / "schemas/synaptic-dataset-result-v1.schema.json").read_text(encoding="utf-8"))

VERBS = {"plan", "preflight", "start", "show", "cancel", "list", "datasets", "validate", "observations"}
RUN = DataRunRef("data-01J", "acme")
TARGETS = (DataScenarioTarget("vault_search", 3), DataScenarioTarget("vault_move", 2))
OUTCOMES = (DataScenarioOutcome("vault_search", 3, 3), DataScenarioOutcome("vault_move", 2, 2))
ARTIFACTS = (
    VerifiedArtifact("dataset_jsonl", "c" * 64, 220114),
    VerifiedArtifact("dataset_metadata", "d" * 64, 412),
)


def _request(**changes: object) -> DataRequest:
    values: dict[str, object] = {
        "request_id": "request-1", "project_ref": "acme", "mode": DataMode.GENERATE,
        "backend": "lmstudio", "model": "qwen3-8b", "output_ref": "/data/out/run.jsonl",
        "scenarios": TARGETS, "tags": ("smoke",),
    }
    values.update(changes)
    return DataRequest(**values)  # type: ignore[arg-type]


def _improve_request(**changes: object) -> DataRequest:
    values: dict[str, object] = {
        "request_id": "request-2", "project_ref": "acme", "mode": DataMode.IMPROVE,
        "backend": "ollama", "model": "qwen3-8b", "output_ref": "/data/out/improved.jsonl",
        "input_dataset_ref": "/data/in/seed.jsonl", "rubric_refs": ("data_quality",),
        "max_iterations": 2,
    }
    values.update(changes)
    return DataRequest(**values)  # type: ignore[arg-type]


def _plan(request: DataRequest | None = None) -> DataPlan:
    request = request or _request()
    scenarios = request.scenarios or (DataScenarioTarget(request.input_dataset_ref, 5),)  # type: ignore[arg-type]
    return DataPlan("synaptic-dataset-plan/v1", request, scenarios, sum(item.rows for item in scenarios), "b" * 64)


def _preflight(plan: DataPlan, **changes: object) -> DataPreflight:
    values: dict[str, object] = {
        "plan_fingerprint": plan.plan_fingerprint, "ready": True,
        "checked_at": "2026-09-17T12:00:00Z", "expires_at": "2026-09-17T12:05:00Z",
        "authorization": (AuthorizationRequirement("data.start", False),),
    }
    values.update(changes)
    return DataPreflight(**values)  # type: ignore[arg-type]


def _outcome(run: DataRunRef = RUN, **changes: object) -> DataOutcome:
    values: dict[str, object] = {
        "run": run, "state": DataRunState.RUNNING, "rows_requested": 5, "rows_written": 1,
        "scenarios": (DataScenarioOutcome("vault_search", 3, 1), DataScenarioOutcome("vault_move", 2, 0)),
    }
    values.update(changes)
    return DataOutcome(**values)  # type: ignore[arg-type]


def _measured_usage() -> UsageRecordV1:
    return UsageRecordV1(
        "synaptic-usage/v1", UsageAvailability.MEASURED, 412334, 88120, 1874, "USD",
        SpendRef("openrouter", "acct-main", "spend-01J"),
    )


def _result(run: DataRunRef = RUN, **changes: object) -> DataResult:
    values: dict[str, object] = {
        "schema_version": "synaptic-dataset-result/v1", "run": run, "state": DataRunState.SUCCEEDED,
        "mode": DataMode.GENERATE, "backend": "lmstudio", "model": "qwen3-8b",
        "rows_requested": 5, "rows_written": 5, "scenarios": OUTCOMES, "artifacts": ARTIFACTS,
        "diagnostic_code": None,
    }
    values.update(changes)
    return DataResult(**values)  # type: ignore[arg-type]


def _stream() -> ObservationStreamRef:
    return ObservationStreamRef(ObservationFamily.DATA, "acme", "data-01J")


def _observation(sequence: int = 41) -> ObservationRecordV1:
    return ObservationRecordV1(
        "synaptic-observation/v1", _stream(), sequence, "2026-09-17T12:00:04Z",
        ObservationKind.DATA_ROW_WRITTEN, DataRowWrittenPayloadV1("vault_search", 1),
    )


def _descriptors(*names: str) -> tuple[DatasetDescriptor, ...]:
    return tuple(DatasetDescriptor(f"/data/out/{name}.jsonl", 10, f"/data/out/{name}.jsonl.meta.json") for name in names)


# --- exports, verbs, closed vocabularies ---------------------------------------


def test_root_data_exports_are_the_canonical_facade_identities() -> None:
    for name in data_facade.__all__:
        assert getattr(v1, name) is getattr(data_facade, name), name
    assert set(data_facade.__all__) <= set(v1.__all__)


def test_data_api_and_operations_have_only_the_accepted_verbs() -> None:
    def verbs(owner: type) -> set[str]:
        return {
            name for name, member in owner.__dict__.items()
            if not name.startswith("_") and inspect.isfunction(member)
        }
    assert verbs(DataAPI) == VERBS
    assert verbs(DataOperations) == VERBS
    for name in VERBS:
        assert tuple(inspect.signature(getattr(DataAPI, name)).parameters) == tuple(
            inspect.signature(getattr(DataOperations, name)).parameters
        ), name
    assert inspect.signature(DataAPI.cancel).parameters["reason"].annotation == "str"
    assert "result" not in verbs(DataAPI) and "reconcile" not in verbs(DataAPI)


def test_closed_vocabularies_match_the_architecture() -> None:
    assert tuple(item.value for item in DataRunState) == (
        "planned", "running", "succeeded", "partially_succeeded", "failed",
        "cancel_requested", "cancelled",
    )
    assert {item for item in DataRunState if item.terminal} == {
        DataRunState.SUCCEEDED, DataRunState.PARTIALLY_SUCCEEDED, DataRunState.FAILED, DataRunState.CANCELLED,
    }
    assert tuple(item.value for item in DataOperationCode) == (
        "run_missing", "cursor_invalid", "scenario_invalid", "manifest_invalid", "backend_unavailable",
        "backend_unmetered", "spend_indeterminate", "output_conflict", "cancel_ineligible",
        "dataset_missing", "state_conflict", "integrity_error",
    )
    assert tuple(item.value for item in DataMode) == ("generate", "improve")
    assert set(RESULT_SCHEMA["properties"]["state"]["enum"]) == {item.value for item in DataRunState}
    assert set(RESULT_SCHEMA["properties"]["mode"]["enum"]) == {item.value for item in DataMode}
    assert set(RESULT_SCHEMA["properties"]["artifacts"]["items"]["properties"]["role"]["enum"]) == {
        "dataset_jsonl", "dataset_metadata",
    }
    error = DataOperationError(DataOperationCode.BACKEND_UNMETERED)
    assert isinstance(error, ValueError) and str(error) == "backend_unmetered"
    assert error.code is DataOperationCode.BACKEND_UNMETERED
    with pytest.raises(TypeError):
        DataOperationError("backend_unmetered")  # type: ignore[arg-type]


def test_data_run_ref_is_its_own_exact_type() -> None:
    assert DataRunRef is not TrainingRunRef and DataRunRef is not EvaluationRunRef
    assert DataRunRef("r", "p") != TrainingRunRef("r", "p")
    assert DataRunRef("r", "p") != EvaluationRunRef("r", "p")
    assert DataRunRef.from_dict(RUN.to_dict()) == RUN


# --- records and schemas ----------------------------------------------------------


def test_result_matches_the_architecture_document_and_its_closed_schema() -> None:
    document = _result(usage=_measured_usage()).to_dict()
    assert tuple(document) == (
        "schema_version", "run", "state", "mode", "backend", "model", "rows_requested",
        "rows_written", "scenarios", "artifacts", "diagnostic_code", "usage",
    )
    assert document["artifacts"] == [
        {"role": "dataset_jsonl", "sha256": "c" * 64, "size_bytes": 220114},
        {"role": "dataset_metadata", "sha256": "d" * 64, "size_bytes": 412},
    ]
    jsonschema.Draft202012Validator(RESULT_SCHEMA).validate(document)
    assert DataResult.from_dict(json.loads(json.dumps(document))) == _result(usage=_measured_usage())
    assert RESULT_SCHEMA["additionalProperties"] is False
    for name, subschema in RESULT_SCHEMA["properties"].items():
        if subschema.get("type") == "object":
            assert subschema["additionalProperties"] is False, name
    assert set(RESULT_SCHEMA["required"]) == set(document) - {"usage"}
    assert _result().outcome() == DataOutcome(RUN, DataRunState.SUCCEEDED, 5, 5, OUTCOMES, ARTIFACTS, None)


def test_unavailable_usage_is_absent_from_the_result_not_null() -> None:
    document = _result().to_dict()
    assert "usage" not in document
    jsonschema.Draft202012Validator(RESULT_SCHEMA).validate(document)
    with pytest.raises(ValueError, match="omitted when unavailable"):
        DataResult.from_dict({**document, "usage": None})
    with pytest.raises(ValueError, match="unavailable usage"):
        _result(usage=UsageRecordV1("synaptic-usage/v1", UsageAvailability.UNAVAILABLE, 0, 0, None, None, None))


@pytest.mark.parametrize("request_builder", [_request, _improve_request])
def test_plan_round_trips_through_its_closed_schema_and_derives_its_fingerprint(request_builder) -> None:
    plan = _plan(request_builder())
    document = plan.to_dict()
    assert tuple(document) == ("schema_version", "request", "scenarios", "rows_requested", "source_digest")
    jsonschema.Draft202012Validator(PLAN_SCHEMA).validate(document)
    assert DataPlan.from_dict(json.loads(json.dumps(document))) == plan
    assert plan.plan_fingerprint == DataPlan.from_dict(document).plan_fingerprint
    assert len(plan.plan_fingerprint) == 64
    assert plan.plan_fingerprint != _plan(request_builder(request_id="other")).plan_fingerprint
    assert PLAN_SCHEMA["additionalProperties"] is False
    assert PLAN_SCHEMA["properties"]["request"]["additionalProperties"] is False


def test_plan_schema_is_closed_per_mode() -> None:
    validator = jsonschema.Draft202012Validator(PLAN_SCHEMA)
    generate = _plan().to_dict()
    generate["request"]["input_dataset_ref"] = "/data/in/seed.jsonl"
    assert not validator.is_valid(generate)
    improve = _plan(_improve_request()).to_dict()
    improve["request"]["scenarios"] = [{"scenario_ref": "x", "rows": 1}]
    assert not validator.is_valid(improve)
    improve = _plan(_improve_request()).to_dict()
    improve["request"]["rubric_refs"] = []
    assert not validator.is_valid(improve)
    assert not validator.is_valid({**_plan().to_dict(), "extra": 1})


def test_request_mode_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="at least 1 scenario"):
        _request(scenarios=())
    with pytest.raises(ValueError, match="takes no input_dataset_ref"):
        _request(input_dataset_ref="/data/in/seed.jsonl")
    with pytest.raises(ValueError, match="takes no scenario targets"):
        _improve_request(scenarios=TARGETS)
    with pytest.raises(ValueError, match="requires input_dataset_ref"):
        _improve_request(input_dataset_ref=None)
    with pytest.raises(ValueError, match="at least 1 rubric"):
        _improve_request(rubric_refs=())
    with pytest.raises(ValueError, match="must sum"):
        DataPlan("synaptic-dataset-plan/v1", _request(), TARGETS, 4, "b" * 64)
    with pytest.raises(ValueError, match="must equal the request targets"):
        DataPlan("synaptic-dataset-plan/v1", _request(), (DataScenarioTarget("other", 5),), 5, "b" * 64)


def test_preflight_binds_the_exact_plan_and_round_trips() -> None:
    plan = _plan()
    preflight = _preflight(plan)
    assert preflight.binds(plan) and not preflight.binds(_plan(_request(request_id="other")))
    assert DataPreflight.from_dict(preflight.to_dict()) == preflight
    assert not preflight.is_expired("2026-09-17T12:04:59Z") and preflight.is_expired("2026-09-17T12:05:00Z")
    refused = _preflight(plan, ready=False, diagnostic_codes=("backend_unmetered",))
    assert refused.diagnostic_codes == ("backend_unmetered",)
    with pytest.raises(ValueError, match="requires a diagnostic code"):
        _preflight(plan, ready=False)
    with pytest.raises(ValueError, match="later than"):
        _preflight(plan, expires_at="2026-09-17T12:00:00Z")


def test_partially_succeeded_and_cancelled_are_terminal_and_list_their_artifact() -> None:
    partial_scenarios = (DataScenarioOutcome("vault_search", 3, 3), DataScenarioOutcome("vault_move", 2, 1))
    for state in (DataRunState.PARTIALLY_SUCCEEDED, DataRunState.CANCELLED):
        assert state.terminal
        result = _result(
            state=state, rows_written=4, scenarios=partial_scenarios, diagnostic_code="backend_unavailable",
        )
        document = result.to_dict()
        jsonschema.Draft202012Validator(RESULT_SCHEMA).validate(document)
        assert DataResult.from_dict(document).artifacts == ARTIFACTS
        assert result.outcome().artifacts == ARTIFACTS and result.outcome().state is state
    assert not DataRunState.CANCEL_REQUESTED.terminal and not DataRunState.RUNNING.terminal


def test_result_and_outcome_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="rows_written must not exceed"):
        _result(rows_written=6)
    with pytest.raises(ValueError, match="rows_requested must sum"):
        _result(scenarios=(DataScenarioOutcome("vault_search", 5, 5), DataScenarioOutcome("vault_move", 2, 0)))
    with pytest.raises(ValueError, match="rows_written must sum"):
        _result(scenarios=(DataScenarioOutcome("vault_search", 3, 2), DataScenarioOutcome("vault_move", 2, 2)))
    with pytest.raises(ValueError, match="scenario refs must be unique"):
        _result(scenarios=(DataScenarioOutcome("vault_search", 3, 3), DataScenarioOutcome("vault_search", 2, 2)))
    with pytest.raises(ValueError, match="roles must be one of"):
        _result(artifacts=(VerifiedArtifact("evaluation_results", "c" * 64, 1),))
    with pytest.raises(ValueError, match="roles must be unique"):
        _result(artifacts=(ARTIFACTS[0], VerifiedArtifact("dataset_jsonl", "e" * 64, 2)))
    with pytest.raises(ValueError):
        DataScenarioOutcome("vault_search", 3, 4)
    with pytest.raises(ValueError):
        DataScenarioTarget("vault_search", 0)
    with pytest.raises(ValueError, match="unsupported dataset result schema"):
        _result(schema_version="synaptic-evaluation-result/v1")
    with pytest.raises(TypeError):
        _result(mode="generate")
    with pytest.raises(ValueError):
        DataResult.from_dict({**_result().to_dict(), "extra": 1})
    with pytest.raises(ValueError):
        DataOutcome.from_dict({**_outcome().to_dict(), "rows": 1})


# --- paging: list, datasets, validate -------------------------------------------------


def test_list_paging_rule_matches_the_runs_facade() -> None:
    listing = DataListRequest("acme", limit=1)
    assert DataPage(listing, (_outcome(),), "c", True).truncated
    with pytest.raises(ValueError, match="matrix"):
        DataPage(listing, (_outcome(),), None, True)
    with pytest.raises(ValueError, match="matrix"):
        DataPage(listing, (_outcome(),), "c", False)
    with pytest.raises(ValueError, match="a truncated page"):
        DataPage(listing, (), "c", True)
    with pytest.raises(ValueError, match="exceed requested limit"):
        DataPage(listing, (_outcome(), _outcome(DataRunRef("data-02", "acme"))))
    with pytest.raises(ValueError, match="project does not match"):
        DataPage(listing, (_outcome(DataRunRef("data-02", "other")),))
    with pytest.raises(ValueError, match="limit must be"):
        DataListRequest("acme", limit=101)
    with pytest.raises(ValueError, match="ASCII"):
        DataListRequest("acme", cursor="é")


def test_datasets_paging_rule_is_settled_and_refs_are_ordered() -> None:
    listing = DatasetListRequest("acme", "/data/out", limit=2)
    page = DatasetPage(listing, _descriptors("a", "b"), "c", True)
    assert page.truncated and DatasetPage(listing, _descriptors("a")).next_cursor is None
    with pytest.raises(ValueError, match="matrix"):
        DatasetPage(listing, _descriptors("a"), None, True)
    with pytest.raises(ValueError, match="matrix"):
        DatasetPage(listing, _descriptors("a"), "c", False)
    with pytest.raises(ValueError, match="a truncated page"):
        DatasetPage(listing, (), "c", True)
    with pytest.raises(ValueError, match="exceed requested limit"):
        DatasetPage(listing, _descriptors("a", "b", "c"))
    with pytest.raises(ValueError, match="strictly increasing"):
        DatasetPage(listing, _descriptors("b", "a"))
    with pytest.raises(ValueError, match="strictly increasing"):
        DatasetPage(listing, _descriptors("a", "a"))
    with pytest.raises(ValueError, match="limit must be"):
        DatasetListRequest("acme", "/data/out", limit=0)
    with pytest.raises(ValueError, match="ASCII"):
        DatasetListRequest("acme", "/data/out", cursor="é")
    assert DatasetDescriptor("/data/out/a.jsonl", 0).metadata_ref is None


def test_validate_report_counts_findings_exactly_and_pages() -> None:
    request = DatasetValidateRequest("acme", "/data/out/a.jsonl", limit=4)
    findings = (ValidationFinding(2, ValidationFindingCode.MALFORMED_JSON), ValidationFinding(3, ValidationFindingCode.ROW_NOT_OBJECT))
    report = ValidationReport(request, 4, 2, findings, "v1.4", True)
    assert report.truncated and ValidationReport(request, 0, 0).findings == ()
    assert tuple(item.value for item in ValidationFindingCode) == (
        "malformed_json", "row_not_object", "legacy_metadata_row", "conversations_missing", "conversations_invalid",
    )
    with pytest.raises(ValueError, match="exactly the invalid rows"):
        ValidationReport(request, 4, 3, findings)
    with pytest.raises(ValueError, match="rows_valid must not exceed"):
        ValidationReport(request, 1, 2)
    with pytest.raises(ValueError, match="exceed requested limit"):
        ValidationReport(request, 5, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        ValidationReport(request, 4, 2, (findings[1], findings[0]))
    with pytest.raises(ValueError, match="matrix"):
        ValidationReport(request, 4, 2, findings, None, True)
    with pytest.raises(ValueError, match="a truncated report"):
        ValidationReport(request, 0, 0, (), "v1.0", True)
    with pytest.raises(ValueError):
        ValidationFinding(0, ValidationFindingCode.MALFORMED_JSON)
    with pytest.raises(TypeError, match="exact ValidationFindingCode"):
        ValidationFinding(1, "malformed_json")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ASCII"):
        DatasetValidateRequest("acme", "/data/out/a.jsonl", cursor="é")


# --- facade discipline -------------------------------------------------------------


def test_data_facade_reconstructs_and_binds_every_callback_result() -> None:
    request = _request()
    plan = _plan(request)
    preflight = _preflight(plan)
    outcome = _outcome()
    listing = DataListRequest("acme", limit=1)
    dataset_listing = DatasetListRequest("acme", "/data/out", limit=1)
    validate_request = DatasetValidateRequest("acme", "/data/out/a.jsonl", limit=4)
    observations = ObservationsRequest(_stream(), after_sequence=40, limit=1)
    record = _observation()
    report = ValidationReport(validate_request, 4, 3, (ValidationFinding(2, ValidationFindingCode.MALFORMED_JSON),))

    class Operations:
        def plan(self, supplied):
            assert supplied is not request and supplied == request
            return plan

        def preflight(self, supplied):
            assert supplied is not plan and supplied == plan
            return preflight

        def start(self, supplied):
            assert supplied is not plan and supplied == plan
            return DataStart(RUN, True)

        def show(self, supplied):
            assert supplied is not RUN and supplied == RUN
            return outcome

        def cancel(self, supplied, reason):
            assert supplied is not RUN and reason == "operator request"
            return outcome

        def list(self, supplied):
            assert supplied is not listing and supplied == listing
            return DataPage(supplied, (outcome,))

        def datasets(self, supplied):
            assert supplied is not dataset_listing and supplied == dataset_listing
            return DatasetPage(supplied, _descriptors("a"))

        def validate(self, supplied):
            assert supplied is not validate_request and supplied == validate_request
            return report

        def observations(self, supplied):
            assert supplied is not observations and supplied == observations
            return ObservationPage(supplied, (record,), 41, True)

    api = DataAPI(Operations())
    assert api.plan(request) == plan
    assert api.preflight(plan) == preflight
    assert api.start(plan) == DataStart(RUN, True)
    assert api.show(RUN) == outcome
    assert api.cancel(RUN, "operator request") == outcome
    assert api.list(listing) == DataPage(listing, (outcome,))
    assert api.datasets(dataset_listing) == DatasetPage(dataset_listing, _descriptors("a"))
    assert api.validate(validate_request) == report
    assert api.observations(observations) == ObservationPage(observations, (record,), 41, True)

    public = api.show(RUN)
    assert public is not outcome and public.run is not outcome.run
    object.__setattr__(outcome.run, "run_id", "mutated")
    assert public.run == RUN


@pytest.mark.parametrize(
    ("verb", "callback", "argument"),
    [
        ("plan", lambda supplied: _plan(_request(request_id="other")), _request()),
        ("preflight", lambda supplied: _preflight(_plan(_request(backend="ollama"))), _plan()),
        ("start", lambda supplied: DataStart(DataRunRef("data-02", "other"), True), _plan()),
        ("show", lambda supplied: _outcome(DataRunRef("data-02", "acme")), RUN),
        ("cancel", lambda supplied, reason: _outcome(DataRunRef("data-02", "acme")), RUN),
        ("list", lambda supplied: DataPage(DataListRequest("acme", limit=2), ()), DataListRequest("acme", limit=1)),
        (
            "datasets",
            lambda supplied: DatasetPage(DatasetListRequest("acme", "/data/other", limit=1), ()),
            DatasetListRequest("acme", "/data/out", limit=1),
        ),
        (
            "validate",
            lambda supplied: ValidationReport(DatasetValidateRequest("acme", "/data/out/b.jsonl"), 0, 0),
            DatasetValidateRequest("acme", "/data/out/a.jsonl"),
        ),
        (
            "observations",
            lambda supplied: ObservationPage(ObservationsRequest(_stream(), limit=2), ()),
            ObservationsRequest(_stream(), limit=1),
        ),
    ],
)
def test_data_facade_rejects_callback_identity_drift(verb, callback, argument) -> None:
    operations = type("Operations", (), {verb: staticmethod(callback)})()
    extra = ("operator request",) if verb == "cancel" else ()
    with pytest.raises(ValueError, match="bind"):
        getattr(DataAPI(operations), verb)(argument, *extra)


@pytest.mark.parametrize("verb", sorted(VERBS))
@pytest.mark.parametrize("raises", [False, True])
def test_data_facade_rejects_presented_input_mutation_on_return_and_raise(verb, raises) -> None:
    request = _request()
    plan = _plan(request)
    listing = DataListRequest("acme", limit=1)
    dataset_listing = DatasetListRequest("acme", "/data/out", limit=1)
    validate_request = DatasetValidateRequest("acme", "/data/out/a.jsonl")
    observations = ObservationsRequest(_stream(), limit=1)

    class Operations:
        def __getattr__(self, name):
            def callback(value, *extra):
                if name == "plan":
                    object.__setattr__(value, "request_id", "changed")
                elif name in {"preflight", "start"}:
                    object.__setattr__(value.request, "request_id", "changed")
                elif name in {"list", "datasets", "validate"}:
                    object.__setattr__(value, "project_ref", "changed")
                elif name == "observations":
                    object.__setattr__(value.stream, "entity_id", "changed")
                else:
                    object.__setattr__(value, "run_id", "changed")
                if raises:
                    raise RuntimeError("collaborator detail")
                return object()
            return callback

    api = DataAPI(Operations())
    invocation = {
        "plan": lambda: api.plan(request),
        "preflight": lambda: api.preflight(plan),
        "start": lambda: api.start(plan),
        "show": lambda: api.show(RUN),
        "cancel": lambda: api.cancel(RUN, "operator request"),
        "list": lambda: api.list(listing),
        "datasets": lambda: api.datasets(dataset_listing),
        "validate": lambda: api.validate(validate_request),
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


def test_operation_errors_pass_through_the_facade_unchanged() -> None:
    class Operations:
        def start(self, supplied):
            raise DataOperationError(DataOperationCode.BACKEND_UNMETERED)

        def show(self, supplied):
            raise DataOperationError(DataOperationCode.RUN_MISSING)

    api = DataAPI(Operations())
    with pytest.raises(DataOperationError) as captured:
        api.start(_plan())
    assert captured.value.code is DataOperationCode.BACKEND_UNMETERED
    with pytest.raises(DataOperationError) as captured:
        api.show(RUN)
    assert captured.value.code is DataOperationCode.RUN_MISSING


def test_observations_verb_admits_only_data_streams_before_any_callback() -> None:
    calls = []

    class Operations:
        def observations(self, supplied):
            calls.append(supplied)
            return ObservationPage(supplied, ())

    api = DataAPI(Operations())
    foreign = ObservationsRequest(ObservationStreamRef(ObservationFamily.EVALUATION, "acme", "ev-01J"), limit=1)
    with pytest.raises(ValueError, match="data stream"):
        api.observations(foreign)
    assert calls == []
    accepted = ObservationsRequest(_stream(), limit=1)
    assert api.observations(accepted) == ObservationPage(accepted, ())
    assert len(calls) == 1


def test_facade_rejects_nonexact_results_and_wrong_types() -> None:
    class SubOutcome(DataOutcome):
        pass

    class Operations:
        def show(self, supplied):
            return SubOutcome(RUN, DataRunState.RUNNING, 5, 0, (DataScenarioOutcome("vault_search", 5, 0),))

        def datasets(self, supplied):
            return DataPage(DataListRequest("acme"), ())

        def validate(self, supplied):
            return None

    api = DataAPI(Operations())
    with pytest.raises(TypeError, match="exact DataOutcome"):
        api.show(RUN)
    with pytest.raises(TypeError, match="exact DatasetPage"):
        api.datasets(DatasetListRequest("acme", "/data/out"))
    with pytest.raises(TypeError, match="exact ValidationReport"):
        api.validate(DatasetValidateRequest("acme", "/data/out/a.jsonl"))
    with pytest.raises(TypeError, match="exact DataRunRef"):
        api.show(TrainingRunRef("r", "acme"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact DataPlan"):
        api.start(_request())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact DataRequest"):
        api.plan(_plan())  # type: ignore[arg-type]


@pytest.mark.parametrize("container", [MappingProxyType, dict])
@pytest.mark.parametrize("parser", [DataResult.from_dict, DataPlan.from_dict, DataOutcome.from_dict])
def test_data_parsers_require_exact_builtin_objects(container, parser) -> None:
    source = {
        DataResult.from_dict: _result().to_dict(),
        DataPlan.from_dict: _plan().to_dict(),
        DataOutcome.from_dict: _outcome().to_dict(),
    }[parser]
    if container is dict:
        assert parser(json.loads(json.dumps(source))) is not None
    else:
        with pytest.raises(TypeError):
            parser(container(source))  # type: ignore[arg-type]


@pytest.mark.parametrize("parser", [DataResult.from_dict, DataPlan.from_dict, DataOutcome.from_dict])
def test_data_parsers_reject_hostile_field_name_subclasses_without_callbacks(parser) -> None:
    calls = []

    class HostileKey(str):
        def __eq__(self, other):
            calls.append("eq")
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    source = {
        DataResult.from_dict: _result().to_dict(),
        DataPlan.from_dict: _plan().to_dict(),
        DataOutcome.from_dict: _outcome().to_dict(),
    }[parser]
    hostile = {HostileKey(key): value for key, value in source.items()}
    with pytest.raises(TypeError):
        parser(hostile)
    assert calls == []


# --- host composition and import gate ----------------------------------------------------


def test_api_host_composes_the_data_facade() -> None:
    class Clock:
        def now(self):
            return "2026-09-17T12:00:00Z"

    class Operations:
        def show(self, run):
            return _outcome()

    ports = HostPorts(
        training=object(), runs=object(), artifacts=None, evaluation=None,
        chat=None, data=Operations(), pipelines=None, clock=Clock(),
    )
    host = APIHost(ports)
    assert type(host.data) is DataAPI
    assert host.data.show(RUN) == _outcome()
    with pytest.raises(RuntimeError, match="did not compose the 'data' family"):
        APIHost(HostPorts(
            training=object(), runs=object(), artifacts=None, evaluation=None,
            chat=None, data=None, pipelines=None, clock=Clock(),
        )).data


def test_data_facade_imports_no_engine_provider_synthchat_or_database_modules() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1.data_facade
print(json.dumps(sorted(n for n in sys.modules if n in ('tuner', 'sqlite3', 'modal', 'huggingface_hub', 'runpod', 'SynthChat', 'shared') or n.startswith(('tuner.', 'modal.', 'sqlite3.', 'huggingface_hub.', 'runpod.', 'SynthChat.', 'shared.')))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
