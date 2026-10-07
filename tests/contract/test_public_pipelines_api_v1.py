"""Public ``PipelinesAPI`` contract (api-facade slice 10).

Pins the closed vocabularies (states, stage names, operation codes), the exact
fields of ``synaptic-pipeline-record/v1`` against the architecture example and
its closed schema, the protocol shape, ``attempt_key`` determinism, plan-time
refusal of unadmitted stages, the facade's detach-and-revalidate discipline and
the package import gate. No reference module is imported here.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1 import pipelines_facade
from synaptic_tuner.api.v1.evaluation_facade import EvaluationRunRef
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    ObservationFamily, ObservationKind, ObservationsRequest, ObservationStreamRef,
    PIPELINE_STAGE_OUTCOMES,
)
from synaptic_tuner.api.v1.pipelines_facade import (
    ADMITTED_STAGES, PipelineEvaluateSpec, PipelineListRequest, PipelineOperationCode,
    PipelineOperationError, PipelinePage, PipelinePlan, PipelineRecord, PipelineRef,
    PipelineRequest, PipelineStage, PipelineStageName, PipelineStart, PipelineState,
    PipelineTrainSpec, PipelinesAPI, PipelinesOperations, StageState, stage_attempt_key,
    stage_input_digest,
)
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact


ROOT = Path(__file__).resolve().parents[2]
RECORD_SCHEMA = json.loads((ROOT / "schemas/synaptic-pipeline-record-v1.schema.json").read_text(encoding="utf-8"))
VERBS = frozenset({"plan", "start", "show", "resume", "cancel", "list", "observations"})
D = tuple(character * 64 for character in "123456789abcdef")
PIPELINE = PipelineRef("pl-01J", "acme")
ADAPTER = VerifiedArtifact("adapter", D[0], 84)
TRAIN_RUN = TrainingRunRef("modal-sft-01J", "acme")
EVAL_RUN = EvaluationRunRef("ev-01J", "acme")


def _request(**changes: object) -> PipelineRequest:
    values: dict[str, object] = {
        "request_id": "req-1", "project_ref": "acme",
        "stages": (PipelineStageName.TRAIN, PipelineStageName.EVALUATE),
        "train": PipelineTrainSpec('{"method":"sft"}', ProviderRef("fake", "profile")),
        "evaluate": PipelineEvaluateSpec("local", ("alpha.yaml",), "fake-model"),
    }
    values.update(changes)
    return PipelineRequest(**values)  # type: ignore[arg-type]


def _record(state: PipelineState = PipelineState.PARTIALLY_SUCCEEDED, spec_digest: str = D[1]) -> PipelineRecord:
    train = PipelineStage(
        PipelineStageName.TRAIN, StageState.SUCCEEDED,
        stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.TRAIN, spec_digest, ()),
        TRAIN_RUN, (), (ADAPTER,),
    )
    evaluate = PipelineStage(
        PipelineStageName.EVALUATE, StageState.FAILED,
        stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.EVALUATE, spec_digest, (ADAPTER,)),
        EVAL_RUN, (ADAPTER,), (),
    )
    return PipelineRecord("synaptic-pipeline-record/v1", PIPELINE, spec_digest, state, 6, (train, evaluate))


# --- exports and vocabularies -----------------------------------------------------------


def test_root_pipeline_exports_are_the_canonical_facade_identities() -> None:
    for name in pipelines_facade.__all__:
        assert getattr(v1, name) is getattr(pipelines_facade, name), name
        assert name in v1.__all__, name
    assert set(v1._LAZY_MODULE_ATTRIBUTES["pipelines_facade"]) == set(pipelines_facade.__all__)
    with pytest.raises(ModuleNotFoundError):
        __import__("synaptic_tuner.api.v1.pipelines")


def test_pipelines_api_and_operations_have_only_the_accepted_verbs() -> None:
    def verbs(owner: type) -> set[str]:
        return {
            name for name, member in owner.__dict__.items()
            if not name.startswith("_") and inspect.isfunction(member)
        }
    assert verbs(PipelinesAPI) == VERBS
    assert verbs(PipelinesOperations) == VERBS
    for name in VERBS:
        assert tuple(inspect.signature(getattr(PipelinesAPI, name)).parameters) == tuple(
            inspect.signature(getattr(PipelinesOperations, name)).parameters
        ), name
    assert PipelinesAPI.__slots__ == ("_operations",)


def test_closed_vocabularies_match_the_architecture() -> None:
    assert tuple(item.value for item in PipelineState) == (
        "planned", "running", "succeeded", "partially_succeeded", "failed",
        "reconcile_required", "cancel_requested", "cancelled",
    )
    assert tuple(item.value for item in PipelineStageName) == (
        "train", "evaluate", "loss", "analysis", "recommendation",
    )
    assert ADMITTED_STAGES == frozenset({PipelineStageName.TRAIN, PipelineStageName.EVALUATE})
    assert tuple(item.value for item in StageState) == (
        "planned", "running", "succeeded", "failed", "skipped", "cancelled", "reconcile_required",
    )
    assert tuple(item.value for item in PipelineOperationCode) == (
        "pipeline_missing", "cursor_invalid", "spec_invalid", "stage_unsupported",
        "stage_input_missing", "stage_digest_mismatch", "resume_ineligible", "cancel_ineligible",
        "state_conflict", "integrity_error",
    )
    # Every terminal stage state that reaches the observation stream is one of §4.2's outcomes.
    terminal = {item.value for item in pipelines_facade.TERMINAL_STAGE_STATES}
    assert terminal == PIPELINE_STAGE_OUTCOMES
    assert set(RECORD_SCHEMA["properties"]["state"]["enum"]) == {item.value for item in PipelineState}
    stage_schema = RECORD_SCHEMA["$defs"]["stage"]["properties"]
    assert set(stage_schema["state"]["enum"]) == {item.value for item in StageState}
    assert set(stage_schema["name"]["enum"]) == {item.value for item in PipelineStageName}
    error = PipelineOperationError(PipelineOperationCode.STAGE_UNSUPPORTED)
    assert isinstance(error, ValueError) and str(error) == "stage_unsupported"
    with pytest.raises(TypeError):
        PipelineOperationError("stage_unsupported")  # type: ignore[arg-type]


def test_stage_run_reference_is_typed_by_stage_name() -> None:
    key = stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.TRAIN, D[1], ())
    with pytest.raises(TypeError, match="exact TrainingRunRef"):
        PipelineStage(PipelineStageName.TRAIN, StageState.SUCCEEDED, key, EVAL_RUN, (), (ADAPTER,))
    key = stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.EVALUATE, D[1], ())
    with pytest.raises(TypeError, match="exact EvaluationRunRef"):
        PipelineStage(PipelineStageName.EVALUATE, StageState.SUCCEEDED, key, TRAIN_RUN, (), (ADAPTER,))
    key = stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.LOSS, D[1], ())
    with pytest.raises(TypeError, match="exact None"):
        PipelineStage(PipelineStageName.LOSS, StageState.SUCCEEDED, key, TRAIN_RUN, (), ())
    with pytest.raises(ValueError, match="no run and no outputs"):
        PipelineStage(PipelineStageName.LOSS, StageState.PLANNED, key, None, (), (ADAPTER,))


# --- canonical documents and schemas ---------------------------------------------------


def test_record_matches_the_architecture_example_and_its_closed_schema() -> None:
    record = _record()
    document = record.to_dict()
    assert document == {
        "schema_version": "synaptic-pipeline-record/v1",
        "pipeline": {"pipeline_id": "pl-01J", "project_ref": "acme"},
        "spec_digest": D[1], "state": "partially_succeeded", "revision": 6,
        "stages": [
            {"name": "train", "state": "succeeded", "attempt_key": record.stages[0].attempt_key,
             "run": {"run_id": "modal-sft-01J", "project_ref": "acme"},
             "inputs": [], "outputs": [{"role": "adapter", "sha256": D[0], "size_bytes": 84}]},
            {"name": "evaluate", "state": "failed", "attempt_key": record.stages[1].attempt_key,
             "run": {"run_id": "ev-01J", "project_ref": "acme"},
             "inputs": [{"role": "adapter", "sha256": D[0], "size_bytes": 84}], "outputs": []},
        ],
    }
    jsonschema.Draft202012Validator.check_schema(RECORD_SCHEMA)
    jsonschema.validate(document, RECORD_SCHEMA)
    assert PipelineRecord.from_dict(json.loads(json.dumps(document))) == record
    assert record.stage(PipelineStageName.EVALUATE).run == EVAL_RUN
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, extra=1), RECORD_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, revision=0), RECORD_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, state="completed"), RECORD_SCHEMA)
    planned = dict(document["stages"][1], state="planned")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(document, stages=[document["stages"][0], planned]), RECORD_SCHEMA)
    with pytest.raises(ValueError, match="unknown fields: extra"):
        PipelineRecord.from_dict(dict(document, extra=1))


def test_record_invariants_are_enforced() -> None:
    with pytest.raises(ValueError, match="unsupported pipeline record schema version"):
        PipelineRecord("synaptic-pipeline-record/v2", PIPELINE, D[1], PipelineState.PLANNED, 1, _record().stages)
    with pytest.raises(ValueError, match="inconsistent with its stages"):
        _record(PipelineState.SUCCEEDED)
    with pytest.raises(ValueError, match="inconsistent with its stages"):
        _record(PipelineState.RUNNING)
    with pytest.raises(ValueError, match="does not derive from its inputs"):
        PipelineRecord("synaptic-pipeline-record/v1", PIPELINE, D[2], PipelineState.PARTIALLY_SUCCEEDED, 6, _record().stages)
    other = PipelineStage(
        PipelineStageName.TRAIN, StageState.SUCCEEDED,
        stage_attempt_key(PIPELINE.pipeline_id, PipelineStageName.TRAIN, D[1], ()),
        TrainingRunRef("r", "other-project"), (), (ADAPTER,),
    )
    with pytest.raises(ValueError, match="run project does not match"):
        PipelineRecord("synaptic-pipeline-record/v1", PIPELINE, D[1], PipelineState.SUCCEEDED, 1, (other,))
    with pytest.raises(ValueError, match="revision"):
        PipelineRecord("synaptic-pipeline-record/v1", PIPELINE, D[1], PipelineState.PARTIALLY_SUCCEEDED, 0, _record().stages)


def test_attempt_key_is_deterministic_over_ordered_inputs_and_changes_with_any_component() -> None:
    other = VerifiedArtifact("results", D[2], 10)
    key = stage_attempt_key("pl-1", PipelineStageName.EVALUATE, D[1], (ADAPTER, other))
    assert key == stage_attempt_key("pl-1", PipelineStageName.EVALUATE, D[1], (other, ADAPTER))
    assert len(key) == 64 and int(key, 16) >= 0
    assert stage_input_digest((ADAPTER, other)) == stage_input_digest((other, ADAPTER))
    assert stage_input_digest(()) != stage_input_digest((ADAPTER,))
    assert key != stage_attempt_key("pl-2", PipelineStageName.EVALUATE, D[1], (ADAPTER, other))
    assert key != stage_attempt_key("pl-1", PipelineStageName.TRAIN, D[1], (ADAPTER, other))
    assert key != stage_attempt_key("pl-1", PipelineStageName.EVALUATE, D[2], (ADAPTER, other))
    assert key != stage_attempt_key("pl-1", PipelineStageName.EVALUATE, D[1], (ADAPTER,))
    changed = VerifiedArtifact("adapter", D[3], 84)
    assert key != stage_attempt_key("pl-1", PipelineStageName.EVALUATE, D[1], (changed, other))
    with pytest.raises(TypeError):
        stage_attempt_key("pl-1", "evaluate", D[1], ())  # type: ignore[arg-type]


def test_request_admits_only_train_and_evaluate_and_binds_specs_to_stages() -> None:
    request = _request()
    assert request.unadmitted_stages == ()
    assert request.spec_digest == PipelineRequest.from_dict(request.to_dict()).spec_digest
    plan = PipelinePlan("synaptic-pipeline-plan/v1", request)
    assert plan.spec_digest == request.spec_digest
    assert PipelinePlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    loss = _request(stages=(PipelineStageName.TRAIN, PipelineStageName.EVALUATE, PipelineStageName.LOSS))
    assert loss.unadmitted_stages == (PipelineStageName.LOSS,)
    with pytest.raises(ValueError, match="admits only train and evaluate"):
        PipelinePlan("synaptic-pipeline-plan/v1", loss)
    with pytest.raises(ValueError, match="evaluate must follow train"):
        _request(stages=(PipelineStageName.EVALUATE, PipelineStageName.TRAIN))
    with pytest.raises(ValueError, match="evaluate must follow train"):
        _request(stages=(PipelineStageName.EVALUATE,), train=None)
    with pytest.raises(ValueError, match="present exactly when"):
        _request(stages=(PipelineStageName.TRAIN,))
    with pytest.raises(ValueError, match="present exactly when"):
        _request(evaluate=None)
    with pytest.raises(ValueError, match="unique"):
        _request(stages=(PipelineStageName.TRAIN, PipelineStageName.TRAIN), evaluate=None)
    with pytest.raises(ValueError, match="at least 1"):
        _request(stages=(), train=None, evaluate=None)
    # The spec digest covers every stage spec, so a changed scenario list changes the digest.
    other = _request(evaluate=PipelineEvaluateSpec("local", ("beta.yaml",), "fake-model"))
    assert other.spec_digest != request.spec_digest


def test_paging_rule_matches_the_runs_facade() -> None:
    request = PipelineListRequest("acme")
    assert request.limit == 100 and request.cursor is None
    with pytest.raises(ValueError):
        PipelineListRequest("acme", limit=0)
    with pytest.raises(ValueError):
        PipelineListRequest("acme", limit=101)
    page = PipelinePage(request, (), None, False)
    assert page.truncated is False
    with pytest.raises(ValueError):
        PipelinePage(request, (), "cursor", False)
    with pytest.raises(ValueError):
        PipelinePage(request, (), None, True)


# --- facade discipline --------------------------------------------------------------


class Recorder:
    """``PipelinesOperations`` that records what it was handed and answers from a script."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def plan(self, request):
        self.calls.append(("plan", (request,)))
        return self.answers["plan"]

    def start(self, plan):
        self.calls.append(("start", (plan,)))
        return self.answers["start"]

    def show(self, pipeline):
        self.calls.append(("show", (pipeline,)))
        return self.answers["show"]

    def resume(self, pipeline):
        self.calls.append(("resume", (pipeline,)))
        return self.answers["resume"]

    def cancel(self, pipeline, reason):
        self.calls.append(("cancel", (pipeline, reason)))
        return self.answers["cancel"]

    def list(self, request):
        self.calls.append(("list", (request,)))
        return self.answers["list"]

    def observations(self, request):
        self.calls.append(("observations", (request,)))
        return self.answers["observations"]


def test_plan_refuses_unadmitted_stages_before_any_callback() -> None:
    operations = Recorder({})
    api = PipelinesAPI(operations)
    for extra in (PipelineStageName.LOSS, PipelineStageName.ANALYSIS, PipelineStageName.RECOMMENDATION):
        request = _request(stages=(PipelineStageName.TRAIN, PipelineStageName.EVALUATE, extra))
        with pytest.raises(PipelineOperationError) as refused:
            api.plan(request)
        assert refused.value.code is PipelineOperationCode.STAGE_UNSUPPORTED
    assert operations.calls == []


def test_pipelines_facade_reconstructs_and_binds_every_callback_result() -> None:
    request = _request()
    plan = PipelinePlan("synaptic-pipeline-plan/v1", request)
    record = _record()
    stream = ObservationStreamRef(ObservationFamily.PIPELINE, "acme", "pl-01J")
    operations = Recorder({
        "plan": plan, "start": PipelineStart(PIPELINE, True), "show": record, "resume": record,
        "cancel": record, "list": PipelinePage(PipelineListRequest("acme"), (record,), None, False),
        "observations": v1.ObservationPage(ObservationsRequest(stream), (), None, False),
    })
    api = PipelinesAPI(operations)
    assert api.plan(request) == plan and api.plan(request) is not plan
    assert api.start(plan) == PipelineStart(PIPELINE, True)
    assert api.show(PIPELINE) == record and api.show(PIPELINE) is not record
    assert api.resume(PIPELINE) == record
    assert api.cancel(PIPELINE, "operator") == record
    assert api.list(PipelineListRequest("acme")).records == (record,)
    assert api.observations(ObservationsRequest(stream)).records == ()
    presented = [arguments[0] for _, arguments in operations.calls]
    assert presented[0] == request and presented[0] is not request
    assert presented[3] == PIPELINE and presented[3] is not PIPELINE
    assert operations.calls[6][1][1] == "operator"

    with pytest.raises(ValueError, match="does not bind"):
        PipelinesAPI(Recorder({"show": record})).show(PipelineRef("pl-other", "acme"))
    with pytest.raises(ValueError, match="does not bind"):
        PipelinesAPI(Recorder({"plan": plan})).plan(_request(request_id="req-2"))
    with pytest.raises(TypeError, match="exact PipelineRecord"):
        PipelinesAPI(Recorder({"show": object()})).show(PIPELINE)
    with pytest.raises(TypeError, match="exact PipelineRef"):
        api.show(TRAIN_RUN)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        api.cancel(PIPELINE, "")


def test_observations_verb_admits_only_pipeline_streams_before_any_callback() -> None:
    operations = Recorder({})
    api = PipelinesAPI(operations)
    stream = ObservationStreamRef(ObservationFamily.TRAINING, "acme", "run-1")
    with pytest.raises(ValueError, match="pipeline stream"):
        api.observations(ObservationsRequest(stream))
    assert operations.calls == []


def test_pipelines_facade_rejects_presented_input_mutation_on_return_and_raise() -> None:
    class Mutating:
        def __init__(self, raises: bool) -> None:
            self.raises = raises

        def show(self, pipeline):
            object.__setattr__(pipeline, "project_ref", "evil")
            if self.raises:
                raise RuntimeError("boom")
            return _record()

    for raises in (False, True):
        with pytest.raises(ValueError, match="changed during callback"):
            PipelinesAPI(Mutating(raises)).show(PIPELINE)  # type: ignore[arg-type]


def test_observation_kinds_for_pipelines_are_the_two_stage_kinds() -> None:
    kinds = {kind for kind in ObservationKind if kind.value.startswith("pipeline_")}
    assert kinds == {ObservationKind.PIPELINE_STAGE_STARTED, ObservationKind.PIPELINE_STAGE_COMPLETED}


# --- host wiring and import gate ----------------------------------------------------------


def test_api_host_wraps_pipelines_operations_in_the_public_facade() -> None:
    class Clock:
        def now(self) -> str:
            return "2026-08-30T12:00:00Z"

    ports = HostPorts(
        training=object(), runs=object(), artifacts=None, evaluation=None,
        chat=None, data=None, pipelines=Recorder({}), clock=Clock(),
    )
    host = APIHost(ports)
    assert type(host.pipelines) is PipelinesAPI
    empty = APIHost(HostPorts(
        training=object(), runs=object(), artifacts=None, evaluation=None,
        chat=None, data=None, pipelines=None, clock=Clock(),
    ))
    with pytest.raises(RuntimeError, match="did not compose the 'pipelines' family"):
        empty.pipelines


def test_pipelines_facade_imports_nothing_from_the_engine_packages() -> None:
    source = (ROOT / "synaptic_tuner/api/v1/pipelines_facade.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(source)
    roots = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    roots.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert roots.isdisjoint({"tuner", "Evaluator", "SynthChat", "shared", "modal", "sqlite3"})
    code = """
import sys
import synaptic_tuner.api.v1
from synaptic_tuner.api.v1 import PipelinesAPI, PipelineRecord
for prefix in ('huggingface_hub', 'modal', 'runpod', 'sqlite3', 'tuner'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, cwd=ROOT)
    assert result.returncode == 0, result.stderr
