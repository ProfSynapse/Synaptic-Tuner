"""Reference ``PipelinesAPI`` over the fake provider family and local evaluation (api-facade slice 10).

Composes ``compose_reference_host`` with ``ReferencePipelinePortsV1`` over the
public host ports (in-memory record and stream stores, a fixed clock, a
scripted grant authority), the fake provider family (slice 4) and the
reference evaluation family with a scripted local backend (slice 6), then
drives ``APIHost.pipelines``: plan, start, show, resume, cancel, list and
observations. The train child is a real coordinator run on the fake provider;
the evaluate child is a real reference evaluation run. Every child is
referenced, never owned.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.evaluation_facade import EvaluationRunRef, EvaluationRunState
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.observations import (
    ObservationFamily, ObservationKind, ObservationsRequest, ObservationStreamRef,
)
from synaptic_tuner.api.v1.pipelines_facade import (
    PipelineEvaluateSpec, PipelineListRequest, PipelineOperationCode, PipelineOperationError,
    PipelineRef, PipelineRequest, PipelineStageName, PipelineState, PipelineTrainSpec,
    PipelinesAPI, StageState, stage_attempt_key,
)
from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest, TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.reference import (
    DirectoryEvaluationArtifactSinkV1, EvaluationBackendRegistryV1, ProviderFamilyV1,
    ReferenceEvaluationPortsV1, ReferenceHostPortsV1, ReferencePipelinePortsV1,
    ReferenceRequestPortsV1, compose_reference_authority, compose_reference_host,
    compose_reference_pipelines,
)
from synaptic_tuner.api.v1.reference.pipelines import (
    PIPELINE_EVENT_BUDGET_BASE, PIPELINE_EVENT_BUDGET_PER_STAGE, pipeline_event_budget,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import RunListRequest, RunLogEntry, RunLogLevel, RunOutcome
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingPreflight, TrainingRequest,
)
from Evaluator.protocols import BackendResponse
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1, ProviderRunPhaseV1
from tuner.execution.fake_provider_v1 import (
    FakeArtifactV1, FakeEffectResultV1, FakeProviderConfigV1, FakeProviderFamilyV1,
)
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1


ROOT = Path(__file__).resolve().parents[2]
D = tuple(character * 64 for character in "123456789abcdef")
SECRET = SecretRef("env", "SYNAPTIC_REFERENCE_AUTHORITY")
PROVIDER = ProviderRef("fake", "profile")
RUN = TrainingRunRef("run-1", "project")
PROJECT = "project"
NOW = "2026-08-27T00:00:00Z"
ADAPTER_BYTES = b"adapter-data"
ADAPTER = VerifiedArtifact("adapter", hashlib.sha256(ADAPTER_BYTES).hexdigest(), len(ADAPTER_BYTES))
RESPONSES = {"Say hello": "hello there", "Summarise the document": "Summary: fine"}


class Clock:
    def now(self) -> str:
        return NOW

    def now_epoch(self) -> int:
        return 150


class Secrets:
    def resolve(self, reference: SecretRef) -> str:
        assert reference == SECRET
        return "reference-test-master-secret-value-" + "x" * 32


class HostGrants:
    """``GrantAuthorityPort`` whose ``bind`` scripts the fake provider per effect kind."""

    def __init__(self, family_holder):
        self._holder = family_holder
        self.bound = []

    def authorize(self, requirements):
        return ExecutionGrant("host-grant-1")

    def bind(self, grant, *, operation, requirements):
        kind = operation.operation.effect.kind.value
        self.bound.append(kind)
        self._holder.family.register_command(
            operation, dispatch=(FakeEffectResultV1(ObservationDisposition.FOUND),), reconciliation=(),
        )
        return object()


class Holder:
    family = None


class Loader:
    def load(self, canonical_json):
        return TrainingRequest("request", PROJECT, canonical_json)


class Resolver:
    def resolve(self, request):
        return ResolvedTrainingRequest(
            "synaptic-resolved-training-request/v1", request.request_id, request.project_ref, *D[:5],
        )


class Identity:
    def for_plan(self, plan):
        return RUN


class ScriptedClient:
    def __init__(self, backend, responses):
        self._backend = backend
        self._responses = responses

    def chat(self, messages):
        question = messages[-1]["content"]
        self._backend.chats.append(question)
        message = self._responses[question]
        raw = {"choices": [{"message": {"role": "assistant", "content": message}}]}
        return BackendResponse(message=message, raw=raw, latency_s=0.01)


class ScriptedBackend:
    """Local ``EvaluationBackendPort``: no spend effect, scripted answers, records the model."""

    def __init__(self, responses=RESPONSES):
        self._responses = dict(responses)
        self.opened = []
        self.chats = []

    def requirement(self):
        return AuthorizationRequirement("evaluation.start", False)

    def open(self, model):
        self.opened.append(model)
        return ScriptedClient(self, self._responses)


class Pacing:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def _write_config(root: Path) -> Path:
    scenarios = root / "config" / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "alpha.yaml").write_text(
        "name: alpha\n"
        "tests:\n"
        "  - id: greet\n"
        "    question: Say hello\n"
        "    tags: [smoke]\n"
        "    correct:\n"
        "      any:\n"
        "        - name: hello\n"
        "          assertions:\n"
        "            - {type: jsonpath_regex, path: '$.content', pattern: 'hello'}\n"
        "  - id: summary\n"
        "    question: Summarise the document\n"
        "    tags: [smoke]\n"
        "    correct:\n"
        "      any:\n"
        "        - name: summary\n"
        "          assertions:\n"
        "            - {type: jsonpath_regex, path: '$.content', pattern: '^Summary:'}\n",
        encoding="utf-8",
    )
    (root / "config" / "eval_run.yaml").write_text(
        "run:\n  name: Reference\n  scenarios: [alpha.yaml]\npresets: {}\n", encoding="utf-8",
    )
    return root / "config"


def _host(
    tmp_path: Path, *, backends, records=None, streams=None, family=None,
    maximum_child_polls=10_000, pacing=None, pipelines=True,
):
    """Compose a reference host with the evaluation and pipelines families over the fake provider.

    ``records``/``streams`` reuse existing host stores (a recomposed host);
    ``family`` reuses an existing fake provider, which stands for the one
    external provider both hosts talk to.
    """
    records = records if records is not None else InMemoryDurableRecordStoreV1()
    streams = streams if streams is not None else InMemoryDurableStreamStoreV1()
    config_root = tmp_path / "config"
    if not config_root.exists():
        _write_config(tmp_path)
    sink_root = tmp_path / "sink"
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", PROVIDER.provider_id, "Fake provider", "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    basis = TrainingPlanBasisV1("synaptic-training-plan-basis/v1", "request", PROJECT, *D[:5])
    plan_context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", PROVIDER, basis.basis_digest,
        descriptor.descriptor_digest, D[5],
    )
    executor = ExecutorDescriptorV1(PROVIDER.provider_id, "executor", "1.0.0")
    adapter = AdapterDescriptorV1(PROVIDER.provider_id, "adapter", "1.0.0")
    scope = ExecutionScopeV1("account", "namespace")
    binding = ProviderExecutionBindingV1(
        PROVIDER, descriptor.descriptor_digest, plan_context.profile_digest, scope, executor,
        adapter.digest, D[7], D[8], D[9],
    )
    config = FakeProviderConfigV1(
        PROVIDER, descriptor, plan_context.profile_digest, scope.account_ref, scope.namespace_ref,
        executor, adapter, (), (ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED),
        (RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "progress", "ok", 2),),
        (FakeArtifactV1(ADAPTER, ADAPTER_BYTES),),
    )
    holder = Holder()
    grants = HostGrants(holder)
    ports = ReferenceHostPortsV1(records, streams, Clock(), grants, Secrets())
    authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets, authority_secret=SECRET)
    fake = family if family is not None else FakeProviderFamilyV1(
        config, evidence_key=b"e" * 32,
        foundation_authenticator=authority.foundation_authenticator,
        assessment_authenticator=authority.assessment_authority,
    )
    holder.family = fake

    class Planning:
        def describe(self, provider):
            return descriptor

        def context(self, resolved, provider):
            return plan_context

        def preflight(self, plan):
            return TrainingPreflight(
                plan.plan_fingerprint, True, "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
                (AuthorizationRequirement("training.start", True, 100, "USD"),),
            )

    class Preparation:
        def resolve(self, provider, plan_context):
            return binding

        def prepare(self, plan, run, binding):
            return CanonicalPreparationV2.build(
                provider=binding.provider, scope=binding.scope, project_ref=run.project_ref,
                run_id=run.run_id, plan_fingerprint=plan.plan_fingerprint,
                source_digest=plan.basis.source_digest, workload_digest=plan.basis.workload_digest,
                runtime_digest=plan.basis.runtime_digest, resource_digest=binding.resource_digest,
                artifact_contract_digest=plan.basis.artifact_policy_digest,
                quote_digest=binding.quote_digest,
                secret_requirements_digest=binding.secret_requirements_digest,
                execution_binding_digest=binding.binding_digest,
            )

        def payload(self, preparation, kind):
            return CanonicalProviderPayloadV1.build(
                PROVIDER.provider_id, f"{kind.value}-payload/v2", preparation.workload_digest,
            )

    provider_family = ProviderFamilyV1(
        descriptor, Planning(), Preparation(), fake.reader, fake.executor_resolver,
        fake.reconciliation_resolver, fake.evidence_authority, fake.artifact_verifier,
        fake.evidence_authority, fake.evidence_authority,
    )
    evaluation = None
    if backends is not None:
        evaluation = ReferenceEvaluationPortsV1(
            config_root, EvaluationBackendRegistryV1(backends),
            DirectoryEvaluationArtifactSinkV1(sink_root), None,
        )
    composition = compose_reference_host(
        family=provider_family, ports=ports,
        requests=ReferenceRequestPortsV1(Loader(), Resolver(), Identity()),
        authority=authority, evaluation=evaluation,
        pipelines=ReferencePipelinePortsV1(maximum_child_polls, pacing) if pipelines else None,
    )
    return composition, composition.api(), fake, grants, records, streams


def _request(*, stages=(PipelineStageName.TRAIN, PipelineStageName.EVALUATE), request_id="pipe-1", scenario="alpha.yaml"):
    return PipelineRequest(
        request_id, PROJECT, tuple(stages),
        PipelineTrainSpec('{"method":"sft"}', PROVIDER) if PipelineStageName.TRAIN in stages else None,
        PipelineEvaluateSpec("local", (scenario,), "fake-model") if PipelineStageName.EVALUATE in stages else None,
    )


def _start(api, **changes):
    plan = api.pipelines.plan(_request(**changes))
    started = api.pipelines.start(plan)
    assert started.accepted is True and started.pipeline.project_ref == PROJECT
    return started.pipeline


def _stores(records):
    found = {}
    for partition in StoragePartition:
        page = records.list_page(partition=partition.value, prefix="", after_key=None, limit=100)
        if page.records:
            found[partition.value] = tuple(record.key for record in page.records)
    return found


def _events(records, streams):
    """Every durable lifecycle event of the single pipeline in the ``pipeline`` partition."""
    keys = [key for key in _stores(records)[StoragePartition.PIPELINE.value] if key.startswith("pipeline/")]
    (key,) = keys
    page = streams.read_page(
        partition=StoragePartition.PIPELINE.value, stream_key=key, after_sequence=None, limit=100,
    )
    assert page.truncated is False
    return key, [json.loads(entry.canonical.decode("utf-8")) for entry in page.entries]


def _observations(api, pipeline):
    stream = ObservationStreamRef(ObservationFamily.PIPELINE, pipeline.project_ref, pipeline.pipeline_id)
    page = api.pipelines.observations(ObservationsRequest(stream, None, 200))
    return [(record.kind, record.payload.stage, getattr(record.payload, "outcome", None)) for record in page.records]


def _no_legacy_dirs(root: Path) -> None:
    found = {path.name for path in root.rglob("*") if path.name in {".tracking", ".synaptic"}}
    assert found == set(), found


# --- tests -----------------------------------------------------------------------------


def test_train_then_evaluate_converges_with_referenced_children_and_artifact_handoff(tmp_path) -> None:
    backend = ScriptedBackend()
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": backend})
    pipeline = _start(api)
    record = api.pipelines.show(pipeline)
    assert record.state is PipelineState.SUCCEEDED and record.revision >= 6
    train, evaluate = record.stages
    assert (train.state, evaluate.state) == (StageState.SUCCEEDED, StageState.SUCCEEDED)
    # The train child is the host's own coordinator run, referenced by identity.
    assert train.run == RUN and train.inputs == () and train.outputs == (ADAPTER,)
    assert api.runs.show(RUN).state is TrainingRunState.SUCCEEDED
    assert api.runs.show(RUN).artifacts == (ADAPTER,)
    assert grants.bound == ["stage", "submit"]
    # Handoff is by VerifiedArtifact: the evaluate inputs are the train outputs and the
    # evaluated model revision is the adapter's digest, not a path.
    assert type(evaluate.run) is EvaluationRunRef and evaluate.inputs == (ADAPTER,)
    assert [model.model_revision for model in backend.opened] == [ADAPTER.sha256]
    assert len(evaluate.outputs) == 2 and api.evaluation.show(evaluate.run).artifacts == evaluate.outputs
    assert api.evaluation.show(evaluate.run).state is EvaluationRunState.SUCCEEDED
    assert train.attempt_key == stage_attempt_key(pipeline.pipeline_id, PipelineStageName.TRAIN, record.spec_digest, ())
    assert evaluate.attempt_key == stage_attempt_key(pipeline.pipeline_id, PipelineStageName.EVALUATE, record.spec_digest, (ADAPTER,))

    # Idempotent start: the same plan names the same pipeline and drives nothing again.
    again = api.pipelines.start(api.pipelines.plan(_request()))
    assert again.pipeline == pipeline and api.pipelines.show(pipeline) == record
    assert grants.bound == ["stage", "submit"]

    # Public observations: two stage lifecycles in order.
    assert _observations(api, pipeline) == [
        (ObservationKind.PIPELINE_STAGE_STARTED, "train", None),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "train", "succeeded"),
        (ObservationKind.PIPELINE_STAGE_STARTED, "evaluate", None),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate", "succeeded"),
    ]
    # Durable layout: plan and head under the pipeline partition, children in their own partitions.
    stored = _stores(records)
    assert sorted(key.split("/")[0] for key in stored[StoragePartition.PIPELINE.value]) == ["pipeline", "plan"]
    assert StoragePartition.WORKFLOW.value in stored and StoragePartition.EVALUATION.value in stored
    key, events = _events(records, streams)
    assert [event["event"]["kind"] for event in events] == [
        "created", "started", "stage_started", "stage_finished", "stage_started", "stage_finished",
    ]
    assert len(events) <= pipeline_event_budget(2) == PIPELINE_EVENT_BUDGET_BASE + 2 * PIPELINE_EVENT_BUDGET_PER_STAGE
    head = records.read(partition=StoragePartition.PIPELINE.value, key=key)
    assert head.revision == record.revision == len(events)
    page = api.pipelines.list(PipelineListRequest(PROJECT))
    assert page.records == (record,) and page.next_cursor is None
    _no_legacy_dirs(tmp_path)


def test_failed_evaluate_leaves_the_train_child_succeeded_and_the_pipeline_partially_succeeded(tmp_path) -> None:
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": ScriptedBackend(responses={})})
    pipeline = _start(api)
    record = api.pipelines.show(pipeline)
    assert record.state is PipelineState.PARTIALLY_SUCCEEDED
    train, evaluate = record.stages
    assert train.state is StageState.SUCCEEDED and train.run == RUN and train.outputs == (ADAPTER,)
    assert evaluate.state is StageState.FAILED and type(evaluate.run) is EvaluationRunRef
    assert evaluate.inputs == (ADAPTER,) and evaluate.outputs == ()
    assert api.runs.show(RUN).state is TrainingRunState.SUCCEEDED
    assert api.evaluation.show(evaluate.run).state is EvaluationRunState.FAILED
    assert _observations(api, pipeline)[-1] == (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate", "failed")
    public = json.dumps(record.to_dict())
    assert "KeyError" not in public and "Say hello" not in public
    with pytest.raises(PipelineOperationError) as refused:
        api.pipelines.cancel(pipeline, "too late")
    assert refused.value.code is PipelineOperationCode.CANCEL_INELIGIBLE


def test_a_spec_naming_an_unadmitted_stage_is_refused_at_plan_time(tmp_path) -> None:
    composition, api, _, grants, _, _ = _host(tmp_path, backends={"local": ScriptedBackend()})
    request = _request(stages=(PipelineStageName.TRAIN, PipelineStageName.EVALUATE, PipelineStageName.LOSS))
    with pytest.raises(PipelineOperationError) as refused:
        composition.pipelines.plan(request)
    assert refused.value.code is PipelineOperationCode.STAGE_UNSUPPORTED
    with pytest.raises(PipelineOperationError) as refused:
        api.pipelines.plan(request)
    assert refused.value.code is PipelineOperationCode.STAGE_UNSUPPORTED
    assert grants.bound == []
    # A host without an evaluation family cannot admit an evaluate stage either.
    _, bare, _, _, _, _ = _host(tmp_path / "bare", backends=None)
    with pytest.raises(PipelineOperationError) as refused:
        bare.pipelines.plan(_request())
    assert refused.value.code is PipelineOperationCode.STAGE_UNSUPPORTED
    # An unknown backend or scenario is a spec problem, refused before anything durable.
    with pytest.raises(PipelineOperationError) as refused:
        api.pipelines.plan(_request(scenario="missing.yaml"))
    assert refused.value.code is PipelineOperationCode.SPEC_INVALID
    with pytest.raises(PipelineOperationError) as refused:
        api.pipelines.show(PipelineRef("pl-missing", PROJECT))
    assert refused.value.code is PipelineOperationCode.PIPELINE_MISSING


def test_interrupted_train_then_resume_converges_without_a_duplicate_child_run(tmp_path) -> None:
    """The gate: a pipeline interrupted mid-train re-attaches to its child on resume."""
    pacing = Pacing()
    composition, api, fake, grants, records, streams = _host(
        tmp_path, backends={"local": ScriptedBackend()}, maximum_child_polls=1, pacing=pacing,
    )
    pipeline = _start(api)
    interrupted = api.pipelines.show(pipeline)
    assert interrupted.state is PipelineState.RUNNING
    train, evaluate = interrupted.stages
    assert train.state is StageState.RUNNING and train.run == RUN and evaluate.state is StageState.PLANNED
    assert grants.bound == ["stage", "submit"] and pacing.waits == 0
    assert api.runs.show(RUN).state is TrainingRunState.RUNNING
    observed = fake.trace.snapshot().count(("reader", "observe"))

    # A recomposed host over the same stores and the same provider resumes the pipeline.
    _, again, _, grants_again, _, _ = _host(
        tmp_path, backends={"local": ScriptedBackend()}, records=records, streams=streams, family=fake,
    )
    record = again.pipelines.resume(pipeline)
    assert record.state is PipelineState.SUCCEEDED
    assert record.stages[0].run == RUN and record.stages[0].outputs == (ADAPTER,)
    assert record.stages[1].state is StageState.SUCCEEDED and record.stages[1].inputs == (ADAPTER,)
    # No second training start: no new grant was bound and the project still has exactly one run.
    assert grants_again.bound == [] and grants.bound == ["stage", "submit"]
    assert [item.run for item in again.runs.list(RunListRequest(PROJECT)).outcomes] == [RUN]
    assert fake.trace.snapshot().count(("reader", "observe")) == observed + 1
    assert _observations(again, pipeline) == [
        (ObservationKind.PIPELINE_STAGE_STARTED, "train", None),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "train", "succeeded"),
        (ObservationKind.PIPELINE_STAGE_STARTED, "evaluate", None),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate", "succeeded"),
    ]
    _, events = _events(records, streams)
    assert [event["event"]["kind"] for event in events] == [
        "created", "started", "stage_started", "stage_finished", "stage_started", "stage_finished",
    ]
    # A succeeded pipeline has nothing to resume.
    with pytest.raises(PipelineOperationError) as refused:
        again.pipelines.resume(pipeline)
    assert refused.value.code is PipelineOperationCode.RESUME_INELIGIBLE
    _no_legacy_dirs(tmp_path)


def test_resume_skips_the_matching_train_stage_and_retries_the_failed_evaluate_with_a_fresh_child(tmp_path) -> None:
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": ScriptedBackend(responses={})})
    pipeline = _start(api)
    failed = api.pipelines.show(pipeline)
    assert failed.state is PipelineState.PARTIALLY_SUCCEEDED
    first_evaluation = failed.stages[1].run

    # Same stores, same provider, a backend that now answers: resume retries only the evaluate stage.
    _, again, _, grants_again, _, _ = _host(
        tmp_path, backends={"local": ScriptedBackend()}, records=records, streams=streams, family=fake,
    )
    record = again.pipelines.resume(pipeline)
    assert record.state is PipelineState.SUCCEEDED
    assert record.stages[0] == failed.stages[0]  # same run, same attempt_key, same outputs: skipped
    assert record.stages[1].attempt_key == failed.stages[1].attempt_key  # same inputs, same key
    assert record.stages[1].run != first_evaluation and record.stages[1].state is StageState.SUCCEEDED
    assert grants_again.bound == [] and [item.run for item in again.runs.list(RunListRequest(PROJECT)).outcomes] == [RUN]
    # Both evaluation children still exist; the pipeline references the second and never touched the first.
    assert again.evaluation.show(first_evaluation).state is EvaluationRunState.FAILED
    _, events = _events(records, streams)
    assert [event["event"]["kind"] for event in events][6:] == ["resumed", "stage_started", "stage_finished"]
    assert len(events) == 9 <= pipeline_event_budget(2)
    assert _observations(again, pipeline)[4:] == [
        (ObservationKind.PIPELINE_STAGE_STARTED, "evaluate", None),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate", "succeeded"),
    ]


class DriftingRuns:
    """``RunsOperations`` over the real coordinator whose train child now reports other artifacts."""

    def __init__(self, inner, artifacts):
        self._inner = inner
        self._artifacts = artifacts

    def _drift(self, outcome):
        if outcome.artifacts:
            return RunOutcome(outcome.schema_version, outcome.run, outcome.state, self._artifacts, outcome.diagnostic_code)
        return outcome

    def list(self, request):
        return self._inner.list(request)

    def show(self, run):
        return self._drift(self._inner.show(run))

    def outcome(self, run):
        return self._drift(self._inner.outcome(run))

    def logs(self, request):
        return self._inner.logs(request)

    def cancel(self, run, reason):
        return self._inner.cancel(run, reason)

    def reconcile(self, run):
        return self._inner.reconcile(run)

    def verify(self, run):
        return self._inner.verify(run)

    def reverify(self, run):
        return self._inner.reverify(run)

    def artifacts(self, request):
        return self._inner.artifacts(request)


def _recomposed_pipelines(tmp_path, composition, records, streams, fake, *, artifacts, backend):
    """Pipelines over the same stores, the same training family and a train child whose artifacts drifted."""
    evaluation_host, _, _, _, _, _ = _host(
        tmp_path, backends={"local": backend}, records=records, streams=streams, family=fake,
    )
    operations = compose_reference_pipelines(
        records=records, streams=streams, clock=Clock(), training=composition.training,
        runs=DriftingRuns(composition.runs, artifacts), evaluation=evaluation_host.evaluation,
        pipelines=ReferencePipelinePortsV1(),
    )
    return PipelinesAPI(operations)


def test_resume_reruns_a_downstream_stage_whose_input_digest_changed(tmp_path) -> None:
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": ScriptedBackend(responses={})})
    pipeline = _start(api)
    failed = api.pipelines.show(pipeline)
    assert failed.stages[1].state is StageState.FAILED and failed.stages[1].inputs == (ADAPTER,)
    drifted = VerifiedArtifact("adapter", hashlib.sha256(b"other").hexdigest(), 5)
    backend = ScriptedBackend()
    pipelines = _recomposed_pipelines(tmp_path, composition, records, streams, fake, artifacts=(drifted,), backend=backend)
    record = pipelines.resume(pipeline)
    assert record.state is PipelineState.SUCCEEDED
    train, evaluate = record.stages
    # The train stage keeps its key and its child; its outputs now name the child's live artifacts.
    assert train.run == RUN and train.attempt_key == failed.stages[0].attempt_key and train.outputs == (drifted,)
    # The evaluate stage re-ran on the new inputs under a new key against the new model revision.
    assert evaluate.inputs == (drifted,) and evaluate.attempt_key != failed.stages[1].attempt_key
    assert evaluate.attempt_key == stage_attempt_key(pipeline.pipeline_id, PipelineStageName.EVALUATE, record.spec_digest, (drifted,))
    assert [model.model_revision for model in backend.opened] == [drifted.sha256]
    assert grants.bound == ["stage", "submit"]


def test_resume_refuses_with_stage_digest_mismatch_when_a_succeeded_stage_was_built_on_stale_inputs(tmp_path) -> None:
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": ScriptedBackend()})
    pipeline = _start(api)
    done = api.pipelines.show(pipeline)
    assert done.state is PipelineState.SUCCEEDED and done.stages[1].inputs == (ADAPTER,)
    # Intact children: a succeeded pipeline simply has nothing to resume.
    intact = _recomposed_pipelines(tmp_path, composition, records, streams, fake, artifacts=(ADAPTER,), backend=ScriptedBackend())
    with pytest.raises(PipelineOperationError) as refused:
        intact.resume(pipeline)
    assert refused.value.code is PipelineOperationCode.RESUME_INELIGIBLE
    # The train child now reports another adapter: the evaluate stage's recorded input digest no
    # longer matches the upstream output, and the recorded success is refused rather than rewritten.
    drifted = VerifiedArtifact("adapter", hashlib.sha256(b"other").hexdigest(), 5)
    stale = _recomposed_pipelines(tmp_path, composition, records, streams, fake, artifacts=(drifted,), backend=ScriptedBackend())
    with pytest.raises(PipelineOperationError) as refused:
        stale.resume(pipeline)
    assert refused.value.code is PipelineOperationCode.STAGE_DIGEST_MISMATCH
    assert stale.show(pipeline) == done  # nothing was rewritten
    assert grants.bound == ["stage", "submit"]


def test_cancel_requests_cancel_on_the_in_flight_child_only_and_never_touches_completed_children(tmp_path) -> None:
    composition, api, fake, grants, records, streams = _host(
        tmp_path, backends={"local": ScriptedBackend()}, maximum_child_polls=1,
    )
    pipeline = _start(api)
    assert api.pipelines.show(pipeline).state is PipelineState.RUNNING
    before = fake.trace.snapshot().count(("executor", "cancel"))
    record = api.pipelines.cancel(pipeline, "operator")
    assert record.state is PipelineState.CANCEL_REQUESTED
    train, evaluate = record.stages
    assert train.state is StageState.RUNNING and train.run == RUN
    assert evaluate.state is StageState.CANCELLED and evaluate.run is None and evaluate.outputs == ()
    assert fake.trace.snapshot().count(("executor", "cancel")) == before + 1
    assert api.runs.show(RUN).state is TrainingRunState.CANCEL_REQUESTED
    assert grants.bound == ["stage", "submit", "cancel"]
    # Cancel is idempotent: nothing new reaches the child.
    assert api.pipelines.cancel(pipeline, "operator") == record
    assert fake.trace.snapshot().count(("executor", "cancel")) == before + 1
    assert _observations(api, pipeline)[-1] == (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate", "cancelled")
    _, events = _events(records, streams)
    assert [event["event"]["kind"] for event in events] == ["created", "started", "stage_started", "cancel_requested"]
    assert len(events) <= pipeline_event_budget(2)
    # The completed pipeline elsewhere is never touched by cancel.
    _, done, fake_done, _, _, _ = _host(tmp_path / "done", backends={"local": ScriptedBackend()})
    finished = _start(done)
    with pytest.raises(PipelineOperationError) as refused:
        done.pipelines.cancel(finished, "operator")
    assert refused.value.code is PipelineOperationCode.CANCEL_INELIGIBLE
    assert fake_done.trace.snapshot().count(("executor", "cancel")) == 0
    assert done.runs.show(RUN).state is TrainingRunState.SUCCEEDED


def test_budget_is_asserted_and_the_pipeline_writes_no_legacy_directories(tmp_path) -> None:
    assert pipeline_event_budget(1) == 14 and pipeline_event_budget(2) == 20
    composition, api, fake, grants, records, streams = _host(tmp_path, backends={"local": ScriptedBackend()})
    pipeline = _start(api, stages=(PipelineStageName.TRAIN,), request_id="train-only")
    record = api.pipelines.show(pipeline)
    assert record.state is PipelineState.SUCCEEDED and len(record.stages) == 1
    _, events = _events(records, streams)
    assert len(events) == 4 <= pipeline_event_budget(1)
    for event in events:
        assert set(event) == {
            "schema_version", "revision", "previous_sequence", "previous_chain_digest",
            "event", "record_digest", "chain_digest",
        }
        assert set(event["event"]) == {"kind", "occurred_at", "state"}
    # Nothing under the project root but the injected config (and the sink once an evaluation runs).
    assert set(path.name for path in tmp_path.iterdir()) <= {"config", "sink"}
    _no_legacy_dirs(tmp_path)
    assert not any(name.startswith(".") for name in os.listdir(tmp_path))


def test_reference_pipelines_module_imports_no_persistence_paths_or_project_context() -> None:
    source = (ROOT / "synaptic_tuner/api/v1/reference/pipelines.py").read_text(encoding="utf-8")
    for forbidden in (".tracking", ".synaptic", "ProjectContext", "persistence", "os.environ", "getenv", "print(", "sys.exit", "load_dotenv"):
        assert forbidden not in source, forbidden
