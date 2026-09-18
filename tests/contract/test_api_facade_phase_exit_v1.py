"""API facade phase-exit conformance (``docs/plans/api-facade-slices.md``, "Phase exit").

One ``compose_reference_host`` over the fake provider family, in-memory record
and stream stores and all four optional family ports (evaluation, data,
pipelines, chat), wrapped in ``APIHost(HostPorts(...))``. From that single host
every one of the seven facades is driven through its verb ladder, then the
phase-exit clauses are asserted: every facade property is the expected ``*API``
type, both import-closure gates hold (package gate in a fresh ``-I``
interpreter, module gate plus AST source gate over the contract module list),
``synaptic_tuner/host`` is gone, no module under ``api/v1/reference/`` imports
``api/v1/persistence.py``, the effects partition holds only fake-provider
``stage``/``submit``/``cancel`` effects and no ``spend`` effect, nothing under
the tmp project root is a ``.synaptic`` or ``.tracking`` directory, and a socket
guard is active for the whole module.

Fakes and fixtures are reused from the per-family reference tests rather than
re-invented: the fake provider family wiring from
``test_reference_composition_v1``, the scripted evaluation backend and scenario
config from ``test_reference_evaluation_v1``, the scripted SynthChat client from
``test_reference_data_v1``, the pipeline request and pacing from
``test_reference_pipelines_v1`` and the fake chat runtime from
``test_reference_chat_v1``. The only new fakes are a request loader and a run
identity that derive distinct ids per canonical request, so one host can hold
several training runs (the direct ladder, the cancelled run and the pipeline's
train child) instead of the single fixed ``run-1``.

No live provider effect is claimed; the frozen matrix rows for Modal
cancel/reconcile and for publication are untouched by this module.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from synaptic_tuner.api.v1.artifacts_facade import ArtifactsAPI, PublicationRef, PublicationRequest
from synaptic_tuner.api.v1.chat_facade import (
    ChatAPI, ChatListRequest, ChatSessionState, ChatTurnRequest,
)
from synaptic_tuner.api.v1.data_facade import (
    DataAPI, DataListRequest, DataMode, DataRequest, DataRunState, DataScenarioTarget,
    DatasetListRequest, DatasetValidateRequest,
)
from synaptic_tuner.api.v1.evaluation_facade import (
    EvaluationAPI, EvaluationListRequest, EvaluationOperationCode, EvaluationOperationError,
    EvaluationResultRequest, EvaluationRunRef, EvaluationRunState,
)
from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.observations import (
    ObservationFamily, ObservationKind, ObservationsRequest, ObservationStreamRef,
)
from synaptic_tuner.api.v1.pipelines_facade import (
    PipelineListRequest, PipelineState, PipelinesAPI, StageState,
)
from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlanBasisV1
from synaptic_tuner.api.v1.ports import StoragePartition
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.reference import (
    ChatRuntimeRegistryV1, DirectoryEvaluationArtifactSinkV1, EvaluationBackendRegistryV1,
    ProviderFamilyV1, ReferenceChatPortsV1, ReferenceDataPortsV1, ReferenceEvaluationPortsV1,
    ReferenceHostPortsV1, ReferencePipelinePortsV1, ReferenceRequestPortsV1,
    compose_reference_authority, compose_reference_host,
)
from synaptic_tuner.api.v1.reference.data import data_run_ref
from synaptic_tuner.api.v1.reference.evaluation import EVALUATION_RESULTS_ROLE, EVALUATION_TRACE_ROLE
from synaptic_tuner.api.v1.reference.repositories import decode_effect_record
from synaptic_tuner.api.v1.reference.spend import SPEND_EFFECT_KEY_PREFIX
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, TrainingRunState, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest, RunListRequest, RunLogEntry, RunLogLevel, RunLogsRequest, RunsAPI,
)
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingAPI, TrainingPreflight, TrainingRequest,
)
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1, ProviderRunPhaseV1
from tuner.execution.coordinator_v1.publication import PublicationCodeV1, PublicationErrorV1
from tuner.execution.fake_provider_v1 import FakeArtifactV1, FakeProviderConfigV1, FakeProviderFamilyV1
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1, parse_exact_command
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1

from test_reference_chat_v1 import HUB, Client as ChatClient, FakeRuntime, _open as _chat_open
from test_reference_composition_v1 import (
    D, SECRET, Clock, HostGrants, Holder, Resolver, Secrets, _imports, _stores,
)
from test_reference_data_v1 import (
    CHAT_CALLS_PER_ROW, CONFIG_DIR, RUBRICS_DIR, SCENARIO_YAML, FakeLocalClient, Factory,
    _assert_artifacts_match_disk,
)
from test_reference_evaluation_v1 import (
    MODEL, ScriptedBackend, _request as _evaluation_request, _write_config,
)
from test_reference_pipelines_v1 import Pacing, _no_legacy_dirs, _request as _pipeline_request


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "synaptic_tuner" / "api" / "v1" / "reference"
PROJECT = "project"
PROVIDER = ProviderRef("fake", "profile")
ADAPTER_BYTES = b"adapter-data"
ADAPTER = VerifiedArtifact("adapter", hashlib.sha256(ADAPTER_BYTES).hexdigest(), len(ADAPTER_BYTES))
BLOCKED_MODULES = ("huggingface_hub", "modal", "runpod", "sqlite3", "tuner")
CONTRACT_MODULES = (
    "providers", "planning", "results", "training_facade", "runs_facade", "artifacts_facade",
    "observations", "usage", "ports", "reference.stores", "evaluation_facade", "data_facade",
    "pipelines_facade", "chat_facade",
)
CONTRACT_SOURCES = (
    "_contract.py", "providers.py", "planning.py", "results.py", "training_facade.py",
    "runs_facade.py", "artifacts_facade.py", "observations.py", "usage.py", "ports.py",
    "evaluation_facade.py", "data_facade.py", "pipelines_facade.py", "chat_facade.py",
    "reference/__init__.py", "reference/stores.py",
)
# One fake provider, one observe cursor shared by every submitted run (the fake keys it by
# provider job ref, and every submit is ``job-1``): the direct ladder consumes the first
# RUNNING/SUCCEEDED pair, the pipeline's train child the second, and every later observe
# repeats the final SUCCEEDED.
RUN_PHASES = (
    ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED,
    ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED,
)


# --- guards ---------------------------------------------------------------------------


def _refuse_network(*args, **kwargs):
    raise AssertionError("network access is forbidden in the phase-exit conformance test")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse_network)
    monkeypatch.setattr(socket, "create_connection", _refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse_network)


# --- the two new fakes: per-request identities so one host holds several runs ---------------


class Loader:
    """``TrainingRequestLoader`` deriving the request id from the canonical request."""

    def load(self, canonical_json):
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
        return TrainingRequest(f"request-{digest}", PROJECT, canonical_json)


class Identity:
    """``RunIdentityPort`` naming one run per request, so distinct requests are distinct runs."""

    def for_plan(self, plan):
        return TrainingRunRef(plan.basis.request_id.replace("request-", "run-", 1), PROJECT)


# --- the single composition -----------------------------------------------------------------


class Composed:
    """Everything the phase-exit test drives or inspects, from one ``compose_reference_host``."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.records = InMemoryDurableRecordStoreV1()
        self.streams = InMemoryDurableStreamStoreV1()
        descriptor = ProviderDescriptor(
            "synaptic-provider-descriptor/v1", PROVIDER.provider_id, "Fake provider", "1.0.0",
            ProviderCapabilities(True, True, True, True, True, False),
        )
        executor = ExecutorDescriptorV1(PROVIDER.provider_id, "executor", "1.0.0")
        adapter = AdapterDescriptorV1(PROVIDER.provider_id, "adapter", "1.0.0")
        scope = ExecutionScopeV1("account", "namespace")
        binding = ProviderExecutionBindingV1(
            PROVIDER, descriptor.descriptor_digest, D[5], scope, executor, adapter.digest, D[7], D[8], D[9],
        )
        config = FakeProviderConfigV1(
            PROVIDER, descriptor, D[5], scope.account_ref, scope.namespace_ref, executor, adapter, (),
            RUN_PHASES,
            (RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "progress", "ok", 2),),
            (FakeArtifactV1(ADAPTER, ADAPTER_BYTES),),
        )
        holder = Holder()
        self.grants = HostGrants(holder, {})
        ports = ReferenceHostPortsV1(self.records, self.streams, Clock(), self.grants, Secrets())
        authority = compose_reference_authority(clock=ports.clock, secrets=ports.secrets, authority_secret=SECRET)
        self.fake = FakeProviderFamilyV1(
            config, evidence_key=b"e" * 32,
            foundation_authenticator=authority.foundation_authenticator,
            assessment_authenticator=authority.assessment_authority,
        )
        holder.family = self.fake

        class Planning:
            def describe(self, provider):
                assert provider == PROVIDER
                return descriptor

            def context(self, resolved, provider):
                basis = TrainingPlanBasisV1.from_resolved(resolved)
                return ProviderPlanContextV1(
                    "synaptic-provider-plan-context/v1", PROVIDER, basis.basis_digest,
                    descriptor.descriptor_digest, D[5],
                )

            def preflight(self, plan):
                return TrainingPreflight(
                    plan.plan_fingerprint, True, "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
                    (AuthorizationRequirement("training.start", True, 100, "USD"),),
                )

        class Preparation:
            def resolve(self, provider, plan_context):
                assert (provider, plan_context.profile_digest) == (PROVIDER, D[5])
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

        family = ProviderFamilyV1(
            descriptor, Planning(), Preparation(), self.fake.reader, self.fake.executor_resolver,
            self.fake.reconciliation_resolver, self.fake.evidence_authority, self.fake.artifact_verifier,
            self.fake.evidence_authority, self.fake.evidence_authority,
        )

        # Evaluation: the slice-6 scenario config and scripted local backends; the second
        # backend cancels its own run at the first chat, the cooperative cancel path.
        self.config_root = _write_config(tmp_path)
        self.sink_root = tmp_path / "sink"
        self.local_backend = ScriptedBackend()
        self.cancelling_backend = ScriptedBackend(on_chat=self._cancel_running_evaluation)
        evaluation = ReferenceEvaluationPortsV1(
            self.config_root,
            EvaluationBackendRegistryV1({"local": self.local_backend, "cancelling": self.cancelling_backend}),
            DirectoryEvaluationArtifactSinkV1(self.sink_root), None,
        )

        # Data: the slice-9 scenario file under the tmp root and the scripted SynthChat client.
        self.scenarios_dir = tmp_path / "synthchat-scenarios"
        self.scenarios_dir.mkdir()
        (self.scenarios_dir / "behavioral.yaml").write_text(SCENARIO_YAML, encoding="utf-8")
        self.datasets_dir = tmp_path / "datasets"
        self.datasets_dir.mkdir()
        self.data_client = FakeLocalClient([f"text {index}" for index in range(3 * CHAT_CALLS_PER_ROW)])
        self.data_factory = Factory(self.data_client)
        data = ReferenceDataPortsV1(CONFIG_DIR, self.scenarios_dir, RUBRICS_DIR, self.data_factory)

        # Pipelines: one child poll per start/resume so ``resume`` has work to do.
        self.pacing = Pacing()
        pipelines = ReferencePipelinePortsV1(1, self.pacing)

        # Chat: the slice-8 fake runtime over the real engine ``ChatSession``.
        self.chat_runtime = FakeRuntime(ChatClient(lambda messages: f"reply to {messages[-1]['content']}"))
        chat = ReferenceChatPortsV1(ChatRuntimeRegistryV1({"fake": self.chat_runtime}))

        self.composition = compose_reference_host(
            family=family, ports=ports,
            requests=ReferenceRequestPortsV1(Loader(), Resolver(), Identity()),
            authority=authority, evaluation=evaluation, data=data, pipelines=pipelines, chat=chat,
        )
        self.host = APIHost(HostPorts(
            training=self.composition.training, runs=self.composition.runs,
            artifacts=self.composition.artifacts, evaluation=self.composition.evaluation,
            chat=self.composition.chat, data=self.composition.data,
            pipelines=self.composition.pipelines, clock=ports.clock,
        ))
        self.cancelled_evaluation = None

    def _cancel_running_evaluation(self, question):
        if question != "Say hello":
            return
        running = [
            outcome for outcome in self.host.evaluation.list(EvaluationListRequest(PROJECT)).outcomes
            if outcome.state is EvaluationRunState.RUNNING
        ]
        (outcome,) = running
        requested = self.host.evaluation.cancel(outcome.run, "operator stop")
        assert requested.state is EvaluationRunState.CANCEL_REQUESTED
        self.cancelled_evaluation = outcome.run

    def start_training(self, canonical_json):
        api = self.host.training
        request = api.load(canonical_json)
        resolved = api.resolve(request)
        plan = api.plan(resolved, PROVIDER)
        ready = api.preflight(plan)
        assert ready.ready is True and ready.binds(plan)
        started = api.start(plan, ready)
        assert started.accepted is True and started.run.project_ref == PROJECT
        assert api.start(plan, ready) == started
        return started.run

    def observations(self, family, entity_id, **changes):
        api = getattr(self.host, family.value if family is not ObservationFamily.PIPELINE else "pipelines")
        stream = ObservationStreamRef(family, PROJECT, entity_id)
        return api.observations(ObservationsRequest(stream, **changes)).records

    def effects(self):
        """Every stored effect, decoded: ``(key, kind, provider)``."""
        page = self.records.list_page(partition=StoragePartition.EFFECTS.value, prefix="", after_key=None, limit=100)
        assert page.truncated is False
        found = []
        for stored in page.records:
            record = decode_effect_record(stored.canonical, stored.key)
            effect = parse_exact_command(record.command_bytes).operation.effect
            found.append((stored.key, effect.kind, effect.provider))
        return found


# --- the ladder ---------------------------------------------------------------------------


def test_reference_host_drives_all_seven_facades_through_their_ladders_over_the_fake_family(tmp_path) -> None:
    with pytest.raises(AssertionError):
        socket.create_connection(("127.0.0.1", 9))
    composed = Composed(tmp_path)
    host = composed.host

    # Every facade property is the expected public type, from one HostPorts.
    assert type(host.training) is TrainingAPI and type(host.runs) is RunsAPI
    assert type(host.artifacts) is ArtifactsAPI and type(host.evaluation) is EvaluationAPI
    assert type(host.data) is DataAPI and type(host.pipelines) is PipelinesAPI and type(host.chat) is ChatAPI
    assert tuple(HostPorts.__dataclass_fields__) == (
        "training", "runs", "artifacts", "evaluation", "chat", "data", "pipelines", "clock",
    )

    # --- training: load -> resolve -> plan -> preflight -> start; runs: the read ladder ---
    direct = composed.start_training('{"method":"kto"}')
    assert composed.grants.bound == ["stage", "submit"]
    assert host.runs.show(direct).state is TrainingRunState.QUEUED
    assert host.runs.outcome(direct).state is TrainingRunState.RUNNING
    logs = host.runs.logs(RunLogsRequest(direct, limit=1, maximum_bytes=4096))
    assert [entry.message for entry in logs.entries] == ["ok"]
    assert host.runs.outcome(direct).state is TrainingRunState.SUCCEEDED
    assert host.runs.verify(direct).verified is True
    assert host.runs.reverify(direct).verified is True
    stream = host.runs.artifacts(RunArtifactRequest(direct, "adapter", 100))
    assert b"".join(stream.iter_bytes()) == ADAPTER_BYTES
    assert host.runs.show(direct).artifacts == (ADAPTER,)

    # --- artifacts: the read and verify verbs over a host with no destinations ---
    assert host.artifacts.destinations().destinations == ()
    assert host.artifacts.publications("anywhere").publications == ()
    with pytest.raises(PublicationErrorV1) as no_destination:
        host.artifacts.publish(PublicationRequest(direct, "anywhere"))
    assert no_destination.value.code is PublicationCodeV1.DESTINATION_MISSING
    with pytest.raises(PublicationErrorV1) as no_publication:
        host.artifacts.verify(PublicationRef("f" * 64, "anywhere"))
    assert no_publication.value.code is PublicationCodeV1.PUBLICATION_MISSING

    # --- evaluation: plan -> preflight -> start -> show -> result -> list -> observations ---
    evaluation_plan = host.evaluation.plan(_evaluation_request("alpha.yaml"))
    assert evaluation_plan.cases_total == 2
    evaluation_ready = host.evaluation.preflight(evaluation_plan)
    assert evaluation_ready.ready is True and evaluation_ready.authorization == ()
    evaluation_started = host.evaluation.start(evaluation_plan)
    evaluation_run = evaluation_started.run
    assert evaluation_started.accepted is True and type(evaluation_run) is EvaluationRunRef
    evaluation_outcome = host.evaluation.show(evaluation_run)
    assert evaluation_outcome.state is EvaluationRunState.SUCCEEDED
    assert (evaluation_outcome.cases_total, evaluation_outcome.cases_scored) == (2, 2)
    assert tuple(item.role for item in evaluation_outcome.artifacts) == (EVALUATION_RESULTS_ROLE, EVALUATION_TRACE_ROLE)
    for artifact in evaluation_outcome.artifacts:
        raw = (composed.sink_root / evaluation_run.run_id / f"{artifact.role}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact.sha256 and len(raw) == artifact.size_bytes
    result = host.evaluation.result(EvaluationResultRequest(evaluation_run))
    assert result.state is EvaluationRunState.SUCCEEDED and result.model == MODEL and result.usage is None
    assert (result.verdicts.passed, result.verdicts.failed, result.verdicts.inconclusive) == (2, 0, 0)
    assert host.evaluation.list(EvaluationListRequest(PROJECT)).outcomes == (evaluation_outcome,)
    kinds = [record.kind for record in composed.observations(ObservationFamily.EVALUATION, evaluation_run.run_id)]
    assert kinds == [
        ObservationKind.EVALUATION_CASE_STARTED, ObservationKind.EVALUATION_CASE_SCORED,
        ObservationKind.EVALUATION_CASE_STARTED, ObservationKind.EVALUATION_CASE_SCORED,
        ObservationKind.EVALUATION_STAGE_COMPLETED,
    ]
    assert composed.local_backend.opened == [MODEL]

    # --- evaluation: the cooperative cancel path, at a scenario boundary ---
    cancel_plan = host.evaluation.plan(
        _evaluation_request("alpha.yaml", "beta.yaml", backend="cancelling", request_id="req-cancel"),
    )
    assert cancel_plan.cases_total == 4
    cancelled_run = host.evaluation.start(cancel_plan).run
    assert cancelled_run == composed.cancelled_evaluation and cancelled_run != evaluation_run
    cancelled = host.evaluation.show(cancelled_run)
    assert cancelled.state is EvaluationRunState.CANCELLED
    assert (cancelled.cases_total, cancelled.cases_scored) == (4, 2)
    assert composed.cancelling_backend.chats == ["Say hello", "Summarise the document"]
    assert tuple(item.role for item in cancelled.artifacts) == (EVALUATION_RESULTS_ROLE, EVALUATION_TRACE_ROLE)
    with pytest.raises(EvaluationOperationError) as ineligible:
        host.evaluation.cancel(cancelled_run, "again")
    assert ineligible.value.code is EvaluationOperationCode.CANCEL_INELIGIBLE

    # --- data: plan -> preflight -> start -> show -> list -> datasets -> validate -> observations ---
    output = composed.datasets_dir / "run.jsonl"
    data_request = DataRequest(
        request_id="request-data", project_ref=PROJECT, mode=DataMode.GENERATE, backend="lmstudio",
        model="fake-local", output_ref=str(output),
        scenarios=(DataScenarioTarget("greeting", 2), DataScenarioTarget("farewell", 1)),
    )
    data_plan = host.data.plan(data_request)
    assert data_plan.rows_requested == 3 and host.data.plan(data_request) == data_plan
    data_ready = host.data.preflight(data_plan)
    assert data_ready.ready is True and data_ready.binds(data_plan)
    assert [(item.operation, item.paid_effect) for item in data_ready.authorization] == [("data.start", False)]
    data_started = host.data.start(data_plan)
    assert data_started.accepted is True and data_started.run == data_run_ref(data_plan)
    assert composed.data_factory.requests == [("lmstudio", "fake-local")]
    assert composed.data_client.chat_calls == 3 * CHAT_CALLS_PER_ROW
    data_outcome = host.data.show(data_started.run)
    assert data_outcome.state is DataRunState.SUCCEEDED and data_outcome.diagnostic_code is None
    assert (data_outcome.rows_requested, data_outcome.rows_written) == (3, 3)
    _assert_artifacts_match_disk(data_outcome, output)  # homogeneous JSONL, sidecar present, digests match
    assert host.data.list(DataListRequest(PROJECT)).outcomes == (data_outcome,)
    datasets = host.data.datasets(DatasetListRequest(PROJECT, str(composed.datasets_dir)))
    assert [Path(item.dataset_ref).name for item in datasets.datasets] == ["run.jsonl"]
    assert datasets.datasets[0].metadata_ref == str(output.with_name("run.jsonl.meta.json"))
    report = host.data.validate(DatasetValidateRequest(PROJECT, str(output)))
    assert (report.rows_checked, report.rows_valid, report.findings) == (3, 3, ())
    data_kinds = [record.kind for record in composed.observations(ObservationFamily.DATA, data_started.run.run_id)]
    assert data_kinds.count(ObservationKind.DATA_ROW_WRITTEN) == 3
    assert data_kinds.count(ObservationKind.DATA_SCENARIO_COMPLETED) == 2

    # --- pipelines: plan -> start -> show -> resume -> list -> observations over this host's families ---
    pipeline_plan = host.pipelines.plan(_pipeline_request())
    pipeline_started = host.pipelines.start(pipeline_plan)
    pipeline = pipeline_started.pipeline
    assert pipeline_started.accepted is True and pipeline.project_ref == PROJECT
    interrupted = host.pipelines.show(pipeline)
    assert interrupted.state is PipelineState.RUNNING
    train, evaluate = interrupted.stages
    assert train.state is StageState.RUNNING and evaluate.state is StageState.PLANNED
    child_run = train.run
    assert type(child_run) is TrainingRunRef and child_run != direct
    assert host.runs.show(child_run).state is TrainingRunState.RUNNING
    assert composed.grants.bound == ["stage", "submit", "stage", "submit"]
    resumed = host.pipelines.resume(pipeline)
    assert resumed.state is PipelineState.SUCCEEDED
    train, evaluate = resumed.stages
    assert (train.state, evaluate.state) == (StageState.SUCCEEDED, StageState.SUCCEEDED)
    assert train.run == child_run and train.outputs == (ADAPTER,) and evaluate.inputs == (ADAPTER,)
    assert composed.grants.bound == ["stage", "submit", "stage", "submit"]  # resume started nothing new
    # The children are this host's own runs, visible through host.runs and host.evaluation.
    assert host.runs.show(child_run).state is TrainingRunState.SUCCEEDED
    assert host.runs.show(child_run).artifacts == (ADAPTER,)
    assert type(evaluate.run) is EvaluationRunRef and evaluate.run not in (evaluation_run, cancelled_run)
    assert host.evaluation.show(evaluate.run).state is EvaluationRunState.SUCCEEDED
    assert host.evaluation.show(evaluate.run).artifacts == evaluate.outputs
    assert [model.model_revision for model in composed.local_backend.opened] == [MODEL.model_revision, ADAPTER.sha256]
    assert host.pipelines.list(PipelineListRequest(PROJECT)).records == (resumed,)
    pipeline_observations = composed.observations(ObservationFamily.PIPELINE, pipeline.pipeline_id)
    assert [(record.kind, record.payload.stage) for record in pipeline_observations] == [
        (ObservationKind.PIPELINE_STAGE_STARTED, "train"),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "train"),
        (ObservationKind.PIPELINE_STAGE_STARTED, "evaluate"),
        (ObservationKind.PIPELINE_STAGE_COMPLETED, "evaluate"),
    ]

    # --- runs: cancel on a third run, real and idempotent against the fake provider ---
    to_cancel = composed.start_training('{"method":"grpo"}')
    requested = host.runs.cancel(to_cancel, "phase exit")
    assert requested.state is TrainingRunState.CANCEL_REQUESTED
    assert host.runs.cancel(to_cancel, "phase exit") == requested
    assert composed.fake.trace.snapshot().count(("executor", "cancel")) == 1
    assert composed.grants.bound == ["stage", "submit"] * 3 + ["cancel"]
    runs_page = host.runs.list(RunListRequest(PROJECT))
    assert {item.run for item in runs_page.outcomes} == {direct, child_run, to_cancel}
    assert runs_page.next_cursor is None
    assert {item.run for item in host.evaluation.list(EvaluationListRequest(PROJECT)).outcomes} == {
        evaluation_run, cancelled_run, evaluate.run,
    }

    # --- chat: open -> turn -> turn -> show -> list -> close -> observations ---
    session = host.chat.open(_chat_open())
    assert session.state is ChatSessionState.READY and session.model == HUB.identity()
    first = host.chat.turn(ChatTurnRequest(session.ref, "question 1"))
    second = host.chat.turn(ChatTurnRequest(session.ref, "question 2"))
    assert (first.ref.request_id, second.ref.request_id) == (1, 2) and second.ref.follows(first.ref)
    assert (first.content, second.content) == ("reply to question 1", "reply to question 2")
    shown = host.chat.show(session.ref)
    assert shown.state is ChatSessionState.READY and shown.turns == 2
    assert [record.ref for record in host.chat.list(ChatListRequest(PROJECT)).sessions] == [session.ref]
    closed = host.chat.close(session.ref)
    assert closed.state is ChatSessionState.CLOSED and closed.turns == 2
    assert composed.chat_runtime.lease.closes == 1 and composed.chat_runtime.client.calls == 2
    chat_observations = composed.observations(ObservationFamily.CHAT, session.ref.session_id)
    assert [record.kind for record in chat_observations] == [
        ObservationKind.CHAT_TURN_STARTED, ObservationKind.CHAT_TURN_COMPLETED,
    ] * 2
    assert [record.payload.request_id for record in chat_observations] == [1, 1, 2, 2]

    # --- the effects partition: only fake-provider stage/submit/cancel, never spend ---
    effects = composed.effects()
    assert [kind for _, kind, _ in effects].count(EffectKind.STAGE) == 3
    assert [kind for _, kind, _ in effects].count(EffectKind.SUBMIT) == 3
    assert [kind for _, kind, _ in effects].count(EffectKind.CANCEL) == 1
    assert len(effects) == 7 and all(kind is not EffectKind.SPEND for _, kind, _ in effects)
    assert all(provider == PROVIDER for _, _, provider in effects)
    assert not any(key.startswith(SPEND_EFFECT_KEY_PREFIX) or key.startswith("spend") for key, _, _ in effects)
    # Local backends everywhere: only training.start was ever authorized (three starts plus the
    # two idempotent replays in ``start_training``); no evaluation or data grant was requested.
    assert set(composed.grants.authorized) == {(AuthorizationRequirement("training.start", True, 100, "USD"),)}
    assert len(composed.grants.authorized) == 5
    stored = set(_stores(composed.records))
    assert stored >= {
        StoragePartition.WORKFLOW.value, StoragePartition.EVALUATION.value, StoragePartition.DATA.value,
        StoragePartition.PIPELINE.value, StoragePartition.CHAT_SESSION.value, StoragePartition.EFFECTS.value,
        StoragePartition.AUTHORIZATION.value,
    }

    # --- nothing legacy under the project root; the fake executor saw exactly the ledger's effects ---
    _no_legacy_dirs(tmp_path)
    trace = composed.fake.trace.snapshot()
    assert (trace.count(("executor", "stage")), trace.count(("executor", "submit")), trace.count(("executor", "cancel"))) == (3, 3, 1)
    with pytest.raises(AssertionError):
        socket.create_connection(("127.0.0.1", 9))


# --- the two import-closure gates -----------------------------------------------------------


def _blocked_in(modules) -> list[str]:
    return sorted(
        name for name in modules
        if name in BLOCKED_MODULES or name.startswith(tuple(prefix + "." for prefix in BLOCKED_MODULES))
    )


def _fresh_interpreter(script: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return json.loads(completed.stdout)


def test_package_gate_importing_the_public_package_loads_no_provider_engine_or_database_module() -> None:
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
import synaptic_tuner.api.v1
blocked = {BLOCKED_MODULES!r}
print(json.dumps(sorted(n for n in sys.modules if n in blocked or n.startswith(tuple(b + '.' for b in blocked)))))
"""
    assert _fresh_interpreter(script) == []


def test_module_gate_importing_every_contract_submodule_directly_loads_no_engine_module() -> None:
    imports = "\n".join(f"import synaptic_tuner.api.v1.{name}" for name in CONTRACT_MODULES)
    script = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
{imports}
blocked = {BLOCKED_MODULES!r}
print(json.dumps(sorted(n for n in sys.modules if n in blocked or n.startswith(tuple(b + '.' for b in blocked)))))
"""
    assert _fresh_interpreter(script) == []


def test_source_gate_contract_sources_import_no_engine_provider_or_database_module() -> None:
    for name in CONTRACT_SOURCES:
        path = ROOT / "synaptic_tuner" / "api" / "v1" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        found.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        assert found.isdisjoint({"tuner", "modal", "sqlite3"}), name


# --- the deletion and the persistence discipline --------------------------------------------


def test_the_superseded_host_package_is_gone() -> None:
    assert not (ROOT / "synaptic_tuner" / "host").exists()
    assert importlib.util.find_spec("synaptic_tuner.host") is None
    assert not (ROOT / "synaptic_tuner" / "api" / "v1" / "reference" / "host").exists()


def test_no_reference_module_imports_the_frozen_persistence_module() -> None:
    paths = sorted(REFERENCE_DIR.glob("*.py"))
    assert len(paths) >= 14
    for path in paths:
        for module, names in _imports(path):
            assert module != "synaptic_tuner.api.v1.persistence", path.name
            assert module is None or not module.endswith("persistence"), path.name
            assert "persistence" not in names, path.name
