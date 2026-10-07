"""Provider-neutral reference composition over the fake provider family (api-facade slice 4).

Composes ``compose_reference_host`` over the public host ports (in-memory
record and stream stores, a fixed clock, a scripted grant authority and a
secret resolver) and drives the full public ladder through ``APIHost``:
start, outcome, logs, verify, reverify, artifacts, list/show, cancel and
reconcile. Also pins the export-site rulings: ``reference`` is never lazily
exported, no reference module imports ``api/v1/persistence.py`` and no
reference module imports ``ProjectContext``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from synaptic_tuner.api import v1
from synaptic_tuner.api.v1.artifacts_facade import PublicationRequest
from synaptic_tuner.api.v1.execution import ExecutionGrant
from synaptic_tuner.api.v1.planning import (
    ProviderPlanContextV1, ProviderPlanRef, ResolvedTrainingRequest, TrainingPlan,
    TrainingPlanBasisV1,
)
from synaptic_tuner.api.v1.ports import StoragePartition, StoredRecordV1, StoredStreamEntryV1
from synaptic_tuner.api.v1.providers import ProviderCapabilities, ProviderDescriptor, ProviderRef
from synaptic_tuner.api.v1.reference import (
    ProviderFamilyV1, ReferenceHostPortsV1, ReferenceRequestPortsV1,
    compose_reference_authority, compose_reference_host,
)
from synaptic_tuner.api.v1.reference.stores import (
    InMemoryDurableRecordStoreV1, InMemoryDurableStreamStoreV1,
)
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from synaptic_tuner.api.v1.runs_facade import (
    RunArtifactRequest, RunListRequest, RunLogEntry, RunLogLevel, RunLogsRequest,
    RunOperationCode, RunOperationError,
)
from synaptic_tuner.api.v1.secrets import SecretRef
from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement, TrainingPreflight, TrainingRequest,
)
from tuner.execution.coordinator_v1.model import ProviderExecutionBindingV1, ProviderRunPhaseV1
from tuner.execution.coordinator_v1.publication import PublicationCodeV1, PublicationErrorV1
from tuner.execution.fake_provider_v1 import (
    FakeArtifactV1, FakeEffectResultV1, FakeProviderConfigV1, FakeProviderFamilyV1,
)
from tuner.execution.foundation_v2.commands import CanonicalProviderPayloadV1
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1, ExecutorDescriptorV1
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import ExecutionScopeV1


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "synaptic_tuner" / "api" / "v1" / "reference"
D = tuple(character * 64 for character in "123456789abcdef")
SECRET = SecretRef("env", "SYNAPTIC_REFERENCE_AUTHORITY")
PROVIDER = ProviderRef("fake", "profile")
RUN = TrainingRunRef("run-1", "project")
NOW = "2026-08-27T00:00:00Z"


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

    def __init__(self, family_holder, policies):
        self._holder = family_holder
        self._policies = policies
        self.authorized = []
        self.bound = []

    def authorize(self, requirements):
        self.authorized.append(requirements)
        assert tuple(item.operation for item in requirements) == ("training.start",)
        return ExecutionGrant("host-grant-1")

    def bind(self, grant, *, operation, requirements):
        assert grant == ExecutionGrant("host-grant-1")
        kind = operation.operation.effect.kind.value
        self.bound.append(kind)
        policy = self._policies.get(kind, "found")
        if policy == "found":
            dispatch = (FakeEffectResultV1(ObservationDisposition.FOUND),)
            reconciliation = ()
        elif policy == "indeterminate_found":
            dispatch = (FakeEffectResultV1(ObservationDisposition.INDETERMINATE),)
            reconciliation = (FakeEffectResultV1(ObservationDisposition.FOUND),)
        else:
            raise AssertionError("unsupported policy")
        self._holder.family.register_command(operation, dispatch=dispatch, reconciliation=reconciliation)
        return object()


class Holder:
    family = None


class Loader:
    def load(self, canonical_json):
        return TrainingRequest("request", RUN.project_ref, canonical_json)


class Resolver:
    def resolve(self, request):
        return ResolvedTrainingRequest(
            "synaptic-resolved-training-request/v1", request.request_id, request.project_ref, *D[:5],
        )


class Identity:
    def for_plan(self, plan):
        return RUN


def _fixture(*, policies=None, records=None, streams=None, family=None, run_phases=None):
    """Compose a reference host over the fake provider family.

    ``records``/``streams`` reuse existing host stores (a recomposed host);
    ``family`` reuses an existing fake provider, which stands for the one
    external provider both hosts talk to; ``run_phases`` scripts the observed
    provider phases.
    """
    records = records if records is not None else InMemoryDurableRecordStoreV1()
    streams = streams if streams is not None else InMemoryDurableStreamStoreV1()
    descriptor = ProviderDescriptor(
        "synaptic-provider-descriptor/v1", PROVIDER.provider_id, "Fake provider", "1.0.0",
        ProviderCapabilities(True, True, True, True, True, False),
    )
    basis = TrainingPlanBasisV1("synaptic-training-plan-basis/v1", "request", RUN.project_ref, *D[:5])
    context = ProviderPlanContextV1(
        "synaptic-provider-plan-context/v1", PROVIDER, basis.basis_digest,
        descriptor.descriptor_digest, D[5],
    )
    plan = TrainingPlan("synaptic-training-plan/v2", basis, ProviderPlanRef(context.provider_context_digest))
    executor = ExecutorDescriptorV1(PROVIDER.provider_id, "executor", "1.0.0")
    adapter = AdapterDescriptorV1(PROVIDER.provider_id, "adapter", "1.0.0")
    scope = ExecutionScopeV1("account", "namespace")
    binding = ProviderExecutionBindingV1(
        PROVIDER, descriptor.descriptor_digest, context.profile_digest, scope, executor,
        adapter.digest, D[7], D[8], D[9],
    )
    artifact_bytes = b"adapter-data"
    config = FakeProviderConfigV1(
        PROVIDER, descriptor, context.profile_digest, scope.account_ref, scope.namespace_ref,
        executor, adapter, (),
        run_phases or (ProviderRunPhaseV1.RUNNING, ProviderRunPhaseV1.SUCCEEDED),
        (RunLogEntry(1, "2026-08-27T12:00:00Z", RunLogLevel.INFO, "progress", "ok", 2),),
        (FakeArtifactV1(
            VerifiedArtifact("adapter", hashlib.sha256(artifact_bytes).hexdigest(), len(artifact_bytes)),
            artifact_bytes,
        ),),
    )
    holder = Holder()
    grants = HostGrants(holder, policies or {})
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
            assert provider == PROVIDER
            return descriptor

        def context(self, resolved, provider):
            assert (provider, resolved.request_id) == (PROVIDER, "request")
            return context

        def preflight(self, plan):
            return TrainingPreflight(
                plan.plan_fingerprint, True, "2026-08-26T00:00:00Z", "2026-08-28T00:00:00Z",
                (AuthorizationRequirement("training.start", True, 100, "USD"),),
            )

    class Preparation:
        def resolve(self, provider, plan_context):
            assert (provider, plan_context) == (PROVIDER, context)
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
        descriptor, Planning(), Preparation(), fake.reader, fake.executor_resolver,
        fake.reconciliation_resolver, fake.evidence_authority, fake.artifact_verifier,
        fake.evidence_authority, fake.evidence_authority,
    )
    composition = compose_reference_host(
        family=family, ports=ports,
        requests=ReferenceRequestPortsV1(Loader(), Resolver(), Identity()),
        authority=authority,
    )
    return composition, composition.api(), fake, grants, plan, records


def _stores(records):
    """Every stored key per partition, so tests can census the durable layout."""
    found = {}
    for partition in StoragePartition:
        page = records.list_page(partition=partition.value, prefix="", after_key=None, limit=100)
        if page.records:
            found[partition.value] = tuple(record.key for record in page.records)
    return found


def _head(records):
    (head,) = records.list_page(
        partition=StoragePartition.WORKFLOW.value, prefix="run/", after_key=None, limit=10,
    ).records
    return head


def _forge(stored, mutate):
    """Rewrite a stored document in place: the tamper a host store never performs itself."""
    document = json.loads(stored.canonical.decode("utf-8"))
    mutate(document)
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _start(api):
    request = api.training.load('{"method":"sft"}')
    resolved = api.training.resolve(request)
    plan = api.training.plan(resolved, PROVIDER)
    ready = api.training.preflight(plan)
    started = api.training.start(plan, ready)
    assert started.accepted is True and started.run == RUN
    return plan, ready, started


def test_reference_host_runs_the_full_public_ladder_over_the_fake_family() -> None:
    composition, api, fake, grants, expected_plan, records = _fixture()
    plan, ready, started = _start(api)
    assert plan == expected_plan
    assert grants.bound == ["stage", "submit"]
    assert api.training.start(plan, ready) == started and grants.bound == ["stage", "submit"]

    assert api.runs.show(RUN).state.value == "queued"
    page = api.runs.list(RunListRequest(RUN.project_ref))
    assert tuple(item.run for item in page.outcomes) == (RUN,) and page.next_cursor is None
    with pytest.raises(RunOperationError) as hidden:
        api.runs.artifacts(RunArtifactRequest(RUN, "adapter", 100))
    assert hidden.value.code is RunOperationCode.ARTIFACTS_UNVERIFIED

    assert api.runs.outcome(RUN).state.value == "running"
    logs = api.runs.logs(RunLogsRequest(RUN, limit=1, maximum_bytes=4096))
    assert logs.entries[0].message == "ok"
    assert api.runs.outcome(RUN).state.value == "succeeded"
    assert api.runs.verify(RUN).verified is True
    reader_calls = fake.trace.snapshot().count(("reader", "artifacts"))
    assert api.runs.reverify(RUN).verified is True
    assert fake.trace.snapshot().count(("reader", "artifacts")) == reader_calls
    stream = api.runs.artifacts(RunArtifactRequest(RUN, "adapter", 100))
    assert b"".join(stream.iter_bytes()) == b"adapter-data"
    assert api.runs.show(RUN).state.value == "succeeded"

    # The workflow lives in the host record store, not in a coordinator object.
    keys = [record.key for record in records.list_page(
        partition=StoragePartition.WORKFLOW.value, prefix="", after_key=None, limit=10,
    ).records]
    assert len(keys) == 2 and sorted(key.split("/")[0] for key in keys) == ["plan", "run"]
    retained = composition.stores.workflow_store.get(RUN)
    assert retained.phase.value == "verified" and retained.revision >= 8
    # So do the effect ledger and the authorization commitments (slice 4b).
    stored = _stores(records)
    assert sorted(key.split("-")[0] for key in stored["effects"]) == ["stage", "submit"]
    assert sorted(key.split("/")[0] for key in stored["authorization"]) == ["plan", "policy"]


# --- slice 4b: head-document growth --------------------------------------------

HEAD_DOCUMENT_BOUND = 128 * 1024
"""Bytes one workflow head may occupy at any revision of the fake-family ladder.

Measured 59,356 bytes after ``start`` and 64,363 bytes at ``verified``; the head
holds one record, so the bound is two times the largest terminal record.
"""


def test_thousand_running_polls_leave_the_head_bounded_and_the_run_still_terminates() -> None:
    running = (ProviderRunPhaseV1.RUNNING,) * 1001 + (ProviderRunPhaseV1.SUCCEEDED,)
    composition, api, fake, _, _, records = _fixture(run_phases=running)
    _start(api)
    store = composition.stores.workflow_store
    assert api.runs.outcome(RUN).state.value == "running"
    entered = store.get(RUN)
    head_bytes = len(_head(records).canonical)
    head_revision = _head(records).revision
    observed = fake.trace.snapshot().count(("reader", "observe"))

    for _ in range(1000):
        assert api.runs.outcome(RUN).state.value == "running"
    assert fake.trace.snapshot().count(("reader", "observe")) == observed + 1000
    # No phase change: no workflow revision, no head rewrite, no history entry.
    assert store.get(RUN) == entered and store.get(RUN).revision == entered.revision
    assert _head(records).revision == head_revision
    assert len(_head(records).canonical) == head_bytes < HEAD_DOCUMENT_BOUND
    history = composition.stores.workflow_store._streams.read_page(
        partition=StoragePartition.WORKFLOW.value, stream_key=_head(records).key,
        after_sequence=None, limit=1000,
    )
    assert len(history.entries) == entered.revision + 1 and history.truncated is False

    assert api.runs.outcome(RUN).state.value == "succeeded"
    assert api.runs.verify(RUN).verified is True
    final = store.get(RUN)
    assert final.phase.value == "verified" and final.revision == entered.revision + 2
    assert len(_head(records).canonical) < HEAD_DOCUMENT_BOUND
    assert len(final.provider_run_observations) == 2


# --- slice 4b: tamper detection across the split ------------------------------

def test_a_forged_head_is_detected_by_the_same_host_and_by_a_cold_host() -> None:
    composition, api, _, _, _, records = _fixture()
    _start(api)
    streams = composition.stores.workflow_store._streams
    head = _head(records)
    forged = _forge(head, lambda document: document["record"].__setitem__("diagnostic_codes", ["forged"]))
    records._partitions[StoragePartition.WORKFLOW.value][head.key] = StoredRecordV1(head.key, head.revision, forged)
    with pytest.raises(RunOperationError) as warm:
        api.runs.show(RUN)
    assert warm.value.code is RunOperationCode.INTEGRITY_ERROR
    _, cold, _, _, _, _ = _fixture(records=records, streams=streams)
    with pytest.raises(RunOperationError) as fresh:
        cold.runs.show(RUN)
    assert fresh.value.code is RunOperationCode.INTEGRITY_ERROR


def test_a_consistently_forged_head_is_detected_by_the_chain() -> None:
    """A forger who also recomputes the stored record digest still fails the replay."""
    composition, api, _, _, _, records = _fixture()
    _start(api)
    streams = composition.stores.workflow_store._streams
    head = _head(records)
    from synaptic_tuner.api.v1.reference.repositories import decode_workflow_record

    def mutate(document):
        document["record"]["diagnostic_codes"] = ["forged"]
        document["record_digest"] = decode_workflow_record(document["record"]).record_digest

    forged = _forge(head, mutate)
    records._partitions[StoragePartition.WORKFLOW.value][head.key] = StoredRecordV1(head.key, head.revision, forged)
    with pytest.raises(RunOperationError) as warm:
        api.runs.show(RUN)
    assert warm.value.code is RunOperationCode.INTEGRITY_ERROR
    _, cold, _, _, _, _ = _fixture(records=records, streams=streams)
    with pytest.raises(RunOperationError) as fresh:
        cold.runs.show(RUN)
    assert fresh.value.code is RunOperationCode.INTEGRITY_ERROR


@pytest.mark.parametrize("field", ["transition", "record_digest", "chain_digest", "previous_chain_digest"])
def test_a_tampered_history_entry_is_detected_when_a_cold_host_replays_it(field) -> None:
    composition, api, _, _, _, records = _fixture()
    _start(api)
    streams = composition.stores.workflow_store._streams
    key = _head(records).key
    entries = streams._streams[(StoragePartition.WORKFLOW.value, key)]
    target = entries[2]

    def mutate(document):
        if field == "transition":
            document["transition"]["kind"] = "begin_preparation"
        else:
            document[field] = "f" * 64

    entries[2] = StoredStreamEntryV1(target.sequence, _forge(target, mutate))
    _, cold, _, _, _, _ = _fixture(records=records, streams=streams)
    with pytest.raises(RunOperationError) as refused:
        cold.runs.show(RUN)
    assert refused.value.code is RunOperationCode.INTEGRITY_ERROR


def test_reference_host_cancel_is_real_and_idempotent() -> None:
    _, api, fake, _, _, _ = _fixture()
    _start(api)
    requested = api.runs.cancel(RUN, "requested")
    assert requested.state.value == "cancel_requested"
    trace = fake.trace.snapshot()
    assert api.runs.cancel(RUN, "requested") == requested
    assert fake.trace.snapshot() == trace
    assert trace.count(("executor", "cancel")) == 1


def test_reference_host_reconciles_an_indeterminate_submit() -> None:
    composition, api, fake, _, _, _ = _fixture(policies={"submit": "indeterminate_found"})
    _start(api)
    assert composition.stores.workflow_store.get(RUN).phase.value == "submit_reconcile_required"
    assert api.runs.show(RUN).state.value == "reconcile_required"
    resumed = api.runs.reconcile(RUN)
    assert resumed.state.value == "queued"
    assert fake.trace.snapshot().count(("adapter", "lookup")) == 1
    assert api.runs.outcome(RUN).state.value == "running"


def test_reference_host_records_survive_recomposition_over_the_same_stores() -> None:
    composition, api, _, _, _, records = _fixture()
    _start(api)
    assert api.runs.outcome(RUN).state.value == "running"
    _, again, _, _, _, _ = _fixture(records=records, streams=composition.stores.workflow_store._streams)
    assert again.runs.show(RUN).state.value == "running"
    assert tuple(item.run for item in again.runs.list(RunListRequest(RUN.project_ref)).outcomes) == (RUN,)


def test_a_recomposed_host_drives_reconcile_outcome_and_cancel_over_the_same_stores() -> None:
    """Slice 4b: effects and authorization commitments are durable, not process-local."""
    first, api, fake, _, _, records = _fixture(policies={"submit": "indeterminate_found"})
    _start(api)
    assert api.runs.show(RUN).state.value == "reconcile_required"
    streams = first.stores.workflow_store._streams

    second, again, _, grants, _, _ = _fixture(
        records=records, streams=streams, family=fake, policies={"submit": "indeterminate_found"},
    )
    assert grants.bound == []  # nothing re-authorized: the commitment came from the store
    assert again.runs.reconcile(RUN).state.value == "queued"
    assert again.runs.outcome(RUN).state.value == "running"
    assert again.runs.cancel(RUN, "requested").state.value == "cancel_requested"
    assert grants.bound == ["cancel"]
    assert fake.trace.snapshot().count(("executor", "cancel")) == 1
    assert sorted(key.split("-")[0] for key in _stores(records)["effects"]) == ["cancel", "stage", "submit"]

    # The first host sees what the second one did, incrementally from its cursor.
    assert api.runs.show(RUN).state.value == "cancel_requested"
    assert first.stores.workflow_store.get(RUN) == second.stores.workflow_store.get(RUN)


def test_a_recomposed_host_alone_cannot_forge_a_commitment_from_a_tampered_record() -> None:
    first, api, fake, _, _, records = _fixture(policies={"submit": "indeterminate_found"})
    _start(api)
    (policy_key,) = [key for key in _stores(records)["authorization"] if key.startswith("policy/")]
    stored = records.read(partition=StoragePartition.AUTHORIZATION.value, key=policy_key)
    forged = _forge(stored, lambda document: document.__setitem__("expires_at_epoch", 2**40))
    records._partitions[StoragePartition.AUTHORIZATION.value][policy_key] = StoredRecordV1(policy_key, 1, forged)
    _, again, _, _, _, _ = _fixture(
        records=records, streams=first.stores.workflow_store._streams, family=fake,
    )
    with pytest.raises(RunOperationError) as refused:
        again.runs.reconcile(RUN)
    assert refused.value.code is RunOperationCode.STATE_CONFLICT
    assert again.runs.show(RUN).state.value == "reconcile_required"


def test_reference_artifacts_family_has_no_destinations() -> None:
    _, api, _, _, _, _ = _fixture()
    assert api.artifacts.destinations().destinations == ()
    assert api.artifacts.publications("anywhere").publications == ()
    with pytest.raises(PublicationErrorV1) as refused:
        api.artifacts.publish(PublicationRequest(RUN, "anywhere"))
    assert refused.value.code is PublicationCodeV1.DESTINATION_MISSING


def test_reference_host_refuses_a_short_authority_secret() -> None:
    class ShortSecrets:
        def resolve(self, reference):
            return "short"

    with pytest.raises(Exception, match="too short"):
        compose_reference_authority(clock=Clock(), secrets=ShortSecrets(), authority_secret=SECRET)


def test_reference_is_never_lazily_exported() -> None:
    assert "reference" not in v1._LAZY_MODULE_ATTRIBUTES
    assert not any(name.startswith("compose_") for name in v1._FORMAL_EXPORTS)


def _imports(path: Path) -> list[tuple[str | None, tuple[str, ...]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.append(("." * node.level + (node.module or ""), tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            found.append((None, tuple(alias.name for alias in node.names)))
    return found


@pytest.mark.parametrize("path", sorted(REFERENCE_DIR.glob("*.py")), ids=lambda path: path.name)
def test_reference_modules_import_no_persistence_and_no_project_context(path: Path) -> None:
    for module, names in _imports(path):
        assert module != "synaptic_tuner.api.v1.persistence"
        assert module is None or not module.endswith("persistence")
        assert "persistence" not in names
        assert "ProjectContext" not in names
        assert module != "synaptic_tuner.api.v1.context"
        assert module is None or not module.endswith(".context")
        assert "context" not in names
