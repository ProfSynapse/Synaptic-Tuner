from dataclasses import replace
import hashlib

import pytest

import tests.execution.coordinator_v1.test_state_machine as foundation_cases
from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.runs_facade import RunLogEntry, RunLogLevel
from tuner.execution.coordinator_v1.model import (
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    BoundProviderRunRefV1,
    FoundationEffectOutcomeV1,
    ProviderLogQueryV1,
    ProviderReadPurposeV1,
    ProviderRunPhaseV1,
)
from tuner.execution.coordinator_v1.state_machine import provider_run_read_request
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import build_stage_command, build_submit_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.references import ScopedProviderRunRefV1, StagePredecessorV2
from tuner.execution.providers.modal.contracts import ArtifactMemberV1, ArtifactRole
from tuner.execution.providers.modal.control import CrossPlaneIdentityV1
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_reader import (
    ModalArtifactInventory,
    ModalCoordinatorReaderError,
    ModalCoordinatorRunReader,
    ModalLogSnapshot,
    ModalTerminalSnapshot,
)
from tuner.execution.providers.modal.manifest import CompletionManifestV1
from tests.execution.providers.test_modal_coordinator_adapter import composed, inputs
from tests.execution.providers.test_modal_sdk154_adapter import verified
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1


D = foundation_cases.D


def _foundation(monkeypatch):
    adapter, context, plan, execution = composed()
    provider = context.provider
    scope = execution.scope
    run = TrainingRunRef("run-a", "project-a")
    descriptor = adapter.describe(provider)
    preparation = adapter.prepare(plan, run, execution)

    def modal_intent(kind):
        if kind == "stage":
            command = build_stage_command(
                preparation, "nonce-stage", adapter.payload(preparation, EffectKind.STAGE),
                execution.executor_descriptor,
            )
        elif kind == "submit":
            command = build_submit_command(
                preparation, "nonce-submit", adapter.payload(preparation, EffectKind.SUBMIT),
                execution.executor_descriptor,
                StagePredecessorV2(
                    provider.provider_id, provider.profile_ref, scope.account_ref,
                    scope.namespace_ref, run.project_ref, run.run_id,
                    plan.plan_fingerprint, preparation.preparation_digest,
                    preparation.workload_digest, "stage-effect", D[12], D[13],
                ),
            )
        else:
            raise AssertionError("reader fixture only needs stage and submit")
        return foundation_cases.EffectIntentV1.from_command_bytes(command.canonical_bytes)

    for name, value in {
        "PROVIDER": provider, "SCOPE": scope, "RUN": run, "DESC": descriptor,
        "BASIS": plan.basis, "CONTEXT": context, "PLAN": plan,
        "STAGE_REF": foundation_cases.ProviderStageRefV1(
            "modal", provider.profile_ref, scope.account_ref, scope.namespace_ref, "stage-a"
        ),
        "PROVIDER_RUN": ScopedProviderRunRefV1(
            "modal", provider.profile_ref, scope.account_ref, scope.namespace_ref, "job-a"
        ),
        "prep": lambda: preparation,
        "intent": modal_intent,
    }.items():
        monkeypatch.setattr(foundation_cases, name, value)
    workflow, record = foundation_cases.queued_evidence()
    assessed = foundation_cases.assessment(record)
    values = inputs()
    selection = ModalDeploymentSelectionV1.from_profile(
        values["profile"], binding=values["binding"],
        runtime_environment=values["runtime_environment"],
        timeout_seconds=values["timeout_seconds"],
    )
    return workflow, record, assessed, adapter, verified(selection)


class Authority:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.failed_log_outputs = 0
        self.authenticated_values = []
    def authenticate(self, value):
        self.authenticated_values.append(value)
        return self.allowed
    def observation(self, content):
        return AuthenticatedProviderRunObservationV1(content, "authority-a", "key-a", "a" * 64)
    def log_page(self, content):
        if self.failed_log_outputs:
            self.failed_log_outputs -= 1
            return None
        return AuthenticatedProviderLogPageV1(content, "authority-a", "key-a", "b" * 64)


class Catalog:
    def __init__(self, value): self.value = value
    def resolve(self, digest):
        assert digest == self.value.command_digest
        return self.value


class Transport:
    def __init__(self, identity, members=(), contents=()):
        self.identity = identity
        self.members = members
        self.contents = dict(contents)
        self.calls = []
        self.terminal_snapshot = None
        self.log_snapshots = []
    def observe(self, binding, *, provider_job_ref):
        self.calls.append("observe")
        return self.terminal_snapshot or ModalTerminalSnapshot.build(
            ProviderRunPhaseV1.RUNNING, self.identity
        )
    def logs(self, binding, query, *, provider_job_ref):
        self.calls.append("logs")
        entry = RunLogEntry(1, "2026-09-09T00:00:00Z", RunLogLevel.INFO,
                            "progress", "ok", 2)
        if self.log_snapshots:
            return self.log_snapshots.pop(0)
        return ModalLogSnapshot.build(
            (entry,), self.identity, D[0], query.log_query_digest, 1, 1,
        )
    def artifact_inventory(self, binding, *, provider_job_ref):
        self.calls.append("inventory")
        completion = CompletionManifestV1(
            tuple(self.members), self.identity.binding.account_ref,
            self.identity.binding.workspace_ref, self.identity.binding.environment_ref,
            self.identity.binding.client_ref, self.identity.binding.sdk_version,
            self.identity.control_volume_id, self.identity.artifact_volume_id,
            self.identity.job_ref, self.identity.effect_id, self.identity.command_digest,
            self.identity.plan_digest, self.identity.deployment_attestation_digest,
            self.identity.invocation_nonce, self.identity.generation, D[1], D[2],
        )
        return ModalArtifactInventory.build(completion, self.identity)
    def iter_artifact(self, binding, member, *, provider_job_ref, maximum_bytes):
        self.calls.append("bytes")
        yield self.contents[member.role.value]


def _reader(monkeypatch, purpose, *, binding_allowed=True, foundation_allowed=True,
            members=(), contents=()):
    workflow, record, assessed, adapter, deployment = _foundation(monkeypatch)
    if purpose is ProviderReadPurposeV1.ARTIFACTS:
        observation_request, observation = foundation_cases.observation(
            workflow, record, ProviderRunPhaseV1.SUCCEEDED, {"completed": True}
        )
        workflow = foundation_cases.apply_provider_observation(
            workflow, observation_request, observation, foundation_cases.ObservationAuth()
        )
    request = provider_run_read_request(
        workflow, record, assessed, foundation_cases.Auth(), foundation_cases.AssessmentAuth(),
        purpose=purpose,
    )
    command = foundation_cases.parse_exact_command(request.submit_command_bytes)
    bound = ModalCommandBinding(
        request.submit_command_bytes, adapter.snapshot(),
        canonical_bytes(deployment.to_dict()),
    )
    client_binding = bound.client_binding
    identity = CrossPlaneIdentityV1(
        client_binding, "control-volume", "artifact-volume", "job-a",
        command.operation.effect.effect_id, command.digest,
        command.preparation.plan_fingerprint, bound.deployment.attestation_digest,
        command.operation.invocation_nonce, 1, "key-a",
    )
    transport = Transport(identity, members, contents)
    authority = Authority(binding_allowed)
    reader = ModalCoordinatorRunReader(
        catalog=Catalog(bound), binding_authority=authority,
        foundation_authenticator=foundation_cases.Auth(foundation_allowed),
        assessment_authenticator=foundation_cases.AssessmentAuth(),
        evidence_authority=authority, transport=transport,
        observed_at="2026-09-09T00:00:00Z",
    )
    return reader, request, transport


def test_observe_authenticates_complete_foundation_before_transport(monkeypatch):
    reader, request, transport = _reader(
        monkeypatch, ProviderReadPurposeV1.OBSERVE, foundation_allowed=False
    )
    with pytest.raises(ModalCoordinatorReaderError, match="authentication_failed"):
        reader.observe(request)
    assert transport.calls == []


def test_observe_and_logs_preserve_bound_cross_plane_evidence(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    observed = reader.observe(request)
    assert observed.content.phase is ProviderRunPhaseV1.RUNNING
    assert transport.calls == ["observe"]

    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.LOGS)
    page = reader.logs(request, ProviderLogQueryV1(None, 1, 4096))
    assert page.content.entries[0].message == "ok"
    assert page.content.total_bytes == 2


def test_inventory_does_not_read_bodies_and_stream_verifies_exact_bytes(monkeypatch):
    roles = tuple(ArtifactRole)
    contents = {role.value: role.value.encode() for role in roles}
    members = tuple(ArtifactMemberV1(
        role, f"operations/submit-effect/output/{role.value}",
        len(contents[role.value]), hashlib.sha256(contents[role.value]).hexdigest(),
        f"entry-{index}",
    ) for index, role in enumerate(roles))
    reader, request, transport = _reader(
        monkeypatch, ProviderReadPurposeV1.ARTIFACTS,
        members=members, contents=contents.items(),
    )
    # Use the actual effect-specific paths required by the submit command.
    effect_id = request.provider_run.effect_id
    transport.members = tuple(replace(member, path=f"operations/{effect_id}/output/{member.role.value}")
                              for member in members)
    manifest = reader.artifacts(request)
    assert transport.calls == ["inventory"]
    role = ArtifactRole.FINAL_MODEL.value
    body = b"".join(reader.iter_artifact_bytes(
        request, manifest, role, maximum_bytes=len(contents[role])
    ))
    assert body == contents[role]
    assert transport.calls == ["inventory", "inventory", "bytes"]


def test_stream_rejects_truncation_before_successful_completion(monkeypatch):
    content = b"model"
    members = tuple(ArtifactMemberV1(
        role, f"operations/submit-effect/output/{role.value}",
        len(content), hashlib.sha256(content).hexdigest(), f"entry-{index}",
    ) for index, role in enumerate(ArtifactRole))
    contents = {role.value: content for role in ArtifactRole}
    reader, request, transport = _reader(
        monkeypatch, ProviderReadPurposeV1.ARTIFACTS,
        members=members, contents=contents.items(),
    )
    effect_id = request.provider_run.effect_id
    transport.members = tuple(replace(member, path=f"operations/{effect_id}/output/{member.role.value}")
                              for member in members)
    manifest = reader.artifacts(request)
    transport.contents[ArtifactRole.FINAL_MODEL.value] = b"mod"
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        b"".join(reader.iter_artifact_bytes(
            request, manifest, ArtifactRole.FINAL_MODEL.value, maximum_bytes=len(content)
        ))


def test_reader_requires_exact_shared_binding_before_io(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    original = reader._catalog.value
    reader._catalog.value = object.__new__(type("StructuralBinding", (), {}))
    for field in ("command_bytes", "preparation_snapshot", "deployment_bytes"):
        setattr(reader._catalog.value, field, getattr(original, field))
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid|authentication_failed"):
        reader.observe(request)
    assert transport.calls == []


def test_reader_rebuilds_shared_binding_before_authority_authentication(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    retained = reader._catalog.value
    reader.observe(request)
    authenticated = reader._binding_authority.authenticated_values[-1]
    assert type(authenticated) is ModalCommandBinding
    assert authenticated == retained
    assert authenticated is not retained
    assert transport.calls == ["observe"]


def test_forged_self_consistent_run_outcome_cannot_replace_authenticated_receipt(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    old = request.foundation_outcome
    original = request.provider_run
    forged_ref = replace(original.reference, provider_job_ref="forged-job")
    outcome_document = {
        "binding_digest": old.binding_digest,
        "disposition": old.disposition.value,
        "receipts": list(old.authenticated_receipt_digests),
        "stage": None,
        "run": forged_ref.to_dict(),
        "cancel": None,
        "finality": old.finality_proof_digest,
    }
    forged_outcome = FoundationEffectOutcomeV1(
        old.binding_digest, old.kind, old.effect_id, old.command_digest,
        old.preparation_digest, old.foundation_record_digest, old.disposition,
        old.authenticated_receipt_digests, old.receipt_content_digests,
        old.observation_digests, None, forged_ref, None,
        old.finality_proof_digest,
        domain_digest("synaptic-foundation-outcome/v1", canonical_bytes(outcome_document)),
    )
    common = (
        original.effect_id, original.command_digest, original.command_bytes_digest,
        original.preparation_digest, original.foundation_binding_digest,
        forged_outcome.outcome_digest, original.authenticated_receipt_digest,
    )
    bound_document = {
        "reference": forged_ref.to_dict(), "effect_id": common[0],
        "command_digest": common[1], "command_bytes_digest": common[2],
        "preparation_digest": common[3], "foundation_binding_digest": common[4],
        "foundation_outcome_digest": common[5],
        "authenticated_receipt_digest": common[6],
    }
    forged_bound = BoundProviderRunRefV1(
        forged_ref, *common,
        domain_digest("synaptic-submit-evidence-binding/v1", canonical_bytes(bound_document)),
    )
    forged_request = foundation_cases.read_request_variant(
        request, provider_run=forged_bound, foundation_outcome=forged_outcome,
    )
    with pytest.raises(ModalCoordinatorReaderError, match="authentication_failed"):
        reader.observe(forged_request)
    assert transport.calls == []


@pytest.mark.parametrize("field,value", [
    ("deployment_attestation_digest", "f" * 64),
    ("invocation_nonce", "other-nonce"),
])
def test_observation_rejects_rebound_cross_plane_identity(monkeypatch, field, value):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    changed = replace(transport.identity, **{field: value})
    transport.terminal_snapshot = ModalTerminalSnapshot.build(
        ProviderRunPhaseV1.RUNNING, changed
    )
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        reader.observe(request)


def test_inventory_rejects_other_effect_output_prefix(monkeypatch):
    content = b"artifact"
    members = tuple(ArtifactMemberV1(
        role, f"operations/other-effect/output/{role.value}", len(content),
        hashlib.sha256(content).hexdigest(), f"entry-{index}",
    ) for index, role in enumerate(ArtifactRole))
    reader, request, _ = _reader(
        monkeypatch, ProviderReadPurposeV1.ARTIFACTS,
        members=members, contents=((role.value, content) for role in ArtifactRole),
    )
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        reader.artifacts(request)


def test_snapshot_evidence_is_deterministically_bound(monkeypatch):
    _, _, transport = _reader(monkeypatch, ProviderReadPurposeV1.OBSERVE)
    valid = ModalTerminalSnapshot.build(ProviderRunPhaseV1.RUNNING, transport.identity)
    with pytest.raises(ValueError, match="does not bind"):
        replace(valid, canonical_evidence=b"{}")


def test_log_snapshot_rejects_regression_and_same_query_equivocation(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.LOGS)
    query = ProviderLogQueryV1(None, 1, 4096)
    entry = RunLogEntry(1, "2026-09-09T00:00:00Z", RunLogLevel.INFO,
                        "progress", "ok", 2)
    transport.log_snapshots = [
        ModalLogSnapshot.build((entry,), transport.identity, D[0],
                               query.log_query_digest, 2, 2, truncated=True),
        ModalLogSnapshot.build((), transport.identity, D[1],
                               query.log_query_digest, 1, 1, truncated=True),
    ]
    reader.logs(request, query)
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        reader.logs(request, query)


def test_rejected_malformed_log_page_does_not_poison_next_valid_page(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.LOGS)
    query = ProviderLogQueryV1(None, 1, 4096)
    entries = tuple(RunLogEntry(
        sequence, "2026-09-09T00:00:00Z", RunLogLevel.INFO,
        "progress", "ok", 2,
    ) for sequence in (1, 2))
    transport.log_snapshots = [
        ModalLogSnapshot.build(entries, transport.identity, D[0],
                               query.log_query_digest, 100, 2),
        ModalLogSnapshot.build(entries[:1], transport.identity, D[1],
                               query.log_query_digest, 1, 1),
    ]
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        reader.logs(request, query)
    assert reader.logs(request, query).content.entries == entries[:1]


def test_failed_log_authority_does_not_poison_next_valid_page(monkeypatch):
    reader, request, transport = _reader(monkeypatch, ProviderReadPurposeV1.LOGS)
    query = ProviderLogQueryV1(None, 1, 4096)
    high = RunLogEntry(100, "2026-09-09T00:00:00Z", RunLogLevel.INFO,
                       "progress", "high", 4)
    low = RunLogEntry(1, "2026-09-09T00:00:00Z", RunLogLevel.INFO,
                      "progress", "ok", 2)
    transport.log_snapshots = [
        ModalLogSnapshot.build((high,), transport.identity, D[0],
                               query.log_query_digest, 100, 100),
        ModalLogSnapshot.build((low,), transport.identity, D[1],
                               query.log_query_digest, 1, 1),
    ]
    reader._authority.failed_log_outputs = 1
    with pytest.raises(ModalCoordinatorReaderError, match="evidence_invalid"):
        reader.logs(request, query)
    assert reader.logs(request, query).content.entries == (low,)
