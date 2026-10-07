"""Credential-free tests for the bounded Modal Foundation effect slice."""

from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from tests.execution.providers.test_modal_coordinator_adapter import composed, inputs
from tests.execution.providers.test_modal_sdk154_adapter import verified
from tuner.execution.foundation_v2.canonical import FoundationError, canonical_bytes
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.commands import (
    build_cancel_command, build_stage_command, build_submit_command,
)
from tuner.execution.foundation_v2.executors import ExecutionResolutionRequestV2
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import InvalidEvidenceAuthorityV2, ReceiptAuthorityV2
from tuner.execution.foundation_v2.repository import EffectState, InMemoryEffectRepositoryV2
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.foundation_v2.references import (
    CancellationRefV1, ProviderRunRefV1, StagePredecessorV2,
)
from tuner.execution.providers.modal.coordinator_effects import (
    ModalEffectOutcome,
    ModalFoundationEffectExecutor,
    ModalFoundationExecutorResolver,
    ModalFoundationReconciliationAdapter,
)
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1
from tests.execution.foundation_v2.helpers import StrongVerifier


class Catalog:
    def __init__(self, binding):
        self.binding = binding

    def resolve(self, digest):
        return self.binding


class Authority:
    def __init__(self, allowed=True, expected=None):
        self.allowed = allowed
        self.expected = expected

    def authenticate(self, binding):
        return (self.allowed and type(binding) is ModalCommandBinding
                and (self.expected is None or binding.canonical_bytes == self.expected))


class Transport:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def execute_once(self, binding, command):
        self.calls.append((binding, command))
        return self.outcome

    def lookup_once(self, binding, command):
        raise AssertionError("not used")


def stage_case():
    adapter, context, plan, execution = composed()
    prep = adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), execution)
    payload = adapter.payload(prep, EffectKind.STAGE)
    command = build_stage_command(prep, "nonce-a", payload, execution.executor_descriptor)
    client = inputs()["binding"]
    values = inputs()
    selection = ModalDeploymentSelectionV1.from_profile(
        values["profile"], binding=client,
        runtime_environment=values["runtime_environment"],
        timeout_seconds=values["timeout_seconds"],
    )
    deployment = verified(selection)
    binding = ModalCommandBinding(
        command.canonical_bytes, adapter.snapshot(),
        canonical_bytes(deployment.to_dict()),
    )
    request = ExecutionResolutionRequestV2(
        command.digest, command.executor.digest, "modal", context.provider.profile_ref,
        execution.scope.account_ref, execution.scope.namespace_ref, "stage",
        "stage-payload/v2", prep.workload_digest,
    )
    return command, payload, request, binding


def executor(binding, transport, authority=None):
    return ModalFoundationEffectExecutor(
        profile_ref=binding.profile_ref, account_ref=binding.account_ref,
        namespace_ref=binding.namespace_ref, catalog=Catalog(binding),
        authority=authority or Authority(), transport=transport,
    )


def test_stage_uses_authenticated_exact_command_and_calls_transport_once():
    command, payload, request, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.FOUND, "stage-a"))
    observation = executor(binding, transport).execute_once(payload, request)
    assert observation.stage_ref.stage_ref == "stage-a"
    assert observation.command_digest == command.digest
    assert len(transport.calls) == 1
    assert transport.calls[0][0] == binding
    assert transport.calls[0][1].canonical_bytes == command.canonical_bytes


def test_authentication_and_command_substitution_stop_before_transport():
    command, payload, request, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    authority = Authority()
    authority.allowed = False
    with pytest.raises(FoundationError):
        executor(binding, transport, authority).execute_once(payload, request)
    assert transport.calls == []
    authority.allowed = True
    parsed = command.preparation
    changed_command = build_stage_command(
        parsed, "other-nonce", command.payload, command.executor,
    )
    changed = ModalCommandBinding(
        changed_command.canonical_bytes, binding.preparation_snapshot,
        binding.deployment_bytes,
    )
    with pytest.raises((AssertionError, FoundationError)):
        executor(changed, transport, authority).execute_once(payload, request)
    assert transport.calls == []


def test_client_workspace_tuple_cannot_be_relabelled_as_foundation_namespace():
    _, payload, request, binding = stage_case()
    alien_selection = replace(
        binding.deployment.selection, workspace_ref="alien-workspace"
    )
    # The exact binding itself rejects a deployment that differs from the
    # independently retained preparation snapshot.
    with pytest.raises(ValueError):
        ModalCommandBinding(
            binding.command_bytes, binding.preparation_snapshot,
            canonical_bytes(verified(alien_selection).to_dict()),
        )


def test_authority_must_authenticate_complete_reconstructed_content():
    command, payload, request, binding = stage_case()
    changed_command = build_stage_command(
        command.preparation, "other-nonce", command.payload, command.executor,
    )
    changed = ModalCommandBinding(
        changed_command.canonical_bytes, binding.preparation_snapshot,
        binding.deployment_bytes,
    )
    transport = Transport(ModalEffectOutcome(ObservationDisposition.FOUND, "stage-a"))
    authority = Authority(expected=binding.canonical_bytes)
    with pytest.raises(FoundationError):
        executor(changed, transport, authority).execute_once(payload, request)
    assert transport.calls == []


def test_direct_executor_rechecks_own_scope_and_descriptor_before_transport():
    _, payload, request, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.FOUND, "stage-a"))
    wrong = ModalFoundationEffectExecutor(
        profile_ref="other-profile", account_ref=binding.account_ref,
        namespace_ref=binding.namespace_ref, catalog=Catalog(binding),
        authority=Authority(), transport=transport,
    )
    with pytest.raises(FoundationError):
        wrong.execute_once(payload, request)
    assert transport.calls == []

def test_indeterminate_provider_boundary_is_preserved_without_retry():
    _, payload, request, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    observation = executor(binding, transport).execute_once(payload, request)
    assert observation.disposition is ObservationDisposition.INDETERMINATE
    assert len(transport.calls) == 1


def test_weak_lookup_cannot_be_upgraded_to_definite_absence():
    with pytest.raises(ValueError, match="finality proof"):
        ModalEffectOutcome(ObservationDisposition.DEFINITELY_ABSENT)


@pytest.mark.parametrize("kind", [EffectKind.SUBMIT, EffectKind.CANCEL])
def test_submit_and_cancel_return_only_the_matching_typed_reference(kind):
    stage, _, _, stage_binding = stage_case()
    adapter, _, plan, execution = composed()
    prep = adapter.prepare(plan, TrainingRunRef("run-a", "project-a"), execution)
    payload = adapter.payload(prep, kind)
    if kind is EffectKind.SUBMIT:
        predecessor = StagePredecessorV2(
            "modal", prep.provider.profile_ref, prep.scope.account_ref,
            prep.scope.namespace_ref, prep.project_ref, prep.run_id,
            prep.plan_fingerprint, prep.preparation_digest, prep.workload_digest,
            stage.operation.effect.effect_id, "b" * 64, "c" * 64,
        )
        command = build_submit_command(
            prep, "nonce-submit", payload, execution.executor_descriptor, predecessor
        )
        provider_ref = "job-a"
    else:
        provider_ref = "job-a"
        command = build_cancel_command(
            prep, "nonce-cancel", payload, execution.executor_descriptor,
            CancellationRefV1(ProviderRunRefV1(provider_ref), "d" * 64),
        )
    binding = ModalCommandBinding(
        command.canonical_bytes, stage_binding.preparation_snapshot,
        stage_binding.deployment_bytes,
    )
    request = ExecutionResolutionRequestV2(
        command.digest, command.executor.digest, "modal", prep.provider.profile_ref,
        prep.scope.account_ref, prep.scope.namespace_ref, kind.value,
        command.payload.payload_kind, prep.workload_digest,
    )
    transport = Transport(ModalEffectOutcome(ObservationDisposition.FOUND, provider_ref))
    observation = executor(binding, transport).execute_once(payload, request)
    if kind is EffectKind.SUBMIT:
        assert observation.provider_run.provider_job_ref == provider_ref
        assert observation.cancellation is None
    else:
        assert observation.cancellation.run.provider_job_ref == provider_ref
        assert observation.provider_run is None


def test_reconciliation_uses_target_epoch_and_rejects_command_byte_substitution():
    command, _, _, binding = stage_case()

    class LookupTransport(Transport):
        def lookup_once(self, retained, parsed):
            self.calls.append((retained, parsed))
            return self.outcome

    transport = LookupTransport(
        ModalEffectOutcome(ObservationDisposition.FOUND, "stage-a")
    )
    adapter = ModalFoundationReconciliationAdapter(
        profile_ref=binding.profile_ref, account_ref=binding.account_ref,
        namespace_ref=binding.namespace_ref, catalog=Catalog(binding),
        authority=Authority(), transport=transport,
    )
    target = ReconciliationTargetV1(
        command.canonical_bytes, command.digest, command.operation.effect.effect_id,
        "owner-a", 1, 7, 1, "e" * 64,
    )
    observation = adapter.lookup(target, command.preparation)
    assert observation.result_epoch == 7
    assert observation.resolution_digest == "e" * 64
    assert len(transport.calls) == 1
    changed = ReconciliationTargetV1(
        command.canonical_bytes + b" ", command.digest,
        command.operation.effect.effect_id, "owner-a", 1, 7, 1, "e" * 64,
    )
    with pytest.raises(FoundationError):
        adapter.lookup(changed, command.preparation)
    assert len(transport.calls) == 1
    wrong_scope = ModalFoundationReconciliationAdapter(
        profile_ref=binding.profile_ref, account_ref="other-account",
        namespace_ref=binding.namespace_ref, catalog=Catalog(binding),
        authority=Authority(), transport=transport,
    )
    with pytest.raises(FoundationError):
        wrong_scope.lookup(target, command.preparation)
    assert len(transport.calls) == 1


def test_real_foundation_broker_resolver_is_one_shot_and_preserves_ambiguity():
    command, _, _, binding = stage_case()
    transport = Transport(ModalEffectOutcome(ObservationDisposition.FOUND, "stage-a"))
    resolved = executor(binding, transport)
    grants = GrantAuthorityV2("grants", b"g" * 32)
    receipts = ReceiptAuthorityV2("receipts", b"r" * 32)
    invalid = InvalidEvidenceAuthorityV2("invalid", b"i" * 32)
    finality = StrongVerifier()
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, finality, finality, grants
    )
    broker = EffectBrokerV2(
        repository, ModalFoundationExecutorResolver(resolved), grants, receipts, invalid
    )
    grant = grants.issue(
        command.canonical_bytes, grant_ref="grant-a", policy_digest="1" * 64,
        requirement_digest="2" * 64, not_before_epoch=100, expires_at_epoch=200,
    )
    first = broker.execute(command.canonical_bytes, grant, now_epoch=150)
    second = broker.execute(command.canonical_bytes, grant, now_epoch=150)
    assert first == second and first.state is EffectState.FOUND
    assert len(transport.calls) == 1

    class Ambiguous(Transport):
        def execute_once(self, binding, parsed):
            self.calls.append((binding, parsed))
            raise RuntimeError("provider boundary crossed")

    ambiguous = Ambiguous(ModalEffectOutcome(ObservationDisposition.INDETERMINATE))
    other_grants = GrantAuthorityV2("other-grants", b"G" * 32)
    other_receipts = ReceiptAuthorityV2("other-receipts", b"R" * 32)
    other_invalid = InvalidEvidenceAuthorityV2("other-invalid", b"I" * 32)
    other_repository = InMemoryEffectRepositoryV2(
        other_receipts, other_invalid, finality, finality, other_grants
    )
    other = EffectBrokerV2(
        other_repository,
        ModalFoundationExecutorResolver(executor(binding, ambiguous)),
        other_grants, other_receipts, other_invalid,
    )
    other_grant = other_grants.issue(
        command.canonical_bytes, grant_ref="grant-b", policy_digest="1" * 64,
        requirement_digest="2" * 64, not_before_epoch=100, expires_at_epoch=200,
    )
    with pytest.raises(FoundationError, match="effect_ambiguous"):
        other.execute(command.canonical_bytes, other_grant, now_epoch=150)
    assert len(ambiguous.calls) == 1
