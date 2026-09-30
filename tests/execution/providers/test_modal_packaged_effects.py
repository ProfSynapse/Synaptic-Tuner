"""Provider-free effect/reconciliation tests for the packaged Modal adapter."""

from __future__ import annotations

import pytest

from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.executors import ExecutionResolutionRequestV2
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)
from tuner.execution.foundation_v2.repository import (
    EffectState,
    InMemoryEffectRepositoryV2,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationTargetV1
from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
from tuner.execution.providers.modal.packaged_binding import (
    CommittedModalPackagedBindingCatalog,
)
from tuner.execution.providers.modal.packaged_effects import (
    ModalPackagedFoundationEffectExecutor,
    ModalPackagedFoundationExecutorResolver,
    ModalPackagedFoundationReconciliationAdapter,
)

from tests.execution.providers.test_modal_packaged_binding import _binding
from tests.execution.foundation_v2.helpers import StrongVerifier


class Authority:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls = []

    def authenticate(self, binding):
        self.calls.append(binding.authenticated_binding_digest)
        return self.allowed


class Transport:
    def __init__(self, outcome: ModalEffectOutcome) -> None:
        self.outcome = outcome
        self.execute_calls = []
        self.lookup_calls = []

    def execute_once(self, binding, command):
        self.execute_calls.append((binding, command))
        return self.outcome

    def lookup_once(self, binding, command):
        self.lookup_calls.append((binding, command))
        return self.outcome


def _request(command) -> ExecutionResolutionRequestV2:
    preparation = command.preparation
    return ExecutionResolutionRequestV2(
        command.digest, command.executor.digest,
        preparation.provider.provider_id, preparation.provider.profile_ref,
        preparation.scope.account_ref, preparation.scope.namespace_ref,
        command.operation.effect.kind.value, command.payload.payload_kind,
        command.payload.input_digest,
    )


def _executor(*, allowed=True, outcome=None):
    binding = _binding()
    authority = Authority(allowed)
    transport = Transport(
        outcome or ModalEffectOutcome(ObservationDisposition.FOUND, "stage-ref")
    )
    catalog = CommittedModalPackagedBindingCatalog(
        lambda digest: binding if digest == binding.command_digest else None,
    )
    command = binding.command
    executor = ModalPackagedFoundationEffectExecutor(
        profile_ref=command.preparation.provider.profile_ref,
        account_ref=command.preparation.scope.account_ref,
        namespace_ref=command.preparation.scope.namespace_ref,
        catalog=catalog, authority=authority, transport=transport,
    )
    return binding, command, executor, authority, transport, catalog


def test_authenticated_effect_calls_transport_exactly_once() -> None:
    binding, command, executor, authority, transport, _ = _executor()
    observation = executor.execute_once(command.payload, _request(command))
    assert observation.disposition is ObservationDisposition.FOUND
    assert observation.stage_ref.stage_ref == "stage-ref"
    assert authority.calls == [binding.authenticated_binding_digest]
    assert len(transport.execute_calls) == 1
    called_binding, called_command = transport.execute_calls[0]
    assert called_binding == binding
    assert called_command.canonical_bytes == command.canonical_bytes


def test_indeterminate_provider_boundary_is_preserved_without_retry() -> None:
    _, command, executor, _, transport, _ = _executor(
        outcome=ModalEffectOutcome(ObservationDisposition.INDETERMINATE),
    )
    observation = executor.execute_once(command.payload, _request(command))
    assert observation.disposition is ObservationDisposition.INDETERMINATE
    assert len(transport.execute_calls) == 1


@pytest.mark.parametrize("fault", ("authority", "request", "payload", "catalog"))
def test_binding_failures_stop_before_transport(fault: str) -> None:
    binding, command, executor, _, transport, _ = _executor(
        allowed=fault != "authority",
    )
    request = _request(command)
    payload = command.payload
    if fault == "request":
        request = ExecutionResolutionRequestV2(
            "0" * 64, request.descriptor_digest, request.provider_id,
            request.profile_ref, request.account_ref, request.namespace_ref,
            request.effect_kind, request.payload_schema, request.input_digest,
        )
    elif fault == "payload":
        payload = object()
    elif fault == "catalog":
        executor._catalog = CommittedModalPackagedBindingCatalog(lambda _: None)
    with pytest.raises(FoundationError) as caught:
        executor.execute_once(payload, request)
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert transport.execute_calls == []


def test_reconciliation_is_lookup_only_and_never_executes() -> None:
    binding, command, _, authority, transport, catalog = _executor()
    adapter = ModalPackagedFoundationReconciliationAdapter(
        profile_ref=command.preparation.provider.profile_ref,
        account_ref=command.preparation.scope.account_ref,
        namespace_ref=command.preparation.scope.namespace_ref,
        catalog=catalog, authority=authority, transport=transport,
    )
    target = ReconciliationTargetV1(
        command.canonical_bytes, command.digest,
        command.operation.effect.effect_id, "owner", 1, 1, 1, "a" * 64,
    )
    observation = adapter.lookup(target, command.preparation)
    assert observation.disposition is ObservationDisposition.FOUND
    assert len(transport.lookup_calls) == 1
    called_binding, called_command = transport.lookup_calls[0]
    assert called_binding == binding
    assert called_command.canonical_bytes == command.canonical_bytes
    assert transport.execute_calls == []


def test_reconciliation_substitution_does_not_reach_lookup() -> None:
    _, command, _, authority, transport, catalog = _executor()
    adapter = ModalPackagedFoundationReconciliationAdapter(
        profile_ref=command.preparation.provider.profile_ref,
        account_ref=command.preparation.scope.account_ref,
        namespace_ref=command.preparation.scope.namespace_ref,
        catalog=catalog, authority=authority, transport=transport,
    )
    target = ReconciliationTargetV1(
        command.canonical_bytes, command.digest,
        "stage-other", "owner", 1, 1, 1, "a" * 64,
    )
    with pytest.raises(FoundationError) as caught:
        adapter.lookup(target, command.preparation)
    assert caught.value.code is DiagnosticCode.BINDING_MISMATCH
    assert transport.lookup_calls == []


def test_one_foundation_grant_performs_one_effect_even_when_replayed() -> None:
    _, command, executor, _, transport, _ = _executor()
    grants = GrantAuthorityV2("packaged-grants", b"g" * 32)
    receipts = ReceiptAuthorityV2("packaged-receipts", b"r" * 32)
    invalid = InvalidEvidenceAuthorityV2("packaged-invalid", b"i" * 32)
    finality = StrongVerifier()
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, finality, finality, grants,
    )
    broker = EffectBrokerV2(
        repository, ModalPackagedFoundationExecutorResolver(executor),
        grants, receipts, invalid,
    )
    grant = grants.issue(
        command.canonical_bytes, grant_ref="packaged-grant",
        policy_digest="1" * 64, requirement_digest="2" * 64,
        not_before_epoch=100, expires_at_epoch=200,
    )
    first = broker.execute(command.canonical_bytes, grant, now_epoch=150)
    replay = broker.execute(command.canonical_bytes, grant, now_epoch=150)
    assert first == replay
    assert first.state is EffectState.FOUND
    assert len(transport.execute_calls) == 1
