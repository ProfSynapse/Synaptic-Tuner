from dataclasses import replace

import pytest

from tuner.execution.foundation_v2.authority import ReconciliationGrantContentV1
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.commands import build_submit_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.foundation_v2.observations import ObservationDisposition, ProviderObservationV1
from tuner.execution.foundation_v2.receipts import ReceiptContentV1
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import DispatchState, EffectState, InMemoryEffectRepositoryV2
from tuner.execution.foundation_v2.references import ProviderStageRefV1, StagePredecessorV2

from .helpers import *


def _completed_stage():
    command = stage_command()
    executor = bound_executor_local(command, Executor())
    repo, authority, receipts, verifier, _ = environment(executor)
    record = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    prep_value = command.preparation
    predecessor = StagePredecessorV2(
        prep_value.provider.provider_id, prep_value.provider.profile_ref,
        prep_value.scope.account_ref, prep_value.scope.namespace_ref,
        prep_value.project_ref, prep_value.run_id, prep_value.plan_fingerprint,
        prep_value.preparation_digest, prep_value.workload_digest,
        command.operation.effect.effect_id,
        record.results[0].authenticated_receipt_digest, record.record_digest,
    )
    return command, record, predecessor, repo, authority, receipts, verifier


def bound_executor_local(command, executor):
    original = executor.execute_once

    def execute(payload_value, request):
        return replace(original(payload_value, request), effect_id=command.operation.effect.effect_id)

    executor.execute_once = execute
    return executor


def reconciliation_grant_local(authority, command, adapter, *, grant_ref="reconcile", owner="owner", generation=1, epoch=1):
    p = command.preparation
    content = ReconciliationGrantContentV1(
        grant_ref, command.digest, command.operation.effect.effect_id,
        p.preparation_digest, adapter.digest, p.provider.provider_id,
        p.provider.profile_ref, p.scope.account_ref, p.scope.namespace_ref,
        owner, generation, epoch, D[9], D[10], 100, 200,
        authority.epoch, authority.revocation_generation,
    )
    return authority.issue_reconciliation(content)


def test_submit_predecessor_cas_rejects_cross_scope_and_cross_run_reuse():
    _, _, predecessor, repo, authority, receipts, _ = _completed_stage()
    for changed in (
        replace(predecessor, namespace_ref="other"),
        replace(predecessor, run_id="other-run"),
    ):
        p = prep()
        command = build_submit_command(p, "nonce", payload(EffectKind.SUBMIT, p), descriptor(), changed)
        executor = bound_executor_local(command, Executor())
        broker = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts)
        with pytest.raises(FoundationError) as error:
            broker.execute(command.canonical_bytes, execution_grant(authority, command, "submit-" + changed.namespace_ref + changed.run_id), now_epoch=150)
        assert error.value.code is DiagnosticCode.BINDING_MISMATCH
        assert executor.calls == 0


@pytest.mark.parametrize("wrong_ref", [
    ProviderStageRefV1("other", "local", "acct", "ns", "stage-output"),
    ProviderStageRefV1("docker", "other", "acct", "ns", "stage-output"),
])
def test_effect_specific_observation_rejected_before_relinquished_result(wrong_ref):
    command = stage_command()

    class WrongExecutor(Executor):
        def execute_once(self, payload_value, request):
            self.calls += 1
            return ProviderObservationV1(
                command.operation.effect.effect_id, command.digest, command.executor.digest,
                ObservationDisposition.FOUND, request.digest, 1, stage_ref=wrong_ref,
            )

    executor = WrongExecutor()
    repo, authority, receipts, _, _ = environment(executor)
    broker = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts)
    with pytest.raises(FoundationError) as error:
        broker.execute(command.canonical_bytes, execution_grant(authority, command), now_epoch=150)
    record = repo.get(command.operation.effect.effect_id)
    assert error.value.code is DiagnosticCode.EVIDENCE_INVALID
    assert record.dispatch is DispatchState.RELINQUISHED
    assert record.state is EffectState.INDETERMINATE
    assert record.results == ()
    assert record.invalid_codes == (DiagnosticCode.EVIDENCE_INVALID,)


def test_distinct_authenticated_found_content_contradicts_exact_duplicate_noops():
    command = stage_command()
    executor = bound_executor_local(command, Executor())
    repo, authority, receipts, _, _ = environment(executor)
    record = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    conflicting = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[11], 1,
        stage_ref=ProviderStageRefV1("docker", "local", "acct", "ns", "different-stage"),
    )
    receipt = receipts.issue(dispatch_receipt_content(command, conflicting, record))
    record = repo.append_result(command.operation.effect.effect_id, receipt, None, now_epoch=150)
    assert record.state is EffectState.CONTRADICTED
    assert len(record.results) == 2
    assert repo.append_result(command.operation.effect.effect_id, receipt, None, now_epoch=150) is record


def test_cross_provider_runtime_executor_rejected_before_admission():
    command = stage_command()
    executor = Executor(provider_id="other")
    repo, authority, receipts, _, _ = environment(executor)
    broker = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts)
    with pytest.raises(FoundationError) as error:
        broker.execute(command.canonical_bytes, execution_grant(authority, command), now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH
    assert repo.get(command.operation.effect.effect_id) is None
    assert executor.calls == 0


class _BadFinalityVerifier:
    def __init__(self, result):
        self.result = result

    def verify_finality(self, proof, record, receipt, *, now_epoch):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_finality_requires_exact_true_and_verifier_error_preserves_receipt():
    for verifier_result in (1, RuntimeError("secret-verifier-body")):
        command = stage_command()
        executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
        base_repo, authority, receipts, recovery, _ = environment(executor)
        repo = InMemoryEffectRepositoryV2(receipts, recovery, _BadFinalityVerifier(verifier_result), authority)
        broker = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts)
        broker.execute(command.canonical_bytes, execution_grant(authority, command), now_epoch=150)
        current = repo.get(command.operation.effect.effect_id)
        proof = recovery.proof(current, "final_absent")
        observation = observation_for(
            command, ObservationDisposition.DEFINITELY_ABSENT,
            finality_proof=proof,
        )
        receipt = receipts.issue(dispatch_receipt_content(command, observation, current))
        record = repo.append_result(command.operation.effect.effect_id, receipt, proof, now_epoch=150)
        assert receipt in record.results
        assert record.state is EffectState.INDETERMINATE
        assert DiagnosticCode.FINALITY_UNPROVEN in record.invalid_codes
        assert "secret" not in " ".join(code.value for code in record.invalid_codes)


def test_grant_content_is_structurally_validated_and_reconciliation_revokes():
    command = stage_command()
    _, authority, _, _, _ = environment(Executor())
    grant = execution_grant(authority, command)
    with pytest.raises(ValueError):
        replace(grant.content, effect_kind="submit")
    adapter = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    content = ReconciliationGrantContentV1(
        "reconcile", command.digest, command.operation.effect.effect_id,
        command.preparation.preparation_digest, adapter.digest, "docker", "local",
        "acct", "ns", "owner", 1, 1, D[9], D[10], 100, 200,
        authority.epoch, authority.revocation_generation,
    )
    recon_grant = authority.issue_reconciliation(content)
    assert authority.verify_reconciliation(recon_grant, now_epoch=150)
    authority.revoke(content.grant_ref)
    assert not authority.verify_reconciliation(recon_grant, now_epoch=150)


def test_active_reconciliation_transfer_rejected_even_with_valid_proof():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, verifier, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    content = ReconciliationGrantContentV1(
        "owner-one", command.digest, command.operation.effect.effect_id,
        command.preparation.preparation_digest, adapter.digest, "docker", "local",
        "acct", "ns", "owner", 1, 1, D[9], D[10], 100, 200,
        authority.epoch, authority.revocation_generation,
    )
    grant = authority.issue_reconciliation(content)
    record, claim, _ = repo.acquire_reconciliation(command.canonical_bytes, grant, now_epoch=150)
    transfer = authority.issue_reconciliation(replace(
        content, grant_ref="owner-two", owner_ref="other", ownership_epoch=2,
    ))
    proof = verifier.proof(record, "quiescent")
    with pytest.raises(FoundationError) as error:
        repo.transfer_reconciliation(command.canonical_bytes, transfer, proof=proof, now_epoch=150)
    assert error.value.code is DiagnosticCode.RECONCILIATION_CONFLICT
    assert repo.get(command.operation.effect.effect_id).reconciliation == claim


def test_cancel_found_requires_exact_target_and_reason():
    command = cancel_command()
    executor = bound_executor_local(command, Executor())
    repo, authority, receipts, _, _ = environment(executor)
    record = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    assert record.state is EffectState.FOUND

    other = cancel_command(prep(run_id="other-run"))

    class WrongCancellation(Executor):
        def execute_once(self, payload_value, request):
            self.calls += 1
            return ProviderObservationV1(
                other.operation.effect.effect_id, other.digest, other.executor.digest,
                ObservationDisposition.FOUND, request.digest, 1,
                cancellation=CancellationRefV1(ProviderRunRefV1("job-1"), D[9]),
            )

    wrong = WrongCancellation()
    wrong_repo, wrong_authority, wrong_receipts, _, _ = environment(wrong)
    with pytest.raises(FoundationError) as error:
        EffectBrokerV2(wrong_repo, ExecutorResolver(wrong), wrong_authority, wrong_receipts).execute(
            other.canonical_bytes, execution_grant(wrong_authority, other), now_epoch=150,
        )
    assert error.value.code is DiagnosticCode.EVIDENCE_INVALID


def test_cross_provider_runtime_adapter_rejected_before_claim():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter = Adapter(observation_for(command), provider_id="other")
    descriptor_value = adapter.descriptor
    grant = reconciliation_grant_local(authority, command, descriptor_value)
    service = ReconciliationServiceV1(repo, authority, AdapterResolver(adapter), receipts)
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, grant, now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH
    assert repo.get(command.operation.effect.effect_id).reconciliation is None
    assert adapter.calls == 0


class _TruthyRecovery:
    def verify_quiescence(self, proof, record, *, now_epoch):
        return 1


def test_recovery_proof_requires_exact_boolean_true():
    command = stage_command()
    executor = Executor(fail=True)
    _, authority, receipts, finality, _ = environment(executor)
    repo = InMemoryEffectRepositoryV2(receipts, _TruthyRecovery(), finality, authority)
    broker = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts)
    with pytest.raises(FoundationError):
        broker.execute(command.canonical_bytes, execution_grant(authority, command), now_epoch=150)
    with pytest.raises(FoundationError) as error:
        repo.prove_quiescence(command.operation.effect.effect_id, object(), now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID


def test_reconciliation_wrong_result_epoch_is_invalid_and_cannot_complete_claim():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )

    class WrongEpochAdapter(Adapter):
        def lookup(self, target, preparation):
            self.calls += 1
            return self.observation

    adapter = WrongEpochAdapter(observation_for(command, result_epoch=9))
    grant = reconciliation_grant_local(authority, command, adapter.descriptor)
    service = ReconciliationServiceV1(repo, authority, AdapterResolver(adapter), receipts)
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, grant, now_epoch=150)
    record = repo.get(command.operation.effect.effect_id)
    assert error.value.code is DiagnosticCode.EVIDENCE_INVALID
    assert record.reconciliation.active is False
    assert len(record.results) == 1


def test_reconciliation_first_generation_is_repository_derived_and_next_indeterminate_generation_runs_once():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter = Adapter(observation_for(command, ObservationDisposition.INDETERMINATE))
    service = ReconciliationServiceV1(repo, authority, AdapterResolver(adapter), receipts)
    invalid_first = reconciliation_grant_local(
        authority, command, adapter.descriptor, grant_ref="bad-first",
        generation=2, epoch=2,
    )
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, invalid_first, now_epoch=150)
    assert error.value.code is DiagnosticCode.RECONCILIATION_CONFLICT
    assert adapter.calls == 0

    first = reconciliation_grant_local(authority, command, adapter.descriptor, grant_ref="generation-one")
    record = service.reconcile(command.canonical_bytes, first, now_epoch=150)
    assert record.state is EffectState.INDETERMINATE
    assert record.reconciliation.completed and adapter.calls == 1
    second = reconciliation_grant_local(
        authority, command, adapter.descriptor, grant_ref="generation-two",
        generation=2, epoch=2,
    )
    record = service.reconcile(command.canonical_bytes, second, now_epoch=151)
    assert record.state is EffectState.INDETERMINATE
    assert record.reconciliation.generation == 2
    assert record.reconciliation.ownership_epoch == 2
    assert record.reconciliation.completed and adapter.calls == 2


def test_terminal_reconciliation_rejects_even_new_generation_before_lookup():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter = Adapter(observation_for(command))
    service = ReconciliationServiceV1(repo, authority, AdapterResolver(adapter), receipts)
    service.reconcile(
        command.canonical_bytes,
        reconciliation_grant_local(authority, command, adapter.descriptor, grant_ref="terminal-one"),
        now_epoch=150,
    )
    second = reconciliation_grant_local(
        authority, command, adapter.descriptor, grant_ref="terminal-two",
        generation=2, epoch=2,
    )
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, second, now_epoch=151)
    assert error.value.code is DiagnosticCode.EFFECT_INELIGIBLE
    assert adapter.calls == 1


def test_late_old_owner_receipt_is_retained_stale_and_conflicts_with_new_owner_outcome():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, verifier, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter_descriptor = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    first_grant = reconciliation_grant_local(authority, command, adapter_descriptor, grant_ref="old-owner")
    record, first_claim, _ = repo.acquire_reconciliation(command.canonical_bytes, first_grant, now_epoch=150)
    stopped = repo.interrupt_reconciliation(command.operation.effect.effect_id, first_claim)
    transfer_grant = reconciliation_grant_local(
        authority, command, adapter_descriptor, grant_ref="new-owner",
        owner="other-owner", generation=1, epoch=2,
    )
    proof = verifier.proof(repo.get(command.operation.effect.effect_id), "quiescent")
    _, new_claim = repo.transfer_reconciliation(
        command.canonical_bytes, transfer_grant, proof=proof, now_epoch=151,
    )

    late = observation_for(command, result_epoch=1)
    late_receipt = receipts.issue(reconciliation_receipt_content(late, stopped))
    record = repo.append_result(command.operation.effect.effect_id, late_receipt, None, now_epoch=151)
    assert late_receipt.authenticated_receipt_digest == record.results[-1].authenticated_receipt_digest
    assert DiagnosticCode.STALE_RESULT in record.invalid_codes
    assert record.state is EffectState.INDETERMINATE

    current = ProviderObservationV1(
        command.operation.effect.effect_id, command.digest, command.executor.digest,
        ObservationDisposition.FOUND, D[12], 2,
        stage_ref=ProviderStageRefV1("docker", "local", "acct", "ns", "new-stage"),
    )
    current_receipt = receipts.issue(reconciliation_receipt_content(current, new_claim))
    record = repo.complete_reconciliation(
        command.operation.effect.effect_id, new_claim, current_receipt, None, now_epoch=151,
    )
    assert record.state is EffectState.CONTRADICTED
    assert len(record.results) == 3


def test_same_terminal_semantics_with_different_provenance_is_compatible():
    command = stage_command()
    executor = bound_executor_local(command, Executor())
    repo, authority, receipts, _, _ = environment(executor)
    record = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    same_outcome = observation_for(command, resolution_digest=D[12])
    receipt = receipts.issue(dispatch_receipt_content(command, same_outcome, record))
    record = repo.append_result(command.operation.effect.effect_id, receipt, None, now_epoch=151)
    assert record.state is EffectState.FOUND
    assert len(record.results) == 2
    assert len(record.terminal_content_digests) == 1


def test_receipt_envelopes_are_owned_canonical_values_and_malformed_signed_found_is_rejected():
    command = stage_command()
    executor = bound_executor_local(command, Executor())
    repo, authority, receipts, _, _ = environment(executor)
    record = EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    observation = observation_for(command)
    content = dispatch_receipt_content(command, observation, record)
    issued = receipts.issue(content)
    with pytest.raises(Exception):
        content.stage_ref.stage_ref = "mutated"
    assert issued.content is not content

    malformed_observation = replace(
        observation,
        stage_ref=ProviderStageRefV1("other", "local", "acct", "ns", "stage-output"),
    )
    malformed = receipts.issue(dispatch_receipt_content(command, malformed_observation, record))
    before = len(record.results)
    with pytest.raises(FoundationError) as error:
        repo.append_result(command.operation.effect.effect_id, malformed, None, now_epoch=151)
    assert error.value.code is DiagnosticCode.EVIDENCE_INVALID
    assert len(repo.get(command.operation.effect.effect_id).results) == before


def test_receipt_authority_key_and_reconciliation_service_outer_types_are_closed():
    for key in ("x" * 32, b"short"):
        with pytest.raises(ValueError):
            ReceiptAuthorityV1("receipts", key)
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    service = ReconciliationServiceV1(repo, authority, AdapterResolver(Adapter(observation_for(command))), receipts)
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, object(), now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID
    valid = reconciliation_grant_local(authority, command, AdapterDescriptorV1("docker", "lookup", "1.0.0"))
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, valid, now_epoch=150, resume=object())
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID

    class BadResolver:
        def resolve(self, request):
            return object()

    service = ReconciliationServiceV1(repo, authority, BadResolver(), receipts)
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, valid, now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH


def test_same_absence_semantics_with_different_valid_proofs_are_compatible():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))

    class AcceptFinality:
        def verify_finality(self, proof, record, receipt, *, now_epoch):
            return True

    repo, authority, receipts, recovery, _ = environment(executor)
    repo = InMemoryEffectRepositoryV2(receipts, recovery, AcceptFinality(), authority)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )

    class ProofValue:
        def __init__(self, digest):
            self.proof_digest = digest

    for digest in (D[12], D[13]):
        proof = ProofValue(digest)
        observation = observation_for(
            command, ObservationDisposition.DEFINITELY_ABSENT,
            finality_proof=proof,
        )
        record = repo.get(command.operation.effect.effect_id)
        receipt = receipts.issue(dispatch_receipt_content(command, observation, record))
        record = repo.append_result(command.operation.effect.effect_id, receipt, proof, now_epoch=150)
    assert record.state is EffectState.DEFINITELY_ABSENT
    assert len(record.terminal_content_digests) == 1
    assert record.results[-1].content.finality_proof_digest != record.results[-2].content.finality_proof_digest


@pytest.mark.parametrize("resolver_value", [RuntimeError("secret-resolver-body"), object()])
def test_broker_resolver_boundary_is_closed_before_admission(resolver_value):
    command = stage_command()
    executor = Executor()
    repo, authority, receipts, _, _ = environment(executor)

    class BadResolver:
        def resolve(self, request):
            if isinstance(resolver_value, Exception):
                raise resolver_value
            return resolver_value

    broker = EffectBrokerV2(repo, BadResolver(), authority, receipts)
    with pytest.raises(FoundationError) as error:
        broker.execute(command.canonical_bytes, execution_grant(authority, command), now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH
    assert "secret" not in str(error.value)
    assert repo.get(command.operation.effect.effect_id) is None


def test_reconciliation_completion_malformed_result_never_returns_success_shape():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )

    class MalformedCompletionRepository:
        def __getattr__(self, name):
            return getattr(repo, name)

        def complete_reconciliation(self, *args, **kwargs):
            repo.complete_reconciliation(*args, **kwargs)
            return object()

    adapter = Adapter(observation_for(command))
    service = ReconciliationServiceV1(
        MalformedCompletionRepository(), authority, AdapterResolver(adapter), receipts,
    )
    grant = reconciliation_grant_local(authority, command, adapter.descriptor)
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, grant, now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH


def test_repository_acquire_reconciliation_outer_values_are_closed():
    command = stage_command()
    executor = bound_executor_local(command, Executor(ObservationDisposition.INDETERMINATE))
    repo, authority, receipts, _, _ = environment(executor)
    EffectBrokerV2(repo, ExecutorResolver(executor), authority, receipts).execute(
        command.canonical_bytes, execution_grant(authority, command), now_epoch=150,
    )
    adapter = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    valid = reconciliation_grant_local(authority, command, adapter)
    cases = (
        (bytearray(command.canonical_bytes), valid, DiagnosticCode.AUTHORITY_INVALID),
        (b"not-canonical", valid, DiagnosticCode.BINDING_MISMATCH),
        (command.canonical_bytes, object(), DiagnosticCode.AUTHORITY_INVALID),
    )
    for raw, grant, code in cases:
        with pytest.raises(FoundationError) as error:
            repo.acquire_reconciliation(raw, grant, now_epoch=150)
        assert error.value.code is code
    object.__setattr__(valid, "content", object())
    with pytest.raises(FoundationError) as error:
        repo.acquire_reconciliation(command.canonical_bytes, valid, now_epoch=150)
    assert error.value.code is DiagnosticCode.AUTHORITY_INVALID

    class BadClaimRepository:
        def acquire_reconciliation(self, *args, **kwargs):
            return object(), object(), True

    service = ReconciliationServiceV1(
        BadClaimRepository(), authority,
        AdapterResolver(Adapter(observation_for(command))), receipts,
    )
    fresh_valid = reconciliation_grant_local(
        authority, command, adapter, grant_ref="fresh-valid",
    )
    with pytest.raises(FoundationError) as error:
        service.reconcile(command.canonical_bytes, fresh_valid, now_epoch=150)
    assert error.value.code is DiagnosticCode.BINDING_MISMATCH


def test_repository_acquire_missing_effect_is_closed_and_non_mutating():
    command = stage_command()
    repo, authority, _, _, _ = environment(Executor())
    adapter = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    grant = reconciliation_grant_local(authority, command, adapter)
    with pytest.raises(FoundationError) as error:
        repo.acquire_reconciliation(command.canonical_bytes, grant, now_epoch=150)
    assert error.value.code is DiagnosticCode.EFFECT_INELIGIBLE
    assert command.operation.effect.effect_id not in str(error.value)
    assert repo.get(command.operation.effect.effect_id) is None
