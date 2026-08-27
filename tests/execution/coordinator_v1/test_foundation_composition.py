from dataclasses import replace

import pytest

from tuner.execution.coordinator_v1.foundation import (
    ComposedEffectFoundationV1,
    FoundationRecordAssessmentAuthorityV1,
    QuiescenceRecoveryRequestV1,
)
from tuner.execution.coordinator_v1.model import WorkflowPhaseV1
from tuner.execution.coordinator_v1.state_machine import (
    apply_stage_effect_record,
    begin_preparation,
    record_stage_intent,
)
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.authority import ReconciliationGrantContentV1
from tuner.execution.foundation_v2.canonical import DiagnosticCode, FoundationError
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import (
    DispatchState,
    ReconciliationGrantBindingV2,
    ReconciliationOwnershipV2,
)
from tuner.execution.foundation_v2.executors import AdapterDescriptorV1

from tests.execution.foundation_v2.helpers import (
    Adapter,
    AdapterResolver,
    Executor,
    D,
    environment,
    execution_grant,
    observation_for,
    stage_command,
)
from tests.execution.coordinator_v1.test_state_machine import (
    intent as coordinator_intent,
    planned as coordinator_planned,
    prep as coordinator_preparation,
)


class Clock:
    def now_iso(self):
        return "2026-08-27T00:00:00Z"


def bound_executor(command, executor):
    original = executor.execute_once

    def call(payload, request):
        return replace(
            original(payload, request),
            effect_id=command.operation.effect.effect_id,
        )

    executor.execute_once = call
    return executor


class TrustedEvidence:
    def __init__(self, repository, verifier):
        self.repository = repository
        self.verifier = verifier
        self.requests = []

    def obtain(self, request, *, now_epoch):
        self.requests.append((request, now_epoch))
        record = self.repository.get(request.effect_id)
        return self.verifier.proof(record, "quiescent", now_epoch)


def composed(*, disposition=ObservationDisposition.FOUND):
    command = stage_command()
    executor = bound_executor(command, Executor(disposition))
    repository, grants, receipts, invalid, verifier, resolver = environment(executor)
    broker = EffectBrokerV2(repository, resolver, grants, receipts, invalid)
    adapter = Adapter(observation_for(command, disposition))
    reconciliation = ReconciliationServiceV1(
        repository, grants, AdapterResolver(adapter), receipts, invalid
    )
    assessments = FoundationRecordAssessmentAuthorityV1(
        "assessment-authority",
        "assessment-key",
        b"a" * 32,
        assessor_ref="foundation-assessor",
        assessor_version="1.0.0",
        clock=Clock(),
        receipt_authority=receipts,
        invalid_evidence_authority=invalid,
        grant_authority=grants,
    )
    evidence = TrustedEvidence(repository, verifier)
    facade = ComposedEffectFoundationV1(
        repository,
        broker,
        reconciliation,
        grant_authority=grants,
        receipt_authority=receipts,
        invalid_evidence_authority=invalid,
        assessment_authority=assessments,
        trusted_quiescence_evidence=evidence,
    )
    return facade, command, repository, grants, receipts, invalid, verifier, assessments, evidence


def test_execute_get_and_assessment_are_durable_and_authenticated():
    facade, command, repository, grants, _, _, _, authority, _ = composed()
    record = facade.execute(
        command.canonical_bytes,
        execution_grant(grants, command),
        now_epoch=150,
    )
    assert record == repository.get(command.operation.effect.effect_id)
    assert facade.get(command.operation.effect.effect_id) == record
    assessment = facade.assess(record)
    assert facade.authenticate(assessment) is True
    assert assessment.content.foundation_record_digest == record.record_digest
    assert assessment.content.authenticated_receipt_digests == tuple(
        value.authenticated_receipt_digest for value in record.results
    )
    assert "a" * 32 not in repr(authority) and "<redacted>" in repr(authority)


def test_assessment_is_separate_key_canonical_and_tamper_closed():
    facade, command, _, grants, receipts, invalid, _, authority, _ = composed()
    record = facade.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    assessment = authority.assess(record)
    other = FoundationRecordAssessmentAuthorityV1(
        "assessment-authority", "assessment-key", b"b" * 32,
        assessor_ref="foundation-assessor", assessor_version="1.0.0", clock=Clock(),
        receipt_authority=receipts, invalid_evidence_authority=invalid,
        grant_authority=grants,
    )
    assert other.authenticate(assessment) is False
    assert authority.authenticate(replace(assessment, tag=D[0])) is False
    assert authority.authenticate(object()) is False

    for assessor_ref, assessor_version in (
        ("other-assessor", "1.0.0"),
        ("foundation-assessor", "2.0.0"),
    ):
        mismatched_identity = FoundationRecordAssessmentAuthorityV1(
            "assessment-authority", "assessment-key", b"a" * 32,
            assessor_ref=assessor_ref, assessor_version=assessor_version, clock=Clock(),
            receipt_authority=receipts, invalid_evidence_authority=invalid,
            grant_authority=grants,
        )
        assert mismatched_identity.authenticate(assessment) is False


def test_assessment_rejects_tampered_retained_record_and_truthy_auth_is_not_used():
    facade, command, _, grants, _, _, _, authority, _ = composed()
    record = facade.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    object.__setattr__(record, "dispatch_epoch", 2)
    with pytest.raises(FoundationError) as caught:
        authority.assess(record)
    assert caught.value.code is DiagnosticCode.AUTHORITY_INVALID


def test_recover_orphan_uses_only_trusted_internal_proof_and_exact_transition():
    facade, command, repository, grants, _, _, _, _, evidence = composed()
    grant = execution_grant(grants, command)
    record, admitted = repository.consume_attempt(command.canonical_bytes, grant, now_epoch=150)
    assert admitted
    repository.begin_dispatch(command.operation.effect.effect_id)
    orphaned = repository.orphan(command.operation.effect.effect_id)
    recovered = facade.recover_orphan(command.operation.effect.effect_id, now_epoch=150)
    assert recovered == replace(orphaned, dispatch=DispatchState.QUIESCENCE_PROVEN)
    request, epoch = evidence.requests[0]
    assert type(request) is QuiescenceRecoveryRequestV1 and epoch == 150
    assert request.foundation_record_digest == orphaned.record_digest
    assert request.executor_digest == command.executor.digest
    assert request.dispatch_state == DispatchState.ORPHANED_UNPROVEN.value
    assert command.canonical_bytes not in request.canonical_bytes
    assert not hasattr(request, "proof")

    request_digest = request.request_digest
    with pytest.raises(ValueError):
        replace(request, dispatch_state=DispatchState.QUIESCENCE_PROVEN.value)
    assert replace(request, executor_digest=D[0]).request_digest != request_digest

    recovered_again = facade.recover_orphan(command.operation.effect.effect_id, now_epoch=151)
    assert recovered_again is recovered
    assert len(evidence.requests) == 1


def test_recovery_not_found_or_untrusted_not_found_claim_is_insufficient():
    facade, command, repository, grants, receipts, invalid, _, assessments, _ = composed()
    with pytest.raises(FoundationError) as missing:
        facade.recover_orphan(command.operation.effect.effect_id, now_epoch=150)
    assert missing.value.code is DiagnosticCode.EFFECT_INELIGIBLE
    grant = execution_grant(grants, command)
    repository.consume_attempt(command.canonical_bytes, grant, now_epoch=150)
    repository.begin_dispatch(command.operation.effect.effect_id)
    repository.orphan(command.operation.effect.effect_id)

    class NotFoundOnly:
        def obtain(self, request, *, now_epoch):
            return object()

    closed = ComposedEffectFoundationV1(
        repository, facade._broker, facade._reconciliation,
        grant_authority=grants, receipt_authority=receipts,
        invalid_evidence_authority=invalid, assessment_authority=assessments,
        trusted_quiescence_evidence=NotFoundOnly(),
    )
    with pytest.raises(FoundationError) as rejected:
        closed.recover_orphan(command.operation.effect.effect_id, now_epoch=150)
    assert rejected.value.code is DiagnosticCode.AUTHORITY_INVALID


def test_get_and_recovery_reject_valid_record_returned_under_foreign_lookup_key():
    facade, command, repository, grants, _, _, _, _, evidence = composed()
    record = facade.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)

    class ForeignKeyRepository:
        def get(self, effect_id):
            return record

    facade._repo = ForeignKeyRepository()
    with pytest.raises(FoundationError) as direct:
        facade.get("foreign-effect")
    assert direct.value.code is DiagnosticCode.BINDING_MISMATCH
    with pytest.raises(FoundationError) as recovery:
        facade.recover_orphan("foreign-effect", now_epoch=150)
    assert recovery.value.code is DiagnosticCode.BINDING_MISMATCH
    assert evidence.requests == []


def test_recovery_converges_after_quiescence_write_loses_acknowledgement():
    facade, command, repository, grants, _, _, _, _, evidence = composed()
    repository.consume_attempt(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    repository.begin_dispatch(command.operation.effect.effect_id)
    orphaned = repository.orphan(command.operation.effect.effect_id)

    class AppliedThenRaisedRepository:
        def get(self, effect_id):
            return repository.get(effect_id)

        def prove_quiescence(self, effect_id, proof, *, now_epoch):
            repository.prove_quiescence(effect_id, proof, now_epoch=now_epoch)
            raise RuntimeError("lost acknowledgement")

    facade._repo = AppliedThenRaisedRepository()
    recovered = facade.recover_orphan(command.operation.effect.effect_id, now_epoch=150)
    assert recovered == replace(orphaned, dispatch=DispatchState.QUIESCENCE_PROVEN)
    assert len(evidence.requests) == 1
    assert facade.recover_orphan(command.operation.effect.effect_id, now_epoch=151) == recovered
    assert len(evidence.requests) == 1


def test_concrete_assessment_is_consumed_by_stage_reducer_without_translation():
    intent = coordinator_intent("stage")
    command = parse_exact_command(intent.canonical_command_bytes)
    executor = Executor(
        ObservationDisposition.FOUND,
        provider_id="provider-a",
        profile_ref="profile-a",
        account_ref="account-a",
        namespace_ref="namespace-a",
    )
    executor.descriptor = command.executor
    bound_executor(command, executor)
    repository, grants, receipts, invalid, _, resolver = environment(executor)
    broker = EffectBrokerV2(repository, resolver, grants, receipts, invalid)
    foundation_record = broker.execute(
        command.canonical_bytes,
        execution_grant(grants, command),
        now_epoch=150,
    )
    assessments = FoundationRecordAssessmentAuthorityV1(
        "assessment-authority", "assessment-key", b"a" * 32,
        assessor_ref="foundation-assessor", assessor_version="1.0.0", clock=Clock(),
        receipt_authority=receipts, invalid_evidence_authority=invalid,
        grant_authority=grants,
    )

    class FoundationAuthenticator:
        def authenticate_grant(self, grant, command_bytes):
            return grants.authenticate(grant, command_bytes)

        def authenticate_receipt(self, receipt):
            return receipts.verify(receipt)

        def authenticate_invalid_evidence(self, evidence):
            return invalid.verify(evidence)

    workflow = record_stage_intent(
        begin_preparation(coordinator_planned()),
        coordinator_preparation(),
        intent,
    )
    result = apply_stage_effect_record(
        workflow,
        foundation_record,
        assessments.assess(foundation_record),
        FoundationAuthenticator(),
        assessments,
    )
    assert result.phase is WorkflowPhaseV1.STAGED


def test_execute_rejects_return_reload_mismatch_and_raw_errors_are_closed():
    facade, command, repository, grants, receipts, invalid, _, assessments, evidence = composed()
    durable = facade.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    other, other_command, _, other_grants, _, _, _, _, _ = composed(
        disposition=ObservationDisposition.INDETERMINATE
    )
    stale = other.execute(
        other_command.canonical_bytes,
        execution_grant(other_grants, other_command),
        now_epoch=150,
    )

    class StaleBroker:
        def execute(self, command_bytes, grant, *, now_epoch):
            return stale

    mismatch = ComposedEffectFoundationV1(
        repository, StaleBroker(), facade._reconciliation,
        grant_authority=grants, receipt_authority=receipts,
        invalid_evidence_authority=invalid, assessment_authority=assessments,
        trusted_quiescence_evidence=evidence,
    )
    with pytest.raises(FoundationError) as caught:
        mismatch.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    assert caught.value.code is DiagnosticCode.AUTHORITY_INVALID

    class ThrowingBroker:
        def execute(self, command_bytes, grant, *, now_epoch):
            raise RuntimeError("secret-provider-object")

    throwing = ComposedEffectFoundationV1(
        repository, ThrowingBroker(), facade._reconciliation,
        grant_authority=grants, receipt_authority=receipts,
        invalid_evidence_authority=invalid, assessment_authority=assessments,
        trusted_quiescence_evidence=evidence,
    )
    with pytest.raises(FoundationError) as raw:
        throwing.execute(command.canonical_bytes, execution_grant(grants, command), now_epoch=150)
    assert raw.value.code is DiagnosticCode.AUTHORITY_INVALID
    assert "secret-provider-object" not in repr(raw.value)
    assert raw.value.__cause__ is None and raw.value.__context__ is None


def test_reconcile_forwards_exact_continuation_and_revalidates_durable_result():
    facade, command, repository, grants, receipts, invalid, _, assessments, evidence = composed(
        disposition=ObservationDisposition.INDETERMINATE
    )
    durable = facade.execute(
        command.canonical_bytes, execution_grant(grants, command), now_epoch=150
    )
    adapter = AdapterDescriptorV1("docker", "lookup", "1.0.0")
    content = ReconciliationGrantContentV1(
        "reconcile-grant", command.digest, command.operation.effect.effect_id,
        command.preparation.preparation_digest, adapter.digest,
        command.preparation.provider.provider_id,
        command.preparation.provider.profile_ref,
        command.preparation.scope.account_ref,
        command.preparation.scope.namespace_ref,
        "owner", 1, 1, D[9], D[10], 100, 200,
        grants.epoch, grants.revocation_generation,
    )
    grant = grants.issue_reconciliation(content)
    binding = ReconciliationGrantBindingV2(
        content.grant_ref, grant.authenticated_grant_digest,
        len(durable.receipt_admissions), None,
    )
    continuation = ReconciliationOwnershipV2(
        content.owner_ref, 1, 1, 150, D[11], content.grant_ref,
        grant.authenticated_grant_digest, (binding,), True, False,
    )

    class CapturingReconciliation:
        def __init__(self):
            self.continuation = None

        def reconcile(self, command_bytes, supplied, *, now_epoch, continuation=None):
            self.continuation = continuation
            return durable

    service = CapturingReconciliation()
    forwarding = ComposedEffectFoundationV1(
        repository, facade._broker, service,
        grant_authority=grants, receipt_authority=receipts,
        invalid_evidence_authority=invalid, assessment_authority=assessments,
        trusted_quiescence_evidence=evidence,
    )
    assert forwarding.reconcile(
        command.canonical_bytes, grant, now_epoch=150, continuation=continuation
    ) == durable
    assert service.continuation is continuation
    with pytest.raises(FoundationError) as malformed:
        forwarding.reconcile(
            command.canonical_bytes, grant, now_epoch=150, continuation=object()
        )
    assert malformed.value.code is DiagnosticCode.BINDING_MISMATCH


def test_facade_assessment_authenticator_requires_exact_true():
    facade, command, repository, grants, receipts, invalid, _, authority, evidence = composed()
    record = facade.execute(
        command.canonical_bytes, execution_grant(grants, command), now_epoch=150
    )
    assessment = authority.assess(record)

    class TruthyAssessmentAuthority:
        def assess(self, value):
            return assessment

        def authenticate(self, value):
            return object()

    composed_truthy = ComposedEffectFoundationV1(
        repository, facade._broker, facade._reconciliation,
        grant_authority=grants, receipt_authority=receipts,
        invalid_evidence_authority=invalid,
        assessment_authority=TruthyAssessmentAuthority(),
        trusted_quiescence_evidence=evidence,
    )
    assert composed_truthy.authenticate(assessment) is False


def test_composition_is_provider_neutral_and_unexported():
    import tuner.execution.coordinator_v1 as package
    import tuner.execution.coordinator_v1.foundation as module

    assert not hasattr(package, "ComposedEffectFoundationV1")
    source = open(module.__file__, encoding="utf-8").read().lower()
    assert "sqlite" not in source and "huggingface" not in source and "runpod" not in source
