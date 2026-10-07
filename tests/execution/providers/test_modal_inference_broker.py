"""Chat adapters use the real Foundation grant, predecessor and receipt boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.execution.foundation_v2.helpers import StrongVerifier, execution_grant
from tests.execution.foundation_v2.test_bounded_remediation import (
    reconciliation_grant_local,
)
from tests.execution.providers.test_modal_inference_commands import (
    Authority,
    Catalog,
    _bindings,
)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import FoundationError
from tuner.execution.foundation_v2.commands import (
    build_submit_command,
    parse_exact_command,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import (
    DispatchState,
    EffectState,
    InMemoryEffectRepositoryV2,
)
from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_effects import (
    ModalChatEffectExecutor,
    ModalChatExecutorResolver,
    ModalChatReconciliationAdapter,
    ModalChatReconciliationResolver,
)


class Transport:
    def __init__(self):
        self.executions = []
        self.lookups = []
        self.fail_execute = False
        self.fail_lookup = False
        self.disposition = ObservationDisposition.FOUND
        self.finality_proof = None

    def execute_once(self, binding, command):
        self.executions.append(command.digest)
        if self.fail_execute:
            raise RuntimeError("private-provider-response")
        ref = (
            "chat-object" if self.disposition is ObservationDisposition.FOUND else None
        )
        return ModalEffectOutcome(self.disposition, ref, self.finality_proof)

    def lookup_once(self, binding, command):
        self.lookups.append(command.digest)
        if self.fail_lookup:
            raise RuntimeError("private-provider-response")
        return ModalEffectOutcome(ObservationDisposition.FOUND, "chat-object")


def _case(monkeypatch):
    stage, submit, _ = _bindings(monkeypatch)
    command = parse_exact_command(stage.command_bytes)
    catalog = Catalog()
    catalog.values = {value.command_digest: value for value in (stage, submit)}
    authority = Authority([stage.canonical_bytes, submit.canonical_bytes])
    transport = Transport()
    prep = command.preparation
    kwargs = dict(
        profile_ref=prep.provider.profile_ref,
        account_ref=prep.scope.account_ref,
        namespace_ref=prep.scope.namespace_ref,
        implementation_version=command.executor.implementation_version,
        catalog=catalog,
        authority=authority,
        transport=transport,
    )
    executor = ModalChatEffectExecutor(**kwargs)
    adapter = ModalChatReconciliationAdapter(**kwargs)
    grants = GrantAuthorityV2("chat-grants", b"g" * 32)
    receipts = ReceiptAuthorityV2("chat-receipts", b"r" * 32)
    invalid = InvalidEvidenceAuthorityV2("chat-invalid", b"i" * 32)
    verifier = StrongVerifier()
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, verifier, verifier, grants
    )
    broker = EffectBrokerV2(
        repository, ModalChatExecutorResolver(executor), grants, receipts, invalid
    )
    service = ReconciliationServiceV1(
        repository, grants, ModalChatReconciliationResolver(adapter), receipts, invalid
    )
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("failure", ("foreign-authority", "expired", "wrong-command"))
def test_invalid_grant_cannot_reach_retention_or_transport(monkeypatch, failure):
    case = _case(monkeypatch)
    authority = (
        GrantAuthorityV2("foreign", b"f" * 32)
        if failure == "foreign-authority"
        else case.grants
    )
    command = (
        parse_exact_command(case.submit.command_bytes)
        if failure == "wrong-command"
        else case.command
    )
    grant = execution_grant(authority, command)
    with pytest.raises(FoundationError, match="authority_invalid"):
        case.broker.execute(
            case.stage.command_bytes,
            grant,
            now_epoch=201 if failure == "expired" else 150,
        )
    assert case.catalog.calls == case.authority.calls == case.transport.executions == []


def test_stage_then_submit_requires_the_actual_authenticated_stage_record(monkeypatch):
    case = _case(monkeypatch)
    grant = execution_grant(case.grants, case.command)
    stage_record = case.broker.execute(case.stage.command_bytes, grant, now_epoch=150)
    assert stage_record.state is EffectState.FOUND
    assert (
        case.broker.execute(case.stage.command_bytes, grant, now_epoch=150)
        == stage_record
    )
    template = parse_exact_command(case.submit.command_bytes)
    forged_grant = execution_grant(case.grants, template, "forged-predecessor")
    with pytest.raises(FoundationError, match="binding_mismatch"):
        case.broker.execute(template.canonical_bytes, forged_grant, now_epoch=150)
    assert len(case.transport.executions) == 1
    predecessor = replace(
        template.stage_predecessor,
        authenticated_receipt_digest=stage_record.results[
            0
        ].authenticated_receipt_digest,
        record_digest=stage_record.record_digest,
    )
    submit = build_submit_command(
        template.preparation,
        "real-stage-submit",
        template.payload,
        template.executor,
        predecessor,
    )
    retained = ModalInferenceCommandBinding(
        submit.canonical_bytes, case.stage.preparation_snapshot
    )
    case.catalog.values[retained.command_digest] = retained
    case.authority.accepted.add(retained.canonical_bytes)
    submit_grant = execution_grant(case.grants, submit, "real-submit")
    record = case.broker.execute(submit.canonical_bytes, submit_grant, now_epoch=150)
    assert record.state is EffectState.FOUND
    assert record.results[0].content.provider_run.provider_job_ref == "chat-object"
    assert (
        case.broker.execute(submit.canonical_bytes, submit_grant, now_epoch=150)
        == record
    )
    assert case.transport.executions == [case.command.digest, submit.digest]


def test_lost_dispatch_response_is_orphaned_and_never_resubmitted(monkeypatch):
    case = _case(monkeypatch)
    case.transport.fail_execute = True
    grant = execution_grant(case.grants, case.command)
    with pytest.raises(FoundationError, match="^effect_ambiguous$"):
        case.broker.execute(case.stage.command_bytes, grant, now_epoch=150)
    record = case.repository.get(case.command.operation.effect.effect_id)
    assert record.dispatch is DispatchState.ORPHANED_UNPROVEN
    assert record.state is EffectState.UNRESOLVED
    assert record.attempt_count == 1
    assert case.broker.execute(case.stage.command_bytes, grant, now_epoch=150) == record
    assert case.transport.executions == [case.command.digest]
    assert case.transport.lookups == []


def test_non_null_proof_does_not_authenticate_definite_absence(monkeypatch):
    case = _case(monkeypatch)
    case.transport.disposition = ObservationDisposition.DEFINITELY_ABSENT
    # Structurally valid digest, but not a proof issued by the actual verifier.
    case.transport.finality_proof = SimpleNamespace(proof_digest="1" * 64)
    grant = execution_grant(case.grants, case.command)
    record = case.broker.execute(case.stage.command_bytes, grant, now_epoch=150)
    assert record.state is EffectState.INDETERMINATE
    assert record.dispatch is DispatchState.RELINQUISHED
    assert case.broker.execute(case.stage.command_bytes, grant, now_epoch=150) == record
    assert case.transport.executions == [case.command.digest]


@pytest.mark.parametrize("failed_lookup", (False, True))
def test_reconciliation_uses_existing_claim_and_does_not_repeat_lookup(
    monkeypatch, failed_lookup
):
    case = _case(monkeypatch)
    case.transport.disposition = ObservationDisposition.INDETERMINATE
    case.broker.execute(
        case.stage.command_bytes,
        execution_grant(case.grants, case.command),
        now_epoch=150,
    )
    grant = reconciliation_grant_local(
        case.grants, case.command, case.adapter.descriptor
    )
    case.transport.fail_lookup = failed_lookup
    if failed_lookup:
        with pytest.raises(FoundationError, match="^reconciliation_interrupted$"):
            case.service.reconcile(case.stage.command_bytes, grant, now_epoch=150)
        record = case.repository.get(case.command.operation.effect.effect_id)
        assert record.state is EffectState.INDETERMINATE
        assert not record.reconciliation.active
        assert not record.reconciliation.completed
    else:
        record = case.service.reconcile(case.stage.command_bytes, grant, now_epoch=150)
        assert record.state is EffectState.FOUND
        assert record.reconciliation.completed
    with pytest.raises(FoundationError):
        case.service.reconcile(case.stage.command_bytes, grant, now_epoch=151)
    assert case.transport.executions == case.transport.lookups == [case.command.digest]
