import base64
import pytest

from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import (
    build_stage_command,
    parse_exact_command,
)
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.runtime import EnvironmentHmacAuthenticator

from examples.modal_chat.authority import (
    BoundedAuthorization,
    CanonicalCatalog,
    FoundationAuthenticator,
    HMACAuthenticator,
    LogAuthenticator,
    ModalChatAuthorityError,
    PlanningStore,
    ObservationAuthenticator,
    ReaderEvidenceAuthority,
    RetainedBindingAuthority,
    UnavailableQuiescenceEvidence,
    UnavailableRecoveryVerifier,
)
from tests.execution.coordinator_v1.test_state_machine import (
    CONTEXT,
    PLAN,
    intent,
    prep,
)
from tests.execution.coordinator_v1.test_state_machine import (
    observation,
    queued_evidence,
)
from tests.execution.coordinator_v1.test_page_store_and_logs import _log_content
from tuner.execution.coordinator_v1.model import ProviderRunPhaseV1
from tests.execution.providers.test_modal_coordinator_binding import case


class FixedClock:
    def now(self):
        return "2026-08-27T00:00:00Z"

    def now_iso(self):
        return self.now()

    def now_epoch(self):
        return 1787788800


def test_hmac_is_key_and_purpose_separated_and_worker_compatible(monkeypatch):
    authority = HMACAuthenticator(
        {"source-key": b"s" * 32, "stage-key": b"t" * 32},
        allowed_purposes=frozenset({"source-lock-evidence/v1", "modal-stage-claim/v2"}),
    )
    tag = authority.sign("modal-stage-claim/v2", b"claim", "stage-key")
    assert authority.verify("modal-stage-claim/v2", b"claim", tag, "stage-key")
    assert not authority.verify("modal-stage-claim/v2", b"changed", tag, "stage-key")
    assert not authority.verify("modal-stage-claim/v2", b"claim", tag, "source-key")
    assert not authority.verify("modal-launch-claim/v2", b"claim", tag, "stage-key")
    monkeypatch.setenv("MODAL_CHAT_STAGE_KEY", base64.b64encode(b"t" * 32).decode())
    worker = EnvironmentHmacAuthenticator(
        environment_key="MODAL_CHAT_STAGE_KEY",
        key_ref="stage-key",
    )
    assert worker.verify("modal-stage-claim/v2", b"claim", tag, "stage-key")
    worker_tag = worker.sign("modal-stage-claim/v2", b"worker", "stage-key")
    assert authority.verify("modal-stage-claim/v2", b"worker", worker_tag, "stage-key")


def test_reader_evidence_issues_exact_typed_envelopes():
    hmac_authority = HMACAuthenticator(
        {"read-key": b"r" * 32},
        allowed_purposes=frozenset(
            {
                "provider-run-observation/v1",
                "provider-log-page/v1",
            }
        ),
    )
    authority = ReaderEvidenceAuthority("reader-authority", "read-key", hmac_authority)
    workflow, foundation = queued_evidence()
    _, sample = observation(
        workflow, foundation, ProviderRunPhaseV1.RUNNING, {"state": "running"}
    )
    issued_observation = authority.observation(sample.content)
    issued_log = authority.log_page(_log_content())
    assert ObservationAuthenticator(authority).authenticate(issued_observation)
    assert LogAuthenticator(authority).authenticate(issued_log)
    assert not ObservationAuthenticator(authority).authenticate(issued_log)
    assert not LogAuthenticator(authority).authenticate(issued_observation)


def test_planning_store_reconstructs_exact_values_and_is_idempotent():
    store = PlanningStore()
    assert store.put_context_if_absent(CONTEXT)
    assert store.put_plan_if_absent(PLAN)
    assert not store.put_context_if_absent(CONTEXT)
    assert not store.put_plan_if_absent(PLAN)
    assert store.get_context(CONTEXT.provider_context_digest) == CONTEXT
    assert store.get_context(CONTEXT.provider_context_digest) is not CONTEXT
    assert store.get_plan(PLAN.plan_fingerprint) == PLAN
    assert store.get_plan("0" * 64) is None


def _binding(value):
    return ModalCommandBinding(
        value.command_bytes,
        value.preparation_snapshot,
        value.deployment_bytes,
    )


def test_binding_authority_trusts_only_exact_retained_readback():
    adapter, command, deployment = case()
    binding = ModalCommandBinding(
        command.canonical_bytes,
        adapter.snapshot(),
        canonical_bytes(deployment.to_dict()),
    )
    catalog = CanonicalCatalog(ModalCommandBinding, _binding)
    authority = RetainedBindingAuthority(catalog)
    assert not authority.authenticate(binding)
    assert catalog.publish_if_absent(command.digest, binding)
    assert authority.authenticate(binding)
    assert not catalog.publish_if_absent(command.digest, binding)
    assert not authority.authenticate(object())


def test_foundation_authenticator_delegates_real_hmac_authorities():
    grants = GrantAuthorityV2("grants", b"g" * 32)
    receipts = ReceiptAuthorityV2("receipts", b"r" * 32)
    invalid = InvalidEvidenceAuthorityV2("invalid", b"i" * 32)
    auth = FoundationAuthenticator(grants, receipts, invalid)
    command = intent("stage").canonical_command_bytes
    grant = grants.issue(
        command,
        grant_ref="grant-stage",
        policy_digest="1" * 64,
        requirement_digest="2" * 64,
        not_before_epoch=1,
        expires_at_epoch=2000000000,
    )
    assert auth.authenticate_grant(grant, command)
    assert not auth.authenticate_grant(grant, command + b" ")


def test_bounded_authorization_binds_preflight_plan_time_and_command():
    clock = FixedClock()
    preflight = TrainingPreflight(
        PLAN.plan_fingerprint,
        True,
        "2026-08-26T23:59:00Z",
        "2026-08-27T00:10:00Z",
    )
    grants = GrantAuthorityV2("grants", b"g" * 32)
    authority = BoundedAuthorization(
        PLAN, preflight, prep(), grants, clock, maximum_grant_seconds=300
    )
    command = intent("stage").canonical_command_bytes
    with pytest.raises(ModalChatAuthorityError):
        authority.issue_effect_grant(
            command,
            preflight_digest="0" * 64,
            now_epoch=clock.now_epoch(),
        )
    digest = authority.commit_preflight(PLAN, preflight)
    grant = authority.issue_effect_grant(
        command,
        preflight_digest=digest,
        now_epoch=clock.now_epoch(),
    )
    assert grants.verify(grant, command, now_epoch=clock.now_epoch())
    assert grant.content.expires_at_epoch == clock.now_epoch() + 300
    with pytest.raises(ModalChatAuthorityError):
        authority.issue_effect_grant(
            command, preflight_digest="0" * 64, now_epoch=clock.now_epoch()
        )
    parsed = parse_exact_command(command)
    alternate = build_stage_command(
        parsed.preparation,
        "alternate-nonce",
        parsed.payload,
        parsed.executor,
    )
    with pytest.raises(ModalChatAuthorityError):
        authority.issue_effect_grant(
            alternate.canonical_bytes,
            preflight_digest=digest,
            now_epoch=clock.now_epoch(),
        )
    changed_document = parsed.preparation.to_dict()
    changed_document["resource_digest"] = "f" * 64
    changed_preparation = CanonicalPreparationV2.parse(
        canonical_bytes(changed_document)
    )
    changed = build_stage_command(
        changed_preparation,
        parsed.operation.invocation_nonce,
        parsed.payload,
        parsed.executor,
    )
    other = BoundedAuthorization(PLAN, preflight, prep(), grants, clock)
    other.commit_preflight(PLAN, preflight)
    with pytest.raises(ModalChatAuthorityError):
        other.issue_effect_grant(
            changed.canonical_bytes,
            preflight_digest=digest,
            now_epoch=clock.now_epoch(),
        )
    with pytest.raises(ModalChatAuthorityError):
        authority.issue_reconciliation_grant()


def test_recovery_is_explicitly_unavailable():
    verifier = UnavailableRecoveryVerifier()
    assert verifier.verify_quiescence(object(), object(), now_epoch=1) is False
    assert verifier.verify_finality(object(), object(), object(), now_epoch=1) is False
    with pytest.raises(ModalChatAuthorityError, match="quiescence_unavailable"):
        UnavailableQuiescenceEvidence().obtain(object(), now_epoch=1)
