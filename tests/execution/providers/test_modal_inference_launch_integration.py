"""Real Foundation stage evidence feeding the private chat launch boundary."""

from dataclasses import replace
import copy
import hashlib
import hmac
from types import SimpleNamespace

import pytest

from tests.execution.coordinator_v1.test_start_reconcile_service import (
    FoundationAuthenticator,
)
from tests.execution.foundation_v2.helpers import execution_grant
from tests.execution.providers.test_modal_inference_broker import _case
from tuner.execution.coordinator_v1.foundation import (
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    parse_canonical_object,
)
from tuner.execution.foundation_v2.commands import (
    build_submit_command,
    parse_exact_command,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.repository import DispatchState, EffectState
from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_effects import (
    ModalChatEffectExecutor,
    ModalChatExecutorResolver,
)
from tuner.execution.providers.modal.inference_launch import (
    modal_chat_stage_ref,
    prepare_modal_chat_launch,
)
from tuner.execution.providers.modal.inference_wire import (
    ModalChatWorkerExpectation,
    admit_modal_chat_launch,
)


class Clock:
    def now_iso(self):
        return "2026-09-09T12:02:01Z"


class LaunchAuthenticator:
    """Synthetic test key, not a credential or a runtime authority implementation."""

    def __init__(self):
        self.sign_calls = []
        self.verify_calls = []

    @staticmethod
    def _tag(purpose, payload, key_ref):
        return hmac.new(
            b"test-launch-key" * 3,
            purpose.encode() + b"\0" + key_ref.encode() + b"\0" + payload,
            hashlib.sha256,
        ).digest()

    def sign(self, purpose, payload, key_ref):
        self.sign_calls.append((purpose, payload, key_ref))
        return self._tag(purpose, payload, key_ref)

    def verify(self, purpose, payload, tag, key_ref):
        self.verify_calls.append((purpose, payload, key_ref))
        return hmac.compare_digest(tag, self._tag(purpose, payload, key_ref))


def _launch_case(monkeypatch):
    case = _case(monkeypatch)

    class StageTransport:
        def __init__(self):
            self.calls = []

        def execute_once(self, binding, command):
            self.calls.append(command.digest)
            return ModalEffectOutcome(
                ObservationDisposition.FOUND, modal_chat_stage_ref(binding)
            )

    transport = StageTransport()
    executor = ModalChatEffectExecutor(**(case.kwargs | {"transport": transport}))
    broker = EffectBrokerV2(
        case.repository,
        ModalChatExecutorResolver(executor),
        case.grants,
        case.receipts,
        case.invalid,
    )
    stage_record = broker.execute(
        case.stage.command_bytes,
        execution_grant(case.grants, case.command),
        now_epoch=150,
    )
    template = parse_exact_command(case.submit.command_bytes)
    predecessor = replace(
        template.stage_predecessor,
        authenticated_receipt_digest=stage_record.results[
            0
        ].authenticated_receipt_digest,
        record_digest=stage_record.record_digest,
    )
    submit = build_submit_command(
        template.preparation,
        "launch-submit",
        template.payload,
        template.executor,
        predecessor,
    )
    submit_binding = ModalInferenceCommandBinding(
        submit.canonical_bytes, case.stage.preparation_snapshot
    )
    case.authority.accepted.add(submit_binding.canonical_bytes)
    case.catalog.values[submit_binding.command_digest] = submit_binding
    clock = Clock()
    assessment_authority = FoundationRecordAssessmentAuthorityV1(
        "chat-assessment",
        "chat-assessment-key",
        b"a" * 32,
        assessor_ref="chat-assessor",
        assessor_version="test",
        clock=clock,
        receipt_authority=case.receipts,
        invalid_evidence_authority=case.invalid,
        grant_authority=case.grants,
    )
    assessment = assessment_authority.assess(stage_record)
    authenticator = LaunchAuthenticator()
    configuration = parse_canonical_object(
        case.stage.preparation_snapshot, name="preparation"
    )["configuration"]
    expectation = ModalChatWorkerExpectation(
        configuration_bytes=canonical_bytes(configuration),
        submit_command_digest=submit.digest,
        executor_version=submit.executor.implementation_version,
        issuer_ref="host-launch",
        audience_ref="chat-worker",
        key_ref=configuration["volumes"]["key_ref"],
        challenge_nonce="launch-nonce",
        artifact_root="/worker/artifacts",
        control_root="/worker/control",
        cache_root="/worker/cache",
    )
    arguments = dict(
        stage_binding=case.stage,
        submit_binding=submit_binding,
        stage_record=stage_record,
        stage_assessment=assessment,
        foundation_authenticator=FoundationAuthenticator(
            case.grants, case.receipts, case.invalid
        ),
        assessment_authenticator=assessment_authority,
        binding_authority=case.authority,
        signer=authenticator,
        clock=clock,
        expectation=expectation,
        issued_at="2026-09-09T12:02:00Z",
        expires_at="2026-09-09T12:04:00Z",
        evidence_ref="launch-evidence",
    )
    return SimpleNamespace(**locals())


def test_actual_foundation_stage_receipt_reaches_owned_worker_admission(monkeypatch):
    case = _launch_case(monkeypatch)
    envelope = prepare_modal_chat_launch(**case.arguments)
    assert len(case.authenticator.sign_calls) == 1
    admitted = admit_modal_chat_launch(
        envelope.argument_bytes,
        expectation=case.expectation,
        verifier=case.authenticator,
        clock=case.clock,
    )
    assert admitted.stage_command_bytes == case.case.stage.command_bytes
    assert admitted.submit_command_bytes == case.submit.canonical_bytes
    assert admitted.preparation_snapshot == case.submit_binding.preparation_snapshot
    assert len(case.authenticator.verify_calls) == 1
    assert case.transport.calls == [case.case.command.digest]
    # This pure builder/admission is NOT a grant boundary or a provider submit.
    assert case.case.repository.get(case.submit.operation.effect.effect_id) is None


@pytest.mark.parametrize("field", ("authenticated_receipt_digest", "record_digest"))
def test_structural_predecessor_cannot_replace_authenticated_stage_evidence(
    monkeypatch, field
):
    case = _launch_case(monkeypatch)
    predecessor = replace(case.submit.stage_predecessor, **{field: "f" * 64})
    submit = build_submit_command(
        case.submit.preparation,
        "launch-submit",
        case.submit.payload,
        case.submit.executor,
        predecessor,
    )
    binding = ModalInferenceCommandBinding(
        submit.canonical_bytes, case.submit_binding.preparation_snapshot
    )
    case.case.authority.accepted.add(binding.canonical_bytes)
    expectation = replace(case.expectation, submit_command_digest=submit.digest)
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(
            **(case.arguments | {"submit_binding": binding, "expectation": expectation})
        )
    assert case.authenticator.sign_calls == []


def test_changed_authenticated_stage_record_fails_before_launch_signing(monkeypatch):
    case = _launch_case(monkeypatch)
    changed = copy.copy(case.stage_record)
    object.__setattr__(changed, "terminal_content_digests", ())
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(**(case.arguments | {"stage_record": changed}))
    assert case.authenticator.sign_calls == []


def test_foreign_assessment_authority_cannot_attest_stage_lineage(monkeypatch):
    case = _launch_case(monkeypatch)
    foreign = FoundationRecordAssessmentAuthorityV1(
        "foreign-assessment",
        "foreign-key",
        b"f" * 32,
        assessor_ref="other-assessor",
        assessor_version="test",
        clock=case.clock,
        receipt_authority=case.case.receipts,
        invalid_evidence_authority=case.case.invalid,
        grant_authority=case.case.grants,
    )
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(
            **(case.arguments | {"stage_assessment": foreign.assess(case.stage_record)})
        )
    assert case.authenticator.sign_calls == []


def test_launch_composition_runs_inside_one_consumed_foundation_submit(monkeypatch):
    case = _launch_case(monkeypatch)

    class LaunchTransport:
        def __init__(self):
            self.calls = []

        def execute_once(self, binding, command):
            current = case.case.repository.get(command.operation.effect.effect_id)
            assert current.dispatch is DispatchState.OWNED_IN_FLIGHT
            assert current.attempt_count == current.dispatch_epoch == 1
            assert binding.canonical_bytes == case.submit_binding.canonical_bytes
            envelope = prepare_modal_chat_launch(**case.arguments)
            admission = admit_modal_chat_launch(
                envelope.argument_bytes,
                expectation=case.expectation,
                verifier=case.authenticator,
                clock=case.clock,
            )
            self.calls.append(admission.submit_command_bytes)
            # Synthetic provider result: this test never calls an SDK or allocates.
            return ModalEffectOutcome(ObservationDisposition.FOUND, "sandbox-fixture")

    transport = LaunchTransport()
    executor = ModalChatEffectExecutor(**(case.case.kwargs | {"transport": transport}))
    broker = EffectBrokerV2(
        case.case.repository,
        ModalChatExecutorResolver(executor),
        case.case.grants,
        case.case.receipts,
        case.case.invalid,
    )
    grant = execution_grant(case.case.grants, case.submit, "chat-submit-grant")
    first = broker.execute(case.submit.canonical_bytes, grant, now_epoch=150)
    assert first.state is EffectState.FOUND
    assert broker.execute(case.submit.canonical_bytes, grant, now_epoch=150) == first
    assert transport.calls == [case.submit.canonical_bytes]
    assert (
        len(case.authenticator.sign_calls) == len(case.authenticator.verify_calls) == 1
    )


@pytest.mark.parametrize(
    "expiry", ("2026-09-09T12:02:00Z", "2026-09-09T12:01:59Z", "2026-09-09T12:07:01Z")
)
def test_unusable_admission_window_is_rejected_before_signing(monkeypatch, expiry):
    case = _launch_case(monkeypatch)
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(**(case.arguments | {"expires_at": expiry}))
    assert case.authenticator.sign_calls == []


@pytest.mark.parametrize("field", ("configuration", "executor", "root"))
def test_independent_expectation_mismatch_is_rejected_before_signing(
    monkeypatch, field
):
    case = _launch_case(monkeypatch)
    if field == "configuration":
        changed = dict(case.configuration)
        changed["application"] = changed["application"] | {
            "worker_ref": "another-worker"
        }
        expectation = replace(
            case.expectation, configuration_bytes=canonical_bytes(changed)
        )
    elif field == "executor":
        expectation = replace(case.expectation, executor_version="other-executor")
    else:
        expectation = copy.copy(case.expectation)
        object.__setattr__(expectation, "artifact_root", "/")
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(**(case.arguments | {"expectation": expectation}))
    assert case.authenticator.sign_calls == []


@pytest.mark.parametrize(
    "issued,expires",
    (
        ("2026-09-09T12:02:02Z", "2026-09-09T12:04:00Z"),
        ("2026-09-09T13:02:00Z", "2026-09-09T13:04:00Z"),
        ("2026-09-09T11:02:00Z", "2026-09-09T11:04:00Z"),
    ),
)
def test_future_or_stale_claim_is_not_signed(monkeypatch, issued, expires):
    case = _launch_case(monkeypatch)
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(
            **(case.arguments | {"issued_at": issued, "expires_at": expires})
        )
    assert case.authenticator.sign_calls == []


def test_verifier_cannot_rewrite_independent_worker_expectation(monkeypatch):
    case = _launch_case(monkeypatch)
    envelope = prepare_modal_chat_launch(**case.arguments)
    expectation = replace(case.expectation, submit_command_digest="0" * 64)

    class Verifier:
        calls = 0

        def verify(self, purpose, payload, tag, key_ref):
            self.calls += 1
            object.__setattr__(expectation, "submit_command_digest", case.submit.digest)
            return case.authenticator.verify(purpose, payload, tag, key_ref)

    verifier = Verifier()
    with pytest.raises(ValueError, match="^Modal chat launch admission invalid$"):
        admit_modal_chat_launch(
            envelope.argument_bytes,
            expectation=expectation,
            verifier=verifier,
            clock=case.clock,
        )
    assert verifier.calls == 1


def test_signer_cannot_mutate_authority_retained_owned_binding(monkeypatch):
    case = _launch_case(monkeypatch)

    class Signer:
        calls = 0

        def sign(self, purpose, payload, key_ref):
            self.calls += 1
            owned_submit = case.case.authority.calls[-1]
            assert owned_submit is not case.submit_binding
            object.__setattr__(
                owned_submit, "_command_bytes", case.case.stage.command_bytes
            )
            return case.authenticator.sign(purpose, payload, key_ref)

    signer = Signer()
    with pytest.raises(ValueError, match="^Modal chat launch preparation invalid$"):
        prepare_modal_chat_launch(**(case.arguments | {"signer": signer}))
    assert signer.calls == 1
