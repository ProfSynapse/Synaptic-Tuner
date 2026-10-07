"""Host construction of an authenticated Modal chat launch argument."""

from __future__ import annotations

import hashlib
import copy
import inspect
from datetime import timedelta
from typing import Protocol

from tuner.execution.coordinator_v1.model import (
    AuthenticatedFoundationRecordAssessmentV1,
    BoundProviderStageRefV1,
    EffectIntentV1,
    FoundationDispositionV1,
)
from tuner.execution.coordinator_v1.state_machine import _derive_foundation
from tuner.execution.evidence import (
    DEPLOYMENT_EVIDENCE_POLICY,
    canonical_utc,
    parse_utc,
    validate_evidence_window,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import (
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)
from tuner.execution.foundation_v2.repository import EffectRecordV2

from .inference_commands import ModalInferenceCommandBinding
from .inference_preparation import _validate_preparation_snapshot
from .inference_wire import ModalChatWorkerExpectation, _clock_now, _encode_argument

_PURPOSE = "modal-inference-launch/v1"
_SCHEMA = "synaptic-modal-inference-launch/v1"


class _BindingAuthority(Protocol):
    def authenticate(self, binding: ModalInferenceCommandBinding) -> bool: ...


class _Signer(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...


class ModalChatLaunchEnvelope:
    """Immutable launch evidence; Foundation consumption remains host-owned."""

    __slots__ = (
        "_claim",
        "_claim_tag",
        "_stage_command_bytes",
        "_submit_command_bytes",
        "_preparation_snapshot",
    )

    def __init__(
        self,
        claim: bytes,
        claim_tag: bytes,
        stage_command_bytes: bytes,
        submit_command_bytes: bytes,
        preparation_snapshot: bytes,
    ) -> None:
        values = (
            claim,
            claim_tag,
            stage_command_bytes,
            submit_command_bytes,
            preparation_snapshot,
        )
        if any(type(value) is not bytes or not value for value in values):
            raise TypeError("exact nonempty launch bytes are required")
        if len(claim_tag) > 128:
            raise ValueError("launch tag exceeds its bound")
        object.__setattr__(self, "_claim", bytes(claim))
        object.__setattr__(self, "_claim_tag", bytes(claim_tag))
        object.__setattr__(self, "_stage_command_bytes", bytes(stage_command_bytes))
        object.__setattr__(self, "_submit_command_bytes", bytes(submit_command_bytes))
        object.__setattr__(self, "_preparation_snapshot", bytes(preparation_snapshot))
        self.argument_bytes

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalChatLaunchEnvelope is final")

    def __setattr__(self, name, value):
        raise AttributeError("Modal chat launch envelopes are immutable")

    @property
    def claim(self) -> bytes:
        return bytes(self._claim)

    @property
    def claim_tag(self) -> bytes:
        return bytes(self._claim_tag)

    @property
    def stage_command_bytes(self) -> bytes:
        return bytes(self._stage_command_bytes)

    @property
    def submit_command_bytes(self) -> bytes:
        return bytes(self._submit_command_bytes)

    @property
    def preparation_snapshot(self) -> bytes:
        return bytes(self._preparation_snapshot)

    @property
    def argument_bytes(self) -> bytes:
        return _encode_argument(
            self._claim,
            self._claim_tag,
            self._stage_command_bytes,
            self._submit_command_bytes,
            self._preparation_snapshot,
        )


def modal_chat_stage_ref(binding: ModalInferenceCommandBinding) -> str:
    if type(binding) is not ModalInferenceCommandBinding:
        raise TypeError("exact Modal inference command binding required")
    command = parse_exact_command(binding.command_bytes)
    if type(command) is not StageCommandV2:
        raise ValueError("exact chat STAGE binding required")
    return "modal-chat-stage:" + binding.binding_digest


def _owned_binding(
    value: ModalInferenceCommandBinding, authority: _BindingAuthority
) -> tuple[ModalInferenceCommandBinding, object]:
    if type(value) is not ModalInferenceCommandBinding:
        raise TypeError("exact Modal inference command binding required")
    method = inspect.getattr_static(type(authority), "authenticate", None)
    if (
        method is None
        or not callable(method)
        or inspect.getattr_static(authority, "authenticate", None) is not method
    ):
        raise TypeError("exact static binding authority required")
    owned = ModalInferenceCommandBinding(
        value.command_bytes, value.preparation_snapshot
    )
    if authority.authenticate(owned) is not True:
        raise ValueError("Modal inference command binding authentication failed")
    if (value.command_bytes, value.preparation_snapshot) != (
        owned.command_bytes,
        owned.preparation_snapshot,
    ):
        raise ValueError("Modal inference command binding changed")
    return owned, parse_exact_command(owned.command_bytes)


def _claim_document(
    *,
    stage_binding: ModalInferenceCommandBinding,
    submit_binding: ModalInferenceCommandBinding,
    stage: StageCommandV2,
    submit: SubmitCommandV2,
    stage_record: EffectRecordV2,
    stage_assessment: AuthenticatedFoundationRecordAssessmentV1,
    foundation_binding,
    foundation_outcome,
    bound: BoundProviderStageRefV1,
    expectation: ModalChatWorkerExpectation,
    issued_at: str,
    expires_at: str,
    evidence_ref: str,
) -> dict[str, object]:
    snapshot = _validate_preparation_snapshot(submit_binding.preparation_snapshot)
    return {
        "schema_version": _SCHEMA,
        "stage_ref": modal_chat_stage_ref(stage_binding),
        "stage_binding_digest": stage_binding.binding_digest,
        "stage_command_digest": stage.digest,
        "stage_effect_id": stage.operation.effect.effect_id,
        "stage_invocation_nonce": stage.operation.invocation_nonce,
        "submit_binding_digest": submit_binding.binding_digest,
        "submit_command_digest": submit.digest,
        "submit_effect_id": submit.operation.effect.effect_id,
        "submit_invocation_nonce": submit.operation.invocation_nonce,
        "preparation_snapshot_sha256": hashlib.sha256(
            submit_binding.preparation_snapshot
        ).hexdigest(),
        "configuration_digest": snapshot["configuration_digest"],
        "session_id": snapshot["chat_input"]["session_id"],
        "stage_record_digest": stage_record.record_digest,
        "stage_assessment_digest": stage_assessment.authenticated_assessment_digest,
        "stage_foundation_binding_digest": foundation_binding.binding_digest,
        "stage_outcome_digest": foundation_outcome.outcome_digest,
        "stage_bound_reference_digest": bound.binding_digest,
        "stage_authenticated_receipt_digest": bound.authenticated_receipt_digest,
        "issued_at": issued_at,
        "expires_at": expires_at,
        "evidence_ref": evidence_ref,
        "issuer_ref": expectation.issuer_ref,
        "audience_ref": expectation.audience_ref,
        "key_ref": expectation.key_ref,
        "challenge_nonce": expectation.challenge_nonce,
        "artifact_root": expectation.artifact_root,
        "control_root": expectation.control_root,
        "cache_root": expectation.cache_root,
    }


def prepare_modal_chat_launch(
    *,
    stage_binding: ModalInferenceCommandBinding,
    submit_binding: ModalInferenceCommandBinding,
    stage_record: EffectRecordV2,
    stage_assessment: AuthenticatedFoundationRecordAssessmentV1,
    foundation_authenticator,
    assessment_authenticator,
    binding_authority: _BindingAuthority,
    signer: _Signer,
    expectation: ModalChatWorkerExpectation,
    clock,
    issued_at: str,
    expires_at: str,
    evidence_ref: str,
) -> ModalChatLaunchEnvelope:
    """Authenticate a completed chat STAGE and sign its exact launch content."""
    try:
        if type(expectation) is not ModalChatWorkerExpectation:
            raise TypeError("exact Modal chat worker expectation required")
        if (
            type(stage_binding) is not ModalInferenceCommandBinding
            or type(submit_binding) is not ModalInferenceCommandBinding
        ):
            raise TypeError("exact Modal inference command bindings required")
        if type(stage_record) is not EffectRecordV2:
            raise TypeError("exact Foundation stage record required")
        if type(stage_assessment) is not AuthenticatedFoundationRecordAssessmentV1:
            raise TypeError("exact Foundation stage assessment required")
        if type(evidence_ref) is not str:
            raise TypeError("evidence_ref must be an exact string")
        safe_ref(evidence_ref, "evidence_ref")
        issued_at = canonical_utc(issued_at, "issued_at")
        expires_at = canonical_utc(expires_at, "expires_at")
        if type(issued_at) is not str or type(expires_at) is not str:
            raise TypeError("launch timestamps must be exact strings")
        owned_expectation = ModalChatWorkerExpectation(
            **{
                name: getattr(expectation, name)
                for name in expectation.__dataclass_fields__
            }
        )
        initial_stage = (
            stage_binding.command_bytes,
            stage_binding.preparation_snapshot,
        )
        initial_submit = (
            submit_binding.command_bytes,
            submit_binding.preparation_snapshot,
        )
        initial_record_digest = stage_record.record_digest
        initial_assessment = bytes(stage_assessment.canonical_bytes)
        initial_expectation = tuple(
            getattr(expectation, name) for name in expectation.__dataclass_fields__
        )
        owned_record = copy.deepcopy(stage_record)
        owned_assessment = AuthenticatedFoundationRecordAssessmentV1.parse(
            initial_assessment
        )
        owned_stage_snapshot = None
        owned_submit_snapshot = None

        def unchanged() -> bool:
            return (
                (stage_binding.command_bytes, stage_binding.preparation_snapshot)
                == initial_stage
                and (submit_binding.command_bytes, submit_binding.preparation_snapshot)
                == initial_submit
                and stage_record.record_digest == initial_record_digest
                and stage_assessment.canonical_bytes == initial_assessment
                and owned_record.record_digest == initial_record_digest
                and owned_assessment.canonical_bytes == initial_assessment
                and (
                    owned_stage_snapshot is None
                    or (
                        owned_stage.command_bytes,
                        owned_stage.preparation_snapshot,
                    )
                    == owned_stage_snapshot
                )
                and (
                    owned_submit_snapshot is None
                    or (
                        owned_submit.command_bytes,
                        owned_submit.preparation_snapshot,
                    )
                    == owned_submit_snapshot
                )
                and tuple(
                    getattr(expectation, name)
                    for name in expectation.__dataclass_fields__
                )
                == initial_expectation
            )

        now = _clock_now(clock)
        if not unchanged():
            raise ValueError("chat launch inputs changed")
        if parse_utc(issued_at) > parse_utc(now):
            raise ValueError("chat launch issuance is in the future")
        validate_evidence_window(
            verified_at=issued_at,
            expires_at=expires_at,
            now=now,
            policy=DEPLOYMENT_EVIDENCE_POLICY,
        )
        owned_stage, stage = _owned_binding(stage_binding, binding_authority)
        owned_stage_snapshot = (
            owned_stage.command_bytes,
            owned_stage.preparation_snapshot,
        )
        if not unchanged():
            raise ValueError("chat launch inputs changed")
        owned_submit, submit = _owned_binding(submit_binding, binding_authority)
        owned_submit_snapshot = (
            owned_submit.command_bytes,
            owned_submit.preparation_snapshot,
        )
        if not unchanged():
            raise ValueError("chat launch inputs changed")
        if type(stage) is not StageCommandV2 or type(submit) is not SubmitCommandV2:
            raise ValueError("chat launch requires exact STAGE and SUBMIT commands")
        if owned_stage.preparation_snapshot != owned_submit.preparation_snapshot:
            raise ValueError("chat launch preparation snapshots differ")
        intent = EffectIntentV1.from_command_bytes(stage.canonical_bytes)
        foundation_binding, foundation_outcome, bound = _derive_foundation(
            intent,
            owned_record,
            owned_assessment,
            foundation_authenticator,
            assessment_authenticator,
            None,
        )
        if not unchanged():
            raise ValueError("chat launch inputs changed")
        if (
            foundation_outcome.disposition is not FoundationDispositionV1.FOUND
            or type(bound) is not BoundProviderStageRefV1
            or bound.reference.stage_ref != modal_chat_stage_ref(owned_stage)
        ):
            raise ValueError("stage lacks its authenticated chat reference")
        preparation = submit.preparation
        predecessor = submit.stage_predecessor
        expected_predecessor = (
            preparation.provider.provider_id,
            preparation.provider.profile_ref,
            preparation.scope.account_ref,
            preparation.scope.namespace_ref,
            preparation.project_ref,
            preparation.run_id,
            preparation.plan_fingerprint,
            preparation.preparation_digest,
            preparation.workload_digest,
            stage.operation.effect.effect_id,
            bound.authenticated_receipt_digest,
            owned_record.record_digest,
        )
        if (
            stage.preparation != preparation
            or tuple(
                getattr(predecessor, name) for name in predecessor.__dataclass_fields__
            )
            != expected_predecessor
        ):
            raise ValueError("SUBMIT predecessor differs from authenticated chat STAGE")
        if submit.digest != owned_expectation.submit_command_digest:
            raise ValueError("SUBMIT command differs from worker expectation")
        snapshot = _validate_preparation_snapshot(owned_submit.preparation_snapshot)
        configuration = snapshot["configuration"]
        if (
            canonical_bytes(configuration) != owned_expectation.configuration_bytes
            or submit.executor.implementation_version
            != owned_expectation.executor_version
            or parse_utc(expires_at) - parse_utc(issued_at)
            > timedelta(seconds=configuration["policy"]["absolute_lifetime_seconds"])
        ):
            raise ValueError("chat launch differs from worker expectation")
        signer_method = inspect.getattr_static(type(signer), "sign", None)
        if (
            signer_method is None
            or not callable(signer_method)
            or inspect.getattr_static(signer, "sign", None) is not signer_method
        ):
            raise TypeError("exact static evidence signer required")
        claim = canonical_bytes(
            _claim_document(
                stage_binding=owned_stage,
                submit_binding=owned_submit,
                stage=stage,
                submit=submit,
                stage_record=owned_record,
                stage_assessment=owned_assessment,
                foundation_binding=foundation_binding,
                foundation_outcome=foundation_outcome,
                bound=bound,
                expectation=owned_expectation,
                issued_at=issued_at,
                expires_at=expires_at,
                evidence_ref=evidence_ref,
            )
        )
        tag = signer.sign(_PURPOSE, claim, owned_expectation.key_ref)
        if not unchanged():
            raise ValueError("chat launch inputs changed")
        envelope = ModalChatLaunchEnvelope(
            claim,
            tag,
            owned_stage.command_bytes,
            owned_submit.command_bytes,
            owned_submit.preparation_snapshot,
        )
        if (
            not unchanged()
            or owned_record.record_digest != foundation_binding.foundation_record_digest
            or owned_assessment.authenticated_assessment_digest
            != foundation_binding.assessment_digest
        ):
            raise ValueError("chat launch inputs changed")
        return envelope
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError("Modal chat launch preparation invalid") from None


__all__: list[str] = []
