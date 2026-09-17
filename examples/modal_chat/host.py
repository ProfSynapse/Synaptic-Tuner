"""Single-process composition of the real Foundation-native Modal coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from synaptic_tuner.api.v1.host import APIHost, HostPorts
from synaptic_tuner.api.v1.training_facade import TrainingPreflight
from tuner.execution.coordinator_v1.cursors import HMACCursorAuthorityV1
from tuner.execution.coordinator_v1.foundation import (
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.stores import (
    InMemoryExecutionGrantStoreV1,
    InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1,
    InMemoryWorkflowStoreV1,
)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.registry import ProviderReaderFactoryRequestV1
from tuner.execution.foundation_v2.repository import InMemoryEffectRepositoryV2
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_composition import (
    ModalCoordinatorComposition,
    ModalCoordinatorStorePorts,
    ModalFoundationCompositionPorts,
    compose_modal_coordinator,
)
from tuner.execution.providers.modal.coordinator_launch import ModalLaunchEnvelope
from tuner.execution.providers.modal.coordinator_retention import (
    ModalRetainedPreparation,
)
from tuner.execution.providers.modal.coordinator_staging import ModalStageMaterial

from .artifacts import ModalChatArtifactVerifier
from .authority import (
    BoundedAuthorization,
    CanonicalCatalog,
    FoundationAuthenticator,
    HMACAuthenticator,
    LogAuthenticator,
    ObservationAuthenticator,
    PlanningStore,
    ReaderEvidenceAuthority,
    RetainedBindingAuthority,
    RetainedInputs,
    UnavailableQuiescenceEvidence,
    UnavailableRecoveryVerifier,
)
from .requests import ModalChatTrainingRequests


class ModalChatHostError(RuntimeError):
    """Closed host-composition failure."""


class _ExactCatalog:
    def __init__(self, expected_type, rebuild):
        self._type, self._rebuild = expected_type, rebuild
        self._values = {}
        self._lock = RLock()

    def _owned(self, value):
        if type(value) is not self._type:
            raise ModalChatHostError("modal_chat_host_invalid")
        owned = self._rebuild(value)
        if type(owned) is not self._type or owned != value:
            raise ModalChatHostError("modal_chat_host_invalid")
        return owned

    def resolve(self, key):
        with self._lock:
            value = self._values.get(key)
        return None if value is None else self._owned(value)

    def publish_if_absent(self, key, value):
        owned = self._owned(value)
        with self._lock:
            prior = self._values.get(key)
            if prior is None:
                self._values[key] = owned
                return True
            if self._owned(prior) == owned:
                return False
            raise ModalChatHostError("modal_chat_host_conflict")


class _AuthorizationSlot:
    def __init__(
        self, plan, preparation, grants, clock, *, cost, currency, maximum_grant_seconds
    ):
        self._plan, self._preparation = plan, preparation
        self._grants, self._clock = grants, clock
        self._cost, self._currency = cost, currency
        self._maximum = maximum_grant_seconds
        self._delegate = None
        self._lock = RLock()

    def commit_preflight(self, plan, preflight):
        if type(preflight) is not TrainingPreflight or plan != self._plan:
            raise ModalChatHostError("modal_chat_authorization_invalid")
        requirements = preflight.authorization
        if (
            len(requirements) != 1
            or requirements[0].operation != "training.start"
            or requirements[0].paid_effect is not True
            or requirements[0].maximum_cost_minor_units != self._cost
            or requirements[0].currency != self._currency
        ):
            raise ModalChatHostError("modal_chat_authorization_invalid")
        with self._lock:
            if self._delegate is None:
                self._delegate = BoundedAuthorization(
                    plan,
                    preflight,
                    self._preparation,
                    self._grants,
                    self._clock,
                    maximum_grant_seconds=self._maximum,
                )
            return self._delegate.commit_preflight(plan, preflight)

    def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
        with self._lock:
            delegate = self._delegate
        if delegate is None:
            raise ModalChatHostError("modal_chat_authorization_invalid")
        return delegate.issue_effect_grant(
            command_bytes,
            preflight_digest=preflight_digest,
            now_epoch=now_epoch,
        )

    def issue_reconciliation_grant(self, *args, **kwargs):
        raise ModalChatHostError("modal_chat_reconciliation_unsupported")


@dataclass(frozen=True, slots=True)
class ModalChatHost:
    api: APIHost
    composition: ModalCoordinatorComposition
    reader: object
    authorization: object
    foundation_ports: ModalFoundationCompositionPorts
    stores: ModalCoordinatorStorePorts
    artifact_verifier: ModalChatArtifactVerifier
    reader_authority: ReaderEvidenceAuthority
    observation_authenticator: ObservationAuthenticator
    log_authenticator: LogAuthenticator
    clock: object


def compose_modal_chat_host(
    *,
    preparation,
    operational_preflight,
    facade,
    deployment,
    recipes,
    requests: ModalChatTrainingRequests,
    retained: ModalRetainedPreparation,
    evidence_authenticator: HMACAuthenticator,
    evidence_key_ref: str,
    artifact_key: bytes,
    clock,
    observed_at: str,
    grant_key: bytes,
    receipt_key: bytes,
    invalid_evidence_key: bytes,
    assessment_key: bytes,
    cursor_key: bytes,
    approved_cost_minor_units: int,
    approved_currency: str = "USD",
    maximum_artifact_bytes: int = 256 * 1024 * 1024,
    maximum_grant_seconds: int = 900,
) -> ModalChatHost:
    if (
        type(requests) is not ModalChatTrainingRequests
        or type(retained) is not ModalRetainedPreparation
        or type(evidence_authenticator) is not HMACAuthenticator
    ):
        raise TypeError("exact Modal chat host inputs required")
    context, execution, plan = preparation._snapshot()
    run = requests.for_plan(plan)
    expected_preparation = preparation.prepare(plan, run, execution)
    if (
        retained.preparation_snapshot != preparation.snapshot()
        or retained.deployment_bytes != canonical_bytes(deployment.to_dict())
    ):
        raise ModalChatHostError("modal_chat_host_invalid")
    grants = GrantAuthorityV2("modal-chat-grants", grant_key)
    receipts = ReceiptAuthorityV2("modal-chat-receipts", receipt_key)
    invalid = InvalidEvidenceAuthorityV2("modal-chat-invalid", invalid_evidence_key)
    foundation_auth = FoundationAuthenticator(grants, receipts, invalid)
    unavailable = UnavailableRecoveryVerifier()
    repository = InMemoryEffectRepositoryV2(
        receipts,
        invalid,
        unavailable,
        unavailable,
        grants,
    )
    assessments = FoundationRecordAssessmentAuthorityV1(
        "modal-chat-assessments",
        "modal-chat-assessment-key",
        assessment_key,
        assessor_ref="modal-chat-assessor",
        assessor_version="1.0.0",
        clock=clock,
        receipt_authority=receipts,
        invalid_evidence_authority=invalid,
        grant_authority=grants,
    )
    artifact_verifier = ModalChatArtifactVerifier(
        foundation_authenticator=foundation_auth,
        assessment_authenticator=assessments,
        authority_ref="modal-chat-artifacts",
        key_ref="modal-chat-artifact-key",
        key=artifact_key,
        clock=clock,
        maximum_total_bytes=maximum_artifact_bytes,
    )
    bindings = CanonicalCatalog(
        ModalCommandBinding,
        lambda value: ModalCommandBinding(
            value.command_bytes,
            value.preparation_snapshot,
            value.deployment_bytes,
        ),
    )
    binding_authority = RetainedBindingAuthority(bindings)
    stages = _ExactCatalog(
        ModalStageMaterial,
        lambda value: ModalStageMaterial(
            **{name: getattr(value, name) for name in value.__dataclass_fields__}
        ),
    )
    launches = _ExactCatalog(
        ModalLaunchEnvelope,
        lambda value: ModalLaunchEnvelope(
            **{name: getattr(value, name) for name in value.__dataclass_fields__}
        ),
    )
    reader_authority = ReaderEvidenceAuthority(
        "modal-chat-reader",
        evidence_key_ref,
        evidence_authenticator,
    )
    observation_auth = ObservationAuthenticator(reader_authority)
    log_auth = LogAuthenticator(reader_authority)
    authorization = _AuthorizationSlot(
        plan,
        expected_preparation,
        grants,
        clock,
        cost=approved_cost_minor_units,
        currency=approved_currency,
        maximum_grant_seconds=maximum_grant_seconds,
    )
    stores = ModalCoordinatorStorePorts(
        PlanningStore(),
        InMemoryWorkflowStoreV1(
            foundation_auth,
            assessments,
            observation_auth,
            artifact_verifier,
        ),
        InMemoryPreparationStoreV1(),
        InMemoryExecutionGrantStoreV1(grants),
        InMemoryReconciliationGrantStoreV1(grants),
    )
    foundation_ports = ModalFoundationCompositionPorts(
        repository,
        grants,
        receipts,
        invalid,
        assessments,
        foundation_auth,
        UnavailableQuiescenceEvidence(),
        binding_authority,
        evidence_authenticator,
        evidence_authenticator,
        bindings,
        stages,
        launches,
        RetainedInputs(expected_preparation.preparation_digest, retained),
    )
    composed = compose_modal_coordinator(
        preparation=preparation,
        operational_preflight=operational_preflight,
        facade=facade,
        deployment=deployment,
        recipes=recipes,
        evidence_authority=reader_authority,
        evidence_verifier=evidence_authenticator,
        observation_authenticator=observation_auth,
        log_authenticator=log_auth,
        artifact_verifier=artifact_verifier,
        cursor_authority=HMACCursorAuthorityV1(
            "modal-chat-cursors",
            {1: cursor_key},
            active_generation=1,
        ),
        observed_at=observed_at,
        loader=requests,
        resolver=requests,
        authorization=authorization,
        clock=clock,
        run_identity=requests,
        foundation_ports=foundation_ports,
        stores=stores,
    )
    descriptor = composed.registration.provider
    reader_request = ProviderReaderFactoryRequestV1(
        context.provider,
        descriptor.descriptor_digest,
        execution.profile_digest,
        execution.scope.account_ref,
        execution.scope.namespace_ref,
    )
    reader = composed.registration.reader_factory_ref.create(reader_request).reader
    artifact_verifier.bind(reader=reader, foundation=composed.foundation)
    api = APIHost(HostPorts(
        training=composed.training,
        runs=composed.runs,
        artifacts=None,
        evaluation=None,
        chat=None,
        data=None,
        pipelines=None,
        clock=clock,
    ))
    return ModalChatHost(
        api,
        composed,
        reader,
        authorization,
        foundation_ports,
        stores,
        artifact_verifier,
        reader_authority,
        observation_auth,
        log_auth,
        clock,
    )


__all__ = ["ModalChatHost", "ModalChatHostError", "compose_modal_chat_host"]
