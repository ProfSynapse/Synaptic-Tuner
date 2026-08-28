"""Production same-process composition for the Docker v1 coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import secrets
from threading import RLock

from synaptic_tuner.api.v1.docker import (
    DockerCoordinatorHostPortsV1,
    DockerSameProcessLaunchV1,
)
from synaptic_tuner.api.v1.planning import ProviderPlanContextV1, TrainingPlan
from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef
from tuner.execution.coordinator_v1.coordinator import (
    ReconciliationGrantSlotV1,
    TrainingCoordinatorV1,
)
from tuner.execution.coordinator_v1.foundation import (
    ComposedEffectFoundationV1,
    FoundationRecordAssessmentAuthorityV1,
)
from tuner.execution.coordinator_v1.model import WorkflowRecordV1
from tuner.execution.coordinator_v1.stores import (
    InMemoryExecutionGrantStoreV1,
    InMemoryPreparationStoreV1,
    InMemoryReconciliationGrantStoreV1,
    InMemoryWorkflowStoreV1,
)
from tuner.execution.foundation_v2.authority import (
    GrantAuthorityV2,
    ReconciliationGrantContentV1,
)
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.receipts import (
    InvalidEvidenceAuthorityV2,
    ReceiptAuthorityV2,
)
from tuner.execution.foundation_v2.reconciliation import ReconciliationServiceV1
from tuner.execution.foundation_v2.repository import EffectRecordV2, InMemoryEffectRepositoryV2

from .effects import (
    DockerEffectExecutorV1,
    DockerExecutorResolverV1,
    DockerReconciliationAdapterV1,
    DockerReconciliationResolverV1,
)
from .model import (
    AuthenticatedDockerAbsenceV1,
    AuthenticatedDockerCancellationAbsenceV1,
    AuthenticatedDockerCommandBindingV1,
    DockerCommandBindingV1,
    DockerEffectIdentityV1,
)
from .preparation import DockerBindingResolverV1, DockerPreparationMaterializerV1


class _LaunchClockV1:
    __slots__ = ("_epoch", "_iso")

    def __init__(self, launch: DockerSameProcessLaunchV1):
        self._iso = launch.preflight.checked_at
        self._epoch = int(datetime.fromisoformat(self._iso.replace("Z", "+00:00")).timestamp())

    def now_epoch(self) -> int:
        return self._epoch

    def now_iso(self) -> str:
        return self._iso


class _SinglePlanningV1:
    __slots__ = ("_profile",)

    def __init__(self, launch: DockerSameProcessLaunchV1):
        self._profile = launch.profile

    def describe(self, provider: ProviderRef):
        if type(provider) is not ProviderRef or provider != self._profile.provider:
            raise ValueError("Docker provider differs")
        return self._profile.descriptor


class _SinglePlanStoreV1:
    __slots__ = ("_context", "_plan")

    def __init__(self, launch: DockerSameProcessLaunchV1):
        self._context = ProviderPlanContextV1.from_dict(launch.context.to_dict())
        self._plan = TrainingPlan.from_dict(launch.plan.to_dict())

    def get_plan(self, fingerprint: str):
        return self._plan if fingerprint == self._plan.plan_fingerprint else None

    def get_context(self, digest: str):
        return self._context if digest == self._context.provider_context_digest else None


class _SingleRunIdentityV1:
    __slots__ = ("_plan", "_run")

    def __init__(self, launch: DockerSameProcessLaunchV1):
        self._plan = launch.plan
        self._run = launch.run

    def for_plan(self, plan: TrainingPlan):
        if type(plan) is not TrainingPlan or plan != self._plan:
            raise ValueError("Docker plan differs")
        return TrainingRunRef.from_dict(self._run.to_dict())


class _FoundationAuthenticatorV1:
    __slots__ = ("_grants", "_receipts", "_invalid")

    def __init__(self, grants, receipts, invalid):
        self._grants = grants
        self._receipts = receipts
        self._invalid = invalid

    def authenticate_grant(self, grant, command_bytes):
        return self._grants.authenticate(grant, command_bytes)

    def authenticate_receipt(self, receipt):
        return self._receipts.verify(receipt)

    def authenticate_invalid_evidence(self, evidence):
        return self._invalid.verify(evidence)


class _RejectUnusedReadAuthorityV1:
    def authenticate(self, value):
        return False

    def verify(self, value):
        return False


@dataclass(frozen=True, slots=True)
class _QuiescenceProofV1:
    effect_id: str
    command_digest: str
    record_digest: str
    epoch: int
    tag: str

    @property
    def proof_digest(self):
        return domain_digest("synaptic-docker-same-process-quiescence/v1", canonical_bytes({
            "effect_id": self.effect_id,
            "command_digest": self.command_digest,
            "record_digest": self.record_digest,
            "epoch": self.epoch,
            "tag": self.tag,
        }))


class _SameProcessFoundationVerifierV1:
    __slots__ = ("_key", "_evidence")

    def __init__(self, key: bytes, evidence_authority):
        self._key = key
        self._evidence = evidence_authority

    def _tag(self, effect_id, command_digest, record_digest, epoch):
        message = canonical_bytes({
            "effect_id": effect_id, "command_digest": command_digest,
            "record_digest": record_digest, "epoch": epoch,
        })
        return hmac.new(self._key, b"docker-same-process-quiescence-v1\0" + message, hashlib.sha256).hexdigest()

    def issue_quiescence(self, record: EffectRecordV2, epoch: int):
        command = parse_exact_command(record.command_bytes)
        return _QuiescenceProofV1(
            command.operation.effect.effect_id, command.digest, record.record_digest,
            epoch, self._tag(
                command.operation.effect.effect_id, command.digest,
                record.record_digest, epoch,
            ),
        )

    def verify_quiescence(self, proof, record, *, now_epoch):
        try:
            command = parse_exact_command(record.command_bytes)
            expected = self._tag(
                command.operation.effect.effect_id, command.digest,
                record.record_digest, proof.epoch,
            )
            return (
                type(proof) is _QuiescenceProofV1
                and proof.effect_id == command.operation.effect.effect_id
                and proof.command_digest == command.digest
                and proof.record_digest == record.record_digest
                and proof.epoch <= now_epoch
                and hmac.compare_digest(proof.tag, expected)
            )
        except Exception:
            return False

    def verify_finality(self, proof, record, receipt, *, now_epoch):
        try:
            if receipt.content.finality_proof_digest != proof.proof_digest:
                return False
            if type(proof) is AuthenticatedDockerCancellationAbsenceV1:
                return self._evidence.authenticate_cancellation_absence(proof) is True
            if type(proof) is AuthenticatedDockerAbsenceV1:
                return self._evidence.authenticate_absence(proof) is True
            return False
        except Exception:
            return False


class _TrustedQuiescenceEvidenceV1:
    __slots__ = ("_repository", "_verifier")

    def __init__(self, repository, verifier):
        self._repository = repository
        self._verifier = verifier

    def obtain(self, request, *, now_epoch):
        record = self._repository.get(request.effect_id)
        if (type(record) is not EffectRecordV2
                or record.record_digest != request.foundation_record_digest):
            raise ValueError("foundation record differs")
        return self._verifier.issue_quiescence(record, now_epoch)


class _UnsupportedCancellationsV1:
    def stop_once(self, request):
        raise ValueError("cancel is not exposed by the same-process Docker runtime")

    def lookup(self, request):
        raise ValueError("cancel is not exposed by the same-process Docker runtime")


class _DockerAuthorizationV1:
    __slots__ = (
        "_authority", "_binding_store", "_clock", "_expiry_epoch", "_grants",
        "_launch", "_lock", "_materializer", "_preflight_digest",
        "_requirement_digest", "_published",
    )

    def __init__(self, launch, ports, materializer, grants, clock):
        self._launch = launch
        self._authority = ports.binding_authority
        self._binding_store = ports.binding_store
        self._materializer = materializer
        self._grants = grants
        self._clock = clock
        self._expiry_epoch = int(datetime.fromisoformat(
            launch.preflight.expires_at.replace("Z", "+00:00")
        ).timestamp())
        self._preflight_digest = domain_digest(
            "synaptic-docker-same-process-preflight/v1",
            canonical_bytes(launch.preflight.to_dict()),
        )
        self._requirement_digest = domain_digest(
            "synaptic-docker-same-process-requirements/v1",
            canonical_bytes({
                "authorization": [value.to_dict() for value in launch.preflight.authorization],
            }),
        )
        self._published: dict[str, str] = {}
        self._lock = RLock()

    def commit_preflight(self, plan, preflight):
        if plan != self._launch.plan or preflight != self._launch.preflight:
            raise ValueError("preflight differs")
        return self._preflight_digest

    def _binding(self, command):
        prepared = self._materializer.prepared(command.preparation)
        identity = DockerEffectIdentityV1(
            command.digest, command.operation.effect.effect_id,
            command.operation.effect.kind.value, prepared,
        )
        if identity.effect_kind not in {"stage", "submit"}:
            raise ValueError("same-process runtime exposes stage and submit only")
        return DockerCommandBindingV1(identity, command.canonical_bytes)

    def issue_effect_grant(self, command_bytes, *, preflight_digest, now_epoch):
        if preflight_digest != self._preflight_digest or now_epoch != self._clock.now_epoch():
            raise ValueError("grant launch binding differs")
        command = parse_exact_command(command_bytes)
        expected = self._binding(command)
        issued = self._authority.issue(self._binding(command))
        if (type(issued) is not AuthenticatedDockerCommandBindingV1
                or issued.content != expected
                or issued.binding_digest != expected.binding_digest
                or issued.authority_ref != self._binding_store.authority_ref
                or issued.key_ref != self._binding_store.key_ref):
            raise ValueError("issued Docker binding differs")
        retained = self._binding_store.publish_once(issued)
        if retained.content != expected or retained.binding_digest != expected.binding_digest:
            raise ValueError("retained Docker binding differs")
        kind = expected.effect_kind
        with self._lock:
            previous = self._published.get(kind)
            if previous is not None and previous != command.digest:
                raise ValueError("effect kind binding conflict")
            self._published[kind] = command.digest
        return self._grants.issue(
            command.canonical_bytes,
            grant_ref=f"docker-{kind}-{command.digest[:24]}",
            policy_digest=self._preflight_digest,
            requirement_digest=self._requirement_digest,
            not_before_epoch=self._clock.now_epoch(),
            expires_at_epoch=self._expiry_epoch,
        )

    def issue_reconciliation_grant(self, record, binding, *, slot, now_epoch):
        if type(slot) is not ReconciliationGrantSlotV1 or now_epoch != self._clock.now_epoch():
            raise ValueError("reconciliation slot differs")
        command = parse_exact_command(record.command_bytes)
        content = ReconciliationGrantContentV1(
            f"docker-reconcile-{command.digest[:20]}-{slot.generation}-{slot.ownership_epoch}",
            command.digest, command.operation.effect.effect_id,
            command.preparation.preparation_digest,
            self._launch.profile.adapter_descriptor.digest,
            self._launch.profile.provider.provider_id,
            self._launch.profile.provider.profile_ref,
            self._launch.profile.scope.account_ref,
            self._launch.profile.scope.namespace_ref,
            "docker-same-process-owner", slot.generation, slot.ownership_epoch,
            self._preflight_digest, self._requirement_digest,
            self._clock.now_epoch(), self._expiry_epoch,
            self._grants.epoch, self._grants.revocation_generation,
        )
        return self._grants.issue_reconciliation(content)

    def binding(self, effect_kind: str):
        if type(effect_kind) is not str or effect_kind not in {"stage", "submit"}:
            raise ValueError("effect kind must be stage or submit")
        with self._lock:
            digest = self._published.get(effect_kind)
        if digest is None:
            raise LookupError("Docker binding has not been published")
        retained = self._binding_store.resolve(digest)
        if retained.content.effect_kind != effect_kind:
            raise ValueError("Docker binding kind differs")
        return retained


class _DockerSameProcessRuntimeV1:
    __slots__ = ("_authorization", "_coordinator", "_launch")

    def __init__(self, launch, coordinator, authorization):
        self._launch = launch
        self._coordinator = coordinator
        self._authorization = authorization

    def start(self) -> WorkflowRecordV1:
        return self._coordinator.start(self._launch.plan, self._launch.preflight)

    def reconcile(self) -> WorkflowRecordV1:
        return self._coordinator.reconcile(self._launch.run)

    def binding(self, effect_kind: str):
        return self._authorization.binding(effect_kind)


def compose_docker_same_process_coordinator_v1(
    launch: DockerSameProcessLaunchV1,
    ports: DockerCoordinatorHostPortsV1,
):
    if type(launch) is not DockerSameProcessLaunchV1 or type(ports) is not DockerCoordinatorHostPortsV1:
        raise TypeError("exact Docker same-process launch and host ports required")
    clock = _LaunchClockV1(launch)
    materializer = DockerPreparationMaterializerV1(launch.profile)
    grants = GrantAuthorityV2("docker-same-process-grants", secrets.token_bytes(32))
    receipts = ReceiptAuthorityV2("docker-same-process-receipts", secrets.token_bytes(32))
    invalid = InvalidEvidenceAuthorityV2("docker-same-process-invalid", secrets.token_bytes(32))
    foundation_auth = _FoundationAuthenticatorV1(grants, receipts, invalid)
    verifier = _SameProcessFoundationVerifierV1(
        secrets.token_bytes(32), ports.evidence_authority,
    )
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, verifier, verifier, grants,
    )
    assessments = FoundationRecordAssessmentAuthorityV1(
        "docker-same-process-assessments", "docker-same-process-assessment-key",
        secrets.token_bytes(32), assessor_ref="docker-same-process-assessor",
        assessor_version="1.0.0", clock=clock, receipt_authority=receipts,
        invalid_evidence_authority=invalid, grant_authority=grants,
    )
    cancellations = _UnsupportedCancellationsV1()
    executor = DockerEffectExecutorV1(
        launch.profile, ports.binding_store, ports.binding_authority,
        ports.image_inventory, ports.source_seals, ports.control,
        cancellations, ports.evidence_authority,
    )
    adapter = DockerReconciliationAdapterV1(
        launch.profile, ports.binding_store, ports.binding_authority,
        ports.control, ports.source_seals, cancellations,
        ports.evidence_authority,
    )
    broker = EffectBrokerV2(
        repository, DockerExecutorResolverV1(executor), grants, receipts, invalid,
    )
    reconciliation = ReconciliationServiceV1(
        repository, grants, DockerReconciliationResolverV1(adapter),
        receipts, invalid,
    )
    foundation = ComposedEffectFoundationV1(
        repository, broker, reconciliation, grant_authority=grants,
        receipt_authority=receipts, invalid_evidence_authority=invalid,
        assessment_authority=assessments,
        trusted_quiescence_evidence=_TrustedQuiescenceEvidenceV1(repository, verifier),
    )
    authorization = _DockerAuthorizationV1(
        launch, ports, materializer, grants, clock,
    )
    unused = _RejectUnusedReadAuthorityV1()
    coordinator = TrainingCoordinatorV1(
        _SinglePlanningV1(launch), _SinglePlanStoreV1(launch),
        InMemoryWorkflowStoreV1(foundation_auth, assessments, unused, unused),
        InMemoryPreparationStoreV1(), InMemoryExecutionGrantStoreV1(grants),
        InMemoryReconciliationGrantStoreV1(grants),
        DockerBindingResolverV1(launch.profile), materializer, authorization,
        foundation, foundation_auth, clock, _SingleRunIdentityV1(launch),
    )
    return _DockerSameProcessRuntimeV1(launch, coordinator, authorization)


__all__ = ["compose_docker_same_process_coordinator_v1"]
