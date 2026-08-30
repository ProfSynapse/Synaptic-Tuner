"""Public-API Modal operations over host-owned durable state.

This module owns reusable orchestration and verification.  It deliberately
defines only persistence protocols: the consuming project supplies the actual
database, grants, credentials, and authenticated Modal client.
"""

from __future__ import annotations

import hashlib
import base64
import binascii
import json
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, runtime_checkable

from synaptic_tuner.api.v1.execution import (
    ArtifactRef,
    ArtifactState,
    AuthorizationRequirement,
    ErrorCode,
    ExecutionError,
    ExecutionGrant,
    RunRef,
    RunState,
    RunStatus,
)
from synaptic_tuner.api.v1.host import HostPorts
from synaptic_tuner.api.v1.training import (
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    ResolvedTrainingRequest,
    TrainingOutcome,
    TrainingPlan,
    TrainingPreflight,
    TrainingRequest,
    TrainingSubmission,
    RuntimeSpec,
)
from synaptic_tuner.api.v1.sources import ExecutionSourceV1
from tuner.execution._effect_executor import _ProviderEffectExecutor
from tuner.execution.broker import MutationBroker, MutationCommandV1
from tuner.execution.contracts import (
    AttemptDisposition,
    AttemptAdmission,
    EffectDisposition,
    EffectIdentity,
    EffectKind,
    EffectObservation,
    EffectState,
    ExecutionScope,
    EventCode,
    GrantBinding,
    LifecycleEvent,
    LifecyclePhase,
    LifecycleRecord,
    LifecycleRepository,
    MessageCode,
    ProviderRunPhase,
    VerificationStatus,
    RunAlreadyExists,
    digest,
    safe_ref,
)
from tuner.execution.operation import ModalStageTargetV1, OperationBindingV1
from tuner.execution.service import LifecycleService
from tuner.execution.lifecycle import apply_event, initial_record
from tuner.project.context import ProjectContext
from tuner.runtime.artifacts import ArtifactEntry, ArtifactInventory
from tuner.runtime.dispatch import ProcessResult
from tuner.runtime.verification import (
    ArtifactReadError,
    VerificationService,
    VerificationStatus as RuntimeVerificationStatus,
    WorkloadBindingVerifier,
)
from tuner.training.methods.sft import SFT_ARTIFACT_CONTRACT
from tuner.training.recipes import CompiledWorkload
from tuner.training.recipes import RecipeRegistry
from tuner.training.service import TrainingService

from .binding import ModalClientBinding, Readiness, readiness_report
from .bundle import ModalExecutionBundleV1
from .config import ModalProviderProfileV1, ModalRuntimeLockV1
from .contracts import StageReceiptV1, canonical_json, operation_path, sha
from .control import (
    StageControlPlane,
    StageExpectationV1,
    TerminalControlPlane,
    TerminalExpectationV1,
)
from .facade import ExplicitModal154ReadFacade, ModalFunctionCallState
from .logs import LogControlPlane, LogExpectationV1
from .manifest import CompletionControlPlane, CompletionExpectationV1
from .mutation import _ExplicitModal154FunctionMutator
from .resolution import VerifiedModalDeploymentIdentityV1
from .staging import StageMaterialV1, _ExplicitModal154VolumeWriter, prepare_modal_stage


MODAL_PLAN_CONTEXT_SCHEMA = "synaptic-modal-plan-context/v1"


def _closed(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} contains missing or unknown fields")
    return dict(value)


def _resource_digest(plan: TrainingPlan) -> str:
    return sha(
        canonical_json(
            {
                "accelerator": plan.resources.accelerator,
                "accelerator_count": plan.resources.accelerator_count,
                "timeout_seconds": plan.resources.timeout_seconds,
            }
        )
    )


def _secret_requirements_digest(profile: ModalProviderProfileV1) -> str:
    return profile.secret_requirements_digest


def _provider_runtime_requirements_digest(
    runtime_lock: ModalRuntimeLockV1,
    profile: ModalProviderProfileV1,
    deployment,
) -> str:
    return profile.provider_runtime_requirements_digest(
        runtime_lock,
        runtime_environment=deployment.runtime_environment,
        accelerator=deployment.accelerator,
        timeout_seconds=deployment.timeout_seconds,
        max_retries=deployment.max_retries,
    )


@dataclass(frozen=True, slots=True)
class ModalPlanContextV1:
    """Non-secret, restart-safe provider context bound into a public plan."""

    project_ref: str
    profile: str
    deployment: VerifiedModalDeploymentIdentityV1
    binding: ModalClientBinding
    control_volume_id: str
    artifact_volume_id: str
    key_ref: str
    quote_digest: str
    quote_expires_at: str
    maximum_cost_minor_units: int
    currency: str
    effect_id: str
    effect_key: str
    artifact_slot_ref: str
    invocation_nonce: str
    generation: int
    resource_digest: str

    @staticmethod
    def digest_resources(resources: ResourceSpec) -> str:
        """Return the canonical Modal v1 digest for public resource inputs."""
        if not isinstance(resources, ResourceSpec):
            raise TypeError("resources must be a ResourceSpec")
        return sha(
            canonical_json(
                {
                    "accelerator": resources.accelerator,
                    "accelerator_count": resources.accelerator_count,
                    "timeout_seconds": resources.timeout_seconds,
                }
            )
        )

    def __post_init__(self) -> None:
        for name in (
            "project_ref", "profile", "control_volume_id", "artifact_volume_id",
            "key_ref", "effect_id", "effect_key", "artifact_slot_ref",
            "invocation_nonce",
        ):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        if type(self.deployment) is not VerifiedModalDeploymentIdentityV1:
            raise TypeError("deployment must be a verified Modal identity")
        if type(self.binding) is not ModalClientBinding:
            raise TypeError("binding must be a ModalClientBinding")
        if self.binding != ModalClientBinding(
            self.deployment.selection.account_ref,
            self.deployment.selection.workspace_ref,
            self.deployment.selection.environment_ref,
            self.deployment.selection.client_ref,
            self.deployment.selection.sdk_version,
        ):
            raise ValueError("Modal plan context binding differs from deployment")
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("Modal volumes must be distinct")
        object.__setattr__(self, "quote_digest", digest(self.quote_digest, "quote_digest"))
        from tuner.execution.contracts import timestamp
        object.__setattr__(
            self,
            "quote_expires_at",
            timestamp(self.quote_expires_at, "quote_expires_at"),
        )
        object.__setattr__(
            self, "resource_digest", digest(self.resource_digest, "resource_digest")
        )
        if (
            type(self.maximum_cost_minor_units) is not int
            or self.maximum_cost_minor_units < 0
        ):
            raise ValueError("maximum cost must be a non-negative exact integer")
        currency = safe_ref(self.currency.upper(), "currency")
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("currency must be an ISO three-letter code")
        object.__setattr__(self, "currency", currency)
        if type(self.generation) is not int or not 1 <= self.generation <= 2**31 - 1:
            raise ValueError("generation must be a bounded exact integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MODAL_PLAN_CONTEXT_SCHEMA,
            "project_ref": self.project_ref,
            "profile": self.profile,
            "deployment": self.deployment.to_dict(),
            "binding": {
                "account_ref": self.binding.account_ref,
                "workspace_ref": self.binding.workspace_ref,
                "environment_ref": self.binding.environment_ref,
                "client_ref": self.binding.client_ref,
                "sdk_version": self.binding.sdk_version,
            },
            "volumes": {
                "control_volume_id": self.control_volume_id,
                "artifact_volume_id": self.artifact_volume_id,
            },
            "authority": {
                "key_ref": self.key_ref,
                "quote_digest": self.quote_digest,
                "quote_expires_at": self.quote_expires_at,
                "maximum_cost_minor_units": self.maximum_cost_minor_units,
                "currency": self.currency,
            },
            "operation": {
                "effect_id": self.effect_id,
                "effect_key": self.effect_key,
                "artifact_slot_ref": self.artifact_slot_ref,
                "invocation_nonce": self.invocation_nonce,
                "generation": self.generation,
            },
            "resource_digest": self.resource_digest,
        }

    @classmethod
    def from_document(cls, document: CanonicalDocument) -> "ModalPlanContextV1":
        if not isinstance(document, CanonicalDocument):
            raise TypeError("execution context must be a CanonicalDocument")
        root = _closed(
            document.to_dict(),
            {
                "schema_version", "project_ref", "profile", "deployment",
                "binding", "volumes", "authority", "operation", "resource_digest",
            },
            "Modal plan context",
        )
        if root["schema_version"] != MODAL_PLAN_CONTEXT_SCHEMA:
            raise ValueError("unsupported Modal plan context")
        binding = _closed(
            root["binding"],
            {"account_ref", "workspace_ref", "environment_ref", "client_ref", "sdk_version"},
            "Modal binding",
        )
        volumes = _closed(
            root["volumes"], {"control_volume_id", "artifact_volume_id"}, "Modal volumes"
        )
        authority = _closed(
            root["authority"],
            {
                "key_ref", "quote_digest", "quote_expires_at",
                "maximum_cost_minor_units", "currency",
            },
            "Modal authority",
        )
        operation = _closed(
            root["operation"],
            {"effect_id", "effect_key", "artifact_slot_ref", "invocation_nonce", "generation"},
            "Modal operation identity",
        )
        return cls(
            project_ref=root["project_ref"],
            profile=root["profile"],
            deployment=VerifiedModalDeploymentIdentityV1.from_dict(root["deployment"]),
            binding=ModalClientBinding(**binding),
            **volumes,
            **authority,
            **operation,
            resource_digest=root["resource_digest"],
        )


@dataclass(frozen=True, slots=True)
class ModalDurablePreparationV1:
    """Exact non-secret material the host commits before Volume mutation."""

    public_plan_fingerprint: str
    context: ModalPlanContextV1
    operation: OperationBindingV1
    stage: StageMaterialV1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "public_plan_fingerprint",
            digest(self.public_plan_fingerprint, "public_plan_fingerprint"),
        )
        if type(self.context) is not ModalPlanContextV1:
            raise TypeError("context must be ModalPlanContextV1")
        if type(self.operation) is not OperationBindingV1:
            raise TypeError("operation must be OperationBindingV1")
        if type(self.stage) is not StageMaterialV1:
            raise TypeError("stage must be StageMaterialV1")
        if self.stage.expectation.operation != self.operation:
            raise ValueError("stage material differs from durable operation")
        if self.stage.expectation.binding != self.context.binding:
            raise ValueError("stage material differs from durable Modal context")
        if self.public_plan_fingerprint != self.operation.plan_fingerprint:
            raise ValueError("durable plan fingerprint differs from its operation")

    def detached_execution_source(self) -> ExecutionSourceV1:
        """Return the exact staged execution source without exposing bundle internals."""
        try:
            bundle = ModalExecutionBundleV1.parse_transport(self.stage.bundle)
            members = tuple(
                member for member in bundle.members
                if member.name == "execution-source.json"
            )
            if len(members) != 1:
                raise ValueError
            document = json.loads(members[0].content.decode("utf-8"))
            if type(document) is not dict:
                raise ValueError
            return ExecutionSourceV1.from_dict(document)
        except BaseException:
            raise ValueError("durable execution source is unavailable") from None

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": "synaptic-modal-durable-preparation/v1",
                "public_plan_fingerprint": self.public_plan_fingerprint,
                "context": self.context.to_dict(),
                "operation": self.operation.to_dict(),
                "stage": {
                    "bundle_base64": base64.b64encode(self.stage.bundle).decode("ascii"),
                    "claim_base64": base64.b64encode(self.stage.claim).decode("ascii"),
                    "claim_tag_base64": base64.b64encode(self.stage.claim_tag).decode("ascii"),
                },
            }
        )

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> "ModalDurablePreparationV1":
        if not isinstance(value, bytes) or not value or len(value) > 16 * 1024 * 1024:
            raise ValueError("Modal preparation must be bounded canonical JSON bytes")
        try:
            document = json.loads(value.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Modal preparation must be canonical JSON") from exc
        expected = {
            "schema_version", "public_plan_fingerprint", "context", "operation", "stage"
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise ValueError("Modal preparation contains missing or unknown fields")
        if document["schema_version"] != "synaptic-modal-durable-preparation/v1":
            raise ValueError("unsupported Modal preparation schema")
        stage = document["stage"]
        if not isinstance(stage, dict) or set(stage) != {
            "bundle_base64", "claim_base64", "claim_tag_base64"
        }:
            raise ValueError("Modal stage material is malformed")

        def decode(name: str) -> bytes:
            encoded = stage[name]
            if not isinstance(encoded, str) or not encoded.isascii():
                raise ValueError("Modal stage Base64 is invalid")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Modal stage Base64 is invalid") from exc
            if not decoded or base64.b64encode(decoded).decode("ascii") != encoded:
                raise ValueError("Modal stage Base64 is not canonical")
            return decoded

        context = ModalPlanContextV1.from_document(
            CanonicalDocument.from_mapping(document["context"])
        )
        operation = OperationBindingV1.from_dict(document["operation"])
        bundle = decode("bundle_base64")
        claim = decode("claim_base64")
        claim_tag = decode("claim_tag_base64")
        material = StageMaterialV1(
            StageExpectationV1.from_stage(
                operation, context.binding, claim=claim, bundle=bundle
            ),
            bundle,
            claim,
            claim_tag,
        )
        result = cls(
            document["public_plan_fingerprint"], context, operation, material
        )
        if result.canonical_bytes != value:
            raise ValueError("Modal preparation is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ModalPreparedRunV1:
    """One atomically admitted READY lifecycle and its exact Modal preparation."""

    record: LifecycleRecord
    preparation: ModalDurablePreparationV1

    def __post_init__(self) -> None:
        if type(self.record) is not LifecycleRecord:
            raise TypeError("record must be exact LifecycleRecord")
        if type(self.preparation) is not ModalDurablePreparationV1:
            raise TypeError("preparation must be exact ModalDurablePreparationV1")
        try:
            record = LifecycleRecord.from_canonical_bytes(self.record.canonical_bytes)
            preparation = ModalDurablePreparationV1.from_canonical_bytes(
                self.preparation.canonical_bytes
            )
        except BaseException:
            raise ValueError("Modal prepared run is not canonical") from None
        events = record.events
        expected_codes = (
            EventCode.RUN_PLANNED,
            EventCode.AUTHORITY_ACCEPTED,
            EventCode.PREPARATION_STARTED,
            EventCode.PREPARATION_COMPLETED,
        )
        operation = preparation.operation
        context = preparation.context
        binding = record.grant_binding
        if (
            record.revision != 4
            or record.phase is not LifecyclePhase.READY
            or record.verification is not VerificationStatus.NOT_READY
            or record.effects
            or len(events) != 4
            or tuple(event.code for event in events) != expected_codes
            or len({event.occurred_at for event in events}) != 1
            or record.updated_at != events[0].occurred_at
            or events[0].message_code is not MessageCode.PLANNED
            or events[1].message_code is not MessageCode.AUTHORITY_BOUND
            or events[2].message_code is not MessageCode.PREPARING
            or events[3].message_code is not MessageCode.READY
            or record.message_code is not MessageCode.READY
            or events[0].effect is not None
            or events[1].effect is not None
            or events[2].effect is not None
            or events[3].effect is not None
            or events[0].grant_binding is not None
            or events[2].grant_binding is not None
            or events[3].grant_binding is not None
            or type(binding) is not GrantBinding
            or events[1].grant_binding != binding
            or binding.operation != operation
            or binding.grant_ref != operation.grant_ref
            or record.project_ref != operation.project_ref
            or record.run_id != operation.run_id
            or context.project_ref != operation.project_ref
            or context.effect_id != operation.effect.effect_id
            or context.effect_key != operation.effect.effect_key
            or context.artifact_slot_ref != operation.stage_target.artifact_slot_ref
            or context.invocation_nonce != operation.invocation_nonce
            or context.generation != operation.stage_target.generation
            or context.control_volume_id != operation.stage_target.control_volume_id
            or context.artifact_volume_id != operation.stage_target.artifact_volume_id
            or context.key_ref != operation.stage_target.key_ref
            or operation.effect.kind is not EffectKind.SUBMIT
            or operation.effect.scope.provider != "modal"
            or operation.effect.scope.account_ref != context.binding.account_ref
            or operation.effect.scope.namespace_ref
            != context.binding.environment_ref
            or operation.deployment_attestation_digest
            != context.deployment.attestation_digest
            or operation.resource_digest != context.resource_digest
            or operation.quote_digest != context.quote_digest
        ):
            raise ValueError("Modal prepared run identities disagree")
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "preparation", preparation)


def _prepared_record_prefix(record: LifecycleRecord) -> LifecycleRecord:
    if type(record) is not LifecycleRecord or record.revision < 4:
        raise ValueError("Modal prepared lifecycle is unavailable")
    events = record.events[:4]
    return LifecycleRecord(
        run_id=record.run_id,
        project_ref=record.project_ref,
        revision=4,
        phase=LifecyclePhase.READY,
        verification=VerificationStatus.NOT_READY,
        updated_at=events[-1].occurred_at,
        message_code=events[-1].message_code,
        events=events,
        effects=(),
        grant_binding=events[1].grant_binding,
    )


def _snapshot_plan(value: TrainingPlan) -> TrainingPlan:
    """Detach all public plan inputs before the first host callback."""
    if type(value) is not TrainingPlan:
        raise TypeError("plan must be exact TrainingPlan")
    return TrainingPlan(
        ExecutionSourceV1.from_dict(value.execution_source.to_dict()),
        CanonicalDocument(value.execution_context.canonical_json),
        CanonicalDocument(value.resolved_config.canonical_json),
        CanonicalDocument(value.workload.canonical_json),
        RuntimeSpec(
            value.runtime.image,
            value.runtime.dependency_lock_digest,
            value.runtime.python_version,
        ),
        ResourceSpec(
            value.resources.accelerator,
            value.resources.accelerator_count,
            value.resources.timeout_seconds,
        ),
        ArtifactPolicy(
            tuple(value.artifact_policy.required_kinds),
            value.artifact_policy.retain_checkpoints,
        ),
    )


def _snapshot_preflight(value: TrainingPreflight) -> TrainingPreflight:
    if type(value) is not TrainingPreflight:
        raise TypeError("preflight must be exact TrainingPreflight")
    authorization = tuple(
        AuthorizationRequirement(
            item.operation,
            item.paid_effect,
            item.maximum_cost_minor_units,
            item.currency,
        )
        for item in value.authorization
    )
    errors = tuple(
        ExecutionError(item.code, item.message, item.retryable)
        for item in value.errors
    )
    return TrainingPreflight(
        value.plan_fingerprint,
        value.ready,
        value.checked_at,
        value.expires_at,
        authorization,
        errors,
    )


def _snapshot_grant(value: ExecutionGrant) -> ExecutionGrant:
    if type(value) is not ExecutionGrant:
        raise TypeError("grant must be exact ExecutionGrant")
    return ExecutionGrant(value.grant_ref)


def _callable_identity(owner: object, name: str) -> tuple[object | None, object]:
    value = getattr(owner, name)
    return (getattr(value, "__self__", None), getattr(value, "__func__", value))


def _closed_start_failure() -> None:
    raise RuntimeError("host Modal start failed") from None


def _closed_readback_failure() -> None:
    raise RuntimeError("host Modal prepared-run readback failed") from None


def _validated_current_record(
    value: object,
    *,
    prepared: ModalPreparedRunV1,
    expected_effect: EffectIdentity,
    require_ready_grant_ref: str | None,
) -> LifecycleRecord:
    if type(value) is not LifecycleRecord:
        _closed_readback_failure()
    failed = False
    try:
        current = LifecycleRecord.from_canonical_bytes(value.canonical_bytes)
        prefix = _prepared_record_prefix(current)
    except Exception:
        failed = True
    if failed:
        _closed_readback_failure()
    if (
        prefix != prepared.record
        or current.project_ref != prepared.record.project_ref
        or current.run_id != prepared.record.run_id
    ):
        _closed_readback_failure()
    replay_failed = False
    try:
        replayed = LifecycleRecord.from_canonical_bytes(
            prepared.record.canonical_bytes
        )
        for source_event in current.events[4:]:
            if type(source_event) is not LifecycleEvent:
                raise TypeError("lifecycle event must be exact")
            detached_event = LifecycleEvent.from_dict(source_event.to_dict())
            replayed = apply_event(replayed, detached_event)
        replayed = LifecycleRecord.from_canonical_bytes(replayed.canonical_bytes)
    except Exception:
        replay_failed = True
    if replay_failed or replayed.canonical_bytes != current.canonical_bytes:
        _closed_readback_failure()
    if current.phase is LifecyclePhase.READY:
        if require_ready_grant_ref is None or (
            current.grant_binding is None
            or current.grant_binding.operation != prepared.preparation.operation
            or current.grant_binding.grant_ref != require_ready_grant_ref
        ):
            _closed_readback_failure()
    else:
        expected = prepared.preparation.stage.expectation
        receipt = StageReceiptV1(
            expected.effect.effect_id,
            expected.operation_binding_digest,
            expected.control_volume_id,
            expected.artifact_volume_id,
            expected.claim_digest,
            expected.bundle_digest,
        )
        command_failed = False
        try:
            command = MutationCommandV1.from_stage(
                prepared.preparation.operation, receipt
            )
        except Exception:
            command_failed = True
        if command_failed:
            _closed_readback_failure()
        retained = tuple(
            effect for effect in current.effects if effect.identity == expected_effect
        )
        if (
            len(retained) != 1
            or retained[0].identity.kind is not EffectKind.SUBMIT
            or retained[0].command_digest != command.digest
            or retained[0].canonical_command != command.canonical_bytes
        ):
            _closed_readback_failure()
        compatible = {
            LifecyclePhase.SUBMITTING: {EffectState.ATTEMPTED},
            LifecyclePhase.QUEUED: {EffectState.FOUND},
            LifecyclePhase.RUNNING: {EffectState.FOUND},
            LifecyclePhase.VERIFYING: {EffectState.FOUND},
            LifecyclePhase.SUCCEEDED: {EffectState.FOUND},
            LifecyclePhase.FAILED: {EffectState.FOUND},
            LifecyclePhase.CANCELLING: {EffectState.FOUND},
            LifecyclePhase.CANCELLED: {EffectState.FOUND},
            LifecyclePhase.RECONCILE_REQUIRED: {
                EffectState.FOUND,
                EffectState.INDETERMINATE,
                EffectState.DEFINITELY_ABSENT,
            },
        }
        if retained[0].state not in compatible.get(current.phase, set()):
            _closed_readback_failure()
    return current


def _guarded_broker_execute(
    broker: MutationBroker,
    repository: ModalTrainingRepository,
    command: MutationCommandV1,
    *,
    expected_revision: int,
    guard,
) -> EffectObservation:
    """Execute one broker path with a guard around every external seam."""
    admission_failed = False
    guard()
    try:
        admission = repository.compare_and_consume_attempt(
            command.project_ref,
            command.run_id,
            expected_revision=expected_revision,
            grant_ref=command.grant_ref,
            canonical_command=command,
        )
    except Exception:
        admission_failed = True
    guard()
    if admission_failed:
        _closed_start_failure()
    if type(admission) is not AttemptAdmission:
        _closed_start_failure()
    if admission.disposition is AttemptDisposition.LOOKUP_ONLY:
        effect = admission.effect
        if effect.state is EffectState.FOUND:
            return EffectObservation(
                effect.identity,
                EffectDisposition.FOUND,
                effect.provider_job_ref,
                effect.receipt_digest,
            )
        if effect.state is EffectState.DEFINITELY_ABSENT:
            return EffectObservation(
                effect.identity, EffectDisposition.DEFINITELY_ABSENT
            )
        return EffectObservation(effect.identity, EffectDisposition.INDETERMINATE)

    execution_failed = False
    guard()
    try:
        observation = broker._executor.execute_once(command.canonical_bytes)
    except Exception:
        execution_failed = True
    guard()
    if execution_failed:
        _closed_start_failure()
    if type(observation) is not EffectObservation or observation.identity != command.effect:
        _closed_start_failure()

    outcome_failed = False
    guard()
    try:
        repository.record_attempt_outcome(
            command.project_ref,
            command.run_id,
            expected_revision=admission.record.revision,
            command_digest=command.digest,
            observation=observation,
        )
    except Exception:
        outcome_failed = True
    guard()
    if outcome_failed:
        _closed_start_failure()
    return observation


@runtime_checkable
class ModalTrainingRepository(LifecycleRepository, Protocol):
    """Consuming-project database contract; no implementation lives here."""

    def create_modal_prepared_run(self, value: ModalPreparedRunV1) -> None: ...

    def load_modal_preparation(
        self, project_ref: str, run_id: str
    ) -> ModalDurablePreparationV1 | None: ...

    def load_modal_preparation_by_effect(
        self, effect_id: str
    ) -> ModalDurablePreparationV1 | None: ...


class _ExpectationStore:
    """Derive volatile result expectations from durable preparation + attempt."""

    def __init__(self, repository: ModalTrainingRepository) -> None:
        self._repository = repository

    def _preparation(self, effect_id: str) -> ModalDurablePreparationV1:
        value = self._repository.load_modal_preparation_by_effect(effect_id)
        if type(value) is not ModalDurablePreparationV1:
            raise ValueError("Modal preparation unavailable")
        return value

    def load_modal_expectation(self, effect_id: str) -> StageExpectationV1:
        return self._preparation(effect_id).stage.expectation

    def _terminal_values(self, effect_id: str):
        preparation = self._preparation(effect_id)
        operation = preparation.operation
        record = self._repository.load(operation.project_ref, operation.run_id)
        if not isinstance(record, LifecycleRecord):
            raise ValueError("Modal lifecycle unavailable")
        effect = next((item for item in record.effects if item.identity == operation.effect), None)
        if effect is None or effect.state is not EffectState.FOUND:
            raise ValueError("Modal provider job unavailable")
        if effect.canonical_command is None or effect.command_digest is None:
            raise ValueError("Modal command unavailable")
        command = MutationCommandV1.from_bytes(effect.canonical_command)
        if command.operation != operation or command.digest != effect.command_digest:
            raise ValueError("Modal command binding mismatch")
        context = preparation.context
        return preparation, command, effect.provider_job_ref

    def load_terminal_expectation(self, effect_id: str) -> TerminalExpectationV1:
        preparation, command, job_ref = self._terminal_values(effect_id)
        context = preparation.context
        operation = preparation.operation
        return TerminalExpectationV1(
            context.binding, context.control_volume_id, context.artifact_volume_id,
            context.key_ref, job_ref, effect_id, command.digest,
            operation.plan_fingerprint, operation.deployment_attestation_digest,
            operation.invocation_nonce, context.generation,
        )

    def load_log_expectation(self, effect_id: str) -> LogExpectationV1:
        terminal = self.load_terminal_expectation(effect_id)
        return LogExpectationV1(
            terminal.binding, terminal.control_volume_id, terminal.artifact_volume_id,
            terminal.key_ref, terminal.job_ref, terminal.effect_id,
            terminal.command_digest, terminal.plan_digest,
            terminal.deployment_attestation_digest, terminal.invocation_nonce,
            terminal.generation,
        )

    def load_completion_expectation(self, effect_id: str) -> CompletionExpectationV1:
        preparation, command, job_ref = self._terminal_values(effect_id)
        context = preparation.context
        operation = preparation.operation
        return CompletionExpectationV1(
            context.binding, context.control_volume_id, context.artifact_volume_id,
            operation.stage_target.output_prefix, context.key_ref, job_ref, effect_id,
            command.digest, operation.plan_fingerprint,
            operation.deployment_attestation_digest, operation.invocation_nonce,
            context.generation,
        )


class _DurableModalEffectDriver:
    def __init__(
        self,
        repository: ModalTrainingRepository,
        control: StageControlPlane,
        facade: ExplicitModal154ReadFacade,
    ) -> None:
        self._repository = repository
        self._control = control
        self._facade = facade

    def execute_once(self, canonical_command: bytes):
        command = MutationCommandV1.from_bytes(canonical_command)
        preparation = self._repository.load_modal_preparation_by_effect(
            command.effect.effect_id
        )
        if type(preparation) is not ModalDurablePreparationV1:
            raise RuntimeError("modal_preparation_unavailable")
        expected = preparation.stage.expectation
        receipt = StageReceiptV1(
            expected.effect.effect_id,
            expected.operation_binding_digest,
            expected.control_volume_id,
            expected.artifact_volume_id,
            expected.claim_digest,
            expected.bundle_digest,
        )
        self._control.validate(receipt)
        mutator = _ExplicitModal154FunctionMutator(
            self._facade, preparation.context.deployment
        )
        return mutator.execute_once(canonical_command)


def _build_preparation(
    plan: TrainingPlan,
    context: ModalPlanContextV1,
    grant: ExecutionGrant,
    authenticator,
) -> ModalDurablePreparationV1:
    source = plan.execution_source
    if source.run_id != context.effect_key:
        raise ValueError("Modal operation key must equal the finalized source run ID")
    effect = EffectIdentity(
        context.effect_id,
        context.effect_key,
        EffectKind.SUBMIT,
        ExecutionScope("modal", context.binding.account_ref, context.binding.environment_ref),
    )
    output_prefix = operation_path(context.effect_id, "output")
    target = ModalStageTargetV1(
        context.artifact_slot_ref,
        context.control_volume_id,
        context.artifact_volume_id,
        output_prefix,
        context.generation,
        context.key_ref,
    )
    deployment_bytes = canonical_json(context.deployment.to_dict())
    source_bytes = source.canonical_bytes
    workload_bytes = plan.workload.canonical_json.encode("utf-8")
    workload = plan.workload.to_dict()
    artifacts = workload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Modal SFT workload lacks its artifact contract")
    artifact_bytes = canonical_json(dict(artifacts))
    policy_bytes = canonical_json(
        {
            "schema_version": "synaptic-modal-log-terminal-policy/v1",
            "run_id": source.run_id,
            "effect_id": effect.effect_id,
            "generation": context.generation,
            "control_prefix": operation_path(effect.effect_id, "control"),
            "artifact_prefix": output_prefix,
            "max_log_chunks": 1024,
            "max_chunk_bytes": 65536,
            "max_terminal_bytes": 65536,
        }
    )
    plan_bytes = canonical_json(
        {
            "schema_version": "synaptic-training-plan/v1",
            "run_id": source.run_id,
            "effect_id": effect.effect_id,
            "effect_key": effect.effect_key,
            "provider": "modal",
            "account_ref": context.binding.account_ref,
            "namespace_ref": context.binding.environment_ref,
            "artifact_slot_ref": context.artifact_slot_ref,
            "deployment_digest": sha(deployment_bytes),
            "execution_source_digest": sha(source_bytes),
            "workload_digest": sha(workload_bytes),
            "artifact_contract_digest": sha(artifact_bytes),
            "log_policy_digest": sha(policy_bytes),
            "resource_digest": context.resource_digest,
            "quote_digest": context.quote_digest,
            "secret_requirements_digest": source.secret_requirements_digest,
        }
    )
    environment = dict(source.environment)
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + workload_bytes
    ).hexdigest()
    invocation_bytes = canonical_json(
        {
            "schema_version": "synaptic-modal-invocation-intent/v1",
            "run_id": source.run_id,
            "effect_id": effect.effect_id,
            "plan_digest": sha(plan_bytes),
            "deployment_digest": sha(deployment_bytes),
            "execution_source_digest": sha(source_bytes),
            "workload_digest": sha(workload_bytes),
            "interpreter": source.python_executable,
            "argv": [
                source.python_executable,
                source.roots["engine"] + "/Trainers/sft/runtime_v1.py",
                "--canonical-workload-stdin",
            ],
            "cwd": source.roots["tmp"],
            "environment_digest": sha(canonical_json(environment)),
            "invocation_nonce": context.invocation_nonce,
        }
    )
    members = {
        "artifact-contract.json": artifact_bytes,
        "deployment.json": deployment_bytes,
        "execution-source.json": source_bytes,
        "invocation-intent.json": invocation_bytes,
        "log-terminal-policy.json": policy_bytes,
        "plan.json": plan_bytes,
        "workload.json": workload_bytes,
    }
    operation = OperationBindingV1.from_predecessors(
        project_ref=context.project_ref,
        grant_ref=grant.grant_ref,
        effect=effect,
        stage_target=target,
        member_documents=members,
    )
    operation = replace(operation, plan_fingerprint=plan.fingerprint)
    bundle = ModalExecutionBundleV1.build(
        operation=operation, member_documents=members
    ).transport_base64
    stage = prepare_modal_stage(
        operation, context.binding, bundle, authenticator
    )
    return ModalDurablePreparationV1(plan.fingerprint, context, operation, stage)


class ModalTrainingOperations:
    """The sole Modal implementation behind ``TrainingAPI``."""

    def __init__(
        self,
        *,
        planning: TrainingService,
        context: ProjectContext,
        ports: HostPorts,
        profile: ModalProviderProfileV1,
    ) -> None:
        if not isinstance(planning, TrainingService):
            raise TypeError("planning must be TrainingService")
        if not isinstance(context, ProjectContext) or context.mode != "host":
            raise ValueError("Modal training requires a host project context")
        if not isinstance(ports.lifecycle, ModalTrainingRepository):
            raise TypeError("host lifecycle must implement ModalTrainingRepository")
        if type(profile) is not ModalProviderProfileV1:
            raise TypeError("profile must be ModalProviderProfileV1")
        if type(ports.modal_reads) is not ExplicitModal154ReadFacade:
            raise TypeError("host must inject the exact explicit Modal 1.5.4 facade")
        self._planning = planning
        self._context = context
        self._ports = ports
        self._profile = profile
        self._runtime_lock = ModalRuntimeLockV1.packaged()
        self._repository = ports.lifecycle
        self._facade = ports.modal_reads
        self._expectations = _ExpectationStore(self._repository)
        self._stage_control = StageControlPlane(
            self._expectations, ports.authenticator, self._facade
        )
        self._terminal = TerminalControlPlane(
            self._expectations, ports.authenticator, self._facade
        )
        self._logs = LogControlPlane(
            self._expectations, ports.authenticator, self._facade
        )
        self._completion = CompletionControlPlane(
            self._expectations, ports.authenticator, self._facade,
            self._terminal, self._logs,
        )
        driver = _DurableModalEffectDriver(
            self._repository, self._stage_control, self._facade
        )
        self._broker = MutationBroker(
            self._repository, _ProviderEffectExecutor(driver)
        )
        self._lifecycle = LifecycleService(
            self._repository, clock=ports.clock
        )

    def load(self, document: CanonicalDocument) -> TrainingRequest:
        return self._planning.load(document)

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        return self._planning.resolve(request)

    def plan(self, resolved: ResolvedTrainingRequest) -> TrainingPlan:
        return self._planning.plan(resolved)

    def _context_for(self, plan: TrainingPlan) -> ModalPlanContextV1:
        context = ModalPlanContextV1.from_document(plan.execution_context)
        deployment = context.deployment.selection
        if context.profile != self._profile.profile:
            raise ValueError("training plan selected a different Modal profile")
        self._runtime_lock.validate_selection(deployment)
        if (
            deployment.app_name != self._profile.app_name
            or deployment.function_name != self._profile.function_name
            or deployment.deployment_ref != self._profile.deployment_ref
        ):
            raise ValueError("training plan differs from the configured Modal deployment")
        if plan.execution_source.deployment_member_sha256 != sha(
            canonical_json(context.deployment.to_dict())
        ):
            raise ValueError("execution source does not bind the Modal deployment")
        if context.resource_digest != _resource_digest(plan):
            raise ValueError("Modal resource evidence differs from the training plan")
        if plan.runtime.image != self._runtime_lock.registry_reference:
            raise ValueError("training runtime differs from the Modal deployment")
        expected_secret_requirements = _secret_requirements_digest(self._profile)
        expected_provider_requirements = _provider_runtime_requirements_digest(
            self._runtime_lock, self._profile, deployment
        )
        if (
            plan.runtime.dependency_lock_digest
            != self._runtime_lock.locked_digest("dependency_lock")
            or plan.runtime.python_version != self._runtime_lock.python_version
            or plan.execution_source.python_implementation
            != self._runtime_lock.python_implementation
            or plan.execution_source.python_version != self._runtime_lock.python_version
            or plan.execution_source.python_executable
            != self._runtime_lock.python_executable
            or plan.execution_source.python_executable_digest
            != self._runtime_lock.python_executable_digest
            or deployment.secret_requirements_digest != expected_secret_requirements
            or plan.execution_source.secret_requirements_digest
            != expected_secret_requirements
            or deployment.provider_runtime_requirements_digest
            != expected_provider_requirements
            or plan.execution_source.provider_runtime_requirements_digest
            != expected_provider_requirements
            or plan.resources.accelerator != "A10"
            or plan.resources.accelerator_count != 1
            or plan.resources.timeout_seconds != deployment.timeout_seconds
        ):
            raise ValueError("training runtime or resources differ from Modal v1")
        return context

    def preflight(self, plan: TrainingPlan) -> TrainingPreflight:
        context = self._context_for(plan)
        from datetime import timedelta
        from tuner.execution.evidence import parse_utc
        checked_at = self._ports.clock()
        checked = parse_utc(checked_at)
        evidence_expiry = min(
            parse_utc(context.quote_expires_at),
            parse_utc(context.deployment.expires_at),
        )
        public_expiry = (
            evidence_expiry
            if evidence_expiry > checked
            else checked + timedelta(seconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        requirement = AuthorizationRequirement(
            "training.start", True, context.maximum_cost_minor_units, context.currency
        )
        try:
            if evidence_expiry <= checked:
                raise ValueError("Modal preflight evidence expired")
            if readiness_report(context.binding, self._facade).status is not Readiness.READY:
                raise ValueError("Modal client is not ready")
            observed = self._facade.inspect_deployment(
                app_name=self._profile.app_name,
                function_name=self._profile.function_name,
            )
            if observed != context.deployment.selection:
                raise ValueError("Modal deployment changed")
            if (
                self._facade.volume_name(context.control_volume_id)
                != self._profile.control_volume_ref
                or self._facade.volume_name(context.artifact_volume_id)
                != self._profile.artifact_volume_ref
            ):
                raise ValueError("Modal volume identity changed")
            root = operation_path(context.effect_id) + "/"
            if self._facade.list_prefix(
                context.control_volume_id, root, max_entries=1
            ) or self._facade.list_prefix(
                context.artifact_volume_id, root, max_entries=1
            ):
                raise ValueError("Modal operation slot already exists")
        except Exception:
            return TrainingPreflight(
                plan.fingerprint,
                False,
                checked_at,
                public_expiry,
                (requirement,),
                (ExecutionError(ErrorCode.PREFLIGHT_FAILED, "Modal preflight failed"),),
            )
        return TrainingPreflight(
            plan.fingerprint, True, checked_at, public_expiry, (requirement,)
        )

    def start(
        self,
        plan: TrainingPlan,
        preflight: TrainingPreflight,
        grant: ExecutionGrant,
    ) -> TrainingSubmission:
        snapshot_failed = False
        try:
            sealed_plan = _snapshot_plan(plan)
            sealed_preflight = _snapshot_preflight(preflight)
            sealed_grant = _snapshot_grant(grant)
        except Exception:
            snapshot_failed = True
        if snapshot_failed:
            _closed_start_failure()
        if not sealed_preflight.ready or not sealed_preflight.binds(sealed_plan):
            raise ValueError("start requires the exact ready preflight")

        pinned_port_fields = (
            ("lifecycle", self._repository),
            ("grants", self._ports.grants),
            ("authenticator", self._ports.authenticator),
            ("modal_reads", self._facade),
            ("clock", self._ports.clock),
        )
        pinned_attributes = (
            (self, "_ports", self._ports),
            (self, "_repository", self._repository),
            (self, "_facade", self._facade),
            (self, "_stage_control", self._stage_control),
            (self, "_broker", self._broker),
            (self, "_context", self._context),
            (self, "_profile", self._profile),
            (self, "_runtime_lock", self._runtime_lock),
            (self._expectations, "_repository", self._repository),
            (self._stage_control, "store", self._expectations),
            (self._stage_control, "auth", self._ports.authenticator),
            (self._stage_control, "facade", self._facade),
            (self._broker, "_repository", self._repository),
        )
        pinned_executor = self._broker._executor
        pinned_driver = pinned_executor._driver
        pinned_attributes += (
            (self._broker, "_executor", pinned_executor),
            (pinned_executor, "_driver", pinned_driver),
            (pinned_driver, "_repository", self._repository),
            (pinned_driver, "_control", self._stage_control),
            (pinned_driver, "_facade", self._facade),
        )
        pinned_callables = (
            (self._ports, "clock", _callable_identity(self._ports, "clock")),
            (
                self._repository,
                "create_modal_prepared_run",
                _callable_identity(self._repository, "create_modal_prepared_run"),
            ),
            (self._repository, "load", _callable_identity(self._repository, "load")),
            (
                self._repository,
                "load_modal_preparation",
                _callable_identity(self._repository, "load_modal_preparation"),
            ),
            (
                self._repository,
                "load_modal_preparation_by_effect",
                _callable_identity(
                    self._repository, "load_modal_preparation_by_effect"
                ),
            ),
            (
                self._repository,
                "compare_and_consume_attempt",
                _callable_identity(self._repository, "compare_and_consume_attempt"),
            ),
            (
                self._repository,
                "record_attempt_outcome",
                _callable_identity(self._repository, "record_attempt_outcome"),
            ),
            (self._ports.grants, "bind", _callable_identity(self._ports.grants, "bind")),
            (
                self._ports.authenticator,
                "sign",
                _callable_identity(self._ports.authenticator, "sign"),
            ),
            (
                self._ports.authenticator,
                "verify",
                _callable_identity(self._ports.authenticator, "verify"),
            ),
            (
                self._stage_control,
                "validate",
                _callable_identity(self._stage_control, "validate"),
            ),
            (
                pinned_executor,
                "execute_once",
                _callable_identity(pinned_executor, "execute_once"),
            ),
            (
                pinned_driver,
                "execute_once",
                _callable_identity(pinned_driver, "execute_once"),
            ),
        )

        def guard() -> None:
            guard_failed = False
            try:
                inputs_match = (
                    _snapshot_plan(plan) == sealed_plan
                    and _snapshot_preflight(preflight) == sealed_preflight
                    and _snapshot_grant(grant) == sealed_grant
                )
                ports_match = all(
                    getattr(self._ports, name) is identity
                    for name, identity in pinned_port_fields
                )
                attributes_match = all(
                    getattr(owner, name) is identity
                    for owner, name, identity in pinned_attributes
                )
                callables_match = True
                for owner, name, identity in pinned_callables:
                    current = _callable_identity(owner, name)
                    if current[0] is not identity[0] or current[1] is not identity[1]:
                        callables_match = False
                        break
            except Exception:
                guard_failed = True
            if guard_failed:
                _closed_start_failure()
            if (
                not inputs_match
                or not ports_match
                or not attributes_match
                or not callables_match
            ):
                _closed_start_failure()

        from tuner.execution.contracts import timestamp
        guard()
        clock_failed = False
        try:
            occurred_at = self._ports.clock()
        except Exception:
            clock_failed = True
        if clock_failed:
            _closed_start_failure()
        guard()
        if timestamp(occurred_at, "clock") >= timestamp(
            sealed_preflight.expires_at, "preflight expiry"
        ):
            raise ValueError("training preflight has expired")
        context = self._context_for(sealed_plan)
        guard()
        preparation = _build_preparation(
            sealed_plan, context, sealed_grant, self._ports.authenticator
        )
        guard()
        preparation_failed = False
        try:
            sealed_preparation = ModalDurablePreparationV1.from_canonical_bytes(
                preparation.canonical_bytes
            )
        except Exception:
            preparation_failed = True
        if preparation_failed:
            _closed_start_failure()
        guard()
        binding_failed = False
        try:
            binding = self._ports.grants.bind(
                sealed_grant,
                operation=sealed_preparation.operation,
                requirements=sealed_preflight.authorization,
            )
        except Exception:
            binding_failed = True
        if binding_failed:
            _closed_start_failure()
        guard()
        if type(binding) is not GrantBinding:
            raise ValueError("host grant binding is unavailable")
        binding_snapshot_failed = False
        try:
            binding = GrantBinding.from_dict(binding.to_dict())
        except Exception:
            binding_snapshot_failed = True
        if binding_snapshot_failed:
            _closed_start_failure()
        if (
            binding.operation != sealed_preparation.operation
            or binding.grant_ref != sealed_grant.grant_ref
        ):
            raise ValueError("host grant does not bind the exact Modal operation")
        record = initial_record(
            project_ref=context.project_ref,
            run_id=sealed_plan.execution_source.run_id,
            occurred_at=occurred_at,
        )
        record = apply_event(
            record,
            LifecycleEvent(
                EventCode.AUTHORITY_ACCEPTED,
                occurred_at,
                MessageCode.AUTHORITY_BOUND,
                grant_binding=binding,
            ),
        )
        record = apply_event(
            record,
            LifecycleEvent(
                EventCode.PREPARATION_STARTED,
                occurred_at,
                MessageCode.PREPARING,
            ),
        )
        record = apply_event(
            record,
            LifecycleEvent(
                EventCode.PREPARATION_COMPLETED,
                occurred_at,
                MessageCode.READY,
            ),
        )
        baseline = ModalPreparedRunV1(record, sealed_preparation)
        presented = ModalPreparedRunV1(baseline.record, baseline.preparation)
        created = True
        create_failed = False
        guard()
        try:
            self._repository.create_modal_prepared_run(presented)
        except RunAlreadyExists as error:
            if type(error) is RunAlreadyExists:
                created = False
            else:
                create_failed = True
        except Exception:
            create_failed = True
        guard()
        if create_failed:
            _closed_start_failure()
        readback_failed = False
        try:
            guard()
            loaded_record = self._repository.load(
                baseline.record.project_ref, baseline.record.run_id
            )
            guard()
            loaded_preparation = self._repository.load_modal_preparation(
                baseline.record.project_ref, baseline.record.run_id
            )
            guard()
        except Exception:
            readback_failed = True
        if readback_failed:
            _closed_readback_failure()
        pair_failed = False
        try:
            loaded_pair = ModalPreparedRunV1(
                _prepared_record_prefix(loaded_record), loaded_preparation
            )
        except Exception:
            pair_failed = True
        if pair_failed:
            _closed_readback_failure()
        if created:
            if loaded_pair != baseline:
                _closed_readback_failure()
        elif (
            loaded_pair.preparation != baseline.preparation
            or loaded_pair.record.project_ref != baseline.record.project_ref
            or loaded_pair.record.run_id != baseline.record.run_id
            or loaded_pair.record.grant_binding.operation
            != baseline.record.grant_binding.operation
            or loaded_pair.record.grant_binding.grant_ref
            != sealed_grant.grant_ref
        ):
            _closed_readback_failure()
        preparation = loaded_pair.preparation
        guard()
        record = _validated_current_record(
            loaded_record,
            prepared=loaded_pair,
            expected_effect=preparation.operation.effect,
            require_ready_grant_ref=sealed_grant.grant_ref,
        )
        guard()
        if record.phase is not LifecyclePhase.READY:
            return TrainingSubmission(
                RunRef(record.run_id, record.project_ref),
                preparation.public_plan_fingerprint,
                self._submission_time(record, preparation.operation.effect.effect_id),
            )
        expected = preparation.stage.expectation
        receipt = StageReceiptV1(
            expected.effect.effect_id,
            expected.operation_binding_digest,
            expected.control_volume_id,
            expected.artifact_volume_id,
            expected.claim_digest,
            expected.bundle_digest,
        )
        guard()
        try:
            self._stage_control.validate(receipt)
        except Exception:
            guard()
            receipt = _ExplicitModal154VolumeWriter(self._facade).stage_once(
                preparation.stage
            )
        guard()
        command = MutationCommandV1.from_stage(preparation.operation, receipt)
        guard()
        observation = _guarded_broker_execute(
            self._broker,
            self._repository,
            command,
            expected_revision=record.revision,
            guard=guard,
        )
        guard()
        if observation.disposition is EffectDisposition.DEFINITELY_ABSENT:
            raise RuntimeError("Modal submission was definitely absent")
        final_load_failed = False
        guard()
        try:
            final_value = self._repository.load(record.project_ref, record.run_id)
        except Exception:
            final_load_failed = True
        guard()
        if final_load_failed:
            _closed_start_failure()
        record = _validated_current_record(
            final_value,
            prepared=loaded_pair,
            expected_effect=preparation.operation.effect,
            require_ready_grant_ref=sealed_grant.grant_ref,
        )
        guard()
        return TrainingSubmission(
            RunRef(record.run_id, record.project_ref),
            preparation.public_plan_fingerprint,
            self._submission_time(record, preparation.operation.effect.effect_id),
        )

    @staticmethod
    def _submission_time(record: LifecycleRecord, effect_id: str) -> str:
        for event in record.events:
            if (
                event.code
                in {
                    EventCode.EFFECT_ATTEMPTED,
                    EventCode.EFFECT_FOUND,
                    EventCode.EFFECT_INDETERMINATE,
                    EventCode.EFFECT_DEFINITELY_ABSENT,
                }
                and event.effect is not None
                and event.effect.identity.effect_id == effect_id
            ):
                return event.occurred_at
        return record.updated_at

    @staticmethod
    def _public_state(phase: LifecyclePhase) -> RunState:
        return {
            LifecyclePhase.PLANNED: RunState.PLANNED,
            LifecyclePhase.READY: RunState.PLANNED,
            LifecyclePhase.PREPARING: RunState.PLANNED,
            LifecyclePhase.SUBMITTING: RunState.SUBMITTING,
            LifecyclePhase.QUEUED: RunState.QUEUED,
            LifecyclePhase.RUNNING: RunState.RUNNING,
            LifecyclePhase.VERIFYING: RunState.RUNNING,
            LifecyclePhase.SUCCEEDED: RunState.SUCCEEDED,
            LifecyclePhase.FAILED: RunState.FAILED,
            LifecyclePhase.CANCELLING: RunState.CANCELLING,
            LifecyclePhase.CANCELLED: RunState.CANCELLED,
            LifecyclePhase.RECONCILE_REQUIRED: RunState.RECONCILE_REQUIRED,
        }[phase]

    def _status(self, record: LifecycleRecord) -> RunStatus:
        error = None
        if record.phase is LifecyclePhase.FAILED:
            error = ExecutionError(ErrorCode.EXECUTION_FAILED, "training execution failed")
        elif record.phase is LifecyclePhase.RECONCILE_REQUIRED:
            error = ExecutionError(
                ErrorCode.RECONCILE_REQUIRED,
                "training submission or evidence requires reconciliation",
            )
        return RunStatus(
            RunRef(record.run_id, record.project_ref),
            self._public_state(record.phase),
            record.updated_at,
            error,
        )

    @staticmethod
    def _artifacts(submission: TrainingSubmission, manifest) -> tuple[ArtifactRef, ...]:
        return tuple(
            ArtifactRef(
                member.provider_entry_id,
                submission.run,
                member.role.value,
                ArtifactState.VERIFIED,
            )
            for member in sorted(manifest.members, key=lambda item: item.role.value)
        )

    def _verify_semantics(self, preparation: ModalDurablePreparationV1, manifest):
        bundle = ModalExecutionBundleV1.parse_transport(preparation.stage.bundle)
        workload_bytes = next(
            member.content for member in bundle.members if member.name == "workload.json"
        )
        workload_document = CanonicalDocument(workload_bytes.decode("utf-8")).to_dict()
        workload = CompiledWorkload(
            method="sft",
            schema_version=workload_document["schema_version"],
            entrypoint=workload_document["entrypoint"],
            canonical_bytes=workload_bytes,
        )
        inventory = ArtifactInventory(
            tuple(
                ArtifactEntry(
                    member.role.value,
                    member.path,
                    member.sha256,
                    member.size,
                )
                for member in manifest.members
            )
        )
        facade = self._facade
        volume_id = preparation.context.artifact_volume_id

        class Reader:
            def read_bytes(self, artifact, *, maximum):
                try:
                    return facade.read_complete(
                        volume_id, artifact.path, max_bytes=maximum
                    )
                except Exception:
                    raise ArtifactReadError("artifact read unavailable") from None

        report = VerificationService(WorkloadBindingVerifier()).verify(
            provider_completed=True,
            process=ProcessResult(0),
            workload=workload,
            contract=SFT_ARTIFACT_CONTRACT,
            inventory=inventory,
            reader=Reader(),
        )
        if any(
            "artifact_read_failed" in artifact.errors
            for artifact in report.integrity.artifacts
        ):
            return RuntimeVerificationStatus.INCONCLUSIVE
        return report.status

    def outcome(self, submission: TrainingSubmission) -> TrainingOutcome:
        if not isinstance(submission, TrainingSubmission):
            raise TypeError("submission must be TrainingSubmission")
        preparation = self._repository.load_modal_preparation(
            submission.run.project_ref, submission.run.run_id
        )
        if (
            type(preparation) is not ModalDurablePreparationV1
            or preparation.public_plan_fingerprint != submission.plan_fingerprint
        ):
            raise ValueError("durable Modal preparation does not bind the submission")
        record = self._lifecycle.load(
            project_ref=submission.run.project_ref,
            run_id=submission.run.run_id,
        )
        effect_id = preparation.operation.effect.effect_id
        if record.phase is LifecyclePhase.SUCCEEDED:
            manifest = self._completion.validate(effect_id)
            return TrainingOutcome(
                submission, self._status(record), self._artifacts(submission, manifest)
            )
        if record.phase in {LifecyclePhase.FAILED, LifecyclePhase.CANCELLED}:
            return TrainingOutcome(submission, self._status(record))
        retrying_verification = (
            record.phase is LifecyclePhase.RECONCILE_REQUIRED
            and record.verification is VerificationStatus.INCONCLUSIVE
        )
        if record.phase is LifecyclePhase.RECONCILE_REQUIRED and not retrying_verification:
            return TrainingOutcome(submission, self._status(record))
        try:
            terminal = self._terminal.validate(effect_id)
        except Exception:
            found = next(
                (item for item in record.effects if item.state is EffectState.FOUND),
                None,
            )
            if found is not None and found.provider_job_ref is not None:
                call_state = self._facade.observe_function_call(found.provider_job_ref)
                if (
                    call_state is ModalFunctionCallState.PENDING
                    and record.phase is LifecyclePhase.QUEUED
                ):
                    record = self._lifecycle.record_provider_phase(
                        project_ref=record.project_ref,
                        run_id=record.run_id,
                        expected_revision=record.revision,
                        provider_phase=ProviderRunPhase.RUNNING,
                    )
                elif call_state is ModalFunctionCallState.RETURNED:
                    record = self._lifecycle.record_provider_phase(
                        project_ref=record.project_ref,
                        run_id=record.run_id,
                        expected_revision=record.revision,
                        provider_phase=ProviderRunPhase.UNKNOWN,
                    )
            return TrainingOutcome(submission, self._status(record))
        if terminal.evidence.status_code == "failed":
            try:
                self._logs.validate(effect_id)
            except Exception:
                return TrainingOutcome(submission, self._status(record))
            record = self._lifecycle.record_provider_phase(
                project_ref=record.project_ref,
                run_id=record.run_id,
                expected_revision=record.revision,
                provider_phase=ProviderRunPhase.FAILED,
            )
            return TrainingOutcome(submission, self._status(record))
        if terminal.evidence.status_code != "completed":
            return TrainingOutcome(submission, self._status(record))
        if not retrying_verification:
            record = self._lifecycle.record_provider_phase(
                project_ref=record.project_ref,
                run_id=record.run_id,
                expected_revision=record.revision,
                provider_phase=ProviderRunPhase.SUCCEEDED,
            )
        record = self._lifecycle.record_verification(
            project_ref=record.project_ref,
            run_id=record.run_id,
            expected_revision=record.revision,
            verification=VerificationStatus.VERIFYING,
        )
        try:
            manifest = self._completion.validate(effect_id)
        except Exception:
            record = self._lifecycle.record_verification(
                project_ref=record.project_ref,
                run_id=record.run_id,
                expected_revision=record.revision,
                verification=VerificationStatus.INCONCLUSIVE,
            )
            return TrainingOutcome(submission, self._status(record))
        semantic = self._verify_semantics(preparation, manifest)
        if semantic is RuntimeVerificationStatus.INCONCLUSIVE:
            record = self._lifecycle.record_verification(
                project_ref=record.project_ref,
                run_id=record.run_id,
                expected_revision=record.revision,
                verification=VerificationStatus.INCONCLUSIVE,
            )
            return TrainingOutcome(submission, self._status(record))
        if semantic is RuntimeVerificationStatus.INVALID:
            record = self._lifecycle.record_verification(
                project_ref=record.project_ref,
                run_id=record.run_id,
                expected_revision=record.revision,
                verification=VerificationStatus.INVALID,
            )
            return TrainingOutcome(submission, self._status(record))
        record = self._lifecycle.record_verification(
            project_ref=record.project_ref,
            run_id=record.run_id,
            expected_revision=record.revision,
            verification=VerificationStatus.VERIFIED,
        )
        return TrainingOutcome(
            submission, self._status(record), self._artifacts(submission, manifest)
        )

    def reverify(self, submission: TrainingSubmission) -> TrainingOutcome:
        """Explicitly re-adjudicate immutable authenticated artifacts."""
        if not isinstance(submission, TrainingSubmission):
            raise TypeError("submission must be TrainingSubmission")
        preparation = self._repository.load_modal_preparation(
            submission.run.project_ref, submission.run.run_id
        )
        if (
            type(preparation) is not ModalDurablePreparationV1
            or preparation.public_plan_fingerprint != submission.plan_fingerprint
        ):
            raise ValueError("durable Modal preparation does not bind the submission")
        record = self._lifecycle.load(
            project_ref=submission.run.project_ref,
            run_id=submission.run.run_id,
        )
        if record.phase is LifecyclePhase.SUCCEEDED:
            manifest = self._completion.validate(preparation.operation.effect.effect_id)
            return TrainingOutcome(
                submission, self._status(record), self._artifacts(submission, manifest)
            )
        if not (
            record.phase is LifecyclePhase.FAILED
            and record.verification is VerificationStatus.INVALID
        ):
            raise ValueError("reverify requires an invalid terminal verification")
        try:
            manifest = self._completion.validate(preparation.operation.effect.effect_id)
        except Exception:
            return TrainingOutcome(submission, self._status(record))
        semantic = self._verify_semantics(preparation, manifest)
        record = self._lifecycle.reopen_verification(
            project_ref=record.project_ref,
            run_id=record.run_id,
            expected_revision=record.revision,
        )
        verification = {
            RuntimeVerificationStatus.VERIFIED: VerificationStatus.VERIFIED,
            RuntimeVerificationStatus.INVALID: VerificationStatus.INVALID,
            RuntimeVerificationStatus.INCONCLUSIVE: VerificationStatus.INCONCLUSIVE,
        }[semantic]
        record = self._lifecycle.record_verification(
            project_ref=record.project_ref,
            run_id=record.run_id,
            expected_revision=record.revision,
            verification=verification,
        )
        artifacts = (
            self._artifacts(submission, manifest)
            if verification is VerificationStatus.VERIFIED
            else ()
        )
        return TrainingOutcome(submission, self._status(record), artifacts)

def compose_modal_training_operations(
    *,
    context: ProjectContext,
    host_ports: HostPorts,
    provider_config: ModalProviderProfileV1,
    recipe_registry: RecipeRegistry | None = None,
) -> ModalTrainingOperations:
    """Compose the sole Modal training path without choosing host persistence."""
    if recipe_registry is None:
        from tuner.training import default_recipe_registry

        recipe_registry = default_recipe_registry()
    planning = TrainingService(
        context=context,
        resolver=host_ports.training_resolver,
        recipes=recipe_registry,
    )
    return ModalTrainingOperations(
        planning=planning,
        context=context,
        ports=host_ports,
        profile=provider_config,
    )


__all__ = [
    "MODAL_PLAN_CONTEXT_SCHEMA",
    "ModalDurablePreparationV1",
    "ModalPreparedRunV1",
    "ModalPlanContextV1",
    "ModalTrainingOperations",
    "ModalTrainingRepository",
    "compose_modal_training_operations",
]
