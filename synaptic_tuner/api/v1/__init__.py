"""Supported Synaptic Tuner API v1.

The current operational surface remains available during the B1/B2 staged
cutover, but its implementation modules are imported only when an operational
export is requested. Importing a B1 contract module cannot therefore load
``tuner.*``, a provider SDK, SQLite, or host code.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_MODULE_ATTRIBUTES = {
    "artifacts_facade": {
        "ArtifactDestination", "ArtifactsAPI", "ArtifactsOperations",
        "DestinationPage", "PublicationPage", "PublicationRef",
        "PublicationRequest", "PublicationResult", "PublicationState",
        "PublicationVerification",
    },
    "capabilities": {"CapabilityDescriptor"},
    "context": {"PathRef", "ProjectContext"},
    "events": {"EventEnvelope", "ResultEnvelope"},
    "execution": {
        "ArtifactRef", "ArtifactState", "AuthorizationRequirement", "ErrorCode",
        "ExecutionError", "ExecutionGrant", "RunRef", "RunState", "RunStatus",
    },
    "host": {
        "APIHost", "Clock", "EvidenceAuthenticator", "EvidenceReplayStore",
        "GitRemoteReader", "GrantProvider", "HostPorts", "LifecycleRepository",
        "ModalDeploymentReader", "SecretProvider",
    },
    "persistence": {
        "AttemptAdmission", "AttemptDisposition", "AuthorizationMismatch",
        "EffectCollision", "EffectDisposition", "EffectIdentity", "EffectKind",
        "EffectObservation", "EffectRecord", "EffectState",
        "EvidenceReplayRepository", "EventCode", "ExecutionScope", "GrantBinding",
        "InvalidTransition", "LifecycleEvent", "LifecyclePhase", "LifecycleRecord",
        "LifecycleRunPage", "MessageCode", "OperationBindingV1", "ReplayDisposition",
        "RevisionConflict", "RunAlreadyExists", "RunNotFound", "VerificationStatus",
        "apply_lifecycle_event",
    },
    "plugins": {"PluginBinding", "PluginContext"},
    "publication": {
        "ArtifactDestinationRegistryPortV1", "ArtifactSpoolPortV1",
        "AuthenticatedDestinationInventoryV1", "AuthenticatedDestinationV1",
        "AuthenticatedLookupV1", "AuthenticatedPublicationReceiptV1",
        "AuthenticatedPublicationTombstoneV1", "AuthenticatedVerifiedSourceV1",
        "DestinationArtifactV1", "DestinationInventoryV1",
        "DestinationPublicationPortV1", "EvidenceAuthorityPortV1", "LookupOutcomeV1",
        "LookupRecoveryPermitV1", "MaterializedSourceV1", "PublicationCodeV1",
        "PublicationErrorV1", "PublicationEventKindV1", "PublicationEventV1",
        "PublicationOperationsV1", "PublicationPhaseV1", "PublicationRecordV1",
        "PublicationStorePortV1", "RecoveryDecisionV1", "RecoveryDispositionV1",
        "SpooledArtifactV1", "SpoolSinkPortV1", "StrongInMemoryPublicationStoreV1",
        "TransferAdmissionV1", "TransferDispositionV1", "TransferOwnershipV1",
        "VerifiedArtifactSourcePortV1",
    },
    "results": {"TrainingRunRef", "TrainingRunState", "VerifiedArtifact"},
    "runs_facade": {
        "RunArtifactRequest", "RunArtifactStream", "RunListRequest",
        "RunLogEntry", "RunLogLevel", "RunLogPage", "RunLogsRequest",
        "RunOperationCode", "RunOperationError", "RunOutcome", "RunPage",
        "RunVerification", "RunsAPI", "RunsOperations",
    },
    "secrets": {"SecretRef"},
    "sources": {
        "AuthenticatedSourceEvidenceV1", "ExecutionSourceV1",
        "GitCliLocalSourceInspector", "LocalSourceInspectionPort",
        "PushedSourceVerificationPort", "SourceLock", "SourceLockBindingV1",
        "SourceLockProvenanceViewV1", "validate_source_lock_provenance_v1",
    },
    "training": {
        "ArtifactPolicy", "CanonicalDocument", "ResolvedTrainingComponents",
        "ResolvedTrainingRequest", "ResourceSpec", "RuntimeSpec", "TrainingAPI",
        "TrainingOperations", "TrainingOutcome", "TrainingPlan", "TrainingPreflight",
        "TrainingRequest", "TrainingRequestResolver", "TrainingResolutionError",
        "TrainingSubmission",
    },
    "training_input": {
        "SFTTrainingHyperparametersV1", "TrainingArtifactRequirementsV1",
        "TrainingDatasetInputV1", "TrainingDurationV1", "TrainingInputV1",
        "TrainingMethodV1", "TrainingModelInputV1",
    },
    "training_input_loader": {
        "LoadedTrainingInputContractV1", "TrainingInputContractCodeV1",
        "TrainingInputContractErrorV1", "TrainingInputContractIdentityV1",
        "load_training_input_contract_v1",
    },
}

_LAZY_ATTRIBUTES = {
    name: module_name
    for module_name, names in _LAZY_MODULE_ATTRIBUTES.items()
    for name in names
}

_FORMAL_EXPORTS = (
    "APIHost", "AttemptAdmission", "AttemptDisposition", "ArtifactDestination",
    "ArtifactDestinationRegistryPortV1", "ArtifactPolicy", "ArtifactRef",
    "ArtifactSpoolPortV1", "ArtifactState", "ArtifactsAPI", "ArtifactsOperations",
    "AuthenticatedDestinationInventoryV1", "AuthenticatedDestinationV1",
    "AuthenticatedLookupV1", "AuthenticatedPublicationReceiptV1",
    "AuthenticatedPublicationTombstoneV1", "AuthenticatedVerifiedSourceV1",
    "AuthorizationRequirement",
    "AuthorizationMismatch", "CapabilityDescriptor", "CanonicalDocument",
    "ErrorCode", "EffectCollision", "EffectDisposition", "EffectIdentity", "EffectKind",
    "EffectObservation", "EffectRecord", "EffectState", "EvidenceReplayRepository",
    "EventEnvelope", "ExecutionError", "ExecutionGrant", "ExecutionScope",
    "EvidenceAuthenticator", "EvidenceReplayStore", "Clock", "GitRemoteReader",
    "GitCliLocalSourceInspector", "GrantProvider", "GrantBinding", "HostPorts",
    "LifecycleRepository", "LifecycleEvent", "LifecyclePhase", "LifecycleRecord",
    "LifecycleRunPage", "ModalDeploymentReader", "MessageCode", "DestinationArtifactV1",
    "DestinationInventoryV1", "DestinationPage", "DestinationPublicationPortV1",
    "EvidenceAuthorityPortV1", "LookupOutcomeV1", "LookupRecoveryPermitV1",
    "MaterializedSourceV1", "PathRef", "PluginBinding", "PluginContext",
    "OperationBindingV1", "ProjectContext", "PublicationCodeV1", "PublicationErrorV1",
    "PublicationEventKindV1", "PublicationEventV1", "PublicationOperationsV1",
    "PublicationPage", "PublicationPhaseV1", "PublicationRecordV1", "PublicationRef",
    "PublicationRequest", "PublicationResult", "PublicationState",
    "PublicationStorePortV1", "PublicationVerification", "RecoveryDecisionV1",
    "RecoveryDispositionV1", "ResolvedTrainingRequest",
    "ResolvedTrainingComponents", "ResultEnvelope", "ResourceSpec", "RunArtifactRequest",
    "RunArtifactStream", "RunListRequest", "RunLogEntry", "RunLogLevel", "RunLogPage",
    "RunLogsRequest", "RunOperationCode", "RunOperationError", "RunOutcome", "RunPage",
    "RunRef", "RunAlreadyExists",
    "RunNotFound", "RunState", "RunStatus", "RunVerification", "RunsAPI",
    "RunsOperations", "RuntimeSpec", "ReplayDisposition", "RevisionConflict", "SecretRef",
    "SecretProvider", "SourceLock", "SourceLockBindingV1",
    "SourceLockProvenanceViewV1", "validate_source_lock_provenance_v1",
    "AuthenticatedSourceEvidenceV1", "ExecutionSourceV1",
    "LocalSourceInspectionPort", "PushedSourceVerificationPort", "TrainingAPI",
    "TrainingOperations", "TrainingOutcome", "TrainingPlan", "TrainingPreflight",
    "TrainingRequest", "TrainingRequestResolver", "TrainingResolutionError",
    "TrainingRunRef", "TrainingRunState", "TrainingSubmission",
    "TransferAdmissionV1", "TransferDispositionV1", "TransferOwnershipV1",
    "VerificationStatus", "VerifiedArtifact", "VerifiedArtifactSourcePortV1",
    "SpooledArtifactV1", "SpoolSinkPortV1", "StrongInMemoryPublicationStoreV1",
    "InvalidTransition", "apply_lifecycle_event",
    "SFTTrainingHyperparametersV1", "TrainingArtifactRequirementsV1",
    "TrainingDatasetInputV1", "TrainingDurationV1", "TrainingInputV1",
    "TrainingMethodV1", "TrainingModelInputV1",
    "LoadedTrainingInputContractV1", "TrainingInputContractCodeV1",
    "TrainingInputContractErrorV1", "TrainingInputContractIdentityV1",
    "load_training_input_contract_v1",
)

if not set(_FORMAL_EXPORTS).issubset(_LAZY_ATTRIBUTES):  # pragma: no cover - module invariant
    raise RuntimeError("formal API exports must all have an explicit lazy attribute binding")

__all__ = list(_FORMAL_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))
