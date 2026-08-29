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
    "artifacts": {
        "ArtifactPublicationReceipt", "ArtifactPublisher", "PublishedArtifact",
        "VerifiedArtifactDescriptor", "VerifiedArtifactSource",
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
    "runs": {
        "ArtifactPage", "ArtifactsRequest", "CancelResult", "LogEntry", "LogPage",
        "ReconcileRequest", "RunCancelRequest", "RunListRequest", "RunLogsRequest",
        "RunPage", "RunVerification", "RunsAPI", "RunsOperations", "VerifyRequest",
    },
    "secrets": {"SecretRef"},
    "sources": {
        "AuthenticatedSourceEvidenceV1", "ExecutionSourceV1",
        "GitCliLocalSourceInspector", "LocalSourceInspectionPort",
        "PushedSourceVerificationPort", "SourceLock",
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
}

_LAZY_ATTRIBUTES = {
    name: module_name
    for module_name, names in _LAZY_MODULE_ATTRIBUTES.items()
    for name in names
}

_FORMAL_EXPORTS = (
    "APIHost", "AttemptAdmission", "AttemptDisposition", "ArtifactPage",
    "ArtifactPublicationReceipt", "ArtifactPublisher", "ArtifactPolicy", "ArtifactRef",
    "ArtifactState", "ArtifactsRequest", "AuthorizationRequirement",
    "AuthorizationMismatch", "CapabilityDescriptor", "CancelResult", "CanonicalDocument",
    "ErrorCode", "EffectCollision", "EffectDisposition", "EffectIdentity", "EffectKind",
    "EffectObservation", "EffectRecord", "EffectState", "EvidenceReplayRepository",
    "EventEnvelope", "ExecutionError", "ExecutionGrant", "ExecutionScope",
    "EvidenceAuthenticator", "EvidenceReplayStore", "Clock", "GitRemoteReader",
    "GitCliLocalSourceInspector", "GrantProvider", "GrantBinding", "HostPorts",
    "LifecycleRepository", "LifecycleEvent", "LifecyclePhase", "LifecycleRecord",
    "LifecycleRunPage", "ModalDeploymentReader", "MessageCode", "LogEntry", "LogPage",
    "PathRef", "PluginBinding", "PluginContext", "PublishedArtifact", "OperationBindingV1",
    "ProjectContext", "ReconcileRequest", "ResolvedTrainingRequest",
    "ResolvedTrainingComponents", "ResultEnvelope", "ResourceSpec", "RunCancelRequest",
    "RunListRequest", "RunLogsRequest", "RunPage", "RunRef", "RunAlreadyExists",
    "RunNotFound", "RunState", "RunStatus", "RunVerification", "RunsAPI",
    "RunsOperations", "RuntimeSpec", "ReplayDisposition", "RevisionConflict", "SecretRef",
    "SecretProvider", "SourceLock", "AuthenticatedSourceEvidenceV1", "ExecutionSourceV1",
    "LocalSourceInspectionPort", "PushedSourceVerificationPort", "TrainingAPI",
    "TrainingOperations", "TrainingOutcome", "TrainingPlan", "TrainingPreflight",
    "TrainingRequest", "TrainingRequestResolver", "TrainingResolutionError",
    "TrainingSubmission", "VerificationStatus", "VerifiedArtifactDescriptor",
    "VerifiedArtifactSource", "InvalidTransition", "apply_lifecycle_event", "VerifyRequest",
    "SFTTrainingHyperparametersV1", "TrainingArtifactRequirementsV1",
    "TrainingDatasetInputV1", "TrainingDurationV1", "TrainingInputV1",
    "TrainingMethodV1", "TrainingModelInputV1",
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
