"""Provider-neutral artifact-publication composition contracts.

This is the supported host composition boundary for the engine-owned
publication coordinator.  It intentionally exposes no filesystem, database,
provider SDK, credential, or network implementation.
"""

from __future__ import annotations

from tuner.execution.coordinator_v1.publication import (
    ArtifactDestinationRegistryPortV1,
    ArtifactSpoolPortV1,
    AuthenticatedDestinationInventoryV1,
    AuthenticatedDestinationV1,
    AuthenticatedLookupV1,
    AuthenticatedPublicationReceiptV1,
    AuthenticatedPublicationTombstoneV1,
    AuthenticatedVerifiedSourceV1,
    DestinationArtifactV1,
    DestinationInventoryV1,
    DestinationPublicationPortV1,
    EvidenceAuthorityPortV1,
    LookupOutcomeV1,
    LookupRecoveryPermitV1,
    MaterializedSourceV1,
    PublicationCodeV1,
    PublicationErrorV1,
    PublicationEventKindV1,
    PublicationEventV1,
    PublicationOperationsV1,
    PublicationPhaseV1,
    PublicationRecordV1,
    PublicationStorePortV1,
    PublicationTransitionKernelV1,
    RecoveryDecisionV1,
    RecoveryDispositionV1,
    SpooledArtifactV1,
    SpoolSinkPortV1,
    StrongInMemoryPublicationStoreV1,
    TransferAdmissionV1,
    TransferDispositionV1,
    TransferOwnershipV1,
    VerifiedArtifactSourcePortV1,
)


__all__ = [
    "ArtifactDestinationRegistryPortV1", "ArtifactSpoolPortV1",
    "AuthenticatedDestinationInventoryV1", "AuthenticatedDestinationV1",
    "AuthenticatedLookupV1", "AuthenticatedPublicationReceiptV1",
    "AuthenticatedPublicationTombstoneV1", "AuthenticatedVerifiedSourceV1",
    "DestinationArtifactV1", "DestinationInventoryV1",
    "DestinationPublicationPortV1", "EvidenceAuthorityPortV1", "LookupOutcomeV1",
    "LookupRecoveryPermitV1", "MaterializedSourceV1", "PublicationCodeV1",
    "PublicationErrorV1", "PublicationEventKindV1", "PublicationEventV1",
    "PublicationOperationsV1", "PublicationPhaseV1", "PublicationRecordV1",
    "PublicationStorePortV1", "PublicationTransitionKernelV1",
    "RecoveryDecisionV1", "RecoveryDispositionV1",
    "SpooledArtifactV1", "SpoolSinkPortV1", "StrongInMemoryPublicationStoreV1",
    "TransferAdmissionV1", "TransferDispositionV1", "TransferOwnershipV1",
    "VerifiedArtifactSourcePortV1",
]
