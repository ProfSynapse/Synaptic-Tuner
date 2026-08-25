"""Import-light contracts for the isolated ``synaptic.sft.train/v1`` runtime."""

from .contracts import (
    ARTIFACT_ROLES_V1,
    ENTRYPOINT_V1,
    MASK_CONTRACT_V1,
    ArtifactSlotWriterV1,
    BoundWorkloadV1,
    ClosedWorkloadDecoderV1,
    FixtureArtifactWriterV1,
    FixtureDecoderV1,
    FixtureHubAccessV1,
    FixtureRuntimeBindingEvidenceV1,
    HubAccessV1,
    ProductionArtifactWriterV1,
    ProductionDecoderV1,
    ProductionHubAccessV1,
    ProductionRuntimeBindingEvidenceV1,
    ProductionRuntimeCompositionUnavailable,
    RuntimeBindingEvidenceV1,
    RuntimeContractError,
)

__all__ = [name for name in globals() if name.endswith("V1") or name in {
    "ARTIFACT_ROLES_V1", "ENTRYPOINT_V1", "MASK_CONTRACT_V1"
}]