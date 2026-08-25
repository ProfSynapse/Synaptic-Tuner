"""Thin dispatcher facade for the isolated SFT v1 runtime."""

from .v1_runtime.contracts import (
    ARTIFACT_ROLES_V1,
    ENTRYPOINT_V1,
    MASK_CONTRACT_V1,
    FixtureArtifactWriterV1,
    FixtureDecoderV1,
    FixtureHubAccessV1,
    FixtureRuntimeBindingEvidenceV1,
    ProductionArtifactWriterV1,
    ProductionDecoderV1,
    ProductionHubAccessV1,
    ProductionRuntimeBindingEvidenceV1,
    ProductionRuntimeCompositionUnavailable,
    RuntimeContractError,
)
from .v1_runtime.execution import (
    SFTResultV1,
    dispatch_fixture_sft_v1,
    dispatch_sft_v1,
)

__all__ = [
    "ARTIFACT_ROLES_V1",
    "ENTRYPOINT_V1",
    "MASK_CONTRACT_V1",
    "FixtureArtifactWriterV1",
    "FixtureDecoderV1",
    "FixtureHubAccessV1",
    "FixtureRuntimeBindingEvidenceV1",
    "ProductionArtifactWriterV1",
    "ProductionDecoderV1",
    "ProductionHubAccessV1",
    "ProductionRuntimeBindingEvidenceV1",
    "ProductionRuntimeCompositionUnavailable",
    "RuntimeContractError",
    "SFTResultV1",
    "dispatch_fixture_sft_v1",
    "dispatch_sft_v1",
]