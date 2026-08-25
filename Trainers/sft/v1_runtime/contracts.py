"""Narrow trust-boundary contracts for the isolated SFT v1 runtime."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, final

ENTRYPOINT_V1 = "synaptic.sft.train/v1"
SCHEMA_V1 = "synaptic.sft-workload/v1"
MASK_CONTRACT_V1 = "synaptic.sft-mask/conversation-prefix-v1"
ARTIFACT_ROLES_V1 = (
    "workload_record", "training_lineage", "training_metrics", "final_adapter", "tokenizer"
)

_PRODUCTION_COMPOSITION_SEAL = object()


class ProductionRuntimeCompositionUnavailable(RuntimeError):
    """Production runtime composition has not been installed."""


@dataclass(frozen=True, slots=True, init=False)
class ProductionRuntimeBindingEvidenceV1:
    workload_digest: str
    engine_commit: str
    source_digest: str
    dependency_lock_digest: str
    model_revision: str
    tokenizer_revision: str
    dataset_revision: str
    target_profile_id: str
    target_profile_digest: str
    target_modules: tuple[str, ...]
    release_manifest_digest: str
    resolution_attestation_digest: str

    def __init__(self, *args: Any, _composition_seal: object = None, **kwargs: Any) -> None:
        del args, kwargs
        if _composition_seal is not _PRODUCTION_COMPOSITION_SEAL:
            raise ProductionRuntimeCompositionUnavailable(
                "production runtime evidence is unavailable"
            )
        raise ProductionRuntimeCompositionUnavailable(
            "production runtime composition is not implemented"
        )


@dataclass(frozen=True, slots=True)
class FixtureRuntimeBindingEvidenceV1:
    workload_digest: str
    engine_commit: str
    source_digest: str
    dependency_lock_digest: str
    model_revision: str
    tokenizer_revision: str
    dataset_revision: str
    target_profile_id: str
    target_profile_digest: str
    target_modules: tuple[str, ...]
    fixture_manifest_digest: str


RuntimeBindingEvidenceV1 = ProductionRuntimeBindingEvidenceV1 | FixtureRuntimeBindingEvidenceV1


class BoundWorkloadV1(Protocol):
    @property
    def canonical_bytes(self) -> bytes: ...
    @property
    def digest(self) -> str: ...
    @property
    def document(self) -> Mapping[str, Any]: ...
    @property
    def runtime_binding_evidence(self) -> RuntimeBindingEvidenceV1: ...


class ClosedWorkloadDecoderV1(Protocol):
    """Bytes-only trusted-core seam; implementation must remain stdlib-only."""
    def decode_and_bind_sft_v1(self, canonical_bytes: bytes) -> BoundWorkloadV1: ...


class HubAccessV1(Protocol):
    def token_for(self, repository: str, repository_type: str) -> str | None: ...


T = TypeVar("T")


class ArtifactSlotWriterV1(Protocol):
    @property
    def workload_digest(self) -> str: ...
    @property
    def artifact_slot_ref(self) -> str: ...
    def assert_link_safe(self) -> None: ...
    def in_training_workspace(self, operation: Callable[[Path], T]) -> T: ...
    def write_bytes(self, role: str, value: bytes) -> None: ...
    def write_json(self, role: str, value: Mapping[str, Any]) -> None: ...
    def write_directory(self, role: str, producer: Callable[[Path], None]) -> None: ...
    def finish(self, expected_roles: tuple[str, ...]) -> tuple[str, ...]: ...


class RuntimeContractError(ValueError):
    pass


@final
class ProductionDecoderV1:
    """Closed until the trusted production composition root is implemented."""
    def __init__(self, implementation: ClosedWorkloadDecoderV1, *, _composition_seal: object = None) -> None:
        del implementation
        if _composition_seal is not _PRODUCTION_COMPOSITION_SEAL:
            raise ProductionRuntimeCompositionUnavailable("production decoder is unavailable")
        raise ProductionRuntimeCompositionUnavailable("production decoder composition is not implemented")
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ProductionDecoderV1 is sealed")
    def decode_and_bind_sft_v1(self, canonical_bytes: bytes) -> BoundWorkloadV1:
        return self._implementation.decode_and_bind_sft_v1(canonical_bytes)


@final
class FixtureDecoderV1:
    def __init__(self, implementation: ClosedWorkloadDecoderV1) -> None:
        self._implementation = implementation
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("FixtureDecoderV1 is sealed")
    def decode_and_bind_sft_v1(self, canonical_bytes: bytes) -> BoundWorkloadV1:
        return self._implementation.decode_and_bind_sft_v1(canonical_bytes)


class _ArtifactAdapter:
    def __init__(self, implementation: ArtifactSlotWriterV1) -> None:
        self._implementation = implementation
    @property
    def workload_digest(self) -> str:
        return self._implementation.workload_digest
    @property
    def artifact_slot_ref(self) -> str:
        return self._implementation.artifact_slot_ref
    def assert_link_safe(self) -> None:
        self._implementation.assert_link_safe()
    def in_training_workspace(self, operation: Callable[[Path], T]) -> T:
        return self._implementation.in_training_workspace(operation)
    def write_bytes(self, role: str, value: bytes) -> None:
        self._implementation.write_bytes(role, value)
    def write_json(self, role: str, value: Mapping[str, Any]) -> None:
        self._implementation.write_json(role, value)
    def write_directory(self, role: str, producer: Callable[[Path], None]) -> None:
        self._implementation.write_directory(role, producer)
    def finish(self, expected_roles: tuple[str, ...]) -> tuple[str, ...]:
        return self._implementation.finish(expected_roles)


@final
class ProductionArtifactWriterV1(_ArtifactAdapter):
    def __init__(self, implementation: ArtifactSlotWriterV1, *, _composition_seal: object = None) -> None:
        del implementation
        if _composition_seal is not _PRODUCTION_COMPOSITION_SEAL:
            raise ProductionRuntimeCompositionUnavailable("production artifact writer is unavailable")
        raise ProductionRuntimeCompositionUnavailable("production artifact composition is not implemented")
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ProductionArtifactWriterV1 is sealed")


@final
class FixtureArtifactWriterV1(_ArtifactAdapter):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("FixtureArtifactWriterV1 is sealed")


class _HubAdapter:
    def __init__(self, implementation: HubAccessV1) -> None:
        self._implementation = implementation
    def token_for(self, repository: str, repository_type: str) -> str | None:
        return self._implementation.token_for(repository, repository_type)


@final
class ProductionHubAccessV1(_HubAdapter):
    def __init__(self, implementation: HubAccessV1, *, _composition_seal: object = None) -> None:
        del implementation
        if _composition_seal is not _PRODUCTION_COMPOSITION_SEAL:
            raise ProductionRuntimeCompositionUnavailable("production Hub access is unavailable")
        raise ProductionRuntimeCompositionUnavailable("production Hub composition is not implemented")
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ProductionHubAccessV1 is sealed")


@final
class FixtureHubAccessV1(_HubAdapter):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("FixtureHubAccessV1 is sealed")