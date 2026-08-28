"""Injected Docker v1 boundaries. No shell, daemon client, or SDK is imported here."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from ...coordinator_v1.model import (
    AuthenticatedProviderLogPageV1,
    AuthenticatedProviderRunObservationV1,
    ProviderLogPageContentV1,
    ProviderRunObservationContentV1,
    ProviderRunReadRequestV1,
)
from .model import (
    AuthenticatedDockerCommandBindingV1,
    AuthenticatedDockerArtifactInventoryV1,
    AuthenticatedDockerCancellationEvidenceV1,
    AuthenticatedDockerLogPageV1,
    AuthenticatedDockerSourceSealV1,
    DockerArtifactInventoryRequestV1,
    DockerArtifactReadRequestV1,
    DockerArtifactChunkV1,
    DockerArtifactEOFV1,
    DockerCancellationRequestV1,
    DockerCancellationLookupRequestV1,
    DockerCancellationLookupResultV1,
    DockerCommandBindingV1,
    DockerCreateResultV1,
    DockerImageV1,
    DockerLabelsV1,
    DockerLogReadRequestV1,
    DockerLookupRequestV1,
    DockerLookupResultV1,
    DockerRuntimeV1,
    DockerStartResultV1,
    DockerSourceSealRequestV1,
    DockerSourceSealLookupRequestV1,
    DockerSourceSealLookupResultV1,
    DockerWorkloadV1,
)


class DockerCommandCatalogPortV1(Protocol):
    def resolve(self, command_digest: str) -> AuthenticatedDockerCommandBindingV1: ...


class DockerCommandBindingAuthorityPortV1(Protocol):
    authority_ref: str
    key_ref: str
    def authenticate(self, value: AuthenticatedDockerCommandBindingV1) -> bool: ...


class DockerImageInventoryPortV1(Protocol):
    def require_present(self, image: DockerImageV1) -> bool: ...


class DockerSourceSealPortV1(Protocol):
    def seal_read_only(self, request: DockerSourceSealRequestV1) -> AuthenticatedDockerSourceSealV1: ...
    def lookup(self, request: DockerSourceSealLookupRequestV1) -> DockerSourceSealLookupResultV1: ...


class DockerCancellationEvidencePortV1(Protocol):
    def stop_once(self, request: DockerCancellationRequestV1) -> AuthenticatedDockerCancellationEvidenceV1: ...
    def lookup(self, request: DockerCancellationLookupRequestV1) -> DockerCancellationLookupResultV1: ...


class DockerControlPortV1(Protocol):
    def create_once(
        self, *, labels: DockerLabelsV1, image: DockerImageV1,
        runtime: DockerRuntimeV1, workload: DockerWorkloadV1,
        source_ref: str, artifact_ref: str,
    ) -> DockerCreateResultV1: ...
    def start_once(
        self, container_ref: str, labels: DockerLabelsV1
    ) -> DockerStartResultV1: ...
    def lookup(self, request: DockerLookupRequestV1) -> DockerLookupResultV1: ...


class DockerReadPortV1(Protocol):
    def lookup(self, request: DockerLookupRequestV1) -> DockerLookupResultV1: ...
    def logs(self, request: DockerLogReadRequestV1) -> AuthenticatedDockerLogPageV1: ...
    def artifact_inventory(
        self, request: DockerArtifactInventoryRequestV1
    ) -> AuthenticatedDockerArtifactInventoryV1: ...
    def iter_artifact_events(
        self, request: DockerArtifactReadRequestV1
    ) -> Iterator[DockerArtifactChunkV1 | DockerArtifactEOFV1]: ...


class DockerReadAuthorizationPortV1(Protocol):
    def authenticate(self, request: ProviderRunReadRequestV1) -> bool: ...


class DockerEvidenceAuthorityPortV1(Protocol):
    def observation(
        self, content: ProviderRunObservationContentV1
    ) -> AuthenticatedProviderRunObservationV1: ...
    def log_page(self, content: ProviderLogPageContentV1) -> AuthenticatedProviderLogPageV1: ...
    def authenticate_source_seal(self, value: AuthenticatedDockerSourceSealV1) -> bool: ...
    def authenticate_cancellation(self, value: AuthenticatedDockerCancellationEvidenceV1) -> bool: ...
    def authenticate_cancellation_absence(self, value: object) -> bool: ...
    def authenticate_absence(self, value: object) -> bool: ...
    def authenticate_log_page(self, value: AuthenticatedDockerLogPageV1) -> bool: ...
    def authenticate_inventory(self, value: AuthenticatedDockerArtifactInventoryV1) -> bool: ...
    def authenticate_eof(self, value: DockerArtifactEOFV1) -> bool: ...


__all__: list[str] = []
