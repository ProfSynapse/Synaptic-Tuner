"""Final-destination discovery, publication, and verification facade."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol

from ._contract import exact_fields, required_text
from .results import TrainingRunRef, VerifiedArtifact


class PublicationState(str, Enum):
    CLAIMED = "claimed"
    TRANSFERRING = "transferring"
    COMMITTED = "committed"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    FAILED_BEFORE_EFFECT = "failed_before_effect"


@dataclass(frozen=True, slots=True)
class ArtifactDestination:
    destination_ref: str
    display_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination_ref", required_text(self.destination_ref, "destination_ref"))
        object.__setattr__(self, "display_name", required_text(self.display_name, "display_name"))


@dataclass(frozen=True, slots=True)
class DestinationPage:
    destinations: tuple[ArtifactDestination, ...]


@dataclass(frozen=True, slots=True)
class PublicationRef:
    publication_id: str
    destination_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "publication_id", required_text(self.publication_id, "publication_id"))
        object.__setattr__(self, "destination_ref", required_text(self.destination_ref, "destination_ref"))


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    run: TrainingRunRef
    destination_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.run, TrainingRunRef):
            raise TypeError("run must be TrainingRunRef")
        object.__setattr__(self, "destination_ref", required_text(self.destination_ref, "destination_ref"))


@dataclass(frozen=True, slots=True)
class PublicationResult:
    schema_version: str
    publication: PublicationRef
    state: PublicationState
    artifacts: tuple[VerifiedArtifact, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "synaptic-publication-result/v1":
            raise ValueError("unsupported publication result schema version")
        if not isinstance(self.publication, PublicationRef) or not isinstance(self.state, PublicationState):
            raise TypeError("publication/state have invalid types")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, VerifiedArtifact) for item in artifacts):
            raise TypeError("artifacts must contain VerifiedArtifact values")
        roles = tuple(item.role for item in artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.role)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication.publication_id,
            "destination_ref": self.publication.destination_ref,
            "state": self.state.value,
            "artifacts": {
                item.role: {"sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in self.artifacts
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PublicationResult":
        exact_fields(value, frozenset({"schema_version", "publication_id", "destination_ref", "state", "artifacts"}), "publication_result")
        raw_artifacts = value["artifacts"]
        if not isinstance(raw_artifacts, Mapping):
            raise TypeError("artifacts must be a role-keyed object")
        artifacts = []
        for role in sorted(raw_artifacts):
            required_text(role, "artifact_role")
            descriptor = raw_artifacts[role]
            if not isinstance(descriptor, Mapping):
                raise TypeError("artifact descriptors must be objects")
            exact_fields(descriptor, frozenset({"sha256", "size_bytes"}), "artifact_descriptor")
            artifacts.append(VerifiedArtifact(role, descriptor["sha256"], descriptor["size_bytes"]))  # type: ignore[arg-type]
        return cls(
            value["schema_version"],  # type: ignore[arg-type]
            PublicationRef(value["publication_id"], value["destination_ref"]),  # type: ignore[arg-type]
            PublicationState(value["state"]),  # type: ignore[arg-type]
            tuple(artifacts),
        )


@dataclass(frozen=True, slots=True)
class PublicationPage:
    publications: tuple[PublicationResult, ...]


@dataclass(frozen=True, slots=True)
class PublicationVerification:
    publication: PublicationRef
    verified: bool
    checked_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.publication, PublicationRef) or not isinstance(self.verified, bool):
            raise TypeError("publication/verified have invalid types")
        object.__setattr__(self, "checked_at", required_text(self.checked_at, "checked_at"))


class ArtifactsOperations(Protocol):
    def destinations(self) -> DestinationPage: ...
    def publications(self, destination_ref: str) -> PublicationPage: ...
    def publish(self, request: PublicationRequest) -> PublicationResult: ...
    def verify(self, publication: PublicationRef) -> PublicationVerification: ...


class ArtifactsAPI:
    __slots__ = ("_operations",)
    def __init__(self, operations: ArtifactsOperations) -> None: self._operations = operations
    def destinations(self) -> DestinationPage: return self._operations.destinations()
    def publications(self, destination_ref: str) -> PublicationPage: return self._operations.publications(destination_ref)
    def publish(self, request: PublicationRequest) -> PublicationResult: return self._operations.publish(request)
    def verify(self, publication: PublicationRef) -> PublicationVerification: return self._operations.verify(publication)


__all__ = [
    "ArtifactDestination", "ArtifactsAPI", "ArtifactsOperations", "DestinationPage",
    "PublicationPage", "PublicationRef", "PublicationRequest", "PublicationResult",
    "PublicationState", "PublicationVerification",
]
