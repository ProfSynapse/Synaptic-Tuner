"""Final-destination discovery, publication, and verification facade."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ._contract import required_text
from .results import TrainingRunRef, VerifiedArtifact


class PublicationState(str, Enum):
    CLAIMED = "claimed"
    TRANSFERRING = "transferring"
    COMMITTED = "committed"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    FAILED_BEFORE_EFFECT = "failed_before_effect"


def _text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return required_text(value, name)


def _exact_object(
    value: object, expected: frozenset[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    keys = tuple(dict.keys(value))
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{name} field names must be exact strings")
    actual = frozenset(keys)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")
    return {key: dict.__getitem__(value, key) for key in keys}


@dataclass(frozen=True, slots=True)
class ArtifactDestination:
    destination_ref: str
    display_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination_ref", _text(self.destination_ref, "destination_ref"))
        object.__setattr__(self, "display_name", _text(self.display_name, "display_name"))


@dataclass(frozen=True, slots=True)
class DestinationPage:
    destinations: tuple[ArtifactDestination, ...]

    def __post_init__(self) -> None:
        if (type(self.destinations) is not tuple
                or any(type(item) is not ArtifactDestination for item in self.destinations)):
            raise TypeError("destinations must be an exact tuple of ArtifactDestination")
        refs = tuple(item.destination_ref for item in self.destinations)
        if refs != tuple(sorted(refs)) or len(refs) != len(set(refs)):
            raise ValueError("destination references must be unique and ascending")
        object.__setattr__(self, "destinations", tuple(
            ArtifactDestination(item.destination_ref, item.display_name)
            for item in self.destinations
        ))


@dataclass(frozen=True, slots=True)
class PublicationRef:
    publication_id: str
    destination_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "publication_id", _text(self.publication_id, "publication_id"))
        object.__setattr__(self, "destination_ref", _text(self.destination_ref, "destination_ref"))


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    run: TrainingRunRef
    destination_ref: str

    def __post_init__(self) -> None:
        if type(self.run) is not TrainingRunRef:
            raise TypeError("run must be exact TrainingRunRef")
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
        object.__setattr__(self, "destination_ref", _text(self.destination_ref, "destination_ref"))


@dataclass(frozen=True, slots=True)
class PublicationResult:
    schema_version: str
    publication: PublicationRef
    run: TrainingRunRef
    state: PublicationState
    artifacts: tuple[VerifiedArtifact, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "synaptic-publication-result/v1":
            raise ValueError("unsupported publication result schema version")
        if (type(self.publication) is not PublicationRef
                or type(self.run) is not TrainingRunRef
                or type(self.state) is not PublicationState):
            raise TypeError("publication/run/state have invalid types")
        object.__setattr__(self, "publication", PublicationRef(
            self.publication.publication_id, self.publication.destination_ref,
        ))
        object.__setattr__(self, "run", TrainingRunRef.from_dict(self.run.to_dict()))
        if (type(self.artifacts) is not tuple
                or any(type(item) is not VerifiedArtifact for item in self.artifacts)):
            raise TypeError("artifacts must be an exact tuple of VerifiedArtifact")
        artifacts = tuple(VerifiedArtifact.from_dict(item.to_dict()) for item in self.artifacts)
        roles = tuple(item.role for item in artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact roles must be unique")
        object.__setattr__(self, "artifacts", tuple(sorted(artifacts, key=lambda item: item.role)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication.publication_id,
            "destination_ref": self.publication.destination_ref,
            "run": self.run.to_dict(),
            "state": self.state.value,
            "artifacts": {
                item.role: {"sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in self.artifacts
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PublicationResult":
        value = _exact_object(value, frozenset({"schema_version", "publication_id", "destination_ref", "run", "state", "artifacts"}), "publication_result")
        raw_run = _exact_object(value["run"], frozenset({"run_id", "project_ref"}), "run")
        raw_artifacts = value["artifacts"]
        if type(raw_artifacts) is not dict:
            raise TypeError("artifacts must be a role-keyed object")
        artifacts = []
        roles = tuple(dict.keys(raw_artifacts))
        if any(type(role) is not str for role in roles):
            raise TypeError("artifact roles must be exact strings")
        for role in sorted(roles):
            descriptor = _exact_object(
                dict.__getitem__(raw_artifacts, role),
                frozenset({"sha256", "size_bytes"}),
                "artifact_descriptor",
            )
            artifacts.append(VerifiedArtifact(
                _text(role, "artifact_role"),
                _text(descriptor["sha256"], "sha256"),
                descriptor["size_bytes"],  # type: ignore[arg-type]
            ))
        state = value["state"]
        if type(state) is not str:
            raise TypeError("state must be an exact string")
        return cls(
            _text(value["schema_version"], "schema_version"),
            PublicationRef(
                _text(value["publication_id"], "publication_id"),
                _text(value["destination_ref"], "destination_ref"),
            ),
            TrainingRunRef(
                _text(raw_run["run_id"], "run_id"),
                _text(raw_run["project_ref"], "project_ref"),
            ),
            PublicationState(state),
            tuple(artifacts),
        )


@dataclass(frozen=True, slots=True)
class PublicationPage:
    publications: tuple[PublicationResult, ...]

    def __post_init__(self) -> None:
        if (type(self.publications) is not tuple
                or any(type(item) is not PublicationResult for item in self.publications)):
            raise TypeError("publications must be an exact tuple of PublicationResult")
        object.__setattr__(self, "publications", tuple(
            PublicationResult.from_dict(item.to_dict()) for item in self.publications
        ))


@dataclass(frozen=True, slots=True)
class PublicationVerification:
    publication: PublicationRef
    verified: bool
    checked_at: str

    def __post_init__(self) -> None:
        if type(self.publication) is not PublicationRef or type(self.verified) is not bool:
            raise TypeError("publication/verified have invalid types")
        object.__setattr__(self, "publication", PublicationRef(
            self.publication.publication_id, self.publication.destination_ref,
        ))
        object.__setattr__(self, "checked_at", _text(self.checked_at, "checked_at"))


class ArtifactsOperations(Protocol):
    def destinations(self) -> DestinationPage: ...
    def publications(self, destination_ref: str) -> PublicationPage: ...
    def publish(self, request: PublicationRequest) -> PublicationResult: ...
    def verify(self, publication: PublicationRef) -> PublicationVerification: ...


class ArtifactsAPI:
    __slots__ = ("_operations",)
    def __init__(self, operations: ArtifactsOperations) -> None:
        self._operations = operations

    @staticmethod
    def _request(value: PublicationRequest) -> PublicationRequest:
        if type(value) is not PublicationRequest:
            raise TypeError("request must be exact PublicationRequest")
        return PublicationRequest(TrainingRunRef.from_dict(value.run.to_dict()), value.destination_ref)

    @staticmethod
    def _publication(value: PublicationRef) -> PublicationRef:
        if type(value) is not PublicationRef:
            raise TypeError("publication must be exact PublicationRef")
        return PublicationRef(value.publication_id, value.destination_ref)

    @staticmethod
    def _matches(current: object, baseline: object, rebuild) -> bool:
        try:
            return rebuild(current) == baseline
        except BaseException:
            return False

    @staticmethod
    def _changed() -> None:
        raise ValueError("artifact operation input changed during callback") from None

    @classmethod
    def _unchanged(cls, current: object, baseline: object, rebuild) -> None:
        if not cls._matches(current, baseline, rebuild):
            cls._changed()

    @classmethod
    def _call(cls, callback, original, baseline, presentation, rebuild):
        mutation_after_failure = False
        try:
            result = callback(presentation)
        except BaseException:
            mutation_after_failure = (
                not cls._matches(original, baseline, rebuild)
                or not cls._matches(presentation, baseline, rebuild)
            )
            if not mutation_after_failure:
                raise
        if mutation_after_failure:
            cls._changed()
        cls._unchanged(original, baseline, rebuild)
        cls._unchanged(presentation, baseline, rebuild)
        return result

    def destinations(self) -> DestinationPage:
        result = self._operations.destinations()
        if type(result) is not DestinationPage:
            raise TypeError("destination result must be exact DestinationPage")
        return DestinationPage(tuple(
            ArtifactDestination(item.destination_ref, item.display_name)
            for item in result.destinations
        ))

    def publications(self, destination_ref: str) -> PublicationPage:
        destination_ref = _text(destination_ref, "destination_ref")
        result = self._operations.publications(destination_ref)
        if type(result) is not PublicationPage:
            raise TypeError("publication page must be exact PublicationPage")
        rebuilt = PublicationPage(tuple(
            PublicationResult.from_dict(item.to_dict()) for item in result.publications
        ))
        if any(item.publication.destination_ref != destination_ref for item in rebuilt.publications):
            raise ValueError("publication page does not bind the destination")
        return rebuilt

    def publish(self, request: PublicationRequest) -> PublicationResult:
        baseline = self._request(request)
        presented = self._request(baseline)
        result = self._call(self._operations.publish, request, baseline, presented, self._request)
        if type(result) is not PublicationResult:
            raise TypeError("publication result must be exact PublicationResult")
        rebuilt = PublicationResult.from_dict(result.to_dict())
        if (rebuilt.publication.destination_ref != baseline.destination_ref
                or rebuilt.run != baseline.run):
            raise ValueError("publication result does not bind the request")
        self._unchanged(request, baseline, self._request)
        self._unchanged(presented, baseline, self._request)
        return rebuilt

    def verify(self, publication: PublicationRef) -> PublicationVerification:
        baseline = self._publication(publication)
        presented = self._publication(baseline)
        result = self._call(self._operations.verify, publication, baseline, presented, self._publication)
        if type(result) is not PublicationVerification:
            raise TypeError("verification result must be exact PublicationVerification")
        rebuilt = PublicationVerification(
            PublicationRef(
                result.publication.publication_id,
                result.publication.destination_ref,
            ),
            result.verified,
            result.checked_at,
        )
        if rebuilt.publication != baseline:
            raise ValueError("verification result does not bind the request")
        self._unchanged(publication, baseline, self._publication)
        self._unchanged(presented, baseline, self._publication)
        return rebuilt


__all__ = [
    "ArtifactDestination", "ArtifactsAPI", "ArtifactsOperations", "DestinationPage",
    "PublicationPage", "PublicationRef", "PublicationRequest", "PublicationResult",
    "PublicationState", "PublicationVerification",
]
