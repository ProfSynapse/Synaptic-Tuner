"""Provider-neutral verified artifact publication contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from .execution import RunRef


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _digest(value: str) -> str:
    value = _text(value, "sha256").lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be a lowercase hexadecimal digest")
    return value


@dataclass(frozen=True, slots=True)
class VerifiedArtifactDescriptor:
    kind: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "sha256", _digest(self.sha256))
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("size must be a non-negative integer")


@runtime_checkable
class VerifiedArtifactSource(Protocol):
    run: RunRef
    plan_fingerprint: str
    artifacts: tuple[VerifiedArtifactDescriptor, ...]

    def iter_bytes(self, kind: str, *, maximum: int) -> Iterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    kind: str
    uri: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        descriptor = VerifiedArtifactDescriptor(self.kind, self.sha256, self.size)
        object.__setattr__(self, "kind", descriptor.kind)
        object.__setattr__(self, "sha256", descriptor.sha256)
        object.__setattr__(self, "size", descriptor.size)
        object.__setattr__(self, "uri", _text(self.uri, "uri"))


@dataclass(frozen=True, slots=True)
class ArtifactPublicationReceipt:
    run: RunRef
    plan_fingerprint: str
    destination_ref: str
    artifacts: tuple[PublishedArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRef):
            raise TypeError("run must be a RunRef")
        object.__setattr__(
            self, "plan_fingerprint", _digest(self.plan_fingerprint)
        )
        object.__setattr__(
            self, "destination_ref", _text(self.destination_ref, "destination_ref")
        )
        artifacts = tuple(self.artifacts)
        if not artifacts or any(not isinstance(item, PublishedArtifact) for item in artifacts):
            raise ValueError("artifacts must contain PublishedArtifact values")
        if len({item.kind for item in artifacts}) != len(artifacts):
            raise ValueError("published artifact kinds must be unique")
        object.__setattr__(self, "artifacts", artifacts)


class ArtifactPublisher(Protocol):
    def publish(
        self, source: VerifiedArtifactSource, destination_ref: str
    ) -> ArtifactPublicationReceipt: ...


__all__ = [
    "ArtifactPublicationReceipt",
    "ArtifactPublisher",
    "PublishedArtifact",
    "VerifiedArtifactDescriptor",
    "VerifiedArtifactSource",
]
