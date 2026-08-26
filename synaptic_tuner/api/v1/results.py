"""Provider-neutral training result and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from ._contract import canonical_integer, digest_text, exact_fields, required_text


class TrainingRunState(str, Enum):
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    RECONCILE_REQUIRED = "reconcile_required"


@dataclass(frozen=True, slots=True)
class TrainingRunRef:
    run_id: str
    project_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", required_text(self.run_id, "run_id"))
        object.__setattr__(self, "project_ref", required_text(self.project_ref, "project_ref"))

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "project_ref": self.project_ref}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingRunRef":
        exact_fields(value, frozenset({"run_id", "project_ref"}), "training_run_ref")
        return cls(run_id=value["run_id"], project_ref=value["project_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    role: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", required_text(self.role, "role"))
        object.__setattr__(self, "sha256", digest_text(self.sha256, "sha256"))
        object.__setattr__(
            self, "size_bytes", canonical_integer(self.size_bytes, "size_bytes")
        )

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VerifiedArtifact":
        exact_fields(value, frozenset({"role", "sha256", "size_bytes"}), "verified_artifact")
        return cls(
            role=value["role"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
            size_bytes=value["size_bytes"],  # type: ignore[arg-type]
        )


__all__ = ["TrainingRunRef", "TrainingRunState", "VerifiedArtifact"]
