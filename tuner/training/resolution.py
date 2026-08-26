"""Request resolution seams for provider-neutral training planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from synaptic_tuner.api.v1.training import (
    ArtifactPolicy,
    CanonicalDocument,
    ResourceSpec,
    RuntimeSpec,
    TrainingRequest,
)
from tuner.project.context import ProjectContext
from tuner.project.execution_source import ExecutionSourceV1


class TrainingResolutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedTrainingComponents:
    """Exact resolver output before method-specific workload compilation."""

    execution_source: ExecutionSourceV1
    execution_context: CanonicalDocument
    resolved_config: CanonicalDocument
    runtime: RuntimeSpec
    resources: ResourceSpec
    artifact_policy: ArtifactPolicy = ArtifactPolicy()

    def __post_init__(self) -> None:
        checks = (
            (self.execution_source, ExecutionSourceV1, "execution_source"),
            (self.execution_context, CanonicalDocument, "execution_context"),
            (self.resolved_config, CanonicalDocument, "resolved_config"),
            (self.runtime, RuntimeSpec, "runtime"),
            (self.resources, ResourceSpec, "resources"),
            (self.artifact_policy, ArtifactPolicy, "artifact_policy"),
        )
        for value, expected, name in checks:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")


@runtime_checkable
class TrainingRequestResolver(Protocol):
    """Host seam for config, source, model, and dataset resolution."""

    def resolve(
        self,
        request: TrainingRequest,
        *,
        context: ProjectContext,
    ) -> ResolvedTrainingComponents: ...


def validate_source_topology(
    context: ProjectContext, execution_source: ExecutionSourceV1
) -> None:
    """Bind a finalized execution lock to its discovered host provenance."""

    if not isinstance(context, ProjectContext):
        raise TypeError("context must be a ProjectContext")
    if not isinstance(execution_source, ExecutionSourceV1):
        raise TypeError("execution_source must be an ExecutionSourceV1")
    if context.mode != "host":
        raise TrainingResolutionError("finalized cloud execution requires a host context")
    try:
        actual = context.engine_root.resolve().relative_to(context.project_root.resolve())
    except ValueError as exc:
        raise TrainingResolutionError(
            "superproject engine root is outside the project root"
        ) from exc
    if Path(execution_source.engine_submodule_path).as_posix() != actual.as_posix():
        raise TrainingResolutionError("engine root does not match the finalized submodule path")


__all__ = [
    "ResolvedTrainingComponents",
    "TrainingRequestResolver",
    "TrainingResolutionError",
    "validate_source_topology",
]
