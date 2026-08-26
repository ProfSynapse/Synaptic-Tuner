"""Request resolution seams for provider-neutral training planning."""

from __future__ import annotations

from pathlib import Path

from synaptic_tuner.api.v1.training import (
    ResolvedTrainingComponents,
    TrainingRequestResolver,
    TrainingResolutionError,
)
from tuner.project.context import ProjectContext
from tuner.project.execution_source import ExecutionSourceV1


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
