"""Config-first training planning service behind the public v1 facade."""

from __future__ import annotations

from synaptic_tuner.api.v1.training import (
    CanonicalDocument,
    ResolvedTrainingRequest,
    TrainingPlan,
    TrainingRequest,
)
from tuner.project.context import ProjectContext

from .recipes import RecipeRegistry
from .resolution import TrainingRequestResolver, validate_source_topology


class TrainingService:
    """Load, resolve, and compile training requests without executing them.

    Provider preflight, authorization, submission, persistence, and observation
    intentionally remain outside this planning core.
    """

    def __init__(
        self,
        *,
        context: ProjectContext,
        resolver: TrainingRequestResolver,
        recipes: RecipeRegistry,
    ) -> None:
        if not isinstance(context, ProjectContext):
            raise TypeError("context must be a ProjectContext")
        if not isinstance(resolver, TrainingRequestResolver):
            raise TypeError("resolver must implement TrainingRequestResolver")
        if not isinstance(recipes, RecipeRegistry):
            raise TypeError("recipes must be a RecipeRegistry")
        self._context = context
        self._resolver = resolver
        self._recipes = recipes

    def load(self, document: CanonicalDocument) -> TrainingRequest:
        if not isinstance(document, CanonicalDocument):
            raise TypeError("document must be a CanonicalDocument")
        method = document.to_dict().get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("training request requires a method")
        self._recipes.resolve(method)
        return TrainingRequest(document)

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        if not isinstance(request, TrainingRequest):
            raise TypeError("request must be a TrainingRequest")
        components = self._resolver.resolve(request, context=self._context)
        validate_source_topology(self._context, components.execution_source)
        config = components.resolved_config.to_dict()
        method = config.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("resolved config requires a method")
        workload = self._recipes.resolve(method).compile(
            resolved_config=components.resolved_config,
            execution_source=components.execution_source,
        )
        workload_document = workload.document
        artifact_section = workload_document.get("artifacts")
        requirements = (
            artifact_section.get("requirements")
            if isinstance(artifact_section, dict)
            else None
        )
        available_roles = {
            item.get("role")
            for item in requirements
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        } if isinstance(requirements, list) else set()
        unsupported = set(components.artifact_policy.required_kinds) - available_roles
        if unsupported:
            raise ValueError(
                "artifact policy requires roles absent from the method contract: "
                + ", ".join(sorted(unsupported))
            )
        return ResolvedTrainingRequest(
            request=request,
            execution_source=components.execution_source,
            execution_context=components.execution_context,
            resolved_config=components.resolved_config,
            workload=CanonicalDocument(workload.canonical_bytes.decode("utf-8")),
            runtime=components.runtime,
            resources=components.resources,
            artifact_policy=components.artifact_policy,
        )

    def plan(self, resolved: ResolvedTrainingRequest) -> TrainingPlan:
        if not isinstance(resolved, ResolvedTrainingRequest):
            raise TypeError("resolved must be a ResolvedTrainingRequest")
        config = resolved.resolved_config.to_dict()
        method = config.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("resolved config requires a method")
        workload = self._recipes.resolve(method).compile(
            resolved_config=resolved.resolved_config,
            execution_source=resolved.execution_source,
        )
        compiled_document = CanonicalDocument(workload.canonical_bytes.decode("utf-8"))
        if compiled_document != resolved.workload:
            raise ValueError("resolved workload does not match deterministic compilation")
        return TrainingPlan(
            execution_source=resolved.execution_source,
            execution_context=resolved.execution_context,
            resolved_config=resolved.resolved_config,
            workload=resolved.workload,
            runtime=resolved.runtime,
            resources=resolved.resources,
            artifact_policy=resolved.artifact_policy,
        )


__all__ = ["TrainingService"]
