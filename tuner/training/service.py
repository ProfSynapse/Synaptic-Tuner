"""Config-first training planning service behind the public v1 facade."""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType

from tuner.training.contracts import (
    CanonicalDocument,
    ResolvedTrainingRequest,
    TrainingPlan,
    TrainingRequest,
    TrainingRequestResolver,
)
from tuner.project.context import ProjectContext

from .recipes import CompiledWorkload, RecipeRegistry, compile_execution_workload, selected_execution_mode
from .resolution import validate_source_topology


@dataclass(frozen=True, slots=True)
class _PackagedCompilationHooks:
    """Explicit host composition; the legacy service supplies no packaged hook."""

    compile: object

    def __post_init__(self):
        if type(self.compile) is not FunctionType:
            raise TypeError("packaged compilation hook must be an exact function")


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
        packaged_hooks: _PackagedCompilationHooks | None = None,
    ) -> None:
        if not isinstance(context, ProjectContext):
            raise TypeError("context must be a ProjectContext")
        if not isinstance(resolver, TrainingRequestResolver):
            raise TypeError("resolver must implement TrainingRequestResolver")
        if not isinstance(recipes, RecipeRegistry):
            raise TypeError("recipes must be a RecipeRegistry")
        if packaged_hooks is not None and type(packaged_hooks) is not _PackagedCompilationHooks:
            raise TypeError("exact packaged compilation hooks required")
        self._context = context
        self._resolver = resolver
        self._recipes = recipes
        self._packaged_hooks = packaged_hooks

    def _compile_packaged(self, request, material):
        if self._packaged_hooks is None:
            raise ValueError("packaged mode requires explicit host boundary composition")
        compiled = self._packaged_hooks.compile(request=request, material=material)
        if type(compiled) is not CompiledWorkload:
            raise TypeError("packaged hook must return exact CompiledWorkload")
        return compiled

    def load(self, document: CanonicalDocument) -> TrainingRequest:
        if not isinstance(document, CanonicalDocument):
            raise TypeError("document must be a CanonicalDocument")
        method = document.to_dict().get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("training request requires a method")
        self._recipes.resolve(method)
        return TrainingRequest(document)

    def resolve(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        """Historical Git-only resolver; never selects packaged execution."""
        return self._resolve(request, selected=False)

    def resolve_selected(self, request: TrainingRequest) -> ResolvedTrainingRequest:
        """Resolve the explicitly selected mode without inference or fallback."""
        return self._resolve(request, selected=True)

    def _resolve(self, request: TrainingRequest, *, selected: bool) -> ResolvedTrainingRequest:
        if not isinstance(request, TrainingRequest):
            raise TypeError("request must be a TrainingRequest")
        components = self._resolver.resolve(request, context=self._context)
        if selected and selected_execution_mode(components.resolved_config) == "packaged_runtime":
            workload = self._compile_packaged(request, components)
        else:
            validate_source_topology(self._context, components.execution_source)
            workload = compile_execution_workload(
                resolved_config=components.resolved_config,
                execution_source=components.execution_source,
                recipes=self._recipes, require_mode=selected,
            )
        config = components.resolved_config.to_dict()
        method = config.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("resolved config requires a method")
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
        if selected_execution_mode(resolved.resolved_config, required=False) == "packaged_runtime":
            workload = self._compile_packaged(resolved.request, resolved)
        else:
            workload = compile_execution_workload(
                resolved_config=resolved.resolved_config,
                execution_source=resolved.execution_source,
                recipes=self._recipes,
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
