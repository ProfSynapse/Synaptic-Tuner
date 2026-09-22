"""Explicit host boundary for packaged execution and selected training composition.

This is outside the fixed legacy worker closure. It owns the runtime union and
packaged admission imports; legacy services and coordinator kernels receive
explicit hooks and never select this boundary implicitly.

Before provider dispatch the adapter must resolve the committed
ProviderRuntimeBindingV1, invoke execution.validate_bindings(release, binding),
and authenticate adapter-owned native facts. Planning is not dispatch authority.
"""

from __future__ import annotations

import re

from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.project.context import ProjectContext
from tuner.runtime.releases import PackagedExecutionBindingV1, PackagedTrainingRuntimeReleaseV1
from .contracts import (
    CanonicalDocument, GitExecutionSourceV1, ResolvedTrainingRequest,
    RuntimeSpec, TrainingPlan, TrainingRequest, TrainingRequestResolver,
    _safe_training_request_resolver,
)
from .coordinator_material import (
    CoordinatorResolvedMaterial, _derive_coordinator_material, _require_material_inputs,
)
from .recipes import CompiledWorkload, RecipeRegistry, canonical_json_bytes
from .service import TrainingService, _PackagedCompilationHooks
from .packaged_compilation import (
    PACKAGED_CONTEXT_SCHEMA, PACKAGED_ENTRYPOINT, PACKAGED_SFT_WORKLOAD_SCHEMA,
    _fields, compile_packaged_sft_workload, packaged_artifact_policy_digest,
)


ExecutionMaterialV1 = GitExecutionSourceV1 | PackagedExecutionBindingV1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compile_bound_packaged_workload(config, binding) -> CompiledWorkload:
    if type(binding) is not PackagedExecutionBindingV1:
        raise TypeError("exact packaged execution binding required")
    # Reconstruct, rather than trusting a frozen object that may have been mutated.
    binding = PackagedExecutionBindingV1.from_dict(binding.to_dict())
    workload = compile_packaged_sft_workload(resolved_config=config)
    if (binding.workload_digest != workload.fingerprint
            or binding.configuration_digest != workload.document["configuration"]["digest"]
            or binding.to_dict()["prepared_input"] != workload.document["identities"]["dataset"]):
        raise ValueError("packaged binding differs from compiled workload or prepared identity")
    return workload


def validate_packaged_material(*, request: TrainingRequest, material) -> CompiledWorkload:
    """Recompute shared commitments, without granting provider-dispatch authority.

    The adapter must resolve the committed provider binding and call the
    execution binding's validate_bindings() before any provider dispatch.
    """
    from .contracts import ResolvedTrainingComponents, ResolvedTrainingRequest

    if type(material) not in (ResolvedTrainingComponents, ResolvedTrainingRequest):
        raise TypeError("exact rich packaged material required")
    if type(request) is not TrainingRequest or type(request.document) is not CanonicalDocument:
        raise TypeError("exact training request required")
    binding = material.execution_source
    workload = compile_bound_packaged_workload(material.resolved_config, binding)
    if binding.artifact_policy_digest != packaged_artifact_policy_digest(material.artifact_policy):
        raise ValueError("packaged binding differs from artifact policy")
    if type(material.execution_context) is not CanonicalDocument:
        raise TypeError("exact packaged context required")
    context = _fields(material.execution_context.to_dict(), {"schema_version", "runtime_release"}, "packaged context")
    if context["schema_version"] != PACKAGED_CONTEXT_SCHEMA:
        raise ValueError("unsupported packaged context")
    release = PackagedTrainingRuntimeReleaseV1.from_dict(context["runtime_release"])
    config = material.resolved_config.to_dict()
    if (
        release.manifest_digest != binding.runtime_release_digest
        or release.worker_entrypoint != PACKAGED_ENTRYPOINT
        or "sft" not in release.compatible_methods
        or (config["model"]["ref"], config["model"]["revision"]) not in release.compatible_models
        or config["dataset"]["format"] not in release.compatible_dataset_formats
        or release.workload_schema != PACKAGED_SFT_WORKLOAD_SCHEMA
        or release.prepared_input_schema != "synaptic-prepared-training-input/v1"
        or release.artifact_contract_schema != workload.document["artifacts"]["schema_version"]
    ):
        raise ValueError("runtime release does not admit the packaged workload")
    expected_runtime = RuntimeSpec(release.image_ref, release.installed_distributions_digest,
                                   release.python_version)
    if type(material.runtime) is not RuntimeSpec or material.runtime != expected_runtime:
        raise ValueError("packaged runtime differs from admitted release")
    public = TrainingInputV1.from_json(request.document.canonical_json)
    if (
        canonical_json_bytes(public.to_dict()) != canonical_json_bytes(request.document.to_dict())
        or public.method.value != "sft"
        or public.model.to_dict() != {key: config["model"][key] for key in ("ref", "revision", "tokenizer_revision")}
        or public.dataset.ref != config["dataset"]["ref"]
        or canonical_json_bytes(public.hyperparameters.to_dict()) != canonical_json_bytes(config["sft"])
        or public.artifacts.required_kinds != material.artifact_policy.required_kinds
        or public.artifacts.retain_checkpoints != material.artifact_policy.retain_checkpoints
    ):
        raise ValueError("packaged configuration differs from canonical request")
    return workload


def create_execution_training_service(
    *, context: ProjectContext, resolver: TrainingRequestResolver, recipes: RecipeRegistry,
) -> TrainingService:
    """Compose both explicit modes with the exact packaged admission hook."""
    return TrainingService(
        context=context, resolver=resolver, recipes=recipes,
        packaged_hooks=_PackagedCompilationHooks(validate_packaged_material),
    )


def compile_training_plan_for_execution_v1(
    *, training_input: TrainingInputV1, context: ProjectContext,
    resolver: TrainingRequestResolver,
) -> TrainingPlan:
    """Compile the mandatory explicit mode in the host-resolved configuration."""
    from . import default_recipe_registry

    if type(training_input) is not TrainingInputV1 or type(context) is not ProjectContext:
        raise TypeError("exact training input and project context required")
    service = create_execution_training_service(
        context=context, resolver=_safe_training_request_resolver(resolver),
        recipes=default_recipe_registry(),
    )
    request = service.load(CanonicalDocument(training_input.canonical_json()))
    plan = service.plan(service.resolve_selected(request))
    if type(plan) is not TrainingPlan:
        raise TypeError("planning must return exact TrainingPlan")
    if type(plan.fingerprint) is not str or _SHA256_PATTERN.fullmatch(plan.fingerprint) is None:
        raise ValueError("training plan fingerprint is invalid")
    return plan


def derive_packaged_coordinator_material(
    resolved: ResolvedTrainingRequest, recipes: RecipeRegistry, *,
    request_id: str, project_ref: str, run_id: str,
) -> CoordinatorResolvedMaterial:
    """Admit only packaged material before entering the neutral retention kernel."""
    if type(resolved) is not ResolvedTrainingRequest:
        raise TypeError("exact resolved training request required")
    _require_material_inputs(resolved, recipes, request_id, project_ref, run_id)
    if type(resolved.execution_source) is not PackagedExecutionBindingV1:
        raise TypeError("exact packaged execution binding required")
    if resolved.execution_source.run_ref != run_id:
        raise ValueError("allocated coordinator run differs from execution source run")
    compiled = validate_packaged_material(request=resolved.request, material=resolved)
    return _derive_coordinator_material(
        resolved, recipes, request_id=request_id, project_ref=project_ref, run_id=run_id,
        execution_run=resolved.execution_source.run_ref,
        source_digest=resolved.execution_source.binding_digest, compiled=compiled,
    )


def parse_packaged_coordinator_material(
    payload: bytes, recipes: RecipeRegistry,
) -> CoordinatorResolvedMaterial:
    """Parse and independently re-admit only packaged coordinator material."""
    return CoordinatorResolvedMaterial._parse(
        payload, recipes, decode_source=PackagedExecutionBindingV1.from_dict,
        derive=derive_packaged_coordinator_material,
    )


__all__ = [
    "ExecutionMaterialV1", "compile_training_plan_for_execution_v1",
    "create_execution_training_service", "derive_packaged_coordinator_material",
    "parse_packaged_coordinator_material",
]
