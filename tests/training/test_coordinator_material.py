"""Canonical rich-input derivation for coordinator planning."""

from dataclasses import replace
import json

import pytest

from tuner.training.contracts import CanonicalDocument
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.recipes import CompiledWorkload, RecipeRegistry
from tuner.training.coordinator_material import (
    CoordinatorResolvedMaterial, derive_coordinator_material,
)

from tests.training.test_training_service import Resolver, _execution_source


def resolved(tmp_path):
    from tuner.project.context import ProjectContext
    engine = tmp_path / "project" / "vendor" / "engine"
    engine.mkdir(parents=True)
    service = TrainingService(
        context=ProjectContext.host(engine_root=engine, project_root=tmp_path / "project"),
        resolver=Resolver(_execution_source("vendor/engine")),
        recipes=default_recipe_registry(),
    )
    return service.resolve(service.load(CanonicalDocument.from_mapping({"method": "sft"})))


def material(tmp_path):
    return derive_coordinator_material(
        resolved(tmp_path), default_recipe_registry(), request_id="request-a",
        project_ref="project-a", run_id="run-service",
    )


def test_derives_exact_planning_and_bundle_input_identities(tmp_path):
    value = material(tmp_path)
    planning = value.planning_request
    assert planning.request_id == "request-a"
    assert planning.project_ref == "project-a"
    assert planning.source_digest == _execution_source("vendor/engine").fingerprint
    assert planning.workload_digest != __import__("hashlib").sha256(value.workload_bytes).hexdigest()
    assert value.artifact_contract_bytes == json.dumps(
        json.loads(value.workload_bytes)["artifacts"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    ).encode()
    assert value.artifact_contract_sha256 == __import__("hashlib").sha256(
        value.artifact_contract_bytes
    ).hexdigest()


def test_parse_reconstructs_independent_types_and_recompiles(tmp_path):
    original = material(tmp_path)
    restored = CoordinatorResolvedMaterial.parse(
        original.canonical_bytes, default_recipe_registry(),
    )
    assert restored.canonical_bytes == original.canonical_bytes
    assert restored.planning_request == original.planning_request
    assert restored.execution_source_bytes is not original.execution_source_bytes


def test_material_never_reads_mutated_caller_or_returned_planning_objects(tmp_path):
    source = resolved(tmp_path)
    value = derive_coordinator_material(
        source, default_recipe_registry(), request_id="request-a",
        project_ref="project-a", run_id="run-service",
    )
    before = (value.canonical_bytes, value.execution_source_bytes, value.run_id,
              value.planning_request)
    returned = value.planning_request
    object.__setattr__(source.execution_source, "run_id", "poisoned-run")
    object.__setattr__(returned, "request_id", "poisoned-request")
    assert (value.canonical_bytes, value.execution_source_bytes, value.run_id,
            value.planning_request) == before


def test_nested_subclass_and_compiled_workload_subclass_are_refused(tmp_path):
    class DocumentSubclass(CanonicalDocument):
        pass

    source = resolved(tmp_path)
    poisoned = replace(
        source,
        execution_context=DocumentSubclass(source.execution_context.canonical_json),
    )
    with pytest.raises(TypeError, match="exact execution context"):
        derive_coordinator_material(
            poisoned, default_recipe_registry(), request_id="request-a",
            project_ref="project-a", run_id="run-service",
        )

    compiled = default_recipe_registry().resolve("sft").compile(
        resolved_config=source.resolved_config,
        execution_source=source.execution_source,
    )

    class CompiledSubclass(CompiledWorkload):
        pass

    class PoisonRecipe:
        method = "sft"
        def compile(self, **_):
            return CompiledSubclass(
                compiled.method, compiled.schema_version, compiled.entrypoint,
                compiled.canonical_bytes,
            )

    registry = RecipeRegistry()
    registry.register(PoisonRecipe())
    with pytest.raises(TypeError, match="exact CompiledWorkload"):
        derive_coordinator_material(
            source, registry, request_id="request-a",
            project_ref="project-a", run_id="run-service",
        )


def test_artifact_policy_must_be_supported_by_compiled_contract(tmp_path):
    source = resolved(tmp_path)
    poisoned = replace(
        source,
        artifact_policy=replace(source.artifact_policy, required_kinds=("foreign",)),
    )
    with pytest.raises(ValueError, match="roles absent"):
        derive_coordinator_material(
            poisoned, default_recipe_registry(), request_id="request-a",
            project_ref="project-a", run_id="run-service",
        )


@pytest.mark.parametrize("field", [
    "request", "execution_context", "resolved_config", "workload", "runtime",
    "resources", "artifact_policy", "artifact_contract", "digests",
])
def test_every_retained_input_is_checked_on_restore(tmp_path, field):
    original = material(tmp_path)
    document = json.loads(original.canonical_bytes)
    if field == "digests":
        document[field]["runtime"] = "f" * 64
    elif field == "runtime":
        document[field]["python_version"] = "3.12.8"
    elif field == "resources":
        document[field]["timeout_seconds"] += 1
    elif field == "artifact_policy":
        document[field]["retain_checkpoints"] = not document[field]["retain_checkpoints"]
    elif field in {"workload", "artifact_contract"}:
        document[field]["test_poison"] = True
    else:
        document[field]["test_poison"] = True
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises((TypeError, ValueError)):
        CoordinatorResolvedMaterial.parse(payload, default_recipe_registry())


def test_allocated_run_must_precede_source_finalization(tmp_path):
    value = resolved(tmp_path)
    with pytest.raises(ValueError, match="allocated coordinator run"):
        derive_coordinator_material(
            value, default_recipe_registry(), request_id="request-a",
            project_ref="project-a", run_id="another-run",
        )


def test_changed_rich_domains_change_the_corresponding_planning_commitment(tmp_path):
    original = resolved(tmp_path)
    first = derive_coordinator_material(
        original, default_recipe_registry(), request_id="request-a",
        project_ref="project-a", run_id="run-service",
    )
    changed = replace(
        original,
        execution_context=CanonicalDocument.from_mapping({"schema_version": "changed/v1"}),
    )
    second = derive_coordinator_material(
        changed, default_recipe_registry(), request_id="request-a",
        project_ref="project-a", run_id="run-service",
    )
    assert first.planning_request.resolved_config_digest != second.planning_request.resolved_config_digest
    assert first.planning_request.source_digest == second.planning_request.source_digest
