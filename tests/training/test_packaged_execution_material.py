"""Provider-free conformance for the explicit packaged/developer boundary."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest

from synaptic_tuner.api.v1.execution import PreparedTrainingInputIdentity
from synaptic_tuner.api.v1.training_input import TrainingInputV1
from tuner.project.context import ProjectContext
from tuner.runtime.releases import PackagedExecutionBindingV1, PackagedTrainingRuntimeReleaseV1
from tuner.training import TrainingService, default_recipe_registry
from tuner.training.contracts import (
    ArtifactPolicy, CanonicalDocument, ResolvedTrainingComponents, ResourceSpec,
    RuntimeSpec, TrainingRequest, compile_training_plan_for_execution_v1,
)
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.packaged_boundary import (
    create_execution_training_service,
    derive_packaged_coordinator_material as derive_coordinator_material,
    parse_packaged_coordinator_material,
)
from tuner.training.packaged_compilation import (
    PACKAGED_CONTEXT_SCHEMA, PACKAGED_ENTRYPOINT, PACKAGED_SFT_CONFIG_SCHEMA,
    PACKAGED_SFT_WORKLOAD_SCHEMA, compile_packaged_sft_workload,
    packaged_configuration_digest, packaged_artifact_policy_digest,
)
from tests.runtime.test_packaged_runtime_releases import _release, _provider
from tests.training.test_training_compiler import _training_input


def packaged_fixture():
    identity = PreparedTrainingInputIdentity("prepared://sha256/" + "a" * 64,
                                             "a" * 64, "b" * 64, 42, "syntunia-sft-row/v2")
    public = _training_input().to_dict()
    public["model"].update(revision="1" * 40, tokenizer_revision="2" * 40)
    public["dataset"] = {"ref": identity.ref}
    public["hyperparameters"].update(
        dataset_format="messages", completion_only_loss=True, assistant_only_loss=False,
        use_preassigned_splits=True, prompt_render="prompt_completion", packing=False,
        require_memory_efficient_loss=False,
    )
    training_input = TrainingInputV1.from_dict(public)
    config = CanonicalDocument.from_mapping({
        "schema_version": PACKAGED_SFT_CONFIG_SCHEMA, "method": "sft",
        "execution": {"mode": "packaged_runtime"},
        "model": {**public["model"], "load_in_4bit": False},
        "dataset": identity.to_dict(), "sft": public["hyperparameters"],
    })
    release = _release(worker_entrypoint=PACKAGED_ENTRYPOINT,
                       workload_schema=PACKAGED_SFT_WORKLOAD_SCHEMA,
                       compatible_models=((public["model"]["ref"], "1" * 40),))
    policy = ArtifactPolicy(training_input.artifacts.required_kinds,
                            training_input.artifacts.retain_checkpoints)
    compiled = compile_packaged_sft_workload(resolved_config=config)
    binding = PackagedExecutionBindingV1.build(
        run_ref="run-packaged", runtime_release=release, provider_runtime_binding=_provider(release),
        **identity.execution_binding_fields(), workload_digest=compiled.fingerprint,
        configuration_digest=packaged_configuration_digest(config),
        artifact_policy_digest=packaged_artifact_policy_digest(policy),
    )
    components = ResolvedTrainingComponents(
        binding, CanonicalDocument.from_mapping({"schema_version": PACKAGED_CONTEXT_SCHEMA,
                                                "runtime_release": release.to_dict()}),
        config, RuntimeSpec(release.image_ref, release.installed_distributions_digest,
                            release.python_version), ResourceSpec("cpu"), policy,
    )
    return training_input, components


class Resolver:
    def __init__(self, components):
        self.components = components

    def resolve(self, request, *, context):
        return self.components


def service(tmp_path, components):
    return create_execution_training_service(
        context=ProjectContext.standalone(engine_root=tmp_path),
        resolver=Resolver(components), recipes=default_recipe_registry(),
    )


def resolved(tmp_path, components=None):
    training_input, baseline = packaged_fixture()
    compiler = service(tmp_path, components or baseline)
    return compiler.resolve_selected(compiler.load(CanonicalDocument(training_input.canonical_json())))


def rebound(binding, **changes):
    document = binding.to_dict()
    document.update(changes)
    # Produce an internally authentic binding with inconsistent external commitments.
    unsigned = dict(document)
    unsigned.pop("binding_digest")
    import hashlib
    document["binding_digest"] = hashlib.sha256(
        b"synaptic-packaged-execution-binding/v1\0" + json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return PackagedExecutionBindingV1.from_dict(document)


def test_explicit_packaged_plan_needs_no_git_context(tmp_path):
    training_input, components = packaged_fixture()
    plan = compile_training_plan_for_execution_v1(
        training_input=training_input, context=ProjectContext.standalone(engine_root=tmp_path),
        resolver=Resolver(components),
    )
    assert plan.execution_source == components.execution_source
    assert "execution_source" not in plan.workload.to_dict()
    assert "binding_digest" not in plan.workload.canonical_json
    assert plan.workload.to_dict()["configuration"]["document"]["sft"]["learning_rate"] == 0.0002


def test_workload_is_deterministic_and_acyclic():
    _, components = packaged_fixture()
    config = components.resolved_config
    reverse = CanonicalDocument.from_mapping(dict(reversed(list(config.to_dict().items()))))
    first = compile_packaged_sft_workload(resolved_config=config)
    assert first == compile_packaged_sft_workload(resolved_config=reverse)
    assert first.fingerprint == components.execution_source.workload_digest
    assert components.execution_source.configuration_digest == packaged_configuration_digest(config)


def test_unicode_and_fractional_values_have_stable_utf8_workload_bytes():
    _, components = packaged_fixture()
    config = components.resolved_config.to_dict()
    config["model"]["ref"] = "example/雪"
    config["sft"]["lora_dropout"] = 0.125
    workload = compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(config))
    assert "雪".encode() in workload.canonical_bytes
    assert workload.document["configuration"]["document"]["sft"]["lora_dropout"] == 0.125
    assert workload == compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(config))


def test_closed_execution_union_rejects_structural_impostors_and_subclasses():
    from tuner.training.contracts import ExecutionMaterialV1, GitExecutionSourceV1
    from tuner.project.execution_source import ExecutionSourceV1
    from typing import get_args

    assert GitExecutionSourceV1 is ExecutionSourceV1
    assert set(get_args(ExecutionMaterialV1)) == {ExecutionSourceV1, PackagedExecutionBindingV1}
    _, components = packaged_fixture()
    class Subclass(PackagedExecutionBindingV1):
        pass
    for value in (object(), object.__new__(Subclass), components.execution_source.to_dict()):
        with pytest.raises(TypeError, match="exact"):
            replace(components, execution_source=value)


def test_runtime_request_and_context_are_checked_before_planning(tmp_path):
    training_input, components = packaged_fixture()
    context = components.execution_context.to_dict()
    context["provider_object_id"] = "native-id"
    for bad in (
        replace(components, runtime=replace(components.runtime, python_version="3.12.4")),
        replace(components, execution_context=CanonicalDocument.from_mapping(context)),
    ):
        with pytest.raises(ValueError):
            resolved(tmp_path, bad)
    request = training_input.to_dict()
    request["hyperparameters"]["learning_rate"] = 0.001
    compiler = service(tmp_path, components)
    with pytest.raises(ValueError, match="canonical request"):
        compiler.resolve_selected(TrainingRequest(CanonicalDocument.from_mapping(request)))


def test_packaged_workload_schema_and_closed_structure():
    _, components = packaged_fixture()
    document = compile_packaged_sft_workload(resolved_config=components.resolved_config).document
    schema = json.loads(Path("schemas/synaptic-packaged-sft-workload-v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(document, schema)
    document["execution_source"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema)


@pytest.mark.parametrize("selection", [None, {}, {"mode": "auto"}, {"mode": "packaged_runtime", "git": {}},
                                         {"mode": ["packaged_runtime", "developer_integration"]},
                                         {"mode": "developer_integration"}])
def test_explicit_selection_rejects_missing_unknown_mixed_modes(tmp_path, selection):
    _, components = packaged_fixture()
    config = components.resolved_config.to_dict()
    if selection is None:
        config.pop("execution")
    else:
        config["execution"] = selection
    with pytest.raises((TypeError, ValueError)):
        resolved(tmp_path, replace(components, resolved_config=CanonicalDocument.from_mapping(config)))


@pytest.mark.parametrize("target,key,value", [
    ("root", "sources", {}), ("root", "source_evidence", {}),
    ("root", "clone", "https://example.org/repo"), ("root", "command", "run"),
    ("root", "image", "unreviewed:latest"), ("model", "path", "/private/model"),
    ("dataset", "path", "/private/input"), ("sft", "command", "run"),
    ("execution", "provider_id", "native-object"),
])
def test_packaged_configuration_rejects_source_and_launch_injection(target, key, value):
    _, components = packaged_fixture()
    config = components.resolved_config.to_dict()
    (config if target == "root" else config[target])[key] = value
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(config))


@pytest.mark.parametrize("field,value", [
    ("workload_digest", "c" * 64), ("configuration_digest", "c" * 64),
    ("artifact_policy_digest", "c" * 64), ("runtime_release_digest", "c" * 64),
])
def test_valid_binding_cannot_substitute_compilation_commitments(tmp_path, field, value):
    _, components = packaged_fixture()
    with pytest.raises(ValueError):
        resolved(tmp_path, replace(components, execution_source=rebound(components.execution_source, **{field: value})))


@pytest.mark.parametrize("field,value", [("content_digest", "c" * 64), ("size_bytes", 43),
                                          ("format", "syntunia-sft-row/v1"), ("semantic", "c" * 64)])
def test_full_prepared_identity_is_bound(tmp_path, field, value):
    _, components = packaged_fixture()
    prepared = components.execution_source.to_dict()["prepared_input"]
    if field == "semantic":
        prepared.update(ref="prepared://sha256/" + value, revision=value)
    else:
        prepared[field] = value
    with pytest.raises(ValueError, match="prepared identity"):
        resolved(tmp_path, replace(components, execution_source=rebound(components.execution_source, prepared_input=prepared)))


def test_legacy_resolver_does_not_admit_packaged_material(tmp_path):
    training_input, components = packaged_fixture()
    compiler = service(tmp_path, components)
    with pytest.raises(TypeError, match="ExecutionSourceV1"):
        compiler.resolve(TrainingRequest(CanonicalDocument(training_input.canonical_json())))


def test_coordinator_reconstructs_packaged_material_and_opaque_source_digest(tmp_path):
    rich = resolved(tmp_path)
    material = derive_coordinator_material(rich, default_recipe_registry(), request_id="request",
                                          project_ref="project", run_id="run-packaged")
    assert material.planning_request.source_digest == rich.execution_source.binding_digest
    assert material.planning_request.artifact_policy_digest == rich.execution_source.artifact_policy_digest
    assert parse_packaged_coordinator_material(material.canonical_bytes, default_recipe_registry()) == material


@pytest.mark.parametrize("field", ["runtime", "workload", "resolved_config", "execution_context",
                                    "request", "artifact_policy", "execution_source", "run_id"])
def test_coordinator_rejects_packaged_tampering(tmp_path, field):
    rich = resolved(tmp_path)
    material = derive_coordinator_material(rich, default_recipe_registry(), request_id="request",
                                          project_ref="project", run_id="run-packaged")
    document = json.loads(material.canonical_bytes)
    if field == "run_id":
        document[field] = "other"
    else:
        document[field]["injected"] = True
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    with pytest.raises((TypeError, ValueError)):
        parse_packaged_coordinator_material(payload, default_recipe_registry())


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}',
                                      '{"a":1e999}', '{"a":' + '[' * 40 + '0' + ']' * 40 + '}'])
def test_compilation_json_rejects_ambiguous_or_unbounded_inputs(payload):
    with pytest.raises(ValueError):
        CanonicalDocument(payload)


def test_compilation_bounds_cycles_size_and_keys():
    cycle = {}
    cycle["cycle"] = cycle
    for value in (cycle, {"text": "x" * (512 * 1024 + 1)}, {1: "ambiguous"},
                  {"list": [0] * 4097}):
        with pytest.raises(ValueError):
            CanonicalDocument.from_mapping(value)


def test_legacy_import_path_does_not_load_packaged_or_provider_modules():
    script = """
import json, sys
from tuner.training.contracts import compile_training_plan_v1
from tuner.training import default_recipe_registry, TrainingService
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
default_recipe_registry()
print(json.dumps(sorted(n for n in sys.modules if n.startswith((
    'tuner.runtime.releases', 'tuner.training.packaged_compilation',
    'tuner.execution.providers', 'modal', 'huggingface_hub', 'runpod', 'torch')))))
"""
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_rich_type_introspection_does_not_load_packaged_code():
    script = """
import json, sys, typing
from tuner.training.contracts import (
    ResolvedTrainingRequest, ResolvedTrainingComponents, TrainingPlan,
    execution_material_digest, execution_material_run,
)
for record in (ResolvedTrainingRequest, ResolvedTrainingComponents, TrainingPlan):
    assert typing.get_type_hints(record)['execution_source'] is object
for helper in (execution_material_digest, execution_material_run):
    assert typing.get_type_hints(helper)['value'] is object
print(json.dumps(sorted(n for n in sys.modules if n.startswith((
    'tuner.runtime.releases', 'tuner.training.packaged_compilation',
    'tuner.execution.providers', 'modal', 'huggingface_hub', 'runpod', 'torch')))))
"""
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("field,value", [
    ("lora_dropout", 0), ("lora_dropout", -0.0), ("lora_dropout", False),
    ("learning_rate", 1), ("learning_rate", True), ("learning_rate", -0.0),
    ("duration.num_epochs", 1), ("duration.num_epochs", True),
    ("duration.num_epochs", -0.0),
])
def test_float_hyperparameters_reject_numeric_aliases(field, value):
    _, components = packaged_fixture()
    config = components.resolved_config.to_dict()
    if field == "duration.num_epochs":
        config["sft"]["duration"] = {"max_steps": None, "num_epochs": value}
    else:
        config["sft"][field] = value
    candidate = CanonicalDocument.from_mapping(config)
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_sft_workload(resolved_config=candidate)
    with pytest.raises((TypeError, ValueError)):
        packaged_configuration_digest(candidate)


@pytest.mark.parametrize("field", [
    "batch_size", "gradient_accumulation_steps", "max_seq_length", "seed",
    "save_steps", "save_total_limit", "lora_rank", "lora_alpha", "duration.max_steps",
])
@pytest.mark.parametrize("alias", ["float", "bool"])
def test_integer_hyperparameters_reject_float_and_boolean_aliases(field, alias):
    _, components = packaged_fixture()
    config = components.resolved_config.to_dict()
    target = config["sft"]["duration"] if field == "duration.max_steps" else config["sft"]
    key = field.split(".")[-1]
    target[key] = float(target[key]) if alias == "float" else True
    with pytest.raises((TypeError, ValueError)):
        compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(config))


@pytest.mark.parametrize("alias", [0, -0.0, False])
def test_raw_request_cannot_alias_canonical_config_numbers(tmp_path, alias):
    training_input, components = packaged_fixture()
    request = training_input.to_dict()
    request["hyperparameters"]["lora_dropout"] = alias
    with pytest.raises((TypeError, ValueError)):
        service(tmp_path, components).resolve_selected(
            TrainingRequest(CanonicalDocument.from_mapping(request))
        )


def test_admitted_numeric_values_have_one_representation_and_distinct_digests():
    _, components = packaged_fixture()
    baseline = components.resolved_config.to_dict()
    candidates = [baseline]
    for field, value in (("lora_dropout", 0.125), ("learning_rate", 1.0),
                         ("duration", {"max_steps": None, "num_epochs": 1.0})):
        config = components.resolved_config.to_dict()
        config["sft"][field] = value
        candidates.append(config)
    configuration_digests, workload_digests = set(), set()
    for config in candidates:
        document = CanonicalDocument.from_mapping(config)
        reordered = CanonicalDocument.from_mapping(dict(reversed(list(config.items()))))
        digest = packaged_configuration_digest(document)
        workload = compile_packaged_sft_workload(resolved_config=document)
        assert digest == packaged_configuration_digest(reordered)
        assert workload == compile_packaged_sft_workload(resolved_config=reordered)
        configuration_digests.add(digest)
        workload_digests.add(workload.fingerprint)
    assert len(configuration_digests) == len(workload_digests) == len(candidates)
    assert type(baseline["sft"]["lora_dropout"]) is float


def test_slice_e_must_resolve_and_validate_provider_binding_before_dispatch(tmp_path):
    """Planning is not dispatch authority; adapters must resolve and validate.

    No provider facts are added to shared material to perform this check.
    """
    _, components = packaged_fixture()
    release = PackagedTrainingRuntimeReleaseV1.from_dict(
        components.execution_context.to_dict()["runtime_release"]
    )
    provider_binding = _provider(release)
    committed_catalog = {provider_binding.binding_digest: provider_binding}
    execution = components.execution_source
    selected = committed_catalog[execution.provider_runtime_binding_digest]
    execution.validate_bindings(release, selected)

    # Shared compilation cannot authenticate an unresolved opaque commitment.
    unresolved = rebound(execution, provider_runtime_binding_digest="f" * 64)
    rich = resolved(tmp_path, replace(components, execution_source=unresolved))
    material = derive_coordinator_material(rich, default_recipe_registry(), request_id="request",
                                          project_ref="project", run_id="run-packaged")
    assert json.loads(material.execution_source_bytes)["provider_runtime_binding_digest"] == "f" * 64
    with pytest.raises(KeyError):
        committed_catalog[unresolved.provider_runtime_binding_digest]
    # Substituting another available binding instead of resolving the committed
    # identity must fail the required adapter check before its dispatch call.
    with pytest.raises(ValueError, match="cross-binding"):
        unresolved.validate_bindings(release, selected)
