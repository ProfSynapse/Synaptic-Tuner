from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.results import TrainingRunRef
from synaptic_tuner.api.v1.training import (
    ArtifactPolicy, CanonicalDocument, ResolvedTrainingRequest, ResourceSpec,
    RuntimeSpec, TrainingRequest,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.foundation_v2.commands import build_stage_command
from tuner.execution.foundation_v2.identities import EffectKind
from tuner.execution.providers.modal.coordinator_binding import ModalCommandBinding
from tuner.execution.providers.modal.coordinator_bundle import (
    MEMBER_NAMES, ModalCoordinatorBundle,
)
from tuner.execution.providers.modal.config import ModalRuntimeLockV1
from tuner.execution.providers.modal.resolution import ModalDeploymentSelectionV1
from tuner.runtime.offline_sft_worker import closure_digest, load_packaged_offline_sft_worker_manifest
from tuner.training.coordinator_material import derive_coordinator_material
from tuner.training.methods.sft import SFTRecipe, compile_sft_workload
from tuner.training.recipes import RecipeRegistry
from tests.execution.providers.test_modal_coordinator_adapter import composed, inputs
from tests.execution.providers.test_modal_sdk154_adapter import verified
from tests.training.test_sft_compilation import _config, _execution_source


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _large_canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()


def _fixture(*, runtime=None, resources=None, request=None, config_extra=None):
    values = inputs()
    selection = ModalDeploymentSelectionV1.from_profile(
        values["profile"], binding=values["binding"],
        runtime_environment=values["runtime_environment"],
        timeout_seconds=values["timeout_seconds"],
    )
    deployment = verified(selection)
    deployment_bytes = canonical_bytes(deployment.to_dict())
    original = _execution_source()
    source = replace(
        original, deployment_member_sha256=_sha(deployment_bytes),
        python_version=selection.python_version,
        python_executable=selection.python_executable,
        python_executable_digest=selection.python_executable_digest,
        environment=dict(original.environment) | selection.runtime_environment,
        secret_requirements_digest=selection.secret_requirements_digest,
        provider_runtime_requirements_digest=selection.provider_runtime_requirements_digest,
    )
    config = CanonicalDocument.from_mapping(_config().to_dict() | (config_extra or {}))
    workload = compile_sft_workload(resolved_config=config, execution_source=source)
    runtime_lock = ModalRuntimeLockV1.packaged()
    recipes = RecipeRegistry()
    recipes.register(SFTRecipe())
    material = derive_coordinator_material(
        ResolvedTrainingRequest(
            TrainingRequest(CanonicalDocument.from_mapping(
                request or {"schema_version": "request/v1"}
            )),
            source,
            CanonicalDocument.from_mapping({"schema_version": "context/v1"}),
            config, CanonicalDocument(workload.canonical_bytes.decode()),
            runtime or RuntimeSpec(
                runtime_lock.registry_reference,
                selection.dependency_lock_digest, selection.python_version,
            ),
            resources or ResourceSpec(selection.accelerator, 1, selection.timeout_seconds),
            ArtifactPolicy(tuple(item["role"] for item in workload.document["artifacts"]["requirements"]), False),
        ), recipes, request_id="request-a", project_ref="project-a", run_id="run-1",
    )
    adapter, context, plan, execution = composed(resolved=material.planning_request)
    preparation = adapter.prepare(plan, TrainingRunRef("run-1", "project-a"), execution)
    command = build_stage_command(
        preparation, "nonce-stage", adapter.payload(preparation, EffectKind.STAGE),
        execution.executor_descriptor,
    )
    policy = canonical_bytes({
        "schema_version": "synaptic-modal-log-terminal-policy/v2",
        "generation": 1, "max_log_chunks": 1024,
        "max_chunk_bytes": 65536, "max_terminal_bytes": 65536,
    })
    closure = load_packaged_offline_sft_worker_manifest().canonical_bytes
    binding = ModalCommandBinding(
        command.canonical_bytes, adapter.snapshot(), deployment_bytes,
    )
    return binding, material, recipes, policy, closure


def _bundle():
    binding, material, recipes, policy, closure = _fixture()
    return ModalCoordinatorBundle.build(
        binding, material, recipes, log_terminal_policy=policy,
        worker_closure_manifest=closure,
    )


def _mutate(document: bytes, path: tuple[object, ...], value: object) -> bytes:
    result = json.loads(document)
    cursor = result
    for name in path[:-1]:
        cursor = cursor[name]
    cursor[path[-1]] = value
    return canonical_bytes(result)


def test_round_trip_binds_exact_stage_command_and_eight_members():
    value = _bundle()
    assert tuple(member.name for member in value.members) == MEMBER_NAMES
    assert ModalCoordinatorBundle.parse_transport(
        value.transport_bytes, binding=value.binding, recipes=value.recipes,
    ) == value
    plan = json.loads(next(x.content for x in value.members if x.name == "stage-plan.json"))
    assert plan["stage_command_digest"] == value.binding.command_digest
    assert plan["workload_digest"] == json.loads(
        value.binding.command_bytes,
    )["preparation"]["workload_digest"]
    assert plan["workload_fingerprint"] == plan["workload_digest"]


def test_bundle_has_no_submit_or_invocation_authority():
    value = _bundle()
    encoded = value.canonical_bytes
    assert b"submit" not in encoded
    assert b"provider_job_ref" not in encoded
    assert b'"argv"' not in encoded
    assert b'"cwd"' not in encoded
    policy = json.loads(next(x.content for x in value.members if x.name == "log-terminal-policy.json"))
    assert "effect_id" not in policy and "run_id" not in policy


@pytest.mark.parametrize("field", ["generation", "max_log_chunks", "max_chunk_bytes", "max_terminal_bytes"])
def test_policy_rejects_nonpositive_or_boolean_bounds(field):
    binding, material, recipes, policy, closure = _fixture()
    policy = _mutate(policy, (field,), False)
    with pytest.raises(ValueError, match=field):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


@pytest.mark.parametrize("field", ["image", "dependency", "python"])
def test_rejects_resolved_runtime_different_from_deployment(field):
    runtime_lock = ModalRuntimeLockV1.packaged()
    values = {
        "image": runtime_lock.registry_reference,
        "dependency": runtime_lock.locked_digest("dependency_lock"),
        "python": runtime_lock.python_version,
    }
    values[field] = {
        "image": "other/image@sha256:" + "f" * 64,
        "dependency": "e" * 64,
        "python": "3.12.8",
    }[field]
    binding, material, recipes, policy, closure = _fixture(
        runtime=RuntimeSpec(values["image"], values["dependency"], values["python"]),
    )
    with pytest.raises(ValueError, match="runtime or resources"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


def test_rejects_other_registry_image_with_the_same_digest():
    runtime_lock = ModalRuntimeLockV1.packaged()
    binding, material, recipes, policy, closure = _fixture(runtime=RuntimeSpec(
        "attacker.invalid/other@sha256:" + runtime_lock.image_digest,
        runtime_lock.locked_digest("dependency_lock"), runtime_lock.python_version,
    ))
    with pytest.raises(ValueError, match="runtime or resources"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


@pytest.mark.parametrize("resources", [
    ResourceSpec("T4", 1, 900), ResourceSpec("A10", 2, 900),
    ResourceSpec("A10", 1, 901),
])
def test_rejects_resolved_resources_different_from_deployment(resources):
    binding, material, recipes, policy, closure = _fixture(resources=resources)
    with pytest.raises(ValueError, match="runtime or resources"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


@pytest.mark.parametrize("extra", [
    {"api_key": "literal-secret"},
    {"custom_api_key": "literal-secret"},
    {"client_password": "literal-secret"},
    {"refresh_token_value": "literal-secret"},
    {"endpoint": "https://user:password@example.invalid/data"},
    {"endpoint": "https://example.invalid/data?custom_api_key=value"},
    {"note": "aaaaaaaaaaaa.bbbbbbbbbbbb.cccccccccccc"},
])
def test_rejects_reconstructed_material_containing_secret_material(extra):
    binding, material, recipes, policy, closure = _fixture(
        request={"schema_version": "request/v1"} | extra,
    )
    with pytest.raises(ValueError, match="secret|credential|sensitive"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


def test_rejects_secret_field_compiled_into_workload():
    binding, material, recipes, policy, closure = _fixture(
        config_extra={"custom_api_key": "literal-secret"},
    )
    with pytest.raises(ValueError, match="forbidden secret field"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=closure,
        )


def test_tokenizer_revision_is_not_misclassified_as_a_secret():
    _bundle()


def test_rejects_structurally_valid_nonpackaged_worker_closure():
    binding, material, recipes, policy, closure = _fixture()
    document = json.loads(closure)
    document["members"][0]["sha256"] = "f" * 64
    document["closure_digest"] = closure_digest(document)
    changed = canonical_bytes(document) + b"\n"
    with pytest.raises(ValueError, match="packaged offline SFT worker"):
        ModalCoordinatorBundle.build(
            binding, material, recipes, log_terminal_policy=policy,
            worker_closure_manifest=changed,
        )


def test_artifact_cardinality_rejects_boolean_one():
    value = _bundle()
    transport = _tampered_transport(
        value, "artifact-contract.json", ("requirements", 0, "minimum"), True,
    )
    with pytest.raises(ValueError, match="exact singletons"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def _tampered_transport(value, name, path, replacement):
    outer = json.loads(base64.b64decode(value.transport_bytes))
    member = next(item for item in outer["members"] if item["name"] == name)
    content = _mutate(base64.b64decode(member["content_base64"]), path, replacement)
    member["content_base64"] = base64.b64encode(content).decode()
    member["size"] = len(content)
    member["sha256"] = _sha(content)
    if name != "stage-plan.json":
        plan_member = next(item for item in outer["members"] if item["name"] == "stage-plan.json")
        plan = json.loads(base64.b64decode(plan_member["content_base64"]))
        plan["members"][name] = {"size": len(content), "sha256": _sha(content)}
        plan_content = canonical_bytes(plan)
        plan_member["content_base64"] = base64.b64encode(plan_content).decode()
        plan_member["size"] = len(plan_content)
        plan_member["sha256"] = _sha(plan_content)
    return base64.b64encode(_large_canonical(outer))


def test_rejects_deployment_other_than_complete_binding():
    value = _bundle()
    changed = json.loads(value.binding.deployment_bytes)
    changed["selection"]["client_ref"] = "other-client"
    transport = _tampered_transport(
        value, "deployment.json", ("selection", "client_ref"), "other-client",
    )
    with pytest.raises(ValueError, match="deployment|attestation"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def test_rejects_workload_embedding_different_source():
    value = _bundle()
    transport = _tampered_transport(
        value, "workload.json", ("execution_source", "run_id"), "other-run",
    )
    with pytest.raises(ValueError, match="resolved material|different execution source"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def test_rejects_source_runtime_environment_not_selected():
    value = _bundle()
    transport = _tampered_transport(
        value, "execution-source.json", ("runtime", "environment", "variables", "LANG"), "other",
    )
    with pytest.raises(ValueError, match="resolved material|environment differs"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def test_rejects_artifact_contract_not_embedded_in_workload():
    value = _bundle()
    transport = _tampered_transport(
        value, "artifact-contract.json", ("requirements", 0, "role"), "not-a-role",
    )
    with pytest.raises(ValueError, match="artifact role set"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def test_transport_rejects_member_digest_tampering():
    value = _bundle()
    outer = json.loads(base64.b64decode(value.transport_bytes))
    outer["members"][0]["sha256"] = "0" * 64
    transport = base64.b64encode(_large_canonical(outer))
    with pytest.raises(ValueError, match="member digest mismatch"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )


def test_transport_rejects_binding_for_another_stage_command():
    value = _bundle()
    adapter, _, plan, execution = composed(resolved=value.material.planning_request)
    preparation = adapter.prepare(plan, TrainingRunRef("run-1", "project-a"), execution)
    other = build_stage_command(
        preparation, "other-nonce", adapter.payload(preparation, EffectKind.STAGE),
        execution.executor_descriptor,
    )
    other_binding = ModalCommandBinding(
        other.canonical_bytes, value.binding.preparation_snapshot, value.binding.deployment_bytes,
    )
    with pytest.raises(ValueError, match="identity differs"):
        ModalCoordinatorBundle.parse_transport(
            value.transport_bytes, binding=other_binding, recipes=value.recipes,
        )


def test_stage_plan_tampering_is_rejected_even_with_updated_outer_member_hash():
    value = _bundle()
    outer = json.loads(base64.b64decode(value.transport_bytes))
    member = next(item for item in outer["members"] if item["name"] == "stage-plan.json")
    plan = json.loads(base64.b64decode(member["content_base64"]))
    plan["workload_fingerprint"] = "f" * 64
    content = canonical_bytes(plan)
    member["content_base64"] = base64.b64encode(content).decode()
    member["size"] = len(content)
    member["sha256"] = _sha(content)
    transport = base64.b64encode(_large_canonical(outer))
    with pytest.raises(ValueError, match="stage plan"):
        ModalCoordinatorBundle.parse_transport(
            transport, binding=value.binding, recipes=value.recipes,
        )
