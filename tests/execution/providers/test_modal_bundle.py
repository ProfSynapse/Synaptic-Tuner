from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for

from tests.training.test_sft_compilation import _config, _execution_source
from tuner.execution.contracts import EffectIdentity, EffectKind, ExecutionScope
from tuner.execution.providers.modal.bundle import (
    MAX_TRANSPORT_BASE64_BYTES,
    ModalBundleMemberV1,
    ModalExecutionBundleV1,
    REQUIRED_MODAL_MEMBERS,
)
from tuner.execution.operation import ModalStageTargetV1, OperationBindingV1
from tuner.training.methods.sft import compile_sft_workload
from tuner.runtime.offline_sft_worker import load_packaged_offline_sft_worker_manifest


D = "d" * 64


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def effect() -> EffectIdentity:
    return EffectIdentity(
        "effect-1", "train-run-1", EffectKind.SUBMIT,
        ExecutionScope("modal", "acct", "env"),
    )


def documents() -> dict[str, bytes]:
    deployment = {
        "schema_version": "synaptic-verified-modal-deployment/v1",
        "selection": {
            "schema_version": "synaptic-modal-deployment-selection/v1",
            "account_ref": "acct", "workspace_ref": "workspace", "environment_ref": "env",
            "client_ref": "client", "sdk_version": "1.5.4", "app_name": "synaptic-training-v1",
            "function_name": "run_sft_v1_11111111111111111111111111111111",
            "deployment_ref": "modal-deployment-11111111111111111111111111111111",
            "image_digest": "1" * 64, "dependency_lock_digest": "2" * 64,
            "wrapper_digest": "3" * 64, "runtime_digest": "4" * 64,
            "python_version": "3.12.7", "python_executable": "/usr/local/bin/python3.12",
            "python_executable_digest": "5" * 64, "runtime_environment": {"PATH": "/usr/local/bin"},
            "secret_requirements_digest": "6" * 64,
            "provider_runtime_requirements_digest": "7" * 64,
            "accelerator": "A10", "timeout_seconds": 3600, "max_retries": 0,
        },
        "issuer_ref": "modal-verifier", "evidence_ref": "deployment-proof",
        "audience_ref": "project/run-1", "challenge_nonce": "deployment-nonce",
        "verified_at": "2026-08-25T12:02:00Z", "expires_at": "2026-08-25T12:07:00Z",
        "key_ref": "deployment-key", "tag_base64": "dGFn",
        "attestation_digest": "8" * 64,
    }
    deployment_bytes = canonical(deployment)
    source = replace(
        _execution_source(), deployment_member_sha256=sha(deployment_bytes),
        python_version="3.12.7", python_executable="/usr/local/bin/python3.12",
    )
    source_bytes = source.canonical_bytes
    workload = compile_sft_workload(resolved_config=_config(), execution_source=source)
    workload_bytes = workload.canonical_bytes
    artifact_bytes = canonical(workload.document["artifacts"])
    policy = {
        "schema_version": "synaptic-modal-log-terminal-policy/v1", "run_id": source.run_id,
        "effect_id": effect().effect_id, "generation": 1, "control_prefix": "control",
        "artifact_prefix": "output", "max_log_chunks": 1024,
        "max_chunk_bytes": 65536, "max_terminal_bytes": 65536,
    }
    policy_bytes = canonical(policy)
    closure = load_packaged_offline_sft_worker_manifest()
    plan = {
        "schema_version": "synaptic-training-plan/v1", "run_id": source.run_id,
        "effect_id": effect().effect_id, "effect_key": effect().effect_key,
        "provider": "modal", "account_ref": "acct", "namespace_ref": "env",
        "artifact_slot_ref": "slot-1", "deployment_digest": sha(deployment_bytes),
        "execution_source_digest": sha(source_bytes), "workload_digest": sha(workload_bytes),
        "artifact_contract_digest": sha(artifact_bytes), "log_policy_digest": sha(policy_bytes),
        "resource_digest": "9" * 64, "quote_digest": "a" * 64,
        "secret_requirements_digest": source.secret_requirements_digest,
        "worker_closure_manifest_sha256": closure.sha256,
        "worker_closure_digest": closure.closure.closure_digest,
    }
    plan_bytes = canonical(plan)
    environment = dict(source.environment)
    environment.pop("PYTHONPATH")
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = workload.fingerprint
    environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = (
        "/workspace/control/operations/effect-1/input/offline-sft-worker-v1.json"
    )
    environment["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = closure.closure.closure_digest
    environment["SYNAPTIC_MODEL_SNAPSHOT"] = (
        source.roots["cache"] + "/model/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/" + "b" * 40
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    invocation = {
        "schema_version": "synaptic-modal-invocation-intent/v1", "run_id": source.run_id,
        "effect_id": effect().effect_id, "plan_digest": sha(plan_bytes),
        "deployment_digest": sha(deployment_bytes), "execution_source_digest": sha(source_bytes),
        "workload_digest": sha(workload_bytes), "interpreter": source.python_executable,
        "argv": [source.python_executable, source.roots["engine"] + "/Trainers/sft/runtime_v1.py", "--canonical-workload-stdin"],
        "cwd": source.roots["tmp"], "environment_digest": sha(canonical(environment)),
        "invocation_nonce": "nonce-1",
    }
    invocation_bytes = canonical(invocation)
    return {
        "artifact-contract.json": artifact_bytes,
        "deployment.json": deployment_bytes,
        "execution-source.json": source_bytes,
        "invocation-intent.json": invocation_bytes,
        "log-terminal-policy.json": policy_bytes,
        "plan.json": plan_bytes,
        "workload.json": workload_bytes,
        "worker-closure-manifest.json": closure.canonical_bytes,
    }


def operation(values: dict[str, bytes] | None = None) -> OperationBindingV1:
    members = documents() if values is None else values
    return OperationBindingV1.from_predecessors(
        project_ref="project-1", grant_ref="grant-1", effect=effect(),
        stage_target=ModalStageTargetV1("slot-1", "control", "artifacts", "operations/effect-1/output", 1, "key-1"),
        member_documents=members,
    )


def bundle(values: dict[str, bytes] | None = None) -> ModalExecutionBundleV1:
    members = documents() if values is None else values
    return ModalExecutionBundleV1.build(
        operation=operation(members), member_documents=members,
    )


def mutate(values: dict[str, bytes], name: str, path: tuple[str, ...], replacement: object) -> dict[str, bytes]:
    result = dict(values)
    document = json.loads(result[name])
    cursor = document
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    result[name] = canonical(document)
    return result


def test_bundle_round_trips_and_binds_one_exact_intent() -> None:
    value = bundle()
    assert tuple(member.name for member in value.members) == REQUIRED_MODAL_MEMBERS
    assert value.to_stage_bundle().payload == value.transport_base64
    assert ModalExecutionBundleV1.parse_transport(value.transport_base64) == value


def test_bundle_wire_document_validates_against_checked_in_schema() -> None:
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "synaptic-modal-execution-bundle-v1.schema.json").read_text(encoding="utf-8")
    )
    validator = validator_for(schema)(schema)
    validator.validate(bundle().to_dict())


def test_schema_rejects_legacy_eight_member_bundle_without_worker_closure_manifest() -> None:
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "synaptic-modal-execution-bundle-v1.schema.json").read_text(encoding="utf-8")
    )
    validator = validator_for(schema)(schema)
    document = bundle().to_dict()
    document["members"] = [
        member for member in document["members"]
        if member["name"] != "worker-closure-manifest.json"
    ]
    with pytest.raises(ValidationError):
        validator.validate(document)


@pytest.mark.parametrize(
    ("name", "path", "replacement"),
    [
        ("execution-source.json", ("deployment_member_sha256",), D),
        ("workload.json", ("execution_source", "run_id"), "other"),
        ("workload.json", ("artifacts", "requirements",), []),
        ("plan.json", ("workload_digest",), D),
        ("plan.json", ("effect_id",), "other"),
        ("log-terminal-policy.json", ("run_id",), "other"),
        ("invocation-intent.json", ("plan_digest",), D),
        ("invocation-intent.json", ("argv",), ["python"]),
    ],
)
def test_every_cross_member_binding_fails_closed(name, path, replacement) -> None:
    with pytest.raises(ValueError):
        bundle(mutate(documents(), name, path, replacement))


def test_unknown_fields_and_secret_spelling_variants_are_rejected() -> None:
    values = documents()
    deployment = json.loads(values["deployment.json"])
    deployment["selection"]["hf_token"] = "hf_literal"
    values["deployment.json"] = canonical(deployment)
    with pytest.raises(ValueError, match="secret field|unknown fields"):
        bundle(values)

    values = documents()
    plan = json.loads(values["plan.json"])
    plan["extra"] = "value"
    values["plan.json"] = canonical(plan)
    with pytest.raises(ValueError, match="unknown fields"):
        bundle(values)


@pytest.mark.parametrize(
    "literal",
    [
        "Bearer abcdefghijklmnopqrstuvwxyz", "-----BEGIN PRIVATE KEY-----",
        "aaaaaaaaaaaa.bbbbbbbbbbbb.cccccccccccc",
        "https://user:password@example.com/repo.git",
        "https://example.com/path?access_token=value",
    ],
)
def test_literal_credential_shapes_are_rejected(literal: str) -> None:
    values = documents()
    policy = json.loads(values["log-terminal-policy.json"])
    policy["control_prefix"] = literal
    values["log-terminal-policy.json"] = canonical(policy)
    with pytest.raises(ValueError, match="credential|userinfo|sensitive"):
        bundle(values)


@pytest.mark.parametrize("field", ["bundle_digest", "stage_claim_digest", "command_digest"])
def test_stage_intent_cannot_create_a_digest_cycle(field: str) -> None:
    value = bundle()
    members = {member.name: member for member in value.members}
    stage = json.loads(members["stage-intent.json"].content)
    stage[field] = D
    with pytest.raises(ValueError, match="cyclic|unknown fields"):
        ModalExecutionBundleV1(
            value.operation,
            tuple(member for name, member in members.items() if name != "stage-intent.json")
            + (ModalBundleMemberV1("stage-intent.json", canonical(stage)),),
        )


def test_caller_cannot_substitute_a_different_valid_operation_digest() -> None:
    value = bundle()
    members = {member.name: member for member in value.members}
    stage = json.loads(members["stage-intent.json"].content)
    stage["operation_binding_digest"] = "c" * 64
    with pytest.raises(ValueError, match="derived operation"):
        ModalExecutionBundleV1(
            value.operation,
            tuple(member for name, member in members.items() if name != "stage-intent.json")
            + (ModalBundleMemberV1("stage-intent.json", canonical(stage)),),
        )


def test_direct_operation_construction_cannot_override_a_derived_predecessor_digest() -> None:
    values=documents();forged=replace(operation(values),invocation_arguments_digest="c"*64)
    with pytest.raises(ValueError,match="derived from the exact predecessor"):
        ModalExecutionBundleV1.build(operation=forged,member_documents=values)


def test_bundle_factory_rejects_an_externally_supplied_stage_member() -> None:
    members = documents()
    members["stage-intent.json"] = canonical({"schema_version": "synaptic-modal-stage-intent/v1"})
    with pytest.raises(ValueError, match="eight predecessor"):
        ModalExecutionBundleV1.build(operation=operation(), member_documents=members)


def test_transport_rejects_noncanonical_or_oversized_base64() -> None:
    value = bundle()
    with pytest.raises(ValueError):
        ModalExecutionBundleV1.parse_transport(value.transport_base64 + b"=")
    with pytest.raises(ValueError):
        ModalExecutionBundleV1.parse_transport(b"A" * (MAX_TRANSPORT_BASE64_BYTES + 4))


def test_member_rejects_bom_duplicate_and_nonfinite_json() -> None:
    for content in (
        b'\xef\xbb\xbf{"schema_version":"synaptic-training-plan/v1"}',
        b'{"schema_version":"synaptic-training-plan/v1","schema_version":"synaptic-training-plan/v1"}',
        b'{"schema_version":"synaptic-training-plan/v1","value":NaN}',
    ):
        with pytest.raises(ValueError):
            ModalBundleMemberV1("plan.json", content)
