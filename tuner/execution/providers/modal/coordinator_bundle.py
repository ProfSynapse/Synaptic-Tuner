"""Bounded Foundation-native Modal stage bundle.

This internal codec proves byte-level and typed relationships available at the
stage boundary using reconstructable coordinator resolved material.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qsl, urlsplit

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes, digest_text, parse_canonical_object,
)
from tuner.execution.foundation_v2.commands import StageCommandV2, parse_exact_command
from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest as PlanningRequest
from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.offline_sft_worker import (
    load_packaged_offline_sft_worker_manifest, parse_offline_sft_worker_manifest,
)
from tuner.training.recipes import CompiledWorkload
from tuner.training.coordinator_material import CoordinatorResolvedMaterial
from tuner.training.recipes import RecipeRegistry

from .config import ModalRuntimeLockV1
from .coordinator_binding import ModalCommandBinding
from .resolution import VerifiedModalDeploymentIdentityV1


MAX_TRANSPORT_BYTES = 8_388_608
MAX_CANONICAL_BYTES = 6_291_456
MAX_MEMBER_BYTES = 1_048_576
MAX_MEMBER_TOTAL_BYTES = 4_194_304
BUNDLE_SCHEMA = "synaptic-modal-coordinator-bundle/v2"
MEMBER_NAMES = (
    "artifact-contract.json",
    "deployment.json",
    "execution-source.json",
    "log-terminal-policy.json",
    "resolved-material.json",
    "stage-plan.json",
    "worker-closure-manifest.json",
    "workload.json",
)
_B64 = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
_JWT = re.compile(r"^[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}$")
_SENSITIVE = frozenset({
    "secret", "secrets", "credential", "credentials", "token", "apitoken",
    "hftoken", "modaltoken", "modaltokensecret", "accesstoken", "apikey",
    "accesskey", "privatekey", "authorization", "cookie", "password",
})
_ALLOWED_SENSITIVE = frozenset({
    "secret_requirements_digest", "provider_runtime_requirements_digest",
})


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _large_canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("bundle contains a non-canonical JSON value") from None


def _large_object(value: bytes) -> dict[str, object]:
    def pairs(items):
        result = {}
        for key, member in items:
            if key in result:
                raise ValueError("bundle contains a duplicate JSON key")
            result[key] = member
        return result
    try:
        document = json.loads(
            value.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("bundle is not strict UTF-8 JSON") from None
    if type(document) is not dict or _large_canonical(document) != value:
        raise ValueError("bundle is not a canonical JSON object")
    return document


def _object(value: bytes, name: str) -> dict[str, object]:
    document = parse_canonical_object(value, name=name)
    if canonical_bytes(document) != value:
        raise ValueError(f"{name} is not canonical")
    _reject_secrets(document)
    return document


def _sensitive_key(key: str) -> bool:
    # Preserve word boundaries before normalizing so `tokenizer_revision` is
    # not confused with credential-bearing `refresh_token_value`.
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    parts = tuple(part.lower() for part in re.split(r"[^A-Za-z0-9]+", separated) if part)
    normalized = "".join(parts)
    if normalized in _SENSITIVE:
        return True
    if any(part in {"secret", "secrets", "credential", "credentials", "token", "password", "cookie", "authorization"} for part in parts):
        return True
    return any(
        pair in {("api", "key"), ("access", "key"), ("private", "key")}
        for pair in zip(parts, parts[1:])
    )


def _reject_secrets(value: object) -> None:
    if type(value) is dict:
        for key, member in value.items():
            if key not in _ALLOWED_SENSITIVE and _sensitive_key(key):
                raise ValueError("bundle contains a forbidden secret field")
            _reject_secrets(member)
    elif type(value) is list:
        for member in value:
            _reject_secrets(member)
    elif type(value) is str:
        lowered = value.lower()
        if "-----begin " in lowered or lowered.startswith(("bearer ", "basic ")) or _JWT.fullmatch(value):
            raise ValueError("bundle contains literal credential material")
        if "://" in value:
            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("bundle URL contains credential material")
            if any(_sensitive_key(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
                raise ValueError("bundle URL contains a sensitive query parameter")


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} contains missing or unknown fields")
    return value


def _decode(value: bytes, *, maximum: int, name: str) -> bytes:
    if (type(value) is not bytes or not value or len(value) > maximum
            or len(value) % 4 or any(item not in _B64 for item in value)):
        raise ValueError(f"{name} is not bounded canonical Base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError(f"{name} is not canonical Base64") from None
    if base64.b64encode(decoded) != value:
        raise ValueError(f"{name} is not canonical Base64")
    return decoded


def _binding(value: ModalCommandBinding) -> tuple[ModalCommandBinding, StageCommandV2]:
    if type(value) is not ModalCommandBinding:
        raise TypeError("exact ModalCommandBinding required")
    rebuilt = ModalCommandBinding(
        value.command_bytes, value.preparation_snapshot, value.deployment_bytes,
    )
    if rebuilt != value:
        raise ValueError("Modal command binding reconstruction mismatch")
    command = parse_exact_command(rebuilt.command_bytes)
    if type(command) is not StageCommandV2:
        raise ValueError("exact Foundation stage command required")
    return rebuilt, command


def _material(
    value: CoordinatorResolvedMaterial, recipes: RecipeRegistry,
) -> CoordinatorResolvedMaterial:
    if type(value) is not CoordinatorResolvedMaterial or type(recipes) is not RecipeRegistry:
        raise TypeError("exact coordinator material and recipe registry required")
    rebuilt = CoordinatorResolvedMaterial.parse(value.canonical_bytes, recipes)
    if rebuilt.canonical_bytes != value.canonical_bytes:
        raise ValueError("coordinator resolved material reconstruction mismatch")
    _reject_secrets(_large_object(rebuilt.canonical_bytes))
    return rebuilt


def _artifact(document: dict[str, object]) -> None:
    _exact(document, {"schema_version", "requirements"}, "artifact contract")
    if document["schema_version"] != "synaptic-sft-artifacts/v1":
        raise ValueError("artifact contract schema unsupported")
    requirements = document["requirements"]
    if type(requirements) is not list or len(requirements) != 5:
        raise ValueError("artifact contract must contain five exact roles")
    roles: list[str] = []
    for item in requirements:
        requirement = _exact(item, {"role", "minimum", "maximum"}, "artifact requirement")
        if (type(requirement["role"]) is not str
                or type(requirement["minimum"]) is not int
                or type(requirement["maximum"]) is not int
                or requirement["minimum"] != 1 or requirement["maximum"] != 1):
            raise ValueError("artifact roles must be exact singletons")
        roles.append(requirement["role"])
    if set(roles) != {
        "workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer",
    } or len(set(roles)) != len(roles):
        raise ValueError("artifact role set is not exact")


def _policy(document: dict[str, object]) -> None:
    _exact(
        document,
        {"schema_version", "generation", "max_log_chunks", "max_chunk_bytes", "max_terminal_bytes"},
        "log-terminal policy",
    )
    if document["schema_version"] != "synaptic-modal-log-terminal-policy/v2":
        raise ValueError("log-terminal policy schema unsupported")
    for name, maximum in (
        ("generation", 2**31 - 1), ("max_log_chunks", 1_000_000),
        ("max_chunk_bytes", 1_048_576), ("max_terminal_bytes", 1_048_576),
    ):
        value = document[name]
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} is outside its closed bound")


def _expected_plan(
    binding: ModalCommandBinding, members: Mapping[str, "CoordinatorBundleMember"],
    source: ExecutionSourceV1, workload: CompiledWorkload, closure,
    material: CoordinatorResolvedMaterial,
) -> dict[str, object]:
    command = parse_exact_command(binding.command_bytes)
    prep = command.preparation
    return {
        "schema_version": "synaptic-modal-foundation-stage-plan/v1",
        "stage_command_digest": command.digest,
        "binding_digest": binding.authenticated_binding_digest,
        "resolved_material_sha256": _sha(material.canonical_bytes),
        "stage_effect_id": command.operation.effect.effect_id,
        "invocation_nonce": command.operation.invocation_nonce,
        "preparation_digest": prep.preparation_digest,
        "plan_fingerprint": prep.plan_fingerprint,
        "source_digest": prep.source_digest,
        "workload_digest": prep.workload_digest,
        "runtime_digest": prep.runtime_digest,
        "resource_digest": prep.resource_digest,
        "artifact_contract_digest": prep.artifact_contract_digest,
        "quote_digest": prep.quote_digest,
        "secret_requirements_digest": prep.secret_requirements_digest,
        "execution_binding_digest": prep.execution_binding_digest,
        "execution_source_fingerprint": source.fingerprint,
        "workload_fingerprint": workload.fingerprint,
        "worker_closure_digest": closure.closure.closure_digest,
        "members": {
            name: {"size": len(member.content), "sha256": member.sha256}
            for name, member in sorted(members.items()) if name != "stage-plan.json"
        },
    }


@dataclass(frozen=True, slots=True)
class CoordinatorBundleMember:
    name: str
    content: bytes

    def __post_init__(self) -> None:
        if self.name not in MEMBER_NAMES:
            raise ValueError("unknown coordinator bundle member")
        if type(self.content) is not bytes or not 0 < len(self.content) <= MAX_MEMBER_BYTES:
            raise ValueError("coordinator bundle member exceeds its bound")

    @property
    def sha256(self) -> str:
        return _sha(self.content)

    def to_dict(self) -> dict[str, object]:
        encoded = base64.b64encode(self.content)
        return {
            "name": self.name, "size": len(self.content), "sha256": self.sha256,
            "content_base64": encoded.decode("ascii"),
        }


def _validate(bundle: "ModalCoordinatorBundle") -> None:
    binding, command = _binding(bundle.binding)
    material = _material(bundle.material, bundle.recipes)
    members = {member.name: member for member in bundle.members}
    if members["resolved-material.json"].content != material.canonical_bytes:
        raise ValueError("resolved-material member differs from reconstructed material")
    planning = material.planning_request
    prep = command.preparation
    snapshot = parse_canonical_object(
        binding.preparation_snapshot, name="Modal preparation snapshot",
    )
    basis = dict(snapshot["basis"])
    basis["schema_version"] = "synaptic-resolved-training-request/v1"
    if PlanningRequest.from_dict(basis) != planning:
        raise ValueError("coordinator material differs from retained planning request")
    if (
        (planning.project_ref, material.run_id, planning.source_digest,
         planning.workload_digest, planning.runtime_digest,
         planning.artifact_policy_digest)
        != (prep.project_ref, prep.run_id, prep.source_digest, prep.workload_digest,
            prep.runtime_digest, prep.artifact_contract_digest)
    ):
        raise ValueError("coordinator material differs from stage preparation")
    deployment = VerifiedModalDeploymentIdentityV1.from_dict(
        _object(members["deployment.json"].content, "deployment")
    )
    if deployment != binding.deployment or members["deployment.json"].content != binding.deployment_bytes:
        raise ValueError("deployment member differs from complete Modal binding")
    ModalRuntimeLockV1.packaged().validate_selection(deployment.selection)

    source = ExecutionSourceV1.from_dict(
        _object(members["execution-source.json"].content, "execution source")
    )
    if source.canonical_bytes != members["execution-source.json"].content:
        raise ValueError("execution source does not round-trip canonically")
    if source.canonical_bytes != material.execution_source_bytes:
        raise ValueError("execution source differs from resolved material")
    selection = deployment.selection
    runtime = _object(material.runtime_bytes, "resolved runtime")
    resources = _object(material.resources_bytes, "resolved resources")
    image = runtime.get("image")
    if (
        type(image) is not str
        or image != ModalRuntimeLockV1.packaged().registry_reference
        or runtime.get("dependency_lock_digest") != selection.dependency_lock_digest
        or runtime.get("python_version") != selection.python_version
        or resources != {
            "accelerator": selection.accelerator,
            "accelerator_count": 1,
            "timeout_seconds": selection.timeout_seconds,
        }
    ):
        raise ValueError("resolved runtime or resources differ from deployment selection")
    environment = dict(source.environment)
    if environment.pop("PYTHONPATH", None) != source.roots["engine"]:
        raise ValueError("execution source PYTHONPATH does not bind its engine root")
    if any(environment.get(name) != value for name, value in selection.runtime_environment.items()):
        raise ValueError("execution source environment differs from deployment runtime")
    if {"PYTHONHOME", "PYTHONUSERBASE", "HF_TOKEN"} & set(environment):
        raise ValueError("execution source contains a forbidden worker variable")
    if (
        source.run_id != command.preparation.run_id
        or source.deployment_member_sha256 != members["deployment.json"].sha256
        or (source.python_version, source.python_executable, source.python_executable_digest)
        != (selection.python_version, selection.python_executable, selection.python_executable_digest)
        or source.secret_requirements_digest != selection.secret_requirements_digest
        or source.provider_runtime_requirements_digest
        != selection.provider_runtime_requirements_digest
    ):
        raise ValueError("execution source differs from stage runtime identity")

    workload_document = _object(members["workload.json"].content, "workload")
    workload = CompiledWorkload(
        workload_document.get("method"), workload_document.get("schema_version"),
        workload_document.get("entrypoint"), members["workload.json"].content,
    )
    if workload.method != "sft" or workload.entrypoint != "Trainers/sft/runtime_v1.py":
        raise ValueError("only the fixed SFT workload is supported")
    if workload.canonical_bytes != material.workload_bytes:
        raise ValueError("workload differs from resolved material")
    if canonical_bytes(workload.document.get("execution_source")) != source.canonical_bytes:
        raise ValueError("workload embeds a different execution source")
    artifact = _object(members["artifact-contract.json"].content, "artifact contract")
    _artifact(artifact)
    if members["artifact-contract.json"].content != material.artifact_contract_bytes:
        raise ValueError("artifact contract differs from resolved material")
    if workload.document.get("artifacts") != artifact:
        raise ValueError("workload embeds a different artifact contract")

    policy = _object(members["log-terminal-policy.json"].content, "log-terminal policy")
    _policy(policy)
    closure = parse_offline_sft_worker_manifest(
        members["worker-closure-manifest.json"].content,
        source_ref="modal-coordinator-bundle:worker-closure-manifest.json",
        manifest_path=Path("worker-closure-manifest.json"),
    )
    packaged_closure = load_packaged_offline_sft_worker_manifest()
    if (
        members["worker-closure-manifest.json"].content != packaged_closure.canonical_bytes
        or closure.closure.closure_digest != packaged_closure.closure.closure_digest
    ):
        raise ValueError("worker closure differs from packaged offline SFT worker")
    plan = _object(members["stage-plan.json"].content, "stage plan")
    expected = _expected_plan(binding, members, source, workload, closure, material)
    if plan != expected:
        raise ValueError("stage plan does not bind exact command and members")


@dataclass(frozen=True, slots=True)
class ModalCoordinatorBundle:
    binding: ModalCommandBinding
    material: CoordinatorResolvedMaterial
    recipes: RecipeRegistry = field(repr=False, compare=False)
    members: tuple[CoordinatorBundleMember, ...]

    def __post_init__(self) -> None:
        rebuilt, _ = _binding(self.binding)
        material = _material(self.material, self.recipes)
        members = tuple(self.members)
        if any(type(member) is not CoordinatorBundleMember for member in members):
            raise TypeError("exact CoordinatorBundleMember values required")
        if tuple(sorted(member.name for member in members)) != MEMBER_NAMES:
            raise ValueError("coordinator bundle requires the exact member set")
        if sum(len(member.content) for member in members) > MAX_MEMBER_TOTAL_BYTES:
            raise ValueError("coordinator bundle members exceed aggregate bound")
        object.__setattr__(self, "binding", rebuilt)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda item: item.name)))
        _validate(self)
        if len(self.canonical_bytes) > MAX_CANONICAL_BYTES or len(self.transport_bytes) > MAX_TRANSPORT_BYTES:
            raise ValueError("coordinator bundle transport exceeds bound")

    def to_dict(self) -> dict[str, object]:
        command = parse_exact_command(self.binding.command_bytes)
        return {
            "schema_version": BUNDLE_SCHEMA,
            "stage_command_digest": command.digest,
            "binding_digest": self.binding.authenticated_binding_digest,
            "stage_effect_id": command.operation.effect.effect_id,
            "invocation_nonce": command.operation.invocation_nonce,
            "members": [member.to_dict() for member in self.members],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _large_canonical(self.to_dict())

    @property
    def transport_bytes(self) -> bytes:
        return base64.b64encode(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        return _sha(self.transport_bytes)

    @classmethod
    def build(
        cls, binding: ModalCommandBinding, material: CoordinatorResolvedMaterial,
        recipes: RecipeRegistry, *, log_terminal_policy: bytes,
        worker_closure_manifest: bytes,
    ) -> "ModalCoordinatorBundle":
        rebuilt, _ = _binding(binding)
        resolved = _material(material, recipes)
        member_documents = {
            "deployment.json": rebuilt.deployment_bytes,
            "execution-source.json": resolved.execution_source_bytes,
            "workload.json": resolved.workload_bytes,
            "artifact-contract.json": resolved.artifact_contract_bytes,
            "log-terminal-policy.json": log_terminal_policy,
            "resolved-material.json": resolved.canonical_bytes,
            "worker-closure-manifest.json": worker_closure_manifest,
        }
        members = {name: CoordinatorBundleMember(name, value) for name, value in member_documents.items()}
        deployment = VerifiedModalDeploymentIdentityV1.from_dict(
            _object(members["deployment.json"].content, "deployment")
        )
        source = ExecutionSourceV1.from_dict(
            _object(members["execution-source.json"].content, "execution source")
        )
        workload_doc = _object(members["workload.json"].content, "workload")
        workload = CompiledWorkload(
            workload_doc.get("method"), workload_doc.get("schema_version"),
            workload_doc.get("entrypoint"), members["workload.json"].content,
        )
        closure = parse_offline_sft_worker_manifest(
            members["worker-closure-manifest.json"].content,
            source_ref="modal-coordinator-bundle:worker-closure-manifest.json",
            manifest_path=Path("worker-closure-manifest.json"),
        )
        if deployment != rebuilt.deployment:
            raise ValueError("deployment member differs from complete Modal binding")
        plan = canonical_bytes(_expected_plan(rebuilt, members, source, workload, closure, resolved))
        members["stage-plan.json"] = CoordinatorBundleMember("stage-plan.json", plan)
        return cls(rebuilt, resolved, recipes, tuple(members.values()))

    @classmethod
    def parse_transport(
        cls, transport: bytes, *, binding: ModalCommandBinding,
        recipes: RecipeRegistry,
    ) -> "ModalCoordinatorBundle":
        rebuilt, command = _binding(binding)
        decoded = _decode(transport, maximum=MAX_TRANSPORT_BYTES, name="coordinator bundle transport")
        if len(decoded) > MAX_CANONICAL_BYTES:
            raise ValueError("decoded coordinator bundle exceeds bound")
        document = _large_object(decoded)
        _exact(document, {"schema_version", "stage_command_digest", "binding_digest", "stage_effect_id", "invocation_nonce", "members"}, "coordinator bundle")
        if document["schema_version"] != BUNDLE_SCHEMA:
            raise ValueError("coordinator bundle schema unsupported")
        if (
            document["stage_command_digest"] != command.digest
            or document["binding_digest"] != rebuilt.authenticated_binding_digest
            or document["stage_effect_id"] != command.operation.effect.effect_id
            or document["invocation_nonce"] != command.operation.invocation_nonce
        ):
            raise ValueError("coordinator bundle identity differs from binding")
        values = document["members"]
        if type(values) is not list or len(values) != len(MEMBER_NAMES):
            raise ValueError("coordinator bundle member collection is malformed")
        members = []
        for value in values:
            item = _exact(value, {"name", "size", "sha256", "content_base64"}, "bundle member")
            if type(item["content_base64"]) is not str or not item["content_base64"].isascii():
                raise ValueError("bundle member content is not ASCII Base64")
            content = _decode(
                item["content_base64"].encode("ascii"),
                maximum=4 * ((MAX_MEMBER_BYTES + 2) // 3), name="bundle member content",
            )
            if type(item["size"]) is not int or item["size"] != len(content):
                raise ValueError("bundle member size mismatch")
            digest_text(item["sha256"], "member sha256")
            if item["sha256"] != _sha(content):
                raise ValueError("bundle member digest mismatch")
            members.append(CoordinatorBundleMember(item["name"], content))
        material_member = next(
            member for member in members if member.name == "resolved-material.json"
        )
        material = CoordinatorResolvedMaterial.parse(material_member.content, recipes)
        result = cls(rebuilt, material, recipes, tuple(members))
        if result.transport_bytes != transport:
            raise ValueError("coordinator bundle does not round-trip canonically")
        return result


__all__: list[str] = []
