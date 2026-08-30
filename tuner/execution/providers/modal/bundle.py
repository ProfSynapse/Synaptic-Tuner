"""Canonical Modal bundle with strict member schemas and cross-plane binding."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Mapping
from urllib.parse import parse_qsl, urlsplit

from tuner.execution.contracts import EffectIdentity, safe_ref
from tuner.execution.operation import OperationBindingV1
from tuner.execution.providers.contracts import StageBundle
from tuner.project.execution_source import ExecutionSourceV1


MAX_TRANSPORT_BASE64_BYTES = 8_388_608
MAX_DECODED_CANONICAL_JSON_BYTES = 6_291_456
MAX_MEMBER_DECODED_BYTES = 1_048_576
MAX_MEMBER_TOTAL_DECODED_BYTES = 4_194_304
MODAL_BUNDLE_SCHEMA = "synaptic-modal-execution-bundle/v1"
REQUIRED_MODAL_MEMBERS = (
    "artifact-contract.json", "deployment.json", "execution-source.json",
    "invocation-intent.json", "log-terminal-policy.json", "plan.json",
    "stage-intent.json", "workload.json",
)
_MEMBER_SCHEMAS = {
    "artifact-contract.json": "synaptic-sft-artifacts/v1",
    "deployment.json": "synaptic-verified-modal-deployment/v1",
    "execution-source.json": "synaptic-execution-source/v1",
    "invocation-intent.json": "synaptic-modal-invocation-intent/v1",
    "log-terminal-policy.json": "synaptic-modal-log-terminal-policy/v1",
    "plan.json": "synaptic-training-plan/v1",
    "stage-intent.json": "synaptic-modal-stage-intent/v1",
    "workload.json": "synaptic-sft-workload/v1",
}
_B64_RE = re.compile(rb"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\Z")
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEYS = frozenset({
    "secret", "secrets", "credential", "credentials", "token", "apitoken",
    "hftoken", "modaltoken", "modaltokensecret", "accesstoken", "apikey",
    "accesskey", "privatekey", "authorization", "cookie", "password",
})
_ALLOWED_SENSITIVE_KEYS = frozenset({
    "secret_requirements_digest", "provider_runtime_requirements_digest",
})
_CYCLIC_KEYS = frozenset({"bundle_digest", "stage_claim_digest", "command_digest", "final_command_digest"})


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON values are forbidden")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("bundle must contain only finite JSON values") from exc


def _normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _reject_secret_value(value: str) -> None:
    lowered = value.lower()
    if (
        "-----begin " in lowered
        or lowered.startswith(("bearer ", "basic "))
        or _JWT_RE.fullmatch(value) is not None
    ):
        raise ValueError("bundle contains literal credential material")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("bundle URL contains userinfo")
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            if _normalize_key(key) in _SENSITIVE_KEYS:
                raise ValueError("bundle URL contains a sensitive query parameter")


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, member in value.items():
            normalized = _normalize_key(key)
            if key not in _ALLOWED_SENSITIVE_KEYS and (
                normalized in _SENSITIVE_KEYS or normalized.endswith("token")
            ):
                raise ValueError("bundle contains a forbidden secret field")
            if key in _CYCLIC_KEYS:
                raise ValueError("bundle member contains a cyclic digest field")
            _reject_forbidden(member)
    elif isinstance(value, list):
        for member in value:
            _reject_forbidden(member)
    elif isinstance(value, str):
        _reject_secret_value(value)


def _strict_json(data: bytes, *, expected_keys: set[str] | None = None) -> dict[str, object]:
    if not isinstance(data, bytes) or not data or data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("record must be nonempty BOM-free bytes")
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("record must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("record must be a JSON object")
    if expected_keys is not None and set(value) != expected_keys:
        raise ValueError("record contains missing or unknown fields")
    if _canonical(value) != data:
        raise ValueError("record must use canonical JSON encoding")
    _reject_forbidden(value)
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} contains missing or unknown fields")
    return value


def _artifact_contract(value: dict[str, object]) -> None:
    _exact(value, {"schema_version", "requirements"}, "artifact contract")
    requirements = value.get("requirements")
    if not isinstance(requirements, list) or len(requirements) != 5:
        raise ValueError("artifact contract requires exactly five roles")
    roles: list[str] = []
    for item in requirements:
        record = _exact(item, {"role", "minimum", "maximum"}, "artifact requirement")
        if record["minimum"] != 1 or record["maximum"] != 1 or not isinstance(record["role"], str):
            raise ValueError("artifact requirements must be exact singleton roles")
        roles.append(record["role"])
    if set(roles) != {"workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer"}:
        raise ValueError("artifact role set is not exact")


_SELECTION_KEYS = {
    "schema_version", "account_ref", "workspace_ref", "environment_ref", "client_ref",
    "sdk_version", "app_name", "function_name", "deployment_ref",
    "image_digest", "dependency_lock_digest", "wrapper_digest", "runtime_digest",
    "python_version", "python_executable", "python_executable_digest",
    "runtime_environment", "secret_requirements_digest",
    "provider_runtime_requirements_digest", "accelerator", "timeout_seconds", "max_retries",
}


def _deployment(value: dict[str, object]) -> None:
    _exact(
        value,
        {"schema_version", "selection", "issuer_ref", "evidence_ref", "audience_ref",
         "challenge_nonce", "verified_at", "expires_at", "key_ref", "tag_base64",
         "attestation_digest"},
        "deployment",
    )
    selection = _exact(value["selection"], _SELECTION_KEYS, "deployment selection")
    if (
        selection["schema_version"] != "synaptic-modal-deployment-selection/v1"
        or selection["sdk_version"] != "1.5.4" or selection["accelerator"] != "A10"
        or selection["max_retries"] != 0 or type(selection["timeout_seconds"]) is not int
    ):
        raise ValueError("deployment selection violates the Modal v1 contract")
    for key in (
        "image_digest", "dependency_lock_digest", "wrapper_digest", "runtime_digest",
        "python_executable_digest", "secret_requirements_digest",
        "provider_runtime_requirements_digest",
    ):
        _digest(selection[key], key)
    _digest(value["attestation_digest"], "deployment attestation")
    if not isinstance(selection["runtime_environment"], dict):
        raise ValueError("deployment runtime environment is malformed")


def _workload(value: dict[str, object]) -> None:
    _exact(
        value,
        {"schema_version", "method", "entrypoint", "execution_source", "configuration",
         "identities", "runtime_requirements", "artifacts"},
        "workload",
    )
    if value["method"] != "sft" or value["entrypoint"] != "Trainers/sft/runtime_v1.py":
        raise ValueError("workload method or entrypoint is unsupported")
    ExecutionSourceV1.from_dict(value["execution_source"])
    _artifact_contract(value["artifacts"])


def _validate_member_shape(name: str, value: dict[str, object]) -> None:
    if value.get("schema_version") != _MEMBER_SCHEMAS[name]:
        raise ValueError("Modal bundle member has the wrong versioned schema")
    if name == "deployment.json":
        _deployment(value)
    elif name == "execution-source.json":
        ExecutionSourceV1.from_dict(value)
    elif name == "workload.json":
        _workload(value)
    elif name == "artifact-contract.json":
        _artifact_contract(value)
    elif name == "log-terminal-policy.json":
        _exact(value, {"schema_version", "run_id", "effect_id", "generation", "control_prefix", "artifact_prefix", "max_log_chunks", "max_chunk_bytes", "max_terminal_bytes"}, "log policy")
    elif name == "plan.json":
        _exact(value, {"schema_version", "run_id", "effect_id", "effect_key", "provider", "account_ref", "namespace_ref", "artifact_slot_ref", "deployment_digest", "execution_source_digest", "workload_digest", "artifact_contract_digest", "log_policy_digest", "resource_digest", "quote_digest", "secret_requirements_digest"}, "training plan")
    elif name == "invocation-intent.json":
        _exact(value, {"schema_version", "run_id", "effect_id", "plan_digest", "deployment_digest", "execution_source_digest", "workload_digest", "interpreter", "argv", "cwd", "environment_digest", "invocation_nonce"}, "invocation intent")
    elif name == "stage-intent.json":
        _exact(value, {"schema_version", "operation_binding", "operation_binding_digest", "members"}, "stage intent")


def _encoded_size(decoded_size: int) -> int:
    if type(decoded_size) is not int or decoded_size < 0:
        raise ValueError("decoded_size must be a non-negative integer")
    return 4 * ((decoded_size + 2) // 3)


def _decode_base64(value: bytes, *, maximum: int, name: str) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) > maximum or len(value) % 4 or _B64_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be strict padded standard Base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{name} must be strict Base64") from exc
    if base64.b64encode(decoded) != value:
        raise ValueError(f"{name} is not canonically Base64 encoded")
    return decoded


@dataclass(frozen=True, slots=True)
class ModalBundleMemberV1:
    name: str
    content: bytes

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_MODAL_MEMBERS:
            raise ValueError("unknown Modal bundle member")
        if not isinstance(self.content, bytes) or not 0 < len(self.content) <= MAX_MEMBER_DECODED_BYTES:
            raise ValueError("Modal bundle member exceeds its decoded bound")
        _validate_member_shape(self.name, _strict_json(self.content))

    @property
    def document(self) -> dict[str, object]:
        return _strict_json(self.content)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def to_dict(self) -> dict[str, object]:
        encoded = base64.b64encode(self.content)
        return {"name": self.name, "decoded_size": len(self.content), "encoded_size": len(encoded), "sha256": self.sha256, "content_base64": encoded.decode("ascii")}


def _validate_semantics(bundle: "ModalExecutionBundleV1") -> None:
    members = {member.name: member for member in bundle.members}
    documents = {name: member.document for name, member in members.items()}
    deployment = documents["deployment.json"]
    source = documents["execution-source.json"]
    workload = documents["workload.json"]
    artifacts = documents["artifact-contract.json"]
    policy = documents["log-terminal-policy.json"]
    plan = documents["plan.json"]
    invocation = documents["invocation-intent.json"]
    stage = documents["stage-intent.json"]
    selection = deployment["selection"]
    source_value = ExecutionSourceV1.from_dict(source)

    if source_value.deployment_member_sha256 != members["deployment.json"].sha256:
        raise ValueError("execution source does not bind deployment.json")
    if _canonical(workload["execution_source"]) != members["execution-source.json"].content:
        raise ValueError("workload execution_source differs from execution-source.json")
    if workload["artifacts"] != artifacts:
        raise ValueError("workload artifacts differ from artifact-contract.json")
    expected_plan = {
        "run_id": source_value.run_id, "effect_id": bundle.effect.effect_id,
        "effect_key": bundle.effect.effect_key, "provider": bundle.effect.scope.provider,
        "account_ref": bundle.effect.scope.account_ref,
        "namespace_ref": bundle.effect.scope.namespace_ref,
        "deployment_digest": members["deployment.json"].sha256,
        "execution_source_digest": members["execution-source.json"].sha256,
        "workload_digest": members["workload.json"].sha256,
        "artifact_contract_digest": members["artifact-contract.json"].sha256,
        "log_policy_digest": members["log-terminal-policy.json"].sha256,
        "secret_requirements_digest": source_value.secret_requirements_digest,
    }
    if any(plan.get(key) != value for key, value in expected_plan.items()):
        raise ValueError("training plan does not bind the exact execution inputs")
    if policy.get("run_id") != source_value.run_id or policy.get("effect_id") != bundle.effect.effect_id:
        raise ValueError("log policy identity does not bind the plan")
    environment = dict(source_value.environment)
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = hashlib.sha256(
        b"synaptic-training-workload/v1\0" + members["workload.json"].content
    ).hexdigest()
    expected_invocation = {
        "run_id": source_value.run_id, "effect_id": bundle.effect.effect_id,
        "plan_digest": members["plan.json"].sha256,
        "deployment_digest": members["deployment.json"].sha256,
        "execution_source_digest": members["execution-source.json"].sha256,
        "workload_digest": members["workload.json"].sha256,
        "interpreter": source_value.python_executable,
        "argv": [source_value.python_executable, source_value.roots["engine"] + "/Trainers/sft/runtime_v1.py", "--canonical-workload-stdin"],
        "cwd": source_value.roots["tmp"], "environment_digest": hashlib.sha256(_canonical(environment)).hexdigest(),
        "invocation_nonce": bundle.invocation_nonce,
    }
    if any(invocation.get(key) != value for key, value in expected_invocation.items()):
        raise ValueError("invocation intent does not bind the exact runtime invocation")
    operation = OperationBindingV1.from_dict(stage.get("operation_binding"))
    if operation != bundle.operation or stage.get("operation_binding_digest") != operation.digest:
        raise ValueError("stage intent does not bind the derived operation")
    expected_operation = OperationBindingV1.from_predecessors(
        project_ref=operation.project_ref, grant_ref=operation.grant_ref,
        effect=operation.effect, stage_target=operation.stage_target,
        member_documents={
            name: member.content for name, member in members.items()
            if name != "stage-intent.json"
        },
        target_provider_job_ref=operation.target_provider_job_ref,
    )
    # The public training-plan fingerprint is authenticated by the operation
    # and durable preparation.  The private bundle members remain the existing
    # schema; rebuilding here uses that already authenticated operation value.
    expected_operation = replace(
        expected_operation, plan_fingerprint=operation.plan_fingerprint
    )
    if operation != expected_operation:
        raise ValueError("operation binding was not derived from the exact predecessor members")
    scope = operation.effect.scope
    if (
        operation.grant_ref != bundle.grant_ref
        or operation.effect != bundle.effect
        or operation.invocation_nonce != bundle.invocation_nonce
        or operation.stage_target.artifact_slot_ref != plan.get("artifact_slot_ref")
        or operation.execution_source_digest != members["execution-source.json"].sha256
        or operation.workload_digest != members["workload.json"].sha256
        or operation.artifact_contract_digest != members["artifact-contract.json"].sha256
        or operation.log_policy_digest != members["log-terminal-policy.json"].sha256
        or operation.invocation_intent_digest != members["invocation-intent.json"].sha256
        or operation.deployment_attestation_digest != deployment.get("attestation_digest")
        or operation.resource_digest != plan.get("resource_digest")
        or operation.quote_digest != plan.get("quote_digest")
        or operation.secret_requirements_digest != plan.get("secret_requirements_digest")
    ):
        raise ValueError("stage operation differs from the exact predecessor members")
    declared = stage.get("members")
    expected_members = {
        name: {"sha256": member.sha256, "size": len(member.content)}
        for name, member in members.items() if name != "stage-intent.json"
    }
    if declared != expected_members:
        raise ValueError("stage intent does not bind the exact seven predecessor members")
    if selection["account_ref"] != scope.account_ref or selection["environment_ref"] != scope.namespace_ref:
        raise ValueError("deployment scope does not bind the effect scope")


@dataclass(frozen=True, slots=True)
class ModalExecutionBundleV1:
    operation: OperationBindingV1
    members: tuple[ModalBundleMemberV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operation, OperationBindingV1):
            raise TypeError("operation must be OperationBindingV1")
        members = tuple(self.members)
        if any(not isinstance(member, ModalBundleMemberV1) for member in members):
            raise TypeError("members must be ModalBundleMemberV1 values")
        if tuple(sorted(member.name for member in members)) != REQUIRED_MODAL_MEMBERS:
            raise ValueError("Modal execution bundle requires the exact member set")
        if sum(len(member.content) for member in members) > MAX_MEMBER_TOTAL_DECODED_BYTES:
            raise ValueError("Modal bundle members exceed the aggregate decoded bound")
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda item: item.name)))
        _validate_semantics(self)
        if len(self.canonical_bytes) > MAX_DECODED_CANONICAL_JSON_BYTES or len(self.transport_base64) > MAX_TRANSPORT_BASE64_BYTES:
            raise ValueError("Modal bundle transport exceeds its canonical bound")

    @property
    def grant_ref(self) -> str:
        return self.operation.grant_ref

    @property
    def effect(self) -> EffectIdentity:
        return self.operation.effect

    @property
    def invocation_nonce(self) -> str:
        return self.operation.invocation_nonce

    def to_dict(self) -> dict[str, object]:
        scope = self.effect.scope
        return {"schema_version": MODAL_BUNDLE_SCHEMA, "grant_ref": self.grant_ref, "effect": {"effect_id": self.effect.effect_id, "effect_key": self.effect.effect_key, "kind": self.effect.kind.value, "scope": {"provider": scope.provider, "account_ref": scope.account_ref, "namespace_ref": scope.namespace_ref}}, "invocation_nonce": self.invocation_nonce, "members": [member.to_dict() for member in self.members]}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def transport_base64(self) -> bytes:
        return base64.b64encode(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.transport_base64).hexdigest()

    def to_stage_bundle(self) -> StageBundle:
        return StageBundle(payload=self.transport_base64, sha256=self.sha256)

    @classmethod
    def build(cls, *, operation: OperationBindingV1, member_documents: Mapping[str, bytes]) -> "ModalExecutionBundleV1":
        predecessors = set(REQUIRED_MODAL_MEMBERS) - {"stage-intent.json"}
        if not isinstance(operation, OperationBindingV1):
            raise TypeError("operation must be OperationBindingV1")
        if not isinstance(member_documents, Mapping) or set(member_documents) != predecessors:
            raise ValueError("Modal execution bundle requires exactly seven predecessor members")
        validated = {
            name: ModalBundleMemberV1(name, member_documents[name]) for name in predecessors
        }
        stage = {
            "schema_version": "synaptic-modal-stage-intent/v1",
            "operation_binding": operation.to_dict(),
            "operation_binding_digest": operation.digest,
            "members": {
                name: {"sha256": member.sha256, "size": len(member.content)}
                for name, member in sorted(validated.items())
            },
        }
        members = tuple(validated.values()) + (
            ModalBundleMemberV1("stage-intent.json", _canonical(stage)),
        )
        return cls(operation, members)

    @classmethod
    def parse_transport(cls, transport: bytes) -> "ModalExecutionBundleV1":
        decoded = _decode_base64(transport, maximum=MAX_TRANSPORT_BASE64_BYTES, name="Modal bundle transport")
        if len(decoded) > MAX_DECODED_CANONICAL_JSON_BYTES:
            raise ValueError("decoded canonical Modal bundle exceeds its bound")
        document = _strict_json(decoded, expected_keys={"schema_version", "grant_ref", "effect", "invocation_nonce", "members"})
        if document["schema_version"] != MODAL_BUNDLE_SCHEMA:
            raise ValueError("unsupported Modal execution-bundle schema")
        from tuner.execution.contracts import EffectKind, ExecutionScope
        effect_value = _exact(document["effect"], {"effect_id", "effect_key", "kind", "scope"}, "bundle effect")
        scope_value = _exact(effect_value["scope"], {"provider", "account_ref", "namespace_ref"}, "bundle scope")
        try:
            effect = EffectIdentity(effect_value["effect_id"], effect_value["effect_key"], EffectKind(effect_value["kind"]), ExecutionScope(**scope_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid bundle effect identity") from exc
        raw_members = document["members"]
        if not isinstance(raw_members, list) or len(raw_members) != len(REQUIRED_MODAL_MEMBERS):
            raise ValueError("invalid Modal bundle member collection")
        members: list[ModalBundleMemberV1] = []
        for raw_value in raw_members:
            raw = _exact(raw_value, {"name", "decoded_size", "encoded_size", "sha256", "content_base64"}, "bundle member")
            if type(raw["decoded_size"]) is not int or type(raw["encoded_size"]) is not int:
                raise ValueError("Modal member sizes must be strict integers")
            encoded_text = raw["content_base64"]
            if not isinstance(encoded_text, str) or not encoded_text.isascii():
                raise ValueError("Modal member content must be ASCII Base64")
            encoded = encoded_text.encode("ascii")
            content = _decode_base64(encoded, maximum=_encoded_size(MAX_MEMBER_DECODED_BYTES), name="Modal member content")
            if raw["decoded_size"] != len(content) or raw["encoded_size"] != _encoded_size(len(content)) or raw["encoded_size"] != len(encoded) or raw["sha256"] != hashlib.sha256(content).hexdigest():
                raise ValueError("Modal member size or digest mismatch")
            members.append(ModalBundleMemberV1(raw["name"], content))
        stage_member = next(member for member in members if member.name == "stage-intent.json")
        operation = OperationBindingV1.from_dict(stage_member.document["operation_binding"])
        if (
            document["grant_ref"] != operation.grant_ref
            or effect != operation.effect
            or document["invocation_nonce"] != operation.invocation_nonce
        ):
            raise ValueError("outer bundle identity differs from the operation")
        result = cls(operation, tuple(members))
        if result.transport_base64 != transport:
            raise ValueError("Modal bundle transport does not round-trip canonically")
        return result


__all__ = [
    "MAX_DECODED_CANONICAL_JSON_BYTES", "MAX_MEMBER_DECODED_BYTES",
    "MAX_MEMBER_TOTAL_DECODED_BYTES", "MAX_TRANSPORT_BASE64_BYTES",
    "MODAL_BUNDLE_SCHEMA", "ModalBundleMemberV1", "ModalExecutionBundleV1",
    "REQUIRED_MODAL_MEMBERS",
]
