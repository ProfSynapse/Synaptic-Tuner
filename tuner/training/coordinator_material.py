"""Canonical derivation of rich training inputs for coordinator planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from synaptic_tuner.api.v1.planning import ResolvedTrainingRequest as PlanningRequest
from synaptic_tuner.api.v1.training import (
    ArtifactPolicy, CanonicalDocument, ResolvedTrainingRequest, ResourceSpec,
    RuntimeSpec, TrainingRequest,
)
from tuner.project.execution_source import ExecutionSourceV1

from .recipes import CompiledWorkload, RecipeRegistry


_SCHEMA = "synaptic-coordinator-resolved-material/v1"
_MAX_MATERIAL_BYTES = 512 * 1024
_FIELDS = frozenset({
    "schema_version", "request_id", "project_ref", "run_id", "request",
    "execution_source", "execution_context", "resolved_config", "workload",
    "runtime", "resources", "artifact_policy", "artifact_contract", "digests",
})
_DIGEST_FIELDS = frozenset({
    "source", "resolved_input", "workload", "runtime", "artifact_policy",
    "artifact_contract_sha256",
})


def _canonical(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinator material must contain canonical JSON") from exc
    if len(encoded) > _MAX_MATERIAL_BYTES:
        raise ValueError("coordinator material exceeds its canonical bound")
    return encoded


def _domain(name: str, value: object) -> str:
    return hashlib.sha256(name.encode("ascii") + b"\0" + _canonical(value)).hexdigest()


def _mapping(value, name):
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact object")
    return value


def _runtime(value: RuntimeSpec) -> dict[str, object]:
    return {
        "image": value.image, "dependency_lock_digest": value.dependency_lock_digest,
        "python_version": value.python_version,
    }


def _resources(value: ResourceSpec) -> dict[str, object]:
    return {
        "accelerator": value.accelerator,
        "accelerator_count": value.accelerator_count,
        "timeout_seconds": value.timeout_seconds,
    }


def _policy(value: ArtifactPolicy) -> dict[str, object]:
    return {
        "required_kinds": list(value.required_kinds),
        "retain_checkpoints": value.retain_checkpoints,
    }


@dataclass(frozen=True, slots=True, init=False)
class CoordinatorResolvedMaterial:
    _canonical_bytes: bytes

    @classmethod
    def _validated(cls, document):
        value = object.__new__(cls)
        object.__setattr__(value, "_canonical_bytes", _canonical(document))
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    @property
    def planning_request(self) -> PlanningRequest:
        document = json.loads(self._canonical_bytes)
        digests = document["digests"]
        return PlanningRequest(
            "synaptic-resolved-training-request/v1", document["request_id"],
            document["project_ref"], digests["source"], digests["resolved_input"],
            digests["workload"], digests["runtime"], digests["artifact_policy"],
        )

    @property
    def run_id(self) -> str:
        return json.loads(self._canonical_bytes)["run_id"]

    def _bytes(self, name: str) -> bytes:
        return _canonical(json.loads(self._canonical_bytes)[name])

    @property
    def request_bytes(self): return self._bytes("request")
    @property
    def execution_source_bytes(self): return self._bytes("execution_source")
    @property
    def execution_context_bytes(self): return self._bytes("execution_context")
    @property
    def resolved_config_bytes(self): return self._bytes("resolved_config")
    @property
    def workload_bytes(self): return self._bytes("workload")
    @property
    def runtime_bytes(self): return self._bytes("runtime")
    @property
    def resources_bytes(self): return self._bytes("resources")
    @property
    def artifact_policy_bytes(self): return self._bytes("artifact_policy")
    @property
    def artifact_contract_bytes(self): return self._bytes("artifact_contract")
    @property
    def artifact_contract_sha256(self):
        return hashlib.sha256(self.artifact_contract_bytes).hexdigest()

    @classmethod
    def parse(cls, payload: bytes, recipes: RecipeRegistry) -> "CoordinatorResolvedMaterial":
        if type(payload) is not bytes or not payload or len(payload) > _MAX_MATERIAL_BYTES:
            raise ValueError("coordinator material bytes are invalid")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("coordinator material is invalid JSON") from exc
        if _canonical(document) != payload or type(document) is not dict or frozenset(document) != _FIELDS:
            raise ValueError("coordinator material is not an exact canonical document")
        if document["schema_version"] != _SCHEMA:
            raise ValueError("unsupported coordinator material schema")
        runtime = RuntimeSpec(**_mapping(document["runtime"], "runtime"))
        resources = ResourceSpec(**_mapping(document["resources"], "resources"))
        policy_value = _mapping(document["artifact_policy"], "artifact_policy")
        policy = ArtifactPolicy(
            required_kinds=tuple(policy_value.get("required_kinds", ())),
            retain_checkpoints=policy_value.get("retain_checkpoints"),
        )
        resolved = ResolvedTrainingRequest(
            TrainingRequest(CanonicalDocument.from_mapping(_mapping(document["request"], "request"))),
            ExecutionSourceV1.from_dict(_mapping(document["execution_source"], "execution_source")),
            CanonicalDocument.from_mapping(_mapping(document["execution_context"], "execution_context")),
            CanonicalDocument.from_mapping(_mapping(document["resolved_config"], "resolved_config")),
            CanonicalDocument.from_mapping(_mapping(document["workload"], "workload")),
            runtime, resources, policy,
        )
        rebuilt = derive_coordinator_material(
            resolved, recipes, request_id=document["request_id"],
            project_ref=document["project_ref"], run_id=document["run_id"],
        )
        if rebuilt.canonical_bytes != payload:
            raise ValueError("coordinator material does not match reconstructed inputs")
        return rebuilt


def derive_coordinator_material(
    resolved: ResolvedTrainingRequest, recipes: RecipeRegistry, *,
    request_id: str, project_ref: str, run_id: str,
) -> CoordinatorResolvedMaterial:
    if type(resolved) is not ResolvedTrainingRequest:
        raise TypeError("exact resolved training request required")
    if type(recipes) is not RecipeRegistry:
        raise TypeError("exact recipe registry required")
    if not all(type(value) is str and value and value == value.strip()
               for value in (request_id, project_ref, run_id)):
        raise ValueError("request, project, and run identities must be nonblank canonical text")
    exact = (
        (resolved.request, TrainingRequest, "request"),
        (resolved.request.document, CanonicalDocument, "request document"),
        (resolved.execution_source, ExecutionSourceV1, "execution source"),
        (resolved.execution_context, CanonicalDocument, "execution context"),
        (resolved.resolved_config, CanonicalDocument, "resolved config"),
        (resolved.workload, CanonicalDocument, "workload"),
        (resolved.runtime, RuntimeSpec, "runtime"),
        (resolved.resources, ResourceSpec, "resources"),
        (resolved.artifact_policy, ArtifactPolicy, "artifact policy"),
    )
    if any(type(value) is not kind for value, kind, _ in exact):
        failed = next(name for value, kind, name in exact if type(value) is not kind)
        raise TypeError(f"exact {failed} required")
    if resolved.execution_source.run_id != run_id:
        raise ValueError("allocated coordinator run differs from execution source run")
    config = resolved.resolved_config.to_dict()
    method = config.get("method")
    if type(method) is not str or not method:
        raise ValueError("resolved config requires a method")
    compiled = recipes.resolve(method).compile(
        resolved_config=resolved.resolved_config,
        execution_source=resolved.execution_source,
    )
    if type(compiled) is not CompiledWorkload:
        raise TypeError("recipe must return exact CompiledWorkload")
    workload = CanonicalDocument(compiled.canonical_bytes.decode("utf-8"))
    if workload != resolved.workload:
        raise ValueError("resolved workload differs from deterministic compilation")
    artifact_contract = _mapping(workload.to_dict().get("artifacts"), "artifact contract")
    requirements = artifact_contract.get("requirements")
    if type(requirements) is not list:
        raise ValueError("compiled artifact contract requires exact requirements")
    roles = {
        item.get("role") for item in requirements
        if type(item) is dict and type(item.get("role")) is str
    }
    if not set(resolved.artifact_policy.required_kinds).issubset(roles):
        raise ValueError("artifact policy requires roles absent from compiled contract")
    artifact_contract_bytes = _canonical(artifact_contract)
    request = resolved.request.document.to_dict()
    context = resolved.execution_context.to_dict()
    resources = _resources(resolved.resources)
    runtime = _runtime(resolved.runtime)
    policy = _policy(resolved.artifact_policy)
    resolved_input = {
        "request": request, "execution_context": context,
        "resolved_config": config, "resources": resources,
    }
    planning = PlanningRequest(
        "synaptic-resolved-training-request/v1", request_id, project_ref,
        resolved.execution_source.fingerprint,
        _domain("synaptic-coordinator-resolved-input/v1", resolved_input),
        compiled.fingerprint,
        _domain("synaptic-coordinator-runtime/v1", runtime),
        _domain("synaptic-coordinator-artifact-policy/v1", policy),
    )
    document = {
        "schema_version": _SCHEMA, "request_id": request_id,
        "project_ref": project_ref, "run_id": run_id, "request": request,
        "execution_source": resolved.execution_source.to_dict(),
        "execution_context": context, "resolved_config": config,
        "workload": workload.to_dict(), "runtime": runtime,
        "resources": resources, "artifact_policy": policy,
        "artifact_contract": artifact_contract,
        "digests": {
            "source": planning.source_digest,
            "resolved_input": planning.resolved_config_digest,
            "workload": planning.workload_digest,
            "runtime": planning.runtime_digest,
            "artifact_policy": planning.artifact_policy_digest,
            "artifact_contract_sha256": hashlib.sha256(artifact_contract_bytes).hexdigest(),
        },
    }
    return CoordinatorResolvedMaterial._validated(document)


__all__: list[str] = []
