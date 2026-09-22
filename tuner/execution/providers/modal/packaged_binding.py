"""Closed bindings for the packaged-runtime Modal adapter.

The provider facts are deliberately separate from the provider-neutral runtime
release.  A consumer retains and authenticates the complete command binding;
digests are identifiers, never mutation authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Callable, Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.runtime.releases import (
    PackagedExecutionBindingV1,
    PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)

from .binding import ModalClientBinding
from .facade import EXACT_MODAL_SDK_VERSION


MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA = "synaptic-modal-packaged-runtime-facts/v1"
EXACT_ARTIFACT_ROLES = frozenset({
    "workload_record", "training_lineage", "training_metrics", "final_model",
    "tokenizer",
})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_IDS = {
    "app_id": re.compile(r"^ap-[A-Za-z0-9]{1,64}$"),
    "function_id": re.compile(r"^fu-[A-Za-z0-9]{1,64}$"),
    "self_check_function_id": re.compile(r"^fu-[A-Za-z0-9]{1,64}$"),
    "image_id": re.compile(r"^im-[A-Za-z0-9]{1,64}$"),
    "control_volume_id": re.compile(r"^vo-[A-Za-z0-9]{1,64}$"),
    "artifact_volume_id": re.compile(r"^vo-[A-Za-z0-9]{1,64}$"),
}


def _provider_id(value: str, name: str) -> str:
    value = safe_ref(value, name)
    if _PROVIDER_IDS[name].fullmatch(value) is None:
        raise ValueError(f"{name} is not a supported Modal identifier")
    return value


@dataclass(frozen=True, slots=True)
class ModalPackagedRuntimeFactsV1:
    """Authenticated current-state facts for one private packaged deployment."""

    account_ref: str
    workspace_ref: str
    environment_ref: str
    client_ref: str
    sdk_version: str
    app_name: str
    app_id: str
    deployment_generation: int
    function_name: str
    function_id: str
    self_check_function_name: str
    self_check_function_id: str
    deployment_spec_digest: str
    image_id: str
    image_digest: str
    package_digest: str
    installed_distributions_digest: str
    worker_entrypoint: str
    worker_closure_digest: str
    control_volume_id: str
    artifact_volume_id: str
    schema_version: str = MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA:
            raise ValueError("unsupported Modal packaged runtime facts")
        for name in (
            "account_ref", "workspace_ref", "environment_ref", "client_ref",
            "sdk_version", "app_name", "function_name",
            "self_check_function_name", "worker_entrypoint",
        ):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        if self.sdk_version != EXACT_MODAL_SDK_VERSION:
            raise ValueError("Modal SDK version differs from the packaged adapter")
        for name in _PROVIDER_IDS:
            object.__setattr__(self, name, _provider_id(getattr(self, name), name))
        if self.control_volume_id == self.artifact_volume_id:
            raise ValueError("Modal packaged deployment volumes must differ")
        if self.function_name == self.self_check_function_name \
                or self.function_id == self.self_check_function_id:
            raise ValueError("Modal packaged deployment requires two distinct functions")
        if type(self.deployment_generation) is not int or self.deployment_generation < 1:
            raise ValueError("deployment_generation must be a positive integer")
        for name in (
            "deployment_spec_digest", "image_digest", "package_digest", "installed_distributions_digest",
            "worker_closure_digest",
        ):
            object.__setattr__(self, name, digest_text(getattr(self, name), name))

    @property
    def client_binding(self) -> ModalClientBinding:
        return ModalClientBinding(
            self.account_ref, self.workspace_ref, self.environment_ref,
            self.client_ref, self.sdk_version,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": {
                "account_ref": self.account_ref,
                "workspace_ref": self.workspace_ref,
                "environment_ref": self.environment_ref,
                "client_ref": self.client_ref,
                "sdk_version": self.sdk_version,
            },
            "deployment": {
                "app_name": self.app_name,
                "app_id": self.app_id,
                "generation": self.deployment_generation,
                "deployment_spec_digest": self.deployment_spec_digest,
                "functions": [
                    {
                        "role": "training", "name": self.function_name,
                        "function_id": self.function_id, "private": True,
                    },
                    {
                        "role": "self_check", "name": self.self_check_function_name,
                        "function_id": self.self_check_function_id, "private": True,
                    },
                ],
                "classes": [],
                "private": True,
            },
            "runtime": {
                "image_id": self.image_id,
                "image_digest": self.image_digest,
                "package_digest": self.package_digest,
                "installed_distributions_digest": self.installed_distributions_digest,
                "worker_entrypoint": self.worker_entrypoint,
                "worker_closure_digest": self.worker_closure_digest,
            },
            "volumes": {
                "control_id": self.control_volume_id,
                "artifact_id": self.artifact_volume_id,
            },
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def facts_digest(self) -> str:
        return domain_digest(MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA, self.canonical_bytes)

    def validate_release(self, release: PackagedTrainingRuntimeReleaseV1) -> None:
        if type(release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("exact packaged runtime release required")
        if (
            self.image_digest != release.image_digest
            or self.package_digest != release.package_digest
            or self.installed_distributions_digest
            != release.installed_distributions_digest
            or self.worker_entrypoint != release.worker_entrypoint
            or self.worker_closure_digest != release.worker_closure_digest
        ):
            raise ValueError("Modal deployment facts differ from the runtime release")

    def build_provider_binding(
        self, release: PackagedTrainingRuntimeReleaseV1,
    ) -> ProviderRuntimeBindingV1:
        self.validate_release(release)
        return ProviderRuntimeBindingV1.build(
            provider_ref="modal",
            runtime_release=release,
            provider_facts_schema=MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA,
            provider_facts_digest=self.facts_digest,
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ModalPackagedRuntimeFactsV1":
        if type(value) is not dict or set(value) != {
            "schema_version", "scope", "deployment", "runtime", "volumes",
        }:
            raise ValueError("Modal packaged runtime facts have unknown fields")
        scope, deployment = value["scope"], value["deployment"]
        runtime, volumes = value["runtime"], value["volumes"]
        if type(scope) is not dict or set(scope) != {
            "account_ref", "workspace_ref", "environment_ref", "client_ref", "sdk_version",
        }:
            raise ValueError("Modal packaged scope is invalid")
        if type(deployment) is not dict or set(deployment) != {
            "app_name", "app_id", "generation", "deployment_spec_digest",
            "functions", "classes", "private",
        } or deployment.get("private") is not True or deployment.get("classes") != []:
            raise ValueError("Modal packaged deployment is not exactly private")
        functions = deployment.get("functions")
        if type(functions) is not list or len(functions) != 2:
            raise ValueError("Modal packaged deployment function layout is invalid")
        for item, role in zip(functions, ("training", "self_check"), strict=True):
            if type(item) is not dict or set(item) != {
                "role", "name", "function_id", "private",
            } or item.get("role") != role or item.get("private") is not True:
                raise ValueError("Modal packaged deployment function layout is invalid")
        if type(runtime) is not dict or set(runtime) != {
            "image_id", "image_digest", "package_digest",
            "installed_distributions_digest", "worker_entrypoint", "worker_closure_digest",
        }:
            raise ValueError("Modal packaged runtime identity is invalid")
        if type(volumes) is not dict or set(volumes) != {"control_id", "artifact_id"}:
            raise ValueError("Modal packaged volume identity is invalid")
        return cls(
            **scope,
            app_name=deployment["app_name"], app_id=deployment["app_id"],
            deployment_generation=deployment["generation"],
            function_name=functions[0]["name"], function_id=functions[0]["function_id"],
            self_check_function_name=functions[1]["name"],
            self_check_function_id=functions[1]["function_id"],
            deployment_spec_digest=deployment["deployment_spec_digest"],
            **runtime,
            control_volume_id=volumes["control_id"],
            artifact_volume_id=volumes["artifact_id"],
            schema_version=value["schema_version"],
        )  # type: ignore[arg-type]

    @classmethod
    def from_release_deployment(
        cls, value: object,
    ) -> "ModalPackagedRuntimeFactsV1":
        """Project acknowledged release facts into the training adapter binding."""
        from .runtime_release_deployment import ModalRuntimeReleaseDeploymentFactsV1

        if type(value) is not ModalRuntimeReleaseDeploymentFactsV1:
            raise TypeError("exact acknowledged Modal release facts required")
        functions = {item.spec.role: item for item in value.functions}
        volumes = {item.spec.role: item for item in value.volumes}
        if set(functions) != {"training", "self_check"} \
                or not {"control", "artifacts"} <= set(volumes):
            raise ValueError("acknowledged Modal release layout is incomplete")
        training, self_check = functions["training"], functions["self_check"]
        binding = value.client_binding
        return cls(
            account_ref=binding.account_ref,
            workspace_ref=binding.workspace_ref,
            environment_ref=binding.environment_ref,
            client_ref=binding.client_ref,
            sdk_version=binding.sdk_version,
            app_name=value.app_name,
            app_id=value.app_id,
            deployment_generation=value.generation,
            function_name=training.spec.name,
            function_id=training.function_id,
            self_check_function_name=self_check.spec.name,
            self_check_function_id=self_check.function_id,
            deployment_spec_digest=value.deployment_spec_digest,
            image_id=value.image_id,
            image_digest=value.image_digest,
            package_digest=value.package_digest,
            installed_distributions_digest=value.installed_distributions_digest,
            worker_entrypoint=value.worker_entrypoint,
            worker_closure_digest=value.worker_closure_digest,
            control_volume_id=volumes["control"].volume_id,
            artifact_volume_id=volumes["artifacts"].volume_id,
        )

    @classmethod
    def parse(cls, payload: bytes) -> "ModalPackagedRuntimeFactsV1":
        document = parse_canonical_object(payload, name="Modal packaged runtime facts")
        result = cls.from_dict(document)
        if result.canonical_bytes != payload:
            raise ValueError("Modal packaged runtime facts are not canonical")
        return result


def parse_modal_packaged_runtime_facts(payload: bytes) -> ModalPackagedRuntimeFactsV1:
    return ModalPackagedRuntimeFactsV1.parse(payload)


@dataclass(frozen=True, slots=True)
class ModalPackagedCommandBinding:
    """Complete retained binding for one Foundation command."""

    command_bytes: bytes = field(repr=False)
    runtime_release_bytes: bytes = field(repr=False)
    provider_binding_bytes: bytes = field(repr=False)
    provider_facts_bytes: bytes = field(repr=False)
    execution_binding_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if any(type(value) is not bytes or not value for value in (
            self.command_bytes, self.runtime_release_bytes, self.provider_binding_bytes,
            self.provider_facts_bytes, self.execution_binding_bytes,
        )):
            raise TypeError("Modal packaged command binding requires exact bytes")
        command = parse_exact_command(self.command_bytes)
        release = PackagedTrainingRuntimeReleaseV1.from_json(
            self.runtime_release_bytes.decode("utf-8")
        )
        provider = ProviderRuntimeBindingV1.from_json(
            self.provider_binding_bytes.decode("utf-8")
        )
        facts = ModalPackagedRuntimeFactsV1.parse(self.provider_facts_bytes)
        execution = PackagedExecutionBindingV1.from_json(
            self.execution_binding_bytes.decode("utf-8")
        )
        execution.validate_bindings(release, provider)
        facts.validate_release(release)
        expected_provider = facts.build_provider_binding(release)
        preparation = command.preparation
        namespace = domain_digest(
            "synaptic-modal-namespace/v1",
            canonical_bytes({
                "workspace_ref": facts.workspace_ref,
                "environment_ref": facts.environment_ref,
            }),
        )
        if (
            provider != expected_provider
            or preparation.provider.provider_id != "modal"
            or preparation.scope.account_ref != facts.account_ref
            or preparation.scope.namespace_ref != namespace
            or preparation.run_id != execution.run_ref
            or preparation.source_digest != execution.binding_digest
            or preparation.workload_digest != execution.workload_digest
        ):
            raise ValueError("Modal packaged command has a cross-binding mismatch")

    @property
    def command(self):
        return parse_exact_command(self.command_bytes)

    @property
    def command_digest(self) -> str:
        return self.command.digest

    @property
    def runtime_release(self) -> PackagedTrainingRuntimeReleaseV1:
        return PackagedTrainingRuntimeReleaseV1.from_json(self.runtime_release_bytes.decode("utf-8"))

    @property
    def provider_binding(self) -> ProviderRuntimeBindingV1:
        return ProviderRuntimeBindingV1.from_json(self.provider_binding_bytes.decode("utf-8"))

    @property
    def provider_facts(self) -> ModalPackagedRuntimeFactsV1:
        return ModalPackagedRuntimeFactsV1.parse(self.provider_facts_bytes)

    @property
    def execution_binding(self) -> PackagedExecutionBindingV1:
        return PackagedExecutionBindingV1.from_json(self.execution_binding_bytes.decode("utf-8"))

    @property
    def authenticated_binding_digest(self) -> str:
        return domain_digest(
            "synaptic-modal-packaged-command-binding/v1",
            canonical_bytes({
                "command_sha256": hashlib.sha256(self.command_bytes).hexdigest(),
                "runtime_release_sha256": hashlib.sha256(self.runtime_release_bytes).hexdigest(),
                "provider_binding_sha256": hashlib.sha256(self.provider_binding_bytes).hexdigest(),
                "provider_facts_sha256": hashlib.sha256(self.provider_facts_bytes).hexdigest(),
                "execution_binding_sha256": hashlib.sha256(self.execution_binding_bytes).hexdigest(),
            }),
        )

    def reconstructed(self) -> "ModalPackagedCommandBinding":
        return type(self)(
            bytes(self.command_bytes), bytes(self.runtime_release_bytes),
            bytes(self.provider_binding_bytes), bytes(self.provider_facts_bytes),
            bytes(self.execution_binding_bytes),
        )


class ModalPackagedBindingCatalog(Protocol):
    def resolve(self, command_digest: str) -> ModalPackagedCommandBinding | None: ...


class CommittedModalPackagedBindingCatalog:
    """Injectable read boundary; the consumer remains the durable store owner."""

    def __init__(self, resolver: Callable[[str], ModalPackagedCommandBinding | None]):
        if not callable(resolver):
            raise TypeError("binding resolver must be callable")
        self._resolver = resolver

    def resolve(self, command_digest: str) -> ModalPackagedCommandBinding | None:
        digest_text(command_digest, "command_digest")
        value = self._resolver(command_digest)
        if value is None:
            return None
        if type(value) is not ModalPackagedCommandBinding:
            raise TypeError("catalog returned a non-packaged binding")
        rebuilt = value.reconstructed()
        if rebuilt != value or rebuilt.command_digest != command_digest:
            raise ValueError("catalog returned a substituted packaged binding")
        return rebuilt


__all__ = [
    "EXACT_ARTIFACT_ROLES",
    "MODAL_PACKAGED_RUNTIME_FACTS_SCHEMA",
    "CommittedModalPackagedBindingCatalog",
    "ModalPackagedBindingCatalog",
    "ModalPackagedCommandBinding",
    "ModalPackagedRuntimeFactsV1",
    "parse_modal_packaged_runtime_facts",
]
