"""Strict declarative Modal provider profile and packaged runtime lock."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

from ...contracts import digest, safe_ref
from .deployment_identity import validate_modal_function_identity


_REGISTRY = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")


def _closed(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} contains missing or unknown fields")
    return dict(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(member) for key, member in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(member) for member in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_thaw(member) for member in value]
    return value


@dataclass(frozen=True, slots=True)
class ModalSecretProfileV1:
    name: str
    required_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", safe_ref(self.name, "secret name"))
        keys = tuple(self.required_keys)
        if not keys or len(keys) != len(set(keys)) or any(not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key) for key in keys):
            raise ValueError("secret required_keys must be unique environment names")
        object.__setattr__(self, "required_keys", keys)


@dataclass(frozen=True, slots=True)
class ModalProviderProfileV1:
    profile: str
    app_name: str
    function_name: str
    deployment_ref: str
    runtime_lock_ref: str
    control_volume_ref: str
    artifact_volume_ref: str
    secrets: tuple[ModalSecretProfileV1, ...]

    def __post_init__(self) -> None:
        for name in ("profile", "app_name", "control_volume_ref", "artifact_volume_ref"):
            object.__setattr__(self, name, safe_ref(getattr(self, name), name))
        deployment_ref, function_name = validate_modal_function_identity(
            self.deployment_ref, self.function_name
        )
        object.__setattr__(self, "deployment_ref", deployment_ref)
        object.__setattr__(self, "function_name", function_name)
        if self.app_name != "synaptic-training-v1":
            raise ValueError("Modal v1 application name is fixed")
        if self.runtime_lock_ref != "engine://tuner/execution/providers/modal/modal-runtime-v1.lock.json":
            raise ValueError("Modal runtime lock reference is unsupported")
        if self.control_volume_ref == self.artifact_volume_ref:
            raise ValueError("Modal control and artifact volume references must differ")
        secrets = tuple(self.secrets)
        if not secrets or any(type(secret) is not ModalSecretProfileV1 for secret in secrets):
            raise ValueError("at least one canonical Modal secret profile is required")
        if len({secret.name for secret in secrets}) != len(secrets):
            raise ValueError("Modal secret profile names must be unique")
        object.__setattr__(self, "secrets", secrets)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ModalProviderProfileV1":
        root = _closed(value, {"schema_version", "profile", "deployment", "runtime_lock", "volumes", "secrets"}, "Modal provider profile")
        if root["schema_version"] != "synaptic-modal-provider/v1":
            raise ValueError("unsupported Modal provider profile schema")
        deployment = _closed(root["deployment"], {"app_name", "function_name", "deployment_ref"}, "Modal deployment")
        volumes = _closed(root["volumes"], {"control_ref", "artifact_ref"}, "Modal volumes")
        raw_secrets = root["secrets"]
        if not isinstance(raw_secrets, list) or not 1 <= len(raw_secrets) <= 16:
            raise ValueError("Modal secrets must be a bounded list")
        secrets = []
        for raw in raw_secrets:
            item = _closed(raw, {"provider", "name", "required_keys"}, "Modal secret")
            if item["provider"] != "modal" or not isinstance(item["required_keys"], list):
                raise ValueError("Modal secret provider or keys are invalid")
            secrets.append(ModalSecretProfileV1(item["name"], tuple(item["required_keys"])))
        return cls(
            root["profile"], deployment["app_name"], deployment["function_name"],
            deployment["deployment_ref"], root["runtime_lock"],
            volumes["control_ref"], volumes["artifact_ref"], tuple(secrets),
        )

    @property
    def secret_requirements_digest(self) -> str:
        value = {
            "schema_version": "synaptic-modal-secret-requirements/v1",
            "secrets": [
                {"name": item.name, "required_keys": list(item.required_keys)}
                for item in self.secrets
            ],
        }
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    def provider_runtime_requirements_digest(
        self,
        runtime_lock: "ModalRuntimeLockV1",
        *,
        runtime_environment: Mapping[str, str],
        accelerator: str = "A10",
        timeout_seconds: int = 3600,
        max_retries: int = 0,
    ) -> str:
        if type(runtime_lock) is not ModalRuntimeLockV1:
            raise TypeError("canonical Modal runtime lock is required")
        if accelerator != "A10" or type(max_retries) is not int or max_retries != 0:
            raise ValueError("Modal v1 requires A10 with retries disabled")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
            raise ValueError("Modal timeout must be a bounded exact integer")
        environment = dict(runtime_environment)
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("runtime environment must be a closed string map")
        value = {
            "schema_version": "synaptic-modal-provider-runtime-requirements/v1",
            "runtime_lock": runtime_lock.to_dict(),
            "deployment": {
                "app_name": self.app_name,
                "function_name": self.function_name,
                "deployment_ref": self.deployment_ref,
                "accelerator": accelerator,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
            },
            "volumes": {
                "control_ref": self.control_volume_ref,
                "control_mount": "/workspace/control",
                "artifact_ref": self.artifact_volume_ref,
                "artifact_mount": "/workspace/run",
            },
            "secrets": [
                {"name": item.name, "required_keys": list(item.required_keys)}
                for item in self.secrets
            ],
            "environment": environment,
        }
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ModalRuntimeLockV1:
    document: Mapping[str, object]

    def __post_init__(self) -> None:
        root = _closed(self.document, {"schema_version", "sdk_version", "registry_reference", "python", "locked_files", "ml_stack"}, "Modal runtime lock")
        if root["schema_version"] != "synaptic-modal-runtime-lock/v1" or root["sdk_version"] != "1.5.4":
            raise ValueError("unsupported Modal runtime lock")
        if not isinstance(root["registry_reference"], str) or _REGISTRY.fullmatch(root["registry_reference"]) is None:
            raise ValueError("Modal runtime image is not digest pinned")
        python = _closed(root["python"], {"implementation", "version", "executable", "executable_sha256"}, "Modal Python lock")
        if python["implementation"] != "cpython" or python["version"] != "3.11.14" or python["executable"] != "/opt/conda/bin/python3":
            raise ValueError("Modal Python runtime differs from v1")
        digest(python["executable_sha256"], "python executable digest")
        locked = _closed(
            root["locked_files"],
            {
                "dependency_lock", "deployment_wrapper", "modal_mounted_io",
                "modal_runtime", "modal_remote", "modal_producer", "sft_runtime",
            },
            "Modal locked files",
        )
        for member in locked.values():
            item = _closed(member, {"path", "sha256"}, "Modal locked file")
            safe_ref(item["path"], "locked file path")
            digest(item["sha256"], "locked file digest")
        _closed(root["ml_stack"], {"torch", "transformers", "trl"}, "Modal ML stack")
        object.__setattr__(self, "document", _freeze(root))

    @classmethod
    def packaged(cls) -> "ModalRuntimeLockV1":
        resource = files("tuner.execution.providers.modal").joinpath("modal-runtime-v1.lock.json")
        return cls(json.loads(resource.read_text(encoding="utf-8")))

    @property
    def registry_reference(self) -> str:
        return self.document["registry_reference"]

    @property
    def sdk_version(self) -> str:
        return self.document["sdk_version"]

    @property
    def image_digest(self) -> str:
        return self.registry_reference.rsplit("@sha256:", 1)[1]

    @property
    def python_implementation(self) -> str:
        return self.document["python"]["implementation"]

    @property
    def python_version(self) -> str:
        return self.document["python"]["version"]

    @property
    def python_executable(self) -> str:
        return self.document["python"]["executable"]

    @property
    def python_executable_digest(self) -> str:
        return self.document["python"]["executable_sha256"]

    def to_dict(self) -> dict[str, object]:
        return _thaw(self.document)

    def locked_digest(self, name: str) -> str:
        return self.document["locked_files"][name]["sha256"]

    def validate_selection(self, selection: object) -> None:
        expected = {
            "sdk_version": self.sdk_version,
            "image_digest": self.image_digest,
            "dependency_lock_digest": self.locked_digest("dependency_lock"),
            "wrapper_digest": self.locked_digest("deployment_wrapper"),
            "runtime_digest": self.locked_digest("sft_runtime"),
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "python_executable_digest": self.python_executable_digest,
        }
        if any(getattr(selection, name, None) != value for name, value in expected.items()):
            raise ValueError("Modal deployment differs from the packaged runtime lock")


__all__ = ["ModalProviderProfileV1", "ModalRuntimeLockV1", "ModalSecretProfileV1"]
