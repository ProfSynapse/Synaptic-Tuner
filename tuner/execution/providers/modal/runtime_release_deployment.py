"""Protected release-time deployment for the packaged Modal runtime.

This module owns provider mechanics for installing an already-built runtime
release.  It never resolves packages, uploads source, invokes a worker, or
grants training authority.  A caller must retain a durable one-use release
claim before calling :meth:`ModalRuntimeReleaseDeployer.deploy_once`.

Modal's current deployment readback cannot independently prove the relation
between a floating Function and its Image.  Consequently the facts below keep
the IDs acknowledged by the successful ``App.deploy`` call distinct from the
bracketed current-state generation/layout observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import PurePosixPath
import re
import tempfile
import threading
from typing import Callable, Mapping, Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    exact_fields,
    exact_integer,
    parse_canonical_object,
    safe_ref,
)
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1

from .binding import ModalClientBinding
from .facade import EXACT_MODAL_SDK_VERSION, MODAL_VOLUME_V1


MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_SCHEMA = (
    "synaptic-modal-runtime-release-deployment-plan/v1"
)
MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_SCHEMA = (
    "synaptic-modal-runtime-release-deployment-facts/v1"
)
MODAL_RUNTIME_RELEASE_DEPLOYMENT_OBSERVATION_SCHEMA = (
    "synaptic-modal-runtime-release-deployment-observation/v1"
)
EXACT_SELF_CHECK_MODULE = "tuner.runtime.runtime_release_modal_self_check"
EXACT_SELF_CHECK_QUALNAME = "run_runtime_release_self_check"
EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS = (
    "SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY",
)
EXACT_QUALIFICATION_KEY_REF = "modal-runtime-release-qualification-hmac-v1"

_PROVIDER_ID = {
    "app_id": re.compile(r"^ap-[A-Za-z0-9]{1,64}$"),
    "function_id": re.compile(r"^fu-[A-Za-z0-9]{1,64}$"),
    "image_id": re.compile(r"^im-[A-Za-z0-9]{1,64}$"),
    "volume_id": re.compile(r"^vo-[A-Za-z0-9]{1,64}$"),
    "secret_id": re.compile(r"^st-[A-Za-z0-9]{1,64}$"),
}
_PYTHON_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_DEPLOY_CWD_LOCK = threading.Lock()


class ModalRuntimeReleaseDeploymentError(RuntimeError):
    """Closed release-deployment failure without provider response detail."""


def _provider_id(value: object, kind: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    if type(value) is not str or _PROVIDER_ID[kind].fullmatch(value) is None:
        raise ValueError(f"{kind} is not a supported Modal identifier")
    return value


def _absolute_mount(value: object) -> str:
    if type(value) is not str or not value.startswith("/"):
        raise ValueError("Modal mount path must be an absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or value == "/" or ".." in path.parts:
        raise ValueError("Modal mount path is not canonical")
    return value


def _absolute_executable(value: object) -> str:
    if type(value) is not str or not value.startswith("/"):
        raise ValueError("Python executable must be an absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or value == "/" or ".." in path.parts:
        raise ValueError("Python executable path is not canonical")
    return value


def _tuple_of_refs(values: object, name: str, *, minimum: int = 0) -> tuple[str, ...]:
    if type(values) not in (tuple, list):
        raise TypeError(f"{name} must be an exact sequence")
    result = tuple(safe_ref(item, name) for item in values)
    if len(result) < minimum or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique required references")
    return result


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseFunctionSpecV1:
    role: str
    name: str
    module: str
    qualname: str
    volume_roles: tuple[str, ...]
    secret_names: tuple[str, ...]
    cpu_milli: int
    memory_mib: int
    timeout_seconds: int
    gpu: str | None
    block_network: bool
    retries: int = 0
    restrict_modal_access: bool = True
    single_use_containers: bool = True
    serialized: bool = False
    include_source: bool = False

    def __post_init__(self) -> None:
        if self.role not in {"training", "self_check"}:
            raise ValueError("Modal release function role is unsupported")
        object.__setattr__(self, "name", safe_ref(self.name, "function_name"))
        for field in ("module", "qualname"):
            value = getattr(self, field)
            if type(value) is not str or _PYTHON_NAME.fullmatch(value) is None:
                raise ValueError(f"{field} must identify an installed global callable")
        object.__setattr__(
            self, "volume_roles", _tuple_of_refs(self.volume_roles, "volume_role", minimum=2)
        )
        object.__setattr__(
            self, "secret_names", _tuple_of_refs(self.secret_names, "secret_name")
        )
        exact_integer(self.cpu_milli, "cpu_milli", minimum=1)
        exact_integer(self.memory_mib, "memory_mib", minimum=128)
        exact_integer(self.timeout_seconds, "timeout_seconds", minimum=1)
        if self.gpu is not None:
            object.__setattr__(self, "gpu", safe_ref(self.gpu, "gpu"))
        if self.role == "self_check" and self.gpu is not None:
            raise ValueError("Modal release self-check must be CPU-only")
        expected_modal_restriction = self.role != "self_check"
        if self.retries != 0 \
                or self.restrict_modal_access is not expected_modal_restriction \
                or self.single_use_containers is not True \
                or self.serialized is not False or self.include_source is not False:
            raise ValueError("Modal release function safety policy is not exact")
        if type(self.block_network) is not bool:
            raise TypeError("block_network must be an exact boolean")
        if self.role == "self_check" and self.block_network is not True:
            raise ValueError("Modal release self-check must block network access")

    @property
    def entrypoint(self) -> str:
        return f"{self.module}:{self.qualname}"

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "name": self.name,
            "entrypoint": {"module": self.module, "qualname": self.qualname},
            "volume_roles": list(self.volume_roles),
            "secret_names": list(self.secret_names),
            "resources": {
                "cpu_milli": self.cpu_milli,
                "memory_mib": self.memory_mib,
                "timeout_seconds": self.timeout_seconds,
                "gpu": self.gpu,
            },
            "policy": {
                "block_network": self.block_network,
                "retries": self.retries,
                "restrict_modal_access": self.restrict_modal_access,
                "single_use_containers": self.single_use_containers,
                "serialized": self.serialized,
                "include_source": self.include_source,
                "private": True,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> "ModalRuntimeReleaseFunctionSpecV1":
        exact_fields(value, frozenset({
            "role", "name", "entrypoint", "volume_roles", "secret_names",
            "resources", "policy",
        }), "Modal release function")
        assert isinstance(value, Mapping)
        entrypoint = value["entrypoint"]
        resources = value["resources"]
        policy = value["policy"]
        exact_fields(entrypoint, frozenset({"module", "qualname"}), "function entrypoint")
        exact_fields(
            resources,
            frozenset({"cpu_milli", "memory_mib", "timeout_seconds", "gpu"}),
            "function resources",
        )
        exact_fields(
            policy,
            frozenset({
                "block_network", "retries", "restrict_modal_access",
                "single_use_containers", "serialized", "include_source", "private",
            }),
            "function policy",
        )
        assert isinstance(entrypoint, Mapping) and isinstance(resources, Mapping) \
            and isinstance(policy, Mapping)
        if policy["private"] is not True:
            raise ValueError("Modal release function must be private")
        return cls(
            role=value["role"], name=value["name"],
            module=entrypoint["module"], qualname=entrypoint["qualname"],
            volume_roles=value["volume_roles"], secret_names=value["secret_names"],
            cpu_milli=resources["cpu_milli"], memory_mib=resources["memory_mib"],
            timeout_seconds=resources["timeout_seconds"], gpu=resources["gpu"],
            block_network=policy["block_network"], retries=policy["retries"],
            restrict_modal_access=policy["restrict_modal_access"],
            single_use_containers=policy["single_use_containers"],
            serialized=policy["serialized"], include_source=policy["include_source"],
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseVolumeSpecV1:
    role: str
    name: str
    mount_path: str
    version: int = MODAL_VOLUME_V1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", safe_ref(self.role, "volume_role"))
        object.__setattr__(self, "name", safe_ref(self.name, "volume_name"))
        object.__setattr__(self, "mount_path", _absolute_mount(self.mount_path))
        if self.version != MODAL_VOLUME_V1:
            raise ValueError("Modal release Volume version must be v1")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role, "name": self.name,
            "mount_path": self.mount_path, "version": self.version,
            "create_if_missing": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ModalRuntimeReleaseVolumeSpecV1":
        exact_fields(
            value,
            frozenset({"role", "name", "mount_path", "version", "create_if_missing"}),
            "Modal release volume",
        )
        assert isinstance(value, Mapping)
        if value["create_if_missing"] is not False:
            raise ValueError("Modal release Volumes must already exist")
        return cls(value["role"], value["name"], value["mount_path"], value["version"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseSecretSpecV1:
    name: str
    required_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", safe_ref(self.name, "secret_name"))
        object.__setattr__(
            self, "required_keys", _tuple_of_refs(self.required_keys, "required_key", minimum=1)
        )

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "required_keys": list(self.required_keys)}

    @classmethod
    def from_dict(cls, value: object) -> "ModalRuntimeReleaseSecretSpecV1":
        exact_fields(value, frozenset({"name", "required_keys"}), "Modal release secret")
        assert isinstance(value, Mapping)
        return cls(value["name"], value["required_keys"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseDeploymentPlanV1:
    release: PackagedTrainingRuntimeReleaseV1
    app_name: str
    environment_name: str
    functions: tuple[ModalRuntimeReleaseFunctionSpecV1, ...]
    volumes: tuple[ModalRuntimeReleaseVolumeSpecV1, ...]
    secrets: tuple[ModalRuntimeReleaseSecretSpecV1, ...]
    strategy: str = "rolling"
    sdk_version: str = EXACT_MODAL_SDK_VERSION
    schema_version: str = MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_SCHEMA:
            raise ValueError("unsupported Modal runtime release deployment plan")
        if type(self.release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("exact packaged runtime release required")
        if self.sdk_version != EXACT_MODAL_SDK_VERSION or self.strategy != "rolling":
            raise ValueError("Modal release deployment policy is unsupported")
        object.__setattr__(self, "app_name", safe_ref(self.app_name, "app_name"))
        object.__setattr__(
            self, "environment_name", safe_ref(self.environment_name, "environment_name")
        )
        if type(self.functions) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseFunctionSpecV1 for item in self.functions):
            raise TypeError("exact Modal release function specs required")
        functions = tuple(self.functions)
        if tuple(item.role for item in functions) != ("training", "self_check") \
                or len({item.name for item in functions}) != 2 \
                or len({item.entrypoint for item in functions}) != 2:
            raise ValueError("Modal release requires exact training/self-check private layout")
        self_check = functions[1]
        if (
            self_check.module != EXACT_SELF_CHECK_MODULE
            or self_check.qualname != EXACT_SELF_CHECK_QUALNAME
            or self_check.volume_roles != ("control", "artifacts")
            or len(self_check.secret_names) != 1
            or self_check.cpu_milli != 1000
            or self_check.memory_mib != 512
            or self_check.timeout_seconds != 120
            or self_check.gpu is not None
            or self_check.block_network is not True
            or self_check.restrict_modal_access is not False
        ):
            raise ValueError("Modal release self-check specification is not exact")
        object.__setattr__(self, "functions", functions)
        if type(self.volumes) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseVolumeSpecV1 for item in self.volumes):
            raise TypeError("exact Modal release Volume specs required")
        volumes = tuple(self.volumes)
        roles = {item.role for item in volumes}
        if not {"control", "artifacts"} <= roles \
                or len(roles) != len(volumes) \
                or len({item.name for item in volumes}) != len(volumes) \
                or len({item.mount_path for item in volumes}) != len(volumes):
            raise ValueError("Modal release Volume layout is invalid")
        object.__setattr__(self, "volumes", volumes)
        if type(self.secrets) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseSecretSpecV1 for item in self.secrets):
            raise TypeError("exact Modal release Secret specs required")
        secrets = tuple(self.secrets)
        if len({item.name for item in secrets}) != len(secrets):
            raise ValueError("Modal release Secret names must be unique")
        object.__setattr__(self, "secrets", secrets)
        qualification_secret = next(
            (item for item in secrets if item.name == self_check.secret_names[0]), None
        )
        if qualification_secret is None \
                or qualification_secret.required_keys \
                != EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS:
            raise ValueError("Modal release qualification Secret policy is not exact")
        for function in functions:
            if not set(function.volume_roles) <= roles \
                    or not set(function.secret_names) <= {item.name for item in secrets}:
                raise ValueError("Modal release function references an undeclared resource")
        if self.release.image_ref.rsplit("@sha256:", 1)[-1] != self.release.image_digest:
            raise ValueError("runtime release image reference is not digest-bound")

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sdk_version": self.sdk_version,
            "app_name": self.app_name,
            "environment_name": self.environment_name,
            "strategy": self.strategy,
            "runtime_release": self.release.to_dict(),
            "functions": [item.to_dict() for item in self.functions],
            "volumes": [item.to_dict() for item in self.volumes],
            "secrets": [item.to_dict() for item in self.secrets],
        }

    @property
    def deployment_spec_digest(self) -> str:
        return domain_digest(self.schema_version, canonical_bytes(self._unsigned_dict()))

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "deployment_spec_digest": self.deployment_spec_digest}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ModalRuntimeReleaseDeploymentPlanV1":
        exact_fields(value, frozenset({
            "schema_version", "sdk_version", "app_name", "environment_name",
            "strategy", "runtime_release", "functions", "volumes", "secrets",
            "deployment_spec_digest",
        }), "Modal runtime release deployment plan")
        assert isinstance(value, Mapping)
        if type(value["functions"]) is not list or type(value["volumes"]) is not list \
                or type(value["secrets"]) is not list:
            raise TypeError("Modal release plan inventories must be exact arrays")
        result = cls(
            release=PackagedTrainingRuntimeReleaseV1.from_dict(value["runtime_release"]),  # type: ignore[arg-type]
            app_name=value["app_name"], environment_name=value["environment_name"],
            functions=tuple(ModalRuntimeReleaseFunctionSpecV1.from_dict(item) for item in value["functions"]),
            volumes=tuple(ModalRuntimeReleaseVolumeSpecV1.from_dict(item) for item in value["volumes"]),
            secrets=tuple(ModalRuntimeReleaseSecretSpecV1.from_dict(item) for item in value["secrets"]),
            strategy=value["strategy"], sdk_version=value["sdk_version"],
            schema_version=value["schema_version"],
        )  # type: ignore[arg-type]
        if value["deployment_spec_digest"] != result.deployment_spec_digest:
            raise ValueError("deployment_spec_digest does not bind the Modal release plan")
        return result

    @classmethod
    def parse(cls, payload: bytes) -> "ModalRuntimeReleaseDeploymentPlanV1":
        result = cls.from_dict(parse_canonical_object(payload, name="Modal release plan"))
        if result.canonical_bytes != payload:
            raise ValueError("Modal release plan is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseDeploymentObservationV1:
    app_name: str
    environment_name: str
    deployed: bool
    app_id: str
    previous_app_id: str
    generation: int
    function_ids: tuple[tuple[str, str], ...]
    class_ids: tuple[str, ...] = ()
    schema_version: str = MODAL_RUNTIME_RELEASE_DEPLOYMENT_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODAL_RUNTIME_RELEASE_DEPLOYMENT_OBSERVATION_SCHEMA:
            raise ValueError("unsupported Modal release deployment observation")
        object.__setattr__(self, "app_name", safe_ref(self.app_name, "app_name"))
        object.__setattr__(
            self, "environment_name", safe_ref(self.environment_name, "environment_name")
        )
        if type(self.deployed) is not bool:
            raise TypeError("deployed must be an exact boolean")
        app_id = _provider_id(self.app_id, "app_id", optional=True)
        previous = _provider_id(self.previous_app_id, "app_id", optional=True)
        object.__setattr__(self, "app_id", app_id)
        object.__setattr__(self, "previous_app_id", previous)
        exact_integer(self.generation, "generation", minimum=1)
        if self.deployed and not app_id or not self.deployed and app_id:
            raise ValueError("Modal release deployment state is inconsistent")
        if not self.deployed and not previous:
            raise ValueError("stopped Modal deployment lacks prior identity")
        if type(self.function_ids) not in (tuple, list):
            raise TypeError("function_ids must be an exact sequence")
        functions: list[tuple[str, str]] = []
        for item in self.function_ids:
            if type(item) not in (tuple, list) or len(item) != 2:
                raise ValueError("Modal release function layout is invalid")
            functions.append((safe_ref(item[0], "function_name"), _provider_id(item[1], "function_id")))
        if functions != sorted(functions) or len(functions) != len({item[0] for item in functions}):
            raise ValueError("Modal release function layout must be sorted and unique")
        object.__setattr__(self, "function_ids", tuple(functions))
        object.__setattr__(self, "class_ids", _tuple_of_refs(self.class_ids, "class_id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "app_name": self.app_name,
            "environment_name": self.environment_name,
            "deployed": self.deployed,
            "app_id": self.app_id,
            "previous_app_id": self.previous_app_id,
            "generation": self.generation,
            "function_ids": [list(item) for item in self.function_ids],
            "class_ids": list(self.class_ids),
        }


class ModalRuntimeReleaseDeploymentReadPort(Protocol):
    def observe(
        self, *, client: object, app_name: str, environment_name: str,
    ) -> ModalRuntimeReleaseDeploymentObservationV1 | None: ...


class ExplicitModal154ReleaseDeploymentReader:
    """Pinned internal generation/layout readback using one explicit client.

    Modal 1.5.4 has no public API for this complete observation.  The pinned
    protobuf read is bracketed by an identical scoped app lookup and reads no
    function source, arguments, logs, Secret values, or unrelated metadata.
    """

    __slots__ = ("_sdk",)

    def __init__(self, *, sdk: object) -> None:
        if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
            raise ValueError("Modal SDK version differs from release readback")
        self._sdk = sdk

    @staticmethod
    async def _read(client: object, app_name: str, environment_name: str):
        from modal.exception import NotFoundError
        from modal_proto import api_pb2

        request = api_pb2.AppGetByDeploymentNameRequest(
            name=app_name, environment_name=environment_name,
        )
        try:
            before = await client.stub.AppGetByDeploymentName(request)
        except NotFoundError:
            return None
        state = before.lifecycle.app_state
        if before.environment_name != environment_name \
                or state not in (api_pb2.APP_STATE_DEPLOYED, api_pb2.APP_STATE_STOPPED) \
                or type(before.lifecycle.version) is not int \
                or before.lifecycle.version < 1:
            raise ValueError
        deployed = state == api_pb2.APP_STATE_DEPLOYED
        if deployed:
            if not before.app_id:
                raise ValueError
            app_id = _provider_id(before.app_id, "app_id")
            previous = _provider_id(before.previous_app_id, "app_id", optional=True)
            layout_id = app_id
        else:
            if before.app_id or not before.previous_app_id:
                raise ValueError
            app_id = ""
            previous = _provider_id(before.previous_app_id, "app_id")
            layout_id = previous
        layout_response = await client.stub.AppGetLayout(
            api_pb2.AppGetLayoutRequest(app_id=layout_id)
        )
        after = await client.stub.AppGetByDeploymentName(request)
        if after != before:
            raise ValueError
        layout = layout_response.app_layout
        if len(layout.objects) > 4096 or len(layout.function_ids) > 256 \
                or len(layout.class_ids) > 256:
            raise ValueError
        functions = tuple(sorted(
            (safe_ref(name, "function_name"), _provider_id(identity, "function_id"))
            for name, identity in layout.function_ids.items()
        ))
        classes = tuple(sorted(safe_ref(name, "class_name") for name in layout.class_ids))
        function_ids = [identity for _, identity in functions]
        class_object_ids = [
            safe_ref(identity, "class_id") for identity in layout.class_ids.values()
        ]
        if len(function_ids) != len(set(function_ids)) \
                or len(class_object_ids) != len(set(class_object_ids)) \
                or set(function_ids).intersection(class_object_ids):
            raise ValueError
        return ModalRuntimeReleaseDeploymentObservationV1(
            app_name, environment_name, deployed, app_id, previous,
            before.lifecycle.version, functions, classes,
        )

    @classmethod
    async def _bounded_read(cls, client, app_name, environment_name):
        import asyncio

        return await asyncio.wait_for(
            cls._read(client, app_name, environment_name), timeout=30,
        )

    def observe(self, *, client: object, app_name: str, environment_name: str):
        if client is None:
            raise ModalRuntimeReleaseDeploymentError("modal_release_explicit_client_required")
        app_name = safe_ref(app_name, "app_name")
        environment_name = safe_ref(environment_name, "environment_name")
        failed = False
        try:
            from modal._utils.async_utils import synchronizer

            value = synchronizer.create_blocking(self._bounded_read)(
                client, app_name, environment_name,
            )
        except Exception:
            failed, value = True, None
        if failed:
            raise ModalRuntimeReleaseDeploymentError("modal_release_readback_failed")
        return value


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseFunctionFactV1:
    spec: ModalRuntimeReleaseFunctionSpecV1
    function_id: str

    def __post_init__(self) -> None:
        if type(self.spec) is not ModalRuntimeReleaseFunctionSpecV1:
            raise TypeError("exact release function spec required")
        object.__setattr__(self, "function_id", _provider_id(self.function_id, "function_id"))

    def to_dict(self) -> dict[str, object]:
        return {**self.spec.to_dict(), "function_id": self.function_id}


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseVolumeFactV1:
    spec: ModalRuntimeReleaseVolumeSpecV1
    volume_id: str

    def __post_init__(self) -> None:
        if type(self.spec) is not ModalRuntimeReleaseVolumeSpecV1:
            raise TypeError("exact release Volume spec required")
        object.__setattr__(self, "volume_id", _provider_id(self.volume_id, "volume_id"))

    def to_dict(self) -> dict[str, object]:
        return {**self.spec.to_dict(), "volume_id": self.volume_id}


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseSecretFactV1:
    spec: ModalRuntimeReleaseSecretSpecV1
    secret_id: str

    def __post_init__(self) -> None:
        if type(self.spec) is not ModalRuntimeReleaseSecretSpecV1:
            raise TypeError("exact release Secret spec required")
        object.__setattr__(self, "secret_id", _provider_id(self.secret_id, "secret_id"))

    def to_dict(self) -> dict[str, object]:
        return {**self.spec.to_dict(), "secret_id": self.secret_id}


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseDeploymentFactsV1:
    deployment_spec_digest: str
    runtime_release_digest: str
    client_binding: ModalClientBinding
    app_name: str
    app_id: str
    generation: int
    image_reference: str
    image_digest: str
    image_id: str
    functions: tuple[ModalRuntimeReleaseFunctionFactV1, ...]
    volumes: tuple[ModalRuntimeReleaseVolumeFactV1, ...]
    secrets: tuple[ModalRuntimeReleaseSecretFactV1, ...]
    python_implementation: str
    python_version: str
    python_executable: str
    python_executable_digest: str
    package_digest: str
    installed_distributions_digest: str
    worker_entrypoint: str
    worker_closure_digest: str
    current_state_version_pinned: bool = False
    function_image_link_readable: bool = False
    schema_version: str = MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_SCHEMA:
            raise ValueError("unsupported Modal runtime release deployment facts")
        for field in (
            "deployment_spec_digest", "runtime_release_digest", "image_digest",
            "python_executable_digest", "package_digest",
            "installed_distributions_digest", "worker_closure_digest",
        ):
            object.__setattr__(self, field, digest_text(getattr(self, field), field))
        if type(self.client_binding) is not ModalClientBinding \
                or self.client_binding.sdk_version != EXACT_MODAL_SDK_VERSION:
            raise TypeError("exact Modal client binding required")
        object.__setattr__(self, "app_name", safe_ref(self.app_name, "app_name"))
        object.__setattr__(self, "app_id", _provider_id(self.app_id, "app_id"))
        exact_integer(self.generation, "generation", minimum=1)
        object.__setattr__(self, "image_reference", safe_ref(self.image_reference, "image_reference"))
        object.__setattr__(self, "image_id", _provider_id(self.image_id, "image_id"))
        for field in ("python_implementation", "python_version", "worker_entrypoint"):
            object.__setattr__(self, field, safe_ref(getattr(self, field), field))
        object.__setattr__(
            self, "python_executable", _absolute_executable(self.python_executable)
        )
        if type(self.functions) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseFunctionFactV1 for item in self.functions):
            raise TypeError("exact Modal release function facts required")
        functions = tuple(self.functions)
        if tuple(item.spec.role for item in functions) != ("training", "self_check") \
                or len({item.function_id for item in functions}) != 2:
            raise ValueError("Modal release function facts are not exact")
        object.__setattr__(self, "functions", functions)
        if type(self.volumes) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseVolumeFactV1 for item in self.volumes):
            raise TypeError("exact Modal release Volume facts required")
        object.__setattr__(self, "volumes", tuple(self.volumes))
        if type(self.secrets) not in (tuple, list) \
                or any(type(item) is not ModalRuntimeReleaseSecretFactV1 for item in self.secrets):
            raise TypeError("exact Modal release Secret facts required")
        object.__setattr__(self, "secrets", tuple(self.secrets))
        if self.current_state_version_pinned is not False \
                or self.function_image_link_readable is not False:
            raise ValueError("Modal current-state readback limitations must remain explicit")

    def to_dict(self) -> dict[str, object]:
        binding = self.client_binding
        return {
            "schema_version": self.schema_version,
            "deployment_spec_digest": self.deployment_spec_digest,
            "runtime_release_digest": self.runtime_release_digest,
            "scope": {
                "account_ref": binding.account_ref,
                "workspace_ref": binding.workspace_ref,
                "environment_ref": binding.environment_ref,
                "client_ref": binding.client_ref,
                "sdk_version": binding.sdk_version,
            },
            "deployment": {
                "app_name": self.app_name, "app_id": self.app_id,
                "generation": self.generation, "private": True,
                "functions": [item.to_dict() for item in self.functions],
                "classes": [],
            },
            "runtime": {
                "image_reference": self.image_reference,
                "image_digest": self.image_digest,
                "image_id": self.image_id,
                "python": {
                    "implementation": self.python_implementation,
                    "version": self.python_version,
                    "executable": self.python_executable,
                    "executable_digest": self.python_executable_digest,
                },
                "package_digest": self.package_digest,
                "installed_distributions_digest": self.installed_distributions_digest,
                "worker_entrypoint": self.worker_entrypoint,
                "worker_closure_digest": self.worker_closure_digest,
            },
            "volumes": [item.to_dict() for item in self.volumes],
            "secrets": [item.to_dict() for item in self.secrets],
            "readback_limits": {
                "current_state_version_pinned": self.current_state_version_pinned,
                "function_image_link_readable": self.function_image_link_readable,
            },
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def facts_digest(self) -> str:
        return domain_digest(self.schema_version, self.canonical_bytes)

    def validate_plan(self, plan: ModalRuntimeReleaseDeploymentPlanV1) -> None:
        if type(plan) is not ModalRuntimeReleaseDeploymentPlanV1 \
                or self.deployment_spec_digest != plan.deployment_spec_digest \
                or self.runtime_release_digest != plan.release.manifest_digest \
                or self.app_name != plan.app_name \
                or self.client_binding.environment_ref != plan.environment_name \
                or self.image_reference != plan.release.image_ref \
                or self.image_digest != plan.release.image_digest \
                or tuple(item.spec for item in self.functions) != plan.functions \
                or tuple(item.spec for item in self.volumes) != plan.volumes \
                or tuple(item.spec for item in self.secrets) != plan.secrets:
            raise ValueError("Modal deployment facts differ from the release plan")

    @classmethod
    def from_dict(cls, value: object) -> "ModalRuntimeReleaseDeploymentFactsV1":
        exact_fields(value, frozenset({
            "schema_version", "deployment_spec_digest", "runtime_release_digest",
            "scope", "deployment", "runtime", "volumes", "secrets", "readback_limits",
        }), "Modal runtime release deployment facts")
        assert isinstance(value, Mapping)
        scope, deployment, runtime = value["scope"], value["deployment"], value["runtime"]
        limits = value["readback_limits"]
        exact_fields(scope, frozenset({
            "account_ref", "workspace_ref", "environment_ref", "client_ref", "sdk_version",
        }), "Modal release scope")
        exact_fields(deployment, frozenset({
            "app_name", "app_id", "generation", "private", "functions", "classes",
        }), "Modal release deployment")
        exact_fields(runtime, frozenset({
            "image_reference", "image_digest", "image_id", "python", "package_digest",
            "installed_distributions_digest", "worker_entrypoint", "worker_closure_digest",
        }), "Modal release runtime")
        exact_fields(limits, frozenset({
            "current_state_version_pinned", "function_image_link_readable",
        }), "Modal release readback limits")
        assert isinstance(scope, Mapping) and isinstance(deployment, Mapping) \
            and isinstance(runtime, Mapping) and isinstance(limits, Mapping)
        python = runtime["python"]
        exact_fields(python, frozenset({
            "implementation", "version", "executable", "executable_digest",
        }), "Modal release Python identity")
        assert isinstance(python, Mapping)
        if deployment["private"] is not True or deployment["classes"] != [] \
                or type(deployment["functions"]) is not list \
                or type(value["volumes"]) is not list or type(value["secrets"]) is not list:
            raise ValueError("Modal release facts do not describe the exact private layout")
        function_facts = []
        for item in deployment["functions"]:
            if type(item) is not dict or "function_id" not in item:
                raise ValueError("Modal release function fact is invalid")
            spec_value = dict(item)
            function_id = spec_value.pop("function_id")
            function_facts.append(ModalRuntimeReleaseFunctionFactV1(
                ModalRuntimeReleaseFunctionSpecV1.from_dict(spec_value), function_id,
            ))
        volume_facts = []
        for item in value["volumes"]:
            if type(item) is not dict or "volume_id" not in item:
                raise ValueError("Modal release Volume fact is invalid")
            spec_value = dict(item)
            volume_id = spec_value.pop("volume_id")
            volume_facts.append(ModalRuntimeReleaseVolumeFactV1(
                ModalRuntimeReleaseVolumeSpecV1.from_dict(spec_value), volume_id,
            ))
        secret_facts = []
        for item in value["secrets"]:
            if type(item) is not dict or "secret_id" not in item:
                raise ValueError("Modal release Secret fact is invalid")
            spec_value = dict(item)
            secret_id = spec_value.pop("secret_id")
            secret_facts.append(ModalRuntimeReleaseSecretFactV1(
                ModalRuntimeReleaseSecretSpecV1.from_dict(spec_value), secret_id,
            ))
        return cls(
            deployment_spec_digest=value["deployment_spec_digest"],
            runtime_release_digest=value["runtime_release_digest"],
            client_binding=ModalClientBinding(**scope),  # type: ignore[arg-type]
            app_name=deployment["app_name"], app_id=deployment["app_id"],
            generation=deployment["generation"],
            image_reference=runtime["image_reference"], image_digest=runtime["image_digest"],
            image_id=runtime["image_id"], functions=tuple(function_facts),
            volumes=tuple(volume_facts), secrets=tuple(secret_facts),
            python_implementation=python["implementation"], python_version=python["version"],
            python_executable=python["executable"],
            python_executable_digest=python["executable_digest"],
            package_digest=runtime["package_digest"],
            installed_distributions_digest=runtime["installed_distributions_digest"],
            worker_entrypoint=runtime["worker_entrypoint"],
            worker_closure_digest=runtime["worker_closure_digest"],
            current_state_version_pinned=limits["current_state_version_pinned"],
            function_image_link_readable=limits["function_image_link_readable"],
            schema_version=value["schema_version"],
        )  # type: ignore[arg-type]

    @classmethod
    def parse(cls, payload: bytes) -> "ModalRuntimeReleaseDeploymentFactsV1":
        result = cls.from_dict(parse_canonical_object(payload, name="Modal release facts"))
        if result.canonical_bytes != payload:
            raise ValueError("Modal release facts are not canonical")
        return result


class ModalRuntimeReleaseDeployer:
    """Construct and deploy one exact packaged runtime using an explicit client."""

    __slots__ = ("_sdk", "_client", "_binding", "_reader")

    def __init__(
        self, *, sdk: object, client: object, client_binding: ModalClientBinding,
        reader: ModalRuntimeReleaseDeploymentReadPort,
    ) -> None:
        if getattr(sdk, "__version__", None) != EXACT_MODAL_SDK_VERSION:
            raise ValueError("Modal SDK version differs from the release plan")
        if client is None or type(client_binding) is not ModalClientBinding:
            raise TypeError("explicit Modal client and exact binding required")
        if client_binding.sdk_version != EXACT_MODAL_SDK_VERSION \
                or not callable(getattr(reader, "observe", None)):
            raise TypeError("exact Modal deployment read port required")
        self._sdk, self._client, self._binding, self._reader = (
            sdk, client, client_binding, reader,
        )

    def _observe(self, plan: ModalRuntimeReleaseDeploymentPlanV1):
        failed = False
        try:
            value = self._reader.observe(
                client=self._client, app_name=plan.app_name,
                environment_name=plan.environment_name,
            )
        except Exception:
            failed, value = True, None
        if failed:
            raise ModalRuntimeReleaseDeploymentError("modal_release_observation_unavailable")
        if value is not None and type(value) is not ModalRuntimeReleaseDeploymentObservationV1:
            raise ModalRuntimeReleaseDeploymentError("modal_release_observation_invalid")
        return value

    def _observe_scope(self, environment_name: str) -> None:
        failed = False
        try:
            workspace = self._sdk.Workspace.from_context(client=self._client)
            workspace.hydrate(self._client)
            environment = self._sdk.Environment.from_name(
                environment_name, create_if_missing=False, client=self._client,
            )
            environment.hydrate(self._client)
            if getattr(workspace, "is_hydrated", False) is not True \
                    or getattr(environment, "is_hydrated", False) is not True \
                    or safe_ref(getattr(workspace, "name", None), "workspace_ref") \
                    != self._binding.workspace_ref \
                    or self._binding.account_ref != self._binding.workspace_ref \
                    or safe_ref(getattr(environment, "name", None), "environment_ref") \
                    != environment_name:
                raise ValueError
        except Exception:
            failed = True
        if failed:
            raise ModalRuntimeReleaseDeploymentError("modal_release_scope_unavailable")

    def observe(self, plan: ModalRuntimeReleaseDeploymentPlanV1):
        if type(plan) is not ModalRuntimeReleaseDeploymentPlanV1:
            raise TypeError("exact Modal release deployment plan required")
        if plan.environment_name != self._binding.environment_ref:
            raise ValueError("Modal release plan differs from the explicit client scope")
        self._observe_scope(plan.environment_name)
        return self._observe(plan)

    def inspect_resources(self, plan: ModalRuntimeReleaseDeploymentPlanV1):
        """Read exact existing resource identities without deployment mutation."""
        if type(plan) is not ModalRuntimeReleaseDeploymentPlanV1:
            raise TypeError("exact Modal release deployment plan required")
        if plan.environment_name != self._binding.environment_ref:
            raise ValueError("Modal release plan differs from the explicit client scope")
        self._observe_scope(plan.environment_name)
        volumes, secrets = self._hydrate_resources(plan)
        return (
            tuple(item[1] for item in volumes.values()),
            tuple(item[1] for item in secrets.values()),
        )

    def _hydrate_resources(self, plan: ModalRuntimeReleaseDeploymentPlanV1):
        volumes: dict[str, tuple[object, ModalRuntimeReleaseVolumeFactV1]] = {}
        secrets: dict[str, tuple[object, ModalRuntimeReleaseSecretFactV1]] = {}
        failed = False
        try:
            for spec in plan.volumes:
                value = self._sdk.Volume.from_name(
                    spec.name, environment_name=plan.environment_name,
                    create_if_missing=False, version=spec.version, client=self._client,
                )
                value.hydrate(self._client)
                if getattr(value, "is_hydrated", False) is not True:
                    raise ValueError
                fact = ModalRuntimeReleaseVolumeFactV1(
                    spec, getattr(value, "object_id", None),
                )
                volumes[spec.role] = (value, fact)
            for spec in plan.secrets:
                value = self._sdk.Secret.from_name(
                    spec.name, environment_name=plan.environment_name,
                    required_keys=list(spec.required_keys), client=self._client,
                )
                value.hydrate(self._client)
                if getattr(value, "is_hydrated", False) is not True:
                    raise ValueError
                fact = ModalRuntimeReleaseSecretFactV1(
                    spec, getattr(value, "object_id", None),
                )
                secrets[spec.name] = (value, fact)
        except Exception:
            failed = True
        if failed:
            raise ModalRuntimeReleaseDeploymentError("modal_release_resource_unavailable")
        return volumes, secrets

    @staticmethod
    def _validate_entrypoint(spec: ModalRuntimeReleaseFunctionSpecV1, value: object) -> Callable:
        if not callable(value) or getattr(value, "__module__", None) != spec.module \
                or getattr(value, "__qualname__", None) != spec.qualname \
                or "<locals>" in spec.qualname:
            raise ValueError("Modal release entrypoint differs from the installed package")
        return value

    @staticmethod
    def _provider_identity(value: object, attribute: str, kind: str) -> str:
        if getattr(value, "is_hydrated", True) is not True:
            raise ValueError("Modal release provider object is not hydrated")
        return _provider_id(getattr(value, attribute, None), kind)

    @staticmethod
    def _validate_prior(
        prior: ModalRuntimeReleaseDeploymentObservationV1 | None,
        plan: ModalRuntimeReleaseDeploymentPlanV1,
    ) -> None:
        if prior is None:
            return
        expected_names = tuple(sorted(item.name for item in plan.functions))
        if prior.deployed is False:
            historical = {name for name, _ in prior.function_ids}
            if prior.class_ids or historical.intersection(expected_names):
                raise ValueError("stopped Modal app collides with the selected layout")
            return
        if prior.class_ids or tuple(name for name, _ in prior.function_ids) != expected_names:
            raise ValueError("current Modal app has an unowned layout")

    @staticmethod
    def _validate_transition(
        prior: ModalRuntimeReleaseDeploymentObservationV1 | None,
        current: ModalRuntimeReleaseDeploymentObservationV1,
        *, app_id: str, function_ids: tuple[tuple[str, str], ...],
    ) -> None:
        if not current.deployed or current.app_id != app_id \
                or current.class_ids or current.function_ids != function_ids:
            raise ValueError("Modal release deployment layout differs")
        if prior is None:
            valid = current.generation == 1
        elif prior.deployed:
            valid = current.app_id == prior.app_id \
                and current.generation == prior.generation + 1
        else:
            valid = current.app_id != prior.previous_app_id and current.generation == 1
        if not valid:
            raise ValueError("Modal release deployment generation did not advance exactly once")

    @staticmethod
    def _deploy_from_private_nonrepo(app: object, *, client: object, plan) -> None:
        # Modal 1.5.4's public App.deploy probes local Git metadata.  An empty
        # private cwd ensures that probe finds no repository or source metadata.
        with _DEPLOY_CWD_LOCK:
            original = os.getcwd()
            with tempfile.TemporaryDirectory(prefix="synaptic-modal-release-") as directory:
                try:
                    if os.name != "nt":
                        os.chmod(directory, 0o700)
                    os.chdir(directory)
                    app.deploy(
                        environment_name=plan.environment_name,
                        client=client,
                        strategy=plan.strategy,
                    )
                finally:
                    os.chdir(original)

    def deploy_once(
        self, plan: ModalRuntimeReleaseDeploymentPlanV1, *,
        entrypoints: Mapping[str, Callable],
    ) -> ModalRuntimeReleaseDeploymentFactsV1:
        """Perform one already-authorized deployment call, never a retry."""
        if type(plan) is not ModalRuntimeReleaseDeploymentPlanV1:
            raise TypeError("exact Modal release deployment plan required")
        if plan.environment_name != self._binding.environment_ref:
            raise ValueError("Modal release plan differs from the explicit client scope")
        if type(entrypoints) is not dict or set(entrypoints) != {"training", "self_check"}:
            raise TypeError("exact packaged training and self-check entrypoints required")
        resolved = {
            spec.role: self._validate_entrypoint(spec, entrypoints[spec.role])
            for spec in plan.functions
        }
        self._observe_scope(plan.environment_name)
        prior = self._observe(plan)
        self._validate_prior(prior, plan)
        volume_objects, secret_objects = self._hydrate_resources(plan)
        try:
            image = self._sdk.Image.from_registry(plan.release.image_ref).entrypoint([])
            app = self._sdk.App(plan.app_name, image=image, include_source=False)
            function_objects: list[tuple[ModalRuntimeReleaseFunctionSpecV1, object]] = []
            for spec in plan.functions:
                volumes = {
                    volume_objects[role][1].spec.mount_path: volume_objects[role][0]
                    for role in spec.volume_roles
                }
                secrets = [secret_objects[name][0] for name in spec.secret_names]
                function = app.function(
                    name=spec.name,
                    image=image,
                    cpu=spec.cpu_milli / 1000,
                    memory=spec.memory_mib,
                    gpu=spec.gpu,
                    timeout=spec.timeout_seconds,
                    retries=0,
                    volumes=volumes,
                    secrets=secrets,
                    block_network=spec.block_network,
                    restrict_modal_access=spec.restrict_modal_access,
                    single_use_containers=True,
                    serialized=False,
                    include_source=False,
                )(resolved[spec.role])
                function_objects.append((spec, function))
        except Exception:
            raise ModalRuntimeReleaseDeploymentError("modal_release_construction_failed") from None

        # Crossing this call boundary is mutation. Any failure is ambiguous and
        # the caller must retain its consumed claim without retry or facts.
        failed = False
        try:
            self._deploy_from_private_nonrepo(app, client=self._client, plan=plan)
        except Exception:
            failed = True
        if failed:
            raise ModalRuntimeReleaseDeploymentError("modal_release_deployment_indeterminate")

        try:
            app_id = self._provider_identity(app, "app_id", "app_id")
            image_id = self._provider_identity(image, "object_id", "image_id")
            function_facts = tuple(
                ModalRuntimeReleaseFunctionFactV1(
                    spec, self._provider_identity(function, "object_id", "function_id"),
                )
                for spec, function in function_objects
            )
            layout = tuple(sorted(
                (item.spec.name, item.function_id) for item in function_facts
            ))
            current = self._observe(plan)
            if current is None:
                raise ValueError
            self._validate_transition(prior, current, app_id=app_id, function_ids=layout)
            release = plan.release
            facts = ModalRuntimeReleaseDeploymentFactsV1(
                deployment_spec_digest=plan.deployment_spec_digest,
                runtime_release_digest=release.manifest_digest,
                client_binding=self._binding,
                app_name=plan.app_name, app_id=app_id, generation=current.generation,
                image_reference=release.image_ref, image_digest=release.image_digest,
                image_id=image_id, functions=function_facts,
                volumes=tuple(item[1] for item in volume_objects.values()),
                secrets=tuple(item[1] for item in secret_objects.values()),
                python_implementation=release.python_implementation,
                python_version=release.python_version,
                python_executable=release.python_executable,
                python_executable_digest=release.python_executable_digest,
                package_digest=release.package_digest,
                installed_distributions_digest=release.installed_distributions_digest,
                worker_entrypoint=release.worker_entrypoint,
                worker_closure_digest=release.worker_closure_digest,
            )
            facts.validate_plan(plan)
            return facts
        except ModalRuntimeReleaseDeploymentError:
            raise
        except Exception:
            raise ModalRuntimeReleaseDeploymentError(
                "modal_release_acknowledgement_invalid"
            ) from None


def parse_modal_runtime_release_deployment_plan(
    payload: bytes,
) -> ModalRuntimeReleaseDeploymentPlanV1:
    return ModalRuntimeReleaseDeploymentPlanV1.parse(payload)


def parse_modal_runtime_release_deployment_facts(
    payload: bytes,
) -> ModalRuntimeReleaseDeploymentFactsV1:
    return ModalRuntimeReleaseDeploymentFactsV1.parse(payload)


__all__ = [
    "MODAL_RUNTIME_RELEASE_DEPLOYMENT_FACTS_SCHEMA",
    "MODAL_RUNTIME_RELEASE_DEPLOYMENT_OBSERVATION_SCHEMA",
    "MODAL_RUNTIME_RELEASE_DEPLOYMENT_PLAN_SCHEMA",
    "EXACT_SELF_CHECK_MODULE",
    "EXACT_SELF_CHECK_QUALNAME",
    "EXACT_QUALIFICATION_SECRET_REQUIRED_KEYS",
    "EXACT_QUALIFICATION_KEY_REF",
    "ExplicitModal154ReleaseDeploymentReader",
    "ModalRuntimeReleaseDeployer",
    "ModalRuntimeReleaseDeploymentError",
    "ModalRuntimeReleaseDeploymentFactsV1",
    "ModalRuntimeReleaseDeploymentObservationV1",
    "ModalRuntimeReleaseDeploymentPlanV1",
    "ModalRuntimeReleaseDeploymentReadPort",
    "ModalRuntimeReleaseFunctionFactV1",
    "ModalRuntimeReleaseFunctionSpecV1",
    "ModalRuntimeReleaseSecretFactV1",
    "ModalRuntimeReleaseSecretSpecV1",
    "ModalRuntimeReleaseVolumeFactV1",
    "ModalRuntimeReleaseVolumeSpecV1",
    "parse_modal_runtime_release_deployment_facts",
    "parse_modal_runtime_release_deployment_plan",
]
