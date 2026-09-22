"""Provider-neutral packaged training runtime release contracts.

The contracts in this module deliberately stop before provider execution.  A
release describes immutable package/runtime compatibility facts. A provider
binding attaches only a versioned digest commitment to provider-adapter-owned
facts; provider-native facts and secret references never enter this shared
document. An execution binding joins those identities to one prepared run.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


PACKAGED_RUNTIME_RELEASE_SCHEMA = "synaptic-packaged-training-runtime-release/v1"
PROVIDER_RUNTIME_BINDING_SCHEMA = "synaptic-provider-runtime-binding/v1"
PACKAGED_EXECUTION_BINDING_SCHEMA = "synaptic-packaged-execution-binding/v1"

_MAX_JSON_BYTES = 128 * 1024
_MAX_FACT_BYTES = 64 * 1024
_MAX_FACT_DEPTH = 8
_MAX_FACT_NODES = 1024
_MAX_CONTAINER_ITEMS = 128
_MAX_FACT_TEXT_BYTES = 32 * 1024
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}$")
_OPAQUE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+\-]{0,255}$")
_SCHEMA_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}/v[1-9][0-9]*$")
_FACT_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_PREPARED_REF = re.compile(r"^prepared://sha256/(?P<digest>[0-9a-f]{64})$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:[A-Za-z0-9.+\-]{0,64})?$")
_PACKAGE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_ENTRYPOINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_OCI_IMAGE = re.compile(
    r"^(?P<name>[a-z0-9](?:[a-z0-9._:/-]{0,510}[a-z0-9])?)@sha256:(?P<digest>[0-9a-f]{64})$"
)
_MODEL_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
def _text(value: object, name: str, *, maximum: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value:
        raise ValueError(f"{name} must be nonblank canonical text")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its byte limit")
    if value != value.strip():
        raise ValueError(f"{name} must be nonblank canonical text")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must be NFC")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{name} must not contain control characters")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be valid UTF-8 text") from None
    if len(encoded) > maximum:
        raise ValueError(f"{name} exceeds its byte limit")
    return value


def _safe_ref(value: object, name: str) -> str:
    result = _text(value, name)
    if _SAFE_REF.fullmatch(result) is None:
        raise ValueError(f"{name} must be a bounded safe reference")
    return result


def _opaque_ref(value: object, name: str) -> str:
    """Validate identity material that is never interpreted as a path or URI."""

    result = _text(value, name, maximum=256)
    if _OPAQUE_REF.fullmatch(result) is None:
        raise ValueError(f"{name} must be a bounded opaque non-path reference")
    return result


def _schema_identity(value: object, name: str) -> str:
    result = _text(value, name, maximum=132)
    if _SCHEMA_IDENTITY.fullmatch(result) is None:
        raise ValueError(f"{name} must be a versioned provider-specific schema identity")
    return result


def _prepared_ref(value: object) -> tuple[str, str]:
    result = _text(value, "prepared_input.ref", maximum=512)
    match = _PREPARED_REF.fullmatch(result)
    if match is None:
        raise ValueError(
            "prepared_input.ref must be exact prepared://sha256/<semantic-digest> identity"
        )
    return result, match.group("digest")


def _digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object, name: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    if len(value) != len(expected):
        raise ValueError(f"{name} has missing or unknown fields")
    snapshot = dict.copy(value)
    if any(type(key) is not str for key in dict.keys(snapshot)):
        raise TypeError(f"{name} field names must be exact strings")
    if frozenset(snapshot) != expected:
        raise ValueError(f"{name} has missing or unknown fields")
    return snapshot


@dataclass(slots=True)
class _FactBudget:
    nodes: int = 0
    text_bytes: int = 0


def _json_value(
    value: object,
    name: str,
    *,
    depth: int = 0,
    budget: _FactBudget | None = None,
    active: set[int] | None = None,
) -> object:
    """Bound and snapshot exact JSON without aliases or cross-runtime numbers."""

    budget = _FactBudget() if budget is None else budget
    active = set() if active is None else active
    budget.nodes += 1
    if budget.nodes > _MAX_FACT_NODES:
        raise ValueError(f"{name} exceeds its node limit")
    if depth > _MAX_FACT_DEPTH:
        raise ValueError(f"{name} exceeds its depth limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise ValueError(f"{name} integer exceeds the cross-runtime safe range")
        return value
    if type(value) is float:
        raise TypeError(f"{name} must encode fractional values as normalized strings")
    if type(value) is str:
        result = _text(value, name, maximum=4096)
        budget.text_bytes += len(result.encode("utf-8"))
        if budget.text_bytes > _MAX_FACT_TEXT_BYTES:
            raise ValueError(f"{name} exceeds its text limit")
        return result
    if type(value) not in (list, dict):
        raise TypeError(f"{name} must contain only exact canonical JSON values")
    identity = id(value)
    if identity in active:
        raise ValueError(f"{name} must not contain cycles")
    if len(value) > _MAX_CONTAINER_ITEMS:
        raise ValueError(f"{name} container exceeds its item limit")
    active.add(identity)
    try:
        if type(value) is list:
            return [
                _json_value(
                    item, name, depth=depth + 1, budget=budget, active=active
                )
                for item in list(value)
            ]
        snapshot = dict.copy(value)
        if any(type(key) is not str for key in dict.keys(snapshot)):
            raise TypeError(f"{name} contains a non-string field name")
        result: dict[str, object] = {}
        for key, item in dict.items(snapshot):
            normalized_key = _text(key, name, maximum=128)
            if _FACT_KEY.fullmatch(normalized_key) is None:
                raise ValueError(f"{name} keys must be bounded ASCII identifiers")
            budget.text_bytes += len(normalized_key.encode("utf-8"))
            if budget.text_bytes > _MAX_FACT_TEXT_BYTES:
                raise ValueError(f"{name} exceeds its text limit")
            result[normalized_key] = _json_value(
                item, name, depth=depth + 1, budget=budget, active=active
            )
        return result
    finally:
        active.remove(identity)


def _canonical(value: object, *, maximum: int = _MAX_JSON_BYTES) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
        raise ValueError("contract must contain only canonical JSON values") from None
    if len(encoded) > maximum:
        raise ValueError("canonical contract exceeds its byte limit")
    return encoded


def _domain_digest(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical(value)).hexdigest()


def _parse_json(value: str, name: str) -> dict[str, object]:
    if type(value) is not str:
        raise TypeError(f"{name} JSON must be an exact string")
    if not value or len(value) > _MAX_JSON_BYTES:
        raise ValueError(f"{name} JSON exceeds its byte limit")
    try:
        raw = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} JSON is malformed") from None
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{name} JSON exceeds its byte limit")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} JSON contains duplicate fields")
            result[key] = item
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError(f"{name} JSON contains a non-finite number")

    try:
        document = json.loads(
            value, object_pairs_hook=unique, parse_constant=reject_constant
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError, MemoryError):
        raise ValueError(f"{name} JSON is malformed") from None
    if type(document) is not dict:
        raise TypeError(f"{name} JSON must encode an object")
    return document


def _canonical_tuple(value: object, name: str) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise TypeError(f"{name} must be an exact array")
    if not value or len(value) > 256:
        raise ValueError(f"{name} must be nonempty, unique, ascending, and bounded")
    result = tuple(_safe_ref(item, name) for item in value)
    if not result or len(result) > 256 or result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be nonempty, unique, and ascending")
    return result


def _models(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) not in (tuple, list):
        raise TypeError("compatibility.models must be an exact array")
    if not value or len(value) > 256:
        raise ValueError(
            "compatibility.models must be nonempty, unique, ascending, and bounded"
        )
    models: list[tuple[str, str]] = []
    for item in value:
        if type(item) is tuple:
            if len(item) != 2:
                raise ValueError("compatible model tuple must have two items")
            ref, revision = item
        else:
            parsed = _fields(
                item, frozenset({"ref", "revision"}), "compatibility model"
            )
            ref, revision = parsed["ref"], parsed["revision"]
        model_ref = _safe_ref(ref, "model.ref")
        if (
            type(revision) is not str
            or len(revision) not in (40, 64)
            or _MODEL_REVISION.fullmatch(revision) is None
        ):
            raise ValueError("model.revision must be an immutable lowercase revision")
        models.append((model_ref, revision))
    result = tuple(models)
    if not result or len(result) > 256 or result != tuple(sorted(set(result))):
        raise ValueError("compatibility.models must be nonempty, unique, and ascending")
    return result


def _fact_bytes(value: object, name: str) -> bytes:
    if type(value) is bytes:
        raw = value
        if not raw or len(raw) > _MAX_FACT_BYTES:
            raise ValueError(f"{name} bytes exceed their limit")
        try:
            document = _parse_json(raw.decode("utf-8"), name)
        except UnicodeError:
            raise ValueError(f"{name} must be canonical UTF-8 JSON") from None
        snapshot = _json_value(document, name)
        if _canonical(snapshot, maximum=_MAX_FACT_BYTES) != raw:
            raise ValueError(f"{name} bytes must be canonical JSON")
        return raw
    snapshot = _json_value(value, name)
    if type(snapshot) is not dict:
        raise TypeError(f"{name} must be an exact object")
    return _canonical(snapshot, maximum=_MAX_FACT_BYTES)


def _facts(value: bytes) -> dict[str, object]:
    return json.loads(value.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class PackagedTrainingRuntimeReleaseV1:
    release_ref: str
    package_name: str
    package_version: str
    package_digest: str
    source_provenance_digest: str
    worker_entrypoint: str
    worker_closure_digest: str
    image_ref: str
    image_digest: str
    python_implementation: str
    python_version: str
    python_executable: str
    python_executable_digest: str
    installed_distributions_digest: str
    installed_distribution_count: int
    platform_system: str
    platform_machine: str
    cuda_version: str | None
    runtime_facts: bytes
    compatible_methods: tuple[str, ...]
    compatible_models: tuple[tuple[str, str], ...]
    compatible_dataset_formats: tuple[str, ...]
    workload_schema: str
    prepared_input_schema: str
    artifact_contract_schema: str
    manifest_digest: str
    schema_version: str = PACKAGED_RUNTIME_RELEASE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != PACKAGED_RUNTIME_RELEASE_SCHEMA:
            raise ValueError("packaged runtime release schema is unsupported")
        object.__setattr__(self, "release_ref", _opaque_ref(self.release_ref, "release_ref"))
        if (
            type(self.package_name) is not str
            or len(self.package_name) > 128
            or _PACKAGE.fullmatch(self.package_name) is None
        ):
            raise ValueError("package.name is invalid")
        package_version = _text(self.package_version, "package.version", maximum=128)
        if _VERSION.fullmatch(package_version) is None:
            raise ValueError("package.version must be an exact semantic version")
        for field in (
            "package_digest", "source_provenance_digest", "worker_closure_digest",
            "image_digest", "python_executable_digest", "installed_distributions_digest",
            "manifest_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        worker_entrypoint = _text(
            self.worker_entrypoint, "worker.entrypoint", maximum=512
        )
        if _ENTRYPOINT.fullmatch(worker_entrypoint) is None:
            raise ValueError("worker.entrypoint must be a canonical package entrypoint")
        image = _text(self.image_ref, "image.reference")
        match = _OCI_IMAGE.fullmatch(image)
        if match is None or match.group("digest") != self.image_digest:
            raise ValueError("image.reference must be immutable and match image.digest")
        if self.python_implementation != "cpython":
            raise ValueError("python.implementation must be cpython")
        python_version = _text(self.python_version, "python.version", maximum=128)
        if _VERSION.fullmatch(python_version) is None:
            raise ValueError("python.version must be an exact semantic version")
        executable = _text(self.python_executable, "python.executable")
        path = PurePosixPath(executable)
        if not path.is_absolute() or path.as_posix() != executable or any(
            part in {"", ".", ".."} for part in path.parts[1:]
        ):
            raise ValueError("python.executable must be an absolute canonical POSIX path")
        object.__setattr__(
            self, "installed_distribution_count",
            _integer(
                self.installed_distribution_count,
                "installed_distributions.count",
                minimum=1,
                maximum=1_000_000,
            ),
        )
        object.__setattr__(self, "platform_system", _safe_ref(self.platform_system, "platform.system"))
        object.__setattr__(self, "platform_machine", _safe_ref(self.platform_machine, "platform.machine"))
        if self.cuda_version is not None:
            cuda_version = _text(
                self.cuda_version, "platform.cuda_version", maximum=128
            )
            if _VERSION.fullmatch(cuda_version) is None:
                raise ValueError("platform.cuda_version must be null or an exact semantic version")
        object.__setattr__(self, "runtime_facts", _fact_bytes(self.runtime_facts, "runtime_facts"))
        object.__setattr__(self, "compatible_methods", _canonical_tuple(self.compatible_methods, "compatibility.methods"))
        object.__setattr__(self, "compatible_models", _models(self.compatible_models))
        object.__setattr__(self, "compatible_dataset_formats", _canonical_tuple(self.compatible_dataset_formats, "compatibility.dataset_formats"))
        for field in ("workload_schema", "prepared_input_schema", "artifact_contract_schema"):
            object.__setattr__(self, field, _safe_ref(getattr(self, field), f"contracts.{field}"))
        if self.manifest_digest != self.expected_manifest_digest:
            raise ValueError("manifest_digest does not bind the runtime release")

    @classmethod
    def build(cls, **values: Any) -> "PackagedTrainingRuntimeReleaseV1":
        if "manifest_digest" in values or "schema_version" in values:
            raise ValueError("build computes release schema and manifest digest")
        provisional = cls.__new__(cls)
        for field in cls.__dataclass_fields__:
            if field not in {"manifest_digest", "schema_version"}:
                if field not in values:
                    raise TypeError(f"missing release field: {field}")
                object.__setattr__(provisional, field, values.pop(field))
        if values:
            raise TypeError("unknown release build fields")
        object.__setattr__(provisional, "schema_version", PACKAGED_RUNTIME_RELEASE_SCHEMA)
        object.__setattr__(provisional, "manifest_digest", "0" * 64)
        # Normalize all fields once; the placeholder digest intentionally fails last.
        try:
            provisional.__post_init__()
        except ValueError as error:
            if str(error) != "manifest_digest does not bind the runtime release":
                raise
        object.__setattr__(provisional, "manifest_digest", provisional.expected_manifest_digest)
        provisional.__post_init__()
        return provisional

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_ref": self.release_ref,
            "package": {
                "name": self.package_name, "version": self.package_version,
                "digest": self.package_digest,
                "source_provenance_digest": self.source_provenance_digest,
            },
            "worker": {"entrypoint": self.worker_entrypoint, "closure_digest": self.worker_closure_digest},
            "image": {"reference": self.image_ref, "digest": self.image_digest},
            "python": {
                "implementation": self.python_implementation, "version": self.python_version,
                "executable": self.python_executable,
                "executable_digest": self.python_executable_digest,
            },
            "installed_distributions": {
                "digest": self.installed_distributions_digest,
                "count": self.installed_distribution_count,
            },
            "platform": {
                "system": self.platform_system, "machine": self.platform_machine,
                "cuda_version": self.cuda_version, "runtime_facts": _facts(self.runtime_facts),
            },
            "compatibility": {
                "methods": list(self.compatible_methods),
                "models": [{"ref": ref, "revision": revision} for ref, revision in self.compatible_models],
                "dataset_formats": list(self.compatible_dataset_formats),
            },
            "contracts": {
                "workload_schema": self.workload_schema,
                "prepared_input_schema": self.prepared_input_schema,
                "artifact_contract_schema": self.artifact_contract_schema,
            },
        }

    @property
    def expected_manifest_digest(self) -> str:
        return _domain_digest(PACKAGED_RUNTIME_RELEASE_SCHEMA, self._unsigned_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "manifest_digest": self.manifest_digest}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PackagedTrainingRuntimeReleaseV1":
        root = _fields(value, frozenset({
            "schema_version", "release_ref", "package", "worker", "image", "python",
            "installed_distributions", "platform", "compatibility", "contracts", "manifest_digest",
        }), "runtime release")
        package = _fields(root["package"], frozenset({"name", "version", "digest", "source_provenance_digest"}), "package")
        worker = _fields(root["worker"], frozenset({"entrypoint", "closure_digest"}), "worker")
        image = _fields(root["image"], frozenset({"reference", "digest"}), "image")
        python = _fields(root["python"], frozenset({"implementation", "version", "executable", "executable_digest"}), "python")
        installed = _fields(root["installed_distributions"], frozenset({"digest", "count"}), "installed_distributions")
        platform = _fields(root["platform"], frozenset({"system", "machine", "cuda_version", "runtime_facts"}), "platform")
        compatibility = _fields(root["compatibility"], frozenset({"methods", "models", "dataset_formats"}), "compatibility")
        contracts = _fields(root["contracts"], frozenset({"workload_schema", "prepared_input_schema", "artifact_contract_schema"}), "contracts")
        return cls(
            release_ref=root["release_ref"], package_name=package["name"], package_version=package["version"],
            package_digest=package["digest"], source_provenance_digest=package["source_provenance_digest"],
            worker_entrypoint=worker["entrypoint"], worker_closure_digest=worker["closure_digest"],
            image_ref=image["reference"], image_digest=image["digest"],
            python_implementation=python["implementation"], python_version=python["version"],
            python_executable=python["executable"], python_executable_digest=python["executable_digest"],
            installed_distributions_digest=installed["digest"], installed_distribution_count=installed["count"],
            platform_system=platform["system"], platform_machine=platform["machine"], cuda_version=platform["cuda_version"],
            runtime_facts=_fact_bytes(platform["runtime_facts"], "runtime_facts"),
            compatible_methods=compatibility["methods"], compatible_models=compatibility["models"],
            compatible_dataset_formats=compatibility["dataset_formats"],
            workload_schema=contracts["workload_schema"], prepared_input_schema=contracts["prepared_input_schema"],
            artifact_contract_schema=contracts["artifact_contract_schema"], manifest_digest=root["manifest_digest"],
            schema_version=root["schema_version"],
        )  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, value: str) -> "PackagedTrainingRuntimeReleaseV1":
        result = cls.from_dict(_parse_json(value, "runtime release"))
        if result.canonical_json() != value:
            raise ValueError("runtime release JSON must be canonical")
        return result


@dataclass(frozen=True, slots=True)
class ProviderRuntimeBindingV1:
    provider_ref: str
    runtime_release_digest: str
    provider_facts_schema: str
    provider_facts_digest: str
    binding_digest: str
    schema_version: str = PROVIDER_RUNTIME_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != PROVIDER_RUNTIME_BINDING_SCHEMA:
            raise ValueError("provider runtime binding schema is unsupported")
        object.__setattr__(self, "provider_ref", _opaque_ref(self.provider_ref, "provider_ref"))
        object.__setattr__(self, "runtime_release_digest", _digest(self.runtime_release_digest, "runtime_release_digest"))
        object.__setattr__(
            self,
            "provider_facts_schema",
            _schema_identity(self.provider_facts_schema, "provider_facts_schema"),
        )
        object.__setattr__(
            self,
            "provider_facts_digest",
            _digest(self.provider_facts_digest, "provider_facts_digest"),
        )
        object.__setattr__(self, "binding_digest", _digest(self.binding_digest, "binding_digest"))
        if self.binding_digest != self.expected_binding_digest:
            raise ValueError("binding_digest does not bind the provider runtime")

    @classmethod
    def build(
        cls, *, provider_ref: str,
        runtime_release: PackagedTrainingRuntimeReleaseV1,
        provider_facts_schema: str, provider_facts_digest: str,
    ) -> "ProviderRuntimeBindingV1":
        if type(runtime_release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("runtime_release must be exact PackagedTrainingRuntimeReleaseV1")
        provider_ref = _opaque_ref(provider_ref, "provider_ref")
        provider_facts_schema = _schema_identity(
            provider_facts_schema, "provider_facts_schema"
        )
        provider_facts_digest = _digest(
            provider_facts_digest, "provider_facts_digest"
        )
        unsigned = {
            "schema_version": PROVIDER_RUNTIME_BINDING_SCHEMA,
            "provider_ref": provider_ref,
            "runtime_release_digest": runtime_release.manifest_digest,
            "provider_facts_schema": provider_facts_schema,
            "provider_facts_digest": provider_facts_digest,
        }
        return cls(
            provider_ref, runtime_release.manifest_digest,
            provider_facts_schema, provider_facts_digest,
            _domain_digest(PROVIDER_RUNTIME_BINDING_SCHEMA, unsigned),
        )

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "provider_ref": self.provider_ref,
            "runtime_release_digest": self.runtime_release_digest,
            "provider_facts_schema": self.provider_facts_schema,
            "provider_facts_digest": self.provider_facts_digest,
        }

    @property
    def expected_binding_digest(self) -> str:
        return _domain_digest(PROVIDER_RUNTIME_BINDING_SCHEMA, self._unsigned_dict())

    def binds(self, runtime_release: PackagedTrainingRuntimeReleaseV1) -> bool:
        return type(runtime_release) is PackagedTrainingRuntimeReleaseV1 and self.runtime_release_digest == runtime_release.manifest_digest

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "binding_digest": self.binding_digest}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ProviderRuntimeBindingV1":
        root = _fields(value, frozenset({
            "schema_version", "provider_ref", "runtime_release_digest",
            "provider_facts_schema", "provider_facts_digest", "binding_digest",
        }), "provider runtime binding")
        return cls(
            root["provider_ref"], root["runtime_release_digest"],
            root["provider_facts_schema"], root["provider_facts_digest"],
            root["binding_digest"], root["schema_version"],
        )  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, value: str) -> "ProviderRuntimeBindingV1":
        result = cls.from_dict(_parse_json(value, "provider runtime binding"))
        if result.canonical_json() != value:
            raise ValueError("provider runtime binding JSON must be canonical")
        return result


@dataclass(frozen=True, slots=True)
class PackagedExecutionBindingV1:
    run_ref: str
    runtime_release_digest: str
    provider_runtime_binding_digest: str
    prepared_input_ref: str
    prepared_input_revision: str
    prepared_input_content_digest: str
    prepared_input_size_bytes: int
    prepared_input_format: str
    workload_digest: str
    configuration_digest: str
    artifact_policy_digest: str
    binding_digest: str
    schema_version: str = PACKAGED_EXECUTION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != PACKAGED_EXECUTION_BINDING_SCHEMA:
            raise ValueError("packaged execution binding schema is unsupported")
        object.__setattr__(self, "run_ref", _opaque_ref(self.run_ref, "run_ref"))
        prepared_ref, semantic_digest = _prepared_ref(self.prepared_input_ref)
        object.__setattr__(self, "prepared_input_ref", prepared_ref)
        object.__setattr__(
            self,
            "prepared_input_revision",
            _digest(self.prepared_input_revision, "prepared_input_revision"),
        )
        object.__setattr__(
            self,
            "prepared_input_content_digest",
            _digest(
                self.prepared_input_content_digest,
                "prepared_input_content_digest",
            ),
        )
        object.__setattr__(
            self,
            "prepared_input_size_bytes",
            _integer(
                self.prepared_input_size_bytes,
                "prepared_input_size_bytes",
                minimum=1,
                maximum=64 * 1024 * 1024,
            ),
        )
        object.__setattr__(
            self,
            "prepared_input_format",
            _safe_ref(self.prepared_input_format, "prepared_input_format"),
        )
        for field in (
            "runtime_release_digest", "provider_runtime_binding_digest",
            "workload_digest", "configuration_digest", "artifact_policy_digest", "binding_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.prepared_input_revision != semantic_digest:
            raise ValueError(
                "prepared_input.revision must match the semantic digest embedded in prepared_input.ref"
            )
        if self.binding_digest != self.expected_binding_digest:
            raise ValueError("binding_digest does not bind the packaged execution")

    @classmethod
    def build(
        cls, *, run_ref: str, runtime_release: PackagedTrainingRuntimeReleaseV1,
        provider_runtime_binding: ProviderRuntimeBindingV1, prepared_input_ref: str,
        prepared_input_revision: str, prepared_input_content_digest: str,
        prepared_input_size_bytes: int, prepared_input_format: str,
        workload_digest: str, configuration_digest: str,
        artifact_policy_digest: str,
    ) -> "PackagedExecutionBindingV1":
        if type(runtime_release) is not PackagedTrainingRuntimeReleaseV1:
            raise TypeError("runtime_release must be exact PackagedTrainingRuntimeReleaseV1")
        if type(provider_runtime_binding) is not ProviderRuntimeBindingV1:
            raise TypeError("provider_runtime_binding must be exact ProviderRuntimeBindingV1")
        if not provider_runtime_binding.binds(runtime_release):
            raise ValueError("provider runtime binding targets a different runtime release")
        run_ref = _opaque_ref(run_ref, "run_ref")
        prepared_input_ref, semantic_digest = _prepared_ref(prepared_input_ref)
        prepared_input_revision = _digest(
            prepared_input_revision, "prepared_input_revision"
        )
        if prepared_input_revision != semantic_digest:
            raise ValueError(
                "prepared_input.revision must match the semantic digest embedded in prepared_input.ref"
            )
        prepared_input_content_digest = _digest(
            prepared_input_content_digest, "prepared_input_content_digest"
        )
        prepared_input_size_bytes = _integer(
            prepared_input_size_bytes, "prepared_input_size_bytes",
            minimum=1, maximum=64 * 1024 * 1024,
        )
        prepared_input_format = _safe_ref(
            prepared_input_format, "prepared_input_format"
        )
        workload_digest = _digest(workload_digest, "workload_digest")
        configuration_digest = _digest(
            configuration_digest, "configuration_digest"
        )
        artifact_policy_digest = _digest(
            artifact_policy_digest, "artifact_policy_digest"
        )
        values = {
            "schema_version": PACKAGED_EXECUTION_BINDING_SCHEMA,
            "run_ref": run_ref, "runtime_release_digest": runtime_release.manifest_digest,
            "provider_runtime_binding_digest": provider_runtime_binding.binding_digest,
            "prepared_input": {
                "ref": prepared_input_ref, "revision": prepared_input_revision,
                "content_digest": prepared_input_content_digest,
                "size_bytes": prepared_input_size_bytes,
                "format": prepared_input_format,
            },
            "workload_digest": workload_digest, "configuration_digest": configuration_digest,
            "artifact_policy_digest": artifact_policy_digest,
        }
        return cls(
            run_ref, runtime_release.manifest_digest, provider_runtime_binding.binding_digest,
            prepared_input_ref, prepared_input_revision,
            prepared_input_content_digest, prepared_input_size_bytes,
            prepared_input_format,
            workload_digest, configuration_digest, artifact_policy_digest,
            _domain_digest(PACKAGED_EXECUTION_BINDING_SCHEMA, values),
        )

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "run_ref": self.run_ref,
            "runtime_release_digest": self.runtime_release_digest,
            "provider_runtime_binding_digest": self.provider_runtime_binding_digest,
            "prepared_input": {
                "ref": self.prepared_input_ref, "revision": self.prepared_input_revision,
                "content_digest": self.prepared_input_content_digest,
                "size_bytes": self.prepared_input_size_bytes,
                "format": self.prepared_input_format,
            },
            "workload_digest": self.workload_digest,
            "configuration_digest": self.configuration_digest,
            "artifact_policy_digest": self.artifact_policy_digest,
        }

    @property
    def expected_binding_digest(self) -> str:
        return _domain_digest(PACKAGED_EXECUTION_BINDING_SCHEMA, self._unsigned_dict())

    def validate_bindings(
        self, runtime_release: PackagedTrainingRuntimeReleaseV1,
        provider_runtime_binding: ProviderRuntimeBindingV1,
    ) -> None:
        if type(runtime_release) is not PackagedTrainingRuntimeReleaseV1 or type(provider_runtime_binding) is not ProviderRuntimeBindingV1:
            raise TypeError("exact runtime release and provider binding are required")
        if (
            self.runtime_release_digest != runtime_release.manifest_digest
            or self.provider_runtime_binding_digest != provider_runtime_binding.binding_digest
            or not provider_runtime_binding.binds(runtime_release)
        ):
            raise ValueError("packaged execution has a cross-binding mismatch")

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "binding_digest": self.binding_digest}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PackagedExecutionBindingV1":
        root = _fields(value, frozenset({
            "schema_version", "run_ref", "runtime_release_digest", "provider_runtime_binding_digest",
            "prepared_input", "workload_digest", "configuration_digest", "artifact_policy_digest", "binding_digest",
        }), "packaged execution binding")
        prepared = _fields(
            root["prepared_input"],
            frozenset({"ref", "revision", "content_digest", "size_bytes", "format"}),
            "prepared_input",
        )
        return cls(
            root["run_ref"], root["runtime_release_digest"], root["provider_runtime_binding_digest"],
            prepared["ref"], prepared["revision"], prepared["content_digest"],
            prepared["size_bytes"], prepared["format"], root["workload_digest"],
            root["configuration_digest"], root["artifact_policy_digest"], root["binding_digest"], root["schema_version"],
        )  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, value: str) -> "PackagedExecutionBindingV1":
        result = cls.from_dict(_parse_json(value, "packaged execution binding"))
        if result.canonical_json() != value:
            raise ValueError("packaged execution binding JSON must be canonical")
        return result


__all__ = [
    "PACKAGED_EXECUTION_BINDING_SCHEMA",
    "PACKAGED_RUNTIME_RELEASE_SCHEMA",
    "PROVIDER_RUNTIME_BINDING_SCHEMA",
    "PackagedExecutionBindingV1",
    "PackagedTrainingRuntimeReleaseV1",
    "ProviderRuntimeBindingV1",
]
