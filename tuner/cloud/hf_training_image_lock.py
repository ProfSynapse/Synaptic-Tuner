"""Fail-closed candidate capture for the protected HF training image."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import signal
import ssl
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from tuner.cloud.hf_training_docker_archive import (
    DockerArchiveCommand,
    DockerArchiveError,
    MAX_ARCHIVE_BYTES,
    inspect_docker_archive,
    save_docker_archive,
)
from tuner.cloud.hf_training_oci_registry import (
    CHILD_MEDIA_TYPE,
    CONFIG_MEDIA_TYPES,
    HTTPRequest,
    HTTPResponse,
    PROVIDER_REPOSITORY,
    REGISTRY_REPOSITORY,
    OCIRegistryError,
    RegistryDocuments,
    fetch_registry_documents,
    parse_reference as parse_registry_reference,
)
from tuner.cloud.hf_training_image_operation_lock import (
    ImageOperationLockError,
    image_operation_lock,
)
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
)


CANDIDATE_SCHEMA = "synaptic-hf-training-runtime-lock-candidate/v1"
RUNTIME_LOCK_SCHEMA = "synaptic-hf-training-runtime-lock/v1"
PLATFORM = "linux/amd64"
MAX_DESCRIPTOR_BYTES = 4 * 1024 * 1024
MAX_RUNTIME_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 120
WARM_PULL_TIMEOUT_SECONDS = 3600
WARM_OVERALL_TIMEOUT_SECONDS = 3900
CAPTURE_OVERALL_TIMEOUT_SECONDS = 2700
REGISTRY_AGGREGATE_TIMEOUT_SECONDS = 120
REGISTRY_REQUEST_TIMEOUT_SECONDS = 30
RUNTIME_TIMEOUT_SECONDS = 300
DIAGNOSTIC_OVERALL_TIMEOUT_SECONDS = 900
SAVE_ARCHIVE_TIMEOUT_SECONDS = 900
CLEANUP_TIMEOUT_SECONDS = 60
INSPECT_FORMAT = '{"Id":{{json .Id}},"RepoDigests":{{json .RepoDigests}},"Os":{{json .Os}},"Architecture":{{json .Architecture}},"RootFS":{{json .RootFS}}}'
VERSION_FORMAT = '{"ClientVersion":{{json .Client.Version}},"ServerVersion":{{json .Server.Version}}}'
INFO_FORMAT = '{"ID":{{json .ID}},"ServerVersion":{{json .ServerVersion}},"OSType":{{json .OSType}},"Architecture":{{json .Architecture}},"Name":{{json .Name}},"DockerRootDir":{{json .DockerRootDir}},"Driver":{{json .Driver}},"SecurityOptions":{{json .SecurityOptions}}}'
CONTEXT_FORMAT = '{"Name":{{json .Name}},"DockerEndpoint":{{json .Endpoints.docker.Host}},"SkipTLSVerify":{{json .Endpoints.docker.SkipTLSVerify}}}'
LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.layer.v1.tar",
        "application/vnd.oci.image.layer.v1.tar+gzip",
        "application/vnd.oci.image.layer.v1.tar+zstd",
        "application/vnd.docker.image.rootfs.diff.tar.gzip",
        "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip",
    }
)
REASON_CODES = frozenset(
    {
        "IMAGE_INVALID",
        "DOCKER_CONFIG_INVALID",
        "EVIDENCE_INVALID",
        "INSPECTOR_INVALID",
        "COMMAND_FAILED",
        "OUTPUT_INVALID",
        "PROMOTION_FORBIDDEN",
        "OPERATION_LOCK_INVALID",
        "OPERATION_LOCK_TIMEOUT",
        "OPERATION_LOCK_CLEANUP_FAILED",
        "OPERATION_TIMEOUT",
        "CACHE_IDENTITY_INVALID",
    }
)
_RUNTIME_PACKAGES = (
    "accelerate", "datasets", "huggingface-hub", "numpy", "peft", "safetensors",
    "torch", "transformers", "trl", "unsloth", "unsloth-zoo",
)
_RUNTIME_SIGNATURES = (
    "TrainerCallback.on_optimizer_step", "safetensors.safe_open", "torch.load",
    "unsloth.import",
)
_UNSLOTH_IMPORT_SENTINEL = "GPU_RUNTIME_REQUIRED"
_DIAGNOSTIC_STAGES = frozenset({
    "preflight", "registry_initial", "operation_lock", "docker_authority_initial",
    "cache_identity_initial", "runtime_metadata", "cache_identity_final",
    "docker_authority_final", "registry_final", "final_integrity",
})
_DIAGNOSTIC_CATEGORIES = frozenset({
    "timeout", "nonzero", "identity", "document", "runtime", "cleanup",
})
_RUNNER_DIAGNOSTIC_HINTS = frozenset({"nonzero", "cleanup"})
_RUNTIME_SUBSTAGES = (
    "child_unreported", "python_bootstrap", "python_runtime", "workspace_setup",
    "distribution_metadata", "torch_import", "safetensors_import",
    "transformers_import", "signature_introspection", "unsloth_spec",
    "unsloth_origin", "unsloth_package_root", "site_roots", "site_membership",
    "user_site_isolation", "origin_chain", "result_serialization",
)
_RUNTIME_SUBSTAGE_SCHEMA = "synaptic-hf-training-runtime-substage/v1"
_RUNTIME_SUBSTAGE_MAX_BYTES = 65536
_RUNTIME_SUBSTAGE_FAILURE_BYTES = MappingProxyType({
    stage: json.dumps(
        {
            "runtime_substage": stage,
            "schema_version": _RUNTIME_SUBSTAGE_SCHEMA,
            "status": "FAILED",
        },
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    for stage in _RUNTIME_SUBSTAGES[1:]
})
_PYTHON_IDENTITY_SCHEMA = "synaptic-hf-training-python-runtime-identity/v1"
_PYTHON_IMPLEMENTATIONS = MappingProxyType({
    "cpython": "CPython", "pypy": "PyPy", "graalpy": "GraalPy",
    "jython": "Jython", "ironpython": "IronPython",
})
_PYTHON_VERSION = re.compile(
    r"^[1-9][0-9]{0,2}\.(?:0|[1-9][0-9]{0,2})\.(?:0|[1-9][0-9]{0,2})$",
)
_PYTHON_IDENTITY_FAILED_BYTES = (
    b'{"reason_code":"PYTHON_RUNTIME_IDENTITY_UNREPORTED",'
    b'"schema_version":"synaptic-hf-training-python-runtime-identity/v1",'
    b'"status":"FAILED"}'
)


class TrainingImageLockError(RuntimeError):
    def __init__(self, reason_code: str, *, diagnostic_category: str | None = None):
        self.reason_code = reason_code if reason_code in REASON_CODES else "EVIDENCE_INVALID"
        self.diagnostic_category = (
            diagnostic_category if diagnostic_category in _RUNNER_DIAGNOSTIC_HINTS else None
        )
        super().__init__(self.reason_code)


class MetadataDiagnosticStageError(RuntimeError):
    __slots__ = ("_failed_stage", "_category")

    def __init__(self, *, failed_stage: str, category: str):
        if failed_stage not in _DIAGNOSTIC_STAGES or category not in _DIAGNOSTIC_CATEGORIES:
            raise ValueError("invalid metadata diagnostic attribution")
        object.__setattr__(self, "_failed_stage", failed_stage)
        object.__setattr__(self, "_category", category)
        super().__init__("DIAGNOSTIC_STAGE_REJECTED")

    @property
    def failed_stage(self) -> str:
        return self._failed_stage

    @property
    def category(self) -> str:
        return self._category

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("metadata diagnostic attribution is immutable")


class RuntimeSubstageDiagnosticError(RuntimeError):
    __slots__ = ("_runtime_substage",)

    def __init__(self, *, runtime_substage: str):
        if runtime_substage not in _RUNTIME_SUBSTAGES:
            raise ValueError("invalid runtime substage attribution")
        object.__setattr__(self, "_runtime_substage", runtime_substage)
        super().__init__("RUNTIME_SUBSTAGE_REJECTED")

    @property
    def runtime_substage(self) -> str:
        return self._runtime_substage

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("runtime substage attribution is immutable")


class PythonRuntimeIdentityDiagnosticError(RuntimeError):
    __slots__ = ()

    def __init__(self):
        super().__init__("PYTHON_RUNTIME_IDENTITY_REJECTED")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("python runtime identity error is immutable")


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    timeout_seconds: int
    maximum_output_bytes: int


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes = b""
    stderr: bytes = b""


@dataclass(frozen=True)
class DockerExecutableIdentity:
    path: str
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


Runner = Callable[[CommandSpec], CommandResult]
RegistryFetcher = Callable[[str], RegistryDocuments]
ArchiveRunner = Callable[[DockerArchiveCommand, Path], None]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _object(raw: bytes, *, maximum: int = MAX_DESCRIPTOR_BYTES) -> dict[str, object]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise TrainingImageLockError("EVIDENCE_INVALID")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    _assert_json_depth(raw)

    def reject_constant(_value: str) -> object:
        raise ValueError("constant")

    def bounded_integer(value: str) -> int:
        if len(value.lstrip("-")) > 20:
            raise ValueError("integer")
        return int(value)

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=lambda _value: (_ for _ in ()).throw(ValueError("float")),
            parse_int=bounded_integer,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise TrainingImageLockError("EVIDENCE_INVALID") from exc
    if not isinstance(value, dict):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    return value


def _runtime_string(value: object, *, maximum: int) -> bool:
    return (
        isinstance(value, str) and 0 < len(value) <= maximum
        and all(32 <= ord(character) < 127 for character in value)
    )


def _validate_runtime_evidence(runtime: dict[str, object]) -> None:
    if set(runtime) != {"python_implementation", "python", "packages", "signatures"}:
        raise TrainingImageLockError("INSPECTOR_INVALID")
    python_implementation = runtime["python_implementation"]
    python = runtime["python"]
    packages = runtime["packages"]
    signatures = runtime["signatures"]
    if (
        python_implementation != RUNTIME_PYTHON_IMPLEMENTATION
        or python != RUNTIME_PYTHON_VERSION
        or not isinstance(packages, dict) or set(packages) != set(_RUNTIME_PACKAGES)
        or any(not _runtime_string(value, maximum=256) for value in packages.values())
        or not isinstance(signatures, dict) or set(signatures) != set(_RUNTIME_SIGNATURES)
        or any(not _runtime_string(value, maximum=8192) for value in signatures.values())
        or signatures["unsloth.import"] != _UNSLOTH_IMPORT_SENTINEL
    ):
        raise TrainingImageLockError("INSPECTOR_INVALID")


def _parse_runtime_substage_evidence(
    raw: bytes,
) -> tuple[dict[str, object] | None, str | None]:
    value = _object(raw, maximum=_RUNTIME_SUBSTAGE_MAX_BYTES)
    status = value.get("status")
    if status == "FAILED":
        runtime_substage = value.get("runtime_substage")
        if (
            len(raw) > 256
            or set(value) != {"runtime_substage", "schema_version", "status"}
            or value.get("schema_version") != _RUNTIME_SUBSTAGE_SCHEMA
            or type(runtime_substage) is not str
            or runtime_substage not in _RUNTIME_SUBSTAGE_FAILURE_BYTES
            or raw != _RUNTIME_SUBSTAGE_FAILURE_BYTES[runtime_substage]
        ):
            raise TrainingImageLockError("INSPECTOR_INVALID")
        return None, str(runtime_substage)
    if (
        status != "PASS"
        or set(value) != {"runtime", "schema_version", "status"}
        or value.get("schema_version") != _RUNTIME_SUBSTAGE_SCHEMA
        or not isinstance(value.get("runtime"), dict)
    ):
        raise TrainingImageLockError("INSPECTOR_INVALID")
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    _validate_runtime_evidence(runtime)
    try:
        canonical = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TrainingImageLockError("INSPECTOR_INVALID") from exc
    if raw != canonical:
        raise TrainingImageLockError("INSPECTOR_INVALID")
    return runtime, None


def _assert_json_depth(raw: bytes, *, maximum: int = 64) -> None:
    depth = 0
    quoted = False
    escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                quoted = False
        elif byte == 0x22:
            quoted = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > maximum:
                raise TrainingImageLockError("EVIDENCE_INVALID")
        elif byte in (0x5D, 0x7D):
            depth -= 1
            if depth < 0:
                raise TrainingImageLockError("EVIDENCE_INVALID")
    if quoted or escaped or depth:
        raise TrainingImageLockError("EVIDENCE_INVALID")


def _closed(value: Mapping[str, object], required: set[str], optional: set[str] | None = None) -> None:
    if not required <= set(value) or not set(value) <= required | (optional or set()):
        raise TrainingImageLockError("EVIDENCE_INVALID")


def parse_image_reference(reference: str) -> tuple[str, str]:
    try:
        digest = parse_registry_reference(reference)
    except OCIRegistryError as exc:
        raise TrainingImageLockError("IMAGE_INVALID") from exc
    return REGISTRY_REPOSITORY, digest


def validate_oci_documents(documents: RegistryDocuments) -> dict[str, object]:
    if _sha256(documents.requested_raw) != documents.requested_digest:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    requested = _object(documents.requested_raw)
    if requested.get("schemaVersion") != 2 or requested.get("mediaType") != documents.requested_media_type:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if documents.requested_kind == "index":
        if documents.child_raw is None:
            raise TrainingImageLockError("EVIDENCE_INVALID")
        _closed(requested, {"schemaVersion", "mediaType", "manifests"}, {"annotations"})
        manifests = requested.get("manifests")
        if not isinstance(manifests, list):
            raise TrainingImageLockError("EVIDENCE_INVALID")
        for item in manifests:
            if not isinstance(item, dict):
                raise TrainingImageLockError("EVIDENCE_INVALID")
            _closed(item, {"mediaType", "digest", "size", "platform"}, {"annotations"})
            platform = item.get("platform")
            if not isinstance(platform, dict) or not {"os", "architecture"} <= set(platform) or not set(platform) <= {"os", "architecture", "variant"}:
                raise TrainingImageLockError("EVIDENCE_INVALID")
        matches = [
            item for item in manifests
            if isinstance(item, dict)
            and item.get("platform") == {"os": "linux", "architecture": "amd64"}
        ]
        if len(matches) != 1 or matches[0] != {
            "mediaType": CHILD_MEDIA_TYPE,
            "digest": documents.child_digest,
            "size": len(documents.child_raw),
            "platform": {"os": "linux", "architecture": "amd64"},
        }:
            raise TrainingImageLockError("EVIDENCE_INVALID")
        child_raw = documents.child_raw
        index_digest: str | None = documents.requested_digest
        index_media_type: str | None = documents.requested_media_type
    elif documents.requested_kind == "manifest":
        if (
            documents.child_raw is not None
            or documents.requested_digest != documents.child_digest
            or documents.requested_media_type != documents.child_media_type
        ):
            raise TrainingImageLockError("EVIDENCE_INVALID")
        child_raw = documents.requested_raw
        index_digest = None
        index_media_type = None
    else:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if _sha256(child_raw) != documents.child_digest or documents.child_media_type != CHILD_MEDIA_TYPE:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    child = _object(child_raw)
    _closed(child, {"schemaVersion", "mediaType", "config", "layers"}, {"annotations"})
    if child.get("schemaVersion") != 2 or child.get("mediaType") != CHILD_MEDIA_TYPE:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    config = child.get("config")
    layers = child.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list) or not 1 <= len(layers) <= 256:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    _closed(config, {"mediaType", "digest", "size"})
    if config != {
        "mediaType": documents.config_media_type,
        "digest": documents.config_digest,
        "size": documents.config_size,
    } or (
        documents.config_media_type not in CONFIG_MEDIA_TYPES
        or type(documents.config_size) is not int
        or type(config.get("size")) is not int
        or not 1 <= documents.config_size <= MAX_DESCRIPTOR_BYTES
    ):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    normalized_layers: list[dict[str, object]] = []
    digests: set[str] = set()
    for value in layers:
        if not isinstance(value, dict):
            raise TrainingImageLockError("EVIDENCE_INVALID")
        _closed(value, {"mediaType", "digest", "size"})
        media, digest, size = value.get("mediaType"), value.get("digest"), value.get("size")
        if (
            media not in LAYER_MEDIA_TYPES or not isinstance(digest, str)
            or not digest.startswith("sha256:") or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
            or digest in digests or type(size) is not int or not 1 <= size <= 1099511627776
        ):
            raise TrainingImageLockError("EVIDENCE_INVALID")
        digests.add(digest)
        normalized_layers.append({"media_type": media, "digest": digest, "size": size})
    if documents.config_size + sum(int(layer["size"]) for layer in normalized_layers) > MAX_ARCHIVE_BYTES:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    return {
        "registry_repository": REGISTRY_REPOSITORY,
        "provider_repository": PROVIDER_REPOSITORY,
        "requested_digest": documents.requested_digest,
        "requested_media_type": documents.requested_media_type,
        "requested_kind": documents.requested_kind,
        "index_digest": index_digest,
        "index_media_type": index_media_type,
        "child_digest": documents.child_digest,
        "child_media_type": CHILD_MEDIA_TYPE,
        "config_digest": documents.config_digest,
        "config_media_type": documents.config_media_type,
        "config_size": documents.config_size,
        "platform": PLATFORM,
        "layers": normalized_layers,
        "provider_reference": f"{PROVIDER_REPOSITORY}@{documents.child_digest}",
    }


def _docker_environment() -> dict[str, str]:
    return {"DOCKER_CONTENT_TRUST": "1", "PATH": os.defpath}


def _validate_provider_reference(provider_reference: str) -> None:
    if provider_reference != f"{PROVIDER_REPOSITORY}@{provider_reference.rsplit('@', 1)[-1]}":
        raise TrainingImageLockError("IMAGE_INVALID")
    try:
        digest = provider_reference.rsplit("@", 1)[1]
        parse_registry_reference(f"{REGISTRY_REPOSITORY}@{digest}")
    except (IndexError, OCIRegistryError) as exc:
        raise TrainingImageLockError("IMAGE_INVALID") from exc


def build_pull_command(*, docker: Path, config_dir: Path, provider_reference: str) -> CommandSpec:
    _validate_provider_reference(provider_reference)
    return CommandSpec(
        (str(docker), "--config", str(config_dir), "--context", "default", "pull", "--quiet", "--platform", PLATFORM, provider_reference),
        _docker_environment(), WARM_PULL_TIMEOUT_SECONDS, MAX_RUNTIME_BYTES,
    )


def _runtime_substage_inspector() -> str:
    failure_lines = {
        stage: raw.decode("ascii")
        for stage, raw in _RUNTIME_SUBSTAGE_FAILURE_BYTES.items()
    }
    steps = (
        ("python_bootstrap", (
            "import json",
            "import importlib.metadata as metadata", "import importlib.util as importlib_util",
            "import inspect", "import os", "import platform", "import site", "import stat",
            "from pathlib import Path",
        )),
        ("python_runtime", (
            f"if platform.python_implementation()!={RUNTIME_PYTHON_IMPLEMENTATION!r} or platform.python_version()!={RUNTIME_PYTHON_VERSION!r}: raise RuntimeError('PYTHON_RUNTIME_INVALID')",
        )),
        ("workspace_setup", (
            "for directory in ('/tmp/home','/tmp/hf','/tmp/xdg','/tmp/torch'): os.makedirs(directory,mode=0o700,exist_ok=True)",
        )),
        ("distribution_metadata", (
            f"names={list(_RUNTIME_PACKAGES)!r}", "packages={name:metadata.version(name) for name in names}",
        )),
        ("torch_import", ("import torch",)),
        ("safetensors_import", ("from safetensors import safe_open",)),
        ("transformers_import", ("from transformers import TrainerCallback",)),
        ("signature_introspection", (
            "signatures={}",
            "signatures['TrainerCallback.on_optimizer_step']=str(inspect.signature(TrainerCallback.on_optimizer_step))",
            "signatures['safetensors.safe_open']=str(inspect.signature(safe_open))",
            "signatures['torch.load']=str(inspect.signature(torch.load))",
        )),
        ("unsloth_spec", (
            "spec=importlib_util.find_spec('unsloth')",
            "if spec is None or spec.loader is None or not isinstance(spec.origin,str) or not spec.origin: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
        )),
        ("unsloth_origin", (
            "origin=Path(spec.origin)",
            "if not origin.is_absolute() or origin.is_symlink(): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
            "resolved=origin.resolve(strict=True)",
            "if resolved != origin or not stat.S_ISREG(origin.lstat().st_mode): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
        )),
        ("unsloth_package_root", (
            "locations=tuple(spec.submodule_search_locations or ())",
            "if len(locations) != 1: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
            "package_root=Path(locations[0])",
            "if package_root.resolve(strict=True) != package_root or package_root != origin.parent: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
        )),
        ("site_roots", (
            "base_roots=tuple(dict.fromkeys(Path(value).resolve(strict=True) for value in (sys.base_prefix,sys.prefix)))",
            "site_roots=tuple(dict.fromkeys(Path(value).resolve(strict=True) for value in site.getsitepackages()))",
        )),
        ("site_membership", (
            "if not site_roots or not all(any(root == candidate or root in candidate.parents for root in base_roots) for candidate in site_roots): raise RuntimeError('UNSLOTH_SITE_INVALID')",
            "selected=next((candidate for candidate in site_roots if candidate == origin.parent or candidate in origin.parents),None)",
            "if selected is None: raise RuntimeError('UNSLOTH_SITE_INVALID')",
        )),
        ("user_site_isolation", (
            "user_site=Path(site.getusersitepackages()).resolve(strict=False)",
            "if user_site == origin or user_site in origin.parents: raise RuntimeError('UNSLOTH_SITE_INVALID')",
        )),
        ("origin_chain", (
            "cursor=origin", "while cursor != selected:",
            " if cursor.is_symlink(): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
            " cursor=cursor.parent",
            "if selected.is_symlink(): raise RuntimeError('UNSLOTH_SITE_INVALID')",
            f"signatures['unsloth.import']={_UNSLOTH_IMPORT_SENTINEL!r}",
        )),
        ("result_serialization", (
            f"result={{'runtime':{{'python_implementation':platform.python_implementation(),'python':platform.python_version(),'packages':packages,'signatures':signatures}},'schema_version':{_RUNTIME_SUBSTAGE_SCHEMA!r},'status':'PASS'}}",
            "sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))",
        )),
    )
    lines = [
        "import sys", "sys.dont_write_bytecode=True",
        f"failure_lines={failure_lines!r}", "stage='python_bootstrap'", "try:",
    ]
    for stage, statements in steps:
        lines.append(f" stage={stage!r}")
        lines.extend(" " + statement for statement in statements)
    lines.extend((
        "except Exception:",
        " sys.stdout.write(failure_lines[stage])",
    ))
    return "\n".join(lines)


def build_runtime_command(
    *, docker: Path, config_dir: Path, repository: str, child_digest: str,
    runtime_substage_attribution: bool = False,
) -> CommandSpec:
    provider_reference = f"{repository}@{child_digest}"
    if repository != PROVIDER_REPOSITORY:
        raise TrainingImageLockError("IMAGE_INVALID")
    _validate_provider_reference(provider_reference)
    inspector = _runtime_substage_inspector() if runtime_substage_attribution else "\n".join((
        "import sys",
        "sys.dont_write_bytecode=True",
        "import importlib.metadata as metadata",
        "import importlib.util as importlib_util",
        "import inspect",
        "import json",
        "import os",
        "import platform",
        "import site",
        "import stat",
        "from pathlib import Path",
        f"if platform.python_implementation()!={RUNTIME_PYTHON_IMPLEMENTATION!r} or platform.python_version()!={RUNTIME_PYTHON_VERSION!r}: raise RuntimeError('PYTHON_RUNTIME_INVALID')",
        "for directory in ('/tmp/home','/tmp/hf','/tmp/xdg','/tmp/torch'): os.makedirs(directory,mode=0o700,exist_ok=True)",
        f"names={list(_RUNTIME_PACKAGES)!r}",
        "packages={name:metadata.version(name) for name in names}",
        "import torch",
        "from safetensors import safe_open",
        "from transformers import TrainerCallback",
        "signatures={",
        " 'TrainerCallback.on_optimizer_step':str(inspect.signature(TrainerCallback.on_optimizer_step)),",
        " 'safetensors.safe_open':str(inspect.signature(safe_open)),",
        " 'torch.load':str(inspect.signature(torch.load)),",
        "}",
        "spec=importlib_util.find_spec('unsloth')",
        "if spec is None or spec.loader is None or not isinstance(spec.origin,str) or not spec.origin: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
        "origin=Path(spec.origin)",
        "if not origin.is_absolute() or origin.is_symlink(): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
        "resolved=origin.resolve(strict=True)",
        "if resolved != origin or not stat.S_ISREG(origin.lstat().st_mode): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
        "locations=tuple(spec.submodule_search_locations or ())",
        "if len(locations) != 1: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
        "package_root=Path(locations[0])",
        "if package_root.resolve(strict=True) != package_root or package_root != origin.parent: raise RuntimeError('UNSLOTH_SPEC_INVALID')",
        "base_roots=tuple(dict.fromkeys(Path(value).resolve(strict=True) for value in (sys.base_prefix,sys.prefix)))",
        "site_roots=tuple(dict.fromkeys(Path(value).resolve(strict=True) for value in site.getsitepackages()))",
        "if not site_roots or not all(any(root == candidate or root in candidate.parents for root in base_roots) for candidate in site_roots): raise RuntimeError('UNSLOTH_SITE_INVALID')",
        "selected=next((candidate for candidate in site_roots if candidate == origin.parent or candidate in origin.parents),None)",
        "if selected is None: raise RuntimeError('UNSLOTH_SITE_INVALID')",
        "user_site=Path(site.getusersitepackages()).resolve(strict=False)",
        "if user_site == origin or user_site in origin.parents: raise RuntimeError('UNSLOTH_SITE_INVALID')",
        "cursor=origin",
        "while cursor != selected:",
        " if cursor.is_symlink(): raise RuntimeError('UNSLOTH_ORIGIN_INVALID')",
        " cursor=cursor.parent",
        "if selected.is_symlink(): raise RuntimeError('UNSLOTH_SITE_INVALID')",
        f"signatures['unsloth.import']={_UNSLOTH_IMPORT_SENTINEL!r}",
        "print(json.dumps({'python_implementation':platform.python_implementation(),'python':platform.python_version(),'packages':packages,'signatures':signatures},sort_keys=True,separators=(',',':')))",
    ))
    return CommandSpec(
        (
            str(docker), "--config", str(config_dir), "--context", "default", "run", "--rm", "--pull=never",
            "--platform", PLATFORM, "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "128", "--memory", "2g",
            "--cpus", "1", "--user", "65534:65534",
            "--workdir", "/tmp",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777,uid=65534,gid=65534",
            "--env", "HOME=/tmp/home", "--env", "HF_HOME=/tmp/hf",
            "--env", "XDG_CACHE_HOME=/tmp/xdg", "--env", "TORCH_HOME=/tmp/torch",
            "--env", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
            "--env", "PYTHONNOUSERSITE=1", "--entrypoint", "python",
            provider_reference, "-I", "-c", inspector,
        ),
        _docker_environment(), RUNTIME_TIMEOUT_SECONDS,
        _RUNTIME_SUBSTAGE_MAX_BYTES if runtime_substage_attribution else MAX_RUNTIME_BYTES,
    )


def build_python_runtime_identity_command(
    *, docker: Path, config_dir: Path, repository: str, child_digest: str,
) -> CommandSpec:
    provider_reference = f"{repository}@{child_digest}"
    if repository != PROVIDER_REPOSITORY:
        raise TrainingImageLockError("IMAGE_INVALID")
    _validate_provider_reference(provider_reference)
    failure = _PYTHON_IDENTITY_FAILED_BYTES.decode("ascii")
    inspector = "\n".join((
        "import sys",
        f"failure={failure!r}",
        f"implementations={dict(_PYTHON_IMPLEMENTATIONS)!r}",
        "try:",
        " raw=sys.implementation.name",
        " implementation=implementations.get(raw.lower()) if type(raw) is str else None",
        " version_info=sys.version_info",
        " major=version_info.major",
        " minor=version_info.minor",
        " micro=version_info.micro",
        " if implementation is None: raise RuntimeError('identity')",
        " if type(major) is not int or not 1 <= major <= 999: raise RuntimeError('version')",
        " if type(minor) is not int or not 0 <= minor <= 999: raise RuntimeError('version')",
        " if type(micro) is not int or not 0 <= micro <= 999: raise RuntimeError('version')",
        " if version_info.releaselevel != 'final': raise RuntimeError('version')",
        " if type(version_info.serial) is not int or version_info.serial != 0: raise RuntimeError('version')",
        " version=f'{major}.{minor}.{micro}'",
        f" sys.stdout.write('{{\"implementation\":\"'+implementation+'\",\"schema_version\":\"{_PYTHON_IDENTITY_SCHEMA}\",\"status\":\"OBSERVED\",\"version\":\"'+version+'\"}}')",
        "except Exception:",
        " sys.stdout.write(failure)",
    ))
    return CommandSpec(
        (
            str(docker), "--config", str(config_dir), "--context", "default", "run", "--rm", "--pull=never",
            "--platform", PLATFORM, "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "128", "--memory", "2g",
            "--cpus", "1", "--user", "65534:65534", "--workdir", "/tmp",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777,uid=65534,gid=65534",
            "--env", "HOME=/tmp/home", "--env", "HF_HOME=/tmp/hf",
            "--env", "XDG_CACHE_HOME=/tmp/xdg", "--env", "TORCH_HOME=/tmp/torch",
            "--env", "HF_HUB_DISABLE_IMPLICIT_TOKEN=1", "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
            "--env", "PYTHONNOUSERSITE=1", "--entrypoint", "python",
            provider_reference, "-I", "-c", inspector,
        ),
        _docker_environment(), RUNTIME_TIMEOUT_SECONDS, 256,
    )


def _parse_python_runtime_identity(raw: bytes) -> dict[str, str] | None:
    if raw == _PYTHON_IDENTITY_FAILED_BYTES:
        return None
    value = _object(raw, maximum=256)
    if (
        set(value) != {"implementation", "schema_version", "status", "version"}
        or value.get("schema_version") != _PYTHON_IDENTITY_SCHEMA
        or value.get("status") != "OBSERVED"
        or value.get("implementation") not in set(_PYTHON_IMPLEMENTATIONS.values())
        or type(value.get("version")) is not str
        or _PYTHON_VERSION.fullmatch(value["version"]) is None
    ):
        raise TrainingImageLockError("INSPECTOR_INVALID")
    try:
        canonical = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TrainingImageLockError("INSPECTOR_INVALID") from exc
    if raw != canonical:
        raise TrainingImageLockError("INSPECTOR_INVALID")
    return {
        "implementation": str(value["implementation"]),
        "schema_version": _PYTHON_IDENTITY_SCHEMA,
        "status": "OBSERVED",
        "version": str(value["version"]),
    }


def build_inspect_command(
    *, docker: Path, config_dir: Path, provider_reference: str,
) -> CommandSpec:
    _validate_provider_reference(provider_reference)
    return CommandSpec(
        (
            str(docker), "--config", str(config_dir), "--context", "default", "image", "inspect",
            "--format", INSPECT_FORMAT, provider_reference,
        ),
        _docker_environment(), DEFAULT_TIMEOUT_SECONDS, MAX_RUNTIME_BYTES,
    )


def build_save_command(
    *, docker: Path, config_dir: Path, provider_reference: str,
) -> DockerArchiveCommand:
    _validate_provider_reference(provider_reference)
    return DockerArchiveCommand(
        (
            str(docker), "--config", str(config_dir), "--context", "default", "image", "save",
            "--platform", PLATFORM, provider_reference,
        ),
        _docker_environment(), SAVE_ARCHIVE_TIMEOUT_SECONDS,
    )


def build_version_command(*, docker: Path, config_dir: Path) -> CommandSpec:
    return CommandSpec(
        (str(docker), "--config", str(config_dir), "--context", "default", "version", "--format", VERSION_FORMAT),
        _docker_environment(), DEFAULT_TIMEOUT_SECONDS, MAX_RUNTIME_BYTES,
    )


def build_info_command(*, docker: Path, config_dir: Path) -> CommandSpec:
    return CommandSpec(
        (str(docker), "--config", str(config_dir), "--context", "default", "info", "--format", INFO_FORMAT),
        _docker_environment(), DEFAULT_TIMEOUT_SECONDS, MAX_RUNTIME_BYTES,
    )


def build_context_command(*, docker: Path, config_dir: Path) -> CommandSpec:
    return CommandSpec(
        (str(docker), "--config", str(config_dir), "context", "inspect", "default", "--format", CONTEXT_FORMAT),
        _docker_environment(), DEFAULT_TIMEOUT_SECONDS, MAX_RUNTIME_BYTES,
    )
def _popen_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt" and getattr(process, "pid", None):
            completed = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False, env={"PATH": os.defpath},
            )
            if completed.returncode != 0:
                process.kill()
        elif getattr(process, "pid", None):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def subprocess_runner(spec: CommandSpec) -> CommandResult:
    process: subprocess.Popen[bytes] | None = None
    pipes: list[object] = []
    reader_candidates: list[threading.Thread] = []
    cleanup_done = False

    def owned_pipes() -> tuple[list[object], bool]:
        observed: list[object] = []
        complete = True
        for stream in pipes:
            if stream is not None and all(stream is not item for item in observed):
                observed.append(stream)
        if process is not None:
            for attribute in ("stdout", "stderr"):
                try:
                    stream = getattr(process, attribute)
                except BaseException:
                    complete = False
                    continue
                if stream is not None and all(stream is not item for item in observed):
                    observed.append(stream)
        return observed, complete

    def close_pipes(*, suppress: bool) -> bool:
        streams, closed = owned_pipes()
        for stream in streams:
            try:
                stream.close()  # type: ignore[attr-defined]
            except BaseException:
                if not suppress:
                    raise
                closed = False
        return closed

    def proven_started_readers() -> tuple[list[threading.Thread], bool]:
        observed: list[threading.Thread] = []
        complete = True
        for thread in reader_candidates:
            started = False
            try:
                started = thread.is_alive()
            except BaseException:
                complete = False
            try:
                started = started or getattr(thread, "ident", None) is not None
            except BaseException:
                complete = False
            if started and all(thread is not item for item in observed):
                observed.append(thread)
        return observed, complete

    def join_started_readers() -> bool:
        threads, joined = proven_started_readers()
        for thread in threads:
            try:
                thread.join(timeout=5)
            except BaseException:
                joined = False
        threads, complete = proven_started_readers()
        joined = joined and complete
        for thread in threads:
            try:
                if thread.is_alive():
                    joined = False
            except BaseException:
                joined = False
        return joined

    def cancel_owned_process() -> bool:
        if process is None:
            return False
        process_gone = False
        try:
            _terminate_process_tree(process)
        except BaseException:
            pass
        try:
            process.wait(timeout=10)
            process_gone = True
        except BaseException:
            try:
                _terminate_process_tree(process)
            except BaseException:
                pass
            try:
                process_gone = process.poll() is not None
            except BaseException:
                process_gone = False
        pipes_closed = close_pipes(suppress=True)
        readers_gone = join_started_readers()
        if not readers_gone:
            try:
                _terminate_process_tree(process)
            except BaseException:
                pass
            pipes_closed = close_pipes(suppress=True) and pipes_closed
            readers_gone = join_started_readers()
        return process_gone and pipes_closed and readers_gone

    try:
        process = subprocess.Popen(
            list(spec.argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=dict(spec.env), close_fds=True, **_popen_group_kwargs(),
        )
        pipes = [process.stdout, process.stderr]
        if any(stream is None for stream in pipes):
            raise RuntimeError("missing owned pipe")
        overflow = threading.Event()
        reader_failure = threading.Event()

        def consume(stream: object) -> bytes:
            content = bytearray()
            try:
                while True:
                    chunk = stream.read(65536)  # type: ignore[attr-defined]
                    if not chunk:
                        break
                    content.extend(chunk[: spec.maximum_output_bytes + 1 - len(content)])
                    if len(content) > spec.maximum_output_bytes:
                        overflow.set()
                        _terminate_process_tree(process)
                        break
            except (OSError, ValueError):
                reader_failure.set()
                _terminate_process_tree(process)
            return bytes(content)

        results: list[bytes | None] = [None, None]

        def store(index: int, stream: object) -> None:
            results[index] = consume(stream)

        for index, stream in enumerate(pipes):
            reader_candidates.append(
                threading.Thread(target=store, args=(index, stream), daemon=True),
            )
        for thread in reader_candidates:
            thread.start()
        try:
            code = process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            cleanup_done = True
            if not cancel_owned_process():
                raise TrainingImageLockError(
                    "COMMAND_FAILED", diagnostic_category="cleanup",
                ) from None
            raise TrainingImageLockError("OPERATION_TIMEOUT") from exc
        for thread in reader_candidates:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in reader_candidates):
            _terminate_process_tree(process)
            close_pipes(suppress=False)
            for thread in reader_candidates:
                thread.join(timeout=5)
        pipes_closed = close_pipes(suppress=False)
        cleanup_done = True
        readers_alive = any(thread.is_alive() for thread in reader_candidates)
        if not pipes_closed or readers_alive:
            raise TrainingImageLockError(
                "COMMAND_FAILED", diagnostic_category="cleanup",
            )
        if code:
            raise TrainingImageLockError(
                "COMMAND_FAILED", diagnostic_category="nonzero",
            )
        if overflow.is_set() or reader_failure.is_set():
            raise TrainingImageLockError("COMMAND_FAILED")
        return CommandResult(results[0] or b"", results[1] or b"")
    except BaseException as exc:
        if process is None:
            if isinstance(exc, OSError):
                raise TrainingImageLockError("COMMAND_FAILED") from exc
            raise
        cleanup_succeeded = True if cleanup_done else cancel_owned_process()
        if not cleanup_succeeded:
            raise TrainingImageLockError(
                "COMMAND_FAILED", diagnostic_category="cleanup",
            ) from None
        if isinstance(exc, Exception):
            if isinstance(exc, TrainingImageLockError):
                raise
            raise TrainingImageLockError(
                "COMMAND_FAILED", diagnostic_category="cleanup",
            ) from exc
        raise


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb", buffering=0) as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise TrainingImageLockError("IMAGE_INVALID") from exc
    return "sha256:" + digest.hexdigest()


def _docker_identity(path: Path) -> DockerExecutableIdentity:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TrainingImageLockError("IMAGE_INVALID") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not resolved.is_absolute():
        raise TrainingImageLockError("IMAGE_INVALID")
    before = resolved.stat()
    digest = _hash_file(resolved)
    after = resolved.stat()
    fields_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    fields_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if fields_before != fields_after or resolved.resolve(strict=True) != resolved:
        raise TrainingImageLockError("IMAGE_INVALID")
    return DockerExecutableIdentity(
        str(resolved), digest, int(after.st_dev), int(after.st_ino),
        int(after.st_size), int(after.st_mtime_ns),
    )


def _validated_docker(path: Path) -> Path:
    return Path(_docker_identity(path).path)


def _assert_docker_identity(expected: DockerExecutableIdentity) -> None:
    if _docker_identity(Path(expected.path)) != expected:
        raise TrainingImageLockError("IMAGE_INVALID")


def _run_authenticated(
    runner: Runner, spec: CommandSpec, identity: DockerExecutableIdentity,
) -> CommandResult:
    _assert_docker_identity(identity)
    result = runner(spec)
    _assert_docker_identity(identity)
    return result


def _docker_authority(
    *, runner: Runner, docker: Path, config: Path,
    executable: DockerExecutableIdentity,
) -> dict[str, object]:
    _empty_docker_config(config)
    version = _object(
        _run_authenticated(runner, build_version_command(docker=docker, config_dir=config), executable).stdout,
        maximum=MAX_RUNTIME_BYTES,
    )
    _empty_docker_config(config)
    info = _object(
        _run_authenticated(runner, build_info_command(docker=docker, config_dir=config), executable).stdout,
        maximum=MAX_RUNTIME_BYTES,
    )
    _empty_docker_config(config)
    context = _object(
        _run_authenticated(runner, build_context_command(docker=docker, config_dir=config), executable).stdout,
        maximum=MAX_RUNTIME_BYTES,
    )
    _empty_docker_config(config)
    if set(version) != {"ClientVersion", "ServerVersion"}:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if set(info) != {"ID", "ServerVersion", "OSType", "Architecture", "Name", "DockerRootDir", "Driver", "SecurityOptions"}:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if set(context) != {"Name", "DockerEndpoint", "SkipTLSVerify"}:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    scalar_values = [version["ClientVersion"], version["ServerVersion"], info["ID"], info["ServerVersion"], info["Name"], info["DockerRootDir"], info["Driver"], context["DockerEndpoint"]]
    if any(not isinstance(value, str) or not value or len(value) > 1024 or any(ord(char) < 32 for char in value) for value in scalar_values):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if (
        version["ServerVersion"] != info["ServerVersion"]
        or info["OSType"] != "linux" or info["Architecture"] not in {"x86_64", "amd64"}
        or context["Name"] != "default" or context["SkipTLSVerify"] is not False
        or not str(context["DockerEndpoint"]).startswith(("npipe://", "unix://"))
        or not isinstance(info["SecurityOptions"], list)
        or any(not isinstance(item, str) or len(item) > 1024 for item in info["SecurityOptions"])
    ):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    return {
        "executable": {
            "path": executable.path, "sha256": executable.sha256,
            "device": executable.device, "inode": executable.inode,
            "size": executable.size, "mtime_ns": executable.mtime_ns,
        },
        "version": version, "daemon": info, "context": context,
    }


def _empty_docker_config(path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise TrainingImageLockError("DOCKER_CONFIG_INVALID")
    except OSError as exc:
        raise TrainingImageLockError("DOCKER_CONFIG_INVALID") from exc


def _fresh_output(path: Path) -> Path:
    absolute = path.resolve(strict=False)
    if not absolute.name.endswith(".candidate.json") or path.exists() or path.is_symlink():
        raise TrainingImageLockError("OUTPUT_INVALID")
    try:
        parent = absolute.parent.resolve(strict=True)
        info = parent.lstat()
        repository = _repository_root().resolve(strict=True)
    except OSError as exc:
        raise TrainingImageLockError("OUTPUT_INVALID") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or parent == repository or repository in parent.parents:
        raise TrainingImageLockError("OUTPUT_INVALID")
    return absolute


def _archive_path(destination: Path) -> Path:
    archive = destination.with_name(destination.name + ".docker-save.tar.tmp")
    if archive.exists() or archive.is_symlink():
        raise TrainingImageLockError("OUTPUT_INVALID")
    return archive


@dataclass(frozen=True)
class _LocalStoreIdentity:
    mode: str
    image_id: str
    repo_digests: tuple[str, ...]
    platform: str
    rootfs_type: str
    diff_ids: tuple[str, ...]


def _inspect_identity(raw: bytes, *, identity: Mapping[str, object]) -> _LocalStoreIdentity:
    value = _object(raw, maximum=MAX_RUNTIME_BYTES)
    if set(value) != {"Id", "RepoDigests", "Os", "Architecture", "RootFS"}:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    image_id = value.get("Id")
    if image_id == identity["config_digest"]:
        mode = "CONFIG_ID"
    elif image_id == identity["child_digest"]:
        mode = "MANIFEST_TARGET_ID"
    else:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    repo_digests = value.get("RepoDigests")
    expected_reference = identity["provider_reference"]
    if (
        not isinstance(repo_digests, list) or len(repo_digests) != 1
        or not isinstance(repo_digests[0], str)
        or len(repo_digests[0]) > 2048
        or repo_digests[0] != expected_reference
    ):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    if value.get("Os") != "linux" or value.get("Architecture") != "amd64":
        raise TrainingImageLockError("EVIDENCE_INVALID")
    rootfs = value.get("RootFS")
    if not isinstance(rootfs, dict) or set(rootfs) != {"Type", "Layers"}:
        raise TrainingImageLockError("EVIDENCE_INVALID")
    layers = rootfs.get("Layers")
    if (
        rootfs.get("Type") != "layers" or not isinstance(layers, list)
        or len(layers) != len(identity["layers"])
        or any(
            not isinstance(digest, str) or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
            for digest in layers
        )
    ):
        raise TrainingImageLockError("EVIDENCE_INVALID")
    return _LocalStoreIdentity(
        mode=mode,
        image_id=image_id,
        repo_digests=(repo_digests[0],),
        platform=PLATFORM,
        rootfs_type="layers",
        diff_ids=tuple(layers),
    )


@dataclass(frozen=True)
class _OperationDeadline:
    expires_at: float

    @classmethod
    def start(cls, seconds: int) -> "_OperationDeadline":
        return cls(time.monotonic() + seconds)

    def remaining_seconds(self, phase_limit: int) -> int:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise TrainingImageLockError("OPERATION_TIMEOUT")
        return max(1, min(phase_limit, math.ceil(remaining)))

    def check(self) -> None:
        if time.monotonic() >= self.expires_at:
            raise TrainingImageLockError("OPERATION_TIMEOUT")


def _deadline_runner(runner: Runner, deadline: _OperationDeadline) -> Runner:
    def run(spec: CommandSpec) -> CommandResult:
        bounded = replace(
            spec,
            timeout_seconds=deadline.remaining_seconds(spec.timeout_seconds),
        )
        result = runner(bounded)
        deadline.check()
        return result
    return run


class _NoRegistryRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


@dataclass
class _RegistryActiveBudget:
    active_seconds: float = 0.0
    phase_started_at: float | None = None

    def remaining_seconds(self) -> float:
        in_flight = (
            max(0.0, time.monotonic() - self.phase_started_at)
            if self.phase_started_at is not None else 0.0
        )
        remaining = REGISTRY_AGGREGATE_TIMEOUT_SECONDS - self.active_seconds - in_flight
        if remaining <= 0:
            raise TrainingImageLockError("OPERATION_TIMEOUT")
        return remaining

    def request_timeout_seconds(self) -> float:
        return min(REGISTRY_REQUEST_TIMEOUT_SECONDS, self.remaining_seconds())

    def transport(self, request: HTTPRequest) -> HTTPResponse:
        timeout = self.request_timeout_seconds()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            _NoRegistryRedirect(),
        )
        req = urllib.request.Request(
            request.url, headers=dict(request.headers), method="GET",
        )
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        except (OSError, urllib.error.URLError) as exc:
            raise OCIRegistryError("HTTP_INVALID") from exc
        try:
            raw = response.read(request.maximum_bytes + 1)
            headers: dict[str, str] = {}
            for key in response.headers:
                values = response.headers.get_all(key) or []
                lowered = key.lower()
                if lowered in headers or len(values) != 1:
                    raise OCIRegistryError("HTTP_INVALID")
                headers[lowered] = values[0]
            return HTTPResponse(
                int(response.status), str(response.geturl()), headers, raw,
            )
        except (OSError, ValueError) as exc:
            raise OCIRegistryError("HTTP_INVALID") from exc
        finally:
            response.close()

    def fetch(self, image: str, registry_fetcher: RegistryFetcher) -> RegistryDocuments:
        self.remaining_seconds()
        self.phase_started_at = time.monotonic()
        failure: Exception | None = None
        documents: RegistryDocuments | None = None
        try:
            if registry_fetcher is fetch_registry_documents:
                documents = fetch_registry_documents(image, transport=self.transport)
            else:
                documents = registry_fetcher(image)
        except Exception as exc:
            failure = exc
        finally:
            if self.phase_started_at is not None:
                self.active_seconds += max(0.0, time.monotonic() - self.phase_started_at)
                self.phase_started_at = None
        self.remaining_seconds()
        if failure is not None:
            raise failure
        if documents is None:
            raise TrainingImageLockError("EVIDENCE_INVALID")
        return documents


def _registry_identity(
    image: str, *, registry_fetcher: RegistryFetcher,
    deadline: _OperationDeadline, registry_budget: _RegistryActiveBudget,
) -> tuple[RegistryDocuments, dict[str, object]]:
    deadline.check()
    try:
        documents = registry_budget.fetch(image, registry_fetcher)
    except OCIRegistryError as exc:
        raise TrainingImageLockError("EVIDENCE_INVALID") from exc
    deadline.check()
    return documents, validate_oci_documents(documents)


def warm_image_cache(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
) -> dict[str, object]:
    deadline = _OperationDeadline.start(WARM_OVERALL_TIMEOUT_SECONDS)
    parse_image_reference(image)
    executable = _docker_identity(docker)
    docker = Path(executable.path)
    _empty_docker_config(docker_config)
    config = docker_config.resolve(strict=True)
    registry_budget = _RegistryActiveBudget()
    documents, identity = _registry_identity(
        image, registry_fetcher=registry_fetcher, deadline=deadline,
        registry_budget=registry_budget,
    )
    del documents
    provider_reference = str(identity["provider_reference"])
    bounded_runner = _deadline_runner(runner, deadline)
    try:
        with image_operation_lock(
            str(identity["registry_repository"]), str(identity["child_digest"]),
        ) as operation_key:
            authority_before = _docker_authority(
                runner=bounded_runner, docker=docker, config=config,
                executable=executable,
            )
            _run_authenticated(
                bounded_runner,
                build_pull_command(
                    docker=docker, config_dir=config,
                    provider_reference=provider_reference,
                ),
                executable,
            )
            _empty_docker_config(config)
            inspect_result = _run_authenticated(
                bounded_runner,
                build_inspect_command(
                    docker=docker, config_dir=config,
                    provider_reference=provider_reference,
                ),
                executable,
            )
            _empty_docker_config(config)
            _inspect_identity(inspect_result.stdout, identity=identity)
            authority_after = _docker_authority(
                runner=bounded_runner, docker=docker, config=config,
                executable=executable,
            )
            if authority_after != authority_before:
                raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
            deadline.check()
    except ImageOperationLockError as exc:
        raise TrainingImageLockError(exc.reason_code) from exc
    return {
        "status": "CACHE_WARMED", "operation_key": operation_key,
        "child_digest": identity["child_digest"],
    }


def _metadata_diagnostic_category(
    stage: str, error: TrainingImageLockError,
) -> str:
    if error.diagnostic_category is not None:
        return error.diagnostic_category
    if error.reason_code in {"OPERATION_TIMEOUT", "OPERATION_LOCK_TIMEOUT"}:
        return "timeout"
    if error.reason_code == "OPERATION_LOCK_CLEANUP_FAILED":
        return "cleanup"
    if stage == "runtime_metadata":
        return "runtime"
    if stage in {"registry_initial", "registry_final"}:
        return "document"
    return "identity"


def _diagnose_runtime_metadata(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher, attributed: bool,
    runtime_substage_attribution: bool, python_runtime_identity: bool,
) -> dict[str, str]:
    stage = "preflight"
    attributed_error: MetadataDiagnosticStageError | None = None
    runtime_substage_error: RuntimeSubstageDiagnosticError | None = None
    python_identity_error: PythonRuntimeIdentityDiagnosticError | None = None
    python_identity_result: dict[str, str] | None = None
    try:
        if platform.python_implementation() != "CPython" or platform.python_version() != "3.12.7":
            raise TrainingImageLockError("INSPECTOR_INVALID")
        parse_image_reference(image)
        executable = _docker_identity(docker)
        docker = Path(executable.path)
        _empty_docker_config(docker_config)
        config = docker_config.resolve(strict=True)
        deadline = _OperationDeadline.start(DIAGNOSTIC_OVERALL_TIMEOUT_SECONDS)
        registry_budget = _RegistryActiveBudget()
        stage = "registry_initial"
        documents, identity = _registry_identity(
            image, registry_fetcher=registry_fetcher, deadline=deadline,
            registry_budget=registry_budget,
        )
        bounded_runner = _deadline_runner(runner, deadline)
        stage = "operation_lock"
        with image_operation_lock(
            str(identity["registry_repository"]), str(identity["child_digest"]),
        ):
            stage = "docker_authority_initial"
            authority_before = _docker_authority(
                runner=bounded_runner, docker=docker, config=config,
                executable=executable,
            )
            stage = "cache_identity_initial"
            inspect_before = _run_authenticated(
                bounded_runner,
                build_inspect_command(
                    docker=docker, config_dir=config,
                    provider_reference=str(identity["provider_reference"]),
                ),
                executable,
            )
            cached_identity = _inspect_identity(inspect_before.stdout, identity=identity)
            stage = "runtime_metadata"
            try:
                runtime_result = _run_authenticated(
                    bounded_runner,
                    (
                        build_python_runtime_identity_command(
                            docker=docker, config_dir=config,
                            repository=PROVIDER_REPOSITORY,
                            child_digest=str(identity["child_digest"]),
                        )
                        if python_runtime_identity else
                        build_runtime_command(
                            docker=docker, config_dir=config,
                            repository=PROVIDER_REPOSITORY,
                            child_digest=str(identity["child_digest"]),
                            runtime_substage_attribution=runtime_substage_attribution,
                        )
                    ),
                    executable,
                )
            except TrainingImageLockError as exc:
                if (
                    (runtime_substage_attribution or python_runtime_identity)
                    and exc.reason_code == "COMMAND_FAILED"
                    and exc.diagnostic_category == "nonzero"
                ):
                    if python_runtime_identity:
                        python_identity_error = PythonRuntimeIdentityDiagnosticError()
                    else:
                        runtime_substage_error = RuntimeSubstageDiagnosticError(
                            runtime_substage="child_unreported",
                        )
                else:
                    raise
            else:
                if python_runtime_identity:
                    try:
                        python_identity_result = _parse_python_runtime_identity(
                            runtime_result.stdout,
                        )
                    except TrainingImageLockError:
                        python_identity_result = None
                    if python_identity_result is None:
                        python_identity_error = PythonRuntimeIdentityDiagnosticError()
                elif runtime_substage_attribution:
                    try:
                        runtime, failed_substage = _parse_runtime_substage_evidence(
                            runtime_result.stdout,
                        )
                    except TrainingImageLockError:
                        runtime = None
                        failed_substage = "child_unreported"
                    if failed_substage is not None:
                        runtime_substage_error = RuntimeSubstageDiagnosticError(
                            runtime_substage=failed_substage,
                        )
                    del runtime
                else:
                    try:
                        runtime = _object(runtime_result.stdout, maximum=MAX_RUNTIME_BYTES)
                    except TrainingImageLockError as exc:
                        raise TrainingImageLockError("INSPECTOR_INVALID") from exc
                    _validate_runtime_evidence(runtime)
            stage = "cache_identity_final"
            inspect_after = _run_authenticated(
                bounded_runner,
                build_inspect_command(
                    docker=docker, config_dir=config,
                    provider_reference=str(identity["provider_reference"]),
                ),
                executable,
            )
            if _inspect_identity(inspect_after.stdout, identity=identity) != cached_identity:
                raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
            stage = "docker_authority_final"
            authority_after = _docker_authority(
                runner=bounded_runner, docker=docker, config=config,
                executable=executable,
            )
            if authority_after != authority_before:
                raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
            stage = "registry_final"
            final_documents, final_identity = _registry_identity(
                image, registry_fetcher=registry_fetcher, deadline=deadline,
                registry_budget=registry_budget,
            )
            if (
                final_identity != identity
                or final_documents.requested_raw != documents.requested_raw
                or final_documents.child_raw != documents.child_raw
            ):
                raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
            stage = "final_integrity"
            _empty_docker_config(config)
            _assert_docker_identity(executable)
            deadline.check()
    except ImageOperationLockError as exc:
        error = TrainingImageLockError(
            exc.reason_code,
            diagnostic_category=(
                "cleanup" if exc.reason_code == "OPERATION_LOCK_CLEANUP_FAILED" else None
            ),
        )
        if attributed:
            attributed_error = MetadataDiagnosticStageError(
                failed_stage="operation_lock",
                category=_metadata_diagnostic_category("operation_lock", error),
            )
        else:
            raise error from exc
    except TrainingImageLockError as exc:
        if attributed:
            attributed_error = MetadataDiagnosticStageError(
                failed_stage=stage, category=_metadata_diagnostic_category(stage, exc),
            )
        else:
            raise
    if attributed_error is not None:
        raise attributed_error
    if runtime_substage_error is not None:
        raise runtime_substage_error
    if python_identity_error is not None:
        raise python_identity_error
    if python_identity_result is not None:
        return python_identity_result
    return {
        "schema_version": "synaptic-hf-training-image-metadata-diagnostic/v1",
        "status": "PASS",
    }


def diagnose_runtime_metadata(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
) -> dict[str, str]:
    """Reauthenticate one cached image and execute its metadata-only runtime probe once."""

    return _diagnose_runtime_metadata(
        image=image, docker=docker, docker_config=docker_config, runner=runner,
        registry_fetcher=registry_fetcher, attributed=False,
        runtime_substage_attribution=False, python_runtime_identity=False,
    )


def diagnose_runtime_metadata_attributed(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
) -> dict[str, str]:
    """Run the metadata diagnostic with closed first-failure stage attribution."""

    return _diagnose_runtime_metadata(
        image=image, docker=docker, docker_config=docker_config, runner=runner,
        registry_fetcher=registry_fetcher, attributed=True,
        runtime_substage_attribution=False, python_runtime_identity=False,
    )


def diagnose_runtime_substage_attributed(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
) -> dict[str, str]:
    """Run the metadata diagnostic with closed runtime-substage attribution."""

    return _diagnose_runtime_metadata(
        image=image, docker=docker, docker_config=docker_config, runner=runner,
        registry_fetcher=registry_fetcher, attributed=True,
        runtime_substage_attribution=True, python_runtime_identity=False,
    )


def observe_python_runtime_identity(
    *, image: str, docker: Path, docker_config: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
) -> dict[str, str]:
    """Observe the child Python identity after all authenticated metadata guards."""

    return _diagnose_runtime_metadata(
        image=image, docker=docker, docker_config=docker_config, runner=runner,
        registry_fetcher=registry_fetcher, attributed=True,
        runtime_substage_attribution=False, python_runtime_identity=True,
    )


def capture_candidate(
    *, image: str, docker: Path, docker_config: Path, output: Path, runner: Runner,
    registry_fetcher: RegistryFetcher = fetch_registry_documents,
    archive_runner: ArchiveRunner = save_docker_archive,
) -> dict[str, object]:
    deadline = _OperationDeadline.start(CAPTURE_OVERALL_TIMEOUT_SECONDS)
    parse_image_reference(image)
    executable = _docker_identity(docker)
    docker = Path(executable.path)
    _empty_docker_config(docker_config)
    config = docker_config.resolve(strict=True)
    destination = _fresh_output(output)
    registry_budget = _RegistryActiveBudget()
    documents, identity = _registry_identity(
        image, registry_fetcher=registry_fetcher, deadline=deadline,
        registry_budget=registry_budget,
    )
    provider_reference = str(identity["provider_reference"])
    archive = _archive_path(destination)
    bounded_runner = _deadline_runner(runner, deadline)
    try:
        with image_operation_lock(
            str(identity["registry_repository"]), str(identity["child_digest"]),
        ):
            archive_owned = False
            try:
                _empty_docker_config(config)
                authority_before = _docker_authority(
                    runner=bounded_runner, docker=docker, config=config,
                    executable=executable,
                )
                try:
                    inspect_result = _run_authenticated(
                        bounded_runner,
                        build_inspect_command(
                            docker=docker, config_dir=config,
                            provider_reference=provider_reference,
                        ),
                        executable,
                    )
                except TrainingImageLockError as exc:
                    if exc.reason_code == "COMMAND_FAILED":
                        raise TrainingImageLockError("CACHE_IDENTITY_INVALID") from exc
                    raise
                _empty_docker_config(config)
                inspected_identity = _inspect_identity(
                    inspect_result.stdout, identity=identity,
                )
                _empty_docker_config(config)
                _assert_docker_identity(executable)
                save_command = build_save_command(
                    docker=docker, config_dir=config,
                    provider_reference=provider_reference,
                )
                save_command = replace(
                    save_command,
                    timeout_seconds=deadline.remaining_seconds(
                        SAVE_ARCHIVE_TIMEOUT_SECONDS,
                    ),
                )
                archive_runner(save_command, archive)
                archive_owned = True
                deadline.check()
                _assert_docker_identity(executable)
                _empty_docker_config(config)
                archive_evidence = inspect_docker_archive(
                    archive,
                    expected_config_digest=str(identity["config_digest"]),
                    expected_config_size=int(identity["config_size"]),
                    expected_layer_count=len(identity["layers"]),
                    expected_child_digest=str(identity["child_digest"]),
                    expected_child_raw=documents.child_raw or documents.requested_raw,
                    expected_child_media_type=str(identity["child_media_type"]),
                    expected_layers=tuple(identity["layers"]),
                    expected_provider_repository=str(identity["provider_repository"]),
                    timeout_seconds=deadline.remaining_seconds(
                        SAVE_ARCHIVE_TIMEOUT_SECONDS,
                    ),
                )
                deadline.check()
                if archive_evidence.diff_ids != inspected_identity.diff_ids:
                    raise TrainingImageLockError("EVIDENCE_INVALID")
                runtime_result = _run_authenticated(
                    bounded_runner,
                    build_runtime_command(
                        docker=docker, config_dir=config,
                        repository=PROVIDER_REPOSITORY,
                        child_digest=str(identity["child_digest"]),
                    ),
                    executable,
                )
                try:
                    runtime = _object(
                        runtime_result.stdout, maximum=MAX_RUNTIME_BYTES,
                    )
                except TrainingImageLockError as exc:
                    raise TrainingImageLockError("INSPECTOR_INVALID") from exc
                _validate_runtime_evidence(runtime)
                final_inspect = _run_authenticated(
                    bounded_runner,
                    build_inspect_command(
                        docker=docker, config_dir=config,
                        provider_reference=provider_reference,
                    ),
                    executable,
                )
                if _inspect_identity(
                    final_inspect.stdout, identity=identity,
                ) != inspected_identity:
                    raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
                authority_after = _docker_authority(
                    runner=bounded_runner, docker=docker, config=config,
                    executable=executable,
                )
                if authority_after != authority_before:
                    raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
                final_documents, final_identity = _registry_identity(
                    image, registry_fetcher=registry_fetcher, deadline=deadline,
                    registry_budget=registry_budget,
                )
                if (
                    final_identity != identity
                    or final_documents.requested_raw != documents.requested_raw
                    or final_documents.child_raw != documents.child_raw
                ):
                    raise TrainingImageLockError("CACHE_IDENTITY_INVALID")
                deadline.check()
                candidate = {
                    "schema_version": CANDIDATE_SCHEMA,
                    "review_status": "CANDIDATE_ONLY",
                    "image": identity,
                    "runtime": runtime,
                    "anonymous_loading": {
                        "token": False, "trust_remote_code": False,
                        "use_safetensors": True,
                    },
                    "capture": {
                        "requested_bytes_sha256": _sha256(documents.requested_raw),
                        "child_bytes_sha256": _sha256(
                            documents.child_raw or documents.requested_raw,
                        ),
                        "config_bytes_sha256": archive_evidence.config_sha256,
                        "archive_format": archive_evidence.archive_format,
                        "compatibility_manifest_sha256": archive_evidence.compatibility_manifest_sha256,
                        "index_source_annotation_sha256": archive_evidence.index_source_annotation_sha256,
                        "ordered_layer_diff_ids": list(
                            archive_evidence.observed_layer_diff_ids,
                        ),
                        "local_store_identity": {
                            "mode": inspected_identity.mode,
                            "image_id": inspected_identity.image_id,
                            "repo_digests": list(inspected_identity.repo_digests),
                            "platform": inspected_identity.platform,
                            "rootfs_type": inspected_identity.rootfs_type,
                            "ordered_layer_diff_ids": list(inspected_identity.diff_ids),
                        },
                        "runtime_bytes_sha256": _sha256(runtime_result.stdout),
                        "docker_authority": authority_before,
                    },
                }
            finally:
                cleanup_started = time.monotonic()
                try:
                    if archive_owned:
                        archive.unlink(missing_ok=True)
                except OSError as exc:
                    raise TrainingImageLockError("OUTPUT_INVALID") from exc
                if time.monotonic() - cleanup_started > CLEANUP_TIMEOUT_SECONDS:
                    raise TrainingImageLockError("OPERATION_TIMEOUT")
        deadline.check()
        _write_candidate(destination, _canonical_bytes(candidate))
    except ImageOperationLockError as exc:
        raise TrainingImageLockError(exc.reason_code) from exc
    except (TrainingImageLockError, DockerArchiveError) as exc:
        if isinstance(exc, DockerArchiveError):
            if str(exc) == "ARCHIVE_OUTPUT_INVALID":
                raise TrainingImageLockError("OUTPUT_INVALID") from exc
            if str(exc) == "ARCHIVE_COMMAND_FAILED":
                raise TrainingImageLockError("COMMAND_FAILED") from exc
            if str(exc) == "ARCHIVE_TIMEOUT":
                raise TrainingImageLockError("OPERATION_TIMEOUT") from exc
            raise TrainingImageLockError("EVIDENCE_INVALID") from exc
        raise
    except Exception as exc:
        raise TrainingImageLockError("COMMAND_FAILED") from exc
    return candidate


def _write_candidate(destination: Path, payload: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if not isinstance(written, int) or written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if destination.read_bytes() != payload or _sha256(destination.read_bytes()) != _sha256(payload):
            raise OSError("readback mismatch")
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise TrainingImageLockError("OUTPUT_INVALID") from exc


def canonical_runtime_lock_from_candidate(*_args: object, **_kwargs: object) -> None:
    raise TrainingImageLockError("PROMOTION_FORBIDDEN")


__all__ = [
    "CANDIDATE_SCHEMA", "CommandResult", "CommandSpec", "DIAGNOSTIC_OVERALL_TIMEOUT_SECONDS",
    "MetadataDiagnosticStageError", "PLATFORM", "PythonRuntimeIdentityDiagnosticError",
    "RuntimeSubstageDiagnosticError", "TrainingImageLockError",
    "build_inspect_command", "build_pull_command", "build_python_runtime_identity_command",
    "build_runtime_command",
    "build_save_command", "canonical_runtime_lock_from_candidate",
    "capture_candidate", "diagnose_runtime_metadata", "diagnose_runtime_metadata_attributed",
    "diagnose_runtime_substage_attributed", "observe_python_runtime_identity",
    "parse_image_reference", "subprocess_runner",
    "validate_oci_documents", "warm_image_cache",
]
