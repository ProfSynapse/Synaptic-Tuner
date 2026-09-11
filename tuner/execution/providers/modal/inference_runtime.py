"""Concrete verification of the separately packaged Modal inference runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

from .inference_preparation import ModalInferencePreparationConfig
from .mounted_io import hash_regular, read_regular

_SCHEMA = "synaptic-modal-inference-runtime-lock/v1"
_RESOURCE = "inference-runtime.lock.json"
_DEPENDENCY_RESOURCE = "tuner/execution/providers/modal/inference-dependencies.lock"
_CLOSURE_RESOURCE = "tuner/execution/providers/modal/inference-worker-closure.json"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MEMBERS = 512
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY = re.compile(r"^[^\s:@]+(?:/[^\s:@]+)+@sha256:[0-9a-f]{64}$")
_DIST_SEPARATORS = re.compile(r"[-_.]+")
_REQUIRED_DISTRIBUTIONS = frozenset(
    {"modal", "vllm", "torch", "transformers", "tokenizers", "safetensors"}
)
_REQUIRED_SOURCES = frozenset(
    {
        "Evaluator/chat_session.py",
        "Evaluator/verified_vllm_chat.py",
        "Evaluator/vllm_runtime.py",
        "tuner/execution/providers/modal/inference_bootstrap.py",
        "tuner/execution/providers/modal/inference_runtime.py",
        "tuner/execution/providers/modal/inference_wire.py",
        "tuner/execution/providers/modal/inference_worker.py",
        "tuner/inference/retrieved_model.py",
        "tuner/inference/serving_target.py",
    }
)


class ModalInferenceRuntimeError(RuntimeError):
    """Closed non-secret runtime verification failure."""


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} is invalid")
    return value


def _text(value: object, label: str, maximum: int = 512) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > maximum
        or "\0" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _path(value: object) -> str:
    text = _text(value, "runtime member path", 512)
    if "\\" in text:
        raise ValueError("runtime member path is invalid")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("runtime member path is invalid")
    return text


def _runtime_root() -> Path:
    source = Path(__file__)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError("runtime verifier source is invalid")
    root = source.parents[4]
    if root.resolve(strict=True) != root:
        raise ValueError("runtime verifier root is not canonical")
    return root


def _manifest() -> tuple[bytes, dict[str, object]]:
    try:
        root = _runtime_root()
        payload = read_regular(
            root,
            root / "tuner" / "execution" / "providers" / "modal" / _RESOURCE,
            _MAX_MANIFEST_BYTES,
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError):
        raise ModalInferenceRuntimeError("modal_inference_runtime_invalid") from None
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_MANIFEST_BYTES:
        raise ModalInferenceRuntimeError("modal_inference_runtime_invalid")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        if payload != _canonical(document):
            raise ValueError("noncanonical manifest")
        root = _object(
            document,
            {
                "schema_version",
                "registry_reference",
                "sdk_version",
                "python",
                "dependency_lock_path",
                "worker_closure_manifest_path",
                "distributions",
                "source_inventory",
            },
            "runtime manifest",
        )
        if root["schema_version"] != _SCHEMA:
            raise ValueError("unsupported runtime manifest")
        reference = _text(root["registry_reference"], "registry reference", 1024)
        if _REGISTRY.fullmatch(reference) is None:
            raise ValueError("registry reference is invalid")
        _text(root["sdk_version"], "SDK version", 64)
        python = _object(
            root["python"],
            {"implementation", "version", "executable", "executable_sha256"},
            "Python runtime",
        )
        _text(python["implementation"], "Python implementation", 32)
        _text(python["version"], "Python version", 64)
        _text(python["executable"], "Python executable", 4096)
        _digest(python["executable_sha256"], "Python executable digest")
        dependency_lock_path = _path(root["dependency_lock_path"])
        closure_path = _path(root["worker_closure_manifest_path"])
        if (
            dependency_lock_path != _DEPENDENCY_RESOURCE
            or closure_path != _CLOSURE_RESOURCE
        ):
            raise ValueError("runtime commitment path is not fixed")
        distributions = root["distributions"]
        if (
            type(distributions) is not dict
            or not distributions
            or len(distributions) > 512
        ):
            raise ValueError("runtime distributions are invalid")
        if list(distributions) != sorted(distributions):
            raise ValueError("runtime distributions are not ordered")
        for name, version in distributions.items():
            _text(name, "distribution name", 128)
            _text(version, "distribution version", 256)
            if name != _DIST_SEPARATORS.sub("-", name).lower():
                raise ValueError("distribution name is not canonical")
        if not _REQUIRED_DISTRIBUTIONS.issubset(distributions):
            raise ValueError("required inference distributions are absent")
        if distributions["modal"] != root["sdk_version"]:
            raise ValueError("installed Modal selection differs from SDK pin")
        inventory = root["source_inventory"]
        if (
            type(inventory) is not list
            or not inventory
            or len(inventory) > _MAX_MEMBERS
        ):
            raise ValueError("runtime source inventory is invalid")
        paths: list[str] = []
        total = 0
        for raw in inventory:
            member = _object(
                raw, {"path", "size_bytes", "sha256"}, "runtime source member"
            )
            member_path = _path(member["path"])
            size = member["size_bytes"]
            if type(size) is not int or not 0 <= size <= _MAX_MEMBER_BYTES:
                raise ValueError("runtime source member size is invalid")
            total += size
            if total > _MAX_TOTAL_BYTES:
                raise ValueError("runtime source inventory exceeds its bound")
            _digest(member["sha256"], "runtime source member digest")
            paths.append(member_path)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("runtime source inventory is not canonical")
        if not _REQUIRED_SOURCES.issubset(paths):
            raise ValueError("runtime source inventory is incomplete")
        for required in (dependency_lock_path, closure_path):
            if required not in paths:
                raise ValueError("runtime commitment file is absent")
        return payload, root
    except (
        TypeError,
        ValueError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        OverflowError,
    ):
        raise ModalInferenceRuntimeError("modal_inference_runtime_invalid") from None


def _source_digest(inventory: list[dict[str, object]]) -> str:
    return hashlib.sha256(_canonical(inventory)).hexdigest()


def _worker_closure(
    payload: bytes,
    inventory: list[dict[str, object]],
    manifest_path: str,
    dependency_path: str,
) -> str:
    if not 0 < len(payload) <= _MAX_MANIFEST_BYTES:
        raise ValueError("worker closure exceeds its bound")
    document = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if payload != _canonical(document):
        raise ValueError("worker closure is not canonical")
    root = _object(
        document,
        {
            "schema_version",
            "entrypoint",
            "member_count",
            "payload_bytes",
            "members",
            "closure_digest",
        },
        "worker closure",
    )
    if (
        root["schema_version"] != "synaptic-modal-inference-worker-closure/v1"
        or root["entrypoint"]
        != "tuner/execution/providers/modal/inference_bootstrap.py"
        or type(root["member_count"]) is not int
        or type(root["payload_bytes"]) is not int
    ):
        raise ValueError("worker closure is invalid")
    recorded = _digest(root["closure_digest"], "worker closure digest")
    members = root["members"]
    if type(members) is not list or not members or len(members) > _MAX_MEMBERS:
        raise ValueError("worker closure members are invalid")
    expected = {
        item["path"]: item
        for item in inventory
        if item["path"] not in {manifest_path, dependency_path}
    }
    paths: list[str] = []
    total = 0
    for raw in members:
        member = _object(raw, {"path", "size_bytes", "sha256"}, "worker closure member")
        path = _path(member["path"])
        size = member["size_bytes"]
        if type(size) is not int or not 0 <= size <= _MAX_MEMBER_BYTES:
            raise ValueError("worker closure member size is invalid")
        _digest(member["sha256"], "worker closure member digest")
        if expected.get(path) != member:
            raise ValueError("worker closure differs from source inventory")
        total += size
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("worker closure exceeds its bound")
        paths.append(path)
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or set(paths) != set(expected)
        or root["member_count"] != len(members)
        or root["payload_bytes"] != total
    ):
        raise ValueError("worker closure is incomplete")
    unsigned = dict(root)
    unsigned.pop("closure_digest")
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != recorded:
        raise ValueError("worker closure digest differs")
    return recorded


def _actual_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _hash_executable(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_MEMBER_BYTES:
            raise ValueError("Python executable is not bounded regular data")
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_MEMBER_BYTES + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_MEMBER_BYTES:
                raise ValueError("Python executable exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if size != before.st_size or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError("Python executable changed during verification")
        return digest.hexdigest()
    finally:
        active_failure = sys.exc_info()[0] is not None
        try:
            os.close(descriptor)
        except BaseException:
            if not active_failure:
                raise


def verify_modal_inference_runtime(
    configuration: ModalInferencePreparationConfig,
) -> None:
    """Verify the fixed packaged inference runtime before any serving effect."""
    try:
        if type(configuration) is not ModalInferencePreparationConfig:
            raise TypeError("exact inference configuration required")
        payload, manifest = _manifest()
        document = configuration.document
        image = document["image"]
        client = document["client"]
        runtime = document["runtime"]
        python = manifest["python"]
        if (
            hashlib.sha256(payload).hexdigest() != runtime["runtime_lock_digest"]
            or manifest["registry_reference"] != image["registry_reference"]
            or manifest["registry_reference"].rsplit("@sha256:", 1)[1]
            != image["image_digest"]
            or manifest["sdk_version"] != client["sdk_version"]
            or python["version"] != runtime["python_version"]
            or python["executable"] != runtime["python_executable"]
            or python["executable_sha256"] != runtime["python_executable_digest"]
        ):
            raise ValueError("configuration differs from inference runtime")
        root = _runtime_root()
        inventory = manifest["source_inventory"]
        by_path = {item["path"]: item for item in inventory}
        for item in inventory:
            size, digest = hash_regular(
                root, root / item["path"], min(item["size_bytes"], _MAX_MEMBER_BYTES)
            )
            if size != item["size_bytes"] or digest != item["sha256"]:
                raise ValueError("runtime source member differs")
        if _source_digest(inventory) != runtime["source_lock_digest"]:
            raise ValueError("runtime source commitment differs")
        dependency = by_path[manifest["dependency_lock_path"]]
        if dependency["sha256"] != runtime["dependency_lock_digest"]:
            raise ValueError("runtime dependency commitment differs")
        closure_path = manifest["worker_closure_manifest_path"]
        closure_member = by_path[closure_path]
        closure_bytes = read_regular(
            root,
            root / closure_path,
            min(closure_member["size_bytes"], _MAX_MANIFEST_BYTES),
        )
        if (
            len(closure_bytes) != closure_member["size_bytes"]
            or hashlib.sha256(closure_bytes).hexdigest() != closure_member["sha256"]
            or _worker_closure(
                closure_bytes,
                inventory,
                closure_path,
                manifest["dependency_lock_path"],
            )
            != runtime["worker_closure_digest"]
        ):
            raise ValueError("runtime worker closure differs")
        executable = Path(python["executable"])
        if (
            python["implementation"] != "cpython"
            or sys.implementation.name != "cpython"
            or _actual_version() != python["version"]
            or not executable.is_absolute()
            or not os.path.samefile(sys.executable, executable)
            or _hash_executable(executable) != python["executable_sha256"]
        ):
            raise ValueError("physical Python runtime differs")
        actual_distributions: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            if type(name) is not str:
                raise ValueError("installed distribution identity is absent")
            normalized = _DIST_SEPARATORS.sub("-", name).lower()
            if normalized in actual_distributions:
                raise ValueError("installed distribution identity is ambiguous")
            actual_distributions[normalized] = distribution.version
        if actual_distributions != manifest["distributions"]:
            raise ValueError("installed inference distributions differ")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalInferenceRuntimeError("modal_inference_runtime_invalid") from None


__all__: list[str] = []
