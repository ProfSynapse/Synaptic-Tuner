"""Immutable closure validation and bootstrap for the offline SFT worker."""

from __future__ import annotations

import hashlib
import hmac
import importlib.abc
import importlib.machinery
import importlib.resources
import importlib.util
import json
import os
import re
import runpy
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Iterable, Mapping, Sequence


OFFLINE_SFT_CLOSURE_SCHEMA = "synaptic-offline-sft-worker-closure/v1"
OFFLINE_SFT_CLOSURE_REF = "synaptic/offline-sft-worker/v1"
OFFLINE_SFT_ENTRYPOINT = "Trainers/sft/runtime_v1.py"
OFFLINE_SFT_TRAINER_ENTRYPOINT = "Trainers/sft/train_sft.py"
OFFLINE_SFT_BOOTSTRAP = "tuner/runtime/offline_sft_worker.py"
OFFLINE_SFT_MANIFEST_NAME = "offline-sft-worker-v1.json"
OFFLINE_SFT_PACKAGED_MANIFEST_SOURCE = (
    "package:tuner.runtime/manifests/offline-sft-worker-v1.json"
)
WORKER_CLOSURE_MANIFEST_ENV = "SYNAPTIC_WORKER_CLOSURE_MANIFEST"
WORKER_CLOSURE_DIGEST_ENV = "SYNAPTIC_WORKER_CLOSURE_DIGEST"
OWNED_MODULE_PREFIXES = (
    "tuner",
    "synaptic_tuner",
    "Trainers",
    "shared",
    "SynthChat",
    "Evaluator",
    "MechInterp",
    "configs",
    "src",
)

_MANIFEST_FIELDS = {
    "schema_version",
    "closure_ref",
    "entrypoint",
    "trainer_entrypoint",
    "owned_module_prefixes",
    "optional_features",
    "member_count",
    "payload_bytes",
    "members",
    "closure_digest",
}
_MEMBER_FIELDS = {"path", "git_mode", "size_bytes", "sha256"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_CLOSURE_BYTES = 64 * 1024 * 1024
_VALUE_FLAGS = frozenset(
    {
        "--model-name",
        "--model-revision",
        "--model-cache-dir",
        "--model-snapshot",
        "--local-file",
        "--output-root",
        "--run-timestamp",
        "--runtime-v1-workload-fingerprint",
        "--runtime-v1-configuration-revision",
        "--runtime-v1-tokenizer-revision",
        "--runtime-v1-dataset-revision",
        "--runtime-v1-dataset-digest",
        "--batch-size",
        "--gradient-accumulation",
        "--learning-rate",
        "--max-steps",
        "--num-epochs",
        "--max-seq-length",
        "--seed",
        "--save-steps",
        "--save-total-limit",
        "--lora-r",
        "--lora-alpha",
        "--lora-dropout",
        "--lora-target-modules",
        "--init-lora-weights",
    }
)
_BOOLEAN_FLAGS = frozenset(
    {
        "--anonymous-model",
        "--no-dashboard",
        "--quiet",
        "--use-dora",
        "--use-rslora",
        "--split-dataset",
        "--load-in-4bit",
        "--no-load-in-4bit",
    }
)


class OfflineSFTWorkerError(RuntimeError):
    """Fail-closed offline worker closure or bootstrap rejection."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is prohibited: {value}")


def _unique_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def closure_digest(document: Mapping[str, object]) -> str:
    """Return the authoritative digest with ``closure_digest`` excluded."""

    payload = dict(document)
    payload.pop("closure_digest", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _strict_document(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise OfflineSFTWorkerError("worker closure manifest is not strict JSON") from exc
    if not isinstance(document, dict):
        raise OfflineSFTWorkerError("worker closure manifest must be an object")
    return document


def _canonical_member_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise OfflineSFTWorkerError("closure member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise OfflineSFTWorkerError("closure member path is not canonical")
    return path


@dataclass(frozen=True, slots=True)
class OfflineSFTClosureMember:
    path: PurePosixPath
    git_mode: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OfflineSFTWorkerClosure:
    manifest_path: Path
    closure_digest: str
    members: tuple[OfflineSFTClosureMember, ...]
    payload_bytes: int
    owned_module_prefixes: tuple[str, ...]

    @property
    def paths(self) -> frozenset[PurePosixPath]:
        return frozenset(member.path for member in self.members)


@dataclass(frozen=True, slots=True)
class OfflineSFTWorkerManifestV1:
    source_ref: str
    canonical_bytes: bytes
    byte_count: int
    sha256: str
    closure: OfflineSFTWorkerClosure


def parse_offline_sft_worker_manifest(
    payload: bytes,
    *,
    source_ref: str,
    manifest_path: Path,
    expected_digest: str | None = None,
) -> OfflineSFTWorkerManifestV1:
    """Parse one canonical closure manifest through the sole manifest authority."""

    if not isinstance(payload, bytes) or not 0 < len(payload) <= _MAX_MANIFEST_BYTES:
        raise OfflineSFTWorkerError("worker closure manifest exceeds its byte bound")
    if not isinstance(source_ref, str) or not source_ref:
        raise TypeError("manifest source_ref must be non-empty text")
    document = _strict_document(payload)
    canonical = _canonical_json(document) + b"\n"
    if payload != canonical:
        raise OfflineSFTWorkerError("worker closure manifest is not canonical JSON")
    recorded = document.get("closure_digest")
    closure = _parse_manifest(
        document,
        manifest_path=manifest_path,
        expected_digest=recorded if expected_digest is None else expected_digest,
    )
    return OfflineSFTWorkerManifestV1(
        source_ref=source_ref,
        canonical_bytes=payload,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        closure=closure,
    )


def load_packaged_offline_sft_worker_manifest() -> OfflineSFTWorkerManifestV1:
    """Load the immutable manifest shipped as ``tuner.runtime`` package data."""

    resource = importlib.resources.files("tuner.runtime").joinpath(
        "manifests", OFFLINE_SFT_MANIFEST_NAME
    )
    try:
        payload = resource.read_bytes()
    except (OSError, TypeError) as exc:
        raise OfflineSFTWorkerError("packaged worker closure manifest is unavailable") from exc
    return parse_offline_sft_worker_manifest(
        payload,
        source_ref=OFFLINE_SFT_PACKAGED_MANIFEST_SOURCE,
        manifest_path=Path(OFFLINE_SFT_PACKAGED_MANIFEST_SOURCE),
    )


def _read_regular(path: Path, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OfflineSFTWorkerError("closure file must be regular and unredirected")
        if not 0 <= before.st_size <= maximum:
            raise OfflineSFTWorkerError("closure file exceeds its byte bound")
        with path.open("rb") as stream:
            payload = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        confirmed = path.lstat()
    except OSError as exc:
        raise OfflineSFTWorkerError("closure file is unavailable") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        len(payload) > maximum
        or identity(before) != identity(after)
        or identity(after) != identity(confirmed)
    ):
        raise OfflineSFTWorkerError("closure file changed while it was read")
    return payload


def _require_unredirected_path(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise OfflineSFTWorkerError("closure member escapes the engine root") from exc
    current = root
    try:
        root_info = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
            raise OfflineSFTWorkerError("engine root must be an unredirected directory")
        for part in relative.parts:
            current = current / part
            info = current.lstat()
            if current.is_symlink() or (
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ):
                raise OfflineSFTWorkerError("closure member traverses a redirect")
    except OSError as exc:
        raise OfflineSFTWorkerError("closure member is unavailable") from exc


def _parse_manifest(
    document: Mapping[str, object],
    *,
    manifest_path: Path,
    expected_digest: str,
) -> OfflineSFTWorkerClosure:
    if set(document) != _MANIFEST_FIELDS:
        raise OfflineSFTWorkerError("worker closure manifest fields are malformed")
    if (
        document["schema_version"] != OFFLINE_SFT_CLOSURE_SCHEMA
        or document["closure_ref"] != OFFLINE_SFT_CLOSURE_REF
        or document["entrypoint"] != OFFLINE_SFT_ENTRYPOINT
        or document["trainer_entrypoint"] != OFFLINE_SFT_TRAINER_ENTRYPOINT
        or document["owned_module_prefixes"] != list(OWNED_MODULE_PREFIXES)
        or document["optional_features"] != []
    ):
        raise OfflineSFTWorkerError("worker closure identity is unsupported")
    recorded_digest = document["closure_digest"]
    observed_digest = closure_digest(document)
    if (
        not isinstance(recorded_digest, str)
        or _DIGEST.fullmatch(recorded_digest) is None
        or not isinstance(expected_digest, str)
        or _DIGEST.fullmatch(expected_digest) is None
        or not (
            hmac.compare_digest(recorded_digest, observed_digest)
            and hmac.compare_digest(recorded_digest, expected_digest)
        )
    ):
        raise OfflineSFTWorkerError("worker closure digest does not match")
    raw_members = document["members"]
    if not isinstance(raw_members, list) or not raw_members:
        raise OfflineSFTWorkerError("worker closure members are malformed")
    members: list[OfflineSFTClosureMember] = []
    for raw in raw_members:
        if not isinstance(raw, dict) or set(raw) != _MEMBER_FIELDS:
            raise OfflineSFTWorkerError("worker closure member is malformed")
        path = _canonical_member_path(raw["path"])
        git_mode = raw["git_mode"]
        size_bytes = raw["size_bytes"]
        sha256 = raw["sha256"]
        if (
            git_mode not in {"100644", "100755"}
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or not 0 <= size_bytes <= _MAX_MEMBER_BYTES
            or not isinstance(sha256, str)
            or _DIGEST.fullmatch(sha256) is None
        ):
            raise OfflineSFTWorkerError("worker closure member metadata is invalid")
        members.append(OfflineSFTClosureMember(path, git_mode, size_bytes, sha256))
    paths = [member.path.as_posix() for member in members]
    member_count = document["member_count"]
    payload_bytes = document["payload_bytes"]
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or not isinstance(member_count, int)
        or isinstance(member_count, bool)
        or member_count != len(members)
        or not isinstance(payload_bytes, int)
        or isinstance(payload_bytes, bool)
        or payload_bytes != sum(member.size_bytes for member in members)
        or not 0 < payload_bytes <= _MAX_CLOSURE_BYTES
    ):
        raise OfflineSFTWorkerError("worker closure totals or ordering are invalid")
    required = {
        OFFLINE_SFT_ENTRYPOINT,
        OFFLINE_SFT_TRAINER_ENTRYPOINT,
        OFFLINE_SFT_BOOTSTRAP,
    }
    if not required.issubset(set(paths)):
        raise OfflineSFTWorkerError("worker closure lacks a required entrypoint")
    return OfflineSFTWorkerClosure(
        manifest_path=manifest_path,
        closure_digest=recorded_digest,
        members=tuple(members),
        payload_bytes=payload_bytes,
        owned_module_prefixes=OWNED_MODULE_PREFIXES,
    )


def _iter_staged_files(root: Path) -> Iterable[PurePosixPath]:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise OfflineSFTWorkerError("worker closure cannot be enumerated") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    raise OfflineSFTWorkerError("worker closure contains a redirect")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    yield PurePosixPath(path.relative_to(root).as_posix())
                else:
                    raise OfflineSFTWorkerError("worker closure contains a special file")
            except OSError as exc:
                raise OfflineSFTWorkerError("worker closure entry is unreadable") from exc


def load_offline_sft_worker_closure(
    manifest_path: Path,
    *,
    expected_digest: str,
    engine_root: Path,
) -> OfflineSFTWorkerClosure:
    """Load and authenticate one exact, selectively staged worker closure."""

    if not isinstance(manifest_path, Path) or not isinstance(engine_root, Path):
        raise TypeError("manifest path and engine root must be Path values")
    if manifest_path.name != OFFLINE_SFT_MANIFEST_NAME:
        raise OfflineSFTWorkerError("worker closure manifest path is not fixed")
    if not manifest_path.is_absolute() or not engine_root.is_absolute():
        raise OfflineSFTWorkerError("worker closure paths must be absolute")
    try:
        if (
            engine_root.resolve(strict=True) != engine_root
            or manifest_path.resolve(strict=True) != manifest_path
        ):
            raise OfflineSFTWorkerError("worker closure paths are not canonical")
    except OSError as exc:
        raise OfflineSFTWorkerError("worker closure paths are unavailable") from exc
    _require_unredirected_path(manifest_path, manifest_path.parent)
    manifest = parse_offline_sft_worker_manifest(
        _read_regular(manifest_path, maximum=_MAX_MANIFEST_BYTES),
        source_ref=manifest_path.as_posix(),
        manifest_path=manifest_path,
        expected_digest=expected_digest,
    )
    closure = manifest.closure
    expected_paths = closure.paths
    observed_paths = frozenset(_iter_staged_files(engine_root))
    if observed_paths != expected_paths:
        raise OfflineSFTWorkerError("staged worker members do not exactly match closure")
    for member in closure.members:
        path = engine_root.joinpath(*member.path.parts)
        _require_unredirected_path(path, engine_root)
        payload = _read_regular(path, maximum=_MAX_MEMBER_BYTES)
        if len(payload) != member.size_bytes or not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(), member.sha256
        ):
            raise OfflineSFTWorkerError("staged worker member does not match closure")
        if os.name == "posix":
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (member.git_mode == "100755"):
                raise OfflineSFTWorkerError("staged worker member mode does not match")
    return closure


def load_offline_sft_worker_environment(
    environment: Mapping[str, str], *, engine_root: Path
) -> OfflineSFTWorkerClosure:
    if not isinstance(environment, Mapping):
        raise TypeError("worker environment must be a mapping")
    manifest = environment.get(WORKER_CLOSURE_MANIFEST_ENV)
    digest = environment.get(WORKER_CLOSURE_DIGEST_ENV)
    if (
        not isinstance(manifest, str)
        or not manifest
        or manifest != manifest.strip()
        or not isinstance(digest, str)
    ):
        raise OfflineSFTWorkerError("worker closure environment is incomplete")
    return load_offline_sft_worker_closure(
        Path(manifest), expected_digest=digest, engine_root=engine_root
    )


def _module_projection(
    closure: OfflineSFTWorkerClosure, engine_root: Path
) -> tuple[dict[str, Path], dict[str, frozenset[Path]]]:
    modules: dict[str, Path] = {}
    namespace_locations: dict[str, set[Path]] = {}
    for member in closure.members:
        if member.path.suffix != ".py":
            continue
        parts = list(member.path.with_suffix("").parts)
        aliases: list[tuple[list[str], tuple[str, ...]]] = [(parts, ())]
        if parts[:3] == ["Trainers", "sft", "configs"]:
            aliases.append((parts[2:], ("Trainers", "sft")))
        elif parts[:3] == ["Trainers", "sft", "src"]:
            aliases.append((parts[2:], ("Trainers", "sft")))
        for alias, base in aliases:
            if alias[-1] == "__init__":
                alias = alias[:-1]
            if alias:
                modules[".".join(alias)] = engine_root.joinpath(*member.path.parts)
            for index in range(1, len(alias)):
                parent = ".".join(alias[:index])
                relative = (*base, *alias[:index])
                namespace_locations.setdefault(parent, set()).add(
                    engine_root.joinpath(*relative)
                )
    namespaces = {
        name: frozenset(path.resolve() for path in paths)
        for name, paths in namespace_locations.items()
        if name not in modules
    }
    return modules, namespaces


class _OwnedModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, closure: OfflineSFTWorkerClosure, engine_root: Path) -> None:
        self._prefixes = frozenset(closure.owned_module_prefixes)
        self._modules, self._namespaces = _module_projection(closure, engine_root)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ):
        del target
        if fullname.partition(".")[0] not in self._prefixes:
            return None
        expected = self._modules.get(fullname)
        expected_namespace = self._namespaces.get(fullname)
        if expected is None and expected_namespace is None:
            raise ModuleNotFoundError(
                f"owned module {fullname!r} is outside the offline SFT closure"
            )
        del path
        if expected is not None:
            try:
                resolved = expected.resolve(strict=True)
            except OSError as exc:
                raise ModuleNotFoundError(
                    f"owned module {fullname!r} has no stable origin"
                ) from exc
            package_locations = [str(resolved.parent)] if expected.name == "__init__.py" else None
            spec = importlib.util.spec_from_file_location(
                fullname, str(resolved), submodule_search_locations=package_locations
            )
            if spec is None:
                raise ModuleNotFoundError(f"owned module {fullname!r} is unavailable")
        else:
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [
                str(value) for value in sorted(expected_namespace, key=str)
            ]
        return spec


def install_owned_module_guard(
    closure: OfflineSFTWorkerClosure, *, engine_root: Path
) -> importlib.abc.MetaPathFinder:
    guard = _OwnedModuleFinder(closure, engine_root)
    sys.meta_path.insert(0, guard)
    return guard


def verify_loaded_owned_module_origins(
    closure: OfflineSFTWorkerClosure, *, engine_root: Path
) -> None:
    modules, namespaces = _module_projection(closure, engine_root)
    prefixes = frozenset(closure.owned_module_prefixes)
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] not in prefixes or module is None:
            continue
        expected = modules.get(name)
        origin = getattr(module, "__file__", None)
        expected_namespace = namespaces.get(name)
        if expected is None:
            locations = getattr(module, "__path__", None)
            if expected_namespace is None or locations is None:
                raise OfflineSFTWorkerError("loaded owned module is outside the closure")
            try:
                observed = frozenset(Path(value).resolve(strict=True) for value in locations)
            except OSError as exc:
                raise OfflineSFTWorkerError(
                    "loaded owned namespace has no stable origin"
                ) from exc
            if observed != expected_namespace:
                raise OfflineSFTWorkerError(
                    "loaded owned namespace resolved outside the closure"
                )
            continue
        if not isinstance(origin, str):
            raise OfflineSFTWorkerError("loaded owned module is outside the closure")
        try:
            if Path(origin).resolve(strict=True) != expected.resolve(strict=True):
                raise OfflineSFTWorkerError(
                    "loaded owned module resolved outside the closure"
                )
        except OSError as exc:
            raise OfflineSFTWorkerError("loaded owned module has no stable origin") from exc


def _validate_trainer_arguments(arguments: Sequence[str]) -> None:
    if not arguments:
        raise OfflineSFTWorkerError("offline SFT trainer arguments are empty")
    index = 0
    seen: set[str] = set()
    while index < len(arguments):
        flag = arguments[index]
        if flag in seen:
            raise OfflineSFTWorkerError("offline SFT trainer flag is duplicated")
        seen.add(flag)
        if flag in _BOOLEAN_FLAGS:
            index += 1
            continue
        if flag in _VALUE_FLAGS and index + 1 < len(arguments):
            value = arguments[index + 1]
            if not isinstance(value, str) or not value or value.startswith("--"):
                raise OfflineSFTWorkerError("offline SFT trainer flag lacks a value")
            index += 2
            continue
        raise OfflineSFTWorkerError("offline SFT trainer feature is unavailable")
    if ("--max-steps" in seen) == ("--num-epochs" in seen):
        raise OfflineSFTWorkerError("offline SFT trainer duration is ambiguous")
    if ("--load-in-4bit" in seen) == ("--no-load-in-4bit" in seen):
        raise OfflineSFTWorkerError("offline SFT quantization is ambiguous")


def run_offline_sft_worker(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments.pop(0) != "--":
        raise OfflineSFTWorkerError("offline SFT bootstrap delimiter is missing")
    if (
        sys.flags.isolated != 1
        or sys.flags.no_user_site != 1
        or getattr(sys.flags, "safe_path", False) is not True
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("PYTHONSAFEPATH") != "1"
    ):
        raise OfflineSFTWorkerError("offline SFT bootstrap is not isolated")
    engine_text = os.environ.get("SYNAPTIC_ENGINE_ROOT")
    if not isinstance(engine_text, str) or not engine_text:
        raise OfflineSFTWorkerError("offline SFT engine root is unavailable")
    engine_root = Path(engine_text)
    closure = load_offline_sft_worker_environment(os.environ, engine_root=engine_root)
    bootstrap = engine_root.joinpath(*PurePosixPath(OFFLINE_SFT_BOOTSTRAP).parts)
    if Path(__file__).resolve(strict=True) != bootstrap.resolve(strict=True):
        raise OfflineSFTWorkerError("offline SFT bootstrap origin is invalid")
    _validate_trainer_arguments(arguments)
    verify_loaded_owned_module_origins(closure, engine_root=engine_root)
    install_owned_module_guard(closure, engine_root=engine_root)
    trainer_directory = engine_root / "Trainers" / "sft"
    sys.path[:0] = [str(trainer_directory), str(engine_root)]
    trainer = engine_root.joinpath(
        *PurePosixPath(OFFLINE_SFT_TRAINER_ENTRYPOINT).parts
    )
    sys.argv = [str(trainer), *arguments]
    runpy.run_path(str(trainer), run_name="__main__")
    verify_loaded_owned_module_origins(closure, engine_root=engine_root)
    return 0


def main() -> int:
    try:
        return run_offline_sft_worker()
    except (OfflineSFTWorkerError, ImportError):
        print("OFFLINE_SFT_WORKER_REJECTED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OFFLINE_SFT_BOOTSTRAP",
    "OFFLINE_SFT_CLOSURE_REF",
    "OFFLINE_SFT_CLOSURE_SCHEMA",
    "OFFLINE_SFT_ENTRYPOINT",
    "OFFLINE_SFT_MANIFEST_NAME",
    "OFFLINE_SFT_PACKAGED_MANIFEST_SOURCE",
    "OFFLINE_SFT_TRAINER_ENTRYPOINT",
    "OWNED_MODULE_PREFIXES",
    "OfflineSFTClosureMember",
    "OfflineSFTWorkerClosure",
    "OfflineSFTWorkerError",
    "OfflineSFTWorkerManifestV1",
    "WORKER_CLOSURE_DIGEST_ENV",
    "WORKER_CLOSURE_MANIFEST_ENV",
    "closure_digest",
    "install_owned_module_guard",
    "load_offline_sft_worker_closure",
    "load_offline_sft_worker_environment",
    "load_packaged_offline_sft_worker_manifest",
    "parse_offline_sft_worker_manifest",
    "run_offline_sft_worker",
    "verify_loaded_owned_module_origins",
]
