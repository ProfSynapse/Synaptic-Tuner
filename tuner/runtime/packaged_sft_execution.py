"""Two-phase, provider-neutral execution of an installed packaged SFT runtime.

Admission is read-only. The caller owns its physical directories and supplies
model preparation explicitly; neither credentials nor provider objects enter
the compiled workload or the offline trainer.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Protocol

from tuner.runtime.packaged_worker_closure import stable_read, stable_file_digest
from tuner.runtime.releases import (
    PackagedExecutionBindingV1, PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)
from tuner.training.contracts import ArtifactPolicy, CanonicalDocument
from tuner.training.packaged_compilation import (
    compile_packaged_sft_workload, packaged_artifact_policy_digest,
)

LINEAGE_SCHEMA = "synaptic-packaged-sft-training-lineage/v1"
TERMINAL_SCHEMA = "synaptic-packaged-sft-terminal/v1"
_ROLES = ("workload_record", "training_lineage", "training_metrics", "final_model", "tokenizer")
_CODES = frozenset({"ADMISSION", "PREPARATION", "REVALIDATION", "INVOCATION", "TRAINER", "EVIDENCE", "ARTIFACT"})
_TOKEN = object()
_MAX_MODEL_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
_MAX_MODEL_BYTES = 512 * 1024 * 1024 * 1024
_MAX_MODEL_MEMBERS = 10000
_MAX_CHILD_BYTES = 4 * 1024 * 1024


def _close_resources(resources):
    """Attempt every close without allowing cleanup exceptions to disclose text."""
    failed = False
    for resource in reversed(tuple(resources)):
        try:
            resource.close()
        except BaseException:
            failed = True
    return failed


class PackagedSFTExecutionError(RuntimeError):
    def __init__(self, stage: str):
        if stage not in _CODES:
            stage = "ADMISSION"
        self.stage = stage
        super().__init__("PACKAGED_SFT_" + stage + "_REJECTED")


class PackagedModelPreparer(Protocol):
    def __call__(self, model: Mapping[str, object], cache_root: Path) -> Path: ...


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _document(raw, maximum=256 * 1024):
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        raise ValueError
    # A canonical-byte comparison also rejects duplicate keys and constants.
    value = json.loads(raw)
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError
    return value


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PackagedSFTPaths:
    prepared_input: Path
    artifacts: Path
    state: Path
    tracking: Path
    cache: Path
    tmp: Path

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute() or Path(os.path.abspath(path)) != path or ".." in path.parts:
                raise ValueError("physical paths must be absolute and normalized")


class _HeldDirectory:
    """Retain directory identity until the admitted execution has completed."""
    def __init__(self, path, *, parent_fd=None):
        self.path = path
        self.fd = None
        self.handle = None
        info = (os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                if parent_fd is not None else path.lstat())
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel.CreateFileW
            create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
            create.restype = wintypes.HANDLE
            self.close_handle = kernel.CloseHandle
            self.close_handle.argtypes = (wintypes.HANDLE,)
            self.handle = create(str(path), 0x80, 3, None, 3, 0x02200000, None)
            if self.handle == wintypes.HANDLE(-1).value:
                self.handle = None
                raise OSError
        else:
            self.fd = os.open(path.name if parent_fd is not None else path,
                              os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            held = os.fstat(self.fd)
            if (held.st_dev, held.st_ino) != (info.st_dev, info.st_ino):
                self.close()
                raise ValueError
        self.identity = (info.st_dev, info.st_ino)
        self.check()

    def check(self):
        info = self.path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                or (info.st_dev, info.st_ino) != self.identity):
            raise ValueError

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.handle is not None:
            self.close_handle(self.handle)
            self.handle = None


@dataclass(frozen=True, slots=True)
class AdmittedPackagedSFT:
    release: PackagedTrainingRuntimeReleaseV1
    provider_binding: ProviderRuntimeBindingV1
    execution: PackagedExecutionBindingV1
    workload_bytes: bytes
    artifact_policy: ArtifactPolicy
    paths: PackagedSFTPaths
    environment: tuple[tuple[str, str], ...]
    _directories: tuple = field(repr=False)
    _seal: object = field(repr=False)
    _commitment: str = field(repr=False)
    _used: list = field(default_factory=list, repr=False)

    def close(self):
        self._used.append(True)
        if _close_resources(self._directories):
            raise PackagedSFTExecutionError("REVALIDATION") from None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True, slots=True)
class PackagedSFTResult:
    workload_fingerprint: str
    inventory_path: Path
    terminal_path: Path
    artifacts: tuple


def _installed_python_digest(executable: str) -> str:
    physical_python = Path(executable).resolve(strict=True)
    return _digest(stable_read(physical_python, 256 * 1024 * 1024))


def _inspect_release(release):
    """Measure the installed wheel graph, never a source checkout or Git tree."""
    from tuner.runtime.packaged_training_worker import inspect_installed_runtime
    expected = json.loads(stable_read(Path("/opt/synaptic-runtime/build-inputs.json")))
    if (sys.implementation.name != release.python_implementation
            or platform.python_version() != release.python_version
            or str(Path(sys.executable)) != release.python_executable):
        raise ValueError
    # A venv executable may be a symlink to the immutable image's physical
    # interpreter. Keep the configured invocation path exact, then hash the
    # resolved regular file through the no-follow stable reader.
    if _installed_python_digest(release.python_executable) != release.python_executable_digest:
        raise ValueError
    measured = inspect_installed_runtime(expected)
    document = release.to_dict()
    for name in ("package", "python", "installed_distributions", "worker", "contracts", "platform"):
        actual = measured[name]
        locked = document[name]
        if name == "installed_distributions":
            actual = {key: actual[key] for key in ("digest", "count")}
        if _canonical(actual) != _canonical(locked):
            raise ValueError
    if _canonical(measured["capabilities"]["compatibility"]) != _canonical(document["compatibility"]):
        raise ValueError
    if (platform.system().lower() != release.platform_system
            or platform.machine().lower() != release.platform_machine):
        raise ValueError
    distribution = importlib.metadata.distribution(release.package_name)
    from io import BytesIO
    from zipfile import ZipFile
    wheel_raw = stable_read(Path("/opt/synaptic-runtime") / expected["wheel"]["filename"], 256 * 1024 * 1024)
    if _digest(wheel_raw) != release.package_digest:
        raise ValueError
    with ZipFile(BytesIO(wheel_raw)) as wheel:
        return _require_trainer_assets(distribution, wheel)


def _require_trainer_assets(distribution, wheel):
    """These two entry assets must belong to both the wheel and installation."""
    trainer = Path(distribution.locate_file("Trainers/sft/train_sft.py"))
    required = {"Trainers/sft/train_sft.py", "Trainers/sft/configs/config.yaml"}
    if (not required <= {str(item).replace("\\", "/") for item in distribution.files or ()}
            or not required <= set(wheel.namelist())):
        raise ValueError
    for member in required:
        if stable_read(Path(distribution.locate_file(member))) != wheel.read(member):
            raise ValueError
    return trainer


def _admit_contracts(release, provider_binding, execution, workload_bytes, policy):
    from tuner.runtime.packaged_training_worker import admit_packaged_training_release
    if type(release) is not PackagedTrainingRuntimeReleaseV1 or type(provider_binding) is not ProviderRuntimeBindingV1 or type(execution) is not PackagedExecutionBindingV1 or type(policy) is not ArtifactPolicy:
        raise TypeError
    release = admit_packaged_training_release(release.canonical_bytes(), expected_release_digest=execution.runtime_release_digest)
    provider_binding = ProviderRuntimeBindingV1.from_dict(provider_binding.to_dict())
    execution = PackagedExecutionBindingV1.from_dict(execution.to_dict())
    execution.validate_bindings(release, provider_binding)
    document = _document(workload_bytes)
    compiled = compile_packaged_sft_workload(resolved_config=CanonicalDocument.from_mapping(document["configuration"]["document"]))
    config = compiled.document["configuration"]["document"]
    # This trainer has one exact tokenizer/model snapshot and a finite CLI
    # projection. Reject unsupported forms before preparation, not at launch.
    from Trainers.sft import runtime_v1 as core
    projection = _runtime_projection(compiled)
    trainer_args = []
    core._append_sft_arguments(trainer_args, projection.document["configuration"]["document"]["sft"], config["model"])
    core._model_snapshot_path(config["model"], Path("/cache"))
    if config["sft"]["packing"] is not False:
        raise ValueError
    if config["sft"]["dataset_format"] == "raw_text" and (config["sft"]["completion_only_loss"] or config["sft"]["assistant_only_loss"]):
        raise ValueError
    if (compiled.canonical_bytes != workload_bytes or compiled.fingerprint != execution.workload_digest
            or compiled.document["configuration"]["digest"] != execution.configuration_digest
            or _canonical(execution.to_dict()["prepared_input"]) != _canonical(config["dataset"])
            or packaged_artifact_policy_digest(policy) != execution.artifact_policy_digest
            or "sft" not in release.compatible_methods
            or (config["model"]["ref"], config["model"]["revision"]) not in release.compatible_models
            or config["model"]["tokenizer_revision"] != config["model"]["revision"]
            or config["dataset"]["format"] not in release.compatible_dataset_formats
            or release.workload_schema != compiled.schema_version
            or release.prepared_input_schema != "synaptic-prepared-training-input/v1"
            or release.artifact_contract_schema != compiled.document["artifacts"]["schema_version"]):
        raise ValueError
    return compiled


def _hold_paths(paths):
    if type(paths) is not PackagedSFTPaths:
        raise TypeError
    roots = tuple(getattr(paths, name) for name in ("artifacts", "state", "tracking", "cache", "tmp"))
    for index, root in enumerate(roots):
        for other in roots[index + 1:]:
            if root == other or root in other.parents or other in root.parents:
                raise ValueError
        if root == paths.prepared_input or root in paths.prepared_input.parents:
            raise ValueError
    directories = []
    try:
        all_paths = set()
        for path in (*roots, paths.prepared_input.parent):
            all_paths.update((path, *path.parents))
        for path in sorted(all_paths, key=lambda item: (len(item.parts), str(item))):
            directories.append(_HeldDirectory(path))
        if any(paths.artifacts.iterdir()) or any(paths.state.iterdir()) or any(paths.tracking.iterdir()) or any(paths.tmp.iterdir()):
            raise ValueError
        return tuple(directories)
    except BaseException:
        for directory in reversed(directories):
            directory.close()
        raise


def _input(paths, execution):
    raw = stable_read(paths.prepared_input, execution.prepared_input_size_bytes)
    if len(raw) != execution.prepared_input_size_bytes or _digest(raw) != execution.prepared_input_content_digest:
        raise ValueError
    return raw


def admit_packaged_sft(*, runtime_release: PackagedTrainingRuntimeReleaseV1,
                       provider_binding: ProviderRuntimeBindingV1,
                       execution_binding: PackagedExecutionBindingV1,
                       workload_bytes: bytes, artifact_policy: ArtifactPolicy,
                       paths: PackagedSFTPaths, environment=()) -> AdmittedPackagedSFT:
    """Read-only admission. No model preparer or trainer is called here."""
    directories = ()
    try:
        compiled = _admit_contracts(runtime_release, provider_binding, execution_binding, workload_bytes, artifact_policy)
        _inspect_release(runtime_release)
        directories = _hold_paths(paths)
        _input(paths, execution_binding)
        # Ambient environment is never inherited, even for allowed keys.
        values = dict(environment)
        allowed = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL"}
        if len(values) != len(environment) or not set(values) <= allowed or any(type(value) is not str or "\0" in value for value in values.values()):
            raise ValueError
        environment = tuple(sorted(values.items()))
        from Trainers.sft import runtime_v1 as core
        from pathlib import PurePosixPath
        physical = SimpleNamespace(release=runtime_release, provider_binding=provider_binding,
            execution=execution_binding, workload_bytes=workload_bytes,
            artifact_policy=artifact_policy, paths=paths, environment=environment)
        _invocation_spec(physical, _runtime_projection(compiled),
            core._model_snapshot_path(compiled.document["configuration"]["document"]["model"], paths.cache),
            PurePosixPath("/proc/self/fd/2147483647"))
        commitment = _physical_commitment(paths, environment)
        return AdmittedPackagedSFT(runtime_release, provider_binding, execution_binding,
            workload_bytes, artifact_policy, paths, environment, directories, _TOKEN, commitment)
    except BaseException:
        _close_resources(directories)
        raise PackagedSFTExecutionError("ADMISSION") from None


def _physical_commitment(paths, environment):
    return _digest(_canonical({"paths": {name: str(getattr(paths, name)) for name in paths.__dataclass_fields__},
                               "environment": list(environment)}))


def _runtime_projection(compiled):
    """Explicit trainer-config projection; never a legacy workload or source."""
    document = compiled.document
    config = dict(document["configuration"]["document"])
    sft = dict(config["sft"])
    sft.pop("schema_version")
    duration = sft.pop("duration")
    sft.update({key: value for key, value in duration.items() if value is not None})
    if "num_epochs" in sft:
        # The installed trainer's CLI accepts integral epoch counts only.
        if not float(sft["num_epochs"]).is_integer():
            raise ValueError
        sft["num_epochs"] = int(sft["num_epochs"])
    config["sft"] = sft
    return SimpleNamespace(fingerprint=compiled.fingerprint, document={
        "configuration": {"revision": document["configuration"]["digest"], "document": config},
        "identities": document["identities"],
    })


def _snapshot_inventory(snapshot, cache):
    if not isinstance(snapshot, Path) or not snapshot.is_absolute() or cache not in snapshot.parents:
        raise ValueError
    held = []
    members = []
    try:
        for parent in reversed(snapshot.parents):
            if parent != cache and cache in parent.parents:
                held.append(_HeldDirectory(parent))
        pending = [snapshot]
        while pending:
            current = pending.pop()
            held.append(_HeldDirectory(current))
            for path in sorted(current.iterdir()):
                info = path.lstat()
                if stat.S_ISDIR(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400:
                    pending.append(path)
                else:
                    size, digest = stable_file_digest(path, 32 * 1024 * 1024 * 1024)
                    members.append((str(path.relative_to(snapshot)), size, digest))
                if len(members) + len(held) + len(pending) > 10000:
                    raise ValueError
        if not members:
            raise ValueError
        return tuple(held), tuple(sorted(members))
    except BaseException:
        for directory in reversed(held):
            directory.close()
        raise


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class _HeldModelFile:
    """Read-only handle to one authenticated private copy, retained through use."""
    def __init__(self, path, parent, member, *, require_readonly=True):
        self.path, self.parent, self.member = path, parent, member
        self.require_readonly = require_readonly
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        self.fd = os.open(path.name, flags, dir_fd=parent.fd) if os.name == "posix" else os.open(path, flags)
        try:
            self.identity = _file_identity(os.fstat(self.fd))
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        self.parent.check()
        before = os.fstat(self.fd)
        named = (os.stat(self.path.name, dir_fd=self.parent.fd, follow_symlinks=False)
                 if os.name == "posix" else self.path.lstat())
        for info in (before, named):
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or getattr(info, "st_file_attributes", 0) & 0x400
                    or (self.require_readonly and info.st_mode & 0o222)
                    or (info.st_dev, info.st_ino, info.st_size) != (self.member["device"], self.member["inode"], self.member["size_bytes"])):
                raise ValueError("PACKAGED_MODEL_COPY_REJECTED")
        if _file_identity(before) != self.identity:
            raise ValueError("PACKAGED_MODEL_COPY_REJECTED")
        digest = hashlib.sha256()
        os.lseek(self.fd, 0, os.SEEK_SET)
        remaining = self.member["size_bytes"]
        while remaining:
            raw = os.read(self.fd, min(1024 * 1024, remaining))
            if not raw:
                raise ValueError("PACKAGED_MODEL_COPY_REJECTED")
            digest.update(raw)
            remaining -= len(raw)
        if os.read(self.fd, 1) or digest.hexdigest() != self.member["sha256"] or _file_identity(os.fstat(self.fd)) != self.identity:
            raise ValueError("PACKAGED_MODEL_COPY_REJECTED")

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)


@dataclass
class _PrivateModelSnapshot:
    manifest: dict
    snapshot: Path
    directories: tuple
    files: tuple

    def check(self):
        for directory in self.directories:
            directory.check()
        for member in self.files:
            member.check()
        # Reject replacement, insertion, and removal as well as inode writes.
        expected = {file.path for file in self.files} | {directory.path for directory in self.directories[1:]}
        observed = {path for directory in self.directories for path in directory.path.iterdir()}
        if observed != expected:
            raise ValueError("PACKAGED_MODEL_COPY_REJECTED")

    def close(self):
        failed = _close_resources(self.files)
        failed = _close_resources(self.directories) or failed
        if failed:
            raise PackagedSFTExecutionError("REVALIDATION") from None


def _private_manifest(manifest, paths, model):
    """Validate only the physical copy binding; canonical workloads stay logical."""
    import re
    from pathlib import PurePosixPath
    from Trainers.sft import runtime_v1 as core
    if type(manifest) is not dict or set(manifest) != {"root", "device", "inode", "members"}:
        raise ValueError
    root = Path(manifest["root"])
    if (str(root) != manifest["root"] or root.parent != paths.state or re.fullmatch(r"packaged-model-[0-9a-f]{32}", root.name) is None
            or any(type(manifest[key]) is not int or manifest[key] < 0 for key in ("device", "inode"))):
        raise ValueError
    members = manifest["members"]
    if type(members) is not list or not 0 < len(members) <= _MAX_MODEL_MEMBERS:
        raise ValueError
    total = 0
    names = []
    for member in members:
        if type(member) is not dict or set(member) != {"path", "size_bytes", "sha256", "device", "inode"}:
            raise ValueError
        name = member["path"]
        if (type(name) is not str or not 0 < len(name) <= 4096 or "\\" in name
                or PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or type(member["size_bytes"]) is not int or not 0 <= member["size_bytes"] <= _MAX_MODEL_MEMBER_BYTES
                or type(member["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", member["sha256"]) is None
                or any(type(member[key]) is not int or member[key] < 0 for key in ("device", "inode"))):
            raise ValueError
        total += member["size_bytes"]
        names.append(name)
    if total > _MAX_MODEL_BYTES or names != sorted(set(names)):
        raise ValueError
    return root, core._model_snapshot_path(model, root)


def _retain_private_snapshot(manifest, paths, model):
    root, snapshot = _private_manifest(manifest, paths, model)
    directories, files = [], []
    try:
        required = {root, snapshot}
        for member in manifest["members"]:
            path = snapshot / member["path"]
            required.update(parent for parent in path.parents if parent == root or root in parent.parents)
        by_path = {}
        for path in sorted(required, key=lambda item: (len(item.parts), str(item))):
            parent = by_path.get(path.parent)
            directory = _HeldDirectory(path, parent_fd=parent.fd if parent is not None and os.name == "posix" else None)
            directories.append(directory)
            by_path[path] = directory
            info = path.lstat()
            if os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                raise ValueError
        if by_path[root].identity != (manifest["device"], manifest["inode"]):
            raise ValueError
        for member in manifest["members"]:
            path = snapshot / member["path"]
            files.append(_HeldModelFile(path, by_path[path.parent], member))
        retained = _PrivateModelSnapshot(manifest, snapshot, tuple(directories), tuple(files))
        retained.check()
        return retained
    except BaseException:
        _close_resources(files)
        _close_resources(directories)
        raise


def _copy_private_snapshot(snapshot, source_directories, inventory, paths, model):
    """Copy hostile cache bytes into a fresh private namespace, never hardlink.

    The boundary excludes arbitrary same-UID compromise of this private execution
    namespace. The shared cache remains hostile: later writes/replacements there
    cannot affect these regular, read-only copies. Parent and child retain and
    authenticate every private inode before and after trainer consumption.
    """
    if os.name != "posix":
        raise ValueError("PACKAGED_MODEL_COPY_PLATFORM_REJECTED")
    from Trainers.sft import runtime_v1 as core
    if not 0 < len(inventory) <= _MAX_MODEL_MEMBERS or sum(item[1] for item in inventory) > _MAX_MODEL_BYTES:
        raise ValueError
    source_parents = {directory.path: directory for directory in source_directories}
    opened = []
    members = []
    try:
        state = _HeldDirectory(paths.state)
        opened.append(state)
        name = "packaged-model-" + secrets.token_hex(16)
        os.mkdir(name, mode=0o700, dir_fd=state.fd)
        root = paths.state / name
        root_handle = _HeldDirectory(root, parent_fd=state.fd)
        opened.append(root_handle)
        private = core._model_snapshot_path(model, root)
        destinations = {root: root_handle}
        for relative, size, expected in inventory:
            if not 0 <= size <= _MAX_MODEL_MEMBER_BYTES:
                raise ValueError
            source = snapshot / relative
            destination = private / relative
            parent = root_handle
            for part in destination.relative_to(root).parts[:-1]:
                path = parent.path / part
                if path not in destinations:
                    os.mkdir(part, mode=0o700, dir_fd=parent.fd)
                    directory = _HeldDirectory(path, parent_fd=parent.fd)
                    opened.append(directory)
                    destinations[path] = directory
                parent = destinations[path]
            source_parent = source_parents[source.parent]
            source_parent.check()
            source_fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_parent.fd)
            target_fd = None
            try:
                before = os.fstat(source_fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != size:
                    raise ValueError
                target_fd = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                    0o600, dir_fd=parent.fd)
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    raw = os.read(source_fd, min(1024 * 1024, remaining))
                    if not raw:
                        raise ValueError
                    remaining -= len(raw)
                    digest.update(raw)
                    view = memoryview(raw)
                    while view:
                        written = os.write(target_fd, view)
                        if written <= 0:
                            raise ValueError
                        view = view[written:]
                named = os.stat(source.name, dir_fd=source_parent.fd, follow_symlinks=False)
                if (os.read(source_fd, 1) or digest.hexdigest() != expected
                        or _file_identity(before) != _file_identity(os.fstat(source_fd))
                        or _file_identity(before) != _file_identity(named) or named.st_nlink != 1):
                    raise ValueError
                os.fsync(target_fd)
                os.fchmod(target_fd, 0o400)
                target = os.fstat(target_fd)
                members.append({"path": relative, "size_bytes": size, "sha256": expected,
                                "device": target.st_dev, "inode": target.st_ino})
            finally:
                os.close(source_fd)
                if target_fd is not None:
                    os.close(target_fd)
        manifest = {"root": str(root), "device": root_handle.identity[0], "inode": root_handle.identity[1],
                    "members": sorted(members, key=lambda member: member["path"])}
        return _retain_private_snapshot(manifest, paths, model)
    finally:
        if _close_resources(opened):
            raise PackagedSFTExecutionError("REVALIDATION") from None


def _invocation_spec(admitted, workload, snapshot, dataset_path, model_snapshot=None):
    from Trainers.sft import runtime_v1 as core
    paths = admitted.paths
    config = workload.document["configuration"]["document"]
    model, dataset = config["model"], config["dataset"]
    output = paths.state / "runtime-v1-trainer" / "output"
    run = output / "runtime-v1"
    final = run / "final_model"
    args = ["--model-name", model["ref"], "--model-revision", model["revision"],
            "--anonymous-model", "--model-cache-dir", str(snapshot.parents[2]),
            "--model-snapshot", str(snapshot), "--local-file", str(dataset_path),
            "--output-root", str(output), "--run-timestamp", "runtime-v1", "--no-dashboard", "--quiet",
            "--runtime-v1-workload-fingerprint", workload.fingerprint,
            "--runtime-v1-configuration-revision", admitted.execution.configuration_digest,
            "--runtime-v1-tokenizer-revision", model["tokenizer_revision"],
            "--runtime-v1-dataset-revision", dataset["revision"],
            "--runtime-v1-dataset-digest", dataset["content_digest"]]
    core._append_sft_arguments(args, config["sft"], model)
    if config["sft"]["dataset_format"] == "raw_text":
        args.extend(("--aux-head-prompt-render", config["sft"]["prompt_render"]))
        if config["sft"]["require_memory_efficient_loss"]:
            args.append("--require-memory-efficient-loss")
    if config["sft"].get("require_memory_efficient_loss") is False and "--require-memory-efficient-loss" in args:
        args.remove("--require-memory-efficient-loss")
    child = {"release": admitted.release.to_dict(), "release_digest": admitted.release.manifest_digest, "arguments": args,
        "provider_binding": admitted.provider_binding.to_dict(), "execution_binding": admitted.execution.to_dict(),
        "workload": _document(admitted.workload_bytes),
        "artifact_policy": {"required_kinds": list(admitted.artifact_policy.required_kinds), "retain_checkpoints": admitted.artifact_policy.retain_checkpoints},
        "paths": {name: str(getattr(paths, name)) for name in paths.__dataclass_fields__},
        "environment": list(admitted.environment), "model_snapshot": model_snapshot}
    child_raw = _canonical(child)
    if len(child_raw) > _MAX_CHILD_BYTES:
        raise ValueError("PACKAGED_CHILD_TRANSPORT_REJECTED")
    child_path = paths.state / "packaged-child.json"
    env = dict(admitted.environment)
    # Search paths are image policy, never caller-controlled run authority.
    from pathlib import PurePosixPath
    env["PATH"] = ":".join(dict.fromkeys((str(PurePosixPath(admitted.release.python_executable).parent),
                                           "/usr/local/bin", "/usr/bin", "/bin")))
    env.setdefault("LC_ALL", "C.UTF-8")
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": str(paths.cache / "huggingface"), "TRANSFORMERS_CACHE": str(paths.cache / "transformers"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
            "TMPDIR": str(paths.tmp), "HOME": str(paths.tmp), "SYNAPTIC_MODEL_SNAPSHOT": str(snapshot)})
    argv = (admitted.release.python_executable, "-I", "-m", "tuner.runtime.packaged_sft_child",
                "--transport", str(child_path), "--digest", _digest(child_raw), "--", *args)
    return argv, env, child_path, child_raw, run, final


def _invocation(admitted, workload, snapshot, dataset_bytes, model_snapshot=None):
    from Trainers.sft import runtime_v1 as core
    paths = admitted.paths
    dataset = workload.document["configuration"]["document"]["dataset"]
    fd, dataset_path = core._sealed_prepared_dataset(dataset_bytes, content_digest=dataset["content_digest"])
    try:
        argv, env, child_path, child_raw, run, final = _invocation_spec(admitted, workload, snapshot, dataset_path, model_snapshot)
        core._write_exclusive(child_path, child_raw)
        return core.TrainerInvocation(argv, paths.tmp, tuple(sorted(env.items())), run, final, final,
            run / "training_lineage.json", run / "runtime_v1_projection.json",
            core._expected_trainer_projection(workload, dataset_path=dataset_path, run_dir=run, final_model_dir=final),
            paths.tracking / "trainer.stdout.log", paths.tracking / "trainer.stderr.log", (fd,))
    except BaseException:
        core._close_retained_fds((fd,))
        raise


def _bindings(admitted):
    return {"runtime_release_digest": admitted.release.manifest_digest,
        "provider_runtime_binding_digest": admitted.provider_binding.binding_digest,
        "execution_binding_digest": admitted.execution.binding_digest,
        "workload_fingerprint": admitted.execution.workload_digest,
        "configuration_digest": admitted.execution.configuration_digest,
        "artifact_policy_digest": admitted.execution.artifact_policy_digest,
        "run_ref": admitted.execution.run_ref}


def _retain_result_artifacts(admitted, result):
    """Bind terminal publication to original runtime descriptors, never disk claims."""
    if result.workload_fingerprint != admitted.execution.workload_digest:
        raise ValueError
    artifacts = [dict(entry) for entry in result.artifacts]
    if tuple(entry.get("role") for entry in artifacts) != _ROLES:
        raise ValueError
    expected = _canonical({"schema_version": "synaptic-artifact-inventory/v1",
                           "workload_fingerprint": result.workload_fingerprint, "artifacts": artifacts})
    if result.inventory_path != admitted.paths.state / "runtime-v1-inventory.json" or stable_read(result.inventory_path) != expected:
        raise ValueError
    parents = {directory.path: directory for directory in admitted._directories}
    retained = []
    try:
        names = ("workload.json", "training_lineage.json", "training_metrics.json", "final_model.tar", "tokenizer.tar")
        records = [(result.inventory_path, len(expected), _digest(expected))]
        for entry, name in zip(artifacts, names):
            if set(entry) != {"role", "path", "sha256", "size"} or entry["path"] != name:
                raise ValueError
            records.append((admitted.paths.artifacts / name, entry["size"], entry["sha256"]))
        for path, size, digest in records:
            info = path.lstat()
            member = {"path": path.name, "size_bytes": size, "sha256": digest,
                      "device": info.st_dev, "inode": info.st_ino}
            retained.append(_HeldModelFile(path, parents[path.parent], member, require_readonly=False))
        return expected, tuple(retained)
    except BaseException:
        _close_resources(retained)
        raise


def _write_terminal_bytes(descriptor, payload):
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ValueError
        view = view[written:]
    os.fsync(descriptor)


def _check_artifact_namespace(parent, files):
    """Enumerate the retained artifact directory, not an independently reopened path."""
    parent.check()
    expected = [resource.path.name for resource in files if resource.path.parent == parent.path]
    if len(expected) != 5 or len(set(expected)) != 5:
        raise ValueError
    names = os.listdir(parent.fd if os.name == "posix" else parent.path)
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError
    for name in names:
        info = (os.stat(name, dir_fd=parent.fd, follow_symlinks=False) if os.name == "posix"
                else (parent.path / name).lstat())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError
    parent.check()


def _write_terminal_exclusive(path, payload, parent, check):
    """Hold every original artifact identity across terminal commit, or reject it.

    If the post-write check fails, remove only our exclusively created terminal
    inode. A caller never receives completed evidence for a drifted artifact set.
    """
    parent.check()
    check()
    descriptor = None
    identity = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = (os.open(path.name, flags, 0o600, dir_fd=parent.fd) if os.name == "posix"
                      else os.open(path, flags, 0o600))
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        _write_terminal_bytes(descriptor, payload)
        check()
        parent.check()
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if identity is not None:
            parent.check()
            named = (os.stat(path.name, dir_fd=parent.fd, follow_symlinks=False) if os.name == "posix" else path.lstat())
            if stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == identity:
                if os.name == "posix":
                    os.unlink(path.name, dir_fd=parent.fd)
                else:
                    path.unlink()
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def execute_admitted_packaged_sft(admitted: AdmittedPackagedSFT, *, model_preparer: PackagedModelPreparer, runner=None) -> PackagedSFTResult:
    """Consume one admitted execution; revalidation cannot renew its authority."""
    core = None
    snapshots = ()
    artifact_directories = []
    artifact_files = ()
    private_copy = None
    stage = "REVALIDATION"
    try:
        from Trainers.sft import runtime_v1 as core
        if (type(admitted) is not AdmittedPackagedSFT or admitted._seal is not _TOKEN or admitted._used
                or _physical_commitment(admitted.paths, admitted.environment) != admitted._commitment):
            raise ValueError
        admitted._used.append(True)
        compiled = _admit_contracts(admitted.release, admitted.provider_binding, admitted.execution,
                                   admitted.workload_bytes, admitted.artifact_policy)
        _inspect_release(admitted.release)
        for directory in admitted._directories:
            directory.check()
        _input(admitted.paths, admitted.execution)
        stage = "PREPARATION"
        model = compiled.document["configuration"]["document"]["model"]
        snapshot = model_preparer(dict(model), admitted.paths.cache)
        if snapshot != core._model_snapshot_path(model, admitted.paths.cache):
            raise ValueError
        snapshots, inventory = _snapshot_inventory(snapshot, admitted.paths.cache)
        stage = "REVALIDATION"
        for directory in (*admitted._directories, *snapshots):
            directory.check()
        dataset_bytes = _input(admitted.paths, admitted.execution)
        _inspect_release(admitted.release)
        confirm, second = _snapshot_inventory(snapshot, admitted.paths.cache)
        try:
            if second != inventory:
                raise ValueError
        finally:
            for directory in reversed(confirm):
                directory.close()
        if any(admitted.paths.artifacts.iterdir()) or any(admitted.paths.state.iterdir()) or any(admitted.paths.tracking.iterdir()) or any(admitted.paths.tmp.iterdir()):
            raise ValueError
        private_copy = _copy_private_snapshot(snapshot, snapshots, inventory, admitted.paths, model)
        projection = _runtime_projection(compiled)
        stage = "INVOCATION"
        def build(workload, roots, environment):
            private_copy.check()
            return _invocation(admitted, workload, private_copy.snapshot, dataset_bytes, private_copy.manifest)
        def lineage(workload, invocation, evidence, execution_evidence):
            execution_evidence = {**execution_evidence, "schema_version": "synaptic-packaged-sft-execution-evidence/v1"}
            return {"schema_version": LINEAGE_SCHEMA, **_bindings(admitted),
                "prepared_input": admitted.execution.to_dict()["prepared_input"],
                "model_snapshot": private_copy.manifest,
                "trainer_projection": dict(evidence.projection), "trainer_lineage": dict(evidence.lineage),
                "execution_evidence": execution_evidence, "execution_evidence_sha256": _digest(_canonical(execution_evidence)),
                "status": "completed", "trainer_exit_code": 0}
        class RevalidatingRunner:
            def run(self, invocation):
                nonlocal stage
                stage = "REVALIDATION"
                for directory in (*admitted._directories, *snapshots):
                    directory.check()
                private_copy.check()
                _input(admitted.paths, admitted.execution)
                held, final_inventory = _snapshot_inventory(snapshot, admitted.paths.cache)
                try:
                    if final_inventory != inventory:
                        raise ValueError
                finally:
                    for directory in reversed(held):
                        directory.close()
                stage = "TRAINER"
                evidence = (runner or core.SubprocessTrainerRunner()).run(invocation)
                stage = "EVIDENCE"
                private_copy.check()
                for directory in admitted._directories:
                    directory.check()
                if type(evidence) is core.TrainerEvidence and evidence.exit_code == 0:
                    if evidence.final_model_dir != invocation.final_model_dir or evidence.tokenizer_dir != invocation.tokenizer_dir:
                        raise ValueError
                    for path in set((evidence.final_model_dir, evidence.tokenizer_dir)):
                        held, _ = _snapshot_inventory(path, admitted.paths.state)
                        artifact_directories.extend(held)
                return evidence
        result = core.execute_compiled_sft(admitted.workload_bytes, workload=projection, roots=admitted.paths,
            environment={}, runner=RevalidatingRunner(), invocation_builder=build, lineage_builder=lineage)
        stage = "ARTIFACT"
        private_copy.check()
        inventory_raw, artifact_files = _retain_result_artifacts(admitted, result)
        terminal = {"schema_version": TERMINAL_SCHEMA, **_bindings(admitted), "status": "completed",
            "inventory_sha256": _digest(inventory_raw), "artifact_roles": list(_ROLES)}
        terminal_path = admitted.paths.state / "packaged-terminal.json"
        verify_packaged_sft_artifacts(admitted=admitted, inventory_bytes=inventory_raw,
                                     terminal_bytes=_canonical(terminal))
        def check_terminal_inputs():
            private_copy.check()
            for resource in artifact_files:
                resource.check()
            _check_artifact_namespace(artifact_root, artifact_files)
        artifact_root = next(directory for directory in admitted._directories if directory.path == admitted.paths.artifacts)
        state = next(directory for directory in admitted._directories if directory.path == admitted.paths.state)
        _write_terminal_exclusive(terminal_path, _canonical(terminal), state, check_terminal_inputs)
        return PackagedSFTResult(result.workload_fingerprint, result.inventory_path, terminal_path, result.artifacts)
    except BaseException as error:
        try:
            if core is not None and isinstance(error, core.RuntimeV1Error):
                code = getattr(error, "diagnostic_code", "")
                stage = {"runtime_trainer_failed": "TRAINER", "runtime_evidence_rejected": "EVIDENCE",
                         "runtime_artifact_rejected": "ARTIFACT"}.get(code, stage)
        except BaseException:
            pass
        raise PackagedSFTExecutionError(stage) from None
    finally:
        resources = [*snapshots, *artifact_directories, *artifact_files]
        if private_copy is not None:
            resources.append(private_copy)
        if type(admitted) is AdmittedPackagedSFT:
            resources.append(admitted)
        if _close_resources(resources):
            raise PackagedSFTExecutionError(stage) from None


def verify_packaged_sft_artifacts(*, admitted, inventory_bytes, terminal_bytes):
    """Authenticate packaged terminal and five-role lineage without legacy Git evidence."""
    from tuner.runtime.verification import _validate_sft_archive_stream
    from Trainers.sft import runtime_v1 as core
    inventory = _document(inventory_bytes)
    terminal = _document(terminal_bytes)
    expected = {"schema_version": TERMINAL_SCHEMA, **_bindings(admitted), "status": "completed",
        "inventory_sha256": _digest(inventory_bytes), "artifact_roles": list(_ROLES)}
    if terminal != expected or set(inventory) != {"schema_version", "workload_fingerprint", "artifacts"} or inventory["schema_version"] != "synaptic-artifact-inventory/v1" or inventory["workload_fingerprint"] != admitted.execution.workload_digest:
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    entries = inventory["artifacts"]
    if type(entries) is not list or [item.get("role") for item in entries] != list(_ROLES):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    contents = {}
    names = ("workload.json", "training_lineage.json", "training_metrics.json", "final_model.tar", "tokenizer.tar")
    if {path.name for path in admitted.paths.artifacts.iterdir()} != set(names):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    for entry, name in zip(entries, names):
        if set(entry) != {"role", "path", "sha256", "size"} or entry["path"] != name or type(entry["size"]) is not int or not 0 < entry["size"] <= 64 * 1024 * 1024 * 1024:
            raise ValueError("PACKAGED_ARTIFACT_REJECTED")
        path = admitted.paths.artifacts / name
        size, digest = stable_file_digest(path, entry["size"])
        if size != entry["size"] or digest != entry["sha256"]:
            raise ValueError("PACKAGED_ARTIFACT_REJECTED")
        if entry["role"] not in {"final_model", "tokenizer"}:
            contents[entry["role"]] = stable_read(path, min(entry["size"], 4 * 1024 * 1024))
        else:
            contents[entry["role"]] = (path, size, digest)
    if contents["workload_record"] != admitted.workload_bytes:
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    lineage = _document(contents["training_lineage"], 4 * 1024 * 1024)
    lineage_fields = {"schema_version", *_bindings(admitted), "prepared_input", "model_snapshot", "trainer_projection", "trainer_lineage",
                      "execution_evidence", "execution_evidence_sha256", "status", "trainer_exit_code"}
    if set(lineage) != lineage_fields:
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    for key, value in _bindings(admitted).items():
        if lineage.get(key) != value:
            raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    if (lineage.get("schema_version") != LINEAGE_SCHEMA or lineage.get("status") != "completed"
            or type(lineage.get("trainer_exit_code")) is not int or lineage["trainer_exit_code"] != 0
            or lineage.get("prepared_input") != admitted.execution.to_dict()["prepared_input"]
            or lineage.get("execution_evidence_sha256") != _digest(_canonical(lineage.get("execution_evidence")))
            or lineage.get("trainer_lineage", {}).get("synaptic_runtime_projection") != lineage.get("trainer_projection")):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    compiled = _admit_contracts(admitted.release, admitted.provider_binding, admitted.execution,
                               admitted.workload_bytes, admitted.artifact_policy)
    projection_workload = _runtime_projection(compiled)
    evidence = lineage["execution_evidence"]
    if evidence.get("schema_version") != "synaptic-packaged-sft-execution-evidence/v1":
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    dataset_path = evidence.get("dataset", {}).get("resolved_path")
    import re
    from pathlib import PurePosixPath
    if type(dataset_path) is not str or re.fullmatch(r"/proc/self/fd/[1-9][0-9]*", dataset_path) is None:
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    run = admitted.paths.state / "runtime-v1-trainer" / "output" / "runtime-v1"
    expected_projection = core._expected_trainer_projection(projection_workload,
        dataset_path=PurePosixPath(dataset_path), run_dir=run, final_model_dir=run / "final_model")
    if _canonical(lineage["trainer_projection"]) != _canonical(expected_projection):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    config = projection_workload.document["configuration"]["document"]
    _, snapshot = _private_manifest(lineage["model_snapshot"], admitted.paths, config["model"])
    retained_copy = _retain_private_snapshot(lineage["model_snapshot"], admitted.paths, config["model"])
    try:
        retained_copy.check()
    finally:
        retained_copy.close()
    argv, env, _, _, _, final = _invocation_spec(admitted, projection_workload, snapshot, PurePosixPath(dataset_path), lineage["model_snapshot"])
    expected_invocation = SimpleNamespace(argv=argv, environment=tuple(sorted(env.items())), cwd=admitted.paths.tmp,
        run_dir=run, final_model_dir=final, tokenizer_dir=final, lineage_path=run / "training_lineage.json")
    expected_evidence = core._build_execution_evidence(projection_workload, expected_invocation, SimpleNamespace(exit_code=0))
    expected_evidence["schema_version"] = "synaptic-packaged-sft-execution-evidence/v1"
    if _canonical(evidence) != _canonical(expected_evidence):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    if (evidence.get("workload_fingerprint") != admitted.execution.workload_digest
            or evidence.get("configuration_revision") != admitted.execution.configuration_digest
            or _canonical(evidence.get("model")) != _canonical(config["model"])
            or _canonical(evidence.get("sft")) != _canonical(config["sft"])
            or evidence.get("result") != {"exit_code": 0, "status": "completed"}
            or evidence.get("cwd") != str(admitted.paths.tmp)):
        raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    _document(contents["training_metrics"], 4 * 1024 * 1024)
    model_ref = _document(admitted.workload_bytes)["configuration"]["document"]["model"]["ref"]
    for role, kind in (("final_model", "model"), ("tokenizer", "tokenizer")):
        path, size, digest = contents[role]
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            _, valid = _validate_sft_archive_stream(stream, kind, locked_model_ref=model_ref)
            after = os.fstat(stream.fileno())
        if not valid or core._file_identity(before) != core._file_identity(after) or stable_file_digest(path, size) != (size, digest):
            raise ValueError("PACKAGED_ARTIFACT_REJECTED")
    return True
