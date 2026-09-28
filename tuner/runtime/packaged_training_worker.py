"""Provider-neutral admission boundary for an installed packaged training runtime.

This module deliberately has no provider SDK, shell, Git, source-copy, package
installation, or credential handling.  Slice D supplies the canonical workload
transport; until then this boundary only admits the immutable runtime release
that a provider-specific worker composition has already made available.
"""

from __future__ import annotations

import hmac
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import sysconfig
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tuner.runtime.packaged_worker_closure import (
    PackagedWorkerClosureError,
    load_packaged_worker_closure,
    stable_read,
)
from tuner.runtime.releases import PackagedRuntimeRelease, parse_packaged_runtime_release


PACKAGED_TRAINING_WORKER_ENTRYPOINT = "tuner.runtime.packaged_training_worker:main"
_MAX_RELEASE_BYTES = 128 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class PackagedTrainingWorkerError(RuntimeError):
    """Fail-closed packaged-runtime admission rejection."""


class PackagedLocalCPUStageError(PackagedTrainingWorkerError):
    """Closed, non-authorizing boundary code for local CPU qualification."""

    def __init__(self, stage: str) -> None:
        if stage not in ("PARENT_RELEASE", "CHILD_RESULT"):
            raise ValueError("invalid local CPU stage")
        self.stage = stage
        super().__init__("PACKAGED_LOCAL_CPU_REJECTED")


def admit_packaged_training_release(
    payload: bytes, *, expected_release_digest: str
) -> PackagedRuntimeRelease:
    """Authenticate one canonical release against the installed worker closure.

    The caller supplies bytes from the immutable image/package layer, not a
    path, environment value, provider object, or host source tree.  Workload
    and mount admission are intentionally deferred to the shared Slice D
    transport so this module does not invent a parallel execution contract.
    """

    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_RELEASE_BYTES:
        raise PackagedTrainingWorkerError("PACKAGED_RELEASE_REJECTED")
    if type(expected_release_digest) is not str or _DIGEST.fullmatch(expected_release_digest) is None:
        raise PackagedTrainingWorkerError("PACKAGED_RELEASE_REJECTED")
    try:
        document = json.loads(payload.decode("utf-8"))
        release = parse_packaged_runtime_release(document)
        if release.canonical_bytes() != payload:
            raise ValueError
        closure = load_packaged_worker_closure()
    except BaseException as exc:
        # The boundary has one externally observable rejection.  In particular,
        # resource/manifest-loader faults must not leak into worker diagnostics.
        raise PackagedTrainingWorkerError("PACKAGED_RELEASE_REJECTED") from None
    if (
        release.worker_entrypoint != PACKAGED_TRAINING_WORKER_ENTRYPOINT
        or not hmac.compare_digest(release.manifest_digest, expected_release_digest)
        or not hmac.compare_digest(release.worker_closure_digest, closure.digest)
    ):
        raise PackagedTrainingWorkerError("PACKAGED_RELEASE_REJECTED")
    return release


def main() -> int:
    """Reserved fixed entrypoint; dispatch framing is owned by Slice D."""

    # A release cannot be accepted from argv or an ambient environment.  The
    # eventual provider-neutral dispatch composition calls the admission helper
    # with its embedded immutable bytes before any training side effects.
    print("PACKAGED_TRAINING_WORKER_REJECTED")
    return 2


def admit_packaged_sft(**kwargs):
    """Lazily enter the complete installed-package SFT admission boundary."""
    from tuner.runtime.packaged_sft_execution import admit_packaged_sft as admit
    return admit(**kwargs)


def execute_admitted_packaged_sft(admitted, **kwargs):
    """Execute a previously admitted provider-neutral SFT workload."""
    from tuner.runtime.packaged_sft_execution import execute_admitted_packaged_sft as execute
    return execute(admitted, **kwargs)


LOCAL_CPU_PROTOCOL = "synaptic-installed-child-cpu/v1"
LOCAL_CPU_DATA = b'{"text":"local qualification only"}\n'
LOCAL_CPU_MODEL = b'{"qualification_fixture":true}\n'
_MAX_BUILD_INPUTS_BYTES = 128 * 1024
_MAX_WHEEL_BYTES = 256 * 1024 * 1024
_MAX_TRAINER_BYTES = 4 * 1024 * 1024
_TRAINER_MEMBER = "Trainers/sft/train_sft.py"
_WHEEL_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl\Z")


def local_cpu_environment(release):
    from pathlib import PurePosixPath
    return {"PATH": ":".join(dict.fromkeys((str(PurePosixPath(release.python_executable).parent),
            "/usr/local/bin", "/usr/bin", "/bin"))), "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": ""}


def local_cpu_result(release, trainer_digest):
    return {"schema_version": LOCAL_CPU_PROTOCOL, "runtime_release_digest": release.manifest_digest,
            "dataset_sha256": hashlib.sha256(LOCAL_CPU_DATA).hexdigest(),
            "snapshot_sha256": hashlib.sha256(LOCAL_CPU_MODEL).hexdigest(),
            "trainer_sha256": trainer_digest, "isolated": True, "sealed_input": True,
            "private_snapshot": True, "trainer_source_compiled": True,
            "container_runtime": "runc", "nvidia_visible_devices": "void", "cuda_visible_devices": "",
            "gpu_device_namespace_absent": True,
            "training_executed": False, "gpu_qualified": False, "provider_qualified": False}


def _parent_trainer_reference(release) -> str:
    """Derive only the pinned trainer reference; installed admission belongs to -I child."""
    raw = stable_read(Path("/opt/synaptic-runtime/build-inputs.json"), _MAX_BUILD_INPUTS_BYTES)
    def reject_constant(_value):
        raise ValueError
    inputs = json.loads(raw, parse_constant=reject_constant)
    if (type(inputs) is not dict
            or (json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
                + "\n").encode("ascii") != raw):
        raise ValueError
    wheel = inputs.get("wheel")
    if type(wheel) is not dict or set(wheel) != {"filename", "distribution", "version", "sha256"}:
        raise ValueError
    filename, digest = wheel["filename"], wheel["sha256"]
    if (type(filename) is not str or len(filename) > 255
            or _WHEEL_BASENAME.fullmatch(filename) is None
            or type(digest) is not str or _DIGEST.fullmatch(digest) is None
            or not hmac.compare_digest(digest, release.package_digest)):
        raise ValueError
    wheel_raw = stable_read(Path("/opt/synaptic-runtime") / filename, _MAX_WHEEL_BYTES)
    if not hmac.compare_digest(hashlib.sha256(wheel_raw).hexdigest(), digest):
        raise ValueError
    with ZipFile(BytesIO(wheel_raw)) as archive:
        members = archive.infolist()
        if not members or len(members) > 10000:
            raise ValueError
        trainer = [member for member in members if member.filename == _TRAINER_MEMBER]
        if len(trainer) != 1 or trainer[0].is_dir() or not 0 < trainer[0].file_size <= _MAX_TRAINER_BYTES:
            raise ValueError
        raw_trainer = archive.read(trainer[0])
        if len(raw_trainer) != trainer[0].file_size:
            raise ValueError
    return hashlib.sha256(raw_trainer).hexdigest()


def qualify_installed_child(release_document):
    """Diagnostic only: exercise the installed child, never synthesize training evidence."""
    import os
    import fcntl
    import subprocess
    import tempfile
    from tuner.runtime.packaged_sft_execution import _canonical, _digest, _HeldDirectory, _HeldModelFile
    release = parse_packaged_runtime_release(release_document)
    release = admit_packaged_training_release(release.canonical_bytes(), expected_release_digest=release.manifest_digest)
    try:
        trainer_digest = _parent_trainer_reference(release)
    except BaseException:
        raise PackagedLocalCPUStageError("PARENT_RELEASE") from None
    if os.name != "posix":
        raise ValueError
    fd = os.memfd_create("qualification-input", os.MFD_ALLOW_SEALING)
    try:
        if os.write(fd, LOCAL_CPU_DATA) != len(LOCAL_CPU_DATA):
            raise ValueError
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        with tempfile.TemporaryDirectory(prefix="qualification-", dir="/tmp") as temporary:
            root = Path(temporary)
            held = _HeldDirectory(root)
            leaf = None
            try:
                target = os.open("fixture.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=held.fd)
                try:
                    if os.write(target, LOCAL_CPU_MODEL) != len(LOCAL_CPU_MODEL): raise ValueError
                    os.fsync(target)
                    os.fchmod(target, 0o400)
                    info = os.fstat(target)
                finally:
                    os.close(target)
                member = {"path": "fixture.json", "size_bytes": len(LOCAL_CPU_MODEL), "sha256": _digest(LOCAL_CPU_MODEL), "device": info.st_dev, "inode": info.st_ino}
                leaf = _HeldModelFile(root / "fixture.json", held, member)
                payload = _canonical({"schema_version": LOCAL_CPU_PROTOCOL, "release": release_document,
                    "input_fd": fd, "root": str(root), "root_identity": list(held.identity), "member": member})
                transport = os.open("transport.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400, dir_fd=held.fd)
                try:
                    if os.write(transport, payload) != len(payload): raise ValueError
                finally:
                    os.close(transport)
                command = [release.python_executable, "-I", "-m", "tuner.runtime.packaged_sft_child",
                           "--qualify-local", str(root / "transport.json"), _digest(payload)]
                # Output goes to bounded private tmpfs; never an unbounded PIPE allocation.
                with tempfile.TemporaryFile(dir=root) as stdout, tempfile.TemporaryFile(dir=root) as stderr:
                    try:
                        child = subprocess.run(command, env=local_cpu_environment(release), cwd=root,
                            pass_fds=(fd,), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=60, check=False)
                        stdout.seek(0); stderr.seek(0)
                        raw, errors = stdout.read(16385), stderr.read(16385)
                    except BaseException:
                        raise PackagedLocalCPUStageError("CHILD_RESULT") from None
                held.check(); leaf.check()
                expected = local_cpu_result(release, trainer_digest)
                if child.returncode != 0 or errors or raw != _canonical(expected):
                    raise PackagedLocalCPUStageError("CHILD_RESULT")
                return expected
            finally:
                if leaf is not None: leaf.close()
                held.close()
    finally:
        os.close(fd)


def local_cpu_main(release_document):
    try:
        from tuner.runtime.packaged_sft_execution import _canonical
        result = qualify_installed_child(release_document)
        sys.stdout.buffer.write(_canonical(result) + b"\n")
        return 0
    except BaseException:
        print("PACKAGED_LOCAL_CPU_REJECTED", file=sys.stderr)
        return 2


_DUPLICATE_REASONS = (
    "IDENTITY_UNPROVEN", "DISTINCT_PHYSICAL", "VERSION_MISMATCH",
    "PHYSICAL_METADATA_MISMATCH",
)
_DUPLICATE_PACKAGES = ("MAIN", "BOOTSTRAP", "OTHER")
_DUPLICATE_LOCATIONS = ("BOTH_IN", "CROSS_ROOT", "BOTH_OUT", "UNKNOWN")
_DUPLICATE_STAGES = frozenset(
    "INVENTORY_DUPLICATE_" + reason + "_" + package + "_" + location
    for reason in _DUPLICATE_REASONS
    for package in _DUPLICATE_PACKAGES
    for location in _DUPLICATE_LOCATIONS
)
INSTALLED_RUNTIME_INSPECTION_STAGES = frozenset({
    "INPUTS", "WHEEL_BYTES", "DISTRIBUTION", "BOOTSTRAP_DEPENDENCIES",
    "PROVENANCE", "MEMBERS", "CLOSURE", "INVENTORY_ENUMERATION",
    "INVENTORY_BOUNDS", "INVENTORY_DUPLICATE", "INVENTORY_ROOT_UNPROVEN",
}) | _DUPLICATE_STAGES


class PackagedInstalledRuntimeInspectionError(ValueError):
    """Fixed inner predicate for installed-runtime inspection."""

    def __init__(self, stage: str) -> None:
        if stage not in INSTALLED_RUNTIME_INSPECTION_STAGES:
            raise ValueError("invalid installed runtime inspection stage")
        self.stage = stage
        super().__init__("PACKAGED_INSTALLED_RUNTIME_INSPECTION_REJECTED")


def _metadata_identity(distribution: object) -> tuple[int, ...] | None:
    if type(distribution) is not importlib.metadata.PathDistribution:
        return None
    path = getattr(distribution, "_path", None)
    if type(path) is not type(Path()):
        return None
    try:
        info = path.stat()
    except (OSError, OverflowError, ValueError):
        return None
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        return None
    return (stat.S_IFMT(info.st_mode), info.st_dev, info.st_ino,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _reviewed_package_roots(expected: dict) -> tuple[tuple[Path, Path, tuple[int, int]], ...] | None:
    try:
        python = expected["python"]
        executable = python["executable"]
        if type(executable) is not str or not Path(executable).is_absolute():
            return None
        prefix = str(Path(executable).parent.parent)
        if sys.prefix != prefix:
            return None
        paths = sysconfig.get_paths(scheme="venv", vars={"base": prefix, "platbase": prefix})
        roots = []
        for key in ("purelib", "platlib"):
            raw = python[key]
            if (type(raw) is not str or not Path(raw).is_absolute()
                    or os.path.normpath(raw) != raw or paths[key] != raw):
                return None
            selected = Path(raw)
            root = selected.resolve(strict=True)
            info = root.stat()
            if not stat.S_ISDIR(info.st_mode):
                return None
            roots.append((selected, root, (info.st_dev, info.st_ino)))
        return tuple(roots)
    except BaseException:
        return None


def _roots_stable(roots: tuple[tuple[Path, Path, tuple[int, int]], ...]) -> bool:
    for selected, resolved, identity in roots:
        if selected.resolve(strict=True) != resolved:
            return False
        info = selected.stat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
            return False
    return True


def _metadata_within_roots(
    distribution: object, identity: tuple[int, ...] | None,
    roots: tuple[tuple[Path, Path, tuple[int, int]], ...] | None,
) -> bool | None:
    if identity is None or roots is None:
        return None
    try:
        path = distribution._path
        if not _roots_stable(roots) or _metadata_identity(distribution) != identity:
            return None
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        observed = (stat.S_IFMT(info.st_mode), info.st_dev, info.st_ino,
                    info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if (observed != identity or path.resolve(strict=True) != resolved
                or not _roots_stable(roots)
                or _metadata_identity(distribution) != identity):
            return None
        return any(resolved == root or root in resolved.parents for _selected, root, _identity in roots)
    except BaseException:
        return None


def _duplicate_stage(reason: str, package: str, first: bool | None, second: bool | None) -> str:
    location = ("UNKNOWN" if first is None or second is None else
                "BOTH_IN" if first and second else
                "BOTH_OUT" if not first and not second else "CROSS_ROOT")
    return "INVENTORY_DUPLICATE_" + reason + "_" + package + "_" + location


def _reviewed_root_inventory(expected: dict) -> list[dict[str, str]] | None:
    """Select only metadata proved inside the retained venv library roots."""
    try:
        roots = _reviewed_package_roots(expected)
        if roots is None or not _roots_stable(roots):
            return None
        paths = tuple(dict.fromkeys(str(selected) for selected, _resolved, _identity in roots))
        context = importlib.metadata.DistributionFinder.Context(path=paths)
        iterator = importlib.metadata.MetadataPathFinder.find_distributions(context)
        by_name = {}
        identities = {}
        physical = {}
        observations = []
        for occurrence, item in enumerate(iterator, 1):
            if occurrence > 4096:
                return None
            identity = _metadata_identity(item)
            if identity is None or _metadata_within_roots(item, identity, roots) is not True:
                return None
            raw_name, version = item.metadata["Name"], item.version
            if type(raw_name) is not str or not raw_name or type(version) is not str or not version:
                return None
            name = re.sub(r"[-_.]+", "-", raw_name.lower())
            if _metadata_identity(item) != identity:
                return None
            prior_physical = physical.get(identity)
            if prior_physical is not None and prior_physical != (name, version):
                return None
            physical[identity] = (name, version)
            observations.append((item, identity, name, version))
            previous = by_name.get(name)
            if previous is not None:
                if previous["version"] != version or identities[name] != identity:
                    return None
                continue
            by_name[name] = {"name": name, "version": version}
            identities[name] = identity
        if not by_name or len(by_name) > 4096 or not _roots_stable(roots):
            return None
        for item, identity, name, version in observations:
            if (_metadata_identity(item) != identity
                    or _metadata_within_roots(item, identity, roots) is not True
                    or re.sub(r"[-_.]+", "-", item.metadata["Name"].lower()) != name
                    or item.version != version):
                return None
        return sorted(by_name.values(), key=lambda item: item["name"])
    except BaseException:
        return None


def inspect_reviewed_root_inventory(release: PackagedRuntimeRelease) -> dict[str, object] | None:
    """Diagnostic-only digest of proved metadata in reviewed venv roots."""
    try:
        retained = stable_read(Path("/opt/synaptic-runtime/build-inputs.json"))
        expected = json.loads(retained)
        if (type(expected) is not dict
                or retained != (json.dumps(expected, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")):
            return None
        python = expected["python"]
        if (python["implementation"] != release.python_implementation
                or python["version"] != release.python_version
                or python["executable"] != release.python_executable
                or python["executable_digest"] != release.python_executable_digest):
            return None
        inventory = _reviewed_root_inventory(expected)
        if inventory is None:
            return None
        return {"digest": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(),
                "count": len(inventory)}
    except BaseException:
        return None


def _ambient_inventory(expected: dict) -> list[dict[str, str]]:
    inventory_by_name = {}
    identities = {}
    locations = {}
    physical = {}
    roots = _reviewed_package_roots(expected)
    main_package = re.sub(r"[-_.]+", "-", expected["wheel"]["distribution"].lower())
    bootstrap_packages = {
        re.sub(r"[-_.]+", "-", wheel["distribution"].lower())
        for wheel in expected["bootstrap"]
    }
    def category(*names):
        return ("MAIN" if main_package in names else
                "BOOTSTRAP" if any(name in bootstrap_packages for name in names) else "OTHER")
    try:
        for occurrence, item in enumerate(importlib.metadata.distributions(), 1):
            if occurrence > 4096:
                raise PackagedInstalledRuntimeInspectionError("INVENTORY_BOUNDS")
            before = _metadata_identity(item)
            name = re.sub(r"[-_.]+", "-", item.metadata["Name"].lower())
            version = item.version
            after = _metadata_identity(item)
            if before != after and (before is not None or after is not None):
                raise PackagedInstalledRuntimeInspectionError("INVENTORY_ENUMERATION")
            identity = before if before is not None and before == after else None
            location = _metadata_within_roots(item, identity, roots)
            previous = inventory_by_name.get(name)
            if previous is not None:
                if previous["version"] != version:
                    reason = "VERSION_MISMATCH"
                elif identity is None or identities.get(name) is None:
                    reason = "IDENTITY_UNPROVEN"
                elif identities[name] != identity:
                    reason = "DISTINCT_PHYSICAL"
                else:
                    continue
                raise PackagedInstalledRuntimeInspectionError(
                    _duplicate_stage(reason, category(name), locations[name], location))
            if identity is not None:
                prior_physical = physical.get(identity)
                if prior_physical is not None and prior_physical[:2] != (name, version):
                    raise PackagedInstalledRuntimeInspectionError(
                        _duplicate_stage("PHYSICAL_METADATA_MISMATCH",
                                         category(name, prior_physical[0]),
                                         prior_physical[2], location))
                physical[identity] = (name, version, location)
                identities[name] = identity
            locations[name] = location
            inventory_by_name[name] = {"name": name, "version": version}
    except PackagedInstalledRuntimeInspectionError:
        raise
    except BaseException:
        raise PackagedInstalledRuntimeInspectionError("INVENTORY_ENUMERATION") from None
    inventory = sorted(inventory_by_name.values(), key=lambda item: item["name"])
    if not inventory or len(inventory) > 4096:
        raise PackagedInstalledRuntimeInspectionError("INVENTORY_BOUNDS") from None
    return inventory


def inspect_installed_runtime(expected: dict, *, inventory_scope: str = "ambient") -> dict:
    """Measure reviewed wheel bytes, installed members and worker closure.

    Called only by the fixed image inspector after exact-interpreter admission.
    Inputs are build-bound profile data; there is no release digest circularity.
    """
    if type(inventory_scope) is not str or inventory_scope not in {"ambient", "reviewed_roots"}:
        raise PackagedInstalledRuntimeInspectionError("INPUTS") from None
    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    try:
        retained = stable_read(Path("/opt/synaptic-runtime/build-inputs.json"))
        if retained != canonical(expected):
            raise ValueError
    except BaseException:
        raise PackagedInstalledRuntimeInspectionError("INPUTS") from None
    provenance = {}
    try:
        from packaging.requirements import Requirement
        bootstrap_versions = {item["distribution"]: item["version"] for item in expected["bootstrap"]}
    except BaseException:
        raise PackagedInstalledRuntimeInspectionError("INPUTS") from None
    for wheel in [expected["wheel"], *expected["bootstrap"]]:
        try:
            wheel_raw = stable_read(Path("/opt/synaptic-runtime") / wheel["filename"], 256 * 1024 * 1024)
            if hashlib.sha256(wheel_raw).hexdigest() != wheel["sha256"]:
                raise ValueError
        except BaseException:
            raise PackagedInstalledRuntimeInspectionError("WHEEL_BYTES") from None
        try:
            distribution = importlib.metadata.distribution(wheel["distribution"])
            if distribution.version != wheel["version"]:
                raise ValueError
        except BaseException:
            raise PackagedInstalledRuntimeInspectionError("DISTRIBUTION") from None
        try:
            if wheel["distribution"] in bootstrap_versions:
                for text in distribution.requires or ():
                    requirement = Requirement(text)
                    if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                        continue
                    name = re.sub(r"[-_.]+", "-", requirement.name.lower())
                    version = bootstrap_versions.get(name)
                    if version is None:
                        # The pinned base may already provide a dependency.
                        version = importlib.metadata.distribution(requirement.name).version
                    if requirement.url or requirement.extras or version not in requirement.specifier:
                        raise ValueError
        except BaseException:
            raise PackagedInstalledRuntimeInspectionError("BOOTSTRAP_DEPENDENCIES") from None
        try:
            direct_file = next((item for item in distribution.files or () if str(item).replace("\\", "/").endswith(".dist-info/direct_url.json")), None)
            if direct_file is None:
                raise ValueError
            direct_raw = stable_read(Path(distribution.locate_file(direct_file)))
            direct = json.loads(direct_raw)
            if (direct.get("url") != "file:///opt/synaptic-runtime/" + wheel["filename"]
                    or direct.get("archive_info", {}).get("hashes", {}).get("sha256") != wheel["sha256"]):
                raise ValueError
            provenance[wheel["distribution"]] = hashlib.sha256(direct_raw).hexdigest()
        except BaseException:
            raise PackagedInstalledRuntimeInspectionError("PROVENANCE") from None
        try:
            with ZipFile(BytesIO(wheel_raw)) as archive:
                members = archive.infolist()
                if not members or len(members) > 10000 or sum(member.file_size for member in members) > 256 * 1024 * 1024:
                    raise ValueError
                seen = set()
                for member in members:
                    name = member.filename
                    if member.is_dir(): continue
                    if (name in seen or name.startswith("/") or "\\" in name or ".." in name.split("/")
                            or any(part.endswith(".data") for part in name.split("/"))
                            or member.file_size > 64 * 1024 * 1024):
                        raise ValueError
                    seen.add(name)
                    # pip rewrites RECORD with installation-generated files.
                    if name.endswith(".dist-info/RECORD"): continue
                    installed = Path(distribution.locate_file(name))
                    payload = stable_read(installed, max(1, member.file_size))
                    if payload != archive.read(member):
                        raise ValueError
        except BaseException:
            raise PackagedInstalledRuntimeInspectionError("MEMBERS") from None
    try:
        closure = load_packaged_worker_closure()
    except BaseException:
        raise PackagedInstalledRuntimeInspectionError("CLOSURE") from None
    if inventory_scope == "ambient":
        inventory = _ambient_inventory(expected)
    else:
        inventory = _reviewed_root_inventory(expected)
        if inventory is None:
            raise PackagedInstalledRuntimeInspectionError("INVENTORY_ROOT_UNPROVEN") from None
    python = {key: expected["python"][key] for key in ("implementation", "version", "executable", "executable_digest")}
    return {
        "schema_version": "synaptic-packaged-runtime-inspector/v1",
        "package": {"name": "synaptic-tuner", "version": expected["wheel"]["version"], "digest": expected["wheel"]["sha256"], "source_provenance_digest": provenance["synaptic-tuner"]},
        "python": python,
        "installed_distributions": {"inventory": inventory, "digest": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(), "count": len(inventory)},
        "platform": {"system": platform.system().lower(), "machine": platform.machine().lower(), "cuda_version": None, "runtime_facts": {"python_cache_tag": sys.implementation.cache_tag}},
        "worker": {"entrypoint": PACKAGED_TRAINING_WORKER_ENTRYPOINT, "closure_digest": closure.digest},
        "closure": {"verified_digest": closure.digest},
        "contracts": expected["capabilities"]["contracts"],
        "capabilities": expected["capabilities"],
        "build_inputs_digest": hashlib.sha256(retained).hexdigest(),
        "provenance": provenance,
    }


__all__ = [
    "PACKAGED_TRAINING_WORKER_ENTRYPOINT",
    "PackagedLocalCPUStageError",
    "PackagedTrainingWorkerError",
    "admit_packaged_training_release",
    "admit_packaged_sft",
    "execute_admitted_packaged_sft",
    "main",
]
