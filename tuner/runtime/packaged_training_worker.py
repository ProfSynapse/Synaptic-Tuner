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
import platform
import re
import sys
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from tuner.runtime.packaged_worker_closure import (
    PackagedWorkerClosureError,
    load_packaged_worker_closure,
    stable_read,
)
from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1


PACKAGED_TRAINING_WORKER_ENTRYPOINT = "tuner.runtime.packaged_training_worker:main"
_MAX_RELEASE_BYTES = 128 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class PackagedTrainingWorkerError(RuntimeError):
    """Fail-closed packaged-runtime admission rejection."""


def admit_packaged_training_release(
    payload: bytes, *, expected_release_digest: str
) -> PackagedTrainingRuntimeReleaseV1:
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
        release = PackagedTrainingRuntimeReleaseV1.from_json(payload.decode("utf-8"))
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


def qualify_installed_child(release_document):
    """Diagnostic only: exercise the installed child, never synthesize training evidence."""
    import os
    import fcntl
    import subprocess
    import tempfile
    from tuner.runtime.packaged_sft_execution import _canonical, _digest, _inspect_release, _HeldDirectory, _HeldModelFile
    release = PackagedTrainingRuntimeReleaseV1.from_dict(release_document)
    release = admit_packaged_training_release(release.canonical_bytes(), expected_release_digest=release.manifest_digest)
    trainer = _inspect_release(release)
    if os.name != "posix" or not sys.flags.isolated:
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
                    child = subprocess.run(command, env=local_cpu_environment(release), cwd=root,
                        pass_fds=(fd,), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=60, check=False)
                    stdout.seek(0); stderr.seek(0)
                    raw, errors = stdout.read(16385), stderr.read(16385)
                held.check(); leaf.check()
                expected = local_cpu_result(release, _digest(stable_read(trainer)))
                if child.returncode != 0 or errors or raw != _canonical(expected): raise ValueError
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


def inspect_installed_runtime(expected: dict) -> dict:
    """Measure reviewed wheel bytes, installed members and worker closure.

    Called only by the fixed image inspector after exact-interpreter admission.
    Inputs are build-bound profile data; there is no release digest circularity.
    """
    def canonical(value):
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    retained = stable_read(Path("/opt/synaptic-runtime/build-inputs.json"))
    if retained != canonical(expected):
        raise ValueError("build inputs differ")
    provenance = {}
    from packaging.requirements import Requirement
    bootstrap_versions = {item["distribution"]: item["version"] for item in expected["bootstrap"]}
    for wheel in [expected["wheel"], *expected["bootstrap"]]:
        wheel_raw = stable_read(Path("/opt/synaptic-runtime") / wheel["filename"], 256 * 1024 * 1024)
        if hashlib.sha256(wheel_raw).hexdigest() != wheel["sha256"]:
            raise ValueError("wheel differs")
        distribution = importlib.metadata.distribution(wheel["distribution"])
        if distribution.version != wheel["version"]:
            raise ValueError("installed version differs")
        if wheel["distribution"] in bootstrap_versions:
            for text in distribution.requires or ():
                requirement = Requirement(text)
                if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                    continue
                name = re.sub(r"[-_.]+", "-", requirement.name.lower())
                version = bootstrap_versions.get(name)
                if requirement.url or requirement.extras or version is None or version not in requirement.specifier:
                    raise ValueError("bootstrap transitive closure incomplete")
        direct_file = next((item for item in distribution.files or () if str(item).replace("\\", "/").endswith(".dist-info/direct_url.json")), None)
        if direct_file is None:
            raise ValueError("missing wheel provenance")
        direct_raw = stable_read(Path(distribution.locate_file(direct_file)))
        direct = json.loads(direct_raw)
        if (direct.get("url") != "file:///opt/synaptic-runtime/" + wheel["filename"]
                or direct.get("archive_info", {}).get("hashes", {}).get("sha256") != wheel["sha256"]):
            raise ValueError("wheel provenance differs")
        provenance[wheel["distribution"]] = hashlib.sha256(direct_raw).hexdigest()
        with ZipFile(BytesIO(wheel_raw)) as archive:
            members = archive.infolist()
            if not members or len(members) > 10000 or sum(member.file_size for member in members) > 256 * 1024 * 1024:
                raise ValueError("wheel inventory limit")
            seen = set()
            for member in members:
                name = member.filename
                if member.is_dir(): continue
                if (name in seen or name.startswith("/") or "\\" in name or ".." in name.split("/")
                        or any(part.endswith(".data") for part in name.split("/"))
                        or member.file_size > 64 * 1024 * 1024):
                    raise ValueError("unsupported wheel member")
                seen.add(name)
                # pip rewrites RECORD with installation-generated files.
                if name.endswith(".dist-info/RECORD"): continue
                installed = Path(distribution.locate_file(name))
                payload = stable_read(installed, max(1, member.file_size))
                if payload != archive.read(member):
                    raise ValueError("installed wheel member differs")
    closure = load_packaged_worker_closure()
    inventory = sorted([{"name": re.sub(r"[-_.]+", "-", item.metadata["Name"].lower()), "version": item.version}
                        for item in importlib.metadata.distributions()], key=lambda item: item["name"])
    if not inventory or len(inventory) > 4096 or len({item["name"] for item in inventory}) != len(inventory):
        raise ValueError("distribution inventory invalid")
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
    "PackagedTrainingWorkerError",
    "admit_packaged_training_release",
    "admit_packaged_sft",
    "execute_admitted_packaged_sft",
    "main",
]
