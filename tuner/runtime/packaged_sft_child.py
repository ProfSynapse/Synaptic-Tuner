"""Isolated installed-distribution child. No source checkout or Git bootstrap."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.metadata
import os
from pathlib import Path
import sys
from io import BytesIO
from zipfile import ZipFile

from tuner.runtime.packaged_worker_closure import stable_read

_OWNED_PREFIXES = frozenset({"tuner", "synaptic_tuner", "shared", "Trainers", "SynthChat", "Evaluator", "MechInterp", "configs", "src"})


class _OwnedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, expected):
        super().__init__(fullname, path)
        self.expected = expected

    def get_code(self, fullname):
        from tuner.runtime.packaged_sft_execution import _digest
        raw = stable_read(Path(self.path))
        if _digest(raw) != self.expected:
            raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
        return self.source_to_code(raw, self.path)


class _OwnedImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, members):
        self.members = members

    def find_spec(self, fullname, path=None, target=None):
        # The historical trainer calls optional dotenv discovery during its
        # bootstrap. Never let that search reintroduce ambient credentials.
        if fullname == "dotenv" or fullname.startswith("dotenv."):
            raise ImportError("PACKAGED_CHILD_DOTENV_REJECTED")
        if fullname.split(".")[0] not in _OWNED_PREFIXES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ImportError("PACKAGED_CHILD_IMPORT_REJECTED")
        if spec.origin is None:
            if not spec.submodule_search_locations or any(
                not any(Path(location) in member.parents for member in self.members)
                for location in spec.submodule_search_locations
            ):
                raise ImportError("PACKAGED_CHILD_IMPORT_REJECTED")
            return spec
        origin = Path(spec.origin)
        if origin not in self.members or origin.suffix != ".py":
            raise ImportError("PACKAGED_CHILD_IMPORT_REJECTED")
        spec.loader = _OwnedSourceLoader(fullname, str(origin), self.members[origin])
        return spec


def _installed_import_guard(release, trainer):
    import json
    from tuner.runtime.packaged_sft_execution import _digest, _require_trainer_assets
    distribution = importlib.metadata.distribution(release.package_name)
    members = {}
    expected = json.loads(stable_read(Path("/opt/synaptic-runtime/build-inputs.json")))
    wheel = stable_read(Path("/opt/synaptic-runtime") / expected["wheel"]["filename"], 256 * 1024 * 1024)
    if _digest(wheel) != release.package_digest:
        raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
    with ZipFile(BytesIO(wheel)) as archive:
        if _require_trainer_assets(distribution, archive) != trainer:
            raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
        for member in archive.infolist():
            if member.filename.endswith(".py"):
                path = Path(distribution.locate_file(member.filename))
                expected_hash = _digest(archive.read(member))
                if _digest(stable_read(path)) != expected_hash:
                    raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
                members[path] = expected_hash
    if trainer not in members:
        raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
    for directory in (trainer.parent / "configs", trainer.parent / "src"):
        # No unowned directory or linked search root may enter sys.path.
        if directory.is_symlink() or directory.resolve(strict=True) != directory or not any(directory in path.parents for path in members):
            raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
    for name, module in tuple(sys.modules.items()):
        if name == "dotenv" or name.startswith("dotenv."):
            raise ValueError("PACKAGED_CHILD_DOTENV_REJECTED")
        if name.split(".")[0] not in _OWNED_PREFIXES:
            continue
        origin = getattr(module, "__file__", None)
        if origin is not None and Path(origin) not in members:
            raise ValueError("PACKAGED_CHILD_IMPORT_REJECTED")
    return _OwnedImportGuard(members)


def run_packaged_child(argv=None):
    from tuner.runtime.packaged_sft_execution import (
        _document, _digest, _inspect_release, _admit_contracts, _runtime_projection,
        _invocation_spec, _canonical, PackagedSFTPaths, _retain_private_snapshot,
        _private_manifest, _MAX_CHILD_BYTES,
    )
    from tuner.runtime.packaged_training_worker import admit_packaged_training_release
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1, ProviderRuntimeBindingV1, PackagedExecutionBindingV1
    from tuner.training.contracts import ArtifactPolicy
    from Trainers.sft import runtime_v1 as core
    from types import SimpleNamespace
    from pathlib import PurePosixPath
    import re
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 7 or args[0] != "--transport" or args[2] != "--digest" or args[4] != "--" or not sys.flags.isolated:
        raise ValueError
    raw = stable_read(Path(args[1]), _MAX_CHILD_BYTES)
    if _digest(raw) != args[3]:
        raise ValueError
    payload = _document(raw, _MAX_CHILD_BYTES)
    if set(payload) != {"release", "release_digest", "arguments", "provider_binding", "execution_binding", "workload", "artifact_policy", "paths", "environment", "model_snapshot"} or payload["arguments"] != args[5:]:
        raise ValueError
    release = PackagedTrainingRuntimeReleaseV1.from_dict(payload["release"])
    release = admit_packaged_training_release(release.canonical_bytes(), expected_release_digest=payload["release_digest"])
    trainer = _inspect_release(release)
    provider = ProviderRuntimeBindingV1.from_dict(payload["provider_binding"])
    execution = PackagedExecutionBindingV1.from_dict(payload["execution_binding"])
    policy_document = payload["artifact_policy"]
    if set(policy_document) != {"required_kinds", "retain_checkpoints"} or type(policy_document["required_kinds"]) is not list:
        raise ValueError
    policy = ArtifactPolicy(tuple(policy_document["required_kinds"]), policy_document["retain_checkpoints"])
    workload_bytes = _canonical(payload["workload"])
    compiled = _admit_contracts(release, provider, execution, workload_bytes, policy)
    paths = PackagedSFTPaths(**{key: Path(value) for key, value in payload["paths"].items()})
    projected = _runtime_projection(compiled)
    dataset = payload["arguments"][payload["arguments"].index("--local-file") + 1]
    match = re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", dataset)
    if match is None:
        raise ValueError
    # Only a sealed inherited descriptor can supply prepared training bytes.
    import fcntl
    fd = int(match.group(1))
    seals = core._LINUX_F_SEAL_SEAL | core._LINUX_F_SEAL_SHRINK | core._LINUX_F_SEAL_GROW | core._LINUX_F_SEAL_WRITE
    if fcntl.fcntl(fd, core._LINUX_F_GET_SEALS) & seals != seals:
        raise ValueError
    content = os.pread(fd, execution.prepared_input_size_bytes + 1, 0)
    if len(content) != execution.prepared_input_size_bytes or _digest(content) != execution.prepared_input_content_digest:
        raise ValueError
    physical = SimpleNamespace(release=release, provider_binding=provider, execution=execution,
        artifact_policy=policy, workload_bytes=workload_bytes, paths=paths,
        environment=tuple(tuple(item) for item in payload["environment"]))
    if not set(dict(physical.environment)) <= {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL"}:
        raise ValueError
    model = compiled.document["configuration"]["document"]["model"]
    _, snapshot = _private_manifest(payload["model_snapshot"], paths, model)
    expected_argv, expected_environment, child_path, expected_raw, _, _ = _invocation_spec(physical, projected, snapshot, PurePosixPath(dataset), payload["model_snapshot"])
    if expected_raw != raw or list(expected_argv[4:]) != args or dict(os.environ) != expected_environment:
        raise ValueError
    guard = _installed_import_guard(release, trainer)
    allowed = {"PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LANG", "LC_ALL",
        "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE", "HF_HOME", "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_DISABLED", "TMPDIR", "HOME", "SYNAPTIC_MODEL_SNAPSHOT"}
    if not set(os.environ) <= allowed or any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE", "PYTHONSAFEPATH")):
        raise ValueError
    sys.meta_path.insert(0, guard)
    sys.dont_write_bytecode = True
    # train_sft's historical configs/src aliases live beside the installed
    # entrypoint. Their import origins remain guarded against substitution.
    sys.path.insert(0, str(trainer.parent))
    sys.argv = [str(trainer), *payload["arguments"]]
    code = _OwnedSourceLoader("__main__", str(trainer), guard.members[trainer]).get_code("__main__")
    _run_private_trainer(code, trainer, payload["model_snapshot"], paths, model)
    return 0


def _run_private_trainer(code, trainer, manifest, paths, model):
    from tuner.runtime.packaged_sft_execution import _retain_private_snapshot
    private_copy = _retain_private_snapshot(manifest, paths, model)
    try:
        private_copy.check()
        exec(code, {"__name__": "__main__", "__file__": str(trainer), "__package__": None})
    finally:
        try:
            private_copy.check()
        finally:
            private_copy.close()


def _check_local_cpu_environment(release):
    from tuner.runtime.packaged_training_worker import local_cpu_environment
    if dict(os.environ) != local_cpu_environment(release): raise ValueError


def _check_no_gpu_devices():
    """Reject GPU device names without following entries or trusting runtime defaults."""
    import stat
    if os.name != "posix": raise ValueError
    descriptor = os.open("/dev", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        held = os.fstat(descriptor)
        named = os.stat("/dev", follow_symlinks=False)
        if (not stat.S_ISDIR(named.st_mode) or getattr(named, "st_file_attributes", 0) & 0x400
                or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)): raise ValueError
        seen = set()
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = entry.name
                if (len(seen) >= 4096 or name in seen or name.startswith("nvidia") or name in {"dri", "kfd"}):
                    raise ValueError
                seen.add(name)
        named = os.stat("/dev", follow_symlinks=False)
        if (not stat.S_ISDIR(named.st_mode) or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)):
            raise ValueError
    finally:
        os.close(descriptor)


def run_local_cpu_child(args):
    """Separate diagnostic protocol; cannot emit training artifacts or success markers."""
    import fcntl
    import stat
    from tuner.runtime.packaged_training_worker import (
        LOCAL_CPU_PROTOCOL, LOCAL_CPU_DATA, LOCAL_CPU_MODEL, local_cpu_environment,
        local_cpu_result, admit_packaged_training_release,
    )
    from tuner.runtime.packaged_sft_execution import _canonical, _document, _digest, _inspect_release, _HeldDirectory, _HeldModelFile
    from tuner.runtime.releases import PackagedTrainingRuntimeReleaseV1
    if len(args) != 3 or args[0] != "--qualify-local" or not sys.flags.isolated or os.name != "posix": raise ValueError
    raw = stable_read(Path(args[1]), 256 * 1024)
    if _digest(raw) != args[2]: raise ValueError
    payload = _document(raw)
    if set(payload) != {"schema_version", "release", "input_fd", "root", "root_identity", "member"} or payload["schema_version"] != LOCAL_CPU_PROTOCOL: raise ValueError
    release = PackagedTrainingRuntimeReleaseV1.from_dict(payload["release"])
    release = admit_packaged_training_release(release.canonical_bytes(), expected_release_digest=release.manifest_digest)
    _check_local_cpu_environment(release)
    trainer = _inspect_release(release)
    fd = payload["input_fd"]
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    if type(fd) is not int or fd < 3 or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals != seals or os.pread(fd, len(LOCAL_CPU_DATA) + 1, 0) != LOCAL_CPU_DATA: raise ValueError
    root = Path(payload["root"])
    if root.parent != Path("/tmp") or not root.name.startswith("qualification-") or Path(args[1]) != root / "transport.json": raise ValueError
    held = _HeldDirectory(root)
    leaf = None
    try:
        info = os.fstat(held.fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700 or list(held.identity) != payload["root_identity"]: raise ValueError
        member = payload["member"]
        if (set(member) != {"path", "size_bytes", "sha256", "device", "inode"}
                or member["path"] != "fixture.json" or member["size_bytes"] != len(LOCAL_CPU_MODEL)
                or member["sha256"] != _digest(LOCAL_CPU_MODEL)): raise ValueError
        leaf = _HeldModelFile(root / "fixture.json", held, member)
        guard = _installed_import_guard(release, trainer)
        _check_no_gpu_devices()
        _OwnedSourceLoader("__main__", str(trainer), guard.members[trainer]).get_code("__main__")
        _check_no_gpu_devices()
        held.check(); leaf.check()
        sys.stdout.buffer.write(_canonical(local_cpu_result(release, guard.members[trainer])))
        return 0
    finally:
        if leaf is not None: leaf.close()
        held.close()


def main():
    try:
        if sys.argv[1:2] == ["--qualify-local"]:
            return run_local_cpu_child(sys.argv[1:])
        return run_packaged_child()
    except SystemExit as error:
        if type(error) is SystemExit and (error.code is None or (type(error.code) is int and error.code == 0)):
            return 0
    except BaseException:
        pass
    print("PACKAGED_SFT_CHILD_REJECTED", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
