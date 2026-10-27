"""Isolated native worker entrypoint for the admitted owned-process transport.

Invoke with -I -S -B and an absolute private request path. This entrypoint grants
no authority itself: the host/Foundation must bind the exact request and runtime
before dispatch. No user-facing native launch command is registered.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile


def _read(path, maximum):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("native bootstrap requires absolute paths")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_mode & 0o022 or not 0 <= before.st_size <= maximum):
            raise ValueError("native bootstrap file is invalid")
        chunks, size = [], 0
        while size <= maximum:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after, current = os.fstat(fd), path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current) or size != before.st_size:
            raise ValueError("native bootstrap file changed")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate native bootstrap field")
            result[key] = value
        return result

    def reject(_):
        raise ValueError("nonfinite native bootstrap number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def _capture_document(raw):
    capture = _json(raw)
    if (type(capture) is not dict or capture.get("schema") not in (
        "synaptic-native-mlx-capture/v1", "synaptic-native-mlx-capture/v2",
    ) or _canonical(capture) != raw):
        raise ValueError("native capture schema differs")
    config = capture.get("config")
    if type(config) is not dict:
        raise ValueError("native capture configuration differs")
    dynamic_fields = {
        "dynamic_system_libraries", "required_builtin_modules", "expected_os_build",
    }
    observation_fields = {"dynamic_system_libraries", "builtin_modules", "dynamic_os_build"}
    if capture["schema"].endswith("/v2"):
        if (not dynamic_fields <= set(config) or not observation_fields <= set(capture)
                or not config["dynamic_system_libraries"]
                or not config["required_builtin_modules"]):
            raise ValueError("native dynamic capture fields differ")
    elif dynamic_fields & set(config) or observation_fields & set(capture):
        raise ValueError("native legacy capture fields differ")
    return capture


class _PinnedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, filename, digest):
        super().__init__(fullname, filename)
        self.digest = digest

    def get_code(self, fullname):
        raw = _read(self.path, 64 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise ImportError("native source bytes differ")
        # Compile pinned source; never execute an ambient timestamp-based cache.
        return self.source_to_code(raw, self.path)


class _PinnedFinder:
    def __init__(self, members):
        self.members = members

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        if spec.origin is None:
            if not spec.submodule_search_locations or any(
                not any(Path(location) in Path(member).parents for member in self.members)
                for location in spec.submodule_search_locations
            ):
                raise ImportError("native namespace origin is outside capture")
            return spec
        expected = self.members.get(str(Path(spec.origin).resolve(strict=True)))
        if expected is None:
            raise ImportError("native module origin is outside capture")
        raw = _read(spec.origin, 512 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ImportError("native imported module bytes differ")
        if isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _PinnedSourceLoader(fullname, spec.origin, expected)
        elif not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            raise ImportError("native loader type is unsupported")
        return spec


def install_capture_imports(capture):
    config = capture["config"]
    members = {}
    for tree in capture["trees"]:
        for item in tree["members"]:
            path = str(Path(tree["root"]) / item["path"])
            if path in members:
                raise ValueError("native import inventory contains duplicates")
            members[path] = item["sha256"]
    for item in capture["files"]:
        if item["path"] in members:
            raise ValueError("native import inventory contains duplicates")
        members[item["path"]] = item["sha256"]
    # No user site, global site-packages, current directory or zip search path.
    source_parents = tuple(dict.fromkeys(str(Path(root).parent) for root in config["sources"]))
    sys.path[:] = [config["stdlib"], str(Path(config["stdlib"]) / "lib-dynload"),
                   *source_parents, config["packages"]]
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        canonical_origin = str(Path(origin).resolve(strict=True))
        if canonical_origin not in members:
            raise ImportError("native bootstrap loaded an uncaptured module")
        if hashlib.sha256(_read(origin, 512 * 1024 * 1024)).hexdigest() != members[canonical_origin]:
            raise ImportError("native bootstrap module bytes differ")
    finder = _PinnedFinder(members)
    sys.meta_path.insert(0, finder)
    return finder


def _snapshot(value):
    from tuner.runtime.native_files import NativeFile
    from tuner.runtime.native_mlx_input import NativeModelSnapshot
    if type(value) is not dict or set(value) != {"repository", "revision", "root", "files"}:
        raise ValueError("native child model snapshot fields differ")
    if type(value["files"]) is not list:
        raise ValueError("native child snapshot inventory is invalid")
    return NativeModelSnapshot(
        value["repository"], value["revision"], value["root"],
        tuple(NativeFile(**item) for item in value["files"]),
    )


def _load_published_tokenizer(archive_bytes, *, request_root, expected_members, use_fast):
    """Prove the exact role payload is loadable, before any optimizer work."""
    from tuner.runtime.native_files import relative_member
    from transformers import AutoTokenizer
    with tempfile.TemporaryDirectory(prefix="tokenizer-publication-", dir=request_root) as name:
        root = Path(name)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if (tuple(sorted(item.name for item in members))
                    != tuple(sorted((*expected_members, "native_tokenizer.json")))
                    or any(not item.isfile() for item in members)):
                raise ValueError("native tokenizer archive inventory differs")
            for member in members:
                parts = relative_member(member.name)
                parent = root.joinpath(*parts[:-1])
                parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                raw = archive.extractfile(member).read()
                if len(raw) != member.size:
                    raise ValueError("native tokenizer archive member differs")
                fd = os.open(root.joinpath(*parts), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
        AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False, use_fast=use_fast,
        )


# Internal worker exits, interpreted only with the exact admitted source and a
# confirmed worker_exited outcome. No request field selects a diagnostic hook.
_GRPO_PHASE_EXITS = {
    "request_compilation": 130, "runtime_material": 131,
    "portable_tokenizer": 132, "input_admission": 133,
    "tokenizer_load": 134, "input_tokenization": 135,
    "ml_imports": 136, "model_load": 137, "adapter_preparation": 138,
    "initial_save": 139, "generation": 140, "reward": 141,
    "reference": 142, "objective": 143, "update": 144,
    "final_save": 145, "reload": 146, "publication": 147,
    "final_custody": 148,
}

_EVALUATION_PHASE_EXITS = {
    "admission": 150, "tokenization": 151, "model": 152, "adapter": 153,
    "generation": 154, "reference": 155, "publication": 156, "custody": 157,
}


def run_request(request, *, request_root, _stage=None, _progress=None):
    """After isolated import admission; all workload controls are recompiled."""
    stage = _stage if _stage is not None else lambda _: None
    stage("request_compilation")
    from tuner.runtime.native_files import NativeTree
    from tuner.runtime.native_mlx_composition import execute_native_sft
    from tuner.runtime.native_mlx_input import prepare_native_input
    from tuner.runtime.native_mlx_publication import NativeArtifactBudget, tokenizer_publication_bytes
    from tuner.training.contracts import RetainedTrainingInputStreamLease
    from tuner.training.native_mlx_compilation import parse_native_workload

    fields = {"schema", "workload", "runtime_capture", "dependency_lock", "dataset",
              "model_snapshot", "tokenizer_snapshot", "tokenizer_members", "artifact_budget", "output"}
    lifecycle = type(request) is dict and request.get("schema") == "synaptic-native-child-request/v2"
    if lifecycle:
        fields.add("lifecycle")
    if (type(request) is not dict or set(request) != fields
            or request["schema"] != ("synaptic-native-child-request/v2" if lifecycle else "synaptic-native-child-request/v1")
            or lifecycle != (_progress is not None)):
        raise ValueError("native child request fields differ")
    projection = parse_native_workload(_canonical(request["workload"]))
    progress = {}
    if lifecycle:
        from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
        from tuner.runtime.native_worker_progress import NativeWorkerProgress
        if (type(_progress) is not NativeWorkerProgress
                or _canonical(_progress._launch.to_dict()) != _canonical(request["lifecycle"])):
            raise ValueError("native child progress selection differs")
        NativeLifecycleLaunchV1.from_dict(request["lifecycle"])
        _progress.bind(projection)
        _progress.emit("preparation", 0, {})
        progress["_progress"] = _progress.emit
    profile = projection.profile
    model, tokenizer = _snapshot(request["model_snapshot"]), _snapshot(request["tokenizer_snapshot"])
    members = tuple(request["tokenizer_members"])
    budget = NativeArtifactBudget(**request["artifact_budget"])
    if (model.repository, model.revision, tokenizer.repository, tokenizer.revision) != (
        profile.model_ref, profile.model_revision, profile.model_ref, profile.tokenizer_revision,
    ):
        raise ValueError("native snapshots differ from compiled model/tokenizer pins")
    for key in ("runtime_capture", "dependency_lock", "dataset", "output"):
        if type(request[key]) is not str or Path(request[key]).parent != request_root:
            raise ValueError("native child transport is outside its private request root")
    with NativeTree(request_root, private=True) as tree:
        runtime = tree.read(Path(request["runtime_capture"]).name, maximum=16 * 1024 * 1024)
        lock = tree.read(Path(request["dependency_lock"]).name, maximum=1024 * 1024)
        dataset = tree.read(Path(request["dataset"]).name, maximum=64 * 1024 * 1024)

        if projection.workload.method == "grpo":
            stage("runtime_material")
            from tuner.runtime.native_mlx_grpo import execute_native_grpo
            from tuner.runtime.native_mlx_material import verify_native_material

            verify_native_material(runtime, expected_digest=profile.runtime_digest, dependency_lock=lock)
            stage("portable_tokenizer")
            with NativeTree(Path(tokenizer.root), private=True) as token_tree:
                published_tokenizer = tokenizer_publication_bytes(
                    tokenizer, members, projection.tokenization, budget, token_tree,
                )
                token_tree.check()
            _load_published_tokenizer(published_tokenizer, request_root=request_root,
                expected_members=members, use_fast=projection.tokenization.use_fast)
            result = execute_native_grpo(projection=projection, dataset=dataset,
                model_snapshot=model, tokenizer_snapshot=tokenizer, runtime_material=runtime,
                dependency_lock=lock, output=Path(request["output"]), _stage=_stage, **progress)
            stage("final_custody")
            tree.check()
            return result

        class Source:
            identity = projection.source

            def open_lease(self):
                return RetainedTrainingInputStreamLease(self.identity, io.BytesIO(dataset))
        bound = prepare_native_input(
            source=Source(), identity=projection.source, tokenizer_snapshot=tokenizer,
            tokenization=projection.tokenization, config=projection.controls,
            workload_digest=projection.workload.fingerprint,
            runtime_material=runtime, runtime_digest=profile.runtime_digest, dependency_lock=lock,
        )
        with NativeTree(Path(tokenizer.root), private=True) as token_tree:
            published_tokenizer = tokenizer_publication_bytes(
                tokenizer, members, projection.tokenization, budget, token_tree,
            )
            token_tree.check()
        _load_published_tokenizer(
            published_tokenizer, request_root=request_root, expected_members=members,
            use_fast=projection.tokenization.use_fast,
        )
        result = execute_native_sft(
            bound=bound, model_snapshot=model, lora=projection.lora, optimizer=projection.optimizer,
            output=Path(request["output"]), runtime_material=runtime,
            runtime_digest=profile.runtime_digest, dependency_lock=lock,
            **progress,
        )
        with NativeTree(Path(request["output"]), private=True) as output:
            output.write_exclusive("workload.json", projection.workload.canonical_bytes)
        tree.check()
        return result


def main():
    # Diagnostics disclose phases only, never exception/source/provider contents.
    phase = 120
    progress = None
    try:
        if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
            raise ValueError("native child must run with -I -S -B")
        if (len(sys.argv) not in (3, 5) or sys.argv[1] != "--request"
                or len(sys.argv) == 5 and sys.argv[3] != "--progress-fd"):
            raise ValueError("native child arguments differ")
        path = Path(sys.argv[2])
        request = _json(_read(path, 2 * 1024 * 1024))
        if (request.get("schema") == "synaptic-native-child-request/v2") != (len(sys.argv) == 5):
            raise ValueError("native child lifecycle argument differs")
        runtime_path = Path(request["runtime_capture"])
        if runtime_path.parent != path.parent:
            raise ValueError("native capture is outside the request root")
        runtime = _read(runtime_path, 16 * 1024 * 1024)
        if hashlib.sha256(runtime).hexdigest() != request["workload"]["profile"]["runtime_digest"]:
            raise ValueError("native capture digest differs from workload")
        capture = _capture_document(runtime)
        phase = 121
        install_capture_imports(capture)
        phase = 122
        if len(sys.argv) == 5:
            from tuner.runtime.native_lifecycle_journal import NativeLifecycleLaunchV1
            from tuner.runtime.native_worker_progress import NativeWorkerProgress
            fd = int(sys.argv[4])
            if str(fd) != sys.argv[4]:
                raise ValueError("native progress descriptor is noncanonical")
            progress = NativeWorkerProgress(fd, NativeLifecycleLaunchV1.from_dict(request["lifecycle"]))
        callbacks = {} if progress is None else {"_progress": progress}
        if request.get("schema") == "synaptic-native-evaluation-child-request/v1":
            from tuner.runtime.native_mlx_evaluation import execute_request

            def mark_evaluation(stage):
                nonlocal phase
                phase = _EVALUATION_PHASE_EXITS[stage]
            phase = _EVALUATION_PHASE_EXITS["admission"]
            execute_request(request, request_root=path.parent, _stage=mark_evaluation)
        elif request["workload"].get("method") == "grpo":
            def mark(stage):
                nonlocal phase
                phase = _GRPO_PHASE_EXITS[stage]
            phase = _GRPO_PHASE_EXITS["request_compilation"]
            run_request(request, request_root=path.parent, _stage=mark, **callbacks)
        else:
            run_request(request, request_root=path.parent, **callbacks)
        return 0
    except BaseException:
        return phase
    finally:
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
