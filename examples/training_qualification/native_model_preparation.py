#!/usr/bin/env python3
"""Private, bounded model snapshot preparation candidate; never grants training authority.

``inspect`` validates an independently reviewed inventory without network access.
``acquire`` is a later, separately authorized effect. It retains a consumed
workspace on any uncertain failure and never retries automatically.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_command import run_bounded_command
from tuner.runtime.native_files import NativeTree, relative_member


_SCHEMA = "synaptic-native-model-acquisition/v1"
_ENDPOINT = "https://huggingface.co"
_MAX_BYTES = 512 * 1024 * 1024
_MAX_FILES = 128
_MEMBER_FIELDS = {"path", "size", "kind", "upstream_digest"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("duplicate model preparation field")
        result[name] = value
    return result


def _read_config(path):
    path = Path(path)
    if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
        raise ValueError("model preparation config path differs")
    with NativeTree(path.parent, private=True) as parent:
        raw = parent.read(path.name, maximum=1 << 20)
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    return value


def _member(value):
    if type(value) is not dict or set(value) != _MEMBER_FIELDS:
        raise ValueError("model member fields differ")
    name = value["path"]
    relative_member(name)
    if (any(part in {".git", ".cache"} for part in name.split("/"))
            or unicodedata.normalize("NFC", name) != name
            or any(character in name for character in "*?[]")):
        raise ValueError("model member path is unsupported")
    size, kind, digest = value["size"], value["kind"], value["upstream_digest"]
    if (type(size) is not int or not 0 <= size <= _MAX_BYTES
            or kind not in ("git", "lfs") or type(kind) is not str
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{40}" if kind == "git" else r"[0-9a-f]{64}", digest) is None):
        raise ValueError("model member metadata is invalid")
    return value


def _validate_common(value):
    fields = {"schema", "repository", "revision", "workspace_root",
              "maximum_bytes", "maximum_files", "timeout_seconds", "members",
              "tokenizer_members"}
    if type(value) is not dict or set(value) != fields or value["schema"] != _SCHEMA:
        raise ValueError("model preparation configuration differs")
    repository, revision = value["repository"], value["revision"]
    if (type(repository) is not str
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
            or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("model repository pin is invalid")
    workspace = Path(value["workspace_root"])
    if (type(value["workspace_root"]) is not str or not workspace.is_absolute()
            or str(workspace) != value["workspace_root"]
            or str(workspace) != os.path.normpath(str(workspace))):
        raise ValueError("model private workspace is invalid")
    if (type(value["maximum_bytes"]) is not int or not 1 <= value["maximum_bytes"] <= _MAX_BYTES
            or type(value["maximum_files"]) is not int or not 1 <= value["maximum_files"] <= _MAX_FILES
            or type(value["timeout_seconds"]) is not int or not 1 <= value["timeout_seconds"] <= 300):
        raise ValueError("model acquisition bounds differ")
    with NativeTree(workspace, private=True):
        pass
    return workspace


def _validate(value):
    workspace = _validate_common(value)
    raw_members = value["members"]
    if (type(raw_members) is not list or not 1 <= len(raw_members) <= value["maximum_files"]):
        raise ValueError("model member count differs")
    members = tuple(_member(item) for item in raw_members)
    names = tuple(item["path"] for item in members)
    if (names != tuple(sorted(set(names)))
            or len({name.casefold() for name in names}) != len(names)
            or sum(item["size"] for item in members) > value["maximum_bytes"]
            or "config.json" not in names or not any(name.endswith(".safetensors") for name in names)):
        raise ValueError("model complete selection or bound differs")
    selected = value["tokenizer_members"]
    if (type(selected) is not list or not selected
            or selected != sorted(set(selected)) or not set(selected) <= set(names)):
        raise ValueError("model tokenizer selection differs")
    with NativeTree(workspace, private=True) as root:
        for name in ("scratch", "model-snapshot", "tokenizer-snapshot",
                     "model-snapshot.json", "tokenizer-snapshot.json", "acquisition-result.json"):
            try:
                os.stat(name, dir_fd=root.fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError("model workspace has a prior acquisition claim")
    return members


def _upstream(member, raw):
    if len(raw) != member["size"]:
        raise ValueError("model member byte count differs")
    if member["kind"] == "lfs":
        digest = hashlib.sha256(raw).hexdigest()
    else:
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if digest != member["upstream_digest"]:
        raise ValueError("model member upstream digest differs")
    return hashlib.sha256(raw).hexdigest()


def _observe_metadata(config, *, api_type):
    info = api_type(endpoint=_ENDPOINT, token=False).model_info(
        config["repository"], revision=config["revision"], files_metadata=True,
        token=False, timeout=30,
    )
    if info.sha != config["revision"] or type(info.siblings) not in (list, tuple):
        raise ValueError("model metadata revision differs")
    observed = []
    for sibling in info.siblings:
        kind = "lfs" if sibling.lfs is not None else "git"
        digest = sibling.lfs.sha256 if kind == "lfs" else sibling.blob_id
        observed.append(_member({"path": sibling.rfilename, "size": sibling.size,
                                 "kind": kind, "upstream_digest": digest}))
        if len(observed) > config["maximum_files"]:
            raise ValueError("model metadata count exceeds bound")
    names = tuple(item["path"] for item in observed)
    if (len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names)
            or sum(item["size"] for item in observed) > config["maximum_bytes"]):
        raise ValueError("model metadata exceeds exact selection bounds")
    return tuple(sorted(observed, key=lambda item: item["path"]))


def _metadata(members, config, *, api_type):
    observed = _observe_metadata(config, api_type=api_type)
    if observed != members:
        raise ValueError("model metadata differs from reviewed complete selection")


def _metadata_candidate(config, *, api_type=None):
    workspace = _validate_common(config)
    if config["members"] != [] or config["tokenizer_members"] != []:
        raise ValueError("metadata candidate must not contain reviewed members")
    with NativeTree(workspace, private=True) as root:
        try:
            os.stat("metadata-candidate.json", dir_fd=root.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("metadata candidate already exists")
    if api_type is None:
        from huggingface_hub import HfApi
        api_type = HfApi
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        observed = _observe_metadata(config, api_type=api_type)
    names = tuple(item["path"] for item in observed)
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("model metadata is incomplete")
    candidate = {"schema": "synaptic-native-model-metadata-candidate/v1",
                 "status": "REVIEW_REQUIRED_NO_AUTHORITY", "repository": config["repository"],
                 "revision": config["revision"], "members": list(observed),
                 "declared_bytes": sum(item["size"] for item in observed),
                 "member_count": len(observed)}
    with NativeTree(workspace, private=True) as root:
        root.write_exclusive("metadata-candidate.json", _canonical(candidate))
    return {"status": candidate["status"], "repository": config["repository"],
            "revision": config["revision"], "member_count": candidate["member_count"],
            "declared_bytes": candidate["declared_bytes"],
            "metadata_candidate_sha256": hashlib.sha256(_canonical(candidate)).hexdigest()}


def _publish_member(root, name, raw):
    parts = relative_member(name)
    with ExitStack() as stack:
        parent = root.fd
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
                raise ValueError("model destination parent is not private")
            parent = fd
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    root.check()


def _acquire(config, *, api_type=None, download=None):
    previous = os.umask(0o077)
    try:
        return _acquire_private(config, api_type=api_type, download=download)
    finally:
        os.umask(previous)


def _acquire_private(config, *, api_type=None, download=None):
    members = _validate(config)
    workspace = Path(config["workspace_root"])
    with NativeTree(workspace, private=True) as root:
        os.mkdir("scratch", mode=0o700, dir_fd=root.fd)
        root.check()
    if api_type is None or download is None:
        from huggingface_hub import HfApi, snapshot_download
        api_type = api_type or HfApi
        download = download or snapshot_download
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        _metadata(members, config, api_type=api_type)
    scratch = workspace / "scratch"
    with NativeTree(scratch, private=True) as held:
        os.mkdir("repository", mode=0o700, dir_fd=held.fd)
        held.check()
    repository = scratch / "repository"
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        result = download(
            repo_id=config["repository"], revision=config["revision"], token=False,
            endpoint=_ENDPOINT, local_dir=repository, cache_dir=scratch / "sdk-cache",
            allow_patterns=[item["path"] for item in members], max_workers=1,
            etag_timeout=10,
        )
    if Path(result) != repository:
        raise ValueError("model SDK destination differs")
    bytes_by_name = {}
    content_hashes = {}
    with NativeTree(repository, private=True) as source:
        for member in members:
            raw = source.read(member["path"], maximum=member["size"])
            content_hashes[member["path"]] = _upstream(member, raw)
            bytes_by_name[member["path"]] = raw
        source.check()
    with NativeTree(workspace, private=True) as root:
        os.mkdir("model-snapshot", mode=0o700, dir_fd=root.fd)
        os.mkdir("tokenizer-snapshot", mode=0o700, dir_fd=root.fd)
        root.check()
    manifests = []
    for role, names in (("model", tuple(item["path"] for item in members)),
                        ("tokenizer", tuple(config["tokenizer_members"]))):
        target = workspace / f"{role}-snapshot"
        with NativeTree(target, private=True) as published:
            for name in names:
                _publish_member(published, name, bytes_by_name[name])
            inventory = published.inventory(maximum_total=config["maximum_bytes"])
            if (tuple(item.path for item in inventory) != names
                    or any(item.sha256 != content_hashes[item.path] for item in inventory)):
                raise ValueError("model publication readback differs")
        manifest = {"repository": config["repository"], "revision": config["revision"],
                    "root": str(target),
                    "files": [{"path": item.path, "size": item.size, "sha256": item.sha256}
                              for item in inventory]}
        manifests.append((role, manifest))
    receipt = {"schema": "synaptic-native-model-acquisition-observation/v1",
               "status": "OBSERVED_NOT_ADMITTED", "repository": config["repository"],
               "revision": config["revision"], "selection_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
               "model_manifest_sha256": hashlib.sha256(_canonical(manifests[0][1])).hexdigest(),
               "tokenizer_manifest_sha256": hashlib.sha256(_canonical(manifests[1][1])).hexdigest()}
    with NativeTree(workspace, private=True) as root:
        for role, manifest in manifests:
            root.write_exclusive(f"{role}-snapshot.json", _canonical(manifest))
        root.write_exclusive("acquisition-result.json", _canonical(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "metadata-candidate", "acquire",
                                          "_metadata_worker", "_worker"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = _read_config(args.config)
        if args.action in {"metadata-candidate", "_metadata_worker"}:
            _validate_common(config)
        else:
            _validate(config)
        if args.action == "inspect":
            result = {"status": "REVIEW_REQUIRED_NO_NETWORK",
                      "selection_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
                      "member_count": len(config["members"]),
                      "declared_bytes": sum(item["size"] for item in config["members"])}
        elif args.action == "_metadata_worker":
            result = _metadata_candidate(config)
        elif args.action == "_worker":
            result = _acquire(config)
        else:
            environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C",
                           "HF_HOME": config["workspace_root"],
                           "HF_HUB_DISABLE_TELEMETRY": "1", "TMPDIR": config["workspace_root"]}
            status, output = run_bounded_command(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                 "_metadata_worker" if args.action == "metadata-candidate" else "_worker",
                 str(args.config)], environment=environment,
                maximum=4096, timeout=config["timeout_seconds"],
            )
            if status != 0:
                raise ValueError("model acquisition worker uncertain")
            worker_result = json.loads(output, object_pairs_hook=_pairs)
            if _canonical(worker_result) != output:
                raise ValueError("model acquisition worker result differs")
            if args.action == "metadata-candidate":
                with NativeTree(Path(config["workspace_root"]), private=True) as root:
                    candidate = json.loads(root.read("metadata-candidate.json", maximum=1 << 20))
                result = {"status": candidate["status"], "repository": config["repository"],
                          "revision": config["revision"], "member_count": candidate["member_count"],
                          "declared_bytes": candidate["declared_bytes"],
                          "metadata_candidate_sha256": hashlib.sha256(_canonical(candidate)).hexdigest()}
                if result != worker_result or result["status"] != "REVIEW_REQUIRED_NO_AUTHORITY":
                    raise ValueError("model metadata candidate differs")
            else:
                with NativeTree(Path(config["workspace_root"]), private=True) as root:
                    result = json.loads(root.read("acquisition-result.json", maximum=1 << 20))
                if (result != worker_result or result.get("status") != "OBSERVED_NOT_ADMITTED"
                        or result.get("selection_sha256") != hashlib.sha256(_canonical(config)).hexdigest()):
                    raise ValueError("model acquisition result differs")
        print(_canonical(result).decode())
        return 0
    except Exception:
        print('{"diagnostic_code":"native_model_preparation_refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
