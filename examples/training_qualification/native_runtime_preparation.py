#!/usr/bin/env python3
"""Prepare an observed private native MLX runtime from exact retained archives.

This workflow never grants native execution authority. `plan` is read-only;
`acquire` and `install` consume distinct create-only claims. Network acquisition
has a hard byte limit before every write. Installation reads only exact local
archives under an owned, private root and has a finite owned-process deadline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples.training_qualification.bounded_command import run_bounded_command
from native_model_preparation import _response
from tuner.runtime.native_files import NativeTree, file_identity


BASE = ROOT / "Trainers/mlx_sft_mac/runtime-candidate"
_SCHEMA = "synaptic-native-private-runtime-preparation/v1"
_ORIGINAL_SHA = "f9a6298260cd17f2c3d567ae3674108a71efa68e99369906487a4d8c0feb7117"
_ARCHIVE_SHA = "d5f10f6fdc6f4afb5d251578edb89d8c0ece8f9f56e38b0019e2a94e8f0f4852"
_ARCHIVE_SIZE = 20051753
_UV_SHA = "c87143aea2480b09f107a9f97fa93899d116f0307b4d210649a735e141bdd889"
_UV_BANNER = b"uv 0.12.23 (Homebrew 2026-10-03 aarch64-apple-darwin)"
_EXTRA = frozenset({"jsonschema", "jsonschema-specifications", "referencing", "rpds-py"})
_CHUNK = 64 * 1024
_MAX_CONFIG = 64 * 1024
_MAX_WHEELS = 54


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("native runtime duplicate config field")
        result[key] = value
    return result


def _load(path):
    path = Path(path)
    if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
        raise ValueError("native runtime config path differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_mode & 0o022 or not 0 < before.st_size <= _MAX_CONFIG):
            raise ValueError("native runtime config file differs")
        raw = os.read(fd, _MAX_CONFIG + 1)
        if (len(raw) != before.st_size or file_identity(before) != file_identity(os.fstat(fd))
                or file_identity(before) != file_identity(path.lstat())):
            raise ValueError("native runtime config changed")
    finally:
        os.close(fd)
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def _lock_bytes():
    original = (BASE / "requirements.lock").read_bytes()
    extra = (BASE / "requirements.host-additions.lock").read_bytes()
    combined = (BASE / "requirements.runtime.lock").read_bytes()
    if (hashlib.sha256(original).hexdigest() != _ORIGINAL_SHA
            or combined != original + extra):
        raise ValueError("native runtime original or combined lock differs")
    return original, extra, combined


def _retained_locks(root):
    expected = _lock_bytes()
    with NativeTree(root, private=True) as tree:
        actual = tuple(tree.read(name, maximum=1 << 20) for name in
                       ("original.lock", "host-additions.lock", "runtime.lock"))
    if actual != expected:
        raise ValueError("native runtime retained lock bytes differ")


def _lock_packages(raw):
    text = raw.decode("ascii")
    matches = list(re.finditer(r"(?m)^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)", text))
    result = {}
    for index, match in enumerate(matches):
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        hashes = frozenset(re.findall(r"--hash=sha256:([0-9a-f]{64})", block))
        if name in result or not hashes:
            raise ValueError("native runtime package lock differs")
        result[name] = (match.group(2), hashes)
    return result


def _compatible_wheel(filename, version):
    stem = filename[:-4]
    pieces = stem.rsplit("-", 3)
    if len(pieces) != 4:
        return False
    _, python_tag, abi_tag, platform_tag = pieces
    if platform_tag == "any":
        return abi_tag == "none" and python_tag in {"py3", "py2.py3", "py311"}
    match = re.fullmatch(r"macosx_(\d+)_(\d+)_(arm64|universal2)", platform_tag)
    if not match or (int(match.group(1)), int(match.group(2))) > version[:2]:
        return False
    if python_tag == "cp311":
        return abi_tag in {"cp311", "abi3"}
    if python_tag in {"cp38", "cp39", "cp310"}:
        return abi_tag == "abi3"
    return python_tag == "py3" and abi_tag == "none"


def _host_target():
    if (platform.system() != "Darwin" or platform.machine() != "arm64"
            or platform.mac_ver()[0] != "27.0.1"):
        raise ValueError("native runtime Mac target differs")


def _config(value):
    fields = {"schema", "status", "private_root", "target", "uv_executable", "uv_sha256", "uv_version",
              "python_archive", "python_install_key", "original_lock_sha256",
              "additions_lock_sha256", "runtime_lock_sha256", "wheel_archive_bytes",
              "maximum_worker_seconds", "maximum_command_output_bytes",
              "maximum_python_unpacked_bytes", "maximum_wheel_unpacked_bytes",
              "maximum_private_tree_bytes", "wheels"}
    if (type(value) is not dict or set(value) != fields or value["schema"] != _SCHEMA
            or value["status"] != "REVIEW_REQUIRED_NO_EXECUTION_AUTHORITY"):
        raise ValueError("native runtime configuration differs")
    if value["target"] != {"system": "Darwin", "machine": "arm64", "macos": "27.0.1"}:
        raise ValueError("native runtime target selection differs")
    private = value["private_root"]
    if (type(private) is not str or not Path(private).is_absolute()
            or private != os.path.normpath(private) or Path(private).name.startswith(".")):
        raise ValueError("native runtime private path differs")
    if (value["uv_executable"] != "/opt/homebrew/bin/uv"
            or value["uv_sha256"] != _UV_SHA
            or value["uv_version"] != "0.12.23"
            or value["python_install_key"] != "cpython-3.11.14-macos-aarch64-none"):
        raise ValueError("native runtime tool selection differs")
    archive = value["python_archive"]
    if (type(archive) is not dict or set(archive) != {"url", "filename", "size", "sha256"}
            or archive["url"] != "https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B20260211-aarch64-apple-darwin-install_only_stripped.tar.gz"
            or archive["filename"] != "cpython-3.11.14+20260211-aarch64-apple-darwin-install_only_stripped.tar.gz"
            or archive["size"] != _ARCHIVE_SIZE or archive["sha256"] != _ARCHIVE_SHA):
        raise ValueError("native runtime interpreter archive differs")
    original, extra, combined = _lock_bytes()
    if (value["original_lock_sha256"] != _ORIGINAL_SHA
            or value["additions_lock_sha256"] != hashlib.sha256(extra).hexdigest()
            or value["runtime_lock_sha256"] != hashlib.sha256(combined).hexdigest()):
        raise ValueError("native runtime lock digest differs")
    original_packages, extra_packages = _lock_packages(original), _lock_packages(extra)
    if (len(original_packages) != 50 or set(extra_packages) != _EXTRA
            or set(original_packages) & set(extra_packages)):
        raise ValueError("native runtime locked inventory differs")
    packages = {**original_packages, **extra_packages}
    wheels = value["wheels"]
    if type(wheels) is not list or len(wheels) != _MAX_WHEELS:
        raise ValueError("native runtime wheel count differs")
    seen = set()
    total = 0
    for wheel in wheels:
        if type(wheel) is not dict or set(wheel) != {"package", "version", "filename", "sha256", "size", "url"}:
            raise ValueError("native runtime wheel fields differ")
        name = wheel["package"]
        if (type(name) is not str or name in seen or name not in packages
                or type(wheel["version"]) is not str
                or (wheel["version"], wheel["sha256"]) not in {
                    (packages[name][0], digest) for digest in packages[name][1]
                }):
            raise ValueError("native runtime wheel lock selection differs")
        filename = wheel["filename"]
        if (type(filename) is not str or "/" in filename or not filename.endswith(".whl")
                or not filename.lower().startswith(name.replace("-", "_") + "-" + wheel["version"] + "-")
                or not _compatible_wheel(filename, (27, 0, 1))
                or type(wheel["size"]) is not int or not 1 <= wheel["size"] <= 128 * 1024 * 1024
                or type(wheel["url"]) is not str
                or not wheel["url"].startswith("https://files.pythonhosted.org/packages/")
                or not wheel["url"].endswith("/" + filename)):
            raise ValueError("native runtime wheel archive selection differs")
        seen.add(name)
        total += wheel["size"]
    if (seen != set(packages) or total != value["wheel_archive_bytes"]
            or total != 146218037 or type(value["maximum_worker_seconds"]) is not int
            or not 1 <= value["maximum_worker_seconds"] <= 300
            or value["maximum_command_output_bytes"] != 4096
            or value["maximum_python_unpacked_bytes"] != 536870912
            or value["maximum_wheel_unpacked_bytes"] != 1073741824
            or value["maximum_private_tree_bytes"] != 2147483648):
        raise ValueError("native runtime complete archive budget differs")
    return Path(private), packages


def _root_layout(root):
    with NativeTree(root.parent, private=True) as parent:
        os.mkdir(root.name, mode=0o700, dir_fd=parent.fd)
        parent.check()
    with NativeTree(root, private=True) as tree:
        for name in ("python-archive", "wheelhouse", "uv-cache", "tmp", "cache", "tools"):
            os.mkdir(name, mode=0o700, dir_fd=tree.fd)
        tree.check()
    with NativeTree(root / "python-archive", private=True) as tree:
        os.mkdir("20260211", mode=0o700, dir_fd=tree.fd)
        tree.check()


def _stream_archive(url, root, filename, size, digest, *, hosts, redirects):
    hasher = hashlib.sha256()
    with NativeTree(root, private=True) as tree:
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=tree.fd)
        count = 0
        with os.fdopen(fd, "wb") as output:
            with _response(url, hosts=hosts, redirects=redirects) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdecimal() or int(declared) != size):
                    raise ValueError("native runtime archive declared size differs")
                while True:
                    chunk = response.read(min(_CHUNK, size + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > size:
                        raise ValueError("native runtime archive transfer exceeds bound")
                    output.write(chunk)
                    hasher.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        tree.check()
    if count != size or hasher.hexdigest() != digest:
        raise ValueError("native runtime archive bytes differ")


def _archive_read(root, name, size, digest):
    with NativeTree(root, private=True) as tree:
        raw = tree.read(name, maximum=size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("native runtime retained archive differs")
    return raw


def _private_tree_bytes(root, maximum):
    """Reject retained excess after a phase; this is not a transient quota."""
    total, count = 0, 0
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in (*folders, *files):
            info = os.lstat(Path(directory) / name)
            count += 1
            total += info.st_size
            if count > 100000 or total > maximum:
                raise ValueError("native runtime private tree budget exceeded")
    return total


def _archives(value, root):
    archive = value["python_archive"]
    python = _archive_read(root / "python-archive/20260211", archive["filename"],
                           archive["size"], archive["sha256"])
    with tarfile.open(fileobj=io.BytesIO(python), mode="r:gz") as tar:
        members = tar.getmembers()
        if (len(members) > 16384
                or sum(member.size for member in members) > value["maximum_python_unpacked_bytes"]
                or any(member.name.startswith("/") or ".." in PurePosixPath(member.name).parts
                       for member in members)):
            raise ValueError("native runtime Python archive expansion differs")
    expanded = 0
    for wheel in value["wheels"]:
        raw = _archive_read(root / "wheelhouse", wheel["filename"],
                            wheel["size"], wheel["sha256"])
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            infos = zipped.infolist()
            if len(infos) > 100000 or any(
                item.filename.startswith("/") or ".." in PurePosixPath(item.filename).parts
                for item in infos
            ):
                raise ValueError("native runtime wheel members differ")
            expanded += sum(item.file_size for item in infos)
            if expanded > value["maximum_wheel_unpacked_bytes"]:
                raise ValueError("native runtime wheel expansion exceeds bound")
    with NativeTree(root / "wheelhouse", private=True) as tree:
        observed = tree.inventory(maximum_total=value["wheel_archive_bytes"])
    if {item.path for item in observed} != {item["filename"] for item in value["wheels"]}:
        raise ValueError("native runtime wheelhouse has extra or missing archives")
    return expanded


def _copy_uv(value, root):
    source = Path(value["uv_executable"])
    with source.open("rb") as input_file:
        raw = input_file.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != value["uv_sha256"]:
        raise ValueError("native runtime uv executable bytes differ")
    with NativeTree(root / "tools", private=True) as tree:
        fd = os.open("uv", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o500, dir_fd=tree.fd)
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        tree.check()


def _environment(root):
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": str(root / "tmp"),
            "UV_CACHE_DIR": str(root / "uv-cache"), "XDG_CACHE_HOME": str(root / "cache"),
            "UV_PYTHON_INSTALL_REGISTRY": "0", "UV_NO_PROGRESS": "1", "UV_NO_CACHE": "1"}


@contextmanager
def _private_environment(root):
    selected = _environment(root)
    prior = {key: os.environ.get(key) for key in selected}
    os.environ.update(selected)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _command(argv, root, value):
    status, output = run_bounded_command(
        argv, cwd=str(root), environment=_environment(root),
        maximum=value["maximum_command_output_bytes"], timeout=value["maximum_worker_seconds"],
    )
    if status:
        raise ValueError("native runtime owned command failed")
    _private_tree_bytes(root, value["maximum_private_tree_bytes"])
    return output


def _acquire(value, root):
    _root_layout(root)
    with NativeTree(root, private=True) as tree:
        for name, raw in zip(("original.lock", "host-additions.lock", "runtime.lock"),
                             _lock_bytes(), strict=True):
            tree.write_exclusive(name, raw)
    with _private_environment(root):
        archive = value["python_archive"]
        _stream_archive(archive["url"], root / "python-archive/20260211",
                        archive["filename"], archive["size"], archive["sha256"],
                        hosts=frozenset({"github.com", "release-assets.githubusercontent.com",
                                         "objects.githubusercontent.com"}), redirects=2)
        transferred = archive["size"]
        for wheel in value["wheels"]:
            if transferred + wheel["size"] > archive["size"] + value["wheel_archive_bytes"]:
                raise ValueError("native runtime aggregate transfer exceeds bound")
            _stream_archive(wheel["url"], root / "wheelhouse", wheel["filename"],
                            wheel["size"], wheel["sha256"],
                            hosts=frozenset({"files.pythonhosted.org"}), redirects=0)
            transferred += wheel["size"]
        expanded = _archives(value, root)
        _copy_uv(value, root)
        _private_tree_bytes(root, value["maximum_private_tree_bytes"])
    receipt = {"schema": "synaptic-native-private-runtime-archives/v1",
               "status": "OBSERVED_NOT_ADMITTED", "selection_sha256": hashlib.sha256(_canonical(value)).hexdigest(),
               "archive_bytes": transferred, "wheel_count": len(value["wheels"]),
               "wheel_unpacked_bytes": expanded,
               "runtime_lock_sha256": value["runtime_lock_sha256"]}
    with NativeTree(root, private=True) as tree:
        tree.write_exclusive("archive-result.json", _canonical(receipt))
    return receipt


def _installed(value, root, packages):
    python = root / "venv/bin/python"
    code = ("import importlib.metadata as m,json,platform,sys,sysconfig;from pathlib import Path;"
            "items=[(x.metadata['Name'].lower().replace('_','-'),x.version) for x in m.distributions()];"
            "print(json.dumps({'version':list(sys.version_info[:3]),'machine':platform.machine(),"
            "'executable':str(Path(sys.executable).resolve()),'stdlib':str(Path(sysconfig.get_path('stdlib')).resolve()),"
            "'distributions':items},sort_keys=True,separators=(',',':')))" )
    raw = _command([str(python), "-I", "-B", "-c", code], root, value)
    observed = json.loads(raw, object_pairs_hook=_pairs)
    items = observed["distributions"]
    if (observed["version"] != [3, 11, 14] or observed["machine"] != "arm64"
            or len(items) != 54 or len({name for name, _ in items}) != 54
            or dict(items) != {name: version for name, (version, _) in packages.items()}):
        raise ValueError("native runtime installed distribution inventory differs")
    executable, stdlib = Path(observed["executable"]), Path(observed["stdlib"])
    managed = root / "uv-python" / value["python_install_key"]
    if (not executable.is_relative_to(managed) or not stdlib.is_relative_to(managed)
            or not executable.is_file() or not stdlib.is_dir()):
        raise ValueError("native runtime private interpreter path differs")
    with NativeTree(executable.parent) as tree:
        executable_sha256 = hashlib.sha256(tree.read(executable.name)).hexdigest()
    return {"python_version": "3.11.14", "distribution_count": len(items),
            "inventory_sha256": hashlib.sha256(_canonical(sorted(items))).hexdigest(),
            "executable": str(executable), "executable_sha256": executable_sha256,
            "stdlib": str(stdlib)}


def _install(value, root, packages):
    _retained_locks(root)
    with NativeTree(root, private=True) as tree:
        archive_receipt = json.loads(tree.read("archive-result.json", maximum=4096))
        if (archive_receipt.get("status") != "OBSERVED_NOT_ADMITTED"
                or archive_receipt.get("selection_sha256") != hashlib.sha256(_canonical(value)).hexdigest()
                or archive_receipt.get("runtime_lock_sha256") != value["runtime_lock_sha256"]):
            raise ValueError("native runtime archive receipt differs")
        tree.write_exclusive("install-claim.json", _canonical({
            "schema": "synaptic-native-runtime-install-claim/v1", "status": "CONSUMED_UNCERTAIN",
            "selection_sha256": archive_receipt["selection_sha256"]}))
    expanded = _archives(value, root)
    uv = root / "tools/uv"
    if hashlib.sha256(uv.read_bytes()).hexdigest() != value["uv_sha256"]:
        raise ValueError("native runtime private uv differs")
    if _command([str(uv), "--version"], root, value).strip() != _UV_BANNER:
        raise ValueError("native runtime uv version differs")
    _command([str(uv), "--no-config", "--offline", "python", "install", "--install-dir",
              str(root / "uv-python"), "--no-bin", "--mirror",
              (root / "python-archive").as_uri(), "cpython@3.11.14"], root, value)
    managed = root / "uv-python" / value["python_install_key"] / "bin/python3.11"
    if not managed.is_file():
        raise ValueError("native runtime private interpreter absent")
    _command([str(uv), "--no-config", "venv", "--python", str(managed),
              "--no-python-downloads", str(root / "venv")], root, value)
    _command([str(uv), "--no-config", "--offline", "pip", "sync", "--python",
              str(root / "venv/bin/python"), "--no-index", "--find-links", str(root / "wheelhouse"),
              "--require-hashes", "--only-binary", ":all:", "--link-mode", "copy",
              str(root / "runtime.lock")], root, value)
    _command([str(uv), "--no-config", "--offline", "pip", "check", "--python",
              str(root / "venv/bin/python")], root, value)
    result = _installed(value, root, packages)
    result.update({"schema": "synaptic-native-private-runtime-install/v1",
                   "status": "OBSERVED_NOT_ADMITTED", "selection_sha256": hashlib.sha256(_canonical(value)).hexdigest(),
                   "wheel_unpacked_bytes": expanded})
    with NativeTree(root, private=True) as tree:
        tree.write_exclusive("install-result.json", _canonical(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "acquire", "install", "verify", "_acquire_worker"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        value = _load(args.config)
        root, packages = _config(value)
        if args.action in {"acquire", "install", "_acquire_worker"}:
            _host_target()
            with NativeTree(args.config.parent, private=True):
                pass
        if args.action == "plan":
            result = {"status": "REVIEW_REQUIRED_NO_EXECUTION_AUTHORITY",
                      "selection_sha256": hashlib.sha256(_canonical(value)).hexdigest(),
                      "wheel_count": 54, "wheel_archive_bytes": value["wheel_archive_bytes"],
                      "total_archive_bytes": value["wheel_archive_bytes"] + _ARCHIVE_SIZE,
                      "runtime_lock_sha256": value["runtime_lock_sha256"]}
        elif args.action == "_acquire_worker":
            prior_umask = os.umask(0o077)
            try:
                result = _acquire(value, root)
            finally:
                os.umask(prior_umask)
        elif args.action == "install":
            # uv commands own separate process groups. Run this phase in the
            # caller so a timed-out outer worker cannot orphan an inner uv.
            prior_umask = os.umask(0o077)
            try:
                result = _install(value, root, packages)
            finally:
                os.umask(prior_umask)
            with NativeTree(root, private=True) as tree:
                retained = json.loads(tree.read("install-result.json", maximum=4096),
                                      object_pairs_hook=_pairs)
            if retained != result:
                raise ValueError("native runtime retained install result differs")
        elif args.action == "verify":
            _retained_locks(root)
            _archives(value, root)
            result = _installed(value, root, packages)
            result["status"] = "OBSERVED_NOT_ADMITTED"
        else:
            status, raw = run_bounded_command(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_acquire_worker", str(args.config)],
                environment={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                maximum=value["maximum_command_output_bytes"], timeout=value["maximum_worker_seconds"],
            )
            if status or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
                raise ValueError("native runtime worker uncertain")
            result = json.loads(raw[:-1], object_pairs_hook=_pairs)
            if _canonical(result) + b"\n" != raw:
                raise ValueError("native runtime worker framing differs")
            with NativeTree(root, private=True) as tree:
                retained = json.loads(tree.read("archive-result.json", maximum=4096),
                                      object_pairs_hook=_pairs)
            if retained != result:
                raise ValueError("native runtime retained result differs")
        print(_canonical(result).decode())
        return 0
    except Exception:
        print('{"diagnostic_code":"native_runtime_preparation_refused"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
