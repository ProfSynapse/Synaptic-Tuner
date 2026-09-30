"""Small, hash-pinned offline wheel builder for the accepted engine commit.

Only the four public build-tool wheels in the packaged lock may be fetched.
They are copied from a private, reverified cache into each isolated builder;
the engine wheel build itself has no index or dependency resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_LOCK_PATH = Path(__file__).with_name("modal-wheel-builder.lock.json")
_NAMES = ("pip", "setuptools", "wheel", "packaging")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_FILE = re.compile(r"^[a-z0-9][a-z0-9_.-]*-py3-none-any\.whl$")
_MAX_WHEEL = 4 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def builder_lock_bytes() -> bytes:
    raw = _LOCK_PATH.read_bytes()
    if not raw or len(raw) > 16 * 1024:
        raise ValueError("Modal wheel builder lock is invalid")
    return raw


def _locked_wheels() -> tuple[dict[str, object], ...]:
    try:
        lock = json.loads(builder_lock_bytes())
        if (type(lock) is not dict or set(lock) != {"schema_version", "python", "wheels"}
                or lock["schema_version"] != "synaptic-modal-wheel-builder-lock/v1"
                or lock["python"] != "3.11" or type(lock["wheels"]) is not list
                or len(lock["wheels"]) != len(_NAMES)):
            raise ValueError
        wheels = []
        for expected, item in zip(_NAMES, lock["wheels"], strict=True):
            if type(item) is not dict or set(item) != {"name", "version", "filename", "url", "sha256", "size"}:
                raise ValueError
            name, version, filename = item["name"], item["version"], item["filename"]
            url, digest, size = item["url"], item["sha256"], item["size"]
            if (name != expected or type(version) is not str
                    or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version) is None
                    or type(filename) is not str or _FILE.fullmatch(filename) is None
                    or filename != f"{name}-{version}-py3-none-any.whl"
                    or type(url) is not str or type(digest) is not str
                    or _HASH.fullmatch(digest) is None or type(size) is not int
                    or not 0 < size <= _MAX_WHEEL):
                raise ValueError
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org"
                    or parsed.query or parsed.fragment or Path(parsed.path).name != filename):
                raise ValueError
            wheels.append(item)
        return tuple(wheels)
    except Exception:
        raise ValueError("Modal wheel builder lock is invalid") from None


def _private_cache_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute() or os.name != "posix":
        raise ValueError("private Modal wheel cache root required")
    parent = root.parent
    info = parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private Modal wheel cache parent is invalid")
    try:
        root.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        pass
    info = root.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private Modal wheel cache root is invalid")
    return root


def _read_verified(path: Path, *, size: int, digest: str) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077 or before.st_size != size):
            raise ValueError("cached Modal wheel identity is invalid")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(size + 1)
        after = os.fstat(fd)
        if (len(raw) != size or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
                != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
                or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError("cached Modal wheel digest is invalid")
        return raw
    finally:
        os.close(fd)


def _download_exact(url: str, size: int) -> bytes:
    opener = build_opener(_NoRedirect)
    request = Request(url, headers={"User-Agent": "synaptic-modal-wheel-builder/1"})
    with opener.open(request, timeout=20) as response:
        if response.status != 200 or response.url != url:
            raise ValueError("pinned Modal wheel download failed")
        raw = response.read(size + 1)
    if len(raw) != size:
        raise ValueError("pinned Modal wheel size differs")
    return raw


def _wheel_bytes(item: dict[str, object], cache_root: Path) -> bytes:
    path = cache_root / str(item["filename"])
    size, digest = int(item["size"]), str(item["sha256"])
    if path.exists() or path.is_symlink():
        return _read_verified(path, size=size, digest=digest)
    raw = _download_exact(str(item["url"]), size)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("pinned Modal wheel download digest differs")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    return _read_verified(path, size=size, digest=digest)


def create_offline_wheel_builder(scratch: Path, cache_root: Path) -> Path:
    """Return a fresh private venv Python with the exact locked build closure."""
    if sys.version_info[:2] != (3, 11) or os.name != "posix":
        raise ValueError("Modal wheel builder requires the locked Linux Python minor")
    wheels = _locked_wheels()
    cache = _private_cache_root(cache_root)
    wheelhouse = scratch / "wheelhouse"
    wheelhouse.mkdir(mode=0o700)
    for item in wheels:
        (wheelhouse / str(item["filename"])).write_bytes(_wheel_bytes(item, cache))
    builder = scratch / "builder"
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
    }
    created = subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(builder)],
        capture_output=True, timeout=60, check=False, env=safe_env,
    )
    if created.returncode != 0:
        raise ValueError("isolated Modal wheel builder creation failed")
    python = builder / "bin" / "python"
    requirements = scratch / "builder-requirements.txt"
    requirements.write_text("".join(
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
        for item in wheels
    ), encoding="ascii")
    bootstrap_env = {**safe_env, "PYTHONPATH": str(wheelhouse / str(wheels[0]["filename"]))}
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps",
         "--require-hashes", "--no-cache-dir", "--no-compile",
         "--disable-pip-version-check", "--find-links", str(wheelhouse),
         "-r", str(requirements)],
        capture_output=True, timeout=120, check=False, env=bootstrap_env,
    )
    if installed.returncode != 0:
        raise ValueError("isolated Modal wheel builder bootstrap failed")
    check = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in m.distributions()}))"],
        capture_output=True, timeout=30, check=False, env=safe_env,
    )
    try:
        actual = json.loads(check.stdout.decode("ascii", "strict"))
    except Exception:
        actual = None
    expected = {str(item["name"]): str(item["version"]) for item in wheels}
    if check.returncode != 0 or actual != expected:
        raise ValueError("isolated Modal wheel builder inventory differs")
    return python


__all__ = ["builder_lock_bytes", "create_offline_wheel_builder"]
