"""Automatic pinned-model preparation; the Hub SDK writes only private scratch.

The persistent store contains ordinary verified repository files, not a replica
of the SDK's cache internals. It is never passed to the SDK. Only authenticated
upstream members cross into the offline worker's link-free snapshot.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import unicodedata

from .mounted_io import copy_regular

_ENDPOINT = "https://huggingface.co"
_MAX_FILES = 20_000
_MAX_BYTES = 1 << 40


def _bind_hub_api(api_type, download) -> None:
    """Check the image-provided SDK surface without constructing a client.

    The deployment image digest locks the dependency bytes. This check catches
    incompatible APIs before any credential-bearing call, without upgrading
    the independently pinned training stack.
    """
    inspect.signature(api_type).bind(endpoint=_ENDPOINT, token=False)
    inspect.signature(api_type.model_info).bind(
        object(), "fixture/model", revision="a" * 40,
        files_metadata=True, token=False,
    )
    inspect.signature(download).bind(
        repo_id="fixture/model", revision="a" * 40, token=False,
        endpoint=_ENDPOINT, local_dir=Path("repository"),
        cache_dir=Path("sdk-cache"), allow_patterns=["config.json"],
        max_workers=1,
    )


def _directory(path: Path) -> None:
    for member in (*reversed(path.parents), path):
        info = member.lstat()
        if not stat.S_ISDIR(info.st_mode) or member.is_symlink() or (
            getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("model directory is redirected")


def _present(root: Path, relative: Path) -> bool:
    """Missing is a cache miss; redirects and special files are failures."""
    _directory(root)
    current = root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return False
        expected = stat.S_ISREG if index == len(relative.parts) - 1 else stat.S_ISDIR
        if not expected(info.st_mode) or current.is_symlink() or (
            getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("model cache member is redirected")
    return True


def _verify(path: Path, size: int, kind: str, expected: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size != size:
        raise ValueError("model member size mismatch")
    digest = hashlib.sha256() if kind == "lfs" else hashlib.sha1()
    if kind == "git":
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("model member digest mismatch")


def prepare_model_snapshot(
    *, model_ref: str, revision: str, token: str | None,
    persistent_root: Path, destination_root: Path, scratch_root: Path,
) -> Path:
    """Reuse verified files, download misses remotely, then hand off offline.

    Roots are physical canonical directories supplied by the provider. The
    destination is a fresh run cache. Credentials are used only by SDK calls
    and no SDK output or exception text is allowed out of this boundary.
    """
    try:
        # Suppress output without accumulating SDK diagnostics in memory.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return _prepare(
                model_ref=model_ref, revision=revision, token=token,
                persistent_root=persistent_root, destination_root=destination_root,
                scratch_root=scratch_root,
            )
    except Exception:
        raise ValueError("model preparation failed") from None


def _prepare(*, model_ref, revision, token, persistent_root, destination_root, scratch_root):
    from huggingface_hub import HfApi, snapshot_download
    _bind_hub_api(HfApi, snapshot_download)

    parts = model_ref.split("/")
    if len(parts) not in (1, 2) or any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", part) is None
        or ".." in part or "--" in part or part.endswith((".", "-"))
        for part in parts
    ) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model identity is not pinned")
    for root in (persistent_root, destination_root, scratch_root):
        _directory(root)
    credential = token if isinstance(token, str) and token.strip() else False
    info = HfApi(endpoint=_ENDPOINT, token=credential).model_info(
        model_ref, revision=revision, files_metadata=True, token=credential,
    )
    if info.sha != revision or not info.siblings or len(info.siblings) > _MAX_FILES:
        raise ValueError("model metadata is not exact")
    members = {}
    folded = set()
    total = 0
    for sibling in info.siblings:
        name = sibling.rfilename
        path = PurePosixPath(name)
        if (
            path.is_absolute() or path.as_posix() != name
            or any(part in {".", "..", ".cache", ".git"} for part in path.parts)
            or not path.parts or "\\" in name or "\0" in name
            or any(character in name for character in "*?[]")
            or unicodedata.normalize("NFC", name) != name
            or name.casefold() in folded
        ):
            raise ValueError("model metadata path is invalid")
        folded.add(name.casefold())
        size = sibling.size
        if type(size) is not int or not 0 <= size <= _MAX_BYTES:
            raise ValueError("model metadata size is invalid")
        total += size
        if total > _MAX_BYTES:
            raise ValueError("model snapshot exceeds its bound")
        kind = "lfs" if sibling.lfs is not None else "git"
        expected = sibling.lfs.sha256 if kind == "lfs" else sibling.blob_id
        if not isinstance(expected, str) or re.fullmatch(
            r"[0-9a-f]{64}" if kind == "lfs" else r"[0-9a-f]{40}", expected
        ) is None:
            raise ValueError("model metadata digest is missing")
        members[name] = (size, kind, expected)

    relative = Path("models--" + "--".join(parts)) / "snapshots" / revision
    destination = destination_root / "model" / relative
    if destination.exists() or destination.is_symlink():
        raise ValueError("model destination already exists")
    with tempfile.TemporaryDirectory(prefix="model-prepare-", dir=scratch_root) as temporary:
        private = Path(temporary)
        repository = private / "repository"
        repository.mkdir()
        missing = []
        for name, (size, kind, expected) in members.items():
            cached = relative / name
            if _present(persistent_root, cached):
                copy_regular(persistent_root, persistent_root / cached, repository, repository / name, maximum=size)
                _verify(repository / name, size, kind, expected)
            else:
                missing.append(name)
        if missing:
            result = snapshot_download(
                repo_id=model_ref, revision=revision, token=credential,
                endpoint=_ENDPOINT, local_dir=repository,
                cache_dir=private / "sdk-cache", allow_patterns=missing,
                max_workers=1,
            )
            if Path(result) != repository:
                raise ValueError("SDK snapshot location differs")
        for name, (size, kind, expected) in members.items():
            _directory((repository / name).parent)
            _verify(repository / name, size, kind, expected)
        for name, (size, kind, expected) in members.items():
            # Exclusive descriptor-relative writes cannot follow cache links.
            if name in missing:
                try:
                    copy_regular(repository, repository / name, persistent_root, persistent_root / relative / name, maximum=size)
                except FileExistsError:
                    # A concurrent writer may have published the same pinned
                    # member. Verify its bytes privately before accepting it.
                    check = private / "concurrent"
                    check.mkdir(exist_ok=True)
                    copy_regular(persistent_root, persistent_root / relative / name, check, check / name, maximum=size)
                    _verify(check / name, size, kind, expected)
            copy_regular(repository, repository / name, destination_root, destination / name, maximum=size)
        # Re-read the shared destination through retained descriptors. Never
        # hash a Volume file by opening a path that can redirect mid-read.
        verification = private / "verification"
        verification.mkdir()
        for name, (size, kind, expected) in members.items():
            copy_regular(destination_root, destination / name, verification, verification / name, maximum=size)
            _verify(verification / name, size, kind, expected)
    return destination
