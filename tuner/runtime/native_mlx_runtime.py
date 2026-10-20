"""Read-only native candidate comparison, not a runtime attestation or installer.

The host supplies reviewed exact pins derived from the retained wheel install
and interpreter capture. This compares metadata and interpreter bytes before
worker ML imports. Native libraries, loaded-module origins and Metal behavior
still require independent qualification; no RuntimeSpec is admitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import sys


_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]{0,127}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MLXRuntimeExpectation:
    """Host-owned candidate pins. Equality is necessary, not execution authority."""

    python_version: str
    executable_sha256: str
    dependency_lock_sha256: str
    distributions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.python_version) is not str or not re.fullmatch(r"3\.\d+\.\d+", self.python_version):
            raise ValueError("native candidate Python pin is invalid")
        for value in (self.executable_sha256, self.dependency_lock_sha256):
            if type(value) is not str or not _DIGEST.fullmatch(value):
                raise ValueError("native candidate digest pin is invalid")
        if type(self.distributions) is not tuple or not 2 <= len(self.distributions) <= 256:
            raise ValueError("native candidate inventory is invalid")
        seen = set()
        for item in self.distributions:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("native candidate distribution pin is invalid")
            name, version = item
            if (
                type(name) is not str or not _NAME.fullmatch(name) or name in seen
                or type(version) is not str or not _VERSION.fullmatch(version)
            ):
                raise ValueError("native candidate distribution pin is invalid")
            seen.add(name)
        if not {"mlx", "mlx-lm"} <= seen or tuple(sorted(self.distributions)) != self.distributions:
            raise ValueError("native candidate inventory is incomplete or unordered")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> MLXRuntimeExpectation:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native candidate has missing or unknown fields")
        distributions = value["distributions"]
        if type(distributions) is not dict:
            raise ValueError("native candidate inventory must be an object")
        return cls(
            value["python_version"], value["executable_sha256"],
            value["dependency_lock_sha256"], tuple(sorted(distributions.items())),
        )


def inspect_mlx_runtime_candidate(
    expected: MLXRuntimeExpectation, *, dependency_lock: bytes,
) -> dict[str, object]:
    """Compare complete observed package versions and the running executable.

    No install, framework import, model access, environment dump, or GPU call.
    The lock is opaque reviewed bytes; this is not a dependency resolver or a
    proof that installed package bytes match the allowed wheel hashes.
    """
    if type(expected) is not MLXRuntimeExpectation:
        raise TypeError("native candidate requires exact configured expectations")
    if (
        type(dependency_lock) is not bytes or not 1 <= len(dependency_lock) <= 1048576
        or hashlib.sha256(dependency_lock).hexdigest() != expected.dependency_lock_sha256
    ):
        raise ValueError("native candidate dependency lock differs")
    if (
        platform.system() != "Darwin" or platform.machine() != "arm64"
        or sys.implementation.name != "cpython"
        or platform.python_version() != expected.python_version
    ):
        raise ValueError("native candidate platform or Python differs")
    executable = Path(sys.executable).resolve(strict=True)
    with executable.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(64 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
        current = executable.stat()
    if (
        len(raw) > 64 * 1024 * 1024 or len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
        or hashlib.sha256(raw).hexdigest() != expected.executable_sha256
    ):
        raise ValueError("native candidate interpreter bytes differ")
    observed = {}
    for index, distribution in enumerate(metadata.distributions()):
        if index >= 256:
            raise ValueError("native candidate distribution count exceeded")
        name = distribution.metadata["Name"]
        version = distribution.version
        if type(name) is not str or len(name) > 128 or type(version) is not str:
            raise ValueError("native candidate distribution metadata is invalid")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if not _NAME.fullmatch(name) or not _VERSION.fullmatch(version) or name in observed:
            raise ValueError("native candidate distribution metadata is ambiguous")
        observed[name] = version
    if tuple(sorted(observed.items())) != expected.distributions:
        raise ValueError("native candidate installed inventory differs")
    return {
        "status": "CANDIDATE_MATCH_ONLY",
        "python_version": expected.python_version,
        "executable_sha256": expected.executable_sha256,
        "dependency_lock_sha256": expected.dependency_lock_sha256,
        "distributions": observed,
    }
