"""Canonical JSON, hashing and durable file primitives (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .errors import ConfigError


def check_json(value: Any, where: str = "$") -> None:
    """Reject anything that is not strictly JSON-representable and finite."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {where}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            check_json(item, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string key at {where}")
            check_json(item, f"{where}.{key}")
        return
    raise ValueError(f"unsupported JSON type {type(value).__name__} at {where}")


def canonical_bytes(value: Any) -> bytes:
    check_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def _no_constant(name: str):
    raise ValueError(f"non-finite constant {name} is not allowed")


def _no_duplicates(pairs):
    out = {}
    for key, val in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = val
    return out


def strict_loads(data: bytes | str) -> Any:
    """Parse JSON rejecting duplicate keys and NaN/Infinity."""
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    return json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_duplicates)


def load_strict_file(path: Path, *, what: str) -> tuple[Any, bytes]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read {what} {path}: {exc.strerror}", code="unreadable") from exc
    try:
        return strict_loads(raw), raw
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{what} {path} is not strict JSON: {exc}") from exc


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_only(path: Path, data: bytes) -> None:
    """Create a new file (never overwrite), fsync it and its directory."""
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            if n <= 0:
                raise OSError("file write made no progress")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(path.parent)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_finite_nonneg(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )
