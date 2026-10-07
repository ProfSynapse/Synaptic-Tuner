"""Canonical physical identity for one protected experiment-tracking root."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .registry import _PathLock


SCHEMA_VERSION = "synaptic-tracking-root-identity/v1"
MARKER_NAME = ".synaptic-tracking-root.json"


class TrackingRootIdentityError(ValueError):
    """The durable tracking root is missing, copied, replaced, or unsafe."""


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrackingRootIdentityError("Tracking-root identity is not canonical JSON data") from exc


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _canonical_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    return os.path.normcase(str(resolved))


def _root_fingerprint(root: Path) -> dict[str, str]:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise TrackingRootIdentityError("Tracking root must be a physical directory")
    info = root.stat()
    return {
        "canonical_path": _canonical_path(root),
        "device": str(info.st_dev),
        "inode": str(info.st_ino),
    }


def _identity_id(document: Mapping[str, object]) -> str:
    body = {key: value for key, value in document.items() if key != "root_id"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _build(root: Path) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "root_id": "0" * 64,
        **_root_fingerprint(root),
    }
    document["root_id"] = _identity_id(document)
    return document


def _validate(document: Mapping[str, object], root: Path) -> dict[str, object]:
    expected_keys = {"schema_version", "root_id", "canonical_path", "device", "inode"}
    if set(document) != expected_keys:
        raise TrackingRootIdentityError("Tracking-root identity has unknown or missing fields")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise TrackingRootIdentityError("Tracking-root identity schema is unsupported")
    if document.get("root_id") != _identity_id(document):
        raise TrackingRootIdentityError("Tracking-root identity digest is invalid")
    if any(not isinstance(document.get(key), str) for key in expected_keys):
        raise TrackingRootIdentityError("Tracking-root identity fields must be strings")
    if document["root_id"] != _build(root)["root_id"]:
        raise TrackingRootIdentityError("Tracking root was copied, moved, or physically replaced")
    return dict(document)


def _read_marker(marker: Path, root: Path) -> dict[str, object]:
    if _is_link_or_reparse(marker) or not marker.is_file():
        raise TrackingRootIdentityError("Tracking-root marker must be a physical regular file")
    try:
        payload: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrackingRootIdentityError("Tracking-root marker is unreadable or invalid") from exc
    if not isinstance(payload, dict):
        raise TrackingRootIdentityError("Tracking-root marker must contain an object")
    canonical = canonical_json_bytes(payload)
    try:
        if marker.read_bytes() != canonical:
            raise TrackingRootIdentityError("Tracking-root marker bytes are not canonical")
    except OSError as exc:
        raise TrackingRootIdentityError("Tracking-root marker is unreadable") from exc
    return _validate(payload, root)


def _create_marker(marker: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(dir=str(marker.parent), suffix=".root.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, marker)
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def ensure_tracking_root_identity(root: str | Path) -> dict[str, object]:
    """Create-or-authenticate the immutable identity for ``root``."""

    candidate = Path(root).expanduser()
    candidate.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(candidate):
        raise TrackingRootIdentityError("Tracking root cannot be a link or reparse point")
    candidate = candidate.resolve(strict=True)
    marker = candidate / MARKER_NAME
    with _PathLock(marker):
        if not marker.exists():
            _create_marker(marker, canonical_json_bytes(_build(candidate)))
        return _read_marker(marker, candidate)


def require_tracking_root_identity(root: str | Path, expected_root_id: str) -> dict[str, object]:
    """Authenticate ``root`` and require the exact previously bound identity."""

    document = ensure_tracking_root_identity(root)
    if document["root_id"] != expected_root_id:
        raise TrackingRootIdentityError("Tracking-root identity does not match the protected record")
    return document


__all__ = [
    "MARKER_NAME",
    "SCHEMA_VERSION",
    "TrackingRootIdentityError",
    "canonical_json_bytes",
    "ensure_tracking_root_identity",
    "require_tracking_root_identity",
]
