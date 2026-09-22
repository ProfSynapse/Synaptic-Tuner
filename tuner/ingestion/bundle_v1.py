"""Private deterministic normalized-bundle storage for ingestion v1.

The bundle is deliberately small: one canonical manifest and one canonical
JSON-lines item stream.  Storage is assumed to be a private, local,
host-owned artifact root.  Verification is a point-in-time attestation of the
exact bytes observed through retained handles; it does not promise that those
bytes cannot be changed after verification.  This module does not discover
sources or interpret a StructureSet; callers supply normalized semantic items.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, NoReturn, TypeVar


ITEM_SCHEMA_VERSION = "syntunia-normalized-item/v1"
BUNDLE_SCHEMA_VERSION = "syntunia-normalized-bundle/v1"
MAX_ITEMS_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_STRUCTURE_SET_BYTES = 1024 * 1024
MAX_ITEM_COUNT = 100_000
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 200_000
MAX_CONTAINER_WIDTH = 100_000
MAX_JSON_SCALAR_BYTES = MAX_ITEMS_BYTES
_PATH_TYPE = type(Path())

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ITEM_FIELDS = frozenset(
    {
        "schema_version",
        "item_id",
        "logical_path",
        "source_sha256",
        "source_size_bytes",
        "structure_ref",
        "fields",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "bundle_digest",
        "structure_set",
        "structure_set_digest",
        "item_count",
        "items_bytes",
        "items_sha256",
        "logical_paths_sha256",
        "item_ids_sha256",
    }
)
_STRUCTURE_REF_FIELDS = frozenset({"name", "version", "digest"})
_EXPECTED_INVENTORY = frozenset({"manifest.json", "items.jsonl"})


class BundleValidationError(ValueError):
    """The requested or retained normalized bundle is invalid."""


class BundleCollisionError(RuntimeError):
    """A bundle destination exists but is not the identical verified bundle."""


class BundleDurabilityError(RuntimeError):
    """A private bundle root could not be safely durability-synchronized."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()


def _invalid(message: str = "normalized bundle is invalid") -> NoReturn:
    raise BundleValidationError(message) from None


def _digest_text(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _invalid(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(slots=True)
class _JsonBudget:
    remaining_bytes: int
    remaining_nodes: int = MAX_JSON_NODES

    def consume(self, size: int) -> None:
        self.remaining_nodes -= 1
        self.remaining_bytes -= size
        if self.remaining_nodes < 0:
            _invalid("canonical JSON exceeds the node limit")
        if self.remaining_bytes < 0:
            _invalid("canonical JSON exceeds the byte limit")


def _encoded_scalar_size(value: object, maximum_chars: int) -> int:
    if type(value) is str and len(value) > maximum_chars:
        _invalid("canonical JSON scalar exceeds the character limit")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError):
        _invalid("value is not canonical JSON")
    if len(encoded) > MAX_JSON_SCALAR_BYTES:
        _invalid("canonical JSON scalar exceeds the byte limit")
    return len(encoded)


def _normalize_json(value: object, *, maximum_bytes: int, depth: int = 0, budget: _JsonBudget | None = None) -> object:
    if budget is None:
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            _invalid("canonical JSON byte limit is invalid")
        budget = _JsonBudget(maximum_bytes, MAX_JSON_NODES)
    if depth > MAX_JSON_DEPTH:
        _invalid("canonical JSON exceeds the nesting limit")
    value_type = type(value)
    if value is None or value_type in (bool, str):
        budget.consume(_encoded_scalar_size(value, min(maximum_bytes, MAX_JSON_SCALAR_BYTES)))
        return value
    if value_type is int:
        if not -(2**63) <= value <= 2**63 - 1:
            _invalid("canonical JSON integer is outside int64")
        budget.consume(_encoded_scalar_size(value, maximum_bytes))
        return value
    if value_type is float:
        if not math.isfinite(value):
            _invalid("canonical JSON number must be finite")
        budget.consume(_encoded_scalar_size(value, maximum_bytes))
        return value
    if value_type is dict:
        if len(value) > MAX_CONTAINER_WIDTH:
            _invalid("canonical JSON object exceeds the width limit")
        budget.consume(2 + max(0, len(value) - 1))
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                _invalid("canonical JSON object keys must be exact strings")
            budget.consume(_encoded_scalar_size(key, min(maximum_bytes, MAX_JSON_SCALAR_BYTES)) + 1)
            result[key] = _normalize_json(
                item, maximum_bytes=maximum_bytes, depth=depth + 1, budget=budget
            )
        return result
    if value_type in (list, tuple):
        if len(value) > MAX_CONTAINER_WIDTH:
            _invalid("canonical JSON array exceeds the width limit")
        budget.consume(2 + max(0, len(value) - 1))
        return [
            _normalize_json(item, maximum_bytes=maximum_bytes, depth=depth + 1, budget=budget)
            for item in value
        ]
    _invalid("value is not exact built-in canonical JSON")


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if type(value) in (dict, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: dict[str, object], maximum_bytes: int = MAX_ITEMS_BYTES) -> bytes:
    try:
        normalized = _normalize_json(value, maximum_bytes=maximum_bytes)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except BundleValidationError:
        raise
    except BaseException:
        _invalid("value is not canonical JSON")
    if len(encoded) > maximum_bytes:
        _invalid("canonical JSON exceeds the byte limit")
    return encoded


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _invalid("canonical JSON contains duplicate keys")
        result[key] = value
    return result


def _preflight_json_shape(raw: bytes, name: str) -> None:
    """Bound parser work before ``json.loads`` materializes containers."""

    depth = 0
    nodes = 1
    in_string = False
    escaped = False
    widths: list[int] = []
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            nodes += 1
            if depth > MAX_JSON_DEPTH:
                _invalid(f"{name} exceeds the nesting limit")
            widths.append(1)
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0 or not widths:
                _invalid(f"{name} has invalid JSON structure")
            widths.pop()
        elif byte in (0x2C, 0x3A):
            nodes += 1
            if widths:
                widths[-1] += 1
                if widths[-1] > MAX_CONTAINER_WIDTH * 2 + 1:
                    _invalid(f"{name} exceeds the width limit")
        if nodes > MAX_JSON_NODES * 2:
            _invalid(f"{name} exceeds the node limit")
    if in_string or depth != 0:
        _invalid(f"{name} has invalid JSON structure")


def _parse_canonical_object(raw: bytes, maximum: int, name: str) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _invalid(f"{name} is not bounded canonical JSON")

    _preflight_json_shape(raw, name)

    def reject_constant(_: str) -> object:
        _invalid(f"{name} contains a non-finite number")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
        )
    except BundleValidationError:
        raise
    except BaseException:
        _invalid(f"{name} is not canonical UTF-8 JSON")
    if type(value) is not dict:
        _invalid(f"{name} is not a canonical JSON object")
    normalized = _normalize_json(value, maximum_bytes=maximum)
    if type(normalized) is not dict or _canonical_bytes(normalized, maximum) != raw:
        _invalid(f"{name} is not a canonical JSON object")
    return normalized


def _domain_digest(domain: str, payload: bytes) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()


def _exact_fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected:
        _invalid(f"{name} has missing or unknown fields")
    return value


def _logical_path(value: object) -> str:
    if type(value) is not str or not value:
        _invalid("logical_path must be bounded nonempty text")
    if len(value) > 4096:
        _invalid("logical_path must be bounded nonempty text")
    try:
        encoded_size = len(value.encode("utf-8"))
    except UnicodeError:
        _invalid("logical_path must be bounded nonempty text")
    if encoded_size > 4096:
        _invalid("logical_path must be bounded nonempty text")
    if value != unicodedata.normalize("NFC", value):
        _invalid("logical_path must be NFC normalized")
    if "\\" in value or "\0" in value or value.startswith("/"):
        _invalid("logical_path must be a relative POSIX path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _invalid("logical_path contains control characters")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or ":" in parts[0]:
        _invalid("logical_path contains an unsafe segment")
    return value


def _structure_ref(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _invalid("structure_ref must be an object")
    normalized = _normalize_json(value, maximum_bytes=4096)
    ref = _exact_fields(normalized, _STRUCTURE_REF_FIELDS, "structure_ref")
    for name in ("name", "version"):
        item = ref[name]
        if type(item) is not str or not item:
            _invalid(f"structure_ref {name} is invalid")
        if len(item) > 128:
            _invalid(f"structure_ref {name} is invalid")
        try:
            encoded_size = len(item.encode("utf-8"))
        except UnicodeError:
            _invalid(f"structure_ref {name} is invalid")
        if encoded_size > 128:
            _invalid(f"structure_ref {name} is invalid")
        if any(ord(character) < 32 or ord(character) == 127 for character in item):
            _invalid(f"structure_ref {name} is invalid")
    _digest_text(ref["digest"], "structure_ref digest")
    return ref


def _structure_set(value: object) -> tuple[dict[str, object], frozenset[bytes], str]:
    if type(value) is not dict:
        _invalid("structure_set must be an object")
    document = _normalize_json(value, maximum_bytes=MAX_STRUCTURE_SET_BYTES)
    document = _exact_fields(document, frozenset({"structures", "bindings"}), "structure_set")
    raw = _canonical_bytes(document, MAX_STRUCTURE_SET_BYTES)
    if len(raw) > MAX_STRUCTURE_SET_BYTES:
        _invalid("structure_set exceeds its byte limit")
    structures = document["structures"]
    bindings = document["bindings"]
    if type(structures) is not list or not structures or type(bindings) is not list or not bindings:
        _invalid("structure_set requires structures and bindings")
    refs: list[bytes] = []
    for structure in structures:
        if type(structure) is not dict or "ref" not in structure:
            _invalid("structure_set contains an invalid structure")
        refs.append(_canonical_bytes(_structure_ref(structure["ref"]), 4096))
    if len(refs) != len(set(refs)):
        _invalid("structure_set contains duplicate refs")
    return document, frozenset(refs), _domain_digest("synaptic-structure-set/v1", raw)


def _item_basis(
    logical_path: str,
    source_sha256: str,
    source_size_bytes: int,
    structure_ref: object,
    fields: object,
) -> dict[str, object]:
    return {
        "logical_path": logical_path,
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
        "structure_ref": _thaw_json(structure_ref),
        "fields": _thaw_json(fields),
    }


def _item_id(basis: dict[str, object]) -> str:
    return "item-" + _domain_digest(ITEM_SCHEMA_VERSION, _canonical_bytes(basis))


@dataclass(frozen=True, slots=True)
class NormalizedItemInputV1:
    """One semantic normalized item; schema and identity are writer-owned."""

    logical_path: str
    source_sha256: str
    source_size_bytes: int
    structure_ref: dict[str, object]
    fields: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_path", _logical_path(self.logical_path))
        object.__setattr__(self, "source_sha256", _digest_text(self.source_sha256, "source_sha256"))
        if type(self.source_size_bytes) is not int or not 0 <= self.source_size_bytes <= 2**63 - 1:
            _invalid("source_size_bytes must be a nonnegative int64")
        ref = _structure_ref(self.structure_ref)
        if type(self.fields) is not dict or not self.fields:
            _invalid("fields must be a nonempty object")
        fields = _normalize_json(self.fields, maximum_bytes=MAX_ITEMS_BYTES)
        if type(fields) is not dict:
            _invalid("fields must be an object")
        object.__setattr__(self, "structure_ref", _freeze_json(ref))
        object.__setattr__(self, "fields", _freeze_json(fields))


@dataclass(frozen=True, slots=True)
class BundleSemanticIdentityV1:
    bundle_id: str
    bundle_digest: str
    structure_set_digest: str
    item_count: int
    items_bytes: int
    items_sha256: str
    logical_paths_sha256: str
    item_ids_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "bundle_digest", "structure_set_digest", "items_sha256",
            "logical_paths_sha256", "item_ids_sha256",
        ):
            _digest_text(getattr(self, name), name)
        if type(self.bundle_id) is not str or self.bundle_id != "bundle-" + self.bundle_digest:
            _invalid("bundle_id does not bind bundle_digest")
        count = _exact_nonnegative_int(self.item_count, "item_count", MAX_ITEM_COUNT)
        size = _exact_nonnegative_int(self.items_bytes, "items_bytes", MAX_ITEMS_BYTES)
        if count < 1 or size < 1:
            _invalid("bundle counts must be positive")


@dataclass(frozen=True, slots=True)
class VerifiedNormalizedBundleV1:
    path: Path
    semantic_identity: BundleSemanticIdentityV1

    def __post_init__(self) -> None:
        if type(self.path) is not _PATH_TYPE:
            raise TypeError("path must be a Path")
        if type(self.semantic_identity) is not BundleSemanticIdentityV1:
            raise TypeError("semantic_identity must be exact BundleSemanticIdentityV1")


@dataclass(frozen=True, slots=True)
class NormalizedBundleItemRecordV1:
    """One immutable, verified item parsed from a normalized bundle."""

    item_id: str
    logical_path: str
    source_sha256: str
    source_size_bytes: int
    structure_ref: MappingProxyType
    fields: MappingProxyType

    def __post_init__(self) -> None:
        if type(self.item_id) is not str or not self.item_id.startswith("item-") or len(self.item_id) != 69:
            _invalid("item_id is invalid")
        object.__setattr__(self, "logical_path", _logical_path(self.logical_path))
        object.__setattr__(self, "source_sha256", _digest_text(self.source_sha256, "source_sha256"))
        if type(self.source_size_bytes) is not int or not 0 <= self.source_size_bytes <= 2**63 - 1:
            _invalid("source_size_bytes must be a nonnegative int64")
        if type(self.structure_ref) is not MappingProxyType or type(self.fields) is not MappingProxyType:
            raise TypeError("normalized bundle item mappings must be immutable")


@dataclass(frozen=True, slots=True)
class LoadedNormalizedBundleV1:
    """Immutable parsed records and identity from one verified bundle observation."""

    path: Path
    semantic_identity: BundleSemanticIdentityV1
    structure_set: MappingProxyType
    items: tuple[NormalizedBundleItemRecordV1, ...]

    def __post_init__(self) -> None:
        if type(self.path) is not _PATH_TYPE:
            raise TypeError("path must be a Path")
        if type(self.semantic_identity) is not BundleSemanticIdentityV1:
            raise TypeError("semantic_identity must be exact BundleSemanticIdentityV1")
        if type(self.structure_set) is not MappingProxyType:
            raise TypeError("structure_set must be immutable")
        if type(self.items) is not tuple or not self.items or len(self.items) > MAX_ITEM_COUNT:
            _invalid("loaded bundle items are invalid")
        if any(type(item) is not NormalizedBundleItemRecordV1 for item in self.items):
            raise TypeError("loaded bundle items must be exact records")
        if len(self.items) != self.semantic_identity.item_count:
            _invalid("loaded bundle item count does not match identity")


class BundlePublicationUncertaintyPhaseV1(str, Enum):
    PARENT_DURABILITY = "parent_durability"
    FINAL_VERIFICATION = "final_verification"


class BundlePublicationUncertainV1(RuntimeError):
    __slots__ = ("semantic_identity", "phase")

    def __init__(
        self,
        semantic_identity: BundleSemanticIdentityV1,
        phase: BundlePublicationUncertaintyPhaseV1,
    ) -> None:
        if type(semantic_identity) is not BundleSemanticIdentityV1:
            raise TypeError("semantic_identity must be exact BundleSemanticIdentityV1")
        if type(phase) is not BundlePublicationUncertaintyPhaseV1:
            raise TypeError("phase must be exact BundlePublicationUncertaintyPhaseV1")
        self.semantic_identity = semantic_identity
        self.phase = phase
        super().__init__()


class _FileSink:
    __slots__ = ("_path", "_handle", "_hash", "_maximum", "_size", "_closed", "_identity")

    def __init__(self, path: Path, maximum_bytes: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.chmod(path, 0o600)
            self._handle = os.fdopen(descriptor, "wb", buffering=0)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        self._path = path
        self._hash = hashlib.sha256()
        self._maximum = maximum_bytes
        self._size = 0
        self._closed = False
        self._identity = _identity(os.fstat(self._handle.fileno()))

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self._identity

    def write(self, chunk: bytes) -> None:
        if self._closed or type(chunk) is not bytes:
            _invalid("bundle sink is not writable")
        if self._size + len(chunk) > self._maximum:
            _invalid("bundle member exceeds its byte limit")
        offset = 0
        while offset < len(chunk):
            written = self._handle.write(chunk[offset:])
            if type(written) is not int or written <= 0:
                raise RuntimeError("bundle member write failed")
            offset += written
        self._hash.update(chunk)
        self._size += len(chunk)

    def finish(self) -> tuple[int, str]:
        if self._closed:
            _invalid("bundle sink is already closed")
        failed = False
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except BaseException:
            failed = True
        finally:
            try:
                self._handle.close()
            except BaseException:
                failed = True
            self._closed = True
        if failed:
            raise RuntimeError("bundle member finalization failed") from None
        return self._size, self._hash.hexdigest()

    def abort(self) -> None:
        if not self._closed:
            try:
                self._handle.close()
            except BaseException:
                pass
            self._closed = True


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _same_file(identity_a: tuple[int, int, int, int, int], identity_b: tuple[int, int, int, int, int]) -> bool:
    return (identity_a[0], identity_a[1], identity_a[3]) == (identity_b[0], identity_b[1], identity_b[3])


def _plain_directory_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
            _invalid("bundle path is not a plain directory")
        if path.resolve(strict=True) != path.absolute():
            _invalid("bundle path traverses a link")
        return _identity(info)
    except BundleValidationError:
        raise
    except BaseException:
        _invalid("bundle directory is unavailable")


def _bounded_inventory(
    path: Path, directory_identity: tuple[int, int, int, int, int]
) -> dict[str, tuple[int, int, int, int, int]]:
    if _plain_directory_identity(path) != directory_identity:
        _invalid("bundle directory changed during verification")
    result: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if len(result) == len(_EXPECTED_INVENTORY):
                    _invalid("bundle inventory must contain exactly two files")
                if type(entry.name) is not str or entry.name not in _EXPECTED_INVENTORY or entry.name in result:
                    _invalid("bundle inventory must contain exactly two files")
                info = (path / entry.name).lstat()
                if stat.S_ISLNK(info.st_mode) or _is_reparse(info) or not stat.S_ISREG(info.st_mode):
                    _invalid("bundle inventory contains a non-regular entry")
                result[entry.name] = _identity(info)
    except BundleValidationError:
        raise
    except BaseException:
        _invalid("bundle inventory could not be inspected safely")
    if frozenset(result) != _EXPECTED_INVENTORY:
        _invalid("bundle inventory must contain exactly two files")
    if _plain_directory_identity(path) != directory_identity:
        _invalid("bundle directory changed during verification")
    return result


class _RetainedMember:
    __slots__ = ("path", "maximum", "handle", "opened_identity", "closed")

    def __init__(
        self,
        path: Path,
        maximum: int,
        expected_identity: tuple[int, int, int, int, int],
    ) -> None:
        self.path = path
        self.maximum = maximum
        self.handle = None
        self.opened_identity = expected_identity
        self.closed = False
        try:
            declared = path.lstat()
            declared_identity = _identity(declared)
            if declared_identity != expected_identity:
                _invalid("bundle member identity changed during verification")
            if stat.S_ISLNK(declared.st_mode) or _is_reparse(declared) or not stat.S_ISREG(declared.st_mode):
                _invalid("bundle member is not a plain regular file")
            if declared.st_size > maximum or path.resolve(strict=True) != path.absolute():
                _invalid("bundle member is not a bounded plain file")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                self.handle = os.fdopen(descriptor, "rb", buffering=0)
            except BaseException:
                os.close(descriptor)
                raise
            opened = _identity(os.fstat(self.handle.fileno()))
            if not _same_file(declared_identity, opened):
                _invalid("bundle member identity changed during verification")
            if opened[3] > maximum:
                _invalid("bundle member exceeds its byte limit")
            if os.name != "nt" and stat.S_IMODE(opened[2]) & 0o077:
                _invalid("bundle member permissions are not private")
            self.opened_identity = opened
        except BundleValidationError:
            self.close()
            raise
        except BaseException:
            self.close()
            _invalid("bundle member could not be retained safely")

    def read(self) -> bytes:
        if self.handle is None or self.closed:
            _invalid("bundle member handle is unavailable")
        try:
            chunks: list[bytes] = []
            size = 0
            while size <= self.maximum:
                chunk = self.handle.read(min(1024 * 1024, self.maximum + 1 - size))
                if not chunk:
                    break
                if type(chunk) is not bytes:
                    _invalid("bundle member read returned invalid bytes")
                chunks.append(chunk)
                size += len(chunk)
            payload = b"".join(chunks)
        except BundleValidationError:
            raise
        except BaseException:
            _invalid("bundle member could not be read safely")
        if len(payload) > self.maximum or len(payload) != self.opened_identity[3]:
            _invalid("bundle member size changed during verification")
        return payload

    def verify_final_identity(self) -> None:
        if self.handle is None or self.closed:
            _invalid("bundle member handle is unavailable")
        try:
            final = _identity(os.fstat(self.handle.fileno()))
        except BaseException:
            _invalid("bundle member identity could not be revalidated")
        if final != self.opened_identity:
            _invalid("bundle member changed during verification")

    def close(self) -> bool:
        if self.closed or self.handle is None:
            return True
        failed = False
        try:
            self.handle.close()
        except BaseException:
            failed = True
        self.closed = True
        return not failed


def _stream_digest(values: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bundle_basis(
    *,
    structure_set: dict[str, object],
    structure_set_digest: str,
    item_count: int,
    items_bytes: int,
    items_sha256: str,
    logical_paths_sha256: str,
    item_ids_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "structure_set": _thaw_json(structure_set),
        "structure_set_digest": structure_set_digest,
        "item_count": item_count,
        "items_bytes": items_bytes,
        "items_sha256": items_sha256,
        "logical_paths_sha256": logical_paths_sha256,
        "item_ids_sha256": item_ids_sha256,
    }


def _manifest(
    *,
    structure_set: dict[str, object],
    structure_set_digest: str,
    item_count: int,
    items_bytes: int,
    items_sha256: str,
    logical_paths_sha256: str,
    item_ids_sha256: str,
) -> tuple[dict[str, object], str, str]:
    basis = _bundle_basis(
        structure_set=structure_set,
        structure_set_digest=structure_set_digest,
        item_count=item_count,
        items_bytes=items_bytes,
        items_sha256=items_sha256,
        logical_paths_sha256=logical_paths_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    bundle_digest = _domain_digest(BUNDLE_SCHEMA_VERSION, _canonical_bytes(basis))
    bundle_id = "bundle-" + bundle_digest
    document = dict(basis)
    document.update({"bundle_id": bundle_id, "bundle_digest": bundle_digest})
    return document, bundle_id, bundle_digest


def _item_document(item: NormalizedItemInputV1) -> tuple[dict[str, object], str]:
    basis = _item_basis(
        item.logical_path,
        item.source_sha256,
        item.source_size_bytes,
        item.structure_ref,
        item.fields,
    )
    item_id = _item_id(basis)
    document = dict(basis)
    document.update({"schema_version": ITEM_SCHEMA_VERSION, "item_id": item_id})
    return document, item_id


def _require_plain_directory(path: Path) -> None:
    _plain_directory_identity(path)


def _exact_nonnegative_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _invalid(f"{name} must be an exact bounded nonnegative integer")
    return value


def _validate_manifest_scalars(manifest: dict[str, object]) -> None:
    if type(manifest["schema_version"]) is not str or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        _invalid("manifest schema version is unsupported")
    bundle_id = manifest["bundle_id"]
    if type(bundle_id) is not str or not bundle_id.startswith("bundle-") or len(bundle_id) != 71:
        _invalid("manifest bundle_id is invalid")
    for name in (
        "bundle_digest",
        "structure_set_digest",
        "items_sha256",
        "logical_paths_sha256",
        "item_ids_sha256",
    ):
        _digest_text(manifest[name], name)
    count = _exact_nonnegative_int(manifest["item_count"], "item_count", MAX_ITEM_COUNT)
    if count < 1:
        _invalid("item_count must be positive")
    size = _exact_nonnegative_int(manifest["items_bytes"], "items_bytes", MAX_ITEMS_BYTES)
    if size < 1:
        _invalid("items_bytes must be positive")


def _bounded_json_lines(raw: bytes) -> list[bytes]:
    if not raw or not raw.endswith(b"\n"):
        _invalid("items.jsonl must be nonempty and newline terminated")
    lines: list[bytes] = []
    start = 0
    while start < len(raw):
        end = raw.find(b"\n", start)
        if end < 0 or end == start:
            _invalid("items.jsonl has an invalid row count")
        lines.append(raw[start:end])
        if len(lines) > MAX_ITEM_COUNT:
            _invalid("items.jsonl has an invalid row count")
        start = end + 1
    return lines


def _load_observed_bytes(
    path: Path,
    manifest_raw: bytes,
    items_raw: bytes,
    *,
    require_content_addressed_name: bool,
) -> LoadedNormalizedBundleV1:
    manifest = _exact_fields(
        _parse_canonical_object(manifest_raw, MAX_MANIFEST_BYTES, "manifest"),
        _MANIFEST_FIELDS,
        "manifest",
    )
    _validate_manifest_scalars(manifest)
    structure_document, structure_refs, structure_set_digest = _structure_set(manifest["structure_set"])
    if manifest["structure_set_digest"] != structure_set_digest:
        _invalid("manifest structure_set digest is invalid")

    lines = _bounded_json_lines(items_raw)
    logical_paths: list[str] = []
    item_ids: list[str] = []
    parsed_items: list[NormalizedBundleItemRecordV1] = []
    for line in lines:
        row = _exact_fields(
            _parse_canonical_object(line, MAX_ITEMS_BYTES, "item row"),
            _ITEM_FIELDS,
            "item row",
        )
        if row["schema_version"] != ITEM_SCHEMA_VERSION:
            _invalid("item schema version is unsupported")
        logical_path = _logical_path(row["logical_path"])
        source_sha256 = _digest_text(row["source_sha256"], "source_sha256")
        source_size = row["source_size_bytes"]
        if type(source_size) is not int or not 0 <= source_size <= 2**63 - 1:
            _invalid("source_size_bytes must be a nonnegative int64")
        structure_ref = _structure_ref(row["structure_ref"])
        if _canonical_bytes(structure_ref) not in structure_refs:
            _invalid("item structure_ref is not declared by the StructureSet")
        fields = row["fields"]
        if type(fields) is not dict or not fields:
            _invalid("item fields must be a nonempty object")
        basis = _item_basis(logical_path, source_sha256, source_size, structure_ref, fields)
        expected_id = _item_id(basis)
        if row["item_id"] != expected_id:
            _invalid("item_id does not bind the item")
        logical_paths.append(logical_path)
        item_ids.append(expected_id)
        parsed_items.append(
            NormalizedBundleItemRecordV1(
                item_id=expected_id,
                logical_path=logical_path,
                source_sha256=source_sha256,
                source_size_bytes=source_size,
                structure_ref=_freeze_json(structure_ref),
                fields=_freeze_json(fields),
            )
        )
    if logical_paths != sorted(logical_paths) or len(logical_paths) != len(set(logical_paths)):
        _invalid("item logical paths must be unique and sorted")
    if len(item_ids) != len(set(item_ids)):
        _invalid("item IDs must be unique")

    items_sha256 = hashlib.sha256(items_raw).hexdigest()
    logical_paths_sha256 = _stream_digest(logical_paths)
    item_ids_sha256 = _stream_digest(item_ids)
    expected_manifest, bundle_id, bundle_digest = _manifest(
        structure_set=structure_document,
        structure_set_digest=structure_set_digest,
        item_count=len(lines),
        items_bytes=len(items_raw),
        items_sha256=items_sha256,
        logical_paths_sha256=logical_paths_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    expected_manifest_raw = _canonical_bytes(expected_manifest, MAX_MANIFEST_BYTES)
    if manifest_raw != expected_manifest_raw:
        _invalid("manifest does not bind the verified bundle")
    if require_content_addressed_name and path.name != bundle_id:
        _invalid("bundle directory name is not content addressed")
    identity = BundleSemanticIdentityV1(
        bundle_id=bundle_id,
        bundle_digest=bundle_digest,
        structure_set_digest=structure_set_digest,
        item_count=len(lines),
        items_bytes=len(items_raw),
        items_sha256=items_sha256,
        logical_paths_sha256=logical_paths_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    frozen_structure_set = _freeze_json(structure_document)
    if type(frozen_structure_set) is not MappingProxyType:
        _invalid("structure_set is invalid")
    return LoadedNormalizedBundleV1(
        path=path,
        semantic_identity=identity,
        structure_set=frozen_structure_set,
        items=tuple(parsed_items),
    )


def _verify_observed_bytes(
    path: Path,
    manifest_raw: bytes,
    items_raw: bytes,
    *,
    require_content_addressed_name: bool,
) -> VerifiedNormalizedBundleV1:
    loaded = _load_observed_bytes(
        path,
        manifest_raw,
        items_raw,
        require_content_addressed_name=require_content_addressed_name,
    )
    return VerifiedNormalizedBundleV1(path=loaded.path, semantic_identity=loaded.semantic_identity)


_ObservedBundleResult = TypeVar("_ObservedBundleResult")


def _attest(
    path: Path,
    *,
    require_content_addressed_name: bool,
    parse_observed_bytes: Callable[..., _ObservedBundleResult],
) -> _ObservedBundleResult:
    """Attest the exact bundle bytes observed during one retained-handle transaction."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be a Path")
    directory_identity = _plain_directory_identity(path)
    member_identities = _bounded_inventory(path, directory_identity)
    retained: list[_RetainedMember] = []
    result: _ObservedBundleResult | None = None
    close_failed = False
    try:
        manifest_member = _RetainedMember(
            path / "manifest.json", MAX_MANIFEST_BYTES, member_identities["manifest.json"]
        )
        retained.append(manifest_member)
        items_member = _RetainedMember(
            path / "items.jsonl", MAX_ITEMS_BYTES, member_identities["items.jsonl"]
        )
        retained.append(items_member)
        manifest_raw = manifest_member.read()
        items_raw = items_member.read()
        result = parse_observed_bytes(
            path,
            manifest_raw,
            items_raw,
            require_content_addressed_name=require_content_addressed_name,
        )
        manifest_member.verify_final_identity()
        items_member.verify_final_identity()
        if _bounded_inventory(path, directory_identity) != member_identities:
            _invalid("bundle inventory changed during verification")
    finally:
        for member in reversed(retained):
            if not member.close():
                close_failed = True
    if close_failed:
        _invalid("bundle member handle could not be closed safely")
    if result is None:
        _invalid("bundle attestation did not complete")
    return result


def _verify(path: Path, *, require_content_addressed_name: bool) -> VerifiedNormalizedBundleV1:
    return _attest(
        path,
        require_content_addressed_name=require_content_addressed_name,
        parse_observed_bytes=_verify_observed_bytes,
    )


def verify_normalized_bundle_v1(path: Path) -> VerifiedNormalizedBundleV1:
    """Attest an existing bundle's exact point-in-time observed bytes."""

    return _verify(path, require_content_addressed_name=True)


def load_verified_normalized_bundle_v1(path: Path) -> LoadedNormalizedBundleV1:
    """Load immutable records from one exact, verified bundle observation."""

    return _attest(
        path,
        require_content_addressed_name=True,
        parse_observed_bytes=_load_observed_bytes,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def retry_bundle_root_durability_v1(bundles_root: Path) -> None:
    """Retry durability for one exact private-local plain bundle root."""

    failed = type(bundles_root) is not _PATH_TYPE
    if not failed:
        try:
            _require_plain_directory(bundles_root)
            _fsync_directory(bundles_root)
        except BaseException:
            failed = True
    if failed:
        raise BundleDurabilityError() from None


def _rename_noreplace(source: Path, destination: Path) -> None:
    if source.parent != destination.parent:
        raise RuntimeError("bundle publication crossed directories")
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _best_effort_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD if os.name == "nt" else 0o400)
    except OSError:
        pass


def write_normalized_bundle_v1(
    bundles_root: Path,
    structure_set: dict[str, object],
    items: tuple[NormalizedItemInputV1, ...] | list[NormalizedItemInputV1],
) -> VerifiedNormalizedBundleV1:
    """Write or identically reuse one deterministic content-addressed bundle."""

    if type(bundles_root) is not _PATH_TYPE:
        raise TypeError("bundles_root must be a Path")
    structure_document, structure_refs, structure_set_digest = _structure_set(structure_set)
    if type(items) not in (tuple, list):
        raise TypeError("items must be an exact tuple or list of NormalizedItemInputV1")
    if len(items) > MAX_ITEM_COUNT:
        _invalid("item count exceeds its limit")
    retained: list[NormalizedItemInputV1] = []
    for item in items:
        if type(item) is not NormalizedItemInputV1:
            raise TypeError("items must contain exact NormalizedItemInputV1 values")
        retained.append(item)
    if not retained:
        _invalid("at least one normalized item is required")
    retained.sort(key=lambda item: item.logical_path)
    paths = [item.logical_path for item in retained]
    if len(paths) != len(set(paths)):
        _invalid("logical paths must be unique")

    item_chunks: list[bytes] = []
    item_ids: list[str] = []
    items_size = 0
    for item in retained:
        retained_ref = _structure_ref(_thaw_json(item.structure_ref))
        if _canonical_bytes(retained_ref, 4096) not in structure_refs:
            _invalid("item structure_ref is not declared by the StructureSet")
        document, item_id = _item_document(item)
        chunk = _canonical_bytes(document) + b"\n"
        items_size += len(chunk)
        if items_size > MAX_ITEMS_BYTES:
            _invalid("items.jsonl exceeds its byte limit")
        item_chunks.append(chunk)
        item_ids.append(item_id)
    if len(item_ids) != len(set(item_ids)):
        _invalid("item IDs must be unique")
    items_raw = b"".join(item_chunks)
    items_sha256 = hashlib.sha256(items_raw).hexdigest()
    logical_paths_sha256 = _stream_digest(paths)
    item_ids_sha256 = _stream_digest(item_ids)
    manifest, bundle_id, bundle_digest = _manifest(
        structure_set=structure_document,
        structure_set_digest=structure_set_digest,
        item_count=len(retained),
        items_bytes=len(items_raw),
        items_sha256=items_sha256,
        logical_paths_sha256=logical_paths_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    semantic_identity = BundleSemanticIdentityV1(
        bundle_id=bundle_id,
        bundle_digest=bundle_digest,
        structure_set_digest=structure_set_digest,
        item_count=len(retained),
        items_bytes=len(items_raw),
        items_sha256=items_sha256,
        logical_paths_sha256=logical_paths_sha256,
        item_ids_sha256=item_ids_sha256,
    )
    manifest_raw = _canonical_bytes(manifest, MAX_MANIFEST_BYTES)
    if len(manifest_raw) > MAX_MANIFEST_BYTES:
        _invalid("manifest exceeds its byte limit")

    try:
        bundles_root.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(bundles_root)
        destination = bundles_root / bundle_id
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            try:
                existing = verify_normalized_bundle_v1(destination)
            except BundleValidationError:
                raise BundleCollisionError("bundle destination collision") from None
            if existing.semantic_identity != semantic_identity:
                raise BundleCollisionError("bundle destination collision") from None
            return existing
        stage = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=bundles_root))
        os.chmod(stage, 0o700)
        _stage_diagnostic_identity = _plain_directory_identity(stage)
    except BundleValidationError:
        raise
    except BundleCollisionError:
        raise
    except OSError:
        _invalid("bundle staging directory could not be created")
    _member_diagnostic_identities: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for name, payload, maximum in (
            ("items.jsonl", items_raw, MAX_ITEMS_BYTES),
            ("manifest.json", manifest_raw, MAX_MANIFEST_BYTES),
        ):
            sink: _FileSink | None = None
            try:
                sink = _FileSink(stage / name, maximum)
                _member_diagnostic_identities[name] = sink.identity
                sink.write(payload)
                sink.finish()
            except BaseException:
                if sink is not None:
                    sink.abort()
                raise
        for name in _EXPECTED_INVENTORY:
            _best_effort_readonly(stage / name)
        # Retained only for local diagnostics; these identities grant no cleanup
        # or deletion authority under the private-host artifact-root model.
        _ = (_stage_diagnostic_identity, tuple(sorted(_member_diagnostic_identities.items())))
        _fsync_directory(stage)
        staged = _verify(stage, require_content_addressed_name=False)
        if staged.semantic_identity != semantic_identity:
            _invalid("staged bundle identity is inconsistent")
        try:
            _rename_noreplace(stage, destination)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise BundleCollisionError("bundle destination collision") from None
            raise
        uncertain_phase: BundlePublicationUncertaintyPhaseV1 | None = None
        try:
            _fsync_directory(bundles_root)
        except BaseException:
            uncertain_phase = BundlePublicationUncertaintyPhaseV1.PARENT_DURABILITY
        if uncertain_phase is not None:
            raise BundlePublicationUncertainV1(semantic_identity, uncertain_phase) from None
        verified: VerifiedNormalizedBundleV1 | None = None
        try:
            verified = verify_normalized_bundle_v1(destination)
        except BaseException:
            uncertain_phase = BundlePublicationUncertaintyPhaseV1.FINAL_VERIFICATION
        if uncertain_phase is not None:
            raise BundlePublicationUncertainV1(semantic_identity, uncertain_phase) from None
        if verified is None or verified.semantic_identity != semantic_identity:
            raise BundlePublicationUncertainV1(
                semantic_identity, BundlePublicationUncertaintyPhaseV1.FINAL_VERIFICATION
            ) from None
        return verified
    except BundlePublicationUncertainV1:
        raise
    except (BundleValidationError, BundleCollisionError, TypeError):
        raise
    except BaseException:
        raise RuntimeError("normalized bundle could not be published") from None


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "ITEM_SCHEMA_VERSION",
    "MAX_ITEMS_BYTES",
    "BundleCollisionError",
    "BundleDurabilityError",
    "BundlePublicationUncertainV1",
    "BundlePublicationUncertaintyPhaseV1",
    "BundleSemanticIdentityV1",
    "BundleValidationError",
    "LoadedNormalizedBundleV1",
    "NormalizedBundleItemRecordV1",
    "NormalizedItemInputV1",
    "VerifiedNormalizedBundleV1",
    "load_verified_normalized_bundle_v1",
    "verify_normalized_bundle_v1",
    "retry_bundle_root_durability_v1",
    "write_normalized_bundle_v1",
]
