"""Strict, non-extracting Docker image archive evidence reader."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


BLOCK_SIZE = 512
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2048
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_LAYER_COUNT = 256
MAX_STDERR_BYTES = 256 * 1024
ARCHIVE_TIMEOUT_SECONDS = 900
DIGEST_PREFIX = "sha256:"


class DockerArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class DockerArchiveCommand:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    timeout_seconds: int = ARCHIVE_TIMEOUT_SECONDS
    maximum_archive_bytes: int = MAX_ARCHIVE_BYTES
    maximum_stderr_bytes: int = MAX_STDERR_BYTES


@dataclass(frozen=True)
class DockerArchiveEvidence:
    config_raw: bytes
    config_sha256: str
    config_size: int
    diff_ids: tuple[str, ...]
    observed_layer_diff_ids: tuple[str, ...]
    layer_members: tuple[str, ...]
    member_count: int
    archive_size: int
    archive_format: str = "LEGACY_DOCKER"
    compatibility_manifest_sha256: str | None = None
    index_source_annotation_sha256: str | None = None


def _sha256(raw: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(raw).hexdigest()


def _json_value(raw: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("constant")

    _assert_json_depth(raw)
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=lambda _value: (_ for _ in ()).throw(ValueError("float")),
            parse_int=_bounded_json_integer,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID") from exc
    return value


def _bounded_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ValueError("integer")
    return int(value)


def _assert_json_depth(raw: bytes, *, maximum: int = 64) -> None:
    depth = 0
    quoted = False
    escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                quoted = False
        elif byte == 0x22:
            quoted = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > maximum:
                raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
        elif byte in (0x5D, 0x7D):
            depth -= 1
            if depth < 0:
                raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    if quoted or escaped or depth:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")


def _json_object(raw: bytes) -> dict[str, object]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    return value


def _field(raw: bytes) -> str:
    head, separator, tail = raw.partition(b"\0")
    if separator and any(tail):
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    try:
        return head.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID") from exc


def _octal(raw: bytes, *, maximum: int) -> int:
    if not raw or raw[0] & 0x80:
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    value = raw.rstrip(b"\0 ")
    if not value or len(value) > 11 or any(byte not in b"01234567" for byte in value):
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    parsed = int(value, 8)
    if parsed > maximum:
        raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
    return parsed


def _member_name(header: bytes, *, member_kind: str) -> str:
    name = _field(header[0:100])
    prefix = _field(header[345:500])
    combined = f"{prefix}/{name}" if prefix else name
    if (
        not combined or len(combined.encode("ascii")) > 255
        or combined.startswith("/") or "\\" in combined
    ):
        raise DockerArchiveError("ARCHIVE_PATH_INVALID")
    if combined.endswith("/"):
        if member_kind != "directory" or combined.endswith("//"):
            raise DockerArchiveError("ARCHIVE_PATH_INVALID")
        combined = combined[:-1]
    parts = combined.split("/")
    if any(part in {"", ".", ".."} for part in parts) or str(PurePosixPath(*parts)) != combined:
        raise DockerArchiveError("ARCHIVE_PATH_INVALID")
    return combined


def _header(header: bytes) -> tuple[str, str, int]:
    if len(header) != BLOCK_SIZE or header[257:263] != b"ustar\0" or header[263:265] != b"00":
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    stored = _octal(header[148:156], maximum=255 * BLOCK_SIZE)
    calculated = sum(header[:148]) + (32 * 8) + sum(header[156:])
    if stored != calculated:
        raise DockerArchiveError("ARCHIVE_CHECKSUM_INVALID")
    kind = header[156:157]
    if kind in {b"", b"\0", b"0"}:
        member_kind = "file"
    elif kind == b"5":
        member_kind = "directory"
    else:
        raise DockerArchiveError("ARCHIVE_TYPE_INVALID")
    size = _octal(header[124:136], maximum=MAX_ARCHIVE_BYTES)
    if member_kind == "directory" and size:
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    if _field(header[157:257]):
        raise DockerArchiveError("ARCHIVE_HEADER_INVALID")
    return _member_name(header, member_kind=member_kind), member_kind, size


def _read_exact(stream, size: int) -> bytes:  # noqa: ANN001
    value = stream.read(size)
    if not isinstance(value, bytes) or len(value) != size:
        raise DockerArchiveError("ARCHIVE_TRUNCATED")
    return value


def _skip_exact(stream, size: int, *, deadline: float | None = None) -> None:  # noqa: ANN001
    remaining = size
    while remaining:
        if deadline is not None and time.monotonic() >= deadline:
            raise DockerArchiveError("ARCHIVE_TIMEOUT")
        chunk = stream.read(min(1024 * 1024, remaining))
        if not isinstance(chunk, bytes) or not chunk:
            raise DockerArchiveError("ARCHIVE_TRUNCATED")
        remaining -= len(chunk)


def _require_zero_remaining(stream, *, deadline: float) -> None:  # noqa: ANN001
    while True:
        if time.monotonic() >= deadline:
            raise DockerArchiveError("ARCHIVE_TIMEOUT")
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return
        if any(chunk):
            raise DockerArchiveError("ARCHIVE_TRAILING_DATA")


def inspect_docker_archive(
    archive: Path, *, expected_config_digest: str, expected_config_size: int,
    expected_layer_count: int, timeout_seconds: int = ARCHIVE_TIMEOUT_SECONDS,
) -> DockerArchiveEvidence:
    if (
        not expected_config_digest.startswith(DIGEST_PREFIX)
        or len(expected_config_digest) != 71
        or any(value not in "0123456789abcdef" for value in expected_config_digest[7:])
        or not 1 <= expected_config_size <= MAX_METADATA_BYTES
        or not 1 <= expected_layer_count <= MAX_LAYER_COUNT
        or not 1 <= timeout_seconds <= ARCHIVE_TIMEOUT_SECONDS
    ):
        raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
    try:
        info = archive.lstat()
        resolved = archive.resolve(strict=True)
    except OSError as exc:
        raise DockerArchiveError("ARCHIVE_FILE_INVALID") from exc
    if archive.is_symlink() or not stat.S_ISREG(info.st_mode) or not resolved.is_absolute():
        raise DockerArchiveError("ARCHIVE_FILE_INVALID")
    if not 1024 <= info.st_size <= MAX_ARCHIVE_BYTES or info.st_size % BLOCK_SIZE:
        raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")

    expected_config_name = expected_config_digest[7:] + ".json"
    names: set[str] = set()
    regular_names: set[str] = set()
    manifest_raw: bytes | None = None
    config_raw: bytes | None = None
    top_level_configs: list[str] = []
    layer_hashes: dict[str, str] = {}
    metadata_bytes = 0
    member_count = 0
    zeros = 0
    deadline = time.monotonic() + timeout_seconds
    with resolved.open("rb", buffering=0) as stream:
        while stream.tell() < info.st_size:
            if time.monotonic() >= deadline:
                raise DockerArchiveError("ARCHIVE_TIMEOUT")
            header = _read_exact(stream, BLOCK_SIZE)
            if header == bytes(BLOCK_SIZE):
                zeros += 1
                if zeros >= 2:
                    _require_zero_remaining(stream, deadline=deadline)
                    break
                continue
            if zeros:
                raise DockerArchiveError("ARCHIVE_TRAILING_DATA")
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
            name, kind, size = _header(header)
            if name in names:
                raise DockerArchiveError("ARCHIVE_DUPLICATE_MEMBER")
            names.add(name)
            if kind == "file":
                regular_names.add(name)
                if "/" not in name and len(name) == 69 and name.endswith(".json"):
                    top_level_configs.append(name)
            capture = name in {"manifest.json", expected_config_name}
            if capture:
                metadata_bytes += size
                if kind != "file" or size > MAX_METADATA_BYTES or metadata_bytes > MAX_METADATA_BYTES:
                    raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
                raw = _read_exact(stream, size)
                if name == "manifest.json":
                    manifest_raw = raw
                else:
                    config_raw = raw
            elif kind == "file" and name.endswith("/layer.tar"):
                if size < 1:
                    raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    if time.monotonic() >= deadline:
                        raise DockerArchiveError("ARCHIVE_TIMEOUT")
                    chunk = _read_exact(stream, min(1024 * 1024, remaining))
                    digest.update(chunk)
                    remaining -= len(chunk)
                layer_hashes[name] = DIGEST_PREFIX + digest.hexdigest()
            else:
                _skip_exact(stream, size, deadline=deadline)
            padding = (-size) % BLOCK_SIZE
            if padding and any(_read_exact(stream, padding)):
                raise DockerArchiveError("ARCHIVE_PADDING_INVALID")
        else:
            raise DockerArchiveError("ARCHIVE_TRUNCATED")
    if zeros < 2 or manifest_raw is None or config_raw is None:
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    if top_level_configs != [expected_config_name]:
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    manifest = _json_value(manifest_raw)
    if not isinstance(manifest, list) or len(manifest) != 1 or not isinstance(manifest[0], dict):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    record = manifest[0]
    if set(record) != {"Config", "RepoTags", "Layers"}:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    tags = record.get("RepoTags")
    layers = record.get("Layers")
    if (
        record.get("Config") != expected_config_name or tags not in (None, [])
        or not isinstance(layers, list) or len(layers) != expected_layer_count
        or any(not isinstance(name, str) or name not in regular_names or name not in layer_hashes for name in layers)
        or len(set(layers)) != len(layers)
        or set(layers) != set(layer_hashes)
    ):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    if len(config_raw) != expected_config_size or _sha256(config_raw) != expected_config_digest:
        raise DockerArchiveError("ARCHIVE_CONFIG_IDENTITY_INVALID")
    config = _json_object(config_raw)
    if config.get("os") != "linux" or config.get("architecture") != "amd64" or "variant" in config:
        raise DockerArchiveError("ARCHIVE_PLATFORM_INVALID")
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, dict) or set(rootfs) != {"type", "diff_ids"}:
        raise DockerArchiveError("ARCHIVE_ROOTFS_INVALID")
    diff_ids = rootfs.get("diff_ids")
    if (
        rootfs.get("type") != "layers" or not isinstance(diff_ids, list)
        or len(diff_ids) != expected_layer_count
        or any(
            not isinstance(value, str) or not value.startswith(DIGEST_PREFIX) or len(value) != 71
            or any(character not in "0123456789abcdef" for character in value[7:])
            for value in diff_ids
        )
    ):
        raise DockerArchiveError("ARCHIVE_ROOTFS_INVALID")
    observed_layer_diff_ids = tuple(layer_hashes[name] for name in layers)
    if observed_layer_diff_ids != tuple(diff_ids):
        raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID")
    return DockerArchiveEvidence(
        config_raw=config_raw, config_sha256=expected_config_digest,
        config_size=len(config_raw), diff_ids=tuple(diff_ids),
        observed_layer_diff_ids=observed_layer_diff_ids,
        layer_members=tuple(layers), member_count=member_count,
        archive_size=info.st_size,
    )


_inspect_legacy_archive = inspect_docker_archive
OCI_LAYOUT_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"
OCI_PLAIN_LAYER = "application/vnd.oci.image.layer.v1.tar"
OCI_GZIP_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
DOCKER_GZIP_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
_OCI_LAYER_MEDIA_TYPES = frozenset({OCI_PLAIN_LAYER, OCI_GZIP_LAYER, DOCKER_GZIP_LAYER})
_CHUNK_BYTES = 1024 * 1024


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DockerArchiveError("ARCHIVE_FILE_INVALID") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise DockerArchiveError("ARCHIVE_FILE_INVALID")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)


def _scan_topology(path: Path, *, deadline: float) -> dict[str, tuple[str, int]]:
    info = path.lstat()
    members: dict[str, tuple[str, int]] = {}
    zeros = 0
    with path.open("rb", buffering=0) as stream:
        while stream.tell() < info.st_size:
            if time.monotonic() >= deadline:
                raise DockerArchiveError("ARCHIVE_TIMEOUT")
            header = _read_exact(stream, BLOCK_SIZE)
            if header == bytes(BLOCK_SIZE):
                zeros += 1
                if zeros >= 2:
                    _require_zero_remaining(stream, deadline=deadline)
                    break
                continue
            if zeros:
                raise DockerArchiveError("ARCHIVE_TRAILING_DATA")
            if len(members) >= MAX_ARCHIVE_MEMBERS:
                raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
            name, kind, size = _header(header)
            if name in members:
                raise DockerArchiveError("ARCHIVE_DUPLICATE_MEMBER")
            members[name] = (kind, size)
            _skip_exact(stream, size, deadline=deadline)
            padding = (-size) % BLOCK_SIZE
            if padding and any(_read_exact(stream, padding)):
                raise DockerArchiveError("ARCHIVE_PADDING_INVALID")
        else:
            raise DockerArchiveError("ARCHIVE_TRUNCATED")
    if zeros < 2:
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    return members


def _stream_oci_layer(
    stream, *, size: int, media_type: str, deadline: float,
    decompressed_total: list[int],
) -> tuple[str, str]:  # noqa: ANN001
    compressed = hashlib.sha256()
    plain = hashlib.sha256()
    remaining = size
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if media_type != OCI_PLAIN_LAYER else None
    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise DockerArchiveError("ARCHIVE_TIMEOUT")
    def consume(output: bytes) -> None:
        check_deadline()
        if len(output) > _CHUNK_BYTES:
            raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
        decompressed_total[0] += len(output)
        if decompressed_total[0] > MAX_ARCHIVE_BYTES:
            raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
        plain.update(output)
    try:
        while remaining:
            check_deadline()
            chunk = _read_exact(stream, min(_CHUNK_BYTES, remaining))
            remaining -= len(chunk)
            compressed.update(chunk)
            if decoder is None:
                consume(chunk)
            else:
                pending = chunk
                while pending:
                    check_deadline()
                    output = decoder.decompress(pending, _CHUNK_BYTES)
                    pending = decoder.unconsumed_tail
                    if output:
                        consume(output)
                    if decoder.unused_data:
                        raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID")
                while True:
                    check_deadline()
                    output = decoder.decompress(b"", _CHUNK_BYTES)
                    if not output:
                        break
                    consume(output)
        if decoder is not None:
            check_deadline()
            tail = decoder.flush(_CHUNK_BYTES)
            if tail:
                consume(tail)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID")
    except zlib.error as exc:
        raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID") from exc
    return DIGEST_PREFIX + compressed.hexdigest(), DIGEST_PREFIX + plain.hexdigest()


def _inspect_oci_archive(
    archive: Path, *, members: dict[str, tuple[str, int]], deadline: float,
    expected_config_digest: str, expected_config_size: int,
    expected_child_digest: str, expected_child_raw: bytes,
    expected_child_media_type: str, expected_layers: tuple[Mapping[str, object], ...],
    expected_provider_repository: str,
) -> DockerArchiveEvidence:
    if (
        not isinstance(expected_child_raw, bytes)
        or len(expected_child_raw) > MAX_METADATA_BYTES
        or not expected_child_digest.startswith(DIGEST_PREFIX)
        or len(expected_child_digest) != 71
        or any(character not in "0123456789abcdef" for character in expected_child_digest[7:])
        or _sha256(expected_child_raw) != expected_child_digest
        or expected_child_media_type not in {
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        }
        or not re.fullmatch(
            r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+",
            expected_provider_repository,
        )
    ):
        raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
    config_path = "blobs/sha256/" + expected_config_digest[7:]
    child_path = "blobs/sha256/" + expected_child_digest[7:]
    layer_paths = tuple("blobs/sha256/" + str(layer["digest"])[7:] for layer in expected_layers)
    if len(set(layer_paths)) != len(layer_paths) or len({config_path, child_path, *layer_paths}) != len(layer_paths) + 2:
        raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
    for layer in expected_layers:
        if (
            set(layer) != {"media_type", "digest", "size"}
            or layer["media_type"] not in _OCI_LAYER_MEDIA_TYPES
            or not isinstance(layer["digest"], str)
            or not str(layer["digest"]).startswith(DIGEST_PREFIX)
            or len(str(layer["digest"])) != 71
            or any(character not in "0123456789abcdef" for character in str(layer["digest"])[7:])
            or type(layer["size"]) is not int or not 1 <= int(layer["size"]) <= MAX_ARCHIVE_BYTES
        ):
            raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
    compatibility = "manifest.json" in members
    required_files = {"oci-layout", "index.json", child_path, config_path, *layer_paths}
    if compatibility:
        required_files.add("manifest.json")
    expected_names = required_files | {"blobs", "blobs/sha256"}
    if set(members) != expected_names:
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    if members.get("blobs") != ("directory", 0) or members.get("blobs/sha256") != ("directory", 0):
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    if any(members[name][0] != "file" for name in required_files):
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")
    metadata_names = {"oci-layout", "index.json", child_path, config_path}
    if compatibility:
        metadata_names.add("manifest.json")
    if sum(members[name][1] for name in metadata_names) > MAX_METADATA_BYTES:
        raise DockerArchiveError("ARCHIVE_METADATA_INVALID")

    captured: dict[str, bytes] = {}
    observed_compressed: dict[str, str] = {}
    observed_diff_ids: dict[str, str] = {}
    decompressed_total = [0]
    info = archive.lstat()
    zeros = 0
    with archive.open("rb", buffering=0) as stream:
        while stream.tell() < info.st_size:
            if time.monotonic() >= deadline:
                raise DockerArchiveError("ARCHIVE_TIMEOUT")
            header = _read_exact(stream, BLOCK_SIZE)
            if header == bytes(BLOCK_SIZE):
                zeros += 1
                if zeros >= 2:
                    _require_zero_remaining(stream, deadline=deadline)
                    break
                continue
            if zeros:
                raise DockerArchiveError("ARCHIVE_TRAILING_DATA")
            name, kind, size = _header(header)
            if kind == "file" and name in metadata_names:
                captured[name] = _read_exact(stream, size)
            elif kind == "file" and name in layer_paths:
                layer = expected_layers[layer_paths.index(name)]
                compressed, diff_id = _stream_oci_layer(
                    stream, size=size, media_type=str(layer["media_type"]),
                    deadline=deadline, decompressed_total=decompressed_total,
                )
                observed_compressed[name] = compressed
                observed_diff_ids[name] = diff_id
            else:
                _skip_exact(stream, size, deadline=deadline)
            padding = (-size) % BLOCK_SIZE
            if padding and any(_read_exact(stream, padding)):
                raise DockerArchiveError("ARCHIVE_PADDING_INVALID")

    if _json_object(captured["oci-layout"]) != {"imageLayoutVersion": "1.0.0"}:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    index = _json_object(captured["index.json"])
    if set(index) != {"schemaVersion", "mediaType", "manifests"} or index.get("schemaVersion") != 2 or index.get("mediaType") != OCI_LAYOUT_MEDIA_TYPE:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    descriptors = index.get("manifests")
    if not isinstance(descriptors, list) or len(descriptors) != 1 or not isinstance(descriptors[0], dict):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    descriptor = descriptors[0]
    if not set(descriptor) <= {"mediaType", "digest", "size", "platform", "annotations"} or not {"mediaType", "digest", "size"} <= set(descriptor):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    annotation_sha: str | None = None
    if "annotations" in descriptor:
        annotations = descriptor["annotations"]
        if annotations != {
            "containerd.io/distribution.source.docker.io": expected_provider_repository,
        }:
            raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
        annotation_sha = _sha256(expected_provider_repository.encode("utf-8"))
    if "platform" in descriptor and descriptor["platform"] != {"os": "linux", "architecture": "amd64"}:
        raise DockerArchiveError("ARCHIVE_PLATFORM_INVALID")
    if descriptor.get("mediaType") != expected_child_media_type or descriptor.get("digest") != expected_child_digest or descriptor.get("size") != len(expected_child_raw):
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    child_raw = captured[child_path]
    if child_raw != expected_child_raw or _sha256(child_raw) != expected_child_digest:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    child = _json_object(child_raw)
    if set(child) != {"schemaVersion", "mediaType", "config", "layers"} or child.get("schemaVersion") != 2 or child.get("mediaType") != expected_child_media_type:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    expected_config_descriptor = {
        "mediaType": child["config"].get("mediaType") if isinstance(child.get("config"), dict) else None,
        "digest": expected_config_digest, "size": expected_config_size,
    }
    if child.get("config") != expected_config_descriptor:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    expected_layer_descriptors = [
        {"mediaType": layer["media_type"], "digest": layer["digest"], "size": layer["size"]}
        for layer in expected_layers
    ]
    if child.get("layers") != expected_layer_descriptors:
        raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    config_raw = captured[config_path]
    if len(config_raw) != expected_config_size or _sha256(config_raw) != expected_config_digest:
        raise DockerArchiveError("ARCHIVE_CONFIG_IDENTITY_INVALID")
    config = _json_object(config_raw)
    if config.get("os") != "linux" or config.get("architecture") != "amd64" or "variant" in config:
        raise DockerArchiveError("ARCHIVE_PLATFORM_INVALID")
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, dict) or set(rootfs) != {"type", "diff_ids"} or rootfs.get("type") != "layers":
        raise DockerArchiveError("ARCHIVE_ROOTFS_INVALID")
    diff_ids = rootfs.get("diff_ids")
    ordered_observed = tuple(observed_diff_ids[path] for path in layer_paths)
    if not isinstance(diff_ids, list) or tuple(diff_ids) != ordered_observed:
        raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID")
    for path, layer in zip(layer_paths, expected_layers, strict=True):
        if members[path][1] != layer["size"] or observed_compressed[path] != layer["digest"]:
            raise DockerArchiveError("ARCHIVE_LAYER_IDENTITY_INVALID")
    compatibility_sha: str | None = None
    if compatibility:
        compatibility_raw = captured["manifest.json"]
        compatibility_sha = _sha256(compatibility_raw)
        value = _json_value(compatibility_raw)
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict) or set(value[0]) != {"Config", "RepoTags", "Layers"}:
            raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
        record = value[0]
        if record.get("Config") != config_path or record.get("Layers") != list(layer_paths) or record.get("RepoTags") not in (None, []):
            raise DockerArchiveError("ARCHIVE_DOCUMENT_INVALID")
    return DockerArchiveEvidence(
        config_raw=config_raw, config_sha256=expected_config_digest,
        config_size=len(config_raw), diff_ids=tuple(diff_ids),
        observed_layer_diff_ids=ordered_observed, layer_members=layer_paths,
        member_count=len(members), archive_size=info.st_size,
        archive_format="OCI_LAYOUT_COMPAT" if compatibility else "OCI_LAYOUT_PURE",
        compatibility_manifest_sha256=compatibility_sha,
        index_source_annotation_sha256=annotation_sha,
    )


def inspect_docker_archive(
    archive: Path, *, expected_config_digest: str, expected_config_size: int,
    expected_layer_count: int, timeout_seconds: int = ARCHIVE_TIMEOUT_SECONDS,
    expected_child_digest: str | None = None,
    expected_child_raw: bytes | None = None,
    expected_child_media_type: str | None = None,
    expected_layers: tuple[Mapping[str, object], ...] | None = None,
    expected_provider_repository: str | None = None,
) -> DockerArchiveEvidence:
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= ARCHIVE_TIMEOUT_SECONDS
        or not expected_config_digest.startswith(DIGEST_PREFIX)
        or len(expected_config_digest) != 71
        or any(character not in "0123456789abcdef" for character in expected_config_digest[7:])
        or type(expected_config_size) is not int
        or not 1 <= expected_config_size <= MAX_METADATA_BYTES
        or type(expected_layer_count) is not int
        or not 1 <= expected_layer_count <= MAX_LAYER_COUNT
    ):
        raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
    before = _file_identity(archive)
    if not 1024 <= before[2] <= MAX_ARCHIVE_BYTES or before[2] % BLOCK_SIZE:
        raise DockerArchiveError("ARCHIVE_LIMIT_EXCEEDED")
    deadline = time.monotonic() + timeout_seconds
    topology = _scan_topology(archive, deadline=deadline)
    is_oci = (
        "oci-layout" in topology or "index.json" in topology
        or any(name == "blobs" or name.startswith("blobs/") for name in topology)
    )
    if is_oci:
        if (
            expected_child_digest is None or expected_child_raw is None
            or expected_child_media_type is None or expected_layers is None
            or expected_provider_repository is None
            or len(expected_layers) != expected_layer_count
        ):
            raise DockerArchiveError("ARCHIVE_EXPECTATION_INVALID")
        evidence = _inspect_oci_archive(
            archive, members=topology, deadline=deadline,
            expected_config_digest=expected_config_digest,
            expected_config_size=expected_config_size,
            expected_child_digest=expected_child_digest,
            expected_child_raw=expected_child_raw,
            expected_child_media_type=expected_child_media_type,
            expected_layers=expected_layers,
            expected_provider_repository=expected_provider_repository,
        )
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DockerArchiveError("ARCHIVE_TIMEOUT")
        evidence = _inspect_legacy_archive(
            archive, expected_config_digest=expected_config_digest,
            expected_config_size=expected_config_size,
            expected_layer_count=expected_layer_count,
            timeout_seconds=max(1, min(ARCHIVE_TIMEOUT_SECONDS, int(remaining))),
        )
    if time.monotonic() >= deadline:
        raise DockerArchiveError("ARCHIVE_TIMEOUT")
    if _file_identity(archive) != before:
        raise DockerArchiveError("ARCHIVE_FILE_INVALID")
    return evidence


def save_docker_archive(command: DockerArchiveCommand, destination: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise DockerArchiveError("ARCHIVE_OUTPUT_INVALID") from exc
    created = True
    succeeded = False
    process = None
    overflow = threading.Event()
    timed_out = threading.Event()
    stderr_value = bytearray()
    try:
        group_kwargs: dict[str, object]
        if os.name == "nt":
            group_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group_kwargs = {"start_new_session": True}
        process = subprocess.Popen(
            list(command.argv), stdout=descriptor, stderr=subprocess.PIPE,
            env=dict(command.env), close_fds=True, **group_kwargs,
        )

        def terminate_tree() -> None:
            try:
                if os.name == "nt" and getattr(process, "pid", None):
                    completed = subprocess.run(
                        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=10, check=False,
                        env={"PATH": os.defpath},
                    )
                    if completed.returncode != 0:
                        process.kill()
                elif getattr(process, "pid", None):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except (OSError, subprocess.SubprocessError):
                try:
                    process.kill()
                except OSError:
                    pass

        def consume_stderr() -> None:
            while True:
                chunk = process.stderr.read(65536)
                if not chunk:
                    return
                remaining = command.maximum_stderr_bytes + 1 - len(stderr_value)
                stderr_value.extend(chunk[:remaining])
                if len(stderr_value) > command.maximum_stderr_bytes:
                    overflow.set()
                    terminate_tree()
                    return

        thread = threading.Thread(target=consume_stderr, daemon=True)
        thread.start()
        deadline = time.monotonic() + command.timeout_seconds
        while process.poll() is None:
            if overflow.is_set() or os.fstat(descriptor).st_size > command.maximum_archive_bytes:
                terminate_tree()
                overflow.set()
                break
            if time.monotonic() >= deadline:
                terminate_tree()
                timed_out.set()
                break
            time.sleep(0.1)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            terminate_tree()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_tree()
            raise DockerArchiveError("ARCHIVE_TIMEOUT") from exc
        thread.join(timeout=5)
        if thread.is_alive():
            terminate_tree()
            thread.join(timeout=5)
        os.fsync(descriptor)
        if (
            code or thread.is_alive() or overflow.is_set()
            or os.fstat(descriptor).st_size > command.maximum_archive_bytes
        ):
            if timed_out.is_set():
                raise DockerArchiveError("ARCHIVE_TIMEOUT")
            raise DockerArchiveError("ARCHIVE_COMMAND_FAILED")
        succeeded = True
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None:
            try:
                terminate_tree()
            except OSError:
                pass
        raise DockerArchiveError("ARCHIVE_COMMAND_FAILED") from exc
    finally:
        cleanup_error: OSError | None = None
        try:
            os.close(descriptor)
        except OSError as exc:
            cleanup_error = exc
            succeeded = False
        if created and not succeeded:
            try:
                destination.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise DockerArchiveError("ARCHIVE_OUTPUT_INVALID") from cleanup_error


__all__ = [
    "ARCHIVE_TIMEOUT_SECONDS", "DockerArchiveCommand", "DockerArchiveError",
    "DockerArchiveEvidence", "MAX_ARCHIVE_BYTES", "MAX_ARCHIVE_MEMBERS",
    "inspect_docker_archive", "save_docker_archive",
]
