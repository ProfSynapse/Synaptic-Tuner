from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.cloud.hf_training_docker_archive import (
    ARCHIVE_TIMEOUT_SECONDS,
    BLOCK_SIZE,
    DockerArchiveCommand,
    DockerArchiveError,
    inspect_docker_archive,
    save_docker_archive,
)


DIFF_ID = "sha256:" + hashlib.sha256(b"layer").hexdigest()
_ABSENT = object()


def test_archive_phase_bound_is_frozen_at_900_seconds() -> None:
    assert ARCHIVE_TIMEOUT_SECONDS == 900


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _corrupt_gzip_crc(raw: bytes) -> bytes:
    value = bytearray(raw)
    value[-8] ^= 1
    return bytes(value)


def _config(**changes: object) -> bytes:
    value: dict[str, object] = {
        "architecture": "amd64",
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": [DIFF_ID]},
    }
    value.update(changes)
    return _raw(value)


def _archive(
    path: Path, *, config: bytes | None = None, config_name: str | None = None,
    repo_tags: object = None, layers: list[str] | None = None,
    extras: list[tuple[tarfile.TarInfo, bytes]] | None = None,
    archive_format: int = tarfile.USTAR_FORMAT,
) -> tuple[str, int]:
    config = config or _config()
    digest = _digest(config)
    config_name = config_name or digest[7:] + ".json"
    layers = layers or ["layer/layer.tar"]
    manifest = _raw([{"Config": config_name, "RepoTags": repo_tags, "Layers": layers}])
    members: list[tuple[tarfile.TarInfo, bytes]] = []
    for name, raw in (("manifest.json", manifest), (config_name, config)):
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        members.append((info, raw))
    for name in layers:
        info = tarfile.TarInfo(name)
        info.size = 5
        members.append((info, b"layer"))
    members.extend(extras or [])
    with tarfile.open(path, "w", format=archive_format) as output:
        for info, raw in members:
            output.addfile(info, io.BytesIO(raw) if info.isreg() else None)
    return digest, len(config)


def _inspect(path: Path, digest: str, size: int, layers: int = 1):
    return inspect_docker_archive(
        path, expected_config_digest=digest, expected_config_size=size,
        expected_layer_count=layers,
    )


def _oci_archive(
    path: Path, *, compatibility: bool = False, plain: bool = False,
    repo_tags: object = None, extra_name: str | None = None,
    stored_layer: bytes | None = None, descriptor_annotations: bool = False,
    layer_media_type: str | None = None,
    descriptor_platform: str = "exact",
    layer_count: int = 1,
    index_raw_override: bytes | None = None,
    compatibility_raw_override: bytes | None = None,
    descriptor_annotations_value: object = _ABSENT,
    descriptor_media_override: str | None = None,
    descriptor_digest_override: str | None = None,
    descriptor_size_override: int | None = None,
) -> dict[str, object]:
    plain_layers = [b"layer" if layer_count == 1 else f"layer-{index}".encode() for index in range(layer_count)]
    layer_blobs = [raw if plain else gzip.compress(raw, mtime=0) for raw in plain_layers]
    media_type = layer_media_type or (
        "application/vnd.oci.image.layer.v1.tar"
        if plain else "application/vnd.oci.image.layer.v1.tar+gzip"
    )
    layer_digests = [_digest(raw) for raw in layer_blobs]
    config = _config(rootfs={"type": "layers", "diff_ids": [_digest(raw) for raw in plain_layers]})
    config_digest = _digest(config)
    config_media = "application/vnd.docker.container.image.v1+json"
    child_media = "application/vnd.docker.distribution.manifest.v2+json"
    child = _raw({
        "schemaVersion": 2, "mediaType": child_media,
        "config": {"mediaType": config_media, "digest": config_digest, "size": len(config)},
        "layers": [
            {"mediaType": media_type, "digest": digest, "size": len(blob)}
            for digest, blob in zip(layer_digests, layer_blobs, strict=True)
        ],
    })
    child_digest = _digest(child)
    descriptor: dict[str, object] = {
        "mediaType": descriptor_media_override or child_media,
        "digest": descriptor_digest_override or child_digest,
        "size": len(child) if descriptor_size_override is None else descriptor_size_override,
    }
    if descriptor_platform == "exact":
        descriptor["platform"] = {"os": "linux", "architecture": "amd64"}
    elif descriptor_platform == "null":
        descriptor["platform"] = None
    if descriptor_annotations:
        descriptor["annotations"] = {"hostile": "value"}
    elif descriptor_annotations_value is not _ABSENT:
        descriptor["annotations"] = descriptor_annotations_value
    index = index_raw_override or _raw({
        "schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [descriptor],
    })
    config_path = "blobs/sha256/" + config_digest[7:]
    child_path = "blobs/sha256/" + child_digest[7:]
    layer_paths = ["blobs/sha256/" + digest[7:] for digest in layer_digests]
    members: list[tuple[str, bytes, bool]] = [
        ("blobs", b"", True), ("blobs/sha256", b"", True),
        ("oci-layout", _raw({"imageLayoutVersion": "1.0.0"}), False),
        ("index.json", index, False), (child_path, child, False),
        (config_path, config, False),
    ]
    for index_value, (layer_path, layer_blob) in enumerate(zip(layer_paths, layer_blobs, strict=True)):
        members.append((layer_path, stored_layer if stored_layer is not None and index_value == 0 else layer_blob, False))
    if compatibility:
        members.append(("manifest.json", compatibility_raw_override or _raw([{
            "Config": config_path, "RepoTags": repo_tags, "Layers": layer_paths,
        }]), False))
    if extra_name is not None:
        members.append((extra_name, b"extra", False))
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as output:
        for name, raw, directory in members:
            info = tarfile.TarInfo(name)
            if directory:
                info.type = tarfile.DIRTYPE
                info.size = 0
                output.addfile(info)
            else:
                info.size = len(raw)
                output.addfile(info, io.BytesIO(raw))
    return {
        "config_digest": config_digest, "config_size": len(config),
        "child_digest": child_digest, "child_raw": child,
        "child_media_type": child_media,
        "provider_repository": "unsloth/unsloth",
        "layers": tuple(
            {"media_type": media_type, "digest": digest, "size": len(blob)}
            for digest, blob in zip(layer_digests, layer_blobs, strict=True)
        ),
    }


def _inspect_oci(path: Path, expected: dict[str, object]):
    return inspect_docker_archive(
        path,
        expected_config_digest=str(expected["config_digest"]),
        expected_config_size=int(expected["config_size"]),
        expected_layer_count=len(expected["layers"]),
        expected_child_digest=str(expected["child_digest"]),
        expected_child_raw=expected["child_raw"],
        expected_child_media_type=str(expected["child_media_type"]),
        expected_layers=expected["layers"],
        expected_provider_repository=str(expected["provider_repository"]),
    )


@pytest.mark.parametrize("compatibility", [False, True])
@pytest.mark.parametrize("plain", [False, True])
def test_strict_oci_layout_authenticates_plain_or_gzip_layers(
    tmp_path: Path, compatibility: bool, plain: bool,
) -> None:
    path = tmp_path / f"oci-{compatibility}-{plain}.tar"
    expected = _oci_archive(path, compatibility=compatibility, plain=plain)
    evidence = _inspect_oci(path, expected)
    assert evidence.archive_format == (
        "OCI_LAYOUT_COMPAT" if compatibility else "OCI_LAYOUT_PURE"
    )
    assert evidence.diff_ids == (DIFF_ID,)
    assert evidence.observed_layer_diff_ids == (DIFF_ID,)
    assert (evidence.compatibility_manifest_sha256 is not None) is compatibility


def test_exact_24_layer_26_blob_oci_compatibility_layout(tmp_path: Path) -> None:
    path = tmp_path / "oci-24.tar"
    expected = _oci_archive(
        path, compatibility=True, layer_count=24,
        descriptor_annotations_value={
            "containerd.io/distribution.source.docker.io": "unsloth/unsloth",
        },
    )
    evidence = _inspect_oci(path, expected)
    assert evidence.archive_format == "OCI_LAYOUT_COMPAT"
    assert evidence.member_count == 31
    assert len(evidence.layer_members) == 24
    assert len(evidence.diff_ids) == 24
    assert evidence.index_source_annotation_sha256 == _digest(b"unsloth/unsloth")


@pytest.mark.parametrize("compatibility", [False, True])
def test_oci_layout_accepts_exact_optional_source_annotation(
    tmp_path: Path, compatibility: bool,
) -> None:
    path = tmp_path / f"annotated-{compatibility}.tar"
    expected = _oci_archive(
        path, compatibility=compatibility,
        descriptor_annotations_value={
            "containerd.io/distribution.source.docker.io": "unsloth/unsloth",
        },
    )
    assert _inspect_oci(path, expected).index_source_annotation_sha256 == _digest(b"unsloth/unsloth")


def test_any_blobs_member_forces_oci_mode_and_rejects_legacy_shape(tmp_path: Path) -> None:
    valid = tmp_path / "valid-legacy.tar"
    digest, size = _archive(valid)
    assert _inspect(valid, digest, size).archive_format == "LEGACY_DOCKER"

    directory = tarfile.TarInfo("blobs/sha256")
    directory.type = tarfile.DIRTYPE
    for suffix, extra in (
        ("directory", (directory, b"")),
        ("blob", (tarfile.TarInfo("blobs/sha256/" + "a" * 64), b"x")),
    ):
        extra[0].size = len(extra[1])
        path = tmp_path / f"legacy-with-{suffix}.tar"
        hostile_digest, hostile_size = _archive(path, extras=[extra])
        with pytest.raises(DockerArchiveError, match="EXPECTATION_INVALID|METADATA_INVALID"):
            _inspect(path, hostile_digest, hostile_size)


def test_oci_compatibility_manifest_accepts_empty_repo_tags(tmp_path: Path) -> None:
    path = tmp_path / "oci-empty-tags.tar"
    expected = _oci_archive(path, compatibility=True, repo_tags=[])
    assert _inspect_oci(path, expected).archive_format == "OCI_LAYOUT_COMPAT"


def test_oci_index_accepts_absent_platform(tmp_path: Path) -> None:
    path = tmp_path / "oci-no-platform.tar"
    expected = _oci_archive(path, descriptor_platform="absent")
    assert _inspect_oci(path, expected).archive_format == "OCI_LAYOUT_PURE"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"extra_name": "repositories"}, "METADATA_INVALID"),
        ({"extra_name": "legacy/layer.tar"}, "METADATA_INVALID"),
        ({"descriptor_annotations": True}, "DOCUMENT_INVALID"),
        ({"descriptor_platform": "null"}, "PLATFORM_INVALID"),
        ({"compatibility": True, "repo_tags": ["mutable:tag"]}, "DOCUMENT_INVALID"),
        ({"stored_layer": b"not-the-authenticated-layer"}, "LAYER_IDENTITY_INVALID"),
        ({"stored_layer": gzip.compress(b"layer", mtime=0) + b"trailing"}, "LAYER_IDENTITY_INVALID"),
        ({"stored_layer": gzip.compress(b"layer", mtime=0) * 2}, "LAYER_IDENTITY_INVALID"),
        ({"stored_layer": gzip.compress(b"layer", mtime=0)[:-4]}, "LAYER_IDENTITY_INVALID"),
        ({"stored_layer": _corrupt_gzip_crc(gzip.compress(b"layer", mtime=0))}, "LAYER_IDENTITY_INVALID"),
        ({"layer_media_type": "application/vnd.oci.image.layer.v1.tar+zstd"}, "EXPECTATION_INVALID"),
        ({"layer_media_type": "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip"}, "EXPECTATION_INVALID"),
    ],
)
def test_strict_oci_layout_rejects_hostile_topology_and_layers(
    tmp_path: Path, changes: dict[str, object], reason: str,
) -> None:
    path = tmp_path / (hashlib.sha256(repr(changes).encode()).hexdigest() + ".tar")
    expected = _oci_archive(path, **changes)
    with pytest.raises(DockerArchiveError, match=reason):
        _inspect_oci(path, expected)


def test_oci_layer_stream_enforces_aggregate_decompressed_limit(monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(DockerArchiveError, match="LIMIT_EXCEEDED"):
        archive_module._stream_oci_layer(
            io.BytesIO(b"ab"), size=2,
            media_type="application/vnd.oci.image.layer.v1.tar",
            deadline=archive_module.time.monotonic() + 10,
            decompressed_total=[0],
        )


def test_oci_layer_stream_checks_deadline_during_single_chunk_drain(monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    payload = gzip.compress(b"x" * (2 * 1024 * 1024), mtime=0)
    readings = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(archive_module.time, "monotonic", lambda: next(readings, 2.0))
    with pytest.raises(DockerArchiveError, match="ARCHIVE_TIMEOUT"):
        archive_module._stream_oci_layer(
            io.BytesIO(payload), size=len(payload),
            media_type="application/vnd.oci.image.layer.v1.tar+gzip",
            deadline=1.0, decompressed_total=[0],
        )


@pytest.mark.parametrize("target", ["index", "compatibility"])
@pytest.mark.parametrize(
    "hostile_raw",
    [
        b'{"schemaVersion":2,"schemaVersion":2}',
        b"[" * 65 + b"]" * 65,
        b"[NaN]",
    ],
)
def test_oci_metadata_json_rejects_duplicate_deep_and_nonfinite(
    tmp_path: Path, target: str, hostile_raw: bytes,
) -> None:
    path = tmp_path / f"hostile-{target}-{hashlib.sha256(hostile_raw).hexdigest()}.tar"
    kwargs = {"compatibility": target == "compatibility"}
    if target == "index":
        kwargs["index_raw_override"] = hostile_raw
    else:
        kwargs["compatibility_raw_override"] = hostile_raw
    expected = _oci_archive(path, **kwargs)
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect_oci(path, expected)


@pytest.mark.parametrize(
    "annotations",
    [
        None, {}, "unsloth/unsloth", [],
        {"containerd.io/distribution.source.docker.io": "wrong/repository"},
        {"containerd.io/distribution.source.docker.io": "docker.io/unsloth/unsloth"},
        {"containerd.io/distribution.source.docker.io": "unsloth/unsloth:latest"},
        {"containerd.io/distribution.source.docker.io": "unsloth/unsloth@sha256:" + "0" * 64},
        {"containerd.io/distribution.source.docker.io": " unsloth/unsloth"},
        {"containerd.io/distribution.source.docker.io": "unsloth/unsloth\0"},
        {"containerd.io/distribution.source.docker.io": "unsloth/unslöth"},
        {"Containerd.io/distribution.source.docker.io": "unsloth/unsloth"},
        {
            "containerd.io/distribution.source.docker.io": "unsloth/unsloth",
            "extra": "value",
        },
    ],
)
def test_oci_source_annotation_rejects_every_nonexact_variant(
    tmp_path: Path, annotations: object,
) -> None:
    path = tmp_path / (hashlib.sha256(repr(annotations).encode()).hexdigest() + ".tar")
    expected = _oci_archive(path, descriptor_annotations_value=annotations)
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect_oci(path, expected)


@pytest.mark.parametrize(
    "tamper",
    [
        {"descriptor_media_override": "application/vnd.oci.image.manifest.v1+json"},
        {"descriptor_digest_override": "sha256:" + "0" * 64},
        {"descriptor_size_override": 1},
    ],
)
def test_oci_annotation_never_authorizes_tampered_core_descriptor(
    tmp_path: Path, tamper: dict[str, object],
) -> None:
    path = tmp_path / (hashlib.sha256(repr(tamper).encode()).hexdigest() + ".tar")
    expected = _oci_archive(
        path,
        descriptor_annotations_value={
            "containerd.io/distribution.source.docker.io": "unsloth/unsloth",
        },
        **tamper,
    )
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect_oci(path, expected)


def test_strict_archive_authenticates_config_platform_rootfs_and_layers(tmp_path: Path) -> None:
    path = tmp_path / "image.tar"
    digest, size = _archive(path)
    evidence = _inspect(path, digest, size)
    assert evidence.config_sha256 == digest
    assert evidence.config_size == size
    assert evidence.diff_ids == (DIFF_ID,)
    assert evidence.observed_layer_diff_ids == (DIFF_ID,)
    assert evidence.layer_members == ("layer/layer.tar",)
    assert evidence.archive_size == path.stat().st_size
    assert evidence.archive_format == "LEGACY_DOCKER"
    assert evidence.compatibility_manifest_sha256 is None
    assert evidence.index_source_annotation_sha256 is None


def test_archive_rejects_file_identity_change_across_inspection(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    path = tmp_path / "identity.tar"
    digest, size = _archive(path)
    original = archive_module._file_identity(path)
    readings = iter((original, (*original[:3], original[3] + 1, original[4])))
    monkeypatch.setattr(archive_module, "_file_identity", lambda _path: next(readings))
    with pytest.raises(DockerArchiveError, match="FILE_INVALID"):
        _inspect(path, digest, size)


@pytest.mark.parametrize(
    "config",
    [
        _config(os="windows"),
        _config(architecture="arm64"),
        _config(variant="v8"),
        _config(rootfs={"type": "layers", "diff_ids": []}),
        _config(rootfs={"type": "layers", "diff_ids": ["not-a-digest"]}),
    ],
)
def test_archive_rejects_platform_or_rootfs_drift(tmp_path: Path, config: bytes) -> None:
    path = tmp_path / "image.tar"
    digest, size = _archive(path, config=config)
    with pytest.raises(DockerArchiveError):
        _inspect(path, digest, size)


def test_archive_rejects_config_digest_or_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "image.tar"
    digest, size = _archive(path)
    with pytest.raises(DockerArchiveError, match="METADATA_INVALID"):
        _inspect(path, "sha256:" + "0" * 64, size)
    with pytest.raises(DockerArchiveError, match="CONFIG_IDENTITY"):
        _inspect(path, digest, size + 1)


@pytest.mark.parametrize("repo_tags", [["mutable:tag"], "mutable:tag", {}])
def test_archive_rejects_repo_tags(tmp_path: Path, repo_tags: object) -> None:
    path = tmp_path / "image.tar"
    digest, size = _archive(path, repo_tags=repo_tags)
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect(path, digest, size)


def test_archive_rejects_links_and_traversal(tmp_path: Path) -> None:
    for name, kind in (("link", tarfile.SYMTYPE), ("../escape", tarfile.REGTYPE)):
        path = tmp_path / f"{kind!r}.tar"
        info = tarfile.TarInfo(name)
        info.type = kind
        info.size = 0
        digest, size = _archive(path, extras=[(info, b"")])
        with pytest.raises(DockerArchiveError):
            _inspect(path, digest, size)


def test_archive_rejects_duplicate_members_and_pax_headers(tmp_path: Path) -> None:
    duplicate = tarfile.TarInfo("manifest.json")
    duplicate.size = 2
    path = tmp_path / "duplicate.tar"
    digest, size = _archive(path, extras=[(duplicate, b"[]")])
    with pytest.raises(DockerArchiveError, match="DUPLICATE"):
        _inspect(path, digest, size)

    path = tmp_path / "pax.tar"
    pax = tarfile.TarInfo("pax-member")
    pax.pax_headers = {"comment": "extended metadata is forbidden"}
    digest, size = _archive(
        path, extras=[(pax, b"")], archive_format=tarfile.PAX_FORMAT,
    )
    with pytest.raises(DockerArchiveError):
        _inspect(path, digest, size)


def test_archive_rejects_truncation_bad_checksum_and_nonzero_trailing_data(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    digest, size = _archive(source)

    truncated = tmp_path / "truncated.tar"
    truncated.write_bytes(source.read_bytes()[: 7 * BLOCK_SIZE])
    with pytest.raises(DockerArchiveError):
        _inspect(truncated, digest, size)

    checksum = bytearray(source.read_bytes())
    checksum[0] ^= 1
    corrupt = tmp_path / "checksum.tar"
    corrupt.write_bytes(checksum)
    with pytest.raises(DockerArchiveError, match="CHECKSUM"):
        _inspect(corrupt, digest, size)

    trailing = bytearray(source.read_bytes())
    trailing[-1] = 1
    hostile = tmp_path / "trailing.tar"
    hostile.write_bytes(trailing)
    with pytest.raises(DockerArchiveError, match="TRAILING"):
        _inspect(hostile, digest, size)


def test_archive_rejects_layer_manifest_count_and_reference_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "image.tar"
    digest, size = _archive(path, layers=["layer/a.tar", "layer/b.tar"])
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect(path, digest, size, layers=1)

    missing = tmp_path / "missing.tar"
    config = _config()
    digest = _digest(config)
    config_name = digest[7:] + ".json"
    manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": ["absent/layer.tar"]}])
    with tarfile.open(missing, "w", format=tarfile.USTAR_FORMAT) as output:
        for name, raw in (("manifest.json", manifest), (config_name, config)):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
        _inspect(missing, digest, len(config))


@pytest.mark.parametrize("hostile", [b"altered", b"", b"same-count-wrong-content"])
def test_archive_rejects_layer_bytes_not_bound_to_diff_ids(tmp_path: Path, hostile: bytes) -> None:
    path = tmp_path / "hostile.tar"
    config = _config()
    digest = _digest(config)
    config_name = digest[7:] + ".json"
    manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": ["layer/layer.tar"]}])
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as output:
        for name, raw in (("manifest.json", manifest), (config_name, config), ("layer/layer.tar", hostile)):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    with pytest.raises(DockerArchiveError, match="LAYER_IDENTITY|DOCUMENT_INVALID"):
        _inspect(path, digest, len(config))


def test_archive_rejects_metadata_or_directory_as_layer(tmp_path: Path) -> None:
    for layer_name in ("manifest.json", "layer/"):
        path = tmp_path / ("metadata.tar" if layer_name == "manifest.json" else "directory.tar")
        config = _config()
        digest = _digest(config)
        config_name = digest[7:] + ".json"
        manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": [layer_name]}])
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as output:
            for name, raw in (("manifest.json", manifest), (config_name, config)):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                output.addfile(info, io.BytesIO(raw))
            if layer_name.endswith("/"):
                info = tarfile.TarInfo(layer_name)
                info.type = tarfile.DIRTYPE
                output.addfile(info)
        with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
            _inspect(path, digest, len(config))


def test_archive_rejects_nonfinite_deep_and_huge_numeric_json(tmp_path: Path) -> None:
    path = tmp_path / "base.tar"
    digest, size = _archive(path)
    for raw in (b"[NaN]", b"[" * 65 + b"]" * 65, b'[{"Config":1e999999,"RepoTags":null,"Layers":[]}]'):
        hostile = tmp_path / (hashlib.sha256(raw).hexdigest() + ".tar")
        config = _config()
        config_name = digest[7:] + ".json"
        with tarfile.open(hostile, "w", format=tarfile.USTAR_FORMAT) as output:
            for name, value in (("manifest.json", raw), (config_name, config), ("layer/layer.tar", b"layer")):
                info = tarfile.TarInfo(name)
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))
        with pytest.raises(DockerArchiveError, match="DOCUMENT_INVALID"):
            _inspect(hostile, digest, size)


def test_archive_enforces_four_mib_aggregate_metadata_budget(tmp_path: Path) -> None:
    path = tmp_path / "aggregate.tar"
    config = _config() + (b" " * (2 * 1024 * 1024))
    digest = _digest(config)
    config_name = digest[7:] + ".json"
    manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": ["layer/layer.tar"]}])
    manifest += b" " * (2 * 1024 * 1024)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as output:
        for name, raw in (("manifest.json", manifest), (config_name, config), ("layer/layer.tar", b"layer")):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    with pytest.raises(DockerArchiveError, match="METADATA_INVALID"):
        _inspect(path, digest, len(config))


def test_archive_rejects_reordered_layer_members(tmp_path: Path) -> None:
    path = tmp_path / "reordered.tar"
    config = _config(rootfs={"type": "layers", "diff_ids": [_digest(b"a"), _digest(b"b")]})
    digest = _digest(config)
    config_name = digest[7:] + ".json"
    manifest = _raw([{"Config": config_name, "RepoTags": None, "Layers": ["b/layer.tar", "a/layer.tar"]}])
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as output:
        for name, raw in (("manifest.json", manifest), (config_name, config), ("a/layer.tar", b"a"), ("b/layer.tar", b"b")):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    with pytest.raises(DockerArchiveError, match="LAYER_IDENTITY_INVALID"):
        _inspect(path, digest, len(config), layers=2)


def test_archive_inspection_has_independent_deadline(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    path = tmp_path / "deadline.tar"
    digest, size = _archive(path)
    readings = iter((0.0, 2.0))
    monkeypatch.setattr(archive_module.time, "monotonic", lambda: next(readings, 2.0))
    with pytest.raises(DockerArchiveError, match="ARCHIVE_TIMEOUT"):
        inspect_docker_archive(
            path, expected_config_digest=digest, expected_config_size=size,
            expected_layer_count=1, timeout_seconds=1,
        )


def test_zero_tail_read_enforces_archive_deadline(monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    monkeypatch.setattr(archive_module.time, "monotonic", lambda: 2.0)
    with pytest.raises(DockerArchiveError, match="ARCHIVE_TIMEOUT"):
        archive_module._require_zero_remaining(io.BytesIO(bytes(BLOCK_SIZE)), deadline=1.0)


def test_save_runner_uses_exclusive_file_and_enforces_byte_bound(tmp_path: Path, monkeypatch) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    class FakeProcess:
        def __init__(self, stdout, payload: bytes) -> None:
            os.write(stdout, payload)
            self.stderr = io.BytesIO()
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0
        def kill(self):
            return None

    payload = b"bounded"
    popen_kwargs = {}
    def popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return FakeProcess(kwargs["stdout"], payload)
    monkeypatch.setattr(archive_module.subprocess, "Popen", popen)
    command = DockerArchiveCommand(("docker", "image", "save", "sha256:" + "a" * 64), {}, maximum_archive_bytes=len(payload))
    destination = tmp_path / "archive.tar"
    save_docker_archive(command, destination)
    assert destination.read_bytes() == payload
    assert popen_kwargs.get("creationflags") == archive_module.subprocess.CREATE_NEW_PROCESS_GROUP
    with pytest.raises(DockerArchiveError, match="OUTPUT_INVALID"):
        save_docker_archive(command, destination)
    assert destination.read_bytes() == payload

    oversized = tmp_path / "oversized.tar"
    too_small = DockerArchiveCommand(command.argv, {}, maximum_archive_bytes=len(payload) - 1)
    with pytest.raises(DockerArchiveError, match="COMMAND_FAILED"):
        save_docker_archive(too_small, oversized)
    assert not oversized.exists()


def test_archive_timeout_uses_bounded_wait_and_falls_back_when_taskkill_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    class FakeProcess:
        pid = 4321
        stderr = io.BytesIO()
        waits: list[float | None] = []
        killed = 0
        def poll(self):
            return None
        def wait(self, timeout=None):
            self.waits.append(timeout)
            return 1
        def kill(self):
            self.killed += 1

    process = FakeProcess()
    monkeypatch.setattr(archive_module.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        archive_module.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1),
    )
    readings = iter((0.0, 2.0))
    monkeypatch.setattr(archive_module.time, "monotonic", lambda: next(readings, 2.0))
    with pytest.raises(DockerArchiveError, match="ARCHIVE_TIMEOUT"):
        save_docker_archive(
            DockerArchiveCommand(("docker",), {}, timeout_seconds=1),
            tmp_path / "timed-out.tar",
        )
    assert process.waits == [10]
    assert process.killed >= 1


def test_archive_live_stderr_reader_triggers_second_tree_stop_and_bounded_join(
    tmp_path: Path, monkeypatch,
) -> None:
    from tuner.cloud import hf_training_docker_archive as archive_module

    class FakeProcess:
        pid = 4321
        stderr = io.BytesIO()
        killed = 0
        def poll(self):
            return 0
        def wait(self, timeout=None):
            assert timeout == 10
            return 0
        def kill(self):
            self.killed += 1

    class ReaderThread:
        joins = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def join(self, timeout=None):
            assert timeout == 5
            self.joins += 1
        def is_alive(self):
            return self.joins < 2

    process = FakeProcess()
    taskkills = []
    monkeypatch.setattr(archive_module.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(archive_module.threading, "Thread", ReaderThread)
    monkeypatch.setattr(
        archive_module.subprocess, "run",
        lambda argv, **kwargs: taskkills.append(tuple(argv)) or SimpleNamespace(returncode=1),
    )
    save_docker_archive(DockerArchiveCommand(("docker",), {}), tmp_path / "reader.tar")
    assert len(taskkills) >= 1 and process.killed >= 1
