"""Immutable prompt JSONL preparation through existing generic host ports."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from synaptic_tuner.api.v1.training_input import TrainingInputV1, TrainingMethodV1, _text
from synaptic_tuner.api.v1.training_sources import LocalTrainingInputPathV1, PreparedTrainingInputIdentity
from tuner.dataset_prep import publication as storage
from tuner.training.contracts import RetainedTrainingInputStreamLease
from .grpo_compilation import GRPO_PREPARED_FORMAT, admit_grpo_snapshot, grpo_prepared_semantic_digest
from .input_preparation import NormalizedTrainingInputV1, PreparedInputVerificationV1, _stable_directory_identity
from .recipes import canonical_json_bytes

_MANIFEST_SCHEMA = "synaptic-grpo-prepared-publication/v1"
_MAX_BYTES = 65536


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate GRPO input field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite GRPO input")


def _rows(raw, prompt_column):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError("GRPO input byte bound exceeded")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("GRPO input must be UTF-8") from None
    lines = raw.splitlines()
    if text.startswith("\ufeff") or not 1 <= len(lines) <= 128:
        raise ValueError("GRPO input BOM or row count is invalid")
    from Trainers.grpo.runtime_v1 import _finite_tree
    for line in lines:
        if not line.strip():
            raise ValueError("blank GRPO input row")
        try:
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
        except RecursionError:
            raise ValueError("GRPO row exceeds its structure bound") from None
        _finite_tree(row)
        if type(row) is not dict:
            raise ValueError("GRPO input row must be an object")
        prompt = row.get(prompt_column)
        if type(prompt) not in (str, list) or not prompt:
            raise ValueError("GRPO prompt must be nonempty text or messages")
        if type(prompt) is list:
            for message in prompt:
                if (type(message) is not dict or set(message) != {"role", "content"}
                        or type(message["role"]) is not str or not message["role"]
                        or type(message["content"]) is not str):
                    raise ValueError("GRPO prompt message is invalid")
        if prompt_column != "prompt" and "prompt" in row:
            raise ValueError("GRPO input collides with derived prompt column")
    return len(lines)


def _manifest(raw, prompt_column):
    count = _rows(raw, prompt_column)
    basis = {"schema_version": _MANIFEST_SCHEMA, "format": GRPO_PREPARED_FORMAT,
             "prompt_column": prompt_column, "content_digest": _sha(raw),
             "size_bytes": len(raw), "row_count": count}
    digest = grpo_prepared_semantic_digest(content_digest=basis["content_digest"],
        size_bytes=basis["size_bytes"], row_count=count, prompt_column=prompt_column)
    return {**basis, "semantic_digest": digest}


def _private(path):
    if sys.platform != "linux":
        raise ValueError("GRPO publication requires Linux retained descriptors")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.getuid() or path.resolve(strict=True) != path.absolute()):
        raise ValueError("GRPO publication root must be private")


def _read_relative(root, name, maximum):
    check = root.verify_binding if type(root) is storage._OwnedStageAuthority else root.verify
    check()
    _private(root.path)
    assert root.descriptor is not None
    declared = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
    if not stat.S_ISREG(declared.st_mode) or declared.st_nlink != 1 or declared.st_size > maximum:
        raise ValueError("GRPO input member must be a bounded regular file")
    # Existing retained-member helper lacks O_NONBLOCK for a racing special leaf.
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=root.descriptor)
    try:
        opened = os.fstat(descriptor)
        def observation(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                    info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
        expected = observation(declared)
        if (observation(opened) != expected or opened.st_uid != os.getuid()
                or stat.S_IMODE(opened.st_mode) & 0o077):
            raise ValueError("GRPO member identity or permissions differ")
        chunks, total = [], 0
        while total <= maximum:
            chunk = os.read(descriptor, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if (total != opened.st_size or total > maximum
                or observation(os.fstat(descriptor)) != expected
                or observation(os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)) != expected):
            raise ValueError("GRPO input member changed")
        check()
        _private(root.path)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _inventory_names(descriptor):
    names = set()
    with os.scandir(descriptor) as entries:
        for entry in entries:
            if entry.name not in {"manifest.json", "dataset.jsonl"} or entry.name in names:
                raise ValueError("GRPO publication inventory differs")
            names.add(entry.name)
    return names


def snapshot_grpo_publication(path):
    _private(path)
    root = storage._OutputRootAuthority(path)
    try:
        assert root.descriptor is not None
        if _inventory_names(root.descriptor) != {"manifest.json", "dataset.jsonl"}:
            raise ValueError("GRPO publication inventory differs")
        manifest_raw = _read_relative(root, "manifest.json", 4096)
        manifest = json.loads(manifest_raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
        if type(manifest) is not dict or set(manifest) != {
            "schema_version", "format", "prompt_column", "content_digest", "size_bytes", "row_count", "semantic_digest"}:
            raise ValueError("GRPO publication manifest fields differ")
        column = _text(manifest["prompt_column"], "prompt_column", maximum_bytes=128)
        raw = _read_relative(root, "dataset.jsonl", _MAX_BYTES)
        expected = _manifest(raw, column)
        if manifest != expected or manifest_raw != canonical_json_bytes(expected):
            raise ValueError("GRPO publication commitments differ")
        if path.name != "dataset-" + expected["semantic_digest"]:
            raise ValueError("GRPO publication name differs")
        if _inventory_names(root.descriptor) != {"manifest.json", "dataset.jsonl"}:
            raise ValueError("GRPO publication inventory changed")
        root.verify()
        _private(path)
        return manifest, raw
    finally:
        if not root.close():
            raise ValueError("GRPO publication root close failed")


def _identity(manifest):
    digest = manifest["semantic_digest"]
    return PreparedTrainingInputIdentity(ref="prepared://sha256/" + digest, revision=digest,
        content_digest=manifest["content_digest"], size_bytes=manifest["size_bytes"],
        format=GRPO_PREPARED_FORMAT)


@dataclass(frozen=True, slots=True)
class GRPONormalizerConfigV1:
    member: str
    prompt_column: str

    def __post_init__(self):
        _text(self.member, "member", maximum_bytes=128)
        _text(self.prompt_column, "prompt_column", maximum_bytes=128)
        if self.member in {".", ".."} or any(c in self.member for c in "/\\:"):
            raise ValueError("GRPO source member must be one leaf")

    @property
    def normalizer_ref(self):
        return "synaptic.grpo-prompts/v1"


class GRPOTrainingInputNormalizerV1:
    def __init__(self, *, allowed_source_roots):
        if type(allowed_source_roots) is not tuple or not allowed_source_roots:
            raise TypeError("explicit GRPO source roots required")
        for root in allowed_source_roots:
            if type(root) is not type(Path()) or not root.is_absolute():
                raise TypeError("GRPO source roots must be absolute Paths")
            _private(root)
        self._roots = tuple((root, _stable_directory_identity(root, "GRPO source root"))
                            for root in allowed_source_roots)

    def prepare(self, source, *, config, output_root):
        if type(source) is not LocalTrainingInputPathV1 or type(config) is not GRPONormalizerConfigV1:
            raise TypeError("exact GRPO local source and normalizer config required")
        source_path = Path(os.path.abspath(source.path))
        if not any(source_path == root and _stable_directory_identity(root, "GRPO source root") == identity
                   for root, identity in self._roots):
            raise ValueError("GRPO source root not admitted")
        _private(source_path)
        root = storage._OutputRootAuthority(source_path)
        try:
            raw = _read_relative(root, config.member, _MAX_BYTES)
        finally:
            if not root.close():
                raise ValueError("GRPO source root close failed")
        manifest = _manifest(raw, config.prompt_column)
        _private(output_root)
        authority = storage._OutputRootAuthority(output_root)
        coordination = None
        stage = None
        destination = output_root / ("dataset-" + manifest["semantic_digest"])
        try:
            coordination = storage._OutputRootCoordination(authority)
            try:
                destination.lstat()
            except FileNotFoundError:
                pass
            else:
                existing, existing_raw = snapshot_grpo_publication(destination)
                if existing != manifest or existing_raw != raw:
                    raise ValueError("GRPO publication collision")
                return NormalizedTrainingInputV1(destination, GRPO_PREPARED_FORMAT)
            stage = storage._OwnedStageAuthority.create(authority, destination.name)
            storage._write_stage_member(stage, "dataset.jsonl", raw)
            storage._write_stage_member(stage, "manifest.json", canonical_json_bytes(manifest))
            assert stage.descriptor is not None
            for name in ("dataset.jsonl", "manifest.json"):
                os.chmod(name, 0o400, dir_fd=stage.descriptor, follow_symlinks=False)
            storage._sync_directory_authority(stage.descriptor, stage.path)
            stage.verify_binding()
            if (_read_relative(stage, "dataset.jsonl", _MAX_BYTES) != raw
                    or _read_relative(stage, "manifest.json", 4096) != canonical_json_bytes(manifest)):
                raise ValueError("GRPO staged publication changed")
            storage._commit_owned_stage(stage, destination)
            storage._sync_directory_authority(authority.descriptor, output_root)
            expected, verified = snapshot_grpo_publication(destination)
            if expected != manifest or verified != raw:
                raise ValueError("GRPO final publication changed")
            return NormalizedTrainingInputV1(destination, GRPO_PREPARED_FORMAT)
        finally:
            # Uncertain partial stages/publications are retained, never adopted or deleted.
            failed = False
            if stage is not None:
                failed |= not stage.close()
            if coordination is not None:
                failed |= not coordination.close()
            failed |= not authority.close()
            if failed:
                raise ValueError("GRPO publication authority close failed")


class _GRPOPreparedSource:
    __slots__ = ("_path", "_identity")

    def __init__(self, path, identity):
        self._path, self._identity = path, identity

    @property
    def identity(self):
        return self._identity

    def open_lease(self):
        manifest, raw = snapshot_grpo_publication(self._path)
        if _identity(manifest) != self._identity:
            raise ValueError("GRPO prepared input changed")
        return RetainedTrainingInputStreamLease(self._identity, BytesIO(raw))

    def __reduce_ex__(self, protocol):
        raise TypeError("host-only GRPO source is not serializable")


class GRPOPreparedInputFormatVerifierV1:
    @property
    def format(self):
        return GRPO_PREPARED_FORMAT

    def verify(self, path):
        manifest, raw = snapshot_grpo_publication(path)
        identity = _identity(manifest)
        return PreparedInputVerificationV1(path, identity.revision, identity.content_digest,
                                           identity.size_bytes, raw)

    def validate_training_input(self, path, identity, canonical_training_json):
        manifest, raw = snapshot_grpo_publication(path)
        value = TrainingInputV1.from_json(canonical_training_json)
        if value.method is not TrainingMethodV1.GRPO:
            raise ValueError("GRPO verifier requires GRPO input")
        if _identity(manifest) != identity or manifest["prompt_column"] != value.hyperparameters.prompt_column:
            raise ValueError("GRPO prepared input or prompt column differs")
        admit_grpo_snapshot(value, identity, raw)

    def source(self, path, identity):
        if type(identity) is not PreparedTrainingInputIdentity or identity.format != self.format:
            raise ValueError("GRPO prepared identity differs")
        return _GRPOPreparedSource(path, identity)
