"""Bounded-memory verification of mounted model artifacts."""

import hashlib
import os

import pytest

from tuner.execution.providers.modal import mounted_io


@pytest.mark.parametrize("payload", [b"", b"artifact", b"x" * (2 * 1024 * 1024 + 7)])
def test_hash_regular_streams_without_whole_file_read(tmp_path, monkeypatch, payload):
    path = tmp_path / "model"
    path.write_bytes(payload)
    reads = []
    original = os.fdopen

    class Reader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, maximum):
            reads.append(maximum)
            assert 0 < maximum <= 1024 * 1024
            return self.stream.read(maximum)

    monkeypatch.setattr(mounted_io.os, "fdopen", lambda *a, **kw: Reader(original(*a, **kw)))
    assert mounted_io.hash_regular(tmp_path, path, len(payload)) == (
        len(payload), hashlib.sha256(payload).hexdigest(),
    )
    assert reads


@pytest.mark.parametrize("maximum", [-1, True, 1.0])
def test_hash_regular_rejects_invalid_bounds_before_io(tmp_path, maximum):
    with pytest.raises(ValueError, match="exact integer"):
        mounted_io.hash_regular(tmp_path / "missing", tmp_path / "missing/file", maximum)


def test_hash_regular_rejects_oversized_and_symlink_files(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"content")
    with pytest.raises(ValueError):
        mounted_io.hash_regular(tmp_path, path, 2)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        mounted_io.hash_regular(tmp_path, alias, 100)


def test_hash_regular_rejects_content_growth_during_read(tmp_path, monkeypatch):
    path = tmp_path / "model"
    path.write_bytes(b"content")
    original = os.fdopen

    class Reader:
        def __init__(self, stream): self.stream, self.changed = stream, False
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, maximum):
            content = self.stream.read(maximum)
            if not self.changed:
                self.changed = True
                with path.open("ab") as writer: writer.write(b"changed")
            return content

    monkeypatch.setattr(mounted_io.os, "fdopen", lambda *a, **kw: Reader(original(*a, **kw)))
    with pytest.raises(ValueError, match="during hash"):
        mounted_io.hash_regular(tmp_path, path, 100)


@pytest.mark.skipif(not mounted_io._SECURE_DIRFD, reason="requires retained Linux directory descriptors")
def test_hash_regular_retains_parent_when_pathname_is_redirected(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "model").write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model").write_bytes(b"redirected")
    original = mounted_io._open_leaf

    def redirected(*args, **kwargs):
        parent.rename(tmp_path / "retained")
        parent.symlink_to(outside, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(mounted_io, "_open_leaf", redirected)
    assert mounted_io.hash_regular(tmp_path, parent / "model", 100) == (
        8, hashlib.sha256(b"original").hexdigest(),
    )


def test_claim_directory_is_exclusive_and_inventory_is_exact(tmp_path):
    directory = tmp_path / "operation" / "output"
    mounted_io.claim_directory(tmp_path, directory)
    assert mounted_io.list_regular_sizes(tmp_path, directory, 5) == ()
    with pytest.raises(FileExistsError):
        mounted_io.claim_directory(tmp_path, directory)
    (directory / "model").write_bytes(b"model")
    (directory / "tokenizer").write_bytes(b"tokenizer")
    assert mounted_io.list_regular_sizes(tmp_path, directory, 2) == (("model", 5), ("tokenizer", 9))
    with pytest.raises(ValueError, match="entry bound"):
        mounted_io.list_regular_sizes(tmp_path, directory, 1)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_inventory_rejects_nonregular_members(tmp_path, kind):
    directory = tmp_path / "output"
    mounted_io.claim_directory(tmp_path, directory)
    if kind == "directory":
        (directory / "nested").mkdir()
    else:
        target = tmp_path / "target"
        target.write_bytes(b"untrusted")
        (directory / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="nonregular"):
        mounted_io.list_regular_sizes(tmp_path, directory, 5)


def test_inventory_refuses_symlink_ancestor_and_invalid_bound(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        mounted_io.list_regular_sizes(tmp_path, alias, 5)
    with pytest.raises(ValueError, match="exact integer"):
        mounted_io.list_regular_sizes(tmp_path, target, True)


@pytest.mark.skipif(not mounted_io._SECURE_DIRFD, reason="requires retained Linux directory descriptors")
def test_inventory_retains_directory_across_pathname_substitution(tmp_path, monkeypatch):
    directory = tmp_path / "output"
    mounted_io.claim_directory(tmp_path, directory)
    (directory / "model").write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "other").write_bytes(b"redirected")
    original = os.scandir

    def redirected(handle):
        directory.rename(tmp_path / "retained")
        directory.symlink_to(outside, target_is_directory=True)
        return original(handle)

    monkeypatch.setattr(mounted_io.os, "scandir", redirected)
    assert mounted_io.list_regular_sizes(tmp_path, directory, 5) == (("model", 8),)


@pytest.mark.parametrize("content", ["not-bytes", bytearray(b"bytes")])
def test_invalid_exclusive_write_does_not_create_any_path(tmp_path, content):
    parent = tmp_path / "missing"
    with pytest.raises(TypeError, match="exact bytes"):
        mounted_io.write_exclusive(tmp_path, parent / "output", content)
    assert not parent.exists()


@pytest.mark.parametrize("maximum", [-1, True, 1.0])
def test_invalid_copy_bound_does_not_create_destination(tmp_path, maximum):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    destination = tmp_path / "missing" / "output"
    with pytest.raises(ValueError, match="exact integer"):
        mounted_io.copy_regular(tmp_path, source, tmp_path, destination, maximum=maximum)
    assert not destination.parent.exists()


@pytest.mark.parametrize("maximum", [-1, True, 1.0])
def test_invalid_read_bound_is_rejected_before_path_lookup(tmp_path, maximum):
    with pytest.raises(ValueError, match="exact integer"):
        mounted_io.read_regular(tmp_path / "missing", tmp_path / "missing/file", maximum)
