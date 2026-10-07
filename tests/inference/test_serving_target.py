from __future__ import annotations

from pathlib import Path
import os

import pytest

from synaptic_tuner.api.v1.runs_facade import RunsAPI
from tuner.inference.serving_target import (
    ServingTarget,
    VerifiedPinnedModelSnapshot,
    capture_pinned_model_snapshot,
    prepare_serving_target,
)
from tuner.inference import serving_target
from tests.inference.test_retrieved_model import _fixture
from tuner.inference.retrieved_model import materialize_verified_sft_model


def _retrieved(tmp_path: Path):
    retrieved_root = tmp_path / "retrieved"
    retrieved_root.mkdir()
    run, _, operations = _fixture(retrieved_root)
    return materialize_verified_sft_model(RunsAPI(operations), run, retrieved_root)


def _snapshot(tmp_path: Path, revision: str = "c" * 40):
    root = tmp_path / "base"
    path = root / "model" / "snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(b'{"model_type":"fixture"}')
    (path / "weights.bin").write_bytes(b"weights")
    return capture_pinned_model_snapshot(
        model_ref="example/model",
        revision=revision,
        root=root,
        snapshot=f"model/snapshots/{revision}",
    )


def test_lora_prepares_exact_pinned_base_once(tmp_path: Path) -> None:
    retrieved = _retrieved(tmp_path)
    snapshot = _snapshot(tmp_path)

    class Preparer:
        def __init__(self) -> None:
            self.calls = []

        def prepare(self, *, model_ref: str, revision: str):
            self.calls.append((model_ref, revision))
            return snapshot

    preparer = Preparer()
    target = prepare_serving_target(retrieved, preparer)
    assert preparer.calls == [("example/model", "c" * 40)]
    assert target.base_model_path == snapshot.path
    assert target.model_path == retrieved.model_path
    assert target.tokenizer_path == retrieved.tokenizer_path


@pytest.mark.parametrize("revision", ("d" * 40, "d" * 64))
def test_lora_rejects_base_revision_or_identity_substitution(
    tmp_path: Path,
    revision: str,
) -> None:
    retrieved = _retrieved(tmp_path)
    snapshot = _snapshot(tmp_path, revision)

    class Preparer:
        def prepare(self, *, model_ref: str, revision: str):
            return snapshot

    with pytest.raises(ValueError, match="identity differs"):
        prepare_serving_target(retrieved, Preparer())


def test_lora_rejects_untyped_preparer_result(tmp_path: Path) -> None:
    retrieved = _retrieved(tmp_path)

    class Preparer:
        def prepare(self, *, model_ref: str, revision: str):
            return object()

    with pytest.raises(TypeError, match="snapshot type"):
        prepare_serving_target(retrieved, Preparer())


def test_snapshot_fresh_validation_detects_changed_member(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    snapshot.path.joinpath("config.json").write_bytes(b"changed")
    with pytest.raises(ValueError):
        snapshot.validate()


def test_snapshot_rejects_structural_substitution(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    changed = VerifiedPinnedModelSnapshot(
        snapshot.model_ref,
        snapshot.revision,
        snapshot.root,
        snapshot.snapshot,
        snapshot.root_identity,
        snapshot.snapshot_identity,
        tuple(reversed(snapshot.files)),
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        changed.validate()


def test_full_target_never_calls_preparer(tmp_path: Path, monkeypatch) -> None:
    retrieved = _retrieved(tmp_path)
    object.__setattr__(retrieved, "model_kind", "full")
    monkeypatch.setattr(type(retrieved), "validate", lambda self: None)

    class Preparer:
        def prepare(self, *, model_ref: str, revision: str):
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="does not accept"):
        prepare_serving_target(retrieved, Preparer())
    target = prepare_serving_target(retrieved)
    assert type(target) is ServingTarget
    assert target.base_model_path is None


@pytest.mark.parametrize("mode", ("member", "aggregate"))
def test_snapshot_quota_fails_before_hash(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    root = tmp_path / "base"
    snapshot = root / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "a").write_bytes(b"aa")
    if mode == "aggregate":
        (snapshot / "b").write_bytes(b"bb")
        monkeypatch.setattr(serving_target, "MAX_PINNED_BYTES", 3)
    else:
        monkeypatch.setattr(serving_target, "MAX_PINNED_BYTES", 1)
    calls = []
    monkeypatch.setattr(serving_target, "_digest", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="bound"):
        capture_pinned_model_snapshot(
            model_ref="example/model",
            revision="c" * 40,
            root=root,
            snapshot="snapshot",
        )
    assert calls == []


@pytest.mark.parametrize("field", ("model_ref", "revision", "root", "snapshot"))
def test_malformed_snapshot_metadata_causes_zero_filesystem_io(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    values = dict(
        model_ref="example/model",
        revision="c" * 40,
        root=tmp_path,
        snapshot="snapshot",
    )
    values[field] = {
        "model_ref": "",
        "revision": "latest",
        "root": "bad",
        "snapshot": "../bad",
    }[field]
    calls = []
    monkeypatch.setattr(serving_target, "_open_root", lambda path: calls.append(path))
    with pytest.raises((TypeError, ValueError)):
        capture_pinned_model_snapshot(**values)
    assert calls == []


def test_unsupported_platform_causes_zero_filesystem_io(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        serving_target,
        "_platform",
        lambda: (_ for _ in ()).throw(RuntimeError("unsupported")),
    )
    monkeypatch.setattr(serving_target, "_open_root", lambda path: calls.append(path))
    with pytest.raises(RuntimeError, match="unsupported"):
        capture_pinned_model_snapshot(
            model_ref="example/model",
            revision="c" * 40,
            root=tmp_path,
            snapshot="snapshot",
        )
    assert calls == []


def test_snapshot_validate_unsupported_platform_causes_zero_filesystem_io(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = _snapshot(tmp_path)
    calls = []
    monkeypatch.setattr(
        serving_target,
        "_platform",
        lambda: (_ for _ in ()).throw(RuntimeError("unsupported")),
    )
    monkeypatch.setattr(serving_target, "_open_root", lambda path: calls.append(path))

    with pytest.raises(RuntimeError, match="unsupported"):
        snapshot.validate()

    assert calls == []


def test_snapshot_inventory_is_globally_sorted_across_shared_prefixes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "base"
    snapshot_path = root / "snapshot"
    nested = snapshot_path / "a"
    nested.mkdir(parents=True)
    (nested / "z").write_bytes(b"nested")
    (snapshot_path / "a.txt").write_bytes(b"peer")

    snapshot = capture_pinned_model_snapshot(
        model_ref="example/model",
        revision="c" * 40,
        root=root,
        snapshot="snapshot",
    )

    assert tuple(item.relative_path for item in snapshot.files) == ("a.txt", "a/z")
    snapshot.validate()


@pytest.mark.parametrize("attack", ("symlink", "fifo", "hardlink"))
def test_snapshot_rejects_redirected_special_or_linked_members(
    tmp_path: Path,
    attack: str,
) -> None:
    root = tmp_path / "base"
    snapshot = root / "snapshot"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    member = snapshot / "member"
    if attack == "symlink":
        member.symlink_to(outside)
    elif attack == "fifo":
        os.mkfifo(member)
    else:
        os.link(outside, member)
    with pytest.raises(ValueError):
        capture_pinned_model_snapshot(
            model_ref="example/model",
            revision="c" * 40,
            root=root,
            snapshot="snapshot",
        )


def test_snapshot_detects_root_snapshot_and_inventory_mutations(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    original_root = snapshot.root
    moved = original_root.with_name("base-moved")
    original_root.rename(moved)
    original_root.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        snapshot.validate()
