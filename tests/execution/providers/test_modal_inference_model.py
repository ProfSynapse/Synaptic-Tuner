from __future__ import annotations

import os
from pathlib import Path

import pytest

import tuner.execution.providers.modal.inference_model as subject
from tuner.execution.providers.modal.inference_model import ModalPinnedModelPreparer

MODEL = "owner/model"
REVISION = "a" * 40


def _preparer(tmp_path: Path) -> ModalPinnedModelPreparer:
    roots = [tmp_path / name for name in ("persistent", "destination", "scratch")]
    for root in roots:
        root.mkdir()
    return ModalPinnedModelPreparer(
        persistent_root=roots[0],
        destination_root=roots[1],
        scratch_root=roots[2],
        token="private-token",
    )


def test_prepares_once_and_captures_exact_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    destination = tmp_path / "destination" / "model" / "snapshot"
    destination.mkdir(parents=True)
    calls: list[tuple[str, dict[str, object]]] = []

    class Snapshot:
        model_ref = MODEL
        revision = REVISION
        root = tmp_path / "destination"
        root_identity = (root.stat().st_dev, root.stat().st_ino)
        snapshot = "model/snapshot"

    sentinel = Snapshot()

    def prepare(**kwargs: object) -> Path:
        calls.append(("prepare", kwargs))
        return destination

    def capture(**kwargs: object):
        calls.append(("capture", kwargs))
        return sentinel

    monkeypatch.setattr(subject, "prepare_model_snapshot", prepare)
    monkeypatch.setattr(subject, "capture_pinned_model_snapshot", capture)
    monkeypatch.setattr(subject, "VerifiedPinnedModelSnapshot", type(sentinel))
    assert preparer.prepare(model_ref=MODEL, revision=REVISION) is sentinel
    assert [name for name, _ in calls] == ["prepare", "capture"]
    assert calls[0][1]["token"] == "private-token"
    assert calls[1][1] == {
        "model_ref": MODEL,
        "revision": REVISION,
        "root": tmp_path / "destination",
        "snapshot": "model/snapshot",
    }
    assert "private-token" not in repr(preparer)


def test_real_capture_returns_exact_small_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    destination = tmp_path / "destination" / "model" / "snapshot"
    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes(b"{}")
    monkeypatch.setattr(subject, "prepare_model_snapshot", lambda **kwargs: destination)
    value = preparer.prepare(model_ref=MODEL, revision=REVISION)
    assert value.model_ref == MODEL
    assert value.revision == REVISION
    assert value.snapshot == "model/snapshot"
    assert tuple((item.relative_path, item.size_bytes) for item in value.files) == (
        ("config.json", 2),
    )
    value.validate()


@pytest.mark.parametrize("revision", ["", "A" * 40, "a" * 39, "a" * 64])
def test_invalid_revision_denied_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    preparer = _preparer(tmp_path)
    monkeypatch.setattr(
        subject,
        "prepare_model_snapshot",
        lambda **kwargs: pytest.fail("loader called"),
    )
    with pytest.raises((TypeError, ValueError)):
        preparer.prepare(model_ref=MODEL, revision=revision)


def test_loader_cannot_return_outside_or_redirected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(subject, "prepare_model_snapshot", lambda **kwargs: outside)
    with pytest.raises(ValueError, match="outside"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)

    redirected = tmp_path / "destination" / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(subject, "prepare_model_snapshot", lambda **kwargs: redirected)
    with pytest.raises(ValueError, match="inventory"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)


def test_constructor_rejects_overlapping_roots_and_secret_repr(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    with pytest.raises(ValueError, match="distinct"):
        ModalPinnedModelPreparer(
            persistent_root=root,
            destination_root=root,
            scratch_root=root,
            token="never-render-this",
        )


def test_platform_and_root_identity_fail_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    loader_calls = 0

    def loader(**kwargs: object) -> Path:
        nonlocal loader_calls
        loader_calls += 1
        scratch = tmp_path / "scratch"
        scratch.rmdir()
        scratch.mkdir()
        destination = tmp_path / "destination" / "model" / "snapshot"
        destination.mkdir(parents=True)
        return destination

    monkeypatch.setattr(subject, "prepare_model_snapshot", loader)
    monkeypatch.setattr(
        subject,
        "capture_pinned_model_snapshot",
        lambda **kwargs: pytest.fail("capture called"),
    )
    with pytest.raises(ValueError, match="identity"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)
    assert loader_calls == 1


def test_root_replacement_during_capture_is_rejected_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    destination = tmp_path / "destination" / "model" / "snapshot"
    destination.mkdir(parents=True)

    class Snapshot:
        model_ref = MODEL
        revision = REVISION
        root = tmp_path / "destination"
        root_identity = (root.stat().st_dev, root.stat().st_ino)
        snapshot = "model/snapshot"

    monkeypatch.setattr(subject, "prepare_model_snapshot", lambda **kwargs: destination)

    def capture(**kwargs: object) -> Snapshot:
        scratch = tmp_path / "scratch"
        scratch.rmdir()
        scratch.mkdir()
        return Snapshot()

    monkeypatch.setattr(subject, "capture_pinned_model_snapshot", capture)
    monkeypatch.setattr(subject, "VerifiedPinnedModelSnapshot", Snapshot)
    with pytest.raises(ValueError, match="identity"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)


@pytest.mark.parametrize("failure", ["prepare", "capture", None])
def test_retained_root_descriptors_close_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    preparer = _preparer(tmp_path)
    destination = tmp_path / "destination" / "model" / "snapshot"
    destination.mkdir(parents=True)
    before = len(os.listdir("/proc/self/fd"))

    class Snapshot:
        model_ref = MODEL
        revision = REVISION
        root = tmp_path / "destination"
        root_identity = (root.stat().st_dev, root.stat().st_ino)
        snapshot = "model/snapshot"

    def prepare(**kwargs: object) -> Path:
        if failure == "prepare":
            raise RuntimeError("closed prepare failure")
        return destination

    def capture(**kwargs: object) -> Snapshot:
        if failure == "capture":
            raise RuntimeError("closed capture failure")
        return Snapshot()

    monkeypatch.setattr(subject, "prepare_model_snapshot", prepare)
    monkeypatch.setattr(subject, "capture_pinned_model_snapshot", capture)
    monkeypatch.setattr(subject, "VerifiedPinnedModelSnapshot", Snapshot)
    if failure is None:
        assert (
            preparer.prepare(model_ref=MODEL, revision=REVISION).snapshot
            == "model/snapshot"
        )
    else:
        with pytest.raises((RuntimeError, ValueError)):
            preparer.prepare(model_ref=MODEL, revision=REVISION)
    assert len(os.listdir("/proc/self/fd")) == before


def test_physical_root_alias_is_rejected_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent = tmp_path / "persistent"
    destination = tmp_path / "destination"
    persistent.mkdir()
    destination.mkdir()
    alias = tmp_path / "scratch"
    alias.symlink_to(persistent, target_is_directory=True)
    preparer = ModalPinnedModelPreparer(
        persistent_root=persistent,
        destination_root=destination,
        scratch_root=alias,
        token=None,
    )
    monkeypatch.setattr(
        subject, "prepare_model_snapshot", lambda **kwargs: pytest.fail("loader called")
    )
    with pytest.raises((OSError, ValueError)):
        preparer.prepare(model_ref=MODEL, revision=REVISION)


def test_transient_destination_substitution_cannot_supply_captured_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    root = tmp_path / "destination"
    destination = root / "model" / "snapshot"
    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes(b"{}")
    replacement = tmp_path / "replacement"
    other_snapshot = replacement / "model" / "snapshot"
    other_snapshot.mkdir(parents=True)
    (other_snapshot / "config.json").write_bytes(b'{"other":true}')
    held = tmp_path / "original"
    capture = subject.capture_pinned_model_snapshot
    monkeypatch.setattr(subject, "prepare_model_snapshot", lambda **kwargs: destination)
    before = len(os.listdir("/proc/self/fd"))

    def capture_from_replacement(**kwargs):
        root.rename(held)
        replacement.rename(root)
        try:
            return capture(**kwargs)
        finally:
            root.rename(replacement)
            held.rename(root)

    monkeypatch.setattr(
        subject, "capture_pinned_model_snapshot", capture_from_replacement
    )
    with pytest.raises(
        ValueError,
        match="snapshot changed model identity|snapshot capture changed model identity",
    ):
        preparer.prepare(model_ref=MODEL, revision=REVISION)
    assert len(os.listdir("/proc/self/fd")) == before


def test_unsupported_secure_platform_denied_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    monkeypatch.setattr(
        subject, "_platform", lambda: (_ for _ in ()).throw(RuntimeError("unsupported"))
    )
    monkeypatch.setattr(
        subject,
        "prepare_model_snapshot",
        lambda **kwargs: pytest.fail("loader called"),
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)
