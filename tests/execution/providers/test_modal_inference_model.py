from __future__ import annotations

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
        persistent_root=roots[0], destination_root=roots[1],
        scratch_root=roots[2], token="private-token",
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
        "model_ref": MODEL, "revision": REVISION,
        "root": tmp_path / "destination", "snapshot": "model/snapshot",
    }
    assert "private-token" not in repr(preparer)


def test_real_capture_returns_exact_small_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    destination = tmp_path / "destination" / "model" / "snapshot"
    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes(b"{}")
    monkeypatch.setattr(
        subject, "prepare_model_snapshot", lambda **kwargs: destination
    )
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
        subject, "prepare_model_snapshot",
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
            persistent_root=root, destination_root=root,
            scratch_root=root, token="never-render-this",
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
        subject, "capture_pinned_model_snapshot",
        lambda **kwargs: pytest.fail("capture called"),
    )
    with pytest.raises(ValueError, match="identity"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)
    assert loader_calls == 1


def test_unsupported_secure_platform_denied_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _preparer(tmp_path)
    monkeypatch.setattr(
        subject, "_platform", lambda: (_ for _ in ()).throw(RuntimeError("unsupported"))
    )
    monkeypatch.setattr(
        subject, "prepare_model_snapshot",
        lambda **kwargs: pytest.fail("loader called"),
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        preparer.prepare(model_ref=MODEL, revision=REVISION)
