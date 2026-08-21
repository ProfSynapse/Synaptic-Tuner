from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shared.experiment_tracking.root_identity import (
    MARKER_NAME,
    TrackingRootIdentityError,
    ensure_tracking_root_identity,
    require_tracking_root_identity,
)


def test_root_identity_is_canonical_stable_and_thread_safe(tmp_path: Path):
    root = tmp_path / "tracking"
    with ThreadPoolExecutor(max_workers=8) as executor:
        documents = list(executor.map(lambda _: ensure_tracking_root_identity(root), range(16)))
    assert len({document["root_id"] for document in documents}) == 1
    marker = root / MARKER_NAME
    assert marker.read_bytes().endswith(b"\n")
    assert json.loads(marker.read_text(encoding="utf-8")) == documents[0]
    assert require_tracking_root_identity(root, str(documents[0]["root_id"])) == documents[0]


def test_copied_root_and_replaced_marker_fail_closed(tmp_path: Path):
    source = tmp_path / "source"
    identity = ensure_tracking_root_identity(source)
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    with pytest.raises(TrackingRootIdentityError, match="copied|moved|replaced"):
        require_tracking_root_identity(copied, str(identity["root_id"]))

    marker = source / MARKER_NAME
    document = json.loads(marker.read_text(encoding="utf-8"))
    document["inode"] = str(int(document["inode"]) + 1)
    marker.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(TrackingRootIdentityError):
        require_tracking_root_identity(source, str(identity["root_id"]))


def test_root_identity_rejects_wrong_expected_id_without_mutation(tmp_path: Path):
    root = tmp_path / "tracking"
    identity = ensure_tracking_root_identity(root)
    before = (root / MARKER_NAME).read_bytes()
    with pytest.raises(TrackingRootIdentityError, match="does not match"):
        require_tracking_root_identity(root, "f" * 64)
    assert (root / MARKER_NAME).read_bytes() == before


def test_root_identity_rejects_symlink_root_when_supported(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(TrackingRootIdentityError):
        ensure_tracking_root_identity(link)
