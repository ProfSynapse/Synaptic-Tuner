from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import tuner.ingestion.local_selection_v1 as local_selection
from tuner.ingestion.local_selection_v1 import (
    AdmissionReportV1,
    ImmutableLocalSnapshotV1,
    LocalDiscoveryPolicyV1,
    LocalSelectionCodeV1,
    LocalSelectionErrorV1,
    LocalSelectionRootV1,
    ProcessLocalSelectionRegistryV1,
    SnapshotEntryV1,
    glob_matches_v1,
)


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    (
        ("*.md", "note.md", True),
        ("*.md", ".note.md", True),
        ("**/*.md", "note.md", True),
        ("**/*.md", "books/one/note.md", True),
        ("books/?/note.md", "books/a/note.md", True),
        ("books/?/note.md", "books/ab/note.md", False),
        ("a/**/b", "a/b", True),
        ("a/**/b", "a/x/y/b", True),
        ("a/*/b", "a/x/y/b", False),
    ),
)
def test_glob_matches_exact_declarative_dp(
    pattern: str, path: str, expected: bool
) -> None:
    assert glob_matches_v1(pattern, path) is expected


@pytest.mark.parametrize(
    "pattern", ("../*.md", "x/[a].md", "x/**foo.md", "x\\*.md", "x/\x00.md")
)
def test_glob_reuses_facade_validation(pattern: str) -> None:
    with pytest.raises(ValueError):
        glob_matches_v1(pattern, "x/a.md")


def test_registry_builds_deterministic_immutable_one_use_snapshot(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    (selected / "nested").mkdir(parents=True)
    (selected / ".private.md").write_bytes(b"private")
    (selected / "z.txt").write_bytes(b"excluded")
    (selected / "nested" / "b.md").write_bytes(b"bravo")
    (selected / "a.md").write_bytes(b"alpha")

    registry = ProcessLocalSelectionRegistryV1()
    authorized = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ("**/skip*.md",)),
    )
    snapshot = registry.consume_snapshot(authorized.source_ref)

    assert tuple(item.logical_path for item in snapshot.entries) == (
        "notes/a.md",
        "notes/nested/b.md",
    )
    assert tuple(item.content for item in snapshot.entries) == (b"alpha", b"bravo")
    assert snapshot.report.discovered_file_count == 3
    assert snapshot.report.excluded_file_count == 1
    assert snapshot.report.total_bytes == 10
    assert len(snapshot.manifest_digest) == 64

    (selected / "a.md").write_bytes(b"changed")
    assert snapshot.entries[0].content == b"alpha"
    with pytest.raises(LocalSelectionErrorV1) as reused:
        registry.consume_snapshot(authorized.source_ref)
    assert reused.value.code is LocalSelectionCodeV1.AUTHORITY_UNAVAILABLE
    with pytest.raises(AttributeError):
        snapshot.entries[0].content = b"mutated"  # type: ignore[misc]


def test_selected_file_alias_is_its_complete_logical_path(tmp_path: Path) -> None:
    source = tmp_path / "arbitrary-host-name"
    source.write_bytes(b"body")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("renamed.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), ()),
    )
    snapshot = registry.consume_snapshot(reference.source_ref)
    assert snapshot.entries[0].logical_path == "renamed.md"


def test_authorized_root_identity_change_fails_closed_and_consumes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"before")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("note.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), ()),
    )
    source.write_bytes(b"after")
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED
    with pytest.raises(LocalSelectionErrorV1) as consumed:
        registry.consume_snapshot(reference.source_ref)
    assert consumed.value.code is LocalSelectionCodeV1.AUTHORITY_UNAVAILABLE


def test_nfc_logical_path_and_alias_collisions_fail_closed(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "\N{LATIN SMALL LETTER E WITH ACUTE}.md").write_bytes(b"one")
    (selected / "e\N{COMBINING ACUTE ACCENT}.md").write_bytes(b"two")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), (), include_hidden=True),
    )
    with pytest.raises(LocalSelectionErrorV1) as collision:
        registry.consume_snapshot(reference.source_ref)
    assert collision.value.code is LocalSelectionCodeV1.SOURCE_UNSAFE


def test_symlink_is_never_followed(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    target = tmp_path / "outside.md"
    target.write_bytes(b"outside")
    link = selected / "link.md"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable to the test process")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    with pytest.raises(LocalSelectionErrorV1) as unsafe:
        registry.consume_snapshot(reference.source_ref)
    assert unsafe.value.code is LocalSelectionCodeV1.SOURCE_UNSAFE


def test_root_and_policy_inputs_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"body")
    root = LocalSelectionRootV1("note.md", source)
    registry = ProcessLocalSelectionRegistryV1()
    with pytest.raises(ValueError):
        registry.authorize(
            "project_01", (root,) * 33, LocalDiscoveryPolicyV1(("*.md",), ())
        )
    with pytest.raises(ValueError):
        LocalDiscoveryPolicyV1((), ())
    with pytest.raises(ValueError):
        LocalSelectionRootV1("e\N{COMBINING ACUTE ACCENT}", source)


def test_replaced_root_is_rejected_after_leaf_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    source = selected / "note.md"
    source.write_bytes(b"same inode")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    try:
        os.link(source, replacement / "note.md")
    except OSError:
        pytest.skip("hard links are unavailable to the test process")

    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    original_read = local_selection._stable_regular_read
    replaced = False

    def replace_root(path: Path, identity: tuple[int, int, int, int, int]) -> bytes:
        nonlocal replaced
        if not replaced:
            displaced = tmp_path / "displaced"
            try:
                selected.rename(displaced)
                replacement.rename(selected)
            except OSError:
                pytest.skip("directory replacement is unavailable to the test process")
            replaced = True
        return original_read(path, identity)

    monkeypatch.setattr(local_selection, "_stable_regular_read", replace_root)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_discovery_budget_is_charged_during_directory_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (selected / name).write_bytes(b"x")
    monkeypatch.setattr(local_selection, "MAX_DISCOVERY_ENTRIES", 2)
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    with pytest.raises(LocalSelectionErrorV1) as bounded:
        registry.consume_snapshot(reference.source_ref)
    assert bounded.value.code is LocalSelectionCodeV1.LIMIT_EXCEEDED


def test_directory_only_depth_is_bounded_before_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    nested = selected
    for name in ("a", "b", "c"):
        nested = nested / name
        nested.mkdir()
    monkeypatch.setattr(local_selection, "MAX_LOGICAL_PATH_SEGMENTS", 3)
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    with pytest.raises(LocalSelectionErrorV1) as bounded:
        registry.consume_snapshot(reference.source_ref)
    assert bounded.value.code is LocalSelectionCodeV1.LIMIT_EXCEEDED


def test_snapshot_rejects_non_exact_report_before_dereference() -> None:
    entry = SnapshotEntryV1("note.md", b"body", 4, hashlib.sha256(b"body").hexdigest())
    with pytest.raises(TypeError, match="exact AdmissionReportV1"):
        ImmutableLocalSnapshotV1(
            "project_01", "source_01", (entry,), object(), "0" * 64  # type: ignore[arg-type]
        )


def test_valid_report_remains_exact_and_immutable() -> None:
    report = AdmissionReportV1(1, 1, 1, 0, 4)
    assert report.total_bytes == 4


def test_snapshot_defensively_rebuilds_caller_owned_values() -> None:
    entry = SnapshotEntryV1("note.md", b"body", 4, hashlib.sha256(b"body").hexdigest())
    report = AdmissionReportV1(1, 1, 1, 0, 4)
    snapshot = ImmutableLocalSnapshotV1(
        "project_01",
        "source_01",
        (entry,),
        report,
        local_selection._manifest_digest((entry,)),
    )
    object.__setattr__(entry, "content", b"changed")
    object.__setattr__(report, "total_bytes", 99)
    assert snapshot.entries[0].content == b"body"
    assert snapshot.report.total_bytes == 4
