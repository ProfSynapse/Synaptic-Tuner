from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import tuner.ingestion.local_selection_v1 as local_selection
from synaptic_tuner.api.v1._contract import canonical_bytes
from tuner.ingestion.local_selection_v1 import (
    AdmissionLimitsV1,
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
    ("member", "total", "error"),
    (
        (True, 1, TypeError),
        (1.0, 1, TypeError),
        (0, 1, ValueError),
        (1, 0, ValueError),
        (
            local_selection.HARD_MAX_MEMBER_BYTES + 1,
            local_selection.HARD_MAX_TOTAL_BYTES,
            ValueError,
        ),
        (1, local_selection.HARD_MAX_TOTAL_BYTES + 1, ValueError),
        (2, 1, ValueError),
    ),
)
def test_admission_limits_require_exact_bounded_integers(
    member: object, total: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        AdmissionLimitsV1(member, total)  # type: ignore[arg-type]


def test_explicit_member_budget_can_admit_above_legacy_default(tmp_path: Path) -> None:
    source = tmp_path / "large.md"
    payload = b"x" * (local_selection.DEFAULT_MAX_MEMBER_BYTES + 1)
    source.write_bytes(payload)
    root = (LocalSelectionRootV1("large.md", source),)
    policy = LocalDiscoveryPolicyV1(("*.md",), ())

    legacy = ProcessLocalSelectionRegistryV1()
    legacy_ref = legacy.authorize("project_01", root, policy)
    with pytest.raises(LocalSelectionErrorV1) as bounded:
        legacy.consume_snapshot(legacy_ref.source_ref)
    assert bounded.value.code is LocalSelectionCodeV1.LIMIT_EXCEEDED

    configured = ProcessLocalSelectionRegistryV1()
    configured_ref = configured.authorize(
        "project_01",
        root,
        policy,
        AdmissionLimitsV1(len(payload), len(payload)),
    )
    snapshot = configured.consume_snapshot(configured_ref.source_ref)
    assert snapshot.entries[0].content == payload


def test_configured_aggregate_budget_fails_with_closed_limit_code(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "a.md").write_bytes(b"1234")
    (selected / "b.md").write_bytes(b"5678")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("members", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
        AdmissionLimitsV1(4, 7),
    )

    with pytest.raises(LocalSelectionErrorV1) as bounded:
        registry.consume_snapshot(reference.source_ref)
    assert bounded.value.code is LocalSelectionCodeV1.LIMIT_EXCEEDED


def test_authority_identity_preserves_legacy_domain_and_binds_explicit_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "member.md"
    source.write_bytes(b"body")
    root = LocalSelectionRootV1("member.md", source)
    policy = LocalDiscoveryPolicyV1(("*.md",), ())
    secret = b"s" * 32
    source_ref = "local_" + "a" * 48
    monkeypatch.setattr(local_selection.secrets, "token_bytes", lambda size: secret)
    monkeypatch.setattr(
        local_selection.secrets, "token_hex", lambda size: "a" * (size * 2)
    )

    legacy = ProcessLocalSelectionRegistryV1().authorize("project_01", (root,), policy)
    prepared = local_selection._prepare_root(root)
    legacy_document = {
        "project_ref": "project_01",
        "source_ref": source_ref,
        "roots": [
            {
                "alias": prepared.alias,
                "path": str(prepared.path),
                "identity": list(prepared.identity),
            }
        ],
        "policy": {
            "include": ["*.md"],
            "exclude": [],
            "include_hidden": False,
        },
    }
    expected_legacy = hashlib.sha256(
        b"synaptic-process-local-selection-authority/v1\0"
        + secret
        + canonical_bytes(legacy_document)
    ).hexdigest()
    assert legacy.authority_digest == expected_legacy

    first = ProcessLocalSelectionRegistryV1().authorize(
        "project_01", (root,), policy, AdmissionLimitsV1(1024, 2048)
    )
    same = ProcessLocalSelectionRegistryV1().authorize(
        "project_01", (root,), policy, AdmissionLimitsV1(1024, 2048)
    )
    changed = ProcessLocalSelectionRegistryV1().authorize(
        "project_01", (root,), policy, AdmissionLimitsV1(2048, 4096)
    )
    assert first.authority_digest == same.authority_digest
    assert first.authority_digest != changed.authority_digest
    assert first.authority_digest != legacy.authority_digest


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

    def replace_root(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        nonlocal replaced
        if not replaced:
            displaced = tmp_path / "displaced"
            try:
                selected.rename(displaced)
                replacement.rename(selected)
            except OSError:
                pytest.skip("directory replacement is unavailable to the test process")
            replaced = True
        return original_read(path, identity, max_member_bytes)

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


@pytest.mark.parametrize("during_read", (False, True))
def test_directory_timestamp_change_does_not_change_selected_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_read: bool
) -> None:
    selected = tmp_path / "selected"
    nested = selected / "nested"
    nested.mkdir(parents=True)
    (nested / "note.md").write_bytes(b"body")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )

    def touch_directories() -> None:
        for directory in (selected, nested):
            original = directory.stat()
            os.utime(directory, ns=(original.st_atime_ns, original.st_mtime_ns + 10**9))

    if during_read:
        original_read = local_selection._stable_regular_read

        def read_and_touch(
            path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
        ) -> bytes:
            payload = original_read(path, identity, max_member_bytes)
            touch_directories()
            return payload

        monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_touch)
    else:
        touch_directories()

    snapshot = registry.consume_snapshot(reference.source_ref)
    assert tuple(item.content for item in snapshot.entries) == (b"body",)


@pytest.mark.parametrize("operation", ("create", "delete"))
@pytest.mark.parametrize("irrelevant_name", ("cache.tmp", "skip.md", "empty-directory"))
def test_irrelevant_namespace_churn_does_not_change_selected_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    irrelevant_name: str,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "note.md").write_bytes(b"body")
    # This directory has no admitted descendants. Its disappearance must not
    # trip a retained-directory check after the selected file has been read.
    unrelated = selected / "unrelated"
    unrelated.mkdir()
    changed_path = unrelated / irrelevant_name

    def create() -> None:
        if irrelevant_name == "empty-directory":
            changed_path.mkdir()
        else:
            changed_path.write_bytes(b"ignored")

    if operation == "delete":
        create()
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ("**/skip.md",)),
    )
    original_read = local_selection._stable_regular_read
    changed = False

    def read_and_change(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        nonlocal changed
        payload = original_read(path, identity, max_member_bytes)
        if changed:
            return payload
        changed = True
        if operation == "create":
            create()
        else:
            if changed_path.is_dir():
                changed_path.rmdir()
            else:
                changed_path.unlink()
            unrelated.rmdir()
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_change)
    snapshot = registry.consume_snapshot(reference.source_ref)
    assert tuple(item.logical_path for item in snapshot.entries) == ("notes/note.md",)


@pytest.mark.parametrize("operation", ("add", "remove", "rename", "modify"))
def test_matching_inventory_changes_after_read_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    source = selected / "note.md"
    source.write_bytes(b"body")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    original_read = local_selection._stable_regular_read

    def read_and_change(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        payload = original_read(path, identity, max_member_bytes)
        if operation == "add":
            (selected / "new.md").write_bytes(b"new")
        elif operation == "remove":
            source.unlink()
        elif operation == "rename":
            source.rename(selected / "renamed.md")
        else:
            source.write_bytes(b"changed body")
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_change)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_earlier_file_mutation_while_reading_later_file_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    earlier = selected / "a.md"
    earlier.write_bytes(b"first")
    (selected / "b.md").write_bytes(b"second")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    original_read = local_selection._stable_regular_read

    def read_and_change(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        payload = original_read(path, identity, max_member_bytes)
        if path.name == "b.md":
            earlier.write_bytes(b"first changed")
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_change)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_selected_file_mutation_during_descriptor_read_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"before")
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("note.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), ()),
    )
    original_read = os.read
    mutated = False

    def read_and_mutate(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, size)
        if not mutated:
            source.write_bytes(b"longer changed payload")
            mutated = True
        return payload

    monkeypatch.setattr(os, "read", read_and_mutate)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_earlier_file_same_length_rewrite_with_restored_mtime_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    earlier = selected / "a.md"
    earlier.write_bytes(b"first")
    (selected / "b.md").write_bytes(b"second")
    original_stat = earlier.stat()
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    original_read = local_selection._stable_regular_read
    mutated = False

    def read_and_rewrite(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        nonlocal mutated
        payload = original_read(path, identity, max_member_bytes)
        if path.name == "b.md" and not mutated:
            earlier.write_bytes(b"other")
            os.utime(earlier, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            assert local_selection._identity(
                earlier.stat()
            ) == local_selection._identity(original_stat)
            mutated = True
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_rewrite)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert mutated
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_descriptor_read_same_length_rewrite_with_restored_mtime_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "note.md"
    source.write_bytes(b"before")
    original_stat = source.stat()
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("note.md", source),),
        LocalDiscoveryPolicyV1(("*.md",), ()),
    )
    original_read = os.read
    mutated = False

    def read_and_rewrite(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, size)
        if not mutated:
            source.write_bytes(b"after!")
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            assert local_selection._identity(
                source.stat()
            ) == local_selection._identity(original_stat)
            mutated = True
        return payload

    monkeypatch.setattr(os, "read", read_and_rewrite)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert mutated
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_replaced_nested_ancestor_with_same_leaf_identity_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    nested = selected / "nested"
    nested.mkdir(parents=True)
    source = nested / "note.md"
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

    def read_and_replace(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        payload = original_read(path, identity, max_member_bytes)
        try:
            nested.rename(tmp_path / "displaced")
            replacement.rename(nested)
        except OSError:
            pytest.skip("directory replacement is unavailable to the test process")
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_replace)
    with pytest.raises(LocalSelectionErrorV1) as changed:
        registry.consume_snapshot(reference.source_ref)
    assert changed.value.code is LocalSelectionCodeV1.SOURCE_CHANGED


def test_second_discovery_pass_enforces_its_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "note.md").write_bytes(b"body")
    monkeypatch.setattr(local_selection, "MAX_DISCOVERY_ENTRIES", 2)
    registry = ProcessLocalSelectionRegistryV1()
    reference = registry.authorize(
        "project_01",
        (LocalSelectionRootV1("notes", selected),),
        LocalDiscoveryPolicyV1(("**/*.md",), ()),
    )
    original_read = local_selection._stable_regular_read

    def read_and_grow(
        path: Path, identity: tuple[int, int, int, int, int], max_member_bytes: int
    ) -> bytes:
        payload = original_read(path, identity, max_member_bytes)
        for name in ("one.tmp", "two.tmp"):
            (selected / name).write_bytes(b"ignored")
        return payload

    monkeypatch.setattr(local_selection, "_stable_regular_read", read_and_grow)
    with pytest.raises(LocalSelectionErrorV1) as bounded:
        registry.consume_snapshot(reference.source_ref)
    assert bounded.value.code is LocalSelectionCodeV1.LIMIT_EXCEEDED
