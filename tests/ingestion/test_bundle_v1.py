from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tuner.ingestion import bundle_v1
from tuner.ingestion.bundle_v1 import (
    BUNDLE_SCHEMA_VERSION,
    ITEM_SCHEMA_VERSION,
    BundleCollisionError,
    BundleDurabilityError,
    BundlePublicationUncertainV1,
    BundlePublicationUncertaintyPhaseV1,
    BundleSemanticIdentityV1,
    BundleValidationError,
    NormalizedItemInputV1,
    verify_normalized_bundle_v1,
    retry_bundle_root_durability_v1,
    write_normalized_bundle_v1,
)


REF = {"name": "note", "version": "1", "digest": "a" * 64}


def _structures() -> dict[str, object]:
    return {
        "structures": [{"ref": dict(REF), "profile": "markdown"}],
        "bindings": [{"binding_id": "all", "structure_ref": dict(REF)}],
    }


def _item(path: str, body: str = "body", *, digest: str = "b" * 64) -> NormalizedItemInputV1:
    return NormalizedItemInputV1(
        logical_path=path,
        source_sha256=digest,
        source_size_bytes=len(body.encode("utf-8")),
        structure_ref=dict(REF),
        fields={"body": body, "metadata": {"score": 1.5, "tags": ["x"]}},
    )


def _write(tmp_path: Path, items=None):
    return write_normalized_bundle_v1(
        tmp_path / "bundles",
        _structures(),
        items or (_item("b.md", "second"), _item("a.md", "first")),
    )


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _make_writable(path: Path) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def test_writer_is_deterministic_content_addressed_and_canonical(tmp_path: Path) -> None:
    first = write_normalized_bundle_v1(
        tmp_path / "one", _structures(), (_item("b.md"), _item("a.md"))
    )
    second = write_normalized_bundle_v1(
        tmp_path / "two", _structures(), (_item("a.md"), _item("b.md"))
    )

    assert first.semantic_identity == second.semantic_identity
    assert first.semantic_identity.bundle_id == "bundle-" + first.semantic_identity.bundle_digest
    assert first.path.name == first.semantic_identity.bundle_id
    assert {entry.name for entry in first.path.iterdir()} == {"manifest.json", "items.jsonl"}
    assert (first.path / "manifest.json").read_bytes() == (second.path / "manifest.json").read_bytes()
    assert (first.path / "items.jsonl").read_bytes() == (second.path / "items.jsonl").read_bytes()

    manifest_raw = (first.path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical(manifest)
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert "plan_fingerprint" not in manifest
    assert manifest["structure_set"] == _structures()
    rows = [json.loads(line) for line in (first.path / "items.jsonl").read_bytes().splitlines()]
    assert [row["logical_path"] for row in rows] == ["a.md", "b.md"]
    assert all(row["schema_version"] == ITEM_SCHEMA_VERSION for row in rows)
    assert all(line == _canonical(row) for line, row in zip((first.path / "items.jsonl").read_bytes().splitlines(), rows))
    assert all(row["item_id"].startswith("item-") and len(row["item_id"]) == 69 for row in rows)


def test_input_is_semantic_and_copies_mutable_mappings() -> None:
    ref = dict(REF)
    fields = {"body": "original", "nested": {"value": 1}}
    item = NormalizedItemInputV1("note.md", "b" * 64, 8, ref, fields)
    ref["name"] = "changed"
    fields["body"] = "changed"
    fields["nested"]["value"] = 2

    assert item.structure_ref["name"] == "note"
    assert item.fields["body"] == "original"
    assert item.fields["nested"]["value"] == 1
    with pytest.raises(TypeError):
        item.fields["body"] = "nope"


def test_verified_identical_destination_is_reused(tmp_path: Path) -> None:
    first = _write(tmp_path)
    second = _write(tmp_path, (_item("a.md", "first"), _item("b.md", "second")))

    assert second == first
    assert verify_normalized_bundle_v1(first.path) == first
    assert not any(entry.name.startswith(f".{first.semantic_identity.bundle_id}.") for entry in first.path.parent.iterdir())


def test_invalid_existing_destination_is_a_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _write(tmp_path)
    _make_writable(result.path / "items.jsonl")
    (result.path / "items.jsonl").write_bytes(b"{}\n")
    monkeypatch.setattr(
        bundle_v1.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("stage was created")),
    )

    with pytest.raises(BundleCollisionError, match="destination collision"):
        _write(tmp_path)


@pytest.mark.parametrize("path", ["/absolute.md", "C:/drive.md", "../escape.md", "a\\b.md", "a//b.md"])
def test_logical_paths_are_safe_relative_posix_paths(path: str) -> None:
    with pytest.raises(BundleValidationError, match="logical_path"):
        _item(path)


def test_duplicate_paths_and_undeclared_structure_refs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(BundleValidationError, match="logical paths must be unique"):
        _write(tmp_path, (_item("same.md", "a"), _item("same.md", "b", digest="c" * 64)))

    other = NormalizedItemInputV1("other.md", "b" * 64, 1, {"name": "other", "version": "1", "digest": "c" * 64}, {"body": "x"})
    with pytest.raises(BundleValidationError, match="not declared"):
        _write(tmp_path, (other,))


def test_items_output_bound_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_v1, "MAX_ITEMS_BYTES", 256)
    with pytest.raises(BundleValidationError, match="character|byte limit"):
        _write(tmp_path, (_item("large.md", "x" * 512),))


def test_verifier_rejects_noncanonical_manifest(tmp_path: Path) -> None:
    result = _write(tmp_path)
    path = result.path / "manifest.json"
    _make_writable(path)
    path.write_text(json.dumps(json.loads(path.read_bytes()), indent=2), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="canonical"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_recomputes_item_id_and_hashes(tmp_path: Path) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    path = result.path / "items.jsonl"
    _make_writable(path)
    row = json.loads(path.read_bytes())
    row["item_id"] = "item-" + "0" * 64
    path.write_bytes(_canonical(row) + b"\n")

    with pytest.raises(BundleValidationError, match="item_id"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_rejects_reordered_rows_even_if_manifest_is_unchanged(tmp_path: Path) -> None:
    result = _write(tmp_path)
    path = result.path / "items.jsonl"
    _make_writable(path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(reversed(lines)))

    with pytest.raises(BundleValidationError, match="sorted"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_rejects_manifest_bundle_digest_tampering(tmp_path: Path) -> None:
    result = _write(tmp_path)
    path = result.path / "manifest.json"
    _make_writable(path)
    manifest = json.loads(path.read_bytes())
    manifest["bundle_digest"] = "0" * 64
    path.write_bytes(_canonical(manifest))

    with pytest.raises(BundleValidationError, match="manifest does not bind"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_requires_exact_two_file_inventory(tmp_path: Path) -> None:
    result = _write(tmp_path)
    extra = result.path / "extra.json"
    extra.write_bytes(b"{}")

    with pytest.raises(BundleValidationError, match="exactly two"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_rejects_symlink_member(tmp_path: Path) -> None:
    result = _write(tmp_path)
    member = result.path / "items.jsonl"
    target = tmp_path / "target.jsonl"
    target.write_bytes(member.read_bytes())
    _make_writable(member)
    member.unlink()
    try:
        member.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this host")

    with pytest.raises(BundleValidationError, match="non-regular|plain regular"):
        verify_normalized_bundle_v1(result.path)


def test_writer_applies_best_effort_readonly_members(tmp_path: Path) -> None:
    result = _write(tmp_path)
    for name in ("manifest.json", "items.jsonl"):
        info = (result.path / name).stat()
        if os.name == "nt":
            assert getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_READONLY", 0x1)
        else:
            assert stat.S_IMODE(info.st_mode) == 0o400


def test_bundle_artifacts_do_not_add_ambient_identifiers(tmp_path: Path) -> None:
    result = _write(tmp_path, (_item("note.md"),))
    manifest = json.loads((result.path / "manifest.json").read_bytes())

    assert "run_id" not in manifest
    assert "timestamp" not in manifest
    assert "created_at" not in manifest
    assert str(tmp_path) not in (result.path / "manifest.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("field,value", [("item_count", True), ("item_count", 2.0), ("items_bytes", False), ("items_bytes", 1.0)])
def test_manifest_counts_require_exact_integers(tmp_path: Path, field: str, value: object) -> None:
    result = _write(tmp_path)
    path = result.path / "manifest.json"
    _make_writable(path)
    manifest = json.loads(path.read_bytes())
    manifest[field] = value
    path.write_bytes(_canonical(manifest))

    with pytest.raises(BundleValidationError, match="exact bounded"):
        verify_normalized_bundle_v1(result.path)


def test_nonidentical_python_equal_manifest_is_not_reused(tmp_path: Path) -> None:
    result = _write(tmp_path)
    path = result.path / "manifest.json"
    _make_writable(path)
    manifest = json.loads(path.read_bytes())
    manifest["item_count"] = float(manifest["item_count"])
    path.write_bytes(_canonical(manifest))

    with pytest.raises(BundleCollisionError, match="destination collision"):
        _write(tmp_path)


def test_input_boundary_rejects_subclasses_and_iterators_without_using_them(tmp_path: Path) -> None:
    class HostileDict(dict):
        def items(self):
            raise RuntimeError("PRIVATE mapping failure")

    class HostileString(str):
        pass

    with pytest.raises(BundleValidationError) as mapping_error:
        NormalizedItemInputV1("note.md", "b" * 64, 1, dict(REF), HostileDict(body="x"))
    assert "PRIVATE" not in str(mapping_error.value)

    with pytest.raises(BundleValidationError) as string_error:
        NormalizedItemInputV1("note.md", HostileString("b" * 64), 1, dict(REF), {"body": "x"})
    assert "PRIVATE" not in str(string_error.value)

    touched = False

    def hostile_items():
        nonlocal touched
        touched = True
        raise RuntimeError("PRIVATE iterator failure")
        yield _item("never.md")

    with pytest.raises(TypeError, match="exact tuple or list"):
        write_normalized_bundle_v1(tmp_path / "bundles", _structures(), hostile_items())
    assert not touched

    with pytest.raises(BundleValidationError) as structure_error:
        write_normalized_bundle_v1(tmp_path / "other", HostileDict(_structures()), (_item("a.md"),))
    assert "PRIVATE" not in str(structure_error.value)


def test_json_normalization_enforces_depth_width_node_and_byte_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    nested: object = "leaf"
    for _ in range(bundle_v1.MAX_JSON_DEPTH + 1):
        nested = [nested]
    with pytest.raises(BundleValidationError, match="nesting"):
        bundle_v1._normalize_json({"body": nested}, maximum_bytes=10_000)

    monkeypatch.setattr(bundle_v1, "MAX_CONTAINER_WIDTH", 2)
    with pytest.raises(BundleValidationError, match="width"):
        bundle_v1._normalize_json({"body": [1, 2, 3]}, maximum_bytes=10_000)

    monkeypatch.setattr(bundle_v1, "MAX_CONTAINER_WIDTH", 100)
    monkeypatch.setattr(bundle_v1, "MAX_JSON_NODES", 5)
    with pytest.raises(BundleValidationError, match="node"):
        bundle_v1._normalize_json({"a": [1, 2, 3]}, maximum_bytes=10_000)

    with pytest.raises(BundleValidationError, match="character|byte"):
        bundle_v1._normalize_json({"x" * 100: "y" * 100}, maximum_bytes=50)


def test_character_limit_is_checked_before_json_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bundle_v1.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("encoding was attempted")),
    )
    with pytest.raises(BundleValidationError, match="character"):
        bundle_v1._normalize_json({"x" * 11: "value"}, maximum_bytes=10)


def test_multibyte_text_still_obeys_encoded_byte_budget() -> None:
    with pytest.raises(BundleValidationError, match="byte"):
        bundle_v1._normalize_json({"a": "ééé"}, maximum_bytes=12)


def test_inventory_enumeration_stops_after_third_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    original_scandir = os.scandir
    with original_scandir(result.path) as scan:
        real_entries = list(scan)

    class ExtraEntry:
        name = "extra"

    class BoundedScan:
        def __init__(self):
            self.values = iter([*real_entries, ExtraEntry(), ExtraEntry()])
            self.yielded = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            self.yielded += 1
            if self.yielded > 3:
                raise AssertionError("inventory enumeration was not bounded")
            return next(self.values)

    bounded = BoundedScan()
    monkeypatch.setattr(bundle_v1.os, "scandir", lambda _path: bounded)
    with pytest.raises(BundleValidationError, match="exactly two"):
        verify_normalized_bundle_v1(result.path)
    assert bounded.yielded == 3


def test_final_verification_detects_member_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    member = result.path / "items.jsonl"
    _make_writable(member)
    original = bundle_v1._verify_observed_bytes

    def mutate_during_semantics(path, manifest_raw, items_raw, **kwargs):
        with member.open("ab") as handle:
            handle.write(b" ")
            handle.flush()
        return original(path, manifest_raw, items_raw, **kwargs)

    monkeypatch.setattr(bundle_v1, "_verify_observed_bytes", mutate_during_semantics)
    with pytest.raises(BundleValidationError, match="changed"):
        verify_normalized_bundle_v1(result.path)


def test_final_verification_detects_inventory_insertion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    original = bundle_v1._verify_observed_bytes

    def insert_during_semantics(path, manifest_raw, items_raw, **kwargs):
        (result.path / "extra").write_bytes(b"hostile")
        return original(path, manifest_raw, items_raw, **kwargs)

    monkeypatch.setattr(bundle_v1, "_verify_observed_bytes", insert_during_semantics)
    with pytest.raises(BundleValidationError, match="changed|exactly two"):
        verify_normalized_bundle_v1(result.path)


def test_verification_rechecks_retained_directory_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    original = bundle_v1._plain_directory_identity
    calls = 0

    def changing_identity(path):
        nonlocal calls
        calls += 1
        identity = original(path)
        if calls >= 4:
            return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(bundle_v1, "_plain_directory_identity", changing_identity)
    with pytest.raises(BundleValidationError, match="directory changed"):
        verify_normalized_bundle_v1(result.path)


def test_verifier_retains_both_handles_and_closes_each_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    original_type = bundle_v1._RetainedMember
    original_semantics = bundle_v1._verify_observed_bytes
    retained = []

    class TrackingMember(original_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            retained.append(self)

        def close(self):
            self.close_calls += 1
            return super().close()

    def assert_both_retained(*args, **kwargs):
        assert len(retained) == 2
        assert all(not member.closed for member in retained)
        return original_semantics(*args, **kwargs)

    monkeypatch.setattr(bundle_v1, "_RetainedMember", TrackingMember)
    monkeypatch.setattr(bundle_v1, "_verify_observed_bytes", assert_both_retained)
    verify_normalized_bundle_v1(result.path)
    assert [member.close_calls for member in retained] == [1, 1]


def test_failed_publication_leaves_inert_dot_prefixed_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bundle_v1, "_fsync_directory", lambda _path: (_ for _ in ()).throw(OSError("PRIVATE fsync"))
    )
    with pytest.raises(RuntimeError, match="could not be published"):
        _write(tmp_path, (_item("a.md"),))

    stages = tuple((tmp_path / "bundles").iterdir())
    assert len(stages) == 1
    assert stages[0].name.startswith(".bundle-")
    assert {entry.name for entry in stages[0].iterdir()} == {"manifest.json", "items.jsonl"}
    assert not hasattr(bundle_v1, "_remove_owned_stage")


def test_identical_destination_is_verified_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write(tmp_path)
    monkeypatch.setattr(
        bundle_v1.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("stage was created")),
    )
    assert _write(tmp_path) == expected


def test_point_in_time_attestation_does_not_promise_future_immutability(tmp_path: Path) -> None:
    historical = _write(tmp_path, (_item("a.md"),))
    observed = verify_normalized_bundle_v1(historical.path)
    member = historical.path / "items.jsonl"
    _make_writable(member)
    with member.open("ab") as handle:
        handle.write(b" ")

    assert observed == historical
    with pytest.raises(BundleValidationError):
        verify_normalized_bundle_v1(historical.path)


def test_parent_fsync_failure_leaves_published_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = bundle_v1._fsync_directory
    calls = 0

    def fail_parent(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("PRIVATE parent fsync")
        return original(path)

    monkeypatch.setattr(bundle_v1, "_fsync_directory", fail_parent)
    with pytest.raises(BundlePublicationUncertainV1) as raised:
        _write(tmp_path, (_item("a.md"),))
    assert raised.value.phase is BundlePublicationUncertaintyPhaseV1.PARENT_DURABILITY
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not hasattr(raised.value, "path")
    assert raised.value.args == ()
    assert "PRIVATE" not in str(raised.value)
    published = tuple(path for path in (tmp_path / "bundles").iterdir() if not path.name.startswith("."))
    assert len(published) == 1
    verified = verify_normalized_bundle_v1(published[0])
    assert verified.semantic_identity == raised.value.semantic_identity


def test_final_verification_failure_has_path_free_typed_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = bundle_v1.verify_normalized_bundle_v1
    calls = 0

    def fail_final(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("PRIVATE final verification")
        return original(path)

    monkeypatch.setattr(bundle_v1, "verify_normalized_bundle_v1", fail_final)
    with pytest.raises(BundlePublicationUncertainV1) as raised:
        _write(tmp_path, (_item("a.md"),))
    assert raised.value.phase is BundlePublicationUncertaintyPhaseV1.FINAL_VERIFICATION
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not hasattr(raised.value, "path")
    assert raised.value.args == ()
    assert "PRIVATE" not in str(raised.value)


def test_semantic_identity_validates_all_summary_hashes() -> None:
    identity = BundleSemanticIdentityV1(
        "bundle-" + "1" * 64,
        "1" * 64,
        "2" * 64,
        1,
        10,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )
    assert identity.logical_paths_sha256 == "4" * 64
    assert identity.item_ids_sha256 == "5" * 64
    with pytest.raises(BundleValidationError, match="bundle_id"):
        BundleSemanticIdentityV1(
            "bundle-" + "0" * 64, "1" * 64, "2" * 64, 1, 10,
            "3" * 64, "4" * 64, "5" * 64,
        )


def test_manifest_identity_is_semantic_only(tmp_path: Path) -> None:
    result = _write(tmp_path, (_item("a.md"),))
    manifest = json.loads((result.path / "manifest.json").read_bytes())
    forbidden = {"plan_fingerprint", "request_id", "run_id", "timestamp", "path"}
    assert forbidden.isdisjoint(manifest)
    assert result.semantic_identity.logical_paths_sha256 == manifest["logical_paths_sha256"]
    assert result.semantic_identity.item_ids_sha256 == manifest["item_ids_sha256"]


def test_bundle_root_durability_retry_succeeds_for_plain_directory(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    root.mkdir()
    assert retry_bundle_root_durability_v1(root) is None


@pytest.mark.parametrize("kind", ["missing", "file", "non_path", "path_subclass"])
def test_bundle_root_durability_retry_rejects_unsafe_inputs(
    tmp_path: Path, kind: str
) -> None:
    value: object = tmp_path / "missing"
    if kind == "file":
        value = tmp_path / "file"
        value.write_bytes(b"not a directory")
    elif kind == "non_path":
        value = str(tmp_path)
    elif kind == "path_subclass":
        hostile_path_type = type("HostilePath", (type(Path()),), {})
        value = hostile_path_type(tmp_path)

    with pytest.raises(BundleDurabilityError) as raised:
        retry_bundle_root_durability_v1(value)  # type: ignore[arg-type]
    assert raised.value.args == ()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not hasattr(raised.value, "path")
    assert str(tmp_path) not in str(raised.value)
    assert str(tmp_path) not in repr(raised.value)


def test_bundle_root_durability_retry_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(BundleDurabilityError):
        retry_bundle_root_durability_v1(link)


def test_bundle_root_durability_retry_rejects_reparse_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundles"
    root.mkdir()
    monkeypatch.setattr(bundle_v1, "_is_reparse", lambda _info: True)
    with pytest.raises(BundleDurabilityError):
        retry_bundle_root_durability_v1(root)


def test_bundle_root_durability_retry_sanitizes_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private-secret-root"
    root.mkdir()
    monkeypatch.setattr(
        bundle_v1,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(RuntimeError("PRIVATE provider detail")),
    )
    with pytest.raises(BundleDurabilityError) as raised:
        retry_bundle_root_durability_v1(root)
    assert raised.value.args == ()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not hasattr(raised.value, "path")
    assert "PRIVATE" not in str(raised.value)
    assert "PRIVATE" not in repr(raised.value)
    assert str(root) not in str(raised.value)
    assert str(root) not in repr(raised.value)


def test_bundle_digest_binds_exact_structure_set_document(tmp_path: Path) -> None:
    first = write_normalized_bundle_v1(tmp_path / "one", _structures(), (_item("a.md"),))
    changed = _structures()
    changed["bindings"][0]["extra_policy"] = "different"
    second = write_normalized_bundle_v1(tmp_path / "two", changed, (_item("a.md"),))
    assert first.semantic_identity.bundle_digest != second.semantic_identity.bundle_digest


@pytest.mark.parametrize("failure", ["flush", "fsync", "close"])
def test_file_sink_has_one_close_owner_and_no_fallback_os_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    sink = bundle_v1._FileSink(tmp_path / "member", 100)
    sink.write(b"payload")
    inner = sink._handle
    close_calls = 0

    class HandleProxy:
        def __init__(self, inner):
            self.inner = inner

        def flush(self):
            if failure == "flush":
                raise RuntimeError("PRIVATE flush failure")
            return self.inner.flush()

        def fileno(self):
            return self.inner.fileno()

        def close(self):
            nonlocal close_calls
            close_calls += 1
            if failure == "close":
                raise RuntimeError("PRIVATE close failure")
            return self.inner.close()

    sink._handle = HandleProxy(sink._handle)
    if failure == "fsync":
        monkeypatch.setattr(bundle_v1.os, "fsync", lambda _descriptor: (_ for _ in ()).throw(RuntimeError("PRIVATE fsync failure")))
    monkeypatch.setattr(
        bundle_v1.os,
        "close",
        lambda _descriptor: (_ for _ in ()).throw(AssertionError("fallback os.close was called")),
    )

    with pytest.raises(RuntimeError, match="finalization failed") as error:
        sink.finish()
    assert "PRIVATE" not in str(error.value)
    assert sink._closed
    assert close_calls == 1
    if failure == "close":
        inner.close()
