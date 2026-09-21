from __future__ import annotations

import json
import errno
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    MarkdownProfileV1,
    SourceMatcher,
    StructureBinding,
    StructureDefinition,
    StructureSet,
    TextProjection,
)
from tuner.dataset_prep import (
    DatasetPrepCollisionError,
    DatasetPrepConfigV1,
    DatasetPrepDurabilityError,
    DatasetPrepValidationError,
    DatasetPublicationUncertainV1,
    DatasetPublicationUncertaintyPhaseV1,
    prepare_dataset_v1,
    retry_dataset_root_durability_v1,
    snapshot_prepared_dataset_v1,
    verify_prepared_dataset_v1,
)
from tuner.dataset_prep import publication
from tuner.ingestion import NormalizedItemInputV1, write_normalized_bundle_v1


def _structure() -> StructureSet:
    definition = StructureDefinition.define(
        name="MarkdownNote",
        version="1",
        markdown=MarkdownProfileV1(FrontmatterMode.OPTIONAL),
        fields=(
            FieldMapping(
                "body",
                FieldSelector(FieldSelectorKind.DOCUMENT_BODY),
                FieldValueKind.STRING,
                True,
            ),
        ),
        text_projections=(TextProjection("text", "body"),),
    )
    return StructureSet(
        (definition,),
        (StructureBinding("markdown", SourceMatcher("**/*.md"), definition.ref),),
    )


def _bundle(tmp_path: Path, texts: tuple[str, ...] = ("Alpha", "Beta", "Gamma")):
    structure = _structure()
    items = tuple(
        NormalizedItemInputV1(
            logical_path=f"chapter-{index:02d}.md",
            source_sha256=(f"{index + 1:x}" * 64)[:64],
            source_size_bytes=len(text.encode("utf-8")),
            structure_ref=structure.structures[0].ref.to_dict(),
            fields={"body": text},
        )
        for index, text in enumerate(texts)
    )
    return structure, write_normalized_bundle_v1(tmp_path / "bundles", structure.to_dict(), items)


def _config(bundle, structure: StructureSet, *, ordering=None, split=None) -> DatasetPrepConfigV1:
    return DatasetPrepConfigV1.from_dict(
        {
            "schema_version": "syntunia-dataset-prep/v1",
            "source": {
                "bundle_path": str(bundle.path),
                "expected_bundle_digest": bundle.semantic_identity.bundle_digest,
            },
            "format": "raw_text",
            "projection": {
                "structure_ref": structure.structures[0].ref.to_dict(),
                "name": "text",
            },
            "ordering": ordering or {"kind": "source_order"},
            "split": split or {"kind": "none"},
        }
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (path / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]


def _prepare_reconciled(config: DatasetPrepConfigV1, output_root: Path):
    """A verified rerun reconciles platforms without directory-fsync proof."""
    try:
        return prepare_dataset_v1(config, output_root)
    except DatasetPublicationUncertainV1 as error:
        assert error.phase is DatasetPublicationUncertaintyPhaseV1.PARENT_DURABILITY
        return prepare_dataset_v1(config, output_root)


def test_prepare_verify_and_idempotently_reuse_exact_artifact(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)

    first = _prepare_reconciled(config, tmp_path / "private")
    second = prepare_dataset_v1(config, tmp_path / "private")
    verified = verify_prepared_dataset_v1(first.path)

    assert first == second == verified
    assert first.path.name == "dataset-" + first.semantic_identity.dataset_digest
    assert {path.name for path in first.path.iterdir()} == {"manifest.json", "dataset.jsonl"}
    rows = _rows(first.path)
    assert [row["text"] for row in rows] == ["Alpha", "Beta", "Gamma"]
    assert all(row["schema_version"] == "syntunia-sft-row/v1" for row in rows)
    assert all(row["format"] == "raw_text" and row["split"] == "train" for row in rows)


def test_snapshot_returns_the_jsonl_bytes_from_the_retained_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path, ("PRIVATE SNAPSHOT SENTINEL",))
    prepared = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    original = publication._RetainedMember.read
    reads: list[tuple[str, bytes]] = []

    def observe(member):
        payload = original(member)
        reads.append((member.path.name, payload))
        return payload

    monkeypatch.setattr(publication._RetainedMember, "read", observe)
    verified, snapshot = snapshot_prepared_dataset_v1(prepared.path)

    assert verified == prepared
    assert [name for name, _payload in reads] == ["manifest.json", "dataset.jsonl"]
    assert snapshot == reads[1][1]
    before = bytes(snapshot)
    dataset_path = prepared.path / "dataset.jsonl"
    dataset_path.chmod(0o600)
    dataset_path.write_bytes(b"changed after the snapshot\n")
    assert snapshot == before


def test_row_identity_excludes_order_and_split_but_dataset_identity_binds_them(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path)
    source_order = _prepare_reconciled(_config(bundle, structure), tmp_path / "one")
    shuffled = _prepare_reconciled(
        _config(
            bundle,
            structure,
            ordering={"kind": "seeded_hash", "seed": "order-seed"},
            split={
                "kind": "hash_rank",
                "seed": "split-seed",
                "allocations": [
                    {"name": "train", "weight": 2},
                    {"name": "validation", "weight": 1},
                ],
            },
        ),
        tmp_path / "two",
    )

    first_by_source = {row["source_item_id"]: row["row_id"] for row in _rows(source_order.path)}
    second_by_source = {row["source_item_id"]: row["row_id"] for row in _rows(shuffled.path)}
    assert first_by_source == second_by_source
    assert source_order.semantic_identity.dataset_digest != shuffled.semantic_identity.dataset_digest
    assert dict(shuffled.semantic_identity.split_counts) == {"train": 2, "validation": 1}


def test_hash_rank_uses_exact_largest_remainder_counts(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path, tuple(f"Text {index}" for index in range(75)))
    result = _prepare_reconciled(
        _config(
            bundle,
            structure,
            split={
                "kind": "hash_rank",
                "seed": "stable-split",
                "allocations": [
                    {"name": "train", "weight": 9},
                    {"name": "validation", "weight": 1},
                ],
            },
        ),
        tmp_path / "private",
    )
    assert dict(result.semantic_identity.split_counts) == {"train": 68, "validation": 7}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(extra=True),
        lambda config: config["projection"].update(extra=True),
        lambda config: config.update(format="messages"),
        lambda config: config.update(ordering={"kind": "source_order", "seed": "forbidden"}),
        lambda config: config.update(split={"kind": "none", "seed": "forbidden", "allocations": []}),
    ],
)
def test_config_rejects_unknown_or_inconsistent_fields(tmp_path: Path, mutation) -> None:
    structure, bundle = _bundle(tmp_path)
    raw = {
        "schema_version": "syntunia-dataset-prep/v1",
        "source": {
            "bundle_path": str(bundle.path),
            "expected_bundle_digest": bundle.semantic_identity.bundle_digest,
        },
        "format": "raw_text",
        "projection": {"structure_ref": structure.structures[0].ref.to_dict(), "name": "text"},
        "ordering": {"kind": "source_order"},
        "split": {"kind": "none"},
    }
    mutation(raw)
    with pytest.raises(DatasetPrepValidationError):
        DatasetPrepConfigV1.from_dict(raw)


def test_hash_rank_rejects_non_trainer_split_names(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path)
    with pytest.raises(DatasetPrepValidationError, match="train then validation"):
        _config(
            bundle,
            structure,
            split={
                "kind": "hash_rank",
                "seed": "split-seed",
                "allocations": [
                    {"name": "training", "weight": 9},
                    {"name": "dev", "weight": 1},
                ],
            },
        )


def test_artifact_recipe_rejects_non_trainer_split_names() -> None:
    recipe = {
        "schema_version": "syntunia-dataset-prep/v1",
        "format": "raw_text",
        "projection": {
            "structure_ref": {"name": "MarkdownNote", "version": "1", "digest": "1" * 64},
            "name": "text",
        },
        "ordering": {"kind": "source_order"},
        "split": {
            "kind": "hash_rank",
            "seed_sha256": "2" * 64,
            "allocations": [
                {"name": "training", "weight": 9},
                {"name": "dev", "weight": 1},
            ],
        },
    }
    with pytest.raises(DatasetPrepValidationError, match="allocation names"):
        publication._validate_recipe(recipe)


def test_requires_exact_bundle_structure_and_projection(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path)
    wrong_projection = _config(bundle, structure)
    object.__setattr__(wrong_projection.projection, "name", "missing")
    with pytest.raises(DatasetPrepValidationError, match="projection"):
        prepare_dataset_v1(wrong_projection, tmp_path / "private")

    wrong_digest = _config(bundle, structure)
    object.__setattr__(wrong_digest, "expected_bundle_digest", "0" * 64)
    with pytest.raises(DatasetPrepValidationError, match="digest"):
        prepare_dataset_v1(wrong_digest, tmp_path / "other")


def test_manifest_has_no_paths_or_corpus_prose(tmp_path: Path) -> None:
    secret_text = "PRIVATE PROSE SENTINEL"
    secret_seed = "PRIVATE SEED SENTINEL"
    structure, bundle = _bundle(tmp_path, (secret_text,))
    result = _prepare_reconciled(
        _config(
            bundle,
            structure,
            ordering={"kind": "seeded_hash", "seed": secret_seed},
        ),
        tmp_path / "private-output",
    )
    manifest_raw = (result.path / "manifest.json").read_text(encoding="utf-8")

    assert secret_text not in manifest_raw
    assert secret_seed not in manifest_raw
    assert str(tmp_path) not in manifest_raw
    assert "chapter-00.md" not in manifest_raw
    assert secret_text in (result.path / "dataset.jsonl").read_text(encoding="utf-8")


def test_tampering_and_destination_collision_fail_closed(tmp_path: Path) -> None:
    structure, bundle = _bundle(tmp_path)
    result = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    dataset_path = result.path / "dataset.jsonl"
    dataset_path.chmod(0o600)
    dataset_path.write_bytes(b"tampered\n")

    with pytest.raises(DatasetPrepValidationError):
        verify_prepared_dataset_v1(result.path)
    with pytest.raises(DatasetPrepCollisionError):
        prepare_dataset_v1(_config(bundle, structure), tmp_path / "private")


def test_publication_uncertainty_is_typed_and_sanitized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    structure, bundle = _bundle(tmp_path)
    original = publication._sync_directory_authority
    calls = 0

    def fail_parent(descriptor, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("PRIVATE filesystem detail")
        original(descriptor, path)

    monkeypatch.setattr(publication, "_sync_directory_authority", fail_parent)
    with pytest.raises(DatasetPublicationUncertainV1) as raised:
        prepare_dataset_v1(_config(bundle, structure), tmp_path / "private")
    assert raised.value.phase is DatasetPublicationUncertaintyPhaseV1.PARENT_DURABILITY
    assert raised.value.args == ()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "PRIVATE" not in repr(raised.value)


def test_authenticated_root_coordination_serializes_cooperative_publishers(tmp_path: Path) -> None:
    output_root = tmp_path / "private"
    output_root.mkdir()
    first_root = publication._OutputRootAuthority(output_root)
    first_lock = publication._OutputRootCoordination(first_root)
    second_started = threading.Event()
    second_acquired = threading.Event()
    failures: list[BaseException] = []

    def acquire_second() -> None:
        second_root = None
        second_lock = None
        try:
            second_root = publication._OutputRootAuthority(output_root)
            second_started.set()
            second_lock = publication._OutputRootCoordination(second_root)
            second_acquired.set()
        except BaseException as error:
            failures.append(error)
        finally:
            if second_lock is not None:
                second_lock.close()
            if second_root is not None:
                second_root.close()

    worker = threading.Thread(target=acquire_second)
    worker.start()
    assert second_started.wait(2)
    assert not second_acquired.wait(0.1)
    assert first_lock.close()
    assert first_root.close()
    worker.join(2)
    assert not worker.is_alive()
    assert second_acquired.is_set()
    assert failures == []


def test_commit_flag_is_set_only_after_destination_identity_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    original = publication._destination_matches_stage
    observed: list[tuple[str | None, bool]] = []

    def observe(stage, destination: Path) -> bool:
        observed.append((stage.published_name, stage.committed))
        return original(stage, destination)

    monkeypatch.setattr(publication, "_destination_matches_stage", observe)
    _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    assert len(observed) == 1
    assert observed[0][0] is not None
    assert observed[0][1] is False


@pytest.mark.skipif(sys.platform != "linux", reason="retained descriptor scrub regression")
def test_post_rename_mismatch_scrubs_private_bytes_before_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path, ("PRIVATE DATASET SENTINEL",))
    config = _config(bundle, structure)
    monkeypatch.setattr(publication, "_destination_matches_stage", lambda _stage, _destination: False)

    with pytest.raises(DatasetPublicationUncertainV1) as raised:
        prepare_dataset_v1(config, tmp_path / "private")

    assert raised.value.phase is DatasetPublicationUncertaintyPhaseV1.FINAL_VERIFICATION
    destination = tmp_path / "private" / raised.value.semantic_identity.dataset_id
    assert (destination / "dataset.jsonl").read_bytes() == b""
    assert (destination / "manifest.json").read_bytes() == b""
    assert raised.value.__context__ is None


def test_cleanup_never_claims_success_without_verified_name_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    observed = []
    monkeypatch.setattr(publication, "_sync_directory_authority", lambda _descriptor, _path: None)
    monkeypatch.setattr(
        publication,
        "_verify_owned_stage",
        lambda _stage: (_ for _ in ()).throw(DatasetPrepValidationError("PRIVATE precommit detail")),
    )

    def deny_absence(stage) -> bool:
        observed.append(stage)
        return False

    monkeypatch.setattr(publication, "_owned_stage_name_absent", deny_absence)
    with pytest.raises(DatasetPublicationUncertainV1) as raised:
        prepare_dataset_v1(_config(bundle, structure), tmp_path / "private")
    assert raised.value.phase is DatasetPublicationUncertaintyPhaseV1.STAGE_CLEANUP
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert "PRIVATE" not in repr(raised.value)
    assert len(observed) == 1
    assert observed[0].cleaned is False


def test_unprovable_directory_durability_is_uncertain_then_verified_rerun_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)

    monkeypatch.setattr(
        publication,
        "_sync_directory_authority",
        lambda _descriptor, _path: (_ for _ in ()).throw(OSError("PRIVATE unsupported durability")),
    )
    monkeypatch.setattr(
        publication,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("PRIVATE unsupported durability")),
    )
    with pytest.raises(DatasetPublicationUncertainV1) as raised:
        prepare_dataset_v1(config, tmp_path / "private")
    assert raised.value.phase is DatasetPublicationUncertaintyPhaseV1.PARENT_DURABILITY
    assert raised.value.args == ()
    assert raised.value.__context__ is None

    reconciled = prepare_dataset_v1(config, tmp_path / "private")
    assert reconciled.semantic_identity == raised.value.semantic_identity
    with pytest.raises(DatasetPrepDurabilityError) as retry:
        retry_dataset_root_durability_v1(tmp_path / "private")
    assert retry.value.args == ()
    assert "PRIVATE" not in repr(retry.value)


def test_windows_directory_sync_never_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    monkeypatch.setattr(publication.os, "name", "nt")
    with pytest.raises(OSError):
        publication._fsync_directory(root)
    with pytest.raises(DatasetPrepDurabilityError) as raised:
        retry_dataset_root_durability_v1(root)
    assert raised.value.args == ()


def test_exact_concurrent_winner_is_verified_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)
    monkeypatch.setattr(publication, "_sync_directory_authority", lambda _descriptor, _path: None)

    def lose_to_exact_winner(stage, destination: Path) -> None:
        shutil.copytree(stage.path, destination)
        raise FileExistsError(errno.EEXIST, "PRIVATE concurrent winner")

    monkeypatch.setattr(publication, "_commit_owned_stage", lose_to_exact_winner)
    result = prepare_dataset_v1(config, tmp_path / "private")
    assert result == verify_prepared_dataset_v1(result.path)
    assert not any(path.name.startswith(".") for path in (tmp_path / "private").iterdir())


def test_invalid_concurrent_winner_is_a_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)
    monkeypatch.setattr(publication, "_sync_directory_authority", lambda _descriptor, _path: None)

    def lose_to_invalid_winner(stage, destination: Path) -> None:
        shutil.copytree(stage.path, destination)
        dataset = destination / "dataset.jsonl"
        dataset.chmod(0o600)
        dataset.write_bytes(b"tampered\n")
        raise FileExistsError(errno.EEXIST, "PRIVATE concurrent winner")

    monkeypatch.setattr(publication, "_commit_owned_stage", lose_to_invalid_winner)
    with pytest.raises(DatasetPrepCollisionError) as raised:
        prepare_dataset_v1(config, tmp_path / "private")
    assert "PRIVATE" not in repr(raised.value)
    assert not any(path.name.startswith(".") for path in (tmp_path / "private").iterdir())


def test_precommit_failure_removes_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)
    root = tmp_path / "private"
    sentinel = root / "keep.txt"
    root.mkdir()
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(publication, "_sync_directory_authority", lambda _descriptor, _path: None)
    monkeypatch.setattr(
        publication,
        "_verify_owned_stage",
        lambda _stage: (_ for _ in ()).throw(DatasetPrepValidationError("synthetic precommit failure")),
    )

    with pytest.raises(DatasetPrepValidationError, match="precommit"):
        prepare_dataset_v1(config, root)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert [path.name for path in root.iterdir()] == ["keep.txt"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor-relative disclosure regression")
def test_linux_stage_path_symlink_swap_cannot_receive_private_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path, ("PRIVATE DATASET SENTINEL",))
    config = _config(bundle, structure)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_verify = publication._OwnedStageAuthority.verify_binding
    swapped = False

    def swap_after_authentication(stage):
        nonlocal swapped
        original_verify(stage)
        if not swapped and stage.descriptor is not None:
            swapped = True
            moved = stage.root.path / ".attacker-moved-stage"
            stage.path.rename(moved)
            stage.path.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(publication._OwnedStageAuthority, "verify_binding", swap_after_authentication)
    with pytest.raises(DatasetPublicationUncertainV1) as raised:
        prepare_dataset_v1(config, tmp_path / "private")
    assert raised.value.phase is DatasetPublicationUncertaintyPhaseV1.STAGE_CLEANUP
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows retained directory handle regression")
def test_windows_stage_handle_blocks_path_substitution_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)
    original_verify = publication._OwnedStageAuthority.verify_binding
    attempted = False
    blocked = False

    def attempt_swap(stage):
        nonlocal attempted, blocked
        original_verify(stage)
        if not attempted and stage.native_handle is not None:
            attempted = True
            try:
                stage.path.rename(stage.root.path / ".attacker-stage")
            except OSError:
                blocked = True

    monkeypatch.setattr(publication._OwnedStageAuthority, "verify_binding", attempt_swap)
    _prepare_reconciled(config, tmp_path / "private")
    assert attempted and blocked


def test_retry_durability_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(Exception) as raised:
        retry_dataset_root_durability_v1(link)
    assert str(link) not in str(raised.value)


def test_verifier_rejects_directory_identity_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    result = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    original = publication._plain_directory
    calls = 0

    def replaced(path: Path):
        nonlocal calls
        calls += 1
        identity = original(path)
        if calls >= 4:
            return (*identity[:4], identity[4] + 1)
        return identity

    monkeypatch.setattr(publication, "_plain_directory", replaced)
    with pytest.raises(DatasetPrepValidationError, match="directory identity changed"):
        verify_prepared_dataset_v1(result.path)


def test_verifier_rejects_member_replacement_before_final_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    result = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    original = publication._inventory
    calls = 0

    def replaced(path: Path, identity):
        nonlocal calls
        calls += 1
        inventory = original(path, identity)
        if calls == 2:
            inventory = dict(inventory)
            member = inventory["dataset.jsonl"]
            inventory["dataset.jsonl"] = (member[0], member[1] + 1, *member[2:])
        return inventory

    monkeypatch.setattr(publication, "_inventory", replaced)
    with pytest.raises(DatasetPrepValidationError, match="inventory changed"):
        verify_prepared_dataset_v1(result.path)


def test_verifier_rejects_member_mutation_while_handle_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    result = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    original = publication._RetainedMember.read
    mutated = False

    def mutate_after_read(member):
        nonlocal mutated
        payload = original(member)
        if member.path.name == "dataset.jsonl" and not mutated:
            mutated = True
            member.path.chmod(0o600)
            replacement = bytearray(payload)
            replacement[-2] = replacement[-2] ^ 1
            member.path.write_bytes(bytes(replacement))
        return payload

    monkeypatch.setattr(publication._RetainedMember, "read", mutate_after_read)
    with pytest.raises(DatasetPrepValidationError, match="changed during verification"):
        verify_prepared_dataset_v1(result.path)


def test_verifier_fails_closed_when_retained_handle_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    result = _prepare_reconciled(_config(bundle, structure), tmp_path / "private")
    original = publication._RetainedMember.close
    failed = False

    def fail_once(member):
        nonlocal failed
        closed = original(member)
        if not failed:
            failed = True
            return False
        return closed

    monkeypatch.setattr(publication._RetainedMember, "close", fail_once)
    with pytest.raises(DatasetPrepValidationError) as raised:
        verify_prepared_dataset_v1(result.path)
    assert raised.value.__context__ is None
    assert str(tmp_path) not in str(raised.value)


def test_stage_final_and_idempotent_reuse_share_retained_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structure, bundle = _bundle(tmp_path)
    config = _config(bundle, structure)
    original = publication._RetainedDatasetDirectory.__init__
    original_stage_verify = publication._verify_owned_stage
    observed: list[str] = []
    stage_verifications = 0

    def record(authority, path: Path):
        observed.append(path.name)
        original(authority, path)

    def record_stage(stage):
        nonlocal stage_verifications
        stage_verifications += 1
        return original_stage_verify(stage)

    monkeypatch.setattr(publication._RetainedDatasetDirectory, "__init__", record)
    monkeypatch.setattr(publication, "_verify_owned_stage", record_stage)
    monkeypatch.setattr(publication, "_sync_directory_authority", lambda _descriptor, _path: None)
    first = prepare_dataset_v1(config, tmp_path / "private")
    after_first = len(observed)
    second = prepare_dataset_v1(config, tmp_path / "private")

    assert first == second
    assert stage_verifications == 1
    assert after_first >= 1  # final destination, plus ambient stage verification on Windows
    assert len(observed) == after_first + 1  # existing-destination reuse
