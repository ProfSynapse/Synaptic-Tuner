from __future__ import annotations

import json
import hashlib
from dataclasses import replace
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
    ContextPackageV2,
    DatasetPrepConfigV2,
    DatasetPrepValidationError,
    GroupSplitV2,
    ItemLineageV2,
    ProjectionV1,
    PromptVariantV2,
    SplitAllocationV1,
    TargetTransformsV2,
    build_prepared_dataset_v2,
)
from tuner.dataset_prep import publication
from tuner.ingestion import (
    NormalizedItemInputV1,
    load_verified_normalized_bundle_v1,
    write_normalized_bundle_v1,
)


def _bundle(tmp_path: Path):
    definition = StructureDefinition.define(
        name="GenericDocument",
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
    structures = StructureSet(
        (definition,),
        (StructureBinding("markdown", SourceMatcher("**/*.md"), definition.ref),),
    )
    documents = {
        "context-root.md": "Earliest whole document.",
        "context-a.md": "Earlier whole document A.",
        "context-b.md": "Earlier whole document B.",
        "target-a.md": "KEEP: answer A\nDROP: private line\n```omit\nhidden\n```\n",
        "target-b.md": "answer B",
        "target-c.md": "answer C",
        "target-d.md": "answer D",
        "target-e.md": "answer E",
    }
    inputs = tuple(
        NormalizedItemInputV1(
            logical_path=path,
            source_sha256=f"{index + 1:064x}",
            source_size_bytes=len(text.encode("utf-8")),
            structure_ref=definition.ref.to_dict(),
            fields={"body": text},
        )
        for index, (path, text) in enumerate(documents.items())
    )
    written = write_normalized_bundle_v1(tmp_path / "bundles", structures.to_dict(), inputs)
    loaded = load_verified_normalized_bundle_v1(written.path)
    ids = {item.logical_path: item.item_id for item in loaded.items}
    return definition, loaded, ids, documents


def _config(tmp_path: Path):
    definition, bundle, ids, documents = _bundle(tmp_path)
    lineage = (
        ItemLineageV2(ids["context-root.md"], "reference-root", 0, (), "reference-root"),
        ItemLineageV2(ids["context-a.md"], "reference-a", 1, (), "reference-a"),
        ItemLineageV2(ids["context-b.md"], "reference-b", 2, (), "reference-b"),
        ItemLineageV2(ids["target-a.md"], "target-a", 10, (), "group-ab"),
        ItemLineageV2(ids["target-b.md"], "target-b", 11, (ids["target-a.md"],), "group-ab"),
        ItemLineageV2(ids["target-c.md"], "target-c", 12, (), "group-c"),
        ItemLineageV2(ids["target-d.md"], "target-d", 13, (), "group-d"),
        ItemLineageV2(ids["target-e.md"], "target-e", 14, (), "group-e"),
    )
    packages = tuple(
        ContextPackageV2(
            ids[f"target-{letter}.md"],
            (ids["context-b.md"], ids["context-a.md"]),
            "alternate" if letter == "e" else "plain",
        )
        for letter in "abcde"
    )
    config = DatasetPrepConfigV2(
        source_bundle_path=bundle.path,
        expected_bundle_digest=bundle.semantic_identity.bundle_digest,
        target_projection=ProjectionV1.from_dict(
            {"structure_ref": definition.ref.to_dict(), "name": "text"}
        ),
        lineage=lineage,
        packages=packages,
        prompt_variants=(
            PromptVariantV2("plain", "Use the context.", "\n\n---\n\n"),
            PromptVariantV2("alternate", "Consider the material.", "\n\n***\n\n"),
        ),
        target_transforms=TargetTransformsV2(("omit",), ("DROP:",)),
        split=GroupSplitV2(
            "stable-fixture-split",
            (SplitAllocationV1("train", 3), SplitAllocationV1("validation", 1)),
        ),
    )
    return bundle, ids, documents, config


def _tail_config(config, ids):
    targets = {ids[f"target-{letter}.md"] for letter in "abcde"}
    lineage = tuple(
        replace(entry, group_id="series") if entry.item_id in targets else entry
        for entry in config.lineage
    )
    return replace(
        config, lineage=lineage,
        split=GroupSplitV2(
            None,
            (SplitAllocationV1("train", 4), SplitAllocationV1("validation", 1)),
            "group_sequence_tail",
        ),
    )


def _relabel_and_rehash(prepared, index: int, split: str):
    rows = [json.loads(line) for line in prepared.dataset_raw.splitlines()]
    rows[index]["split"] = split
    dataset_raw = b"".join(publication._canonical_bytes(row) + b"\n" for row in rows)
    manifest = json.loads(prepared.manifest_raw)
    manifest["dataset_bytes"] = len(dataset_raw)
    manifest["dataset_sha256"] = hashlib.sha256(dataset_raw).hexdigest()
    manifest["split_counts"] = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation")
    }
    basis = {
        "source_bundle_digest": manifest["source"]["bundle_digest"],
        "source_structure_set_digest": manifest["source"]["structure_set_digest"],
        "projection": manifest["projection"],
        "projection_digest": manifest["projection_digest"],
        "recipe": manifest["recipe"],
        "lineage": manifest["lineage"],
        "lineage_digest": manifest["lineage_digest"],
        "group_count": manifest["group_count"],
        "group_ids_sha256": manifest["group_ids_sha256"],
        "row_count": manifest["row_count"],
        "dataset_bytes": manifest["dataset_bytes"],
        "dataset_sha256": manifest["dataset_sha256"],
        "row_ids_sha256": manifest["row_ids_sha256"],
        "split_counts": manifest["split_counts"],
    }
    if "split_lineage" in manifest:
        basis["split_lineage"] = manifest["split_lineage"]
    manifest["dataset_digest"] = publication._domain_digest(publication._DATASET_DOMAIN_V2, basis)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    return manifest, publication._canonical_bytes(manifest), dataset_raw


def test_sequence_tail_preserves_rows_and_verifies_authenticated_assignment(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    old = build_prepared_dataset_v2(bundle, legacy)
    assert [row.messages for row in prepared.rows] == [row.messages for row in old.rows]
    assert [row.split for row in prepared.rows] == ["train"] * 4 + ["validation"]
    assert "split_lineage" in prepared.manifest
    assert "split_lineage" not in old.manifest
    assert prepared.manifest["recipe"]["split"] == {
        "kind": "group_sequence_tail",
        "allocations": [{"name": "train", "weight": 4}, {"name": "validation", "weight": 1}],
    }
    assert publication._verify_bytes_v2(
        tmp_path / prepared.identity.dataset_id,
        prepared.manifest_raw, prepared.dataset_raw,
    ).semantic_identity == prepared.identity
    rows = [
        (row.target_item_id, row.group_id, row.split, row.context_item_ids)
        for row in prepared.rows
    ]
    wrong = list(rows)
    target, group, _split, contexts = wrong[0]
    wrong[0] = (target, group, "validation", contexts)
    with pytest.raises(DatasetPrepValidationError, match="assignment"):
        publication._verify_sequence_tail(
            dict(prepared.manifest), prepared.manifest["recipe"], wrong,
            bundle.semantic_identity.item_count,
        )


def test_sequence_tail_keeps_equal_sequence_cohort_together(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["target-d.md"], sequence=14)
    prepared = build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))
    assert [row.split for row in prepared.rows] == ["train"] * 3 + ["validation"] * 2
    assert publication._verify_bytes_v2(
        tmp_path / prepared.identity.dataset_id,
        prepared.manifest_raw, prepared.dataset_raw,
    ).semantic_identity == prepared.identity


def test_sequence_tail_rejects_single_target_group_and_revision_leak(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = replace(legacy, split=GroupSplitV2(
        None,
        (SplitAllocationV1("train", 4), SplitAllocationV1("validation", 1)),
        "group_sequence_tail",
    ))
    with pytest.raises(DatasetPrepValidationError, match="chronology boundary"):
        build_prepared_dataset_v2(bundle, config)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["context-root.md"], revision_family="target-e")
    packages = (
        replace(config.packages[0], context_item_ids=(ids["context-root.md"], ids["context-a.md"])),
        *config.packages[1:],
    )
    with pytest.raises(DatasetPrepValidationError, match="held-out target lineage"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage, packages=packages))


def test_sequence_tail_allows_validation_to_use_prior_training_target(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    packages = (*config.packages[:-1], replace(config.packages[-1], context_item_ids=(ids["target-d.md"],)))
    prepared = build_prepared_dataset_v2(bundle, replace(config, packages=packages))
    assert prepared.rows[-2].split == "train"
    assert prepared.rows[-1].split == "validation"
    assert prepared.rows[-1].context_item_ids == (ids["target-d.md"],)
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity


def test_sequence_tail_rejects_transitive_heldout_revision_and_target_family_crossing(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    lineage = _lineage_with(config.lineage, ids["context-root.md"], revision_family="target-e")
    lineage = _lineage_with(lineage, ids["context-a.md"], derived_from=(ids["context-root.md"],))
    packages = tuple(
        replace(package, context_item_ids=(ids["context-b.md"],))
        if package.target_item_id == ids["target-e.md"] else package for package in config.packages
    )
    with pytest.raises(DatasetPrepValidationError, match="held-out target lineage"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage, packages=packages))
    same_family = _lineage_with(config.lineage, ids["target-e.md"], revision_family="target-a")
    with pytest.raises(DatasetPrepValidationError, match="revision family crosses"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=same_family))


def test_sequence_tail_assignment_does_not_depend_on_package_array_order(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    config = _tail_config(legacy, ids)
    forward = build_prepared_dataset_v2(bundle, config)
    reversed_rows = build_prepared_dataset_v2(bundle, replace(config, packages=tuple(reversed(config.packages))))
    assert {row.target_item_id: row.split for row in forward.rows} == {
        row.target_item_id: row.split for row in reversed_rows.rows
    }


def test_artifact_verifier_rejects_rehashed_wrong_tail_assignment_and_old_mixed_group(tmp_path: Path) -> None:
    bundle, ids, _documents, legacy = _config(tmp_path)
    tail = build_prepared_dataset_v2(bundle, _tail_config(legacy, ids))
    manifest, manifest_raw, dataset_raw = _relabel_and_rehash(tail, 0, "validation")
    wrong_tail = tmp_path / manifest["dataset_id"]
    wrong_tail.mkdir(mode=0o700)
    publication._write_member(wrong_tail / "manifest.json", manifest_raw)
    publication._write_member(wrong_tail / "dataset.jsonl", dataset_raw)
    with pytest.raises(DatasetPrepValidationError, match="assignment"):
        publication.verify_prepared_dataset_v2(wrong_tail)
    old = build_prepared_dataset_v2(bundle, legacy)
    old_split = old.rows[0].split
    manifest, manifest_raw, dataset_raw = _relabel_and_rehash(
        old, 1, "validation" if old_split == "train" else "train",
    )
    wrong_old = tmp_path / manifest["dataset_id"]
    wrong_old.mkdir(mode=0o700)
    publication._write_member(wrong_old / "manifest.json", manifest_raw)
    publication._write_member(wrong_old / "dataset.jsonl", dataset_raw)
    with pytest.raises(DatasetPrepValidationError, match="group crosses declared splits"):
        publication.verify_prepared_dataset_v2(wrong_old)


def test_five_generic_packages_render_exact_two_message_rows_and_safe_manifest(tmp_path: Path) -> None:
    bundle, ids, documents, config = _config(tmp_path)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    repeated = build_prepared_dataset_v2(bundle, config)

    assert prepared.dataset_raw == repeated.dataset_raw
    assert prepared.manifest_raw == repeated.manifest_raw
    assert len(prepared.rows) == 5
    first = prepared.rows[0].to_dict()
    assert [message["role"] for message in first["messages"]] == ["user", "assistant"]
    assert first["messages"][1]["content"] == "KEEP: answer A\n"
    assert "system" not in json.dumps(first)
    expected_user = (
        "Use the context.\n\n---\n\n"
        + documents["context-b.md"]
        + "\n\n---\n\n"
        + documents["context-a.md"]
    )
    assert first["messages"][0]["content"] == expected_user
    assert tuple(first["context_item_ids"]) == (ids["context-b.md"], ids["context-a.md"])
    assert prepared.rows[0].split == prepared.rows[1].split
    assert prepared.rows[-1].messages[0]["content"].startswith("Consider the material.")

    manifest_text = prepared.manifest_raw.decode("utf-8")
    assert str(tmp_path) not in manifest_text
    assert "Use the context." not in manifest_text
    assert "Earlier whole document" not in manifest_text
    assert "answer A" not in manifest_text
    assert prepared.manifest["lineage_digest"]
    assert prepared.manifest["lineage"][0]["context_item_ids"] == [
        ids["context-b.md"],
        ids["context-a.md"],
    ]


def test_context_order_and_prompt_template_bind_row_identity(tmp_path: Path) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    baseline = build_prepared_dataset_v2(bundle, config)
    changed_package = replace(
        config.packages[0], context_item_ids=tuple(reversed(config.packages[0].context_item_ids))
    )
    reordered = build_prepared_dataset_v2(
        bundle, replace(config, packages=(changed_package, *config.packages[1:]))
    )
    changed_variant = replace(config.prompt_variants[0], prompt="Apply the context.")
    reprompted = build_prepared_dataset_v2(
        bundle, replace(config, prompt_variants=(changed_variant, config.prompt_variants[1]))
    )
    assert baseline.rows[0].row_id != reordered.rows[0].row_id
    assert baseline.rows[0].row_id != reprompted.rows[0].row_id


def test_rejects_target_in_own_context(tmp_path: Path) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    package = config.packages[0]
    with pytest.raises(DatasetPrepValidationError, match="own context"):
        replace(package, context_item_ids=(package.target_item_id,))


def test_rejects_alternate_revision_context(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(entries[context_index], revision_family="target-a")
    with pytest.raises(DatasetPrepValidationError, match="alternate revision"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_transitive_ancestor_in_target_revision_family(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["context-b.md"],
        revision_family="target-a",
        sequence=2,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=3,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="alternate revision"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_context_derived_from_target(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(
        entries[context_index], sequence=11, derived_from=(ids["target-a.md"],)
    )
    with pytest.raises(DatasetPrepValidationError, match="derived from the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_future_context(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    context_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["context-a.md"])
    entries[context_index] = replace(entries[context_index], sequence=99)
    with pytest.raises(DatasetPrepValidationError, match="newer than the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def test_rejects_lineage_child_that_predates_its_source(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["context-b.md"],
        sequence=99,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=1,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="causal sequence"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_transitive_future_ancestor_closure(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    lineage = _lineage_with(
        config.lineage,
        ids["target-a.md"],
        sequence=10,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-b.md"],
        sequence=11,
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=12,
        derived_from=(ids["context-b.md"],),
    )
    with pytest.raises(DatasetPrepValidationError, match="newer than the target"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_rejects_cross_group_derivative_targets(tmp_path: Path) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    entries = list(config.lineage)
    target_index = next(index for index, entry in enumerate(entries) if entry.item_id == ids["target-b.md"])
    entries[target_index] = replace(entries[target_index], group_id="different-group")
    with pytest.raises(DatasetPrepValidationError, match="derivative targets"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=tuple(entries)))


def _lineage_with(entries, item_id: str, **changes):
    return tuple(
        replace(entry, **changes) if entry.item_id == item_id else entry
        for entry in entries
    )


def _context_derivative_config(config, ids, *, transitive: bool, ancestor: str):
    lineage = _lineage_with(
        config.lineage,
        ids["target-b.md"],
        sequence=1,
        derived_from=(),
        group_id="group-b",
    )
    lineage = _lineage_with(
        lineage,
        ids["context-b.md"],
        sequence=2,
        derived_from=(ids[ancestor],),
    )
    lineage = _lineage_with(
        lineage,
        ids["context-a.md"],
        sequence=3,
        derived_from=(ids["context-b.md"],) if transitive else (ids[ancestor],),
    )
    packages = tuple(
        replace(
            package,
            context_item_ids=(
                (ids["context-a.md"],)
                if package.target_item_id == ids["target-a.md"]
                else (ids["context-root.md"],)
            ),
        )
        for package in config.packages
    )
    return replace(
        config,
        lineage=lineage,
        packages=packages,
        split=replace(config.split, seed="held-out-probe-4"),
    )


@pytest.mark.parametrize("transitive", [False, True])
def test_rejects_context_descending_from_target_in_other_split(
    tmp_path: Path, transitive: bool
) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    leaked = _context_derivative_config(
        config,
        ids,
        transitive=transitive,
        ancestor="target-b.md",
    )
    baseline_lineage = _lineage_with(
        leaked.lineage, ids["context-a.md"], derived_from=()
    )
    baseline_lineage = _lineage_with(
        baseline_lineage, ids["context-b.md"], derived_from=()
    )
    baseline = build_prepared_dataset_v2(
        bundle, replace(leaked, lineage=baseline_lineage)
    )
    baseline_rows = {row.target_item_id: row for row in baseline.rows}
    assert baseline_rows[ids["target-a.md"]].split == "train"
    assert baseline_rows[ids["target-b.md"]].split == "validation"
    with pytest.raises(DatasetPrepValidationError, match="different split"):
        build_prepared_dataset_v2(bundle, leaked)


def test_allows_earlier_derived_context_when_ancestor_target_is_split_safe(
    tmp_path: Path,
) -> None:
    bundle, ids, _documents, config = _config(tmp_path)
    safe = _context_derivative_config(
        config,
        ids,
        transitive=False,
        ancestor="target-c.md",
    )
    lineage = _lineage_with(
        safe.lineage,
        ids["target-c.md"],
        sequence=1,
        derived_from=(),
    )
    # context-b is the direct derivative and context-a remains an independently
    # configured derivative of the same target; both precede target A.
    safe = replace(safe, lineage=lineage)
    prepared = build_prepared_dataset_v2(bundle, safe)
    rows = {row.target_item_id: row for row in prepared.rows}
    assert rows[ids["target-a.md"]].split == "train"
    assert rows[ids["target-c.md"]].split == "train"


@pytest.mark.parametrize("kind", ["duplicate", "unresolved"])
def test_rejects_duplicate_or_unresolved_references(tmp_path: Path, kind: str) -> None:
    bundle, _ids, _documents, config = _config(tmp_path)
    if kind == "duplicate":
        bad = replace(config, lineage=(*config.lineage, config.lineage[0]))
        message = "unique"
    else:
        bad_package = replace(config.packages[0], context_item_ids=("item-" + "f" * 64,))
        bad = replace(config, packages=(bad_package, *config.packages[1:]))
        message = "unresolved"
    with pytest.raises(DatasetPrepValidationError, match=message):
        build_prepared_dataset_v2(bundle, bad)
