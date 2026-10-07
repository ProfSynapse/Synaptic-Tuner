"""Explicit retrospective-support policy and public publication checks."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tuner.dataset_prep import DatasetPrepConfigV2, DatasetPrepValidationError, build_prepared_dataset_v2
from tuner.dataset_prep import publication
from tests.dataset_prep.test_context_messages_v2 import _config, _tail_config


def _retrospective_config(config, ids):
    config = _tail_config(config, ids)
    support = ids["context-b.md"]
    outline = ids["context-a.md"]
    target = ids["target-a.md"]
    heldout = ids["target-e.md"]
    lineage = tuple(
        replace(entry, group_id="series", sequence=20,
                derived_from=(target, heldout)) if entry.item_id == support else
        replace(entry, group_id="series", sequence=10,
                derived_from=(target,)) if entry.item_id == outline else entry
        for entry in config.lineage
    )
    packages = tuple(
        replace(package, context_item_ids=(support, outline))
        if package.target_item_id == target else
        replace(package, context_item_ids=(ids["context-root.md"],))
        for package in config.packages
    )
    return replace(config, lineage=lineage, packages=packages,
                   context_projection=config.target_projection,
                   conditioning_policy="declared_retrospective_support/v1",
                   retrospective_support_item_ids=(support,))


def test_retrospective_support_roundtrip_and_own_outline(tmp_path):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _retrospective_config(legacy, ids)
    assert DatasetPrepConfigV2.from_dict(config.to_dict()) == config
    prepared = build_prepared_dataset_v2(bundle, config)
    assert prepared.rows[0].context_item_ids == (ids["context-b.md"], ids["context-a.md"])
    assert prepared.rows[0].split == "train"
    assert prepared.rows[-1].split == "validation"
    assert prepared.manifest["recipe"]["context_package"]["conditioning_policy"] == {
        "kind": "declared_retrospective_support/v1",
        "support_item_ids": [ids["context-b.md"]],
    }
    retained = tmp_path / prepared.identity.dataset_id
    retained.mkdir(mode=0o700)
    publication._write_member(retained / "manifest.json", prepared.manifest_raw)
    publication._write_member(retained / "dataset.jsonl", prepared.dataset_raw)
    assert publication.verify_prepared_dataset_v2(retained).semantic_identity == prepared.identity
    with pytest.raises(DatasetPrepValidationError):
        build_prepared_dataset_v2(bundle, replace(config, retrospective_support_item_ids=(ids["context-a.md"],)))


@pytest.mark.parametrize("bad", ["target", "revision", "unused", "unsorted", "duplicate"])
def test_retrospective_support_selection_rejects_invalid_ids(tmp_path, bad):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _retrospective_config(legacy, ids)
    if bad == "target":
        config = replace(config, retrospective_support_item_ids=(ids["target-e.md"],))
    elif bad == "revision":
        lineage = tuple(replace(entry, revision_family="target-e")
                        if entry.item_id == ids["context-b.md"] else entry for entry in config.lineage)
        config = replace(config, lineage=lineage)
    elif bad == "unused":
        config = replace(config, retrospective_support_item_ids=(ids["context-root.md"],))
        config = replace(config, packages=tuple(replace(p, context_item_ids=(ids["context-b.md"],))
                                                for p in config.packages))
    else:
        raw = config.to_dict()
        values = [ids["context-b.md"], ids["context-a.md"]]
        raw["context_package"]["conditioning_policy"]["support_item_ids"] = (
            sorted(values, reverse=True) if bad == "unsorted" else [values[0], values[0]]
        )
        with pytest.raises(DatasetPrepValidationError):
            DatasetPrepConfigV2.from_dict(raw)
        return
    with pytest.raises(DatasetPrepValidationError):
        build_prepared_dataset_v2(bundle, config)


def test_retrospective_support_missing_target_rejects_closed(tmp_path):
    bundle, _, _, legacy = _config(tmp_path)
    ids = {item.logical_path: item.item_id for item in bundle.items}
    config = _retrospective_config(legacy, ids)
    missing = "item-" + "f" * 64
    config = replace(config, packages=(replace(config.packages[0], target_item_id=missing),
                                       *config.packages[1:]))
    with pytest.raises(DatasetPrepValidationError, match="unresolved"):
        build_prepared_dataset_v2(bundle, config)


def test_retrospective_support_rehashed_policy_tamper_rejected_publicly(tmp_path):
    bundle, ids, _, legacy = _config(tmp_path)
    prepared = build_prepared_dataset_v2(bundle, _retrospective_config(legacy, ids))
    manifest = json.loads(prepared.manifest_raw)
    manifest["recipe"]["context_package"]["conditioning_policy"]["support_item_ids"] = [ids["context-a.md"]]
    basis = {
        "source_bundle_digest": manifest["source"]["bundle_digest"],
        "source_structure_set_digest": manifest["source"]["structure_set_digest"],
        **{name: manifest[name] for name in (
            "projection", "projection_digest", "recipe", "lineage", "lineage_digest",
            "group_count", "group_ids_sha256", "row_count", "dataset_bytes",
            "dataset_sha256", "row_ids_sha256", "split_counts", "split_lineage",
            "context_projection", "context_projection_digest",
        )},
    }
    manifest["dataset_digest"] = publication._domain_digest(publication._DATASET_DOMAIN_V2, basis)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    with pytest.raises(DatasetPrepValidationError):
        publication._verify_bytes_v2(
            tmp_path / manifest["dataset_id"],
            publication._canonical_bytes(manifest), prepared.dataset_raw,
        )


def test_retrospective_policy_requires_exact_kind_and_bounded_selection(tmp_path):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _retrospective_config(legacy, ids)

    class KindSubclass(str):
        pass

    class EqualKind:
        def __eq__(self, other):
            return other == "declared_retrospective_support/v1"

    for value in (KindSubclass("declared_retrospective_support/v1"), EqualKind()):
        with pytest.raises(DatasetPrepValidationError):
            replace(config, conditioning_policy=value)
        raw = config.to_dict()
        raw["context_package"]["conditioning_policy"]["kind"] = value
        with pytest.raises(DatasetPrepValidationError):
            DatasetPrepConfigV2.from_dict(raw)
    for values in ([], [ids["context-b.md"]] * 257, ["item-" + "f" * 64]):
        raw = config.to_dict()
        raw["context_package"]["conditioning_policy"]["support_item_ids"] = values
        if values and len(values) == 1:
            # A well-formed but unresolved ID passes shape parsing, not builder admission.
            parsed = DatasetPrepConfigV2.from_dict(raw)
            with pytest.raises(DatasetPrepValidationError):
                build_prepared_dataset_v2(bundle, parsed)
        else:
            with pytest.raises(DatasetPrepValidationError):
                DatasetPrepConfigV2.from_dict(raw)


def test_selected_support_does_not_exempt_unlisted_descendant(tmp_path):
    bundle, ids, _, legacy = _config(tmp_path)
    config = _retrospective_config(legacy, ids)
    support = ids["context-b.md"]
    descendant = ids["context-a.md"]
    lineage = tuple(replace(entry, sequence=21, derived_from=(support,))
                    if entry.item_id == descendant else entry for entry in config.lineage)
    with pytest.raises(DatasetPrepValidationError, match="newer|held-out"):
        build_prepared_dataset_v2(bundle, replace(config, lineage=lineage))


def test_retrospective_support_hash_split_roundtrip(tmp_path):
    bundle, ids, _, config = _config(tmp_path)
    support = ids["context-b.md"]
    target_a = ids["target-a.md"]
    target_b = ids["target-b.md"]
    lineage = tuple(replace(entry, group_id="group-ab", sequence=20,
                            derived_from=(target_a, target_b))
                    if entry.item_id == support else entry for entry in config.lineage)
    packages = tuple(replace(package, context_item_ids=(support,))
                     if package.target_item_id == target_a else
                     replace(package, context_item_ids=(ids["context-root.md"],))
                     for package in config.packages)
    config = replace(config, lineage=lineage, packages=packages,
                     conditioning_policy="declared_retrospective_support/v1",
                     retrospective_support_item_ids=(support,))
    prepared = build_prepared_dataset_v2(bundle, config)
    assert "conditioning_lineage" in prepared.manifest
    assert "split_lineage" not in prepared.manifest
    assert publication._verify_bytes_v2(
        tmp_path / prepared.identity.dataset_id, prepared.manifest_raw,
        prepared.dataset_raw,
    ).semantic_identity == prepared.identity


@pytest.mark.parametrize("mutation", ["group", "target_revision", "unlisted_heldout"])
def test_retrospective_support_rehashed_lineage_tamper_rejected_publicly(tmp_path, mutation):
    bundle, ids, _, legacy = _config(tmp_path)
    prepared = build_prepared_dataset_v2(bundle, _retrospective_config(legacy, ids))
    manifest = json.loads(prepared.manifest_raw)
    for entry in manifest["split_lineage"]:
        if entry["item_id"] == ids["context-b.md"]:
            if mutation == "group":
                entry["group_id"] = "other-series"
            elif mutation == "target_revision":
                entry["revision_family"] = "target-e"
        elif entry["item_id"] == ids["context-a.md"] and mutation == "unlisted_heldout":
            entry["derived_from"].append(ids["target-e.md"])
            entry["sequence"] = 20
    manifest["recipe"]["context_package"]["lineage_digest"] = publication._domain_digest(
        publication._LINEAGE_DOMAIN_V2, manifest["split_lineage"],
    )
    basis = {
        "source_bundle_digest": manifest["source"]["bundle_digest"],
        "source_structure_set_digest": manifest["source"]["structure_set_digest"],
        **{name: manifest[name] for name in (
            "projection", "projection_digest", "recipe", "lineage", "lineage_digest",
            "group_count", "group_ids_sha256", "row_count", "dataset_bytes",
            "dataset_sha256", "row_ids_sha256", "split_counts", "split_lineage",
            "context_projection", "context_projection_digest",
        )},
    }
    manifest["dataset_digest"] = publication._domain_digest(publication._DATASET_DOMAIN_V2, basis)
    manifest["dataset_id"] = "dataset-" + manifest["dataset_digest"]
    with pytest.raises(DatasetPrepValidationError):
        publication._verify_bytes_v2(
            tmp_path / manifest["dataset_id"],
            publication._canonical_bytes(manifest), prepared.dataset_raw,
        )
