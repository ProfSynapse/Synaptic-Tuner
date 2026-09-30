"""Pure deterministic transformation from a verified bundle to SFT rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from tuner.ingestion import LoadedNormalizedBundleV1

from .models import (
    ARTIFACT_SCHEMA_VERSION,
    RAW_TEXT_FORMAT,
    ROW_SCHEMA_VERSION,
    DatasetPrepConfigV1,
    DatasetPrepValidationError,
    DatasetSemanticIdentityV1,
    SftRawTextRowV1,
)


MAX_DATASET_BYTES = 64 * 1024 * 1024
_ROW_ID_DOMAIN = "syntunia-sft-row-id/v1"
_ORDER_DOMAIN = "syntunia-dataset-order/v1"
_SPLIT_DOMAIN = "syntunia-dataset-split-rank/v1"
_DATASET_DOMAIN = "syntunia-prepared-dataset/v1"
_PROJECTION_DOMAIN = "syntunia-dataset-projection/v1"


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise DatasetPrepValidationError("dataset semantic value is not canonical JSON") from None


def _domain_digest(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value)).hexdigest()


def _stream_digest(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    return value


def _resolve_projection(bundle: LoadedNormalizedBundleV1, config: DatasetPrepConfigV1) -> tuple[str, dict[str, object], str]:
    selected_ref = config.projection.structure_ref.to_dict()
    structure_set = bundle.structure_set
    structures = structure_set.get("structures")
    if type(structures) is not tuple:
        raise DatasetPrepValidationError("verified structure set is invalid") from None
    matches = [item for item in structures if isinstance(item, Mapping) and _plain(item.get("ref")) == selected_ref]
    if len(matches) != 1:
        raise DatasetPrepValidationError("configured structure does not resolve exactly once") from None
    projections = matches[0].get("text_projections")
    if type(projections) is not tuple:
        raise DatasetPrepValidationError("configured structure projections are invalid") from None
    projection_matches = [
        item
        for item in projections
        if isinstance(item, Mapping) and item.get("name") == config.projection.name
    ]
    if len(projection_matches) != 1:
        raise DatasetPrepValidationError("configured projection does not resolve exactly once") from None
    projection = projection_matches[0]
    field_ref = projection.get("field_ref")
    if type(field_ref) is not str or not field_ref:
        raise DatasetPrepValidationError("configured projection field is invalid") from None
    projection_identity = {
        "structure_ref": selected_ref,
        "name": config.projection.name,
        "field_ref": field_ref,
    }
    return field_ref, projection_identity, _domain_digest(_PROJECTION_DOMAIN, projection_identity)


def _rank(domain: str, seed: str, row_id: str) -> tuple[str, str]:
    return _domain_digest(domain, {"seed": seed, "row_id": row_id}), row_id


def _allocation_counts(config: DatasetPrepConfigV1, count: int) -> tuple[tuple[str, int], ...]:
    allocations = config.split.allocations
    total_weight = sum(item.weight for item in allocations)
    bases = [count * item.weight // total_weight for item in allocations]
    remainders = [count * item.weight % total_weight for item in allocations]
    remaining = count - sum(bases)
    winners = sorted(range(len(allocations)), key=lambda index: (-remainders[index], index))[:remaining]
    for index in winners:
        bases[index] += 1
    return tuple((item.name, bases[index]) for index, item in enumerate(allocations))


def _assign_splits(
    config: DatasetPrepConfigV1,
    rows: tuple[SftRawTextRowV1, ...],
) -> tuple[dict[str, str], dict[str, int]]:
    if config.split.kind == "none":
        return ({row.row_id: "train" for row in rows}, {"train": len(rows)})
    assert config.split.seed is not None
    ranked = sorted(rows, key=lambda row: _rank(_SPLIT_DOMAIN, config.split.seed or "", row.row_id))
    assignments: dict[str, str] = {}
    counts = _allocation_counts(config, len(rows))
    offset = 0
    for name, size in counts:
        for row in ranked[offset : offset + size]:
            assignments[row.row_id] = name
        offset += size
    if offset != len(rows) or len(assignments) != len(rows):
        raise DatasetPrepValidationError("split allocation did not cover every row") from None
    return assignments, dict(counts)


@dataclass(frozen=True, slots=True)
class PreparedDatasetV1:
    identity: DatasetSemanticIdentityV1
    rows: tuple[SftRawTextRowV1, ...]
    dataset_raw: bytes
    manifest: MappingProxyType
    manifest_raw: bytes


def build_prepared_dataset_v1(
    bundle: LoadedNormalizedBundleV1,
    config: DatasetPrepConfigV1,
) -> PreparedDatasetV1:
    """Build exact rows and a path-free semantic manifest without filesystem effects."""

    if type(bundle) is not LoadedNormalizedBundleV1:
        raise TypeError("bundle must be exact LoadedNormalizedBundleV1")
    if type(config) is not DatasetPrepConfigV1:
        raise TypeError("config must be exact DatasetPrepConfigV1")
    if bundle.semantic_identity.bundle_digest != config.expected_bundle_digest:
        raise DatasetPrepValidationError("source bundle digest does not match expectation") from None
    field_ref, projection_identity, projection_digest = _resolve_projection(bundle, config)
    selected_ref = config.projection.structure_ref.to_dict()

    rows_without_splits: list[SftRawTextRowV1] = []
    for item in bundle.items:
        if _plain(item.structure_ref) != selected_ref:
            raise DatasetPrepValidationError("source item does not match configured structure") from None
        text = item.fields.get(field_ref)
        if type(text) is not str or not text:
            raise DatasetPrepValidationError("projected text must be a nonempty string") from None
        try:
            text_bytes = text.encode("utf-8")
        except UnicodeError:
            raise DatasetPrepValidationError("projected text must be valid UTF-8") from None
        row_basis = {
            "source_bundle_digest": bundle.semantic_identity.bundle_digest,
            "source_item_id": item.item_id,
            "structure_ref": selected_ref,
            "projection": {"name": config.projection.name, "field_ref": field_ref},
            "projected_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "format": RAW_TEXT_FORMAT,
        }
        row_id = "row-" + _domain_digest(_ROW_ID_DOMAIN, row_basis)
        rows_without_splits.append(SftRawTextRowV1(row_id, item.item_id, "train", text))

    retained = tuple(rows_without_splits)
    assignments, split_counts = _assign_splits(config, retained)
    rows = tuple(
        SftRawTextRowV1(row.row_id, row.source_item_id, assignments[row.row_id], row.text)
        for row in retained
    )
    if config.ordering.kind == "seeded_hash":
        assert config.ordering.seed is not None
        rows = tuple(sorted(rows, key=lambda row: _rank(_ORDER_DOMAIN, config.ordering.seed or "", row.row_id)))

    chunks = tuple(_canonical_bytes(row.to_dict()) + b"\n" for row in rows)
    dataset_raw = b"".join(chunks)
    if not dataset_raw or len(dataset_raw) > MAX_DATASET_BYTES:
        raise DatasetPrepValidationError("prepared dataset exceeds its byte limit") from None
    dataset_sha256 = hashlib.sha256(dataset_raw).hexdigest()
    row_ids_sha256 = _stream_digest(tuple(row.row_id for row in rows))
    recipe = config.semantic_recipe()
    dataset_basis = {
        "source_bundle_digest": bundle.semantic_identity.bundle_digest,
        "source_structure_set_digest": bundle.semantic_identity.structure_set_digest,
        "projection": projection_identity,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "row_count": len(rows),
        "dataset_bytes": len(dataset_raw),
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": split_counts,
    }
    dataset_digest = _domain_digest(_DATASET_DOMAIN, dataset_basis)
    dataset_id = "dataset-" + dataset_digest
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "source": {
            "bundle_digest": bundle.semantic_identity.bundle_digest,
            "structure_set_digest": bundle.semantic_identity.structure_set_digest,
            "item_count": bundle.semantic_identity.item_count,
        },
        "format": RAW_TEXT_FORMAT,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "projection": projection_identity,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "row_count": len(rows),
        "dataset_bytes": len(dataset_raw),
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": split_counts,
    }
    manifest_raw = _canonical_bytes(manifest)
    identity = DatasetSemanticIdentityV1(
        dataset_id=dataset_id,
        dataset_digest=dataset_digest,
        row_count=len(rows),
        dataset_bytes=len(dataset_raw),
        dataset_sha256=dataset_sha256,
        row_ids_sha256=row_ids_sha256,
        split_counts=MappingProxyType(dict(split_counts)),
    )
    return PreparedDatasetV1(
        identity=identity,
        rows=rows,
        dataset_raw=dataset_raw,
        manifest=MappingProxyType(manifest),
        manifest_raw=manifest_raw,
    )
