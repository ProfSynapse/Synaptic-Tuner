"""Deterministic context-package to two-message SFT datasets.

This module is additive to the strict raw-text v1 surface.  Corpus semantics
remain declarative: item selection, lineage, prompt text, separators, and the
small target-only transforms all come from the v2 configuration.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tuner.ingestion import LoadedNormalizedBundleV1

from .models import (
    DatasetPrepValidationError,
    DatasetSemanticIdentityV1,
    ProjectionV1,
    SplitAllocationV1,
    _digest,
    _exact,
    _name,
    _seed,
)
from .prepare import MAX_DATASET_BYTES, _canonical_bytes, _domain_digest, _stream_digest


CONFIG_SCHEMA_VERSION_V2 = "syntunia-dataset-prep/v2"
ROW_SCHEMA_VERSION_V2 = "syntunia-sft-row/v2"
ARTIFACT_SCHEMA_VERSION_V2 = "syntunia-prepared-dataset/v2"
MESSAGES_FORMAT = "messages"

_ITEM_ID = re.compile(r"^item-[0-9a-f]{64}$")
_LOGICAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_ROW_ID_DOMAIN_V2 = "syntunia-sft-row-id/v2"
_DATASET_DOMAIN_V2 = "syntunia-prepared-dataset/v2"
_PROJECTION_DOMAIN_V2 = "syntunia-dataset-projection/v2"
_PROMPT_DOMAIN_V2 = "syntunia-context-prompt/v2"
_TRANSFORM_DOMAIN_V2 = "syntunia-target-transforms/v2"
_LINEAGE_DOMAIN_V2 = "syntunia-context-lineage/v2"
_GROUP_ORDER_DOMAIN_V2 = "syntunia-dataset-group-split-rank/v2"


def _invalid(message: str) -> None:
    raise DatasetPrepValidationError(message) from None


def _item_id(value: object, name: str) -> str:
    if type(value) is not str or _ITEM_ID.fullmatch(value) is None:
        _invalid(f"{name} is invalid")
    return value


def _logical_id(value: object, name: str) -> str:
    if type(value) is not str or _LOGICAL_ID.fullmatch(value) is None:
        _invalid(f"{name} is invalid")
    return value


def _bounded_text(value: object, name: str, *, maximum: int, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _invalid(f"{name} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        _invalid(f"{name} is invalid")
    if len(encoded) > maximum or "\x00" in value:
        _invalid(f"{name} is invalid")
    return value


def _neutral_separator(value: object) -> str:
    separator = _bounded_text(value, "prompt separator", maximum=64)
    if any(char.isalnum() for char in separator):
        _invalid("prompt separator must be neutral punctuation or whitespace")
    return separator


@dataclass(frozen=True, slots=True)
class PromptVariantV2:
    name: str
    prompt: str
    separator: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "prompt variant name"))
        object.__setattr__(self, "prompt", _bounded_text(self.prompt, "prompt text", maximum=16_384))
        object.__setattr__(self, "separator", _neutral_separator(self.separator))

    @classmethod
    def from_dict(cls, value: object) -> "PromptVariantV2":
        item = _exact(value, frozenset({"name", "prompt", "separator"}), "prompt variant")
        return cls(item["name"], item["prompt"], item["separator"])  # type: ignore[arg-type]

    def hashed_recipe(self) -> dict[str, str]:
        return {
            "name": self.name,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "separator_sha256": hashlib.sha256(self.separator.encode("utf-8")).hexdigest(),
        }

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "prompt": self.prompt, "separator": self.separator}


@dataclass(frozen=True, slots=True)
class TargetTransformsV2:
    drop_fenced_block_info_strings: tuple[str, ...] = ()
    drop_standalone_line_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values, label, maximum in (
            (self.drop_fenced_block_info_strings, "fenced block info string", 256),
            (self.drop_standalone_line_prefixes, "standalone line prefix", 512),
        ):
            if type(values) is not tuple or not values:
                if type(values) is not tuple:
                    raise TypeError(f"{label} values must be a tuple")
                continue
            checked = tuple(_bounded_text(value, label, maximum=maximum) for value in values)
            if len(checked) != len(set(checked)):
                _invalid(f"duplicate {label} values are not allowed")
        if any("\n" in value or "\r" in value for value in self.drop_fenced_block_info_strings):
            _invalid("fenced block info strings must be single-line values")
        if any("\n" in value or "\r" in value for value in self.drop_standalone_line_prefixes):
            _invalid("standalone line prefixes must be single-line values")

    @classmethod
    def from_dict(cls, value: object) -> "TargetTransformsV2":
        item = _exact(
            value,
            frozenset({"drop_fenced_block_info_strings", "drop_standalone_line_prefixes"}),
            "target transforms",
        )
        fenced = item["drop_fenced_block_info_strings"]
        prefixes = item["drop_standalone_line_prefixes"]
        if type(fenced) is not list or type(prefixes) is not list:
            _invalid("target transform selections must be lists")
        return cls(tuple(fenced), tuple(prefixes))  # type: ignore[arg-type]

    def hashed_recipe(self) -> dict[str, object]:
        return {
            "kind": "configured_drop/v1",
            "selection_digest": _domain_digest(
                _TRANSFORM_DOMAIN_V2,
                {
                    "drop_fenced_block_info_strings": list(self.drop_fenced_block_info_strings),
                    "drop_standalone_line_prefixes": list(self.drop_standalone_line_prefixes),
                },
            ),
            "fenced_info_count": len(self.drop_fenced_block_info_strings),
            "line_prefix_count": len(self.drop_standalone_line_prefixes),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "drop_fenced_block_info_strings": list(self.drop_fenced_block_info_strings),
            "drop_standalone_line_prefixes": list(self.drop_standalone_line_prefixes),
        }


@dataclass(frozen=True, slots=True)
class ItemLineageV2:
    item_id: str
    revision_family: str
    sequence: int
    derived_from: tuple[str, ...]
    group_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _item_id(self.item_id, "lineage item_id"))
        object.__setattr__(self, "revision_family", _logical_id(self.revision_family, "revision family"))
        object.__setattr__(self, "group_id", _logical_id(self.group_id, "group_id"))
        if type(self.sequence) is not int or not 0 <= self.sequence <= 2**63 - 1:
            _invalid("lineage sequence is invalid")
        if type(self.derived_from) is not tuple:
            raise TypeError("derived_from must be a tuple")
        checked = tuple(_item_id(value, "derived_from reference") for value in self.derived_from)
        if self.item_id in checked or len(checked) != len(set(checked)):
            _invalid("lineage derived_from references are invalid")

    @classmethod
    def from_dict(cls, value: object) -> "ItemLineageV2":
        item = _exact(
            value,
            frozenset({"item_id", "revision_family", "sequence", "derived_from", "group_id"}),
            "item lineage",
        )
        derived = item["derived_from"]
        if type(derived) is not list:
            _invalid("derived_from must be a list")
        return cls(
            item["item_id"],  # type: ignore[arg-type]
            item["revision_family"],  # type: ignore[arg-type]
            item["sequence"],  # type: ignore[arg-type]
            tuple(derived),  # type: ignore[arg-type]
            item["group_id"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "revision_family": self.revision_family,
            "sequence": self.sequence,
            "derived_from": list(self.derived_from),
            "group_id": self.group_id,
        }


@dataclass(frozen=True, slots=True)
class ContextPackageV2:
    target_item_id: str
    context_item_ids: tuple[str, ...]
    prompt_variant: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_item_id", _item_id(self.target_item_id, "target item_id"))
        if type(self.context_item_ids) is not tuple or not self.context_item_ids:
            _invalid("context_item_ids must be a nonempty tuple")
        checked = tuple(_item_id(value, "context item_id") for value in self.context_item_ids)
        if self.target_item_id in checked:
            _invalid("target item cannot appear in its own context")
        if len(checked) != len(set(checked)):
            _invalid("context item references must be unique")
        object.__setattr__(self, "prompt_variant", _name(self.prompt_variant, "prompt variant"))

    @classmethod
    def from_dict(cls, value: object) -> "ContextPackageV2":
        item = _exact(
            value,
            frozenset({"target_item_id", "context_item_ids", "prompt_variant"}),
            "context package",
        )
        context = item["context_item_ids"]
        if type(context) is not list:
            _invalid("context_item_ids must be a list")
        return cls(item["target_item_id"], tuple(context), item["prompt_variant"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_item_id": self.target_item_id,
            "context_item_ids": list(self.context_item_ids),
            "prompt_variant": self.prompt_variant,
        }


@dataclass(frozen=True, slots=True)
class GroupSplitV2:
    seed: str
    allocations: tuple[SplitAllocationV1, ...]
    kind: str = "group_hash_rank"

    def __post_init__(self) -> None:
        if self.kind != "group_hash_rank":
            _invalid("v2 split kind must be group_hash_rank")
        object.__setattr__(self, "seed", _seed(self.seed, "split seed"))
        if type(self.allocations) is not tuple or len(self.allocations) != 2:
            _invalid("group_hash_rank requires train and validation allocations")
        if tuple(item.name for item in self.allocations) != ("train", "validation"):
            _invalid("group_hash_rank allocations must be train then validation")

    @classmethod
    def from_dict(cls, value: object) -> "GroupSplitV2":
        item = _exact(value, frozenset({"kind", "seed", "allocations"}), "split")
        allocations = item["allocations"]
        if type(allocations) is not list:
            _invalid("split allocations must be a list")
        return cls(
            item["seed"],  # type: ignore[arg-type]
            tuple(SplitAllocationV1.from_dict(entry) for entry in allocations),
            item["kind"],  # type: ignore[arg-type]
        )

    def hashed_recipe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "seed_sha256": hashlib.sha256(
                b"syntunia-dataset-seed/v2\0" + self.seed.encode("utf-8")
            ).hexdigest(),
            "allocations": [item.to_dict() for item in self.allocations],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "allocations": [item.to_dict() for item in self.allocations],
        }


@dataclass(frozen=True, slots=True)
class DatasetPrepConfigV2:
    source_bundle_path: Path
    expected_bundle_digest: str
    target_projection: ProjectionV1
    lineage: tuple[ItemLineageV2, ...]
    packages: tuple[ContextPackageV2, ...]
    prompt_variants: tuple[PromptVariantV2, ...]
    target_transforms: TargetTransformsV2
    split: GroupSplitV2
    schema_version: str = CONFIG_SCHEMA_VERSION_V2
    format: str = MESSAGES_FORMAT
    ordering: str = "configured_whole_documents"

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION_V2 or self.format != MESSAGES_FORMAT:
            _invalid("dataset prep v2 version or format is unsupported")
        if self.ordering != "configured_whole_documents":
            _invalid("context package ordering is unsupported")
        object.__setattr__(self, "source_bundle_path", Path(self.source_bundle_path).absolute())
        object.__setattr__(self, "expected_bundle_digest", _digest(self.expected_bundle_digest, "bundle digest"))
        if type(self.target_projection) is not ProjectionV1:
            raise TypeError("target_projection must be exact ProjectionV1")
        if not self.lineage or not self.packages or not self.prompt_variants:
            _invalid("lineage, packages, and prompt variants must be nonempty")
        if any(type(item) is not ItemLineageV2 for item in self.lineage):
            raise TypeError("lineage must contain exact ItemLineageV2 values")
        if any(type(item) is not ContextPackageV2 for item in self.packages):
            raise TypeError("packages must contain exact ContextPackageV2 values")
        if any(type(item) is not PromptVariantV2 for item in self.prompt_variants):
            raise TypeError("prompt_variants must contain exact PromptVariantV2 values")
        if type(self.target_transforms) is not TargetTransformsV2 or type(self.split) is not GroupSplitV2:
            raise TypeError("target_transforms and split must use exact v2 models")

    @classmethod
    def from_dict(cls, value: object) -> "DatasetPrepConfigV2":
        item = _exact(
            value,
            frozenset({"schema_version", "source", "format", "target_projection", "context_package", "split"}),
            "dataset prep v2 config",
        )
        source = _exact(item["source"], frozenset({"bundle_path", "expected_bundle_digest"}), "source")
        bundle_path = source["bundle_path"]
        if type(bundle_path) is not str or not bundle_path:
            _invalid("source bundle_path is invalid")
        package = _exact(
            item["context_package"],
            frozenset({"ordering", "lineage", "packages", "prompt_variants", "target_transforms"}),
            "context_package",
        )
        lineage = package["lineage"]
        packages = package["packages"]
        variants = package["prompt_variants"]
        if type(lineage) is not list or type(packages) is not list or type(variants) is not list:
            _invalid("context_package collections must be lists")
        ordering = _exact(package["ordering"], frozenset({"kind"}), "context package ordering")
        return cls(
            source_bundle_path=Path(bundle_path),
            expected_bundle_digest=source["expected_bundle_digest"],  # type: ignore[arg-type]
            target_projection=ProjectionV1.from_dict(item["target_projection"]),
            lineage=tuple(ItemLineageV2.from_dict(entry) for entry in lineage),
            packages=tuple(ContextPackageV2.from_dict(entry) for entry in packages),
            prompt_variants=tuple(PromptVariantV2.from_dict(entry) for entry in variants),
            target_transforms=TargetTransformsV2.from_dict(package["target_transforms"]),
            split=GroupSplitV2.from_dict(item["split"]),
            schema_version=item["schema_version"],  # type: ignore[arg-type]
            format=item["format"],  # type: ignore[arg-type]
            ordering=ordering["kind"],  # type: ignore[arg-type]
        )

    def semantic_recipe(self) -> dict[str, object]:
        lineage = [entry.to_dict() for entry in self.lineage]
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "target_projection": self.target_projection.to_dict(),
            "context_package": {
                "ordering": {"kind": self.ordering},
                "lineage_digest": _domain_digest(_LINEAGE_DOMAIN_V2, lineage),
                "lineage_count": len(lineage),
                "package_count": len(self.packages),
                "prompt_variants": [variant.hashed_recipe() for variant in self.prompt_variants],
                "target_transforms": self.target_transforms.hashed_recipe(),
            },
            "split": self.split.hashed_recipe(),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "bundle_path": str(self.source_bundle_path),
                "expected_bundle_digest": self.expected_bundle_digest,
            },
            "format": self.format,
            "target_projection": self.target_projection.to_dict(),
            "context_package": {
                "ordering": {"kind": self.ordering},
                "lineage": [entry.to_dict() for entry in self.lineage],
                "packages": [entry.to_dict() for entry in self.packages],
                "prompt_variants": [entry.to_dict() for entry in self.prompt_variants],
                "target_transforms": self.target_transforms.to_dict(),
            },
            "split": self.split.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SftMessagesRowV2:
    row_id: str
    target_item_id: str
    context_item_ids: tuple[str, ...]
    group_id: str
    split: str
    messages: tuple[MappingProxyType, MappingProxyType]
    schema_version: str = ROW_SCHEMA_VERSION_V2
    format: str = MESSAGES_FORMAT

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "row_id": self.row_id,
            "target_item_id": self.target_item_id,
            "context_item_ids": list(self.context_item_ids),
            "group_id": self.group_id,
            "split": self.split,
            "messages": [dict(message) for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class PreparedDatasetV2:
    identity: DatasetSemanticIdentityV1
    rows: tuple[SftMessagesRowV2, ...]
    dataset_raw: bytes
    manifest: MappingProxyType
    manifest_raw: bytes


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    return value


def _resolve_projection(bundle: LoadedNormalizedBundleV1, projection: ProjectionV1) -> tuple[str, dict[str, object], str]:
    selected_ref = projection.structure_ref.to_dict()
    structures = bundle.structure_set.get("structures")
    if type(structures) is not tuple:
        _invalid("verified structure set is invalid")
    matches = [entry for entry in structures if isinstance(entry, Mapping) and _plain(entry.get("ref")) == selected_ref]
    if len(matches) != 1:
        _invalid("configured structure does not resolve exactly once")
    projections = matches[0].get("text_projections")
    if type(projections) is not tuple:
        _invalid("configured structure projections are invalid")
    selected = [entry for entry in projections if isinstance(entry, Mapping) and entry.get("name") == projection.name]
    if len(selected) != 1 or type(selected[0].get("field_ref")) is not str or not selected[0]["field_ref"]:
        _invalid("configured target projection does not resolve exactly once")
    field_ref = selected[0]["field_ref"]
    identity = {"structure_ref": selected_ref, "name": projection.name, "field_ref": field_ref}
    return field_ref, identity, _domain_digest(_PROJECTION_DOMAIN_V2, identity)


def _drop_selected_fences(text: str, selected: frozenset[str]) -> str:
    if not selected:
        return text
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    dropping: tuple[str, int] | None = None
    for line in lines:
        probe = line.rstrip("\r\n")
        stripped = probe.lstrip(" ")
        indent = len(probe) - len(stripped)
        if dropping is None:
            match = re.fullmatch(r"(`{3,}|~{3,})([^\r\n]*)", stripped) if indent <= 3 else None
            if match is not None and match.group(2).strip() in selected:
                dropping = (match.group(1)[0], len(match.group(1)))
                continue
            kept.append(line)
            continue
        marker, minimum = dropping
        close = re.fullmatch(rf"{re.escape(marker)}{{{minimum},}}[ \t]*", stripped) if indent <= 3 else None
        if close is not None:
            dropping = None
    if dropping is not None:
        _invalid("selected fenced block is not closed")
    return "".join(kept)


def apply_target_transforms_v2(text: str, transforms: TargetTransformsV2) -> str:
    result = _drop_selected_fences(text, frozenset(transforms.drop_fenced_block_info_strings))
    if transforms.drop_standalone_line_prefixes:
        result = "".join(
            line
            for line in result.splitlines(keepends=True)
            if not any(line.rstrip("\r\n").startswith(prefix) for prefix in transforms.drop_standalone_line_prefixes)
        )
    if not result:
        _invalid("target transforms removed the entire target projection")
    return result


def _allocation_counts(allocations: tuple[SplitAllocationV1, ...], count: int) -> tuple[int, ...]:
    total = sum(entry.weight for entry in allocations)
    bases = [count * entry.weight // total for entry in allocations]
    remainders = [count * entry.weight % total for entry in allocations]
    for index in sorted(range(len(allocations)), key=lambda item: (-remainders[item], item))[: count - sum(bases)]:
        bases[index] += 1
    return tuple(bases)


def _validate_lineage(config: DatasetPrepConfigV2, available: frozenset[str]) -> dict[str, ItemLineageV2]:
    entries: dict[str, ItemLineageV2] = {}
    for entry in config.lineage:
        if entry.item_id in entries:
            _invalid("lineage item references must be unique")
        if entry.item_id not in available:
            _invalid("lineage item reference is unresolved")
        entries[entry.item_id] = entry
    for entry in entries.values():
        if any(parent not in entries for parent in entry.derived_from):
            _invalid("derived_from reference is unresolved")
        if any(entries[parent].sequence > entry.sequence for parent in entry.derived_from):
            _invalid("lineage derivation violates causal sequence ordering")

    state: dict[str, int] = {}
    def visit(item_id: str) -> None:
        marker = state.get(item_id, 0)
        if marker == 1:
            _invalid("lineage contains a derivation cycle")
        if marker == 2:
            return
        state[item_id] = 1
        for parent in entries[item_id].derived_from:
            visit(parent)
        state[item_id] = 2
    for item_id in entries:
        visit(item_id)
    return entries


def _ancestors(item_id: str, entries: Mapping[str, ItemLineageV2]) -> frozenset[str]:
    result: set[str] = set()
    pending = list(entries[item_id].derived_from)
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            pending.extend(entries[current].derived_from)
    return frozenset(result)


def build_prepared_dataset_v2(bundle: LoadedNormalizedBundleV1, config: DatasetPrepConfigV2) -> PreparedDatasetV2:
    if type(bundle) is not LoadedNormalizedBundleV1 or type(config) is not DatasetPrepConfigV2:
        raise TypeError("bundle and config must use exact v2 models")
    if bundle.semantic_identity.bundle_digest != config.expected_bundle_digest:
        _invalid("source bundle digest does not match expectation")
    field_ref, projection_identity, projection_digest = _resolve_projection(bundle, config.target_projection)
    items = {entry.item_id: entry for entry in bundle.items}
    lineage = _validate_lineage(config, frozenset(items))
    variants: dict[str, PromptVariantV2] = {}
    for variant in config.prompt_variants:
        if variant.name in variants:
            _invalid("prompt variant names must be unique")
        variants[variant.name] = variant

    targets = [package.target_item_id for package in config.packages]
    if len(targets) != len(set(targets)):
        _invalid("target item references must be unique")
    target_set = frozenset(targets)
    for package in config.packages:
        references = (package.target_item_id, *package.context_item_ids)
        if any(reference not in items or reference not in lineage for reference in references):
            _invalid("target or context item reference is unresolved")
        if package.prompt_variant not in variants:
            _invalid("prompt variant reference is unresolved")
        target_lineage = lineage[package.target_item_id]
        for context_id in package.context_item_ids:
            context_lineage = lineage[context_id]
            context_ancestors = _ancestors(context_id, lineage)
            context_closure = (context_id, *context_ancestors)
            if package.target_item_id in context_ancestors:
                _invalid("context is derived from the target")
            if any(
                lineage[member_id].revision_family == target_lineage.revision_family
                for member_id in context_closure
            ):
                _invalid("context contains an alternate revision of the target")
            if any(
                lineage[member_id].sequence > target_lineage.sequence
                for member_id in context_closure
            ):
                _invalid("context is newer than the target")
            if context_id in target_set and context_lineage.group_id != target_lineage.group_id:
                _invalid("a referenced target would cross group-safe splits")

    # Every derivation-connected target must share one group.  This is checked
    # before split assignment, so the implementation never silently falls back
    # to row-level random splitting.
    for target_id in target_set:
        target = lineage[target_id]
        related = set(target.derived_from) | set(_ancestors(target_id, lineage))
        related.update(candidate.item_id for candidate in lineage.values() if target_id in _ancestors(candidate.item_id, lineage))
        for other_id in related & target_set:
            if lineage[other_id].group_id != target.group_id:
                _invalid("derivative targets must share one group_id")

    group_ids = tuple(dict.fromkeys(lineage[target].group_id for target in targets))
    if len(group_ids) < 2:
        _invalid("group-safe train/validation splitting requires at least two groups")
    ranked_groups = sorted(
        group_ids,
        key=lambda group: (_domain_digest(_GROUP_ORDER_DOMAIN_V2, {"seed": config.split.seed, "group_id": group}), group),
    )
    group_sizes = _allocation_counts(config.split.allocations, len(ranked_groups))
    if any(size == 0 for size in group_sizes):
        _invalid("group-safe split allocations require a nonempty group in every split")
    group_splits: dict[str, str] = {}
    offset = 0
    for allocation, size in zip(config.split.allocations, group_sizes):
        for group_id in ranked_groups[offset : offset + size]:
            group_splits[group_id] = allocation.name
        offset += size

    target_splits = {
        target_id: group_splits[lineage[target_id].group_id]
        for target_id in target_set
    }
    for package in config.packages:
        package_split = target_splits[package.target_item_id]
        for context_id in package.context_item_ids:
            ancestor_targets = _ancestors(context_id, lineage) & target_set
            if any(target_splits[ancestor_id] != package_split for ancestor_id in ancestor_targets):
                _invalid("context descends from a target assigned to a different split")

    rows: list[SftMessagesRowV2] = []
    manifest_lineage: list[dict[str, object]] = []
    selected_ref = config.target_projection.structure_ref.to_dict()
    for package in config.packages:
        target_item = items[package.target_item_id]
        if _plain(target_item.structure_ref) != selected_ref:
            _invalid("target item does not match configured target projection structure")
        target = target_item.fields.get(field_ref)
        if type(target) is not str or not target:
            _invalid("target projection must be a nonempty string")
        assistant = apply_target_transforms_v2(target, config.target_transforms)
        context_texts: list[str] = []
        for context_id in package.context_item_ids:
            context_item = items[context_id]
            if _plain(context_item.structure_ref) != selected_ref:
                _invalid("context item does not match configured projection structure")
            text = context_item.fields.get(field_ref)
            if type(text) is not str or not text:
                _invalid("context projection must be a nonempty string")
            context_texts.append(text)
        variant = variants[package.prompt_variant]
        user = variant.prompt + variant.separator + variant.separator.join(context_texts)
        group_id = lineage[package.target_item_id].group_id
        hashes = {
            "prompt_template_sha256": hashlib.sha256(variant.prompt.encode("utf-8")).hexdigest(),
            "separator_sha256": hashlib.sha256(variant.separator.encode("utf-8")).hexdigest(),
            "rendered_user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "assistant_sha256": hashlib.sha256(assistant.encode("utf-8")).hexdigest(),
        }
        basis = {
            "source_bundle_digest": bundle.semantic_identity.bundle_digest,
            "target_item_id": package.target_item_id,
            "context_item_ids": list(package.context_item_ids),
            "target_projection": projection_identity,
            "prompt_variant": package.prompt_variant,
            **hashes,
            "group_id": group_id,
            "format": MESSAGES_FORMAT,
        }
        row_id = "row-" + _domain_digest(_ROW_ID_DOMAIN_V2, basis)
        messages = (
            MappingProxyType({"role": "user", "content": user}),
            MappingProxyType({"role": "assistant", "content": assistant}),
        )
        rows.append(
            SftMessagesRowV2(
                row_id,
                package.target_item_id,
                package.context_item_ids,
                group_id,
                group_splits[group_id],
                messages,
            )
        )
        manifest_lineage.append(
            {
                "row_id": row_id,
                "target_item_id": package.target_item_id,
                "context_item_ids": list(package.context_item_ids),
                "group_id": group_id,
                "prompt_variant": package.prompt_variant,
                **hashes,
            }
        )

    chunks = tuple(_canonical_bytes(row.to_dict()) + b"\n" for row in rows)
    dataset_raw = b"".join(chunks)
    if not dataset_raw or len(dataset_raw) > MAX_DATASET_BYTES:
        _invalid("prepared dataset exceeds its byte limit")
    split_counts = {name: sum(row.split == name for row in rows) for name in ("train", "validation")}
    if any(count == 0 for count in split_counts.values()):
        _invalid("group-safe splits require nonempty train and validation rows")
    dataset_sha256 = hashlib.sha256(dataset_raw).hexdigest()
    row_ids_sha256 = _stream_digest(tuple(row.row_id for row in rows))
    lineage_digest = _domain_digest(_LINEAGE_DOMAIN_V2, manifest_lineage)
    group_ids_sha256 = _stream_digest(tuple(sorted(group_ids)))
    recipe = config.semantic_recipe()
    dataset_basis = {
        "source_bundle_digest": bundle.semantic_identity.bundle_digest,
        "source_structure_set_digest": bundle.semantic_identity.structure_set_digest,
        "projection": projection_identity,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "lineage": manifest_lineage,
        "lineage_digest": lineage_digest,
        "group_count": len(group_ids),
        "group_ids_sha256": group_ids_sha256,
        "row_count": len(rows),
        "dataset_bytes": len(dataset_raw),
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": split_counts,
    }
    dataset_digest = _domain_digest(_DATASET_DOMAIN_V2, dataset_basis)
    dataset_id = "dataset-" + dataset_digest
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION_V2,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "source": {
            "bundle_digest": bundle.semantic_identity.bundle_digest,
            "structure_set_digest": bundle.semantic_identity.structure_set_digest,
            "item_count": bundle.semantic_identity.item_count,
        },
        "format": MESSAGES_FORMAT,
        "row_schema_version": ROW_SCHEMA_VERSION_V2,
        "projection": projection_identity,
        "projection_digest": projection_digest,
        "recipe": recipe,
        "lineage": manifest_lineage,
        "lineage_digest": lineage_digest,
        "group_count": len(group_ids),
        "group_ids_sha256": group_ids_sha256,
        "row_count": len(rows),
        "dataset_bytes": len(dataset_raw),
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_sha256,
        "split_counts": split_counts,
    }
    manifest_raw = _canonical_bytes(manifest)
    identity = DatasetSemanticIdentityV1(
        dataset_id,
        dataset_digest,
        len(rows),
        len(dataset_raw),
        dataset_sha256,
        row_ids_sha256,
        MappingProxyType(split_counts),
    )
    return PreparedDatasetV2(identity, tuple(rows), dataset_raw, MappingProxyType(manifest), manifest_raw)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION_V2",
    "CONFIG_SCHEMA_VERSION_V2",
    "MESSAGES_FORMAT",
    "ROW_SCHEMA_VERSION_V2",
    "ContextPackageV2",
    "DatasetPrepConfigV2",
    "GroupSplitV2",
    "ItemLineageV2",
    "PreparedDatasetV2",
    "PromptVariantV2",
    "SftMessagesRowV2",
    "TargetTransformsV2",
    "apply_target_transforms_v2",
    "build_prepared_dataset_v2",
]
