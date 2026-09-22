"""Immutable public models for deterministic dataset preparation v1."""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType


CONFIG_SCHEMA_VERSION = "syntunia-dataset-prep/v1"
ROW_SCHEMA_VERSION = "syntunia-sft-row/v1"
ARTIFACT_SCHEMA_VERSION = "syntunia-prepared-dataset/v1"
RAW_TEXT_FORMAT = "raw_text"
MAX_SEED_BYTES = 256
MAX_SPLITS = 32
MAX_SPLIT_WEIGHT = 2**31 - 1
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
_PATH_TYPE = type(Path())


class DatasetPrepValidationError(ValueError):
    """Configuration, source, or prepared artifact is invalid."""


class DatasetPrepCollisionError(RuntimeError):
    """The content-addressed destination exists with different bytes."""


class DatasetPrepDurabilityError(RuntimeError):
    """Publication-root durability could not be established."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()


class DatasetPublicationUncertaintyPhaseV1(str, Enum):
    PARENT_DURABILITY = "parent_durability"
    FINAL_VERIFICATION = "final_verification"
    STAGE_CLEANUP = "stage_cleanup"


def _invalid(message: str) -> None:
    raise DatasetPrepValidationError(message) from None


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _invalid(f"{name} must be a lowercase SHA-256 digest")
    return value


def _name(value: object, name: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        _invalid(f"{name} is invalid")
    return value


def _seed(value: object, name: str) -> str:
    if type(value) is not str or not value:
        _invalid(f"{name} is invalid")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        _invalid(f"{name} is invalid")
    if size > MAX_SEED_BYTES or any(ord(char) < 32 or ord(char) == 127 for char in value):
        _invalid(f"{name} is invalid")
    return value


def _version(value: object) -> str:
    if (
        type(value) is not str
        or _VERSION.fullmatch(value) is None
        or ".." in value
        or value.endswith(".")
    ):
        _invalid("structure version is invalid")
    return value


def _exact(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _invalid(f"{name} has missing or unknown fields")
    return value


@dataclass(frozen=True, slots=True)
class StructureRefV1:
    name: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "structure name"))
        object.__setattr__(self, "version", _version(self.version))
        object.__setattr__(self, "digest", _digest(self.digest, "structure digest"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> "StructureRefV1":
        item = _exact(value, frozenset({"name", "version", "digest"}), "structure_ref")
        return cls(item["name"], item["version"], item["digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProjectionV1:
    structure_ref: StructureRefV1
    name: str

    def __post_init__(self) -> None:
        if type(self.structure_ref) is not StructureRefV1:
            raise TypeError("structure_ref must be exact StructureRefV1")
        object.__setattr__(self, "name", _name(self.name, "projection name"))

    def to_dict(self) -> dict[str, object]:
        return {"structure_ref": self.structure_ref.to_dict(), "name": self.name}

    @classmethod
    def from_dict(cls, value: object) -> "ProjectionV1":
        item = _exact(value, frozenset({"structure_ref", "name"}), "projection")
        return cls(StructureRefV1.from_dict(item["structure_ref"]), item["name"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class OrderingV1:
    kind: str
    seed: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"source_order", "seeded_hash"}:
            _invalid("ordering kind is unsupported")
        if self.kind == "source_order":
            if self.seed is not None:
                _invalid("source_order must not define a seed")
        else:
            object.__setattr__(self, "seed", _seed(self.seed, "ordering seed"))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind} if self.seed is None else {"kind": self.kind, "seed": self.seed}

    @classmethod
    def from_dict(cls, value: object) -> "OrderingV1":
        if type(value) is not dict or "kind" not in value:
            _invalid("ordering is invalid")
        expected = frozenset({"kind"}) if value.get("kind") == "source_order" else frozenset({"kind", "seed"})
        item = _exact(value, expected, "ordering")
        return cls(item["kind"], item.get("seed"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SplitAllocationV1:
    name: str
    weight: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "split name"))
        if type(self.weight) is not int or not 1 <= self.weight <= MAX_SPLIT_WEIGHT:
            _invalid("split weight is invalid")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "weight": self.weight}

    @classmethod
    def from_dict(cls, value: object) -> "SplitAllocationV1":
        item = _exact(value, frozenset({"name", "weight"}), "split allocation")
        return cls(item["name"], item["weight"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SplitV1:
    kind: str
    seed: str | None = None
    allocations: tuple[SplitAllocationV1, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"none", "hash_rank"}:
            _invalid("split kind is unsupported")
        if type(self.allocations) is not tuple or any(type(item) is not SplitAllocationV1 for item in self.allocations):
            raise TypeError("allocations must contain exact SplitAllocationV1 values")
        if self.kind == "none":
            if self.seed is not None or self.allocations:
                _invalid("none split must not define seed or allocations")
            return
        object.__setattr__(self, "seed", _seed(self.seed, "split seed"))
        if not 2 <= len(self.allocations) <= MAX_SPLITS:
            _invalid("hash_rank requires two to 32 allocations")
        names = tuple(item.name for item in self.allocations)
        if names != ("train", "validation"):
            _invalid("hash_rank allocations must be train then validation")

    def to_dict(self) -> dict[str, object]:
        if self.kind == "none":
            return {"kind": "none"}
        return {
            "kind": self.kind,
            "seed": self.seed,
            "allocations": [item.to_dict() for item in self.allocations],
        }

    @classmethod
    def from_dict(cls, value: object) -> "SplitV1":
        if type(value) is not dict or "kind" not in value:
            _invalid("split is invalid")
        if value.get("kind") == "none":
            item = _exact(value, frozenset({"kind"}), "split")
            return cls(item["kind"])  # type: ignore[arg-type]
        item = _exact(value, frozenset({"kind", "seed", "allocations"}), "split")
        raw = item["allocations"]
        if type(raw) is not list:
            _invalid("split allocations must be a list")
        return cls(
            item["kind"],  # type: ignore[arg-type]
            item["seed"],  # type: ignore[arg-type]
            tuple(SplitAllocationV1.from_dict(entry) for entry in raw),
        )


@dataclass(frozen=True, slots=True)
class DatasetPrepConfigV1:
    source_bundle_path: Path
    expected_bundle_digest: str
    projection: ProjectionV1
    ordering: OrderingV1
    split: SplitV1
    schema_version: str = CONFIG_SCHEMA_VERSION
    format: str = RAW_TEXT_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            _invalid("dataset prep schema version is unsupported")
        if self.format != RAW_TEXT_FORMAT:
            _invalid("dataset format is unsupported")
        if type(self.source_bundle_path) is not _PATH_TYPE:
            raise TypeError("source_bundle_path must be a Path")
        absolute = Path(os.path.abspath(self.source_bundle_path))
        object.__setattr__(self, "source_bundle_path", absolute)
        object.__setattr__(self, "expected_bundle_digest", _digest(self.expected_bundle_digest, "expected bundle digest"))
        if type(self.projection) is not ProjectionV1 or type(self.ordering) is not OrderingV1 or type(self.split) is not SplitV1:
            raise TypeError("projection, ordering, and split must use exact v1 models")

    @classmethod
    def from_dict(cls, value: object) -> "DatasetPrepConfigV1":
        item = _exact(
            value,
            frozenset({"schema_version", "source", "format", "projection", "ordering", "split"}),
            "dataset prep config",
        )
        source = _exact(item["source"], frozenset({"bundle_path", "expected_bundle_digest"}), "source")
        bundle_path = source["bundle_path"]
        if type(bundle_path) is not str or not bundle_path:
            _invalid("source bundle_path is invalid")
        return cls(
            source_bundle_path=Path(bundle_path),
            expected_bundle_digest=source["expected_bundle_digest"],  # type: ignore[arg-type]
            projection=ProjectionV1.from_dict(item["projection"]),
            ordering=OrderingV1.from_dict(item["ordering"]),
            split=SplitV1.from_dict(item["split"]),
            schema_version=item["schema_version"],  # type: ignore[arg-type]
            format=item["format"],  # type: ignore[arg-type]
        )

    def semantic_recipe(self) -> dict[str, object]:
        ordering = self.ordering.to_dict()
        if self.ordering.seed is not None:
            ordering = {
                "kind": self.ordering.kind,
                "seed_sha256": hashlib.sha256(
                    b"syntunia-dataset-seed/v1\0" + self.ordering.seed.encode("utf-8")
                ).hexdigest(),
            }
        split = self.split.to_dict()
        if self.split.seed is not None:
            split = {
                "kind": self.split.kind,
                "seed_sha256": hashlib.sha256(
                    b"syntunia-dataset-seed/v1\0" + self.split.seed.encode("utf-8")
                ).hexdigest(),
                "allocations": [item.to_dict() for item in self.split.allocations],
            }
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "projection": self.projection.to_dict(),
            "ordering": ordering,
            "split": split,
        }


@dataclass(frozen=True, slots=True)
class SftRawTextRowV1:
    row_id: str
    source_item_id: str
    split: str
    text: str
    schema_version: str = ROW_SCHEMA_VERSION
    format: str = RAW_TEXT_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != ROW_SCHEMA_VERSION or self.format != RAW_TEXT_FORMAT:
            _invalid("row version or format is unsupported")
        if type(self.row_id) is not str or re.fullmatch(r"row-[0-9a-f]{64}", self.row_id) is None:
            _invalid("row_id is invalid")
        if type(self.source_item_id) is not str or re.fullmatch(r"item-[0-9a-f]{64}", self.source_item_id) is None:
            _invalid("source_item_id is invalid")
        object.__setattr__(self, "split", _name(self.split, "split"))
        if type(self.text) is not str or not self.text:
            _invalid("row text is invalid")
        try:
            self.text.encode("utf-8")
        except UnicodeError:
            _invalid("row text is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "row_id": self.row_id,
            "source_item_id": self.source_item_id,
            "split": self.split,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class DatasetSemanticIdentityV1:
    dataset_id: str
    dataset_digest: str
    row_count: int
    dataset_bytes: int
    dataset_sha256: str
    row_ids_sha256: str
    split_counts: MappingProxyType

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_digest", _digest(self.dataset_digest, "dataset digest"))
        if type(self.dataset_id) is not str or self.dataset_id != "dataset-" + self.dataset_digest:
            _invalid("dataset_id does not bind dataset_digest")
        if type(self.row_count) is not int or not 1 <= self.row_count <= 100_000:
            _invalid("row_count is invalid")
        if type(self.dataset_bytes) is not int or not 1 <= self.dataset_bytes <= 64 * 1024 * 1024:
            _invalid("dataset_bytes is invalid")
        object.__setattr__(self, "dataset_sha256", _digest(self.dataset_sha256, "dataset SHA-256"))
        object.__setattr__(self, "row_ids_sha256", _digest(self.row_ids_sha256, "row IDs SHA-256"))
        if type(self.split_counts) is not MappingProxyType or not self.split_counts:
            _invalid("split_counts is invalid")
        total = 0
        for name, count in self.split_counts.items():
            _name(name, "split")
            if type(count) is not int or not 0 <= count <= self.row_count:
                _invalid("split count is invalid")
            total += count
        if total != self.row_count:
            _invalid("split counts do not match row_count")


@dataclass(frozen=True, slots=True)
class VerifiedPreparedDatasetV1:
    path: Path
    semantic_identity: DatasetSemanticIdentityV1

    def __post_init__(self) -> None:
        if type(self.path) is not _PATH_TYPE:
            raise TypeError("path must be a Path")
        if type(self.semantic_identity) is not DatasetSemanticIdentityV1:
            raise TypeError("semantic_identity must be exact DatasetSemanticIdentityV1")


class DatasetPublicationUncertainV1(RuntimeError):
    __slots__ = ("semantic_identity", "phase")

    def __init__(self, semantic_identity: DatasetSemanticIdentityV1, phase: DatasetPublicationUncertaintyPhaseV1) -> None:
        if type(semantic_identity) is not DatasetSemanticIdentityV1:
            raise TypeError("semantic_identity must be exact DatasetSemanticIdentityV1")
        if type(phase) is not DatasetPublicationUncertaintyPhaseV1:
            raise TypeError("phase must be exact DatasetPublicationUncertaintyPhaseV1")
        self.semantic_identity = semantic_identity
        self.phase = phase
        super().__init__()
