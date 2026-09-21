"""Blob-free, provider-neutral Markdown source-ingestion contracts.

V1 admits an already-authorized local selection or finalized upload by opaque
reference.  It declares how Markdown documents with YAML frontmatter are
normalized, but performs no file IO, parsing, picker invocation, or upload
streaming.  Dataset recipes are a separate API concern.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from ._contract import canonical_bytes, contract_digest, digest_text, exact_fields, required_text
from ._timestamps import require_rfc3339
from .training_facade import AuthorizationRequirement


INGESTION_PLAN_SCHEMA_VERSION = "synaptic-ingestion-plan/v1"
INGESTION_RESULT_SCHEMA_VERSION = "synaptic-ingestion-result/v1"

# V1 parser semantics are public constants, not an implementation.
MARKDOWN_ENCODING = "utf-8"
MARKDOWN_STRIP_ONE_LEADING_BOM = True
MARKDOWN_NEWLINE_NORMALIZATION = "crlf_and_cr_to_lf"
MARKDOWN_FRONTMATTER_OPENER = "---\n"
MARKDOWN_FRONTMATTER_CLOSER = "---"
MARKDOWN_FRONTMATTER_CLOSE_RULE = "first_later_exact_line_closer_lf_or_eof"
MARKDOWN_BODY_START_RULE = "immediately_after_closer_following_lf"
FRONTMATTER_MODE_SEMANTICS = (
    "none_all_body", "optional_absent_all_body_unclosed_invalid",
    "required_absent_invalid_unclosed_invalid",
)
MARKDOWN_BODY_POLICY = "normalized_verbatim_nonempty"
YAML_PRESENTATION_SUBSET = "yaml_1_2_presentation_subset"
YAML_TOP_LEVEL = "exactly_one_mapping_document"
YAML_BOOLEAN_LITERALS = ("false", "true")
YAML_INTEGER_PATTERN = r"-?(0|[1-9][0-9]*)"
YAML_FLOAT_PATTERN = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
YAML_NUMBER_CLASSIFICATION = (
    "classify_integer_first", "float_requires_dot_or_exponent", "float_must_be_finite",
)
YAML_TIMESTAMP_LIKE_SCALARS = "string"
YAML_UNQUOTED_NULL_LITERALS_REJECTED = ("null", "Null", "NULL", "~")
YAML_ALLOWED_VALUES = ("boolean", "finite_number", "int64", "map", "sequence", "string")
YAML_REJECTED_FEATURES = (
    "alias", "anchor", "binary", "duplicate_key", "merge", "multidoc", "non_finite",
    "null", "object", "set", "tag", "typed_timestamp",
)
YAML_V2_UNQUOTED_NULL_LITERALS = ("", "null", "Null", "NULL", "~")
YAML_V2_ALLOWED_VALUES = YAML_ALLOWED_VALUES + ("null",)
YAML_KEY_RULE = "all_nested_keys_nonempty_nfc_string_1_to_64_utf8_bytes"
YAML_DUPLICATE_KEY_RULE = "reject_duplicate_keys_after_nfc_normalization"
YAML_LIMIT_ACCOUNTING = "global_across_entire_frontmatter_document"
METADATA_PRECEDENCE = ("frontmatter", "structure_policy", "binding_policy", "request_policy")
METADATA_MERGE = "recursive_object_merge_array_and_scalar_replace_no_delete"
GLOB_PATH_MODEL = "snapshot_relative_nfc_posix_segments"
GLOB_MATCH_MODEL = "whole_path_case_sensitive_unicode_codepoints_dotfiles_ordinary"
GLOB_DOUBLE_STAR = "complete_segment_matching_zero_or_more_path_segments"
GLOB_STAR = "zero_or_more_non_slash_codepoints_within_one_segment"
GLOB_QUESTION = "exactly_one_non_slash_codepoint_within_one_segment"
GLOB_DP_ALGORITHM = (
    "split_pattern_and_path_on_slash",
    "dp[0][0]=true",
    "dp[i][0]=dp[i-1][0] only when pattern_segment[i-1] is **",
    "for **: dp[i][j]=dp[i-1][j] or dp[i][j-1]",
    "otherwise dp[i][j]=dp[i-1][j-1] and segment_dp(pattern[i-1],path[j-1])",
    "segment_dp uses left-to-right DP where * consumes zero or more codepoints and ? consumes exactly one",
    "match iff dp[len(pattern_segments)][len(path_segments)]",
)
GLOB_EXAMPLES = (
    ("*.md", "note.md", True),
    ("*.md", ".note.md", True),
    ("**/*.md", "note.md", True),
    ("**/*.md", "books/one/note.md", True),
    ("books/?/note.md", "books/a/note.md", True),
    ("books/?/note.md", "books/ab/note.md", False),
)
GLOB_REJECTED_SYNTAX = ("backslash", "colon", "control", "empty_segment", "dot_segment", "dotdot_segment", "class", "brace", "extglob", "escape")

MAX_FRONTMATTER_BYTES = 65_536
MAX_YAML_DEPTH = 8
MAX_YAML_NODES = 1_024
MAX_YAML_MAPPING_ENTRIES = 128
MAX_YAML_SEQUENCE_ENTRIES = 256
MAX_YAML_KEY_BYTES = 64
MAX_YAML_SCALAR_BYTES = 4_096
MAX_SOURCE_COUNT = 100_000
MAX_STRUCTURES = 32
MAX_BINDINGS = 128
MAX_FIELDS = 64
MAX_TEXT_PROJECTIONS = 16
MAX_METADATA_DECLARATIONS = 64
MAX_RELATIONSHIPS = 32
MAX_PROPOSALS = 32
MAX_TAGS = 32
MAX_AUTHORIZATION_REQUIREMENTS = 16
MAX_DIAGNOSTICS = 32
MAX_GLOB_BYTES = 512
MAX_GLOB_SEGMENTS = 64
MAX_STRUCTURE_SET_BYTES = 1_048_576
MAX_INGESTION_PREFLIGHT_BYTES = 65_536
# All ingestion-owned RFC3339 timestamp strings use this inclusive UTF-8 bound.
MAX_INGESTION_TIMESTAMP_BYTES = 64
MAX_LIST_PAGE = 100
MAX_OBSERVATION_PAGE = 200

_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")


def _text(value: object, name: str, *, max_bytes: int | None = None) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    value = required_text(value, name)
    if max_bytes is not None and len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return value


def _identity(value: object, name: str) -> str:
    value = _text(value, name)
    if _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identity")
    return value


def validate_ingestion_identity(value: object, name: str) -> str:
    """Validate one public ingestion operational identity."""

    return _identity(value, name)


def _name(value: object, name: str) -> str:
    value = _text(value, name)
    if _NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be a structure name")
    return value


def _version(value: object) -> str:
    value = _text(value, "version")
    if _VERSION.fullmatch(value) is None or ".." in value or value.endswith("."):
        raise ValueError("version has invalid syntax")
    return value


def _optional(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact boolean")
    return value


def _count(value: object, name: str, *, minimum: int = 0, maximum: int = MAX_SOURCE_COUNT) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be from {minimum} through {maximum}")
    return value


def _enum(enum_type: type, value: object, name: str):
    value = _text(value, name)
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"unknown {name}") from None


def _array(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact array")
    return tuple(value)


def _tuple(value: object, item_type: type, name: str, maximum: int, *, minimum: int = 0) -> tuple:
    if type(value) is not tuple or any(type(item) is not item_type for item in value):
        raise TypeError(f"{name} must be an exact tuple of {item_type.__name__} values")
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} requires {minimum} through {maximum} entries")
    return tuple(item_type.from_dict(item.to_dict()) for item in value)


def _parse_tuple(value: object, item_type: type, name: str) -> tuple:
    return tuple(item_type.from_dict(item) for item in _array(value, name))


def _timestamp_text(value: object, name: str) -> str:
    return require_rfc3339(
        _text(value, name, max_bytes=MAX_INGESTION_TIMESTAMP_BYTES), name
    )


def _timestamp(value: object, name: str) -> datetime:
    value = _timestamp_text(value, name)
    return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _tags(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("tags must be an exact tuple")
    if len(value) > MAX_TAGS:
        raise ValueError(f"tags permits at most {MAX_TAGS} entries")
    tags = tuple(_text(item, "tag", max_bytes=64) for item in value)
    if len(tags) != len(set(tags)):
        raise ValueError("tags must be unique")
    return tuple(sorted(tags))


def _cursor(value: object) -> str | None:
    return None if value is None else _identity(value, "cursor")


def _require_canonical_size(
    document: dict[str, object], name: str, maximum_bytes: int
) -> None:
    """Apply an inclusive limit to exact canonical UTF-8 JSON bytes."""
    if len(canonical_bytes(document)) > maximum_bytes:
        raise ValueError(f"{name} exceeds {maximum_bytes} canonical bytes")


def _diagnostics(value: object) -> tuple["IngestionDiagnosticCode", ...]:
    if type(value) is not tuple or any(type(item) is not IngestionDiagnosticCode for item in value):
        raise TypeError("diagnostic_codes must be an exact tuple of IngestionDiagnosticCode values")
    if len(value) > MAX_DIAGNOSTICS or len(value) != len(set(value)):
        raise ValueError("diagnostic_codes must be unique and bounded")
    return tuple(sorted(value, key=lambda item: item.value))


def _glob(value: object) -> str:
    value = _text(value, "pattern", max_bytes=MAX_GLOB_BYTES)
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("pattern must be NFC normalized")
    if value.startswith("/") or "\\" in value or ":" in value or value.endswith("/"):
        raise ValueError("pattern must be a relative POSIX path glob")
    if any(token in value for token in ("[", "]", "{", "}", "(", ")", "!", "\\")):
        raise ValueError("pattern uses an unsupported glob feature")
    segments = value.split("/")
    if not 1 <= len(segments) <= MAX_GLOB_SEGMENTS:
        raise ValueError("pattern has too many segments")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("pattern has an invalid path segment")
    if any("**" in segment and segment != "**" for segment in segments):
        raise ValueError("** must be a complete segment")
    return value


class SourceAdmissionKind(str, Enum):
    LOCAL_SELECTION = "local_selection"
    FINALIZED_UPLOAD = "finalized_upload"


class ParsingProfile(str, Enum):
    MARKDOWN_YAML_FRONTMATTER_V1 = "markdown_yaml_frontmatter_v1"
    MARKDOWN_YAML_FRONTMATTER_V2 = "markdown_yaml_frontmatter_v2"


class UnitBoundary(str, Enum):
    FILE = "file"


class FrontmatterMode(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class FieldSelectorKind(str, Enum):
    DOCUMENT_BODY = "document_body"
    FRONTMATTER_FIELD = "frontmatter_field"
    LOGICAL_PATH = "logical_path"


class FieldValueKind(str, Enum):
    STRING = "string"
    BOOLEAN = "boolean"
    INT64 = "int64"
    NUMBER = "number"
    ARRAY = "array"
    OBJECT = "object"


class IngestionRunState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECONCILE_REQUIRED = "reconcile_required"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class IngestionDiagnosticCode(str, Enum):
    UNMATCHED_SOURCE = "unmatched_source"
    AMBIGUOUS_SOURCE = "ambiguous_source"
    SOURCE_CHANGED = "source_changed"
    STRUCTURE_INVALID = "structure_invalid"
    PARSE_FAILED = "parse_failed"
    METADATA_INVALID = "metadata_invalid"
    RELATIONSHIP_INVALID = "relationship_invalid"
    OUTPUT_CONFLICT = "output_conflict"
    INTERRUPTED = "interrupted"
    EFFECT_UNCERTAIN = "effect_uncertain"
    EXECUTION_FAILED = "execution_failed"
    BUNDLE_INVALID = "bundle_invalid"


class IngestionOperationCode(str, Enum):
    INPUT_MUTATED = "input_mutated"
    OPERATION_FAILED = "operation_failed"
    RESULT_INVALID = "result_invalid"
    RESULT_UNBOUND = "result_unbound"
    ADMISSION_INELIGIBLE = "admission_ineligible"
    INVALID_SELECTION = "invalid_selection"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    SOURCE_UNSAFE = "source_unsafe"
    SOURCE_CHANGED = "source_changed"
    LIMIT_EXCEEDED = "limit_exceeded"
    START_INELIGIBLE = "start_ineligible"
    CANCEL_INELIGIBLE = "cancel_ineligible"
    RESUME_INELIGIBLE = "resume_ineligible"
    RECONCILE_INELIGIBLE = "reconcile_ineligible"
    RESULT_UNAVAILABLE = "result_unavailable"
    VERIFY_INELIGIBLE = "verify_ineligible"
    RUN_MISSING = "run_missing"
    CURSOR_INVALID = "cursor_invalid"
    STATE_CONFLICT = "state_conflict"
    INTEGRITY_ERROR = "integrity_error"


class IngestionOperationError(ValueError):
    def __init__(self, code: IngestionOperationCode) -> None:
        if type(code) is not IngestionOperationCode:
            raise TypeError("code must be exact IngestionOperationCode")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class AuthorizedSourceRef:
    project_ref: str
    kind: SourceAdmissionKind
    source_ref: str
    authority_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        if type(self.kind) is not SourceAdmissionKind:
            raise TypeError("kind must be exact SourceAdmissionKind")
        object.__setattr__(self, "source_ref", _identity(self.source_ref, "source_ref"))
        object.__setattr__(self, "authority_digest", digest_text(self.authority_digest, "authority_digest"))

    def to_dict(self) -> dict[str, object]:
        return {"project_ref": self.project_ref, "kind": self.kind.value, "source_ref": self.source_ref, "authority_digest": self.authority_digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "AuthorizedSourceRef":
        value = exact_fields(value, frozenset({"project_ref", "kind", "source_ref", "authority_digest"}), "authorized_source_ref")
        return cls(value["project_ref"], _enum(SourceAdmissionKind, value["kind"], "kind"), value["source_ref"], value["authority_digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SourceAdmissionRequest:
    request_id: str
    project_ref: str
    source: AuthorizedSourceRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _identity(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        if type(self.source) is not AuthorizedSourceRef:
            raise TypeError("source must be exact AuthorizedSourceRef")
        object.__setattr__(self, "source", AuthorizedSourceRef.from_dict(self.source.to_dict()))
        if self.source.project_ref != self.project_ref:
            raise ValueError("source project must bind the request")

    @property
    def admission_fingerprint(self) -> str:
        return contract_digest("synaptic-source-admission-request/v1", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, "project_ref": self.project_ref, "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SourceAdmissionRequest":
        value = exact_fields(value, frozenset({"request_id", "project_ref", "source"}), "source_admission_request")
        return cls(value["request_id"], value["project_ref"], AuthorizedSourceRef.from_dict(value["source"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SourceSnapshotRef:
    project_ref: str
    snapshot_id: str
    source_kind: SourceAdmissionKind
    manifest_digest: str
    source_count: int
    admission_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        object.__setattr__(self, "snapshot_id", _identity(self.snapshot_id, "snapshot_id"))
        if type(self.source_kind) is not SourceAdmissionKind:
            raise TypeError("source_kind must be exact SourceAdmissionKind")
        object.__setattr__(self, "manifest_digest", digest_text(self.manifest_digest, "manifest_digest"))
        _count(self.source_count, "source_count", minimum=1)
        object.__setattr__(self, "admission_fingerprint", digest_text(self.admission_fingerprint, "admission_fingerprint"))

    def to_dict(self) -> dict[str, object]:
        return {"project_ref": self.project_ref, "snapshot_id": self.snapshot_id, "source_kind": self.source_kind.value,
                "manifest_digest": self.manifest_digest, "source_count": self.source_count,
                "admission_fingerprint": self.admission_fingerprint}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SourceSnapshotRef":
        value = exact_fields(value, frozenset({"project_ref", "snapshot_id", "source_kind", "manifest_digest", "source_count", "admission_fingerprint"}), "source_snapshot_ref")
        return cls(value["project_ref"], value["snapshot_id"], _enum(SourceAdmissionKind, value["source_kind"], "source_kind"), value["manifest_digest"], value["source_count"], value["admission_fingerprint"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MetadataPolicyRef:
    policy_id: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, "policy_id"))
        object.__setattr__(self, "digest", digest_text(self.digest, "digest"))

    def to_dict(self) -> dict[str, object]: return {"policy_id": self.policy_id, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MetadataPolicyRef":
        value = exact_fields(value, frozenset({"policy_id", "digest"}), "metadata_policy_ref")
        return cls(value["policy_id"], value["digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SchemaRef:
    schema_id: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _identity(self.schema_id, "schema_id"))
        object.__setattr__(self, "version", _version(self.version))
        object.__setattr__(self, "digest", digest_text(self.digest, "digest"))

    def to_dict(self) -> dict[str, object]: return {"schema_id": self.schema_id, "version": self.version, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SchemaRef":
        value = exact_fields(value, frozenset({"schema_id", "version", "digest"}), "schema_ref")
        return cls(value["schema_id"], value["version"], value["digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructureRef:
    name: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        object.__setattr__(self, "version", _version(self.version))
        object.__setattr__(self, "digest", digest_text(self.digest, "digest"))

    def to_dict(self) -> dict[str, object]: return {"name": self.name, "version": self.version, "digest": self.digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureRef":
        value = exact_fields(value, frozenset({"name", "version", "digest"}), "structure_ref")
        return cls(value["name"], value["version"], value["digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MarkdownProfileV1:
    frontmatter_mode: FrontmatterMode

    def __post_init__(self) -> None:
        if type(self.frontmatter_mode) is not FrontmatterMode:
            raise TypeError("frontmatter_mode must be exact FrontmatterMode")

    def to_dict(self) -> dict[str, object]: return {"frontmatter_mode": self.frontmatter_mode.value}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MarkdownProfileV1":
        value = exact_fields(value, frozenset({"frontmatter_mode"}), "markdown_profile")
        return cls(_enum(FrontmatterMode, value["frontmatter_mode"], "frontmatter_mode"))


@dataclass(frozen=True, slots=True)
class SourceMatcher:
    pattern: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _glob(self.pattern))

    def to_dict(self) -> dict[str, object]: return {"pattern": self.pattern}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SourceMatcher":
        value = exact_fields(value, frozenset({"pattern"}), "source_matcher")
        return cls(value["pattern"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FieldSelector:
    kind: FieldSelectorKind
    key: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not FieldSelectorKind:
            raise TypeError("kind must be exact FieldSelectorKind")
        if self.key is not None:
            object.__setattr__(self, "key", _text(self.key, "key", max_bytes=MAX_YAML_KEY_BYTES))
        if (self.kind is FieldSelectorKind.FRONTMATTER_FIELD) != (self.key is not None):
            raise ValueError("only frontmatter_field requires a key")

    def to_dict(self) -> dict[str, object]: return {"kind": self.kind.value, "key": self.key}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FieldSelector":
        value = exact_fields(value, frozenset({"kind", "key"}), "field_selector")
        return cls(_enum(FieldSelectorKind, value["kind"], "kind"), value["key"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FieldMapping:
    name: str
    selector: FieldSelector
    value_kind: FieldValueKind
    required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        if type(self.selector) is not FieldSelector or type(self.value_kind) is not FieldValueKind:
            raise TypeError("selector/value_kind have invalid types")
        object.__setattr__(self, "selector", FieldSelector.from_dict(self.selector.to_dict()))
        _bool(self.required, "required")
        if self.selector.kind in {FieldSelectorKind.DOCUMENT_BODY, FieldSelectorKind.LOGICAL_PATH} and self.value_kind is not FieldValueKind.STRING:
            raise ValueError("document_body and logical_path fields must be strings")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "selector": self.selector.to_dict(), "value_kind": self.value_kind.value, "required": self.required}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FieldMapping":
        value = exact_fields(value, frozenset({"name", "selector", "value_kind", "required"}), "field_mapping")
        return cls(value["name"], FieldSelector.from_dict(value["selector"]), _enum(FieldValueKind, value["value_kind"], "value_kind"), value["required"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TextProjection:
    name: str
    field_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        object.__setattr__(self, "field_ref", _name(self.field_ref, "field_ref"))

    def to_dict(self) -> dict[str, object]: return {"name": self.name, "field_ref": self.field_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TextProjection":
        value = exact_fields(value, frozenset({"name", "field_ref"}), "text_projection")
        return cls(value["name"], value["field_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class MetadataDeclaration:
    name: str
    field_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        object.__setattr__(self, "field_ref", _name(self.field_ref, "field_ref"))

    def to_dict(self) -> dict[str, object]: return {"name": self.name, "field_ref": self.field_ref}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MetadataDeclaration":
        value = exact_fields(value, frozenset({"name", "field_ref"}), "metadata_declaration")
        return cls(value["name"], value["field_ref"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RelationshipDeclaration:
    name: str
    target_structure: StructureRef
    source_field: str
    target_field: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        if type(self.target_structure) is not StructureRef:
            raise TypeError("target_structure must be exact StructureRef")
        object.__setattr__(self, "target_structure", StructureRef.from_dict(self.target_structure.to_dict()))
        object.__setattr__(self, "source_field", _name(self.source_field, "source_field"))
        object.__setattr__(self, "target_field", _name(self.target_field, "target_field"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "target_structure": self.target_structure.to_dict(), "source_field": self.source_field, "target_field": self.target_field}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RelationshipDeclaration":
        value = exact_fields(value, frozenset({"name", "target_structure", "source_field", "target_field"}), "relationship_declaration")
        return cls(value["name"], StructureRef.from_dict(value["target_structure"]), value["source_field"], value["target_field"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructureDefinition:
    ref: StructureRef
    parsing_profile: ParsingProfile
    unit: UnitBoundary
    markdown: MarkdownProfileV1
    fields: tuple[FieldMapping, ...]
    text_projections: tuple[TextProjection, ...]
    metadata: tuple[MetadataDeclaration, ...] = ()
    relationships: tuple[RelationshipDeclaration, ...] = ()
    schema_ref: SchemaRef | None = None
    metadata_policy: MetadataPolicyRef | None = None

    def __post_init__(self) -> None:
        if type(self.ref) is not StructureRef or type(self.parsing_profile) is not ParsingProfile or type(self.unit) is not UnitBoundary or type(self.markdown) is not MarkdownProfileV1:
            raise TypeError("structure header has invalid types")
        object.__setattr__(self, "ref", StructureRef.from_dict(self.ref.to_dict()))
        object.__setattr__(self, "markdown", MarkdownProfileV1.from_dict(self.markdown.to_dict()))
        fields = _tuple(self.fields, FieldMapping, "fields", MAX_FIELDS, minimum=1)
        projections = _tuple(self.text_projections, TextProjection, "text_projections", MAX_TEXT_PROJECTIONS, minimum=1)
        metadata = _tuple(self.metadata, MetadataDeclaration, "metadata", MAX_METADATA_DECLARATIONS)
        relationships = _tuple(self.relationships, RelationshipDeclaration, "relationships", MAX_RELATIONSHIPS)
        for label, values in (("fields", fields), ("text_projections", projections), ("metadata", metadata), ("relationships", relationships)):
            names = tuple(item.name for item in values)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} names must be unique")
        field_map = {item.name: item for item in fields}
        body_fields = tuple(item for item in fields if item.selector.kind is FieldSelectorKind.DOCUMENT_BODY)
        if len(body_fields) != 1 or not body_fields[0].required or body_fields[0].value_kind is not FieldValueKind.STRING:
            raise ValueError("structure requires one required string document_body field")
        if any(item.field_ref not in field_map or field_map[item.field_ref].value_kind is not FieldValueKind.STRING for item in projections):
            raise ValueError("text projections must reference declared string fields")
        if any(item.field_ref not in field_map for item in metadata):
            raise ValueError("metadata must reference declared fields")
        if any(item.source_field not in field_map for item in relationships):
            raise ValueError("relationship source fields must be declared")
        object.__setattr__(self, "fields", tuple(sorted(fields, key=lambda item: item.name)))
        object.__setattr__(self, "text_projections", tuple(sorted(projections, key=lambda item: item.name)))
        object.__setattr__(self, "metadata", tuple(sorted(metadata, key=lambda item: item.name)))
        object.__setattr__(self, "relationships", tuple(sorted(relationships, key=lambda item: item.name)))
        for attr, kind in (("schema_ref", SchemaRef), ("metadata_policy", MetadataPolicyRef)):
            item = getattr(self, attr)
            if item is not None:
                if type(item) is not kind:
                    raise TypeError(f"{attr} must be exact {kind.__name__} or None")
                object.__setattr__(self, attr, kind.from_dict(item.to_dict()))
        if self.ref.digest != self.definition_digest:
            raise ValueError("structure ref digest does not bind the definition")

    @property
    def definition_digest(self) -> str:
        return contract_digest("synaptic-markdown-structure/v1", self._digest_document(self.to_dict()))

    @staticmethod
    def _digest_document(document: dict[str, object]) -> dict[str, object]:
        """Exclude relationship target digests to permit cyclic declarations.

        The enclosing StructureSet digest still binds every exact target ref.
        """
        normalized = dict(document)
        ref = document["ref"]
        normalized["ref"] = {"name": ref["name"], "version": ref["version"]}  # type: ignore[index]
        normalized_relationships = []
        for relationship in document["relationships"]:  # type: ignore[union-attr]
            item = dict(relationship)
            target = relationship["target_structure"]
            item["target_structure"] = {"name": target["name"], "version": target["version"]}
            normalized_relationships.append(item)
        normalized["relationships"] = normalized_relationships
        return normalized

    @classmethod
    def define(
        cls, *, name: str, version: str, markdown: MarkdownProfileV1,
        fields: tuple[FieldMapping, ...], text_projections: tuple[TextProjection, ...],
        metadata: tuple[MetadataDeclaration, ...] = (),
        relationships: tuple[RelationshipDeclaration, ...] = (),
        schema_ref: SchemaRef | None = None,
        metadata_policy: MetadataPolicyRef | None = None,
        parsing_profile: ParsingProfile = ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1,
    ) -> "StructureDefinition":
        """Build a supported Markdown declaration and its content-addressed ref."""
        name, version = _name(name, "name"), _version(version)
        if type(markdown) is not MarkdownProfileV1:
            raise TypeError("markdown must be exact MarkdownProfileV1")
        if type(parsing_profile) is not ParsingProfile:
            raise TypeError("parsing_profile must be exact ParsingProfile")
        fields = tuple(sorted(_tuple(fields, FieldMapping, "fields", MAX_FIELDS, minimum=1), key=lambda item: item.name))
        text_projections = tuple(sorted(_tuple(text_projections, TextProjection, "text_projections", MAX_TEXT_PROJECTIONS, minimum=1), key=lambda item: item.name))
        metadata = tuple(sorted(_tuple(metadata, MetadataDeclaration, "metadata", MAX_METADATA_DECLARATIONS), key=lambda item: item.name))
        relationships = tuple(sorted(_tuple(relationships, RelationshipDeclaration, "relationships", MAX_RELATIONSHIPS), key=lambda item: item.name))
        if schema_ref is not None and type(schema_ref) is not SchemaRef:
            raise TypeError("schema_ref must be exact SchemaRef or None")
        if metadata_policy is not None and type(metadata_policy) is not MetadataPolicyRef:
            raise TypeError("metadata_policy must be exact MetadataPolicyRef or None")
        document: dict[str, object] = {
            "ref": {"name": name, "version": version},
            "parsing_profile": parsing_profile.value,
            "unit": UnitBoundary.FILE.value,
            "markdown": markdown.to_dict(),
            "fields": [item.to_dict() for item in fields],
            "text_projections": [item.to_dict() for item in text_projections],
            "metadata": [item.to_dict() for item in metadata],
            "relationships": [item.to_dict() for item in relationships],
            "schema_ref": None if schema_ref is None else schema_ref.to_dict(),
            "metadata_policy": None if metadata_policy is None else metadata_policy.to_dict(),
        }
        ref = StructureRef(name, version, contract_digest("synaptic-markdown-structure/v1", cls._digest_document(document)))
        relationships = tuple(
            RelationshipDeclaration(
                item.name,
                ref if (item.target_structure.name, item.target_structure.version) == (name, version) else item.target_structure,
                item.source_field,
                item.target_field,
            )
            for item in relationships
        )
        return cls(ref, parsing_profile, UnitBoundary.FILE, markdown, fields,
                   text_projections, metadata, relationships, schema_ref, metadata_policy)

    def to_dict(self) -> dict[str, object]:
        return {"ref": self.ref.to_dict(), "parsing_profile": self.parsing_profile.value, "unit": self.unit.value,
                "markdown": self.markdown.to_dict(), "fields": [item.to_dict() for item in self.fields],
                "text_projections": [item.to_dict() for item in self.text_projections], "metadata": [item.to_dict() for item in self.metadata],
                "relationships": [item.to_dict() for item in self.relationships], "schema_ref": None if self.schema_ref is None else self.schema_ref.to_dict(),
                "metadata_policy": None if self.metadata_policy is None else self.metadata_policy.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureDefinition":
        value = exact_fields(value, frozenset({"ref", "parsing_profile", "unit", "markdown", "fields", "text_projections", "metadata", "relationships", "schema_ref", "metadata_policy"}), "structure_definition")
        return cls(StructureRef.from_dict(value["ref"]), _enum(ParsingProfile, value["parsing_profile"], "parsing_profile"),
                   _enum(UnitBoundary, value["unit"], "unit"), MarkdownProfileV1.from_dict(value["markdown"]),
                   _parse_tuple(value["fields"], FieldMapping, "fields"), _parse_tuple(value["text_projections"], TextProjection, "text_projections"),
                   _parse_tuple(value["metadata"], MetadataDeclaration, "metadata"), _parse_tuple(value["relationships"], RelationshipDeclaration, "relationships"),
                   None if value["schema_ref"] is None else SchemaRef.from_dict(value["schema_ref"]),
                   None if value["metadata_policy"] is None else MetadataPolicyRef.from_dict(value["metadata_policy"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructureBinding:
    binding_id: str
    matcher: SourceMatcher
    structure_ref: StructureRef
    metadata_policy: MetadataPolicyRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_id", _name(self.binding_id, "binding_id"))
        if type(self.matcher) is not SourceMatcher or type(self.structure_ref) is not StructureRef:
            raise TypeError("matcher/structure_ref have invalid types")
        object.__setattr__(self, "matcher", SourceMatcher.from_dict(self.matcher.to_dict()))
        object.__setattr__(self, "structure_ref", StructureRef.from_dict(self.structure_ref.to_dict()))
        if self.metadata_policy is not None:
            if type(self.metadata_policy) is not MetadataPolicyRef:
                raise TypeError("metadata_policy must be exact MetadataPolicyRef or None")
            object.__setattr__(self, "metadata_policy", MetadataPolicyRef.from_dict(self.metadata_policy.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {"binding_id": self.binding_id, "matcher": self.matcher.to_dict(), "structure_ref": self.structure_ref.to_dict(),
                "metadata_policy": None if self.metadata_policy is None else self.metadata_policy.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureBinding":
        value = exact_fields(value, frozenset({"binding_id", "matcher", "structure_ref", "metadata_policy"}), "structure_binding")
        return cls(value["binding_id"], SourceMatcher.from_dict(value["matcher"]), StructureRef.from_dict(value["structure_ref"]),
                   None if value["metadata_policy"] is None else MetadataPolicyRef.from_dict(value["metadata_policy"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructureSet:
    structures: tuple[StructureDefinition, ...]
    bindings: tuple[StructureBinding, ...]

    def __post_init__(self) -> None:
        structures = _tuple(self.structures, StructureDefinition, "structures", MAX_STRUCTURES, minimum=1)
        bindings = _tuple(self.bindings, StructureBinding, "bindings", MAX_BINDINGS, minimum=1)
        refs = tuple(item.ref for item in structures)
        identities = tuple((item.ref.name, item.ref.version) for item in structures)
        binding_ids = tuple(item.binding_id for item in bindings)
        if len(refs) != len(set(refs)) or len(identities) != len(set(identities)) or len(binding_ids) != len(set(binding_ids)):
            raise ValueError("structure refs and binding ids must be unique")
        definitions = {item.ref: item for item in structures}
        if any(item.structure_ref not in definitions for item in bindings):
            raise ValueError("every binding must reference a declared structure")
        for structure in structures:
            source_fields = {item.name: item.value_kind for item in structure.fields}
            for relationship in structure.relationships:
                target = definitions.get(relationship.target_structure)
                if target is None:
                    raise ValueError("relationship target structure must be declared")
                target_fields = {item.name: item.value_kind for item in target.fields}
                if relationship.target_field not in target_fields:
                    raise ValueError("relationship target field must be declared")
                if source_fields[relationship.source_field] is not target_fields[relationship.target_field]:
                    raise ValueError("relationship fields must have the same value kind")
        object.__setattr__(self, "structures", tuple(sorted(structures, key=lambda item: (item.ref.name, item.ref.version))))
        object.__setattr__(self, "bindings", tuple(sorted(bindings, key=lambda item: item.binding_id)))
        _require_canonical_size(self.to_dict(), "structure set", MAX_STRUCTURE_SET_BYTES)

    @property
    def digest(self) -> str:
        return contract_digest("synaptic-structure-set/v1", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"structures": [item.to_dict() for item in self.structures], "bindings": [item.to_dict() for item in self.bindings]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureSet":
        value = exact_fields(value, frozenset({"structures", "bindings"}), "structure_set")
        return cls(_parse_tuple(value["structures"], StructureDefinition, "structures"), _parse_tuple(value["bindings"], StructureBinding, "bindings"))


class ProposalEvidenceCode(str, Enum):
    MARKDOWN_EXTENSION = "markdown_extension"
    FRONTMATTER_OPENER = "frontmatter_opener"


@dataclass(frozen=True, slots=True)
class StructureProposalRequest:
    snapshot: SourceSnapshotRef

    def __post_init__(self) -> None:
        if type(self.snapshot) is not SourceSnapshotRef:
            raise TypeError("snapshot must be exact SourceSnapshotRef")
        object.__setattr__(self, "snapshot", SourceSnapshotRef.from_dict(self.snapshot.to_dict()))

    def to_dict(self) -> dict[str, object]: return {"snapshot": self.snapshot.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureProposalRequest":
        value = exact_fields(value, frozenset({"snapshot"}), "structure_proposal_request")
        return cls(SourceSnapshotRef.from_dict(value["snapshot"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StructureProposal:
    proposal_id: str
    snapshot: SourceSnapshotRef
    suggested_structures: StructureSet
    evidence_codes: tuple[ProposalEvidenceCode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_id", _identity(self.proposal_id, "proposal_id"))
        if type(self.snapshot) is not SourceSnapshotRef or type(self.suggested_structures) is not StructureSet:
            raise TypeError("snapshot/suggested_structures have invalid types")
        object.__setattr__(self, "snapshot", SourceSnapshotRef.from_dict(self.snapshot.to_dict()))
        object.__setattr__(self, "suggested_structures", StructureSet.from_dict(self.suggested_structures.to_dict()))
        if type(self.evidence_codes) is not tuple or any(type(item) is not ProposalEvidenceCode for item in self.evidence_codes):
            raise TypeError("evidence_codes must be an exact tuple of ProposalEvidenceCode values")
        if len(self.evidence_codes) > MAX_DIAGNOSTICS or len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("evidence_codes must be unique and bounded")
        object.__setattr__(self, "evidence_codes", tuple(sorted(self.evidence_codes, key=lambda item: item.value)))

    def to_dict(self) -> dict[str, object]:
        return {"proposal_id": self.proposal_id, "snapshot": self.snapshot.to_dict(), "suggested_structures": self.suggested_structures.to_dict(),
                "evidence_codes": [item.value for item in self.evidence_codes]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StructureProposal":
        value = exact_fields(value, frozenset({"proposal_id", "snapshot", "suggested_structures", "evidence_codes"}), "structure_proposal")
        return cls(value["proposal_id"], SourceSnapshotRef.from_dict(value["snapshot"]), StructureSet.from_dict(value["suggested_structures"]),
                   tuple(_enum(ProposalEvidenceCode, item, "evidence_code") for item in _array(value["evidence_codes"], "evidence_codes")))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    request_id: str
    project_ref: str
    snapshot: SourceSnapshotRef
    structures: StructureSet
    output_ref: str
    metadata_policy: MetadataPolicyRef | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _identity(self.request_id, "request_id"))
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        if type(self.snapshot) is not SourceSnapshotRef or type(self.structures) is not StructureSet:
            raise TypeError("snapshot/structures have invalid types")
        object.__setattr__(self, "snapshot", SourceSnapshotRef.from_dict(self.snapshot.to_dict()))
        object.__setattr__(self, "structures", StructureSet.from_dict(self.structures.to_dict()))
        if self.snapshot.project_ref != self.project_ref:
            raise ValueError("snapshot project must bind the request")
        object.__setattr__(self, "output_ref", _identity(self.output_ref, "output_ref"))
        if self.metadata_policy is not None:
            if type(self.metadata_policy) is not MetadataPolicyRef:
                raise TypeError("metadata_policy must be exact MetadataPolicyRef or None")
            object.__setattr__(self, "metadata_policy", MetadataPolicyRef.from_dict(self.metadata_policy.to_dict()))
        object.__setattr__(self, "tags", _tags(self.tags))

    @property
    def request_fingerprint(self) -> str:
        return contract_digest("synaptic-ingestion-request/v1", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"request_id": self.request_id, "project_ref": self.project_ref, "snapshot": self.snapshot.to_dict(),
                "structures": self.structures.to_dict(), "output_ref": self.output_ref,
                "metadata_policy": None if self.metadata_policy is None else self.metadata_policy.to_dict(), "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionRequest":
        value = exact_fields(value, frozenset({"request_id", "project_ref", "snapshot", "structures", "output_ref", "metadata_policy", "tags"}), "ingestion_request")
        return cls(value["request_id"], value["project_ref"], SourceSnapshotRef.from_dict(value["snapshot"]), StructureSet.from_dict(value["structures"]), value["output_ref"],
                   None if value["metadata_policy"] is None else MetadataPolicyRef.from_dict(value["metadata_policy"]), tuple(_array(value["tags"], "tags")))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class BindingPreview:
    binding_id: str
    matched_sources: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_id", _name(self.binding_id, "binding_id"))
        _count(self.matched_sources, "matched_sources")

    def to_dict(self) -> dict[str, object]: return {"binding_id": self.binding_id, "matched_sources": self.matched_sources}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "BindingPreview":
        value = exact_fields(value, frozenset({"binding_id", "matched_sources"}), "binding_preview")
        return cls(value["binding_id"], value["matched_sources"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionPreview:
    source_count: int
    matched_sources: int
    unmatched_sources: int
    ambiguous_sources: int
    bindings: tuple[BindingPreview, ...]

    def __post_init__(self) -> None:
        for name in ("source_count", "matched_sources", "unmatched_sources", "ambiguous_sources"):
            _count(getattr(self, name), name)
        if self.source_count != self.matched_sources + self.unmatched_sources + self.ambiguous_sources:
            raise ValueError("source dispositions must sum to source_count")
        bindings = _tuple(self.bindings, BindingPreview, "bindings", MAX_BINDINGS, minimum=1)
        ids = tuple(item.binding_id for item in bindings)
        if len(ids) != len(set(ids)) or sum(item.matched_sources for item in bindings) != self.matched_sources:
            raise ValueError("binding preview must uniquely partition matched sources")
        object.__setattr__(self, "bindings", tuple(sorted(bindings, key=lambda item: item.binding_id)))

    @property
    def ready(self) -> bool:
        return self.unmatched_sources == 0 and self.ambiguous_sources == 0

    def to_dict(self) -> dict[str, object]:
        return {"source_count": self.source_count, "matched_sources": self.matched_sources, "unmatched_sources": self.unmatched_sources,
                "ambiguous_sources": self.ambiguous_sources, "bindings": [item.to_dict() for item in self.bindings]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionPreview":
        value = exact_fields(value, frozenset({"source_count", "matched_sources", "unmatched_sources", "ambiguous_sources", "bindings"}), "ingestion_preview")
        return cls(value["source_count"], value["matched_sources"], value["unmatched_sources"], value["ambiguous_sources"], _parse_tuple(value["bindings"], BindingPreview, "bindings"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    schema_version: str
    request: IngestionRequest
    preview: IngestionPreview

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != INGESTION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported ingestion plan schema version")
        if type(self.request) is not IngestionRequest or type(self.preview) is not IngestionPreview:
            raise TypeError("request/preview have invalid types")
        object.__setattr__(self, "request", IngestionRequest.from_dict(self.request.to_dict()))
        object.__setattr__(self, "preview", IngestionPreview.from_dict(self.preview.to_dict()))
        if self.preview.source_count != self.request.snapshot.source_count:
            raise ValueError("preview source_count must bind the snapshot")
        if {item.binding_id for item in self.preview.bindings} != {item.binding_id for item in self.request.structures.bindings}:
            raise ValueError("preview must report every binding")

    @property
    def plan_fingerprint(self) -> str:
        return contract_digest(INGESTION_PLAN_SCHEMA_VERSION, self.to_dict())

    def to_dict(self) -> dict[str, object]: return {"schema_version": self.schema_version, "request": self.request.to_dict(), "preview": self.preview.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionPlan":
        value = exact_fields(value, frozenset({"schema_version", "request", "preview"}), "ingestion_plan")
        return cls(value["schema_version"], IngestionRequest.from_dict(value["request"]), IngestionPreview.from_dict(value["preview"]))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionPreflight:
    plan_fingerprint: str
    ready: bool
    checked_at: str
    expires_at: str
    authorization: tuple[AuthorizationRequirement, ...] = ()
    diagnostic_codes: tuple[IngestionDiagnosticCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_fingerprint", digest_text(self.plan_fingerprint, "plan_fingerprint"))
        _bool(self.ready, "ready")
        checked_at = _timestamp_text(self.checked_at, "checked_at")
        expires_at = _timestamp_text(self.expires_at, "expires_at")
        if _timestamp(expires_at, "expires_at") <= _timestamp(checked_at, "checked_at"):
            raise ValueError("expires_at must be later than checked_at")
        if type(self.authorization) is not tuple or any(type(item) is not AuthorizationRequirement for item in self.authorization):
            raise TypeError("authorization must be an exact tuple of AuthorizationRequirement values")
        if len(self.authorization) > MAX_AUTHORIZATION_REQUIREMENTS:
            raise ValueError("authorization exceeds its bound")
        authorization = tuple(AuthorizationRequirement.from_dict(item.to_dict()) for item in self.authorization)
        if len({item.operation for item in authorization}) != len(authorization):
            raise ValueError("authorization operations must be unique")
        diagnostics = _diagnostics(self.diagnostic_codes)
        if self.ready and diagnostics:
            raise ValueError("ready preflight takes no diagnostics")
        if not self.ready and not diagnostics:
            raise ValueError("not-ready preflight requires diagnostics")
        object.__setattr__(self, "checked_at", checked_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "authorization", tuple(sorted(authorization, key=lambda item: item.operation)))
        object.__setattr__(self, "diagnostic_codes", diagnostics)
        _require_canonical_size(
            self.to_dict(), "ingestion preflight", MAX_INGESTION_PREFLIGHT_BYTES
        )

    @property
    def preflight_fingerprint(self) -> str:
        return contract_digest("synaptic-ingestion-preflight/v1", self.to_dict())

    def binds(self, plan: IngestionPlan) -> bool:
        if type(plan) is not IngestionPlan:
            raise TypeError("plan must be exact IngestionPlan")
        return self.plan_fingerprint == plan.plan_fingerprint

    def is_expired(self, now: str) -> bool:
        return _timestamp(now, "now") >= _timestamp(self.expires_at, "expires_at")

    def to_dict(self) -> dict[str, object]:
        return {"plan_fingerprint": self.plan_fingerprint, "ready": self.ready, "checked_at": self.checked_at, "expires_at": self.expires_at,
                "authorization": [item.to_dict() for item in self.authorization], "diagnostic_codes": [item.value for item in self.diagnostic_codes]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionPreflight":
        value = exact_fields(value, frozenset({"plan_fingerprint", "ready", "checked_at", "expires_at", "authorization", "diagnostic_codes"}), "ingestion_preflight")
        return cls(value["plan_fingerprint"], value["ready"], value["checked_at"], value["expires_at"],
                   tuple(AuthorizationRequirement.from_dict(item) for item in _array(value["authorization"], "authorization")),
                   tuple(_enum(IngestionDiagnosticCode, item, "diagnostic_code") for item in _array(value["diagnostic_codes"], "diagnostic_codes")))  # type: ignore[arg-type]


def run_authority_digest(run_id: str, project_ref: str, plan_fingerprint: str, preflight_fingerprint: str) -> str:
    document = {"run_id": _identity(run_id, "run_id"), "project_ref": _identity(project_ref, "project_ref"),
                "plan_fingerprint": digest_text(plan_fingerprint, "plan_fingerprint"),
                "preflight_fingerprint": digest_text(preflight_fingerprint, "preflight_fingerprint")}
    return contract_digest("synaptic-ingestion-run-authority/v1", document)


@dataclass(frozen=True, slots=True)
class IngestionRunRef:
    run_id: str
    project_ref: str
    plan_fingerprint: str
    preflight_fingerprint: str
    authority_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identity(self.run_id, "run_id"))
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        object.__setattr__(self, "plan_fingerprint", digest_text(self.plan_fingerprint, "plan_fingerprint"))
        object.__setattr__(self, "preflight_fingerprint", digest_text(self.preflight_fingerprint, "preflight_fingerprint"))
        object.__setattr__(self, "authority_digest", digest_text(self.authority_digest, "authority_digest"))
        if self.authority_digest != run_authority_digest(self.run_id, self.project_ref, self.plan_fingerprint, self.preflight_fingerprint):
            raise ValueError("authority_digest does not bind the run authority")

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "project_ref": self.project_ref, "plan_fingerprint": self.plan_fingerprint,
                "preflight_fingerprint": self.preflight_fingerprint, "authority_digest": self.authority_digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionRunRef":
        value = exact_fields(value, frozenset({"run_id", "project_ref", "plan_fingerprint", "preflight_fingerprint", "authority_digest"}), "ingestion_run_ref")
        return cls(value["run_id"], value["project_ref"], value["plan_fingerprint"], value["preflight_fingerprint"], value["authority_digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionStart:
    run: IngestionRunRef
    accepted: bool

    def __post_init__(self) -> None:
        if type(self.run) is not IngestionRunRef:
            raise TypeError("run must be exact IngestionRunRef")
        object.__setattr__(self, "run", IngestionRunRef.from_dict(self.run.to_dict()))
        _bool(self.accepted, "accepted")

    def to_dict(self) -> dict[str, object]: return {"run": self.run.to_dict(), "accepted": self.accepted}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionStart":
        value = exact_fields(value, frozenset({"run", "accepted"}), "ingestion_start")
        return cls(IngestionRunRef.from_dict(value["run"]), value["accepted"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class NormalizedBundleRef:
    bundle_id: str
    bundle_digest: str
    document_count: int
    plan_fingerprint: str
    run_authority_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _identity(self.bundle_id, "bundle_id"))
        for attr in ("bundle_digest", "plan_fingerprint", "run_authority_digest"):
            object.__setattr__(self, attr, digest_text(getattr(self, attr), attr))
        _count(self.document_count, "document_count", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {"bundle_id": self.bundle_id, "bundle_digest": self.bundle_digest, "document_count": self.document_count,
                "plan_fingerprint": self.plan_fingerprint, "run_authority_digest": self.run_authority_digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "NormalizedBundleRef":
        value = exact_fields(value, frozenset({"bundle_id", "bundle_digest", "document_count", "plan_fingerprint", "run_authority_digest"}), "normalized_bundle_ref")
        return cls(value["bundle_id"], value["bundle_digest"], value["document_count"], value["plan_fingerprint"], value["run_authority_digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    run: IngestionRunRef
    state: IngestionRunState
    source_count: int
    sources_processed: int
    documents_written: int
    bundle: NormalizedBundleRef | None = None
    diagnostic_code: IngestionDiagnosticCode | None = None

    def __post_init__(self) -> None:
        if type(self.run) is not IngestionRunRef or type(self.state) is not IngestionRunState:
            raise TypeError("run/state have invalid types")
        object.__setattr__(self, "run", IngestionRunRef.from_dict(self.run.to_dict()))
        _count(self.source_count, "source_count", minimum=1)
        _count(self.sources_processed, "sources_processed")
        _count(self.documents_written, "documents_written")
        if self.sources_processed > self.source_count or self.documents_written > self.sources_processed:
            raise ValueError("outcome counts are inconsistent")
        if self.bundle is not None:
            if type(self.bundle) is not NormalizedBundleRef:
                raise TypeError("bundle must be exact NormalizedBundleRef or None")
            object.__setattr__(self, "bundle", NormalizedBundleRef.from_dict(self.bundle.to_dict()))
            if self.bundle.document_count != self.documents_written or self.bundle.plan_fingerprint != self.run.plan_fingerprint or self.bundle.run_authority_digest != self.run.authority_digest:
                raise ValueError("bundle does not bind the outcome")
        if self.diagnostic_code is not None and type(self.diagnostic_code) is not IngestionDiagnosticCode:
            raise TypeError("diagnostic_code must be exact IngestionDiagnosticCode or None")
        no_bundle = self.bundle is None
        no_diagnostic = self.diagnostic_code is None
        if self.state is IngestionRunState.PLANNED:
            valid = self.sources_processed == self.documents_written == 0 and no_bundle and no_diagnostic
        elif self.state in {IngestionRunState.RUNNING, IngestionRunState.CANCEL_REQUESTED}:
            valid = no_bundle and no_diagnostic
        elif self.state is IngestionRunState.RECONCILE_REQUIRED:
            valid = no_bundle and self.diagnostic_code is IngestionDiagnosticCode.EFFECT_UNCERTAIN
        elif self.state is IngestionRunState.SUCCEEDED:
            valid = self.sources_processed == self.source_count and 1 <= self.documents_written <= self.sources_processed and not no_bundle and no_diagnostic
        elif self.state is IngestionRunState.FAILED:
            valid = no_bundle and self.diagnostic_code not in {None, IngestionDiagnosticCode.EFFECT_UNCERTAIN}
        else:
            valid = no_bundle and no_diagnostic
        if not valid:
            raise ValueError("outcome violates the state matrix")

    @property
    def outcome_digest(self) -> str:
        return contract_digest("synaptic-ingestion-outcome/v1", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "state": self.state.value, "source_count": self.source_count,
                "sources_processed": self.sources_processed, "documents_written": self.documents_written,
                "bundle": None if self.bundle is None else self.bundle.to_dict(),
                "diagnostic_code": None if self.diagnostic_code is None else self.diagnostic_code.value}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionOutcome":
        value = exact_fields(value, frozenset({"run", "state", "source_count", "sources_processed", "documents_written", "bundle", "diagnostic_code"}), "ingestion_outcome")
        return cls(IngestionRunRef.from_dict(value["run"]), _enum(IngestionRunState, value["state"], "state"), value["source_count"], value["sources_processed"], value["documents_written"],
                   None if value["bundle"] is None else NormalizedBundleRef.from_dict(value["bundle"]),
                   None if value["diagnostic_code"] is None else _enum(IngestionDiagnosticCode, value["diagnostic_code"], "diagnostic_code"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionResult:
    schema_version: str
    outcome: IngestionOutcome
    outcome_digest: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != INGESTION_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported ingestion result schema version")
        if type(self.outcome) is not IngestionOutcome:
            raise TypeError("outcome must be exact IngestionOutcome")
        if not self.outcome.state.terminal:
            raise ValueError("result requires a terminal outcome")
        object.__setattr__(self, "outcome", IngestionOutcome.from_dict(self.outcome.to_dict()))
        object.__setattr__(self, "outcome_digest", digest_text(self.outcome_digest, "outcome_digest"))
        if self.outcome_digest != self.outcome.outcome_digest:
            raise ValueError("outcome_digest does not bind the outcome")

    def to_dict(self) -> dict[str, object]: return {"schema_version": self.schema_version, "outcome": self.outcome.to_dict(), "outcome_digest": self.outcome_digest}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionResult":
        value = exact_fields(value, frozenset({"schema_version", "outcome", "outcome_digest"}), "ingestion_result")
        return cls(value["schema_version"], IngestionOutcome.from_dict(value["outcome"]), value["outcome_digest"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionVerification:
    run: IngestionRunRef
    outcome_digest: str
    bundle: NormalizedBundleRef
    verified: bool
    checked_at: str
    diagnostic_codes: tuple[IngestionDiagnosticCode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.run) is not IngestionRunRef or type(self.bundle) is not NormalizedBundleRef:
            raise TypeError("run/bundle have invalid types")
        object.__setattr__(self, "run", IngestionRunRef.from_dict(self.run.to_dict()))
        object.__setattr__(self, "bundle", NormalizedBundleRef.from_dict(self.bundle.to_dict()))
        object.__setattr__(self, "outcome_digest", digest_text(self.outcome_digest, "outcome_digest"))
        object.__setattr__(self, "checked_at", _timestamp_text(self.checked_at, "checked_at"))
        _bool(self.verified, "verified")
        diagnostics = _diagnostics(self.diagnostic_codes)
        if self.bundle.plan_fingerprint != self.run.plan_fingerprint or self.bundle.run_authority_digest != self.run.authority_digest:
            raise ValueError("bundle does not bind the run")
        if self.verified == bool(diagnostics):
            raise ValueError("verified/diagnostic_codes matrix invalid")
        object.__setattr__(self, "diagnostic_codes", diagnostics)

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "outcome_digest": self.outcome_digest, "bundle": self.bundle.to_dict(), "verified": self.verified,
                "checked_at": self.checked_at, "diagnostic_codes": [item.value for item in self.diagnostic_codes]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionVerification":
        value = exact_fields(value, frozenset({"run", "outcome_digest", "bundle", "verified", "checked_at", "diagnostic_codes"}), "ingestion_verification")
        return cls(IngestionRunRef.from_dict(value["run"]), value["outcome_digest"], NormalizedBundleRef.from_dict(value["bundle"]), value["verified"], value["checked_at"],
                   tuple(_enum(IngestionDiagnosticCode, item, "diagnostic_code") for item in _array(value["diagnostic_codes"], "diagnostic_codes")))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionListRequest:
    project_ref: str
    cursor: str | None = None
    limit: int = MAX_LIST_PAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_ref", _identity(self.project_ref, "project_ref"))
        object.__setattr__(self, "cursor", _cursor(self.cursor))
        _count(self.limit, "limit", minimum=1, maximum=MAX_LIST_PAGE)

    def to_dict(self) -> dict[str, object]: return {"project_ref": self.project_ref, "cursor": self.cursor, "limit": self.limit}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionListRequest":
        value = exact_fields(value, frozenset({"project_ref", "cursor", "limit"}), "ingestion_list_request")
        return cls(value["project_ref"], value["cursor"], value["limit"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionPage:
    request: IngestionListRequest
    outcomes: tuple[IngestionOutcome, ...]
    next_cursor: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not IngestionListRequest:
            raise TypeError("request must be exact IngestionListRequest")
        request = IngestionListRequest.from_dict(self.request.to_dict())
        outcomes = _tuple(self.outcomes, IngestionOutcome, "outcomes", request.limit)
        if any(item.run.project_ref != request.project_ref for item in outcomes):
            raise ValueError("outcomes do not bind the request")
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "next_cursor", _cursor(self.next_cursor))
        _bool(self.truncated, "truncated")
        if self.truncated != (self.next_cursor is not None) or (self.truncated and not outcomes):
            raise ValueError("next_cursor/truncated matrix invalid")

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "outcomes": [item.to_dict() for item in self.outcomes], "next_cursor": self.next_cursor, "truncated": self.truncated}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionPage":
        value = exact_fields(value, frozenset({"request", "outcomes", "next_cursor", "truncated"}), "ingestion_page")
        return cls(IngestionListRequest.from_dict(value["request"]), _parse_tuple(value["outcomes"], IngestionOutcome, "outcomes"), value["next_cursor"], value["truncated"])  # type: ignore[arg-type]


class IngestionObservationKind(str, Enum):
    SOURCE_PROCESSED = "source_processed"
    DOCUMENT_WRITTEN = "document_written"


@dataclass(frozen=True, slots=True)
class IngestionObservation:
    run: IngestionRunRef
    sequence: int
    occurred_at: str
    kind: IngestionObservationKind
    completed_count: int

    def __post_init__(self) -> None:
        if type(self.run) is not IngestionRunRef or type(self.kind) is not IngestionObservationKind:
            raise TypeError("run/kind have invalid types")
        object.__setattr__(self, "run", IngestionRunRef.from_dict(self.run.to_dict()))
        _count(self.sequence, "sequence", maximum=2**63 - 1)
        object.__setattr__(self, "occurred_at", _timestamp_text(self.occurred_at, "occurred_at"))
        _count(self.completed_count, "completed_count")

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "sequence": self.sequence, "occurred_at": self.occurred_at, "kind": self.kind.value, "completed_count": self.completed_count}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionObservation":
        value = exact_fields(value, frozenset({"run", "sequence", "occurred_at", "kind", "completed_count"}), "ingestion_observation")
        return cls(IngestionRunRef.from_dict(value["run"]), value["sequence"], value["occurred_at"], _enum(IngestionObservationKind, value["kind"], "kind"), value["completed_count"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionObservationsRequest:
    run: IngestionRunRef
    after_sequence: int | None = None
    limit: int = MAX_OBSERVATION_PAGE

    def __post_init__(self) -> None:
        if type(self.run) is not IngestionRunRef:
            raise TypeError("run must be exact IngestionRunRef")
        object.__setattr__(self, "run", IngestionRunRef.from_dict(self.run.to_dict()))
        if self.after_sequence is not None:
            _count(self.after_sequence, "after_sequence", maximum=2**63 - 1)
        _count(self.limit, "limit", minimum=1, maximum=MAX_OBSERVATION_PAGE)

    def to_dict(self) -> dict[str, object]: return {"run": self.run.to_dict(), "after_sequence": self.after_sequence, "limit": self.limit}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionObservationsRequest":
        value = exact_fields(value, frozenset({"run", "after_sequence", "limit"}), "ingestion_observations_request")
        return cls(IngestionRunRef.from_dict(value["run"]), value["after_sequence"], value["limit"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class IngestionObservationPage:
    request: IngestionObservationsRequest
    records: tuple[IngestionObservation, ...]
    next_cursor: int | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.request) is not IngestionObservationsRequest:
            raise TypeError("request must be exact IngestionObservationsRequest")
        request = IngestionObservationsRequest.from_dict(self.request.to_dict())
        records = _tuple(self.records, IngestionObservation, "records", request.limit)
        if any(item.run != request.run for item in records):
            raise ValueError("records do not bind the request")
        sequences = tuple(item.sequence for item in records)
        if any(left >= right for left, right in zip(sequences, sequences[1:])):
            raise ValueError("observation sequences must be strictly increasing")
        if request.after_sequence is not None and sequences and sequences[0] <= request.after_sequence:
            raise ValueError("observations must follow after_sequence")
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "records", records)
        if self.next_cursor is not None:
            _count(self.next_cursor, "next_cursor", maximum=2**63 - 1)
        _bool(self.truncated, "truncated")
        if self.truncated != (self.next_cursor is not None) or (self.truncated and not records):
            raise ValueError("next_cursor/truncated matrix invalid")
        if self.truncated and self.next_cursor != sequences[-1]:
            raise ValueError("next_cursor must equal the last record sequence")

    def to_dict(self) -> dict[str, object]:
        return {"request": self.request.to_dict(), "records": [item.to_dict() for item in self.records], "next_cursor": self.next_cursor, "truncated": self.truncated}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IngestionObservationPage":
        value = exact_fields(value, frozenset({"request", "records", "next_cursor", "truncated"}), "ingestion_observation_page")
        return cls(IngestionObservationsRequest.from_dict(value["request"]), _parse_tuple(value["records"], IngestionObservation, "records"), value["next_cursor"], value["truncated"])  # type: ignore[arg-type]


class IngestionOperations(Protocol):
    def admit(self, request: SourceAdmissionRequest) -> SourceSnapshotRef: ...
    def propose(self, request: StructureProposalRequest) -> tuple[StructureProposal, ...]: ...
    def plan(self, request: IngestionRequest) -> IngestionPlan: ...
    def preflight(self, plan: IngestionPlan) -> IngestionPreflight: ...
    def start(self, plan: IngestionPlan, preflight: IngestionPreflight) -> IngestionStart: ...
    def show(self, run: IngestionRunRef) -> IngestionOutcome: ...
    def result(self, outcome: IngestionOutcome) -> IngestionResult: ...
    def cancel(self, outcome: IngestionOutcome) -> IngestionOutcome: ...
    def resume(self, outcome: IngestionOutcome) -> IngestionOutcome: ...
    def reconcile(self, outcome: IngestionOutcome) -> IngestionOutcome: ...
    def verify(self, result: IngestionResult) -> IngestionVerification: ...
    def list(self, request: IngestionListRequest) -> IngestionPage: ...
    def observations(self, request: IngestionObservationsRequest) -> IngestionObservationPage: ...


class Clock(Protocol):
    def now(self) -> str: ...


class _OperationVerb(str, Enum):
    ADMIT = "admit"
    PROPOSE = "propose"
    PLAN = "plan"
    PREFLIGHT = "preflight"
    START = "start"
    SHOW = "show"
    RESULT = "result"
    CANCEL = "cancel"
    RESUME = "resume"
    RECONCILE = "reconcile"
    VERIFY = "verify"
    LIST = "list"
    OBSERVATIONS = "observations"


class IngestionAPI:
    """Sanitized detach-and-revalidate boundary over host operations."""

    __slots__ = ("_operations", "_clock")

    def __init__(self, operations: IngestionOperations, *, clock: Clock) -> None:
        self._operations = operations
        self._clock = clock

    @staticmethod
    def _copy(value, expected: type):
        if type(value) is not expected:
            raise TypeError(f"value must be exact {expected.__name__}")
        return expected.from_dict(value.to_dict())

    @staticmethod
    def _same(value, baseline, copier) -> bool:
        try:
            return copier(value) == baseline
        except BaseException:
            return False

    def _call(self, verb: _OperationVerb, originals: tuple, baselines: tuple, presented: tuple, copiers: tuple):
        failure_code: IngestionOperationCode | None = None
        result = None
        try:
            if type(verb) is not _OperationVerb:
                raise TypeError
            callback = getattr(self._operations, verb.value)
            result = callback(*presented)
        except BaseException as failure:
            if type(failure) is IngestionOperationError:
                failure_code = failure.code
            else:
                failure_code = IngestionOperationCode.OPERATION_FAILED
        mutated = any(not self._same(value, baseline, copier) for value, baseline, copier in zip(originals, baselines, copiers))
        mutated = mutated or any(not self._same(value, baseline, copier) for value, baseline, copier in zip(presented, baselines, copiers))
        if mutated:
            raise IngestionOperationError(IngestionOperationCode.INPUT_MUTATED) from None
        if failure_code is not None:
            raise IngestionOperationError(failure_code) from None
        return result

    @staticmethod
    def _rebuilt(value, expected: type):
        failed = False
        rebuilt = None
        try:
            if type(value) is not expected:
                raise TypeError
            rebuilt = expected.from_dict(value.to_dict())
        except BaseException:
            failed = True
        if failed:
            raise IngestionOperationError(IngestionOperationCode.RESULT_INVALID) from None
        return rebuilt

    @staticmethod
    def _unbound() -> None:
        raise IngestionOperationError(IngestionOperationCode.RESULT_UNBOUND) from None

    def _clock_now(self) -> datetime:
        failed = False
        now = None
        try:
            now = _timestamp(self._clock.now(), "now")
        except BaseException:
            failed = True
        if failed:
            raise IngestionOperationError(IngestionOperationCode.OPERATION_FAILED) from None
        return now  # type: ignore[return-value]

    def _one(self, verb: _OperationVerb, value, expected: type):
        copier = lambda item: self._copy(item, expected)
        baseline = copier(value)
        presented = copier(baseline)
        return self._call(verb, (value,), (baseline,), (presented,), (copier,)), baseline

    def admit(self, request: SourceAdmissionRequest) -> SourceSnapshotRef:
        result, baseline = self._one(_OperationVerb.ADMIT, request, SourceAdmissionRequest)
        snapshot = self._rebuilt(result, SourceSnapshotRef)
        if snapshot.admission_fingerprint != baseline.admission_fingerprint or snapshot.project_ref != baseline.project_ref or snapshot.source_kind is not baseline.source.kind:
            self._unbound()
        return snapshot

    def propose(self, request: StructureProposalRequest) -> tuple[StructureProposal, ...]:
        result, baseline = self._one(_OperationVerb.PROPOSE, request, StructureProposalRequest)
        failed = False
        proposals: tuple[StructureProposal, ...] = ()
        try:
            if type(result) is not tuple or any(type(item) is not StructureProposal for item in result) or len(result) > MAX_PROPOSALS:
                raise TypeError
            proposals = tuple(StructureProposal.from_dict(item.to_dict()) for item in result)
            proposals = tuple(sorted(proposals, key=lambda item: item.proposal_id))
        except BaseException:
            failed = True
        if failed:
            raise IngestionOperationError(IngestionOperationCode.RESULT_INVALID) from None
        if len({item.proposal_id for item in proposals}) != len(proposals) or any(item.snapshot != baseline.snapshot for item in proposals):
            self._unbound()
        return proposals

    def plan(self, request: IngestionRequest) -> IngestionPlan:
        result, baseline = self._one(_OperationVerb.PLAN, request, IngestionRequest)
        plan = self._rebuilt(result, IngestionPlan)
        if plan.request != baseline:
            self._unbound()
        return plan

    def preflight(self, plan: IngestionPlan) -> IngestionPreflight:
        result, baseline = self._one(_OperationVerb.PREFLIGHT, plan, IngestionPlan)
        preflight = self._rebuilt(result, IngestionPreflight)
        if not preflight.binds(baseline) or (preflight.ready and not baseline.preview.ready):
            self._unbound()
        if baseline.preview.unmatched_sources and IngestionDiagnosticCode.UNMATCHED_SOURCE not in preflight.diagnostic_codes:
            self._unbound()
        if baseline.preview.ambiguous_sources and IngestionDiagnosticCode.AMBIGUOUS_SOURCE not in preflight.diagnostic_codes:
            self._unbound()
        now = self._clock_now()
        if _timestamp(preflight.checked_at, "checked_at") > now or now >= _timestamp(preflight.expires_at, "expires_at"):
            raise IngestionOperationError(IngestionOperationCode.RESULT_INVALID) from None
        return preflight

    def start(self, plan: IngestionPlan, preflight: IngestionPreflight) -> IngestionStart:
        plan_copy = lambda item: self._copy(item, IngestionPlan)
        preflight_copy = lambda item: self._copy(item, IngestionPreflight)
        plan_base, preflight_base = plan_copy(plan), preflight_copy(preflight)
        now = self._clock_now()
        checked = _timestamp(preflight_base.checked_at, "checked_at")
        expires = _timestamp(preflight_base.expires_at, "expires_at")
        if not preflight_base.ready or not plan_base.preview.ready or not preflight_base.binds(plan_base) or checked > now or now >= expires:
            raise IngestionOperationError(IngestionOperationCode.START_INELIGIBLE) from None
        result = self._call(_OperationVerb.START, (plan, preflight), (plan_base, preflight_base),
                            (plan_copy(plan_base), preflight_copy(preflight_base)), (plan_copy, preflight_copy))
        started = self._rebuilt(result, IngestionStart)
        run = started.run
        if not started.accepted:
            raise IngestionOperationError(IngestionOperationCode.START_INELIGIBLE) from None
        if run.project_ref != plan_base.request.project_ref or run.plan_fingerprint != plan_base.plan_fingerprint or run.preflight_fingerprint != preflight_base.preflight_fingerprint:
            self._unbound()
        return started

    def show(self, run: IngestionRunRef) -> IngestionOutcome:
        result, baseline = self._one(_OperationVerb.SHOW, run, IngestionRunRef)
        outcome = self._rebuilt(result, IngestionOutcome)
        if outcome.run != baseline:
            self._unbound()
        return outcome

    def result(self, outcome: IngestionOutcome) -> IngestionResult:
        if type(outcome) is not IngestionOutcome:
            raise TypeError("outcome must be exact IngestionOutcome")
        if not outcome.state.terminal:
            raise IngestionOperationError(IngestionOperationCode.RESULT_UNAVAILABLE) from None
        value, baseline = self._one(_OperationVerb.RESULT, outcome, IngestionOutcome)
        result = self._rebuilt(value, IngestionResult)
        if result.outcome != baseline or result.outcome_digest != baseline.outcome_digest:
            self._unbound()
        return result

    def cancel(self, outcome: IngestionOutcome) -> IngestionOutcome:
        return self._transition(_OperationVerb.CANCEL, outcome, {IngestionRunState.PLANNED, IngestionRunState.RUNNING},
                                {IngestionRunState.CANCEL_REQUESTED, IngestionRunState.CANCELLED}, IngestionOperationCode.CANCEL_INELIGIBLE)

    def resume(self, outcome: IngestionOutcome) -> IngestionOutcome:
        if type(outcome) is not IngestionOutcome or outcome.state is not IngestionRunState.FAILED or outcome.diagnostic_code is not IngestionDiagnosticCode.INTERRUPTED:
            raise IngestionOperationError(IngestionOperationCode.RESUME_INELIGIBLE) from None
        resumed = self._transition(_OperationVerb.RESUME, outcome, {IngestionRunState.FAILED}, {IngestionRunState.RUNNING}, IngestionOperationCode.RESUME_INELIGIBLE)
        if (resumed.sources_processed, resumed.documents_written) != (outcome.sources_processed, outcome.documents_written):
            self._unbound()
        return resumed

    def reconcile(self, outcome: IngestionOutcome) -> IngestionOutcome:
        return self._transition(_OperationVerb.RECONCILE, outcome, {IngestionRunState.RECONCILE_REQUIRED},
                                {IngestionRunState.RUNNING, IngestionRunState.SUCCEEDED, IngestionRunState.FAILED}, IngestionOperationCode.RECONCILE_INELIGIBLE)

    def _transition(self, verb: _OperationVerb, outcome: IngestionOutcome, sources: set[IngestionRunState], targets: set[IngestionRunState], code: IngestionOperationCode) -> IngestionOutcome:
        if type(outcome) is not IngestionOutcome:
            raise TypeError("outcome must be exact IngestionOutcome")
        if outcome.state not in sources:
            raise IngestionOperationError(code) from None
        result, baseline = self._one(verb, outcome, IngestionOutcome)
        transitioned = self._rebuilt(result, IngestionOutcome)
        if transitioned.run != baseline.run or transitioned.source_count != baseline.source_count or transitioned.state not in targets:
            self._unbound()
        if transitioned.sources_processed < baseline.sources_processed or transitioned.documents_written < baseline.documents_written:
            self._unbound()
        return transitioned

    def verify(self, result: IngestionResult) -> IngestionVerification:
        if type(result) is not IngestionResult:
            raise TypeError("result must be exact IngestionResult")
        if result.outcome.state is not IngestionRunState.SUCCEEDED:
            raise IngestionOperationError(IngestionOperationCode.VERIFY_INELIGIBLE) from None
        value, baseline = self._one(_OperationVerb.VERIFY, result, IngestionResult)
        verification = self._rebuilt(value, IngestionVerification)
        if verification.run != baseline.outcome.run or verification.outcome_digest != baseline.outcome_digest or verification.bundle != baseline.outcome.bundle:
            self._unbound()
        return verification

    def list(self, request: IngestionListRequest) -> IngestionPage:
        value, baseline = self._one(_OperationVerb.LIST, request, IngestionListRequest)
        page = self._rebuilt(value, IngestionPage)
        if page.request != baseline:
            self._unbound()
        return page

    def observations(self, request: IngestionObservationsRequest) -> IngestionObservationPage:
        value, baseline = self._one(_OperationVerb.OBSERVATIONS, request, IngestionObservationsRequest)
        page = self._rebuilt(value, IngestionObservationPage)
        if page.request != baseline:
            self._unbound()
        return page


__all__ = [
    "AuthorizedSourceRef", "BindingPreview", "FieldMapping", "FieldSelector", "FieldSelectorKind", "FieldValueKind",
    "FrontmatterMode", "INGESTION_PLAN_SCHEMA_VERSION", "INGESTION_RESULT_SCHEMA_VERSION", "IngestionAPI", "IngestionDiagnosticCode", "IngestionListRequest", "IngestionObservation",
    "IngestionObservationKind", "IngestionObservationPage", "IngestionObservationsRequest", "IngestionOperationCode",
    "IngestionOperationError", "IngestionOperations", "IngestionOutcome", "IngestionPage", "IngestionPlan", "IngestionPreflight",
    "IngestionPreview", "IngestionRequest", "IngestionResult", "IngestionRunRef", "IngestionRunState", "IngestionStart",
    "IngestionVerification", "MarkdownProfileV1", "MetadataDeclaration", "MetadataPolicyRef", "NormalizedBundleRef",
    "ParsingProfile", "ProposalEvidenceCode", "RelationshipDeclaration", "SchemaRef", "SourceAdmissionKind", "SourceAdmissionRequest",
    "SourceMatcher", "SourceSnapshotRef", "StructureBinding", "StructureDefinition", "StructureProposal", "StructureProposalRequest",
    "StructureRef", "StructureSet", "TextProjection", "UnitBoundary", "run_authority_digest", "validate_ingestion_identity",
]
