"""Pure, bounded Markdown plus YAML-frontmatter normalization for ingestion V1."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import yaml
from yaml.events import (
    AliasEvent,
    DocumentEndEvent,
    DocumentStartEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
    StreamEndEvent,
    StreamStartEvent,
)
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    MAX_FRONTMATTER_BYTES,
    MAX_YAML_DEPTH,
    MAX_YAML_KEY_BYTES,
    MAX_YAML_MAPPING_ENTRIES,
    MAX_YAML_NODES,
    MAX_YAML_SCALAR_BYTES,
    MAX_YAML_SEQUENCE_ENTRIES,
    StructureDefinition,
    YAML_FLOAT_PATTERN,
    YAML_INTEGER_PATTERN,
    YAML_UNQUOTED_NULL_LITERALS_REJECTED,
)

from .local_selection_v1 import MAX_FILE_BYTES


_INTEGER = re.compile(rf"^(?:{YAML_INTEGER_PATTERN})$")
_FLOAT = re.compile(rf"^(?:{YAML_FLOAT_PATTERN})$")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_NONFINITE = frozenset({".nan", ".inf", "+.inf", "-.inf"})


class MarkdownParseCodeV1(str, Enum):
    INVALID_ENCODING = "invalid_encoding"
    CONTENT_TOO_LARGE = "content_too_large"
    FRONTMATTER_REQUIRED = "frontmatter_required"
    FRONTMATTER_UNCLOSED = "frontmatter_unclosed"
    FRONTMATTER_INVALID = "frontmatter_invalid"
    BODY_EMPTY = "body_empty"
    FIELD_MAPPING_INVALID = "field_mapping_invalid"


class MarkdownParseErrorV1(ValueError):
    """Closed parser failure that never exposes source text or YAML prose."""

    __slots__ = ("code",)

    def __init__(self, code: MarkdownParseCodeV1) -> None:
        if type(code) is not MarkdownParseCodeV1:
            raise TypeError("code must be exact MarkdownParseCodeV1")
        self.code = code
        super().__init__(code.value)


def _raise(code: MarkdownParseCodeV1) -> None:
    raise MarkdownParseErrorV1(code) from None


def _utf8_size(value: object) -> int | None:
    if type(value) is not str:
        return None
    try:
        encoded = value.encode("utf-8")
    except BaseException:
        return None
    return len(encoded)


def _exact_dict_items(
    value: dict[object, object],
) -> tuple[tuple[object, object], ...] | None:
    items: tuple[tuple[object, object], ...] | None = None
    failed = False
    try:
        items = tuple(value.items())
    except BaseException:
        failed = True
    if failed or type(items) is not tuple:
        return None
    return items


@dataclass(frozen=True, slots=True)
class ParsedMarkdownV1:
    normalized_text: str
    body: str
    frontmatter: dict[str, object]
    frontmatter_mode: FrontmatterMode
    had_frontmatter: bool

    def __post_init__(self) -> None:
        if type(self.normalized_text) is not str or type(self.body) is not str:
            raise TypeError("normalized_text and body must be exact strings")
        normalized_size = _utf8_size(self.normalized_text)
        body_size = _utf8_size(self.body)
        if (
            normalized_size is None
            or body_size is None
            or normalized_size > MAX_FILE_BYTES
            or body_size > MAX_FILE_BYTES
            or not self.body
        ):
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        if type(self.frontmatter) is not dict:
            raise TypeError("frontmatter must be an exact built-in dict")
        if type(self.frontmatter_mode) is not FrontmatterMode:
            raise TypeError("frontmatter_mode must be exact FrontmatterMode")
        if type(self.had_frontmatter) is not bool:
            raise TypeError("had_frontmatter must be an exact boolean")
        sanitized = _sanitize_value(
            self.frontmatter, depth=1, budget=_YamlBudget(), mapping_root=True
        )
        if type(sanitized) is not MappingProxyType:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        object.__setattr__(self, "frontmatter", sanitized)


@dataclass(slots=True)
class _YamlBudget:
    nodes: int = 0
    mapping_entries: int = 0
    sequence_entries: int = 0


@dataclass(slots=True)
class _EventFrame:
    kind: str
    direct_children: int = 0


def _sanitize_value(
    value: object, *, depth: int, budget: _YamlBudget, mapping_root: bool = False
) -> object:
    if depth > MAX_YAML_DEPTH:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    budget.nodes += 1
    if budget.nodes > MAX_YAML_NODES:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if type(value) is dict:
        entry_count = len(value)
        if entry_count > MAX_YAML_MAPPING_ENTRIES - budget.mapping_entries:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        items = _exact_dict_items(value)
        if items is None or len(items) != entry_count:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        budget.mapping_entries += entry_count
        result: dict[str, object] = {}
        normalized_keys: set[str] = set()
        for key, item in items:
            if type(key) is not str:
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            budget.nodes += 1
            key_size = _utf8_size(key)
            if budget.nodes > MAX_YAML_NODES or key_size is None:
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            if (
                not key
                or key == "<<"
                or key_size > MAX_YAML_KEY_BYTES
                or unicodedata.normalize("NFC", key) != key
                or key in normalized_keys
            ):
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            normalized_keys.add(key)
            result[key] = _sanitize_value(item, depth=depth + 1, budget=budget)
        return MappingProxyType(result)
    if mapping_root:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if type(value) in {list, tuple}:
        budget.sequence_entries += len(value)
        if budget.sequence_entries > MAX_YAML_SEQUENCE_ENTRIES:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        return tuple(
            _sanitize_value(item, depth=depth + 1, budget=budget) for item in value
        )
    if type(value) is str:
        size = _utf8_size(value)
        if size is None or size > MAX_YAML_SCALAR_BYTES:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        return value
    if type(value) is bool:
        return value
    if type(value) is int and _INT64_MIN <= value <= _INT64_MAX:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)


def _plain_scalar(value: str) -> object:
    if value in YAML_UNQUOTED_NULL_LITERALS_REJECTED:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if value.casefold() in _NONFINITE:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if value == "true":
        return True
    if value == "false":
        return False
    if _INTEGER.fullmatch(value) is not None:
        parsed = int(value, 10)
        if not _INT64_MIN <= parsed <= _INT64_MAX:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        return parsed
    if _FLOAT.fullmatch(value) is not None and any(token in value for token in ".eE"):
        parsed_float = float(value)
        if not math.isfinite(parsed_float):
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        return parsed_float
    return value


def _convert_scalar(node: ScalarNode) -> object:
    size = _utf8_size(node.value)
    if size is None or size > MAX_YAML_SCALAR_BYTES:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if node.style is None:
        return _plain_scalar(node.value)
    return node.value


def _convert_yaml(node: Node, *, depth: int, budget: _YamlBudget) -> object:
    if depth > MAX_YAML_DEPTH:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    budget.nodes += 1
    if budget.nodes > MAX_YAML_NODES:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if type(node) is ScalarNode:
        return _convert_scalar(node)
    if type(node) is SequenceNode:
        budget.sequence_entries += len(node.value)
        if budget.sequence_entries > MAX_YAML_SEQUENCE_ENTRIES:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        return [
            _convert_yaml(item, depth=depth + 1, budget=budget) for item in node.value
        ]
    if type(node) is MappingNode:
        budget.mapping_entries += len(node.value)
        if budget.mapping_entries > MAX_YAML_MAPPING_ENTRIES:
            _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
        result: dict[str, object] = {}
        normalized_keys: set[str] = set()
        for key_node, value_node in node.value:
            if type(key_node) is not ScalarNode:
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            key_value = _convert_yaml(key_node, depth=depth + 1, budget=budget)
            if type(key_value) is not str or not key_value or key_value == "<<":
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            normalized = unicodedata.normalize("NFC", key_value)
            if normalized in normalized_keys:
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            key_size = _utf8_size(key_value)
            if (
                normalized != key_value
                or key_size is None
                or key_size > MAX_YAML_KEY_BYTES
            ):
                _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
            normalized_keys.add(normalized)
            result[key_value] = _convert_yaml(
                value_node, depth=depth + 1, budget=budget
            )
        return result
    _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)


def _events_within_limits(value: str) -> bool:
    documents = 0
    document_roots = 0
    nodes = 0
    mapping_entries = 0
    sequence_entries = 0
    frames: list[_EventFrame] = []
    failed = False

    def note_node() -> bool:
        nonlocal document_roots, nodes
        nodes += 1
        if nodes > MAX_YAML_NODES:
            return False
        if frames:
            frames[-1].direct_children += 1
        else:
            document_roots += 1
        return True

    try:
        for event in yaml.parse(value, Loader=yaml.BaseLoader):
            if type(event) in {StreamStartEvent, StreamEndEvent}:
                continue
            if type(event) is DocumentStartEvent:
                documents += 1
                document_roots = 0
                if (
                    documents > 1
                    or event.version is not None
                    or event.tags is not None
                    or frames
                ):
                    return False
                continue
            if type(event) is DocumentEndEvent:
                if frames or document_roots != 1:
                    return False
                continue
            if type(event) is AliasEvent:
                return False
            if type(event) is ScalarEvent:
                scalar_size = _utf8_size(event.value)
                is_mapping_key = bool(
                    frames
                    and frames[-1].kind == "mapping"
                    and frames[-1].direct_children % 2 == 0
                )
                if (
                    event.anchor is not None
                    or event.tag is not None
                    or scalar_size is None
                    or scalar_size > MAX_YAML_SCALAR_BYTES
                    or len(frames) + 1 > MAX_YAML_DEPTH
                    or (
                        is_mapping_key
                        and (
                            not event.value
                            or event.value == "<<"
                            or scalar_size > MAX_YAML_KEY_BYTES
                            or unicodedata.normalize("NFC", event.value) != event.value
                        )
                    )
                    or not note_node()
                ):
                    return False
                continue
            if type(event) in {MappingStartEvent, SequenceStartEvent}:
                is_complex_mapping_key = bool(
                    frames
                    and frames[-1].kind == "mapping"
                    and frames[-1].direct_children % 2 == 0
                )
                if (
                    event.anchor is not None
                    or event.tag is not None
                    or is_complex_mapping_key
                    or not note_node()
                    or len(frames) + 1 > MAX_YAML_DEPTH
                ):
                    return False
                frames.append(
                    _EventFrame(
                        "mapping" if type(event) is MappingStartEvent else "sequence"
                    )
                )
                continue
            if type(event) is MappingEndEvent:
                if not frames or frames[-1].kind != "mapping":
                    return False
                frame = frames.pop()
                if frame.direct_children % 2:
                    return False
                mapping_entries += frame.direct_children // 2
                if mapping_entries > MAX_YAML_MAPPING_ENTRIES:
                    return False
                continue
            if type(event) is SequenceEndEvent:
                if not frames or frames[-1].kind != "sequence":
                    return False
                frame = frames.pop()
                sequence_entries += frame.direct_children
                if sequence_entries > MAX_YAML_SEQUENCE_ENTRIES:
                    return False
                continue
            return False
    except BaseException:
        failed = True
    return not failed and documents == 1 and not frames


def _parse_frontmatter(value: str) -> dict[str, object]:
    frontmatter_size = _utf8_size(value)
    if frontmatter_size is None or frontmatter_size > MAX_FRONTMATTER_BYTES:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    if not _events_within_limits(value):
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    node: Node | None = None
    compose_failed = False
    try:
        node = yaml.compose(value, Loader=yaml.BaseLoader)
    except BaseException:
        compose_failed = True
    if compose_failed or type(node) is not MappingNode:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    converted = _convert_yaml(node, depth=1, budget=_YamlBudget())
    if type(converted) is not dict:
        _raise(MarkdownParseCodeV1.FRONTMATTER_INVALID)
    return converted


def _split_frontmatter(value: str) -> tuple[str, str]:
    remainder = value[4:]
    line_start = 0
    while True:
        line_end = remainder.find("\n", line_start)
        if line_end < 0:
            if remainder[line_start:] == "---":
                return remainder[:line_start], ""
            break
        if remainder[line_start:line_end] == "---":
            return remainder[:line_start], remainder[line_end + 1 :]
        line_start = line_end + 1
    _raise(MarkdownParseCodeV1.FRONTMATTER_UNCLOSED)


def parse_markdown_v1(
    content: bytes, frontmatter_mode: FrontmatterMode
) -> ParsedMarkdownV1:
    """Normalize and dry-parse one immutable Markdown source."""

    if type(content) is not bytes:
        raise TypeError("content must be exact bytes")
    if type(frontmatter_mode) is not FrontmatterMode:
        raise TypeError("frontmatter_mode must be exact FrontmatterMode")
    if len(content) > MAX_FILE_BYTES:
        _raise(MarkdownParseCodeV1.CONTENT_TOO_LARGE)
    normalized: str | None = None
    decode_failed = False
    try:
        normalized = content.decode("utf-8")
    except BaseException:
        decode_failed = True
    if decode_failed or type(normalized) is not str:
        _raise(MarkdownParseCodeV1.INVALID_ENCODING)
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    has_opener = normalized.startswith("---\n")
    if frontmatter_mode is FrontmatterMode.NONE:
        body = normalized
        metadata: dict[str, object] = {}
        had_frontmatter = False
    elif not has_opener:
        if frontmatter_mode is FrontmatterMode.REQUIRED:
            _raise(MarkdownParseCodeV1.FRONTMATTER_REQUIRED)
        body = normalized
        metadata = {}
        had_frontmatter = False
    else:
        frontmatter_text, body = _split_frontmatter(normalized)
        metadata = _parse_frontmatter(frontmatter_text)
        had_frontmatter = True
    if body == "":
        _raise(MarkdownParseCodeV1.BODY_EMPTY)
    return ParsedMarkdownV1(
        normalized, body, metadata, frontmatter_mode, had_frontmatter
    )


def _valid_kind(value: object, kind: FieldValueKind) -> bool:
    if kind is FieldValueKind.STRING:
        return type(value) is str
    if kind is FieldValueKind.BOOLEAN:
        return type(value) is bool
    if kind is FieldValueKind.INT64:
        return type(value) is int and _INT64_MIN <= value <= _INT64_MAX
    if kind is FieldValueKind.NUMBER:
        return (type(value) is int and _INT64_MIN <= value <= _INT64_MAX) or (
            type(value) is float and math.isfinite(value)
        )
    if kind is FieldValueKind.ARRAY:
        return type(value) is tuple
    if kind is FieldValueKind.OBJECT:
        return type(value) is MappingProxyType
    return False


def _validate_logical_path(value: object) -> str:
    path_size = _utf8_size(value)
    if (
        type(value) is not str
        or path_size is None
        or unicodedata.normalize("NFC", value) != value
    ):
        _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
    segments = value.split("/")
    if (
        value.startswith("/")
        or value.endswith("/")
        or any(
            not segment
            or segment in {".", ".."}
            or any(
                character in "\\:" or ord(character) < 32 or ord(character) == 127
                for character in segment
            )
            for segment in segments
        )
    ):
        _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
    if len(segments) > 64 or path_size > 1_024:
        _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
    return value


def map_markdown_fields_v1(
    parsed: ParsedMarkdownV1,
    logical_path: str,
    structure_definition: StructureDefinition,
) -> dict[str, object]:
    """Apply only the structure's explicit field selectors and value kinds."""

    if type(parsed) is not ParsedMarkdownV1:
        raise TypeError("parsed must be exact ParsedMarkdownV1")
    if type(structure_definition) is not StructureDefinition:
        raise TypeError("structure_definition must be exact StructureDefinition")
    if parsed.frontmatter_mode is not structure_definition.markdown.frontmatter_mode:
        _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
    path = _validate_logical_path(logical_path)
    result: dict[str, object] = {}
    for field in structure_definition.fields:
        selector = field.selector
        present = True
        if selector.kind is FieldSelectorKind.DOCUMENT_BODY:
            value: object = parsed.body
        elif selector.kind is FieldSelectorKind.LOGICAL_PATH:
            value = path
        elif selector.kind is FieldSelectorKind.FRONTMATTER_FIELD:
            if selector.key not in parsed.frontmatter:
                present = False
                value = None
            else:
                value = parsed.frontmatter[selector.key]
        else:  # pragma: no cover - facade enum is closed
            _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
        if not present:
            if field.required:
                _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
            continue
        if not _valid_kind(value, field.value_kind):
            _raise(MarkdownParseCodeV1.FIELD_MAPPING_INVALID)
        result[field.name] = value
    return result


__all__ = [
    "MarkdownParseCodeV1",
    "MarkdownParseErrorV1",
    "ParsedMarkdownV1",
    "map_markdown_fields_v1",
    "parse_markdown_v1",
]
