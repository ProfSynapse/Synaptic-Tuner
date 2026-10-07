from __future__ import annotations

from types import MappingProxyType

import pytest

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FieldSelector,
    FieldSelectorKind,
    FieldValueKind,
    FrontmatterMode,
    MarkdownProfileV1,
    ParsingProfile,
    StructureDefinition,
    TextProjection,
)
from tuner.ingestion.markdown_v1 import (
    MarkdownParseCodeV1,
    MarkdownParseErrorV1,
    ParsedMarkdownV1,
    map_markdown_fields_v1,
    parse_markdown_v1,
)
import tuner.ingestion.markdown_v1 as markdown


def test_normalizes_one_bom_and_newlines_without_rewriting_body() -> None:
    parsed = parse_markdown_v1(
        b"\xef\xbb\xbfline one\r\nline two\rline three\n", FrontmatterMode.OPTIONAL
    )
    assert parsed.normalized_text == "line one\nline two\nline three\n"
    assert parsed.body == parsed.normalized_text
    assert parsed.frontmatter == {}
    assert parsed.had_frontmatter is False


def test_none_mode_treats_frontmatter_markers_as_verbatim_body() -> None:
    content = b"---\ntitle: Note\n---\nBody"
    parsed = parse_markdown_v1(content, FrontmatterMode.NONE)
    assert parsed.body == content.decode("utf-8")
    assert parsed.frontmatter == {}
    assert parsed.had_frontmatter is False


def test_closed_scalar_resolution_and_recursive_immutability() -> None:
    parsed = parse_markdown_v1(
        b"""---
truth: true
title: true story
quoted: "true"
integer: -12
number: 1.25e2
date: 2026-09-19
capitalized: True
nested:
  values: [one, 2, false]
---
Body
""",
        FrontmatterMode.REQUIRED,
    )
    assert parsed.frontmatter["truth"] is True
    assert parsed.frontmatter["title"] == "true story"
    assert parsed.frontmatter["quoted"] == "true"
    assert parsed.frontmatter["integer"] == -12
    assert parsed.frontmatter["number"] == 125.0
    assert parsed.frontmatter["date"] == "2026-09-19"
    assert parsed.frontmatter["capitalized"] == "True"
    nested = parsed.frontmatter["nested"]
    assert type(nested) is MappingProxyType
    assert nested["values"] == ("one", 2, False)  # type: ignore[index]
    with pytest.raises(TypeError):
        parsed.frontmatter["truth"] = False  # type: ignore[index]


def test_mapping_and_sequence_nulls_are_supported_without_coercing_quotes() -> None:
    parsed = parse_markdown_v1(
        b'''---
mapping:
  lower: null
  capitalized: Null
  upper: NULL
  tilde: ~
  empty:
sequence:
  - null
  - ~
  -
quoted: "null"
---
Body
''',
        FrontmatterMode.REQUIRED,
        ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V2,
    )

    mapping = parsed.frontmatter["mapping"]
    assert type(mapping) is MappingProxyType
    assert dict(mapping) == {
        "lower": None,
        "capitalized": None,
        "upper": None,
        "tilde": None,
        "empty": None,
    }
    assert parsed.frontmatter["sequence"] == (None, None, None)
    assert parsed.frontmatter["quoted"] == "null"


@pytest.mark.parametrize("literal", ("null", "Null", "NULL", "~"))
def test_v1_profile_preserves_null_rejection(literal: str) -> None:
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        parse_markdown_v1(
            f"---\nvalue: {literal}\n---\nBody".encode("utf-8"),
            FrontmatterMode.REQUIRED,
            ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1,
        )
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID


def test_empty_scalar_is_profile_versioned() -> None:
    content = b"---\nvalue:\n---\nBody"
    legacy = parse_markdown_v1(
        content,
        FrontmatterMode.REQUIRED,
        ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1,
    )
    current = parse_markdown_v1(
        content,
        FrontmatterMode.REQUIRED,
        ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V2,
    )
    assert legacy.frontmatter["value"] == ""
    assert current.frontmatter["value"] is None


@pytest.mark.parametrize(
    "frontmatter",
    (
        "value: .nan",
        "value: !!str tagged",
        "base: &base {x: 1}\ncopy: *base",
        "a: 1\na: 2",
        "'e\u0301': one\n'\xe9': two",
        "<<: {x: 1}",
        "- not\n- a\n- mapping",
        "",
    ),
)
def test_rejects_features_outside_the_closed_yaml_subset(frontmatter: str) -> None:
    content = f"---\n{frontmatter}\n---\nBody".encode("utf-8")
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        parse_markdown_v1(content, FrontmatterMode.REQUIRED)
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID


def test_frontmatter_delimiters_are_exact_and_modes_are_closed() -> None:
    with pytest.raises(MarkdownParseErrorV1) as missing:
        parse_markdown_v1(b"Body", FrontmatterMode.REQUIRED)
    assert missing.value.code is MarkdownParseCodeV1.FRONTMATTER_REQUIRED
    with pytest.raises(MarkdownParseErrorV1) as unclosed:
        parse_markdown_v1(b"---\ntitle: Note\nBody", FrontmatterMode.OPTIONAL)
    assert unclosed.value.code is MarkdownParseCodeV1.FRONTMATTER_UNCLOSED
    parsed = parse_markdown_v1(
        b"--- \ntitle: Note\n---\nBody", FrontmatterMode.OPTIONAL
    )
    assert parsed.had_frontmatter is False


@pytest.mark.parametrize("profile", [ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1, ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V2])
def test_nested_rich_metadata_with_242_aggregate_mapping_entries(profile) -> None:
    groups = "\n".join(f"    - name: group_{index}\n      category: example\n      enabled: true" for index in range(71))
    settings = "\n".join(f"    setting_{index}: value_{index}" for index in range(25))
    text = f"---\ntitle: Public fixture\nmetadata:\n  groups:\n{groups}\n  settings:\n{settings}\n---\nBody"
    parsed = parse_markdown_v1(text.encode(), FrontmatterMode.REQUIRED, parsing_profile=profile)
    metadata = parsed.frontmatter["metadata"]
    assert len(metadata["groups"]) == 71
    assert len(metadata["settings"]) == 25
    assert metadata["groups"][70]["name"] == "group_70"
    assert 2 + 2 + 71 * 3 + 25 == 242
    assert parsed.body == "Body"


@pytest.mark.parametrize("frontmatter", ["#" + "x" * 65536, "value: " + "x" * 4097, "values: [" + ", ".join("x" for _ in range(257)) + "]", "\n".join(f"key_{index}: x" for index in range(512)), "value: " + "[" * 9 + "x" + "]" * 9], ids=["bytes", "scalar", "sequence", "nodes", "depth"])
def test_larger_mapping_cap_preserves_other_frontmatter_guards(frontmatter) -> None:
    with pytest.raises(MarkdownParseErrorV1) as rejected:
        parse_markdown_v1(f"---\n{frontmatter}\n---\nBody".encode(), FrontmatterMode.REQUIRED)
    assert rejected.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID


def test_body_requires_nonempty_content_after_exact_closer() -> None:
    with pytest.raises(MarkdownParseErrorV1) as empty:
        parse_markdown_v1(b"---\ntitle: Note\n---", FrontmatterMode.OPTIONAL)
    assert empty.value.code is MarkdownParseCodeV1.BODY_EMPTY


def _structure(*, required_title: bool = True, include_document_text: bool = False) -> StructureDefinition:
    return StructureDefinition.define(
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
            FieldMapping(
                "path",
                FieldSelector(FieldSelectorKind.LOGICAL_PATH),
                FieldValueKind.STRING,
                True,
            ),
            FieldMapping(
                "title",
                FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "title"),
                FieldValueKind.STRING,
                required_title,
            ),
            FieldMapping(
                "published",
                FieldSelector(FieldSelectorKind.FRONTMATTER_FIELD, "published"),
                FieldValueKind.BOOLEAN,
                False,
            ),
            *(
                (FieldMapping(
                    "source_markdown",
                    FieldSelector(FieldSelectorKind.DOCUMENT_TEXT),
                    FieldValueKind.STRING,
                    True,
                ),)
                if include_document_text else ()
            ),
        ),
        text_projections=(
            TextProjection("text", "body"),
            *((TextProjection("complete", "source_markdown"),) if include_document_text else ()),
        ),
    )


def test_maps_only_declared_fields_with_exact_types() -> None:
    parsed = parse_markdown_v1(
        b"---\ntitle: Example\npublished: true\nignored: value\n---\nBody",
        FrontmatterMode.OPTIONAL,
    )
    assert map_markdown_fields_v1(parsed, "notes/example.md", _structure()) == {
        "body": "Body",
        "path": "notes/example.md",
        "published": True,
        "title": "Example",
    }


def test_document_text_maps_full_normalized_markdown_alongside_body_and_frontmatter() -> None:
    content = b"\xef\xbb\xbf---\r\ntitle: Caf\xc3\xa9\r\npublished: true\r\n---\r\nBody \xe2\x9c\xa8\r\n"
    parsed = parse_markdown_v1(content, FrontmatterMode.OPTIONAL)
    mapped = map_markdown_fields_v1(parsed, "notes/example.md", _structure(include_document_text=True))
    assert mapped == {
        "body": "Body ✨\n",
        "path": "notes/example.md",
        "published": True,
        "source_markdown": "---\ntitle: Café\npublished: true\n---\nBody ✨\n",
        "title": "Café",
    }
    assert map_markdown_fields_v1(parsed, "notes/example.md", _structure())["body"] == mapped["body"]

    plain = parse_markdown_v1(b"\xef\xbb\xbfPlain\r\n", FrontmatterMode.OPTIONAL)
    plain_mapped = map_markdown_fields_v1(plain, "notes/plain.md", _structure(required_title=False, include_document_text=True))
    assert plain_mapped["source_markdown"] == plain_mapped["body"] == "Plain\n"


def test_mapping_rejects_missing_required_and_wrong_value_kind() -> None:
    missing = parse_markdown_v1(b"Body", FrontmatterMode.OPTIONAL)
    with pytest.raises(MarkdownParseErrorV1) as required:
        map_markdown_fields_v1(missing, "note.md", _structure())
    assert required.value.code is MarkdownParseCodeV1.FIELD_MAPPING_INVALID

    wrong = parse_markdown_v1(
        b"---\ntitle: Example\npublished: yes\n---\nBody", FrontmatterMode.OPTIONAL
    )
    with pytest.raises(MarkdownParseErrorV1) as type_error:
        map_markdown_fields_v1(wrong, "note.md", _structure())
    assert type_error.value.code is MarkdownParseCodeV1.FIELD_MAPPING_INVALID


def test_optional_missing_field_is_omitted() -> None:
    parsed = parse_markdown_v1(
        b"---\ntitle: Example\n---\nBody", FrontmatterMode.OPTIONAL
    )
    mapped = map_markdown_fields_v1(parsed, "note.md", _structure())
    assert "published" not in mapped


def test_decoded_surrogate_is_closed_without_private_exception_context() -> None:
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        parse_markdown_v1(b'---\nvalue: "\\uD800"\n---\nBody', FrontmatterMode.REQUIRED)
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert invalid.value.__context__ is None


def test_parsed_value_defensively_freezes_exact_builtin_tree() -> None:
    source = {"nested": {"values": ["one"]}}
    parsed = ParsedMarkdownV1("Body", "Body", source, FrontmatterMode.OPTIONAL, True)
    source["nested"]["values"].append("two")  # type: ignore[index,union-attr]
    assert parsed.frontmatter["nested"]["values"] == ("one",)  # type: ignore[index]


def test_parsed_value_rejects_custom_mapping_without_invoking_it() -> None:
    invoked = False

    class HostileDict(dict):
        def items(self):
            nonlocal invoked
            invoked = True
            raise RuntimeError("private")

    with pytest.raises(MarkdownParseErrorV1) as invalid:
        ParsedMarkdownV1(
            "Body",
            "Body",
            {"nested": HostileDict(value="secret")},
            FrontmatterMode.OPTIONAL,
            True,
        )
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert invalid.value.__context__ is None
    assert invoked is False


def test_mapping_proxy_over_hostile_mapping_is_not_accepted() -> None:
    class HostileDict(dict):
        def __iter__(self):  # pragma: no cover - must never be invoked
            raise RuntimeError("private")

    with pytest.raises(TypeError, match="exact built-in dict"):
        ParsedMarkdownV1(
            "Body",
            "Body",
            MappingProxyType(HostileDict(value="secret")),  # type: ignore[arg-type]
            FrontmatterMode.OPTIONAL,
            True,
        )


def test_event_budget_rejects_before_yaml_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_calls = 0

    def forbidden_compose(*_args, **_kwargs):
        nonlocal compose_calls
        compose_calls += 1
        raise AssertionError("compose must not run after event budget failure")

    monkeypatch.setattr(markdown, "MAX_YAML_NODES", 2)
    monkeypatch.setattr(markdown.yaml, "compose", forbidden_compose)
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        parse_markdown_v1(b"---\na: 1\n---\nBody", FrontmatterMode.REQUIRED)
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert compose_calls == 0


def test_surrogate_logical_path_is_closed_without_private_context() -> None:
    parsed = parse_markdown_v1(b"Body", FrontmatterMode.OPTIONAL)
    private_path = "notes/\ud800.md"
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        map_markdown_fields_v1(parsed, private_path, _structure(required_title=False))
    assert invalid.value.code is MarkdownParseCodeV1.FIELD_MAPPING_INVALID
    assert invalid.value.__context__ is None
    assert private_path not in str(invalid.value)


def test_mapping_entry_cap_is_aggregate_across_nested_objects(monkeypatch) -> None:
    copied_sizes = []
    original = markdown._exact_dict_items
    def record_copy(value):
        copied_sizes.append(len(value))
        return original(value)
    monkeypatch.setattr(markdown, "_exact_dict_items", record_copy)
    value = {"left": {f"key_{index}": index for index in range(256)}, "right": {f"key_{index}": index for index in range(255)}}
    with pytest.raises(MarkdownParseErrorV1) as rejected:
        ParsedMarkdownV1("Body", "Body", value, FrontmatterMode.OPTIONAL, True)
    assert rejected.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert copied_sizes == [2, 256]


def test_oversized_513_entry_exact_dict_is_rejected_before_items_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_copy_calls = 0

    def forbidden_items(_value):
        nonlocal item_copy_calls
        item_copy_calls += 1
        raise AssertionError("oversized mappings must fail before tuple copying")

    monkeypatch.setattr(markdown, "_exact_dict_items", forbidden_items)
    oversized = {f"key_{index}": index for index in range(513)}
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        ParsedMarkdownV1(
            "Body", "Body", oversized, FrontmatterMode.OPTIONAL, True
        )
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert item_copy_calls == 0


@pytest.mark.parametrize(
    "frontmatter",
    (
        "a: {b: value}",
        f"{'k' * 65}: value",
        "e\u0301: value",
    ),
)
def test_scalar_depth_and_key_limits_fail_before_compose(
    frontmatter: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose_calls = 0

    def forbidden_compose(*_args, **_kwargs):
        nonlocal compose_calls
        compose_calls += 1
        raise AssertionError("invalid event stream must not reach compose")

    if frontmatter.startswith("a:"):
        monkeypatch.setattr(markdown, "MAX_YAML_DEPTH", 2)
    monkeypatch.setattr(markdown.yaml, "compose", forbidden_compose)
    content = f"---\n{frontmatter}\n---\nBody".encode("utf-8")
    with pytest.raises(MarkdownParseErrorV1) as invalid:
        parse_markdown_v1(content, FrontmatterMode.REQUIRED)
    assert invalid.value.code is MarkdownParseCodeV1.FRONTMATTER_INVALID
    assert compose_calls == 0
