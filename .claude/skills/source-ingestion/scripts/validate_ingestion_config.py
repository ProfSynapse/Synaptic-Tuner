#!/usr/bin/env python3
"""Validate a strict synaptic-ingestion-cli/v1 or v2 JSON configuration.

The validator checks the executable CLI shape and constructs the declared
structure with public ``synaptic_tuner.api.v1`` contract types. It does not read
source files, authorize selections, or import private ingestion runtime modules.

Usage:
  python validate_ingestion_config.py CONFIG

Exit codes:
  0  valid
  1  violations found
  2  usage or I/O error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn


MAX_CONFIG_BYTES = 1_048_576
HARD_MAX_MEMBER_BYTES = 16 * 1024 * 1024
HARD_MAX_TOTAL_BYTES = 32 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ConfigViolation(ValueError):
    """A sanitized configuration-shape violation."""


def _fail(message: str) -> NoReturn:
    raise ConfigViolation(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(_: str) -> NoReturn:
    _fail("non-finite JSON numbers are forbidden")


def _object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        _fail(f"{label} fields differ; missing={missing}, extra={extra}")
    return value


def _array(value: object, minimum: int, maximum: int, label: str) -> list[object]:
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _fail(f"{label} must contain {minimum}..{maximum} items")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _load(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigViolation("config could not be read") from exc
    if not raw or len(raw) > MAX_CONFIG_BYTES:
        _fail(f"config must contain 1..{MAX_CONFIG_BYTES} bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigViolation("config must be strict UTF-8 JSON") from exc
    if type(value) is not dict:
        _fail("config must be an object")
    common = {
        "schema_version",
        "project_ref",
        "admission_request_id",
        "request_id",
        "discovery",
        "structure",
        "binding",
    }
    if value.get("schema_version") == "synaptic-ingestion-cli/v1":
        return _object(value, common, "config")
    if value.get("schema_version") == "synaptic-ingestion-cli/v2":
        return _object(value, common | {"admission_limits"}, "config")
    _fail(
        "schema_version must be synaptic-ingestion-cli/v1 or synaptic-ingestion-cli/v2"
    )


def validate(config: dict[str, object]) -> None:
    if config["schema_version"] == "synaptic-ingestion-cli/v2":
        limits = _object(
            config["admission_limits"],
            {"max_member_bytes", "max_total_bytes"},
            "admission_limits",
        )
        member = limits["max_member_bytes"]
        total = limits["max_total_bytes"]
        if type(member) is not int or not 1 <= member <= HARD_MAX_MEMBER_BYTES:
            _fail("admission_limits.max_member_bytes is outside the supported range")
        if type(total) is not int or not 1 <= total <= HARD_MAX_TOTAL_BYTES:
            _fail("admission_limits.max_total_bytes is outside the supported range")
        if member > total:
            _fail("admission member budget must not exceed aggregate budget")

    discovery = _object(
        config["discovery"], {"include", "exclude", "include_hidden"}, "discovery"
    )
    include = _array(discovery["include"], 1, 128, "discovery.include")
    exclude = _array(discovery["exclude"], 0, 128, "discovery.exclude")
    if any(type(item) is not str or not item for item in include + exclude):
        _fail("discovery patterns must be non-empty strings")
    if len(include) != len(set(include)) or len(exclude) != len(set(exclude)):
        _fail("discovery patterns must be unique within each list")
    if type(discovery["include_hidden"]) is not bool:
        _fail("discovery.include_hidden must be boolean")

    structure = _object(
        config["structure"],
        {
            "name",
            "version",
            "frontmatter_mode",
            "fields",
            "text_projections",
            "metadata",
        },
        "structure",
    )
    binding = _object(config["binding"], {"binding_id", "pattern"}, "binding")

    try:
        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        from synaptic_tuner.api.v1 import (
            FieldMapping,
            FrontmatterMode,
            MarkdownProfileV1,
            MetadataDeclaration,
            ParsingProfile,
            SourceMatcher,
            StructureBinding,
            StructureDefinition,
            StructureSet,
            TextProjection,
            validate_ingestion_identity,
        )

        for key in ("project_ref", "admission_request_id", "request_id"):
            validate_ingestion_identity(config[key], key)
        for pattern in include + exclude:
            SourceMatcher(pattern)

        fields = tuple(
            FieldMapping.from_dict(
                _object(
                    item,
                    {"name", "selector", "value_kind", "required"},
                    f"structure.fields[{index}]",
                )
            )
            for index, item in enumerate(
                _array(structure["fields"], 1, 64, "structure.fields")
            )
        )
        projections = tuple(
            TextProjection.from_dict(
                _object(
                    item,
                    {"name", "field_ref"},
                    f"structure.text_projections[{index}]",
                )
            )
            for index, item in enumerate(
                _array(
                    structure["text_projections"],
                    1,
                    16,
                    "structure.text_projections",
                )
            )
        )
        metadata = tuple(
            MetadataDeclaration.from_dict(
                _object(
                    item,
                    {"name", "field_ref"},
                    f"structure.metadata[{index}]",
                )
            )
            for index, item in enumerate(
                _array(structure["metadata"], 0, 64, "structure.metadata")
            )
        )
        definition = StructureDefinition.define(
            name=_text(structure["name"], "structure.name"),
            version=_text(structure["version"], "structure.version"),
            markdown=MarkdownProfileV1(FrontmatterMode(structure["frontmatter_mode"])),
            fields=fields,
            text_projections=projections,
            metadata=metadata,
            parsing_profile=(
                ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1
                if config["schema_version"] == "synaptic-ingestion-cli/v1"
                else ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V2
            ),
        )
        StructureSet(
            (definition,),
            (
                StructureBinding(
                    _text(binding["binding_id"], "binding.binding_id"),
                    SourceMatcher(_text(binding["pattern"], "binding.pattern")),
                    definition.ref,
                ),
            ),
        )
    except ConfigViolation:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise ConfigViolation(
            f"public ingestion contract rejected the declaration ({type(exc).__name__})"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="strict CLI ingestion JSON file")
    args = parser.parse_args()
    if not args.config.is_file():
        print("error: config must be an existing regular file", file=sys.stderr)
        return 2
    try:
        validate(_load(args.config))
    except ConfigViolation as exc:
        print(f"INVALID: {exc}")
        return 1
    print("VALID: ingestion CLI config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
