"""Deterministic project_v1 configuration merging and provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import ManifestValidationError
from .secrets import redact_secrets, reject_literal_secrets


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class ConfigDocument:
    uri: str
    data: Mapping[str, Any]
    precedence: int
    sha256: str
    declaring_file: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        *,
        uri: str,
        data: Mapping[str, Any],
        precedence: int,
        declaring_file: Path | None = None,
    ) -> "ConfigDocument":
        reject_literal_secrets(data)
        safe = redact_secrets(dict(data))
        return cls(
            uri=uri,
            data=dict(data),
            precedence=precedence,
            sha256=sha256_value(safe),
            declaring_file=declaring_file.resolve() if declaring_file else None,
        )


@dataclass(frozen=True)
class ConfigOverride:
    path: str
    value: Any
    source: str = "cli"

    def __post_init__(self) -> None:
        if not self.path or any(not part for part in self.path.split(".")):
            raise ValueError("Override paths must be non-empty dotted paths")


@dataclass(frozen=True)
class ResolvedConfig:
    config: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...]
    overrides: tuple[Mapping[str, Any], ...]
    source_map: Mapping[str, Mapping[str, Any]]
    schema_version: str = "synaptic-resolved-config/v1"
    resolved_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        reject_literal_secrets(self.config)
        safe_config = redact_secrets(dict(self.config))
        object.__setattr__(self, "resolved_sha256", sha256_value(safe_config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": redact_secrets(dict(self.config)),
            "sources": [dict(source) for source in self.sources],
            "overrides": [redact_secrets(dict(item)) for item in self.overrides],
            "source_map": {key: dict(value) for key, value in sorted(self.source_map.items())},
            "resolved_sha256": self.resolved_sha256,
        }


def load_config_document(path: Path, *, uri: str, precedence: int) -> ConfigDocument:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestValidationError(f"Could not load config document {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestValidationError(f"Config document must be a mapping: {path}")
    return ConfigDocument.from_mapping(
        uri=uri, data=data, precedence=precedence, declaring_file=path
    )


def _leaf_paths(value: object, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        if not value and prefix:
            yield prefix
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_paths(value[key], path)
    else:
        yield prefix


def _deep_merge(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        elif isinstance(value, dict):
            nested: dict[str, Any] = {}
            _deep_merge(nested, value)
            target[key] = nested
        else:
            target[key] = value


def _replacement_paths(
    target: Mapping[str, Any], incoming: Mapping[str, Any], prefix: str = ""
) -> Iterable[str]:
    """Yield branches replaced rather than recursively merged."""

    for key, value in incoming.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        current = target.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            yield from _replacement_paths(current, value, path)
        else:
            yield path


def _prune_source_map(
    source_map: dict[str, Mapping[str, Any]], path: str, *, ancestors: bool = False
) -> None:
    stale = [
        key
        for key in source_map
        if key == path
        or key.startswith(path + ".")
        or (ancestors and path.startswith(key + "."))
    ]
    for key in stale:
        source_map.pop(key, None)


def _set_override(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None or not isinstance(existing, dict):
            existing = {}
            current[part] = existing
        current = existing
    current[parts[-1]] = value


def resolve_config_layers(
    documents: Sequence[ConfigDocument],
    *,
    overrides: Sequence[ConfigOverride] = (),
) -> ResolvedConfig:
    """Merge documents by precedence, retaining declaration order for ties."""

    ordered = sorted(enumerate(documents), key=lambda pair: (pair[1].precedence, pair[0]))
    config: dict[str, Any] = {}
    source_map: dict[str, Mapping[str, Any]] = {}
    source_records: list[Mapping[str, Any]] = []
    for _, document in ordered:
        for replaced_path in _replacement_paths(config, document.data):
            _prune_source_map(source_map, replaced_path)
        _deep_merge(config, document.data)
        record = {
            "uri": document.uri,
            "sha256": document.sha256,
            "precedence": document.precedence,
        }
        source_records.append(record)
        for path in _leaf_paths(document.data):
            if path:
                source_map[path] = record

    override_records: list[Mapping[str, Any]] = []
    for override in overrides:
        _prune_source_map(source_map, override.path, ancestors=True)
        _set_override(config, override.path, override.value)
        record = {"path": override.path, "source": override.source, "value": override.value}
        override_records.append(record)
        source_map[override.path] = {"uri": override.source, "precedence": 100}

    return ResolvedConfig(
        config=config,
        sources=tuple(source_records),
        overrides=tuple(override_records),
        source_map=source_map,
    )
