"""Lean synchronous CLI composition for local Markdown ingestion."""

from __future__ import annotations

import json
import os
import stat
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from synaptic_tuner.api.v1.ingestion_facade import (
    FieldMapping,
    FrontmatterMode,
    IngestionAPI,
    IngestionOperationError,
    IngestionRequest,
    IngestionRunState,
    MarkdownProfileV1,
    MetadataDeclaration,
    ParsingProfile,
    SourceAdmissionRequest,
    SourceMatcher,
    StructureBinding,
    StructureDefinition,
    StructureSet,
    TextProjection,
    validate_ingestion_identity,
)
from tuner.handlers.base import BaseHandler
from tuner.ingestion.local_selection_v1 import (
    HARD_MAX_MEMBER_BYTES,
    HARD_MAX_TOTAL_BYTES,
    MAX_SELECTION_ROOTS,
    AdmissionLimitsV1,
    LocalDiscoveryPolicyV1,
    LocalSelectionErrorV1,
    LocalSelectionRootV1,
)
from tuner.ingestion.runtime_v1 import ProcessLocalIngestionOperationsV1
from tuner.project import ProjectContext


_CONFIG_SCHEMA_V1 = "synaptic-ingestion-cli/v1"
_CONFIG_SCHEMA_V2 = "synaptic-ingestion-cli/v2"
_MAX_CONFIG_BYTES = 1_048_576
_OUTPUT_REF = "primary"


class _ConfigError(ValueError):
    pass


class _SystemClock:
    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail() -> NoReturn:
    raise _ConfigError


def _object(value: object, fields: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail()
    return value


def _array(value: object, *, minimum: int, maximum: int) -> list[object]:
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _fail()
    return value


def _reject_constant(_: str) -> NoReturn:
    _fail()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_config(path_value: object, base: Path) -> str:
    if type(path_value) is not str or not path_value:
        _fail()
    try:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = base / path
        path = Path(os.path.abspath(path))
        before_path = path.lstat()
        if (
            not stat.S_ISREG(before_path.st_mode)
            or stat.S_ISLNK(before_path.st_mode)
            or getattr(before_path, "st_file_attributes", 0) & 0x400
            or before_path.st_size > _MAX_CONFIG_BYTES
        ):
            _fail()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before_open = os.fstat(descriptor)
            if _file_identity(before_path) != _file_identity(before_open):
                _fail()
            chunks: list[bytes] = []
            remaining = _MAX_CONFIG_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        raw = b"".join(chunks)
        identity = _file_identity(before_open)
        if (
            len(raw) > _MAX_CONFIG_BYTES
            or identity != _file_identity(after_open)
            or identity != _file_identity(after_path)
            or len(raw) != before_open.st_size
        ):
            _fail()
        return raw.decode("utf-8")
    except (OSError, UnicodeError, _ConfigError):
        _fail()


def _load_config(path_value: object, base: Path) -> dict[str, object]:
    raw = _read_config(path_value, base)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, _ConfigError):
        _fail()
    if type(value) is not dict:
        _fail()
    common = {
        "schema_version",
        "project_ref",
        "admission_request_id",
        "request_id",
        "discovery",
        "structure",
        "binding",
    }
    if value.get("schema_version") == _CONFIG_SCHEMA_V1:
        return _object(value, frozenset(common))
    if value.get("schema_version") == _CONFIG_SCHEMA_V2:
        return _object(value, frozenset(common | {"admission_limits"}))
    _fail()


def _structures(config: dict[str, object]) -> StructureSet:
    structure = _object(
        config["structure"],
        frozenset(
            {
                "name",
                "version",
                "frontmatter_mode",
                "fields",
                "text_projections",
                "metadata",
            }
        ),
    )
    fields = tuple(
        FieldMapping.from_dict(
            _object(
                item,
                frozenset({"name", "selector", "value_kind", "required"}),
            )
        )
        for item in _array(structure["fields"], minimum=1, maximum=64)
    )
    projections = tuple(
        TextProjection.from_dict(
            _object(item, frozenset({"name", "field_ref"}))
        )
        for item in _array(
            structure["text_projections"], minimum=1, maximum=16
        )
    )
    metadata = tuple(
        MetadataDeclaration.from_dict(
            _object(item, frozenset({"name", "field_ref"}))
        )
        for item in _array(structure["metadata"], minimum=0, maximum=64)
    )
    definition = StructureDefinition.define(
        name=structure["name"],  # type: ignore[arg-type]
        version=structure["version"],  # type: ignore[arg-type]
        markdown=MarkdownProfileV1(
            FrontmatterMode(structure["frontmatter_mode"])  # type: ignore[arg-type]
        ),
        fields=fields,
        text_projections=projections,
        metadata=metadata,
        parsing_profile=(
            ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V1
            if config["schema_version"] == _CONFIG_SCHEMA_V1
            else ParsingProfile.MARKDOWN_YAML_FRONTMATTER_V2
        ),
    )
    binding = _object(
        config["binding"], frozenset({"binding_id", "pattern"})
    )
    return StructureSet(
        (definition,),
        (
            StructureBinding(
                binding["binding_id"],  # type: ignore[arg-type]
                SourceMatcher(binding["pattern"]),  # type: ignore[arg-type]
                definition.ref,
            ),
        ),
    )


def _policy(config: dict[str, object]) -> LocalDiscoveryPolicyV1:
    discovery = _object(
        config["discovery"],
        frozenset({"include", "exclude", "include_hidden"}),
    )
    include = tuple(_array(discovery["include"], minimum=1, maximum=128))
    exclude = tuple(_array(discovery["exclude"], minimum=0, maximum=128))
    if any(type(item) is not str for item in (*include, *exclude)):
        _fail()
    return LocalDiscoveryPolicyV1(
        include,  # type: ignore[arg-type]
        exclude,  # type: ignore[arg-type]
        discovery["include_hidden"],  # type: ignore[arg-type]
    )


def _admission_limits(config: dict[str, object]) -> AdmissionLimitsV1 | None:
    if config["schema_version"] == _CONFIG_SCHEMA_V1:
        return None
    limits = _object(
        config["admission_limits"],
        frozenset({"max_member_bytes", "max_total_bytes"}),
    )
    member = limits["max_member_bytes"]
    total = limits["max_total_bytes"]
    if (
        type(member) is not int
        or type(total) is not int
        or not 1 <= member <= HARD_MAX_MEMBER_BYTES
        or not 1 <= total <= HARD_MAX_TOTAL_BYTES
        or member > total
    ):
        _fail()
    return AdmissionLimitsV1(member, total)


def _selection_roots(values: object, base: Path) -> tuple[LocalSelectionRootV1, ...]:
    if type(values) is not list or not 1 <= len(values) <= MAX_SELECTION_ROOTS:
        _fail()
    roots: list[LocalSelectionRootV1] = []
    aliases: set[str] = set()
    for value in values:
        if type(value) is not str or "=" not in value:
            _fail()
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path or alias in aliases:
            _fail()
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base / path
        path = Path(os.path.abspath(os.fspath(path)))
        roots.append(LocalSelectionRootV1(alias, path))
        aliases.add(alias)
    return tuple(roots)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class IngestionHandler(BaseHandler):
    """Run one bounded local Markdown ingestion to completion."""

    def __init__(
        self, args: Namespace | None = None, context: ProjectContext | None = None
    ) -> None:
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "ingest"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            if not self.json_mode:
                _fail()
            config = _load_config(
                getattr(self.args, "ml_config", None), self.context.invocation_cwd
            )
            for key in ("project_ref", "admission_request_id", "request_id"):
                validate_ingestion_identity(config[key], key)
            structures = _structures(config)
            policy = _policy(config)
            admission_limits = _admission_limits(config)
            roots = _selection_roots(
                getattr(self.args, "ingestion_selections", None),
                self.context.invocation_cwd,
            )
        except Exception:
            _emit({"success": False, "status": "failed", "error_code": "invalid_input"})
            return 2

        try:
            clock = _SystemClock()
            output_root = self.context.tracking_root / "ingestion" / "bundles"
            operations = ProcessLocalIngestionOperationsV1(
                outputs={_OUTPUT_REF: output_root}, clock=clock
            )
            api = IngestionAPI(operations, clock=clock)
            project_ref = config["project_ref"]
            authorized = operations.authorize_local_selection(
                project_ref,
                roots,
                policy,
                admission_limits,  # type: ignore[arg-type]
            )
            snapshot = api.admit(
                SourceAdmissionRequest(
                    config["admission_request_id"],  # type: ignore[arg-type]
                    project_ref,  # type: ignore[arg-type]
                    authorized,
                )
            )
            request = IngestionRequest(
                config["request_id"],  # type: ignore[arg-type]
                project_ref,  # type: ignore[arg-type]
                snapshot,
                structures,
                _OUTPUT_REF,
            )
            plan = api.plan(request)
            preflight = api.preflight(plan)
            if not preflight.ready:
                _emit(
                    {
                        "success": False,
                        "status": "blocked",
                        "error_code": "preflight_blocked",
                        "source_count": plan.preview.source_count,
                        "matched_sources": plan.preview.matched_sources,
                        "unmatched_sources": plan.preview.unmatched_sources,
                        "ambiguous_sources": plan.preview.ambiguous_sources,
                        "plan_fingerprint": plan.plan_fingerprint,
                        "diagnostic_codes": [
                            item.value for item in preflight.diagnostic_codes
                        ],
                    }
                )
                return 1
            started = api.start(plan, preflight)
            outcome = api.show(started.run)
            result = api.result(outcome)
            if outcome.state is not IngestionRunState.SUCCEEDED:
                _emit(
                    {
                        "success": False,
                        "status": outcome.state.value,
                        "error_code": "ingestion_failed",
                        "source_count": outcome.source_count,
                        "sources_processed": outcome.sources_processed,
                        "documents_written": outcome.documents_written,
                        "run_ref": outcome.run.run_id,
                        "outcome_digest": result.outcome_digest,
                        "diagnostic_codes": []
                        if outcome.diagnostic_code is None
                        else [outcome.diagnostic_code.value],
                    }
                )
                return 1
            verification = api.verify(result)
            bundle = outcome.bundle
            if bundle is None:
                raise ValueError
            _emit(
                {
                    "success": verification.verified,
                    "status": "verified" if verification.verified else "verification_failed",
                    "source_count": outcome.source_count,
                    "matched_sources": plan.preview.matched_sources,
                    "unmatched_sources": plan.preview.unmatched_sources,
                    "ambiguous_sources": plan.preview.ambiguous_sources,
                    "sources_processed": outcome.sources_processed,
                    "documents_written": outcome.documents_written,
                    "structure_set_digest": structures.digest,
                    "snapshot_ref": snapshot.snapshot_id,
                    "manifest_digest": snapshot.manifest_digest,
                    "plan_fingerprint": plan.plan_fingerprint,
                    "run_ref": outcome.run.run_id,
                    "outcome_digest": result.outcome_digest,
                    "bundle_ref": bundle.bundle_id,
                    "bundle_digest": bundle.bundle_digest,
                    "diagnostic_codes": [
                        item.value for item in verification.diagnostic_codes
                    ],
                }
            )
            return 0 if verification.verified else 1
        except LocalSelectionErrorV1 as error:
            _emit({"success": False, "status": "failed", "error_code": error.code.value})
            return 1
        except IngestionOperationError as error:
            _emit({"success": False, "status": "failed", "error_code": error.code.value})
            return 1
        except Exception:
            _emit({"success": False, "status": "failed", "error_code": "ingestion_failed"})
            return 1


__all__ = ["IngestionHandler"]
