"""Lean synchronous CLI composition for deterministic dataset preparation."""

from __future__ import annotations

import json
import os
import stat
from argparse import Namespace
from pathlib import Path
from typing import NoReturn

from tuner.dataset_prep import (
    CONFIG_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION_V2,
    DatasetPrepConfigV1,
    DatasetPrepConfigV2,
    DatasetPublicationUncertainV1,
    prepare_dataset_v1,
    prepare_dataset_v2,
)
from tuner.handlers.base import BaseHandler
from tuner.project import ProjectContext


_MAX_CONFIG_BYTES = 1_048_576


class _ConfigError(ValueError):
    pass


def _fail() -> NoReturn:
    raise _ConfigError from None


def _reject_constant(_: str) -> NoReturn:
    _fail()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _assert_plain_components(path: Path) -> tuple[tuple[Path, tuple[int, int, int, int, int]], ...]:
    parts = path.parts
    if not parts:
        _fail()
    current = Path(parts[0])
    observed: list[tuple[Path, tuple[int, int, int, int, int]]] = []
    for index in range(len(parts)):
        if index:
            current = current / parts[index]
        try:
            info = current.lstat()
        except OSError:
            _fail()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            _fail()
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            _fail()
        observed.append((current, _identity(info)))
    return tuple(observed)


def _revalidate_plain_components(
    observed: tuple[tuple[Path, tuple[int, int, int, int, int]], ...]
) -> None:
    for path, identity in observed:
        try:
            info = path.lstat()
        except OSError:
            _fail()
        if (
            _identity(info) != identity
            or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            _fail()


def _read_config(path_value: object, base: Path) -> bytes:
    if type(path_value) is not str or not path_value:
        _fail()
    descriptor: int | None = None
    try:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = base / path
        path = Path(os.path.abspath(os.fspath(path)))
        observed_components = _assert_plain_components(path)
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
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before_open = os.fstat(descriptor)
        if _identity(before_path) != _identity(before_open):
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
        after_path = path.lstat()
        _revalidate_plain_components(observed_components)
        raw = b"".join(chunks)
        if (
            len(raw) > _MAX_CONFIG_BYTES
            or _identity(before_open) != _identity(after_open)
            or _identity(before_open) != _identity(after_path)
            or len(raw) != before_open.st_size
        ):
            _fail()
        return raw
    except (OSError, _ConfigError):
        _fail()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_config(path_value: object, base: Path) -> DatasetPrepConfigV1 | DatasetPrepConfigV2:
    try:
        value = json.loads(
            _read_config(path_value, base).decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict:
            _fail()
        source = value.get("source")
        if type(source) is not dict:
            _fail()
        bundle_value = source.get("bundle_path")
        if type(bundle_value) is not str or not bundle_value:
            _fail()
        bundle_path = Path(bundle_value).expanduser()
        if not bundle_path.is_absolute():
            bundle_path = base / bundle_path
        normalized = dict(value)
        normalized["source"] = {
            **source,
            "bundle_path": os.path.abspath(os.fspath(bundle_path)),
        }
        schema_version = normalized.get("schema_version")
        if schema_version == CONFIG_SCHEMA_VERSION:
            return DatasetPrepConfigV1.from_dict(normalized)
        if schema_version == CONFIG_SCHEMA_VERSION_V2:
            return DatasetPrepConfigV2.from_dict(normalized)
        _fail()
    except (UnicodeError, json.JSONDecodeError, _ConfigError):
        _fail()


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class DatasetPrepareHandler(BaseHandler):
    """Prepare one verified normalized bundle as a private SFT dataset."""

    def __init__(
        self, args: Namespace | None = None, context: ProjectContext | None = None
    ) -> None:
        super().__init__(args=args, context=context)

    @property
    def name(self) -> str:
        return "prepare-dataset"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        try:
            if not self.json_mode:
                _fail()
            config = _load_config(
                getattr(self.args, "ml_config", None), self.context.invocation_cwd
            )
        except Exception:
            _emit({"success": False, "status": "failed", "error_code": "invalid_input"})
            return 2

        try:
            if type(config) is DatasetPrepConfigV1:
                verified = prepare_dataset_v1(config, self.context.tracking_root / "datasets")
            elif type(config) is DatasetPrepConfigV2:
                verified = prepare_dataset_v2(config, self.context.tracking_root / "datasets")
            else:
                raise TypeError("unsupported dataset prep config")
            identity = verified.semantic_identity
            _emit(
                {
                    "success": True,
                    "status": "verified",
                    "dataset_ref": identity.dataset_id,
                    "dataset_digest": identity.dataset_digest,
                    "source_bundle_digest": config.expected_bundle_digest,
                    "row_count": identity.row_count,
                    "split_counts": dict(identity.split_counts),
                    "verified": True,
                }
            )
            return 0
        except DatasetPublicationUncertainV1 as error:
            identity = error.semantic_identity
            _emit(
                {
                    "success": False,
                    "status": "publication_uncertain",
                    "error_code": "dataset_publication_uncertain",
                    "dataset_ref": identity.dataset_id,
                    "dataset_digest": identity.dataset_digest,
                    "row_count": identity.row_count,
                    "split_counts": dict(identity.split_counts),
                    "phase": error.phase.value,
                }
            )
            return 1
        except Exception:
            _emit(
                {
                    "success": False,
                    "status": "failed",
                    "error_code": "dataset_prep_failed",
                }
            )
            return 1


__all__ = ["DatasetPrepareHandler"]
