"""Inspect one inference image runtime without importing its ML packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys

_SCHEMA = "synaptic-modal-inference-runtime-inspection-candidate/v1"
_ERROR_SCHEMA = "synaptic-modal-inference-runtime-inspection-error/v1"
_IMAGE = re.compile(r"docker\.io/vllm/vllm-openai@sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"[1-9][0-9]{0,2}\.(?:0|[1-9][0-9]{0,2})\.(?:0|[1-9][0-9]{0,2})")
_DIST_SEPARATORS = re.compile(r"[-_.]+")
_MAX_DISTRIBUTIONS = 512
_MAX_DISTRIBUTION_OCCURRENCES = 4096
_MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024


class _InspectionFailure(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class _ClosedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _InspectionFailure("ARGUMENT_INVALID")


def _line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _failure(reason_code: str) -> int:
    payload = _line(
        {
            "reason_code": reason_code,
            "schema_version": _ERROR_SCHEMA,
            "status": "FAILED",
        }
    )
    sys.stderr.buffer.write(payload)
    return 125


def _bounded_text(value: object, *, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise _InspectionFailure("METADATA_INVALID")
    return value


def _executable() -> tuple[str, str]:
    selected = Path(_bounded_text(sys.executable, maximum=4096))
    if not selected.is_absolute() or selected != Path(os.path.normpath(selected)):
        raise _InspectionFailure("PYTHON_INVALID")
    try:
        selected_before = selected.lstat()
        path = selected.resolve(strict=True)
    except OSError:
        raise _InspectionFailure("PYTHON_INVALID") from None
    value = _bounded_text(path.as_posix(), maximum=4096)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(path))
        or path.is_symlink()
    ):
        raise _InspectionFailure("PYTHON_INVALID")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_EXECUTABLE_BYTES
        ):
            raise _InspectionFailure("PYTHON_INVALID")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_EXECUTABLE_BYTES + 1 - size),
            )
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_EXECUTABLE_BYTES:
                raise _InspectionFailure("PYTHON_INVALID")
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
        selected_after = selected.lstat()
        selected_identity = lambda info: (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            size != before.st_size
            or identity(after) != identity(before)
            or selected_identity(selected_after) != selected_identity(selected_before)
            or not os.path.samefile(selected, path)
        ):
            raise _InspectionFailure("PYTHON_INVALID")
        return value, digest.hexdigest()
    except _InspectionFailure:
        raise
    except (OSError, OverflowError, ValueError):
        raise _InspectionFailure("PYTHON_INVALID") from None
    finally:
        if descriptor >= 0:
            active_failure = sys.exc_info()[0] is not None
            try:
                os.close(descriptor)
            except (KeyboardInterrupt, SystemExit):
                if not active_failure:
                    raise
            except BaseException:
                if not active_failure:
                    raise _InspectionFailure("PYTHON_INVALID") from None


def _metadata_identity(distribution: object) -> tuple[int, ...] | None:
    if type(distribution) is not importlib.metadata.PathDistribution:
        return None
    path = getattr(distribution, "_path", None)
    if type(path) is not type(Path()):
        return None
    try:
        info = path.stat()
    except (OSError, OverflowError, ValueError):
        return None
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        return None
    return (
        stat.S_IFMT(info.st_mode),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _distributions() -> dict[str, str]:
    result: dict[str, str] = {}
    identities: dict[str, tuple[tuple[int, ...], str]] = {}
    physical: dict[tuple[int, ...], tuple[str, str]] = {}
    occurrences = 0
    try:
        iterator = iter(importlib.metadata.distributions())
    except Exception:
        raise _InspectionFailure("DISTRIBUTION_ENUMERATION_FAILED") from None
    while True:
        try:
            distribution = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise _InspectionFailure("DISTRIBUTION_ENUMERATION_FAILED") from None
        occurrences += 1
        if occurrences > _MAX_DISTRIBUTION_OCCURRENCES:
            raise _InspectionFailure("DISTRIBUTION_COUNT_LIMIT")
        identity_before = _metadata_identity(distribution)
        try:
            metadata = distribution.metadata
            raw_name = metadata.get("Name")
            raw_version = distribution.version
        except Exception:
            raise _InspectionFailure("DISTRIBUTION_METADATA_READ_FAILED") from None
        try:
            name = _bounded_text(raw_name, maximum=128)
        except Exception:
            raise _InspectionFailure("DISTRIBUTION_NAME_INVALID") from None
        try:
            version = _bounded_text(raw_version, maximum=256)
        except Exception:
            raise _InspectionFailure("DISTRIBUTION_VERSION_INVALID") from None
        normalized = _DIST_SEPARATORS.sub("-", name).lower()
        if not normalized:
            raise _InspectionFailure("DISTRIBUTION_NAME_INVALID")
        identity_after = _metadata_identity(distribution)
        if identity_before != identity_after and (
            identity_before is not None or identity_after is not None
        ):
            raise _InspectionFailure("DISTRIBUTION_METADATA_READ_FAILED")
        stable_identity = (
            identity_before
            if identity_before is not None and identity_before == identity_after
            else None
        )
        if normalized in result:
            previous = identities.get(normalized)
            if previous != (stable_identity, version) or stable_identity is None:
                raise _InspectionFailure("DISTRIBUTION_IDENTITY_DUPLICATE")
            continue
        if len(result) >= _MAX_DISTRIBUTIONS:
            raise _InspectionFailure("DISTRIBUTION_COUNT_LIMIT")
        if stable_identity is not None:
            previous_physical = physical.get(stable_identity)
            if previous_physical is not None and previous_physical != (
                normalized,
                version,
            ):
                raise _InspectionFailure("DISTRIBUTION_IDENTITY_DUPLICATE")
            physical[stable_identity] = (normalized, version)
            identities[normalized] = (stable_identity, version)
        result[normalized] = version
    return dict(sorted(result.items()))


def inspect_runtime(*, image: str, source_commit: str) -> dict[str, object]:
    if type(image) is not str or _IMAGE.fullmatch(image) is None:
        raise _InspectionFailure("IMAGE_INVALID")
    if type(source_commit) is not str or _COMMIT.fullmatch(source_commit) is None:
        raise _InspectionFailure("SOURCE_INVALID")
    implementation = _bounded_text(platform.python_implementation(), maximum=32)
    version = _bounded_text(platform.python_version(), maximum=64)
    if implementation != "CPython" or _VERSION.fullmatch(version) is None:
        raise _InspectionFailure("PYTHON_INVALID")
    executable, executable_digest = _executable()
    distributions = _distributions()
    candidate = {
        "distributions": distributions,
        "operator_selection": {"image": image, "source_commit": source_commit},
        "python": {
            "executable": executable,
            "executable_sha256": executable_digest,
            "implementation": "cpython",
            "version": version,
        },
        "requirements": {
            "modal": {
                "present": "modal" in distributions,
                "version": distributions.get("modal"),
            },
            "vllm": {
                "present": "vllm" in distributions,
                "version": distributions.get("vllm"),
            },
        },
        "schema_version": _SCHEMA,
        "status": "CANDIDATE_ONLY",
    }
    if len(_line(candidate)) > _MAX_OUTPUT_BYTES:
        raise _InspectionFailure("OUTPUT_INVALID")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = _ClosedParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        candidate = inspect_runtime(image=args.image, source_commit=args.source_commit)
        sys.stdout.buffer.write(_line(candidate))
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except _InspectionFailure as error:
        return _failure(error.reason_code)
    except Exception:
        return _failure("INSPECTION_FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
