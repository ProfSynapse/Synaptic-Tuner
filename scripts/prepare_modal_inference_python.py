"""Prepare the isolated Python interpreter for a Modal inference image layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import venv

_SCHEMA = "synaptic-modal-inference-python-preparation/v1"
_DESTINATION = Path("/opt/synaptic-inference")
_BASE_SITE = Path("/usr/local/lib/python3.12/dist-packages")
_PTH_NAME = "synaptic-inference-base.pth"
_PTH_BYTES = b"/usr/local/lib/python3.12/dist-packages\n"
_MAX_RESULT_BYTES = 64 * 1024
_DIST_SEPARATORS = re.compile(r"[-_.]+")


class _PreparationFailure(RuntimeError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


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


def _base_site(path: Path) -> None:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise _PreparationFailure("BASE_SITE_INVALID") from None
    if (
        not stat.S_ISDIR(info.st_mode)
        or path != resolved
        or not path.is_absolute()
        or path != Path(os.path.normpath(path))
    ):
        raise _PreparationFailure("BASE_SITE_INVALID")
    matches = []
    try:
        for child in path.iterdir():
            if child.name.lower().endswith(".dist-info"):
                stem = child.name[: -len(".dist-info")].rsplit("-", 1)[0]
                if _DIST_SEPARATORS.sub("-", stem).lower() == "vllm":
                    matches.append(child)
    except OSError:
        raise _PreparationFailure("BASE_SITE_INVALID") from None
    try:
        metadata = matches[0].lstat() if len(matches) == 1 else None
    except OSError:
        raise _PreparationFailure("BASE_ML_METADATA_INVALID") from None
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise _PreparationFailure("BASE_ML_METADATA_INVALID")


def _preflight_code(destination: Path, base_site: Path) -> str:
    return (
        "import json,os,sys;"
        f"d={str(destination)!r};b={str(base_site)!r};"
        "print(json.dumps({'executable':os.path.realpath(sys.executable),"
        "'prefix':os.path.realpath(sys.prefix),'base_prefix':os.path.realpath(sys.base_prefix),"
        "'base_count':sys.path.count(b),'os_site':('/usr/lib/python3/dist-packages' in sys.path)}))"
    )


def _prepare(
    *,
    destination: Path,
    base_site: Path,
    pth_bytes: bytes,
    builder_factory=venv.EnvBuilder,
    runner=subprocess.run,
) -> None:
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
        raise _PreparationFailure("BASE_PYTHON_INVALID")
    _base_site(base_site)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise _PreparationFailure("DESTINATION_EXISTS") from None
    except OSError:
        raise _PreparationFailure("DESTINATION_CREATE_FAILED") from None
    try:
        builder_factory(
            system_site_packages=False,
            clear=False,
            symlinks=False,
            with_pip=False,
        ).create(destination)
        python = destination / "bin" / "python"
        site = destination / "lib" / "python3.12" / "site-packages"
        pth = site / _PTH_NAME
        descriptor = os.open(
            pth,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o444,
        )
        try:
            if os.write(descriptor, pth_bytes) != len(pth_bytes):
                raise _PreparationFailure("PTH_WRITE_FAILED")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        completed = runner(
            [str(python), "-I", "-c", _preflight_code(destination, base_site)],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin"},
            timeout=30,
        )
        if completed.returncode != 0 or len(completed.stdout) > _MAX_RESULT_BYTES:
            raise _PreparationFailure("PREFLIGHT_FAILED")
        result = json.loads(completed.stdout)
        if (
            type(result) is not dict
            or set(result)
            != {
                "base_count",
                "base_prefix",
                "executable",
                "os_site",
                "prefix",
            }
            or result["executable"] != str(python.resolve(strict=True))
            or result["prefix"] != str(destination.resolve(strict=True))
            or result["base_prefix"] == result["prefix"]
            or result["base_count"] != 1
            or result["os_site"] is not False
        ):
            raise _PreparationFailure("PREFLIGHT_FAILED")
    except _PreparationFailure:
        raise
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        raise _PreparationFailure("PREPARATION_FAILED") from None


def main() -> int:
    try:
        _prepare(
            destination=_DESTINATION,
            base_site=_BASE_SITE,
            pth_bytes=_PTH_BYTES,
        )
        sys.stdout.buffer.write(
            _line({"schema_version": _SCHEMA, "status": "CPU_CANDIDATE_PREPARED"})
        )
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except _PreparationFailure as error:
        sys.stderr.buffer.write(
            _line(
                {
                    "reason_code": error.reason_code,
                    "schema_version": _SCHEMA,
                    "status": "FAILED",
                }
            )
        )
        return 125
    except Exception:
        sys.stderr.buffer.write(
            _line(
                {
                    "reason_code": "PREPARATION_FAILED",
                    "schema_version": _SCHEMA,
                    "status": "FAILED",
                }
            )
        )
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
