"""Credential-free installed-runtime inspection for a Modal-built training image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig


_INPUT = Path("/opt/synaptic-runtime/build-inputs.json")
_MAX_INPUT_BYTES = 128 * 1024
_MAX_OUTPUT_BYTES = 128 * 1024


def inspect() -> dict[str, object]:
    from tuner.runtime.packaged_training_worker import inspect_installed_runtime

    raw = _INPUT.read_bytes()
    if not 0 < len(raw) <= _MAX_INPUT_BYTES:
        raise ValueError("build input size is invalid")
    expected = json.loads(raw)
    if type(expected) is not dict:
        raise ValueError("build input is invalid")
    python = expected["python"]
    if (
        os.environ.get("MODAL_IS_REMOTE") != "1"
        or type(os.environ.get("MODAL_IMAGE_ID")) is not str
        or not os.environ["MODAL_IMAGE_ID"].startswith("im-")
        or sys.implementation.name != python["implementation"]
        or platform.python_version() != python["version"]
        or sys.executable != python["executable"]
    ):
        raise ValueError("runtime identity differs")
    physical = Path(sys.executable).resolve(strict=True)
    digest = hashlib.sha256()
    with physical.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != python["executable_digest"]:
        raise ValueError("interpreter differs")
    root = str(Path(sys.executable).parent.parent)
    paths = sysconfig.get_paths(scheme="venv", vars={"base": root, "platbase": root})
    if any(paths[key] != python[key] for key in ("purelib", "platlib")):
        raise ValueError("package locations differ")
    measured = inspect_installed_runtime(expected)
    return {
        "schema_version": "synaptic-modal-build-capture/v1",
        "image_id": os.environ["MODAL_IMAGE_ID"],
        "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measured": measured,
    }


def main() -> int:
    try:
        report = inspect()
        raw = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(raw) > _MAX_OUTPUT_BYTES:
            raise ValueError("runtime capture is too large")
        sys.stdout.buffer.write(raw + b"\n")
        return 0
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
