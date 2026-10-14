"""Credential-free installed-runtime inspection for a Modal-built training image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import sysconfig


_INPUT = Path("/opt/synaptic-runtime/build-inputs.json")
_MAX_INPUT_BYTES = 128 * 1024
_MAX_OUTPUT_BYTES = 128 * 1024
_SANDBOX_ID = re.compile(r"sb-[A-Za-z0-9]{1,64}\Z")


INSPECTOR_FAILURE_PHASES = frozenset(
    {
        "IMPORT",
        "INPUT",
        "RUNTIME_IDENTITY",
        "INTERPRETER",
        "PACKAGE_LOCATIONS",
        "INSTALLED_INVENTORY",
        "OUTPUT",
    }
)


class _InspectionFailure(ValueError):
    def __init__(self, phase: str, inventory_stage: str | None = None):
        self.phase, self.inventory_stage = phase, inventory_stage
        super().__init__("MODAL_BUILD_INSPECTION_REJECTED")


def inspect() -> dict[str, object]:
    phase = "IMPORT"
    try:
        from tuner.runtime.packaged_training_worker import (
            inspect_installed_runtime,
            PackagedInstalledRuntimeInspectionError,
            INSTALLED_RUNTIME_INSPECTION_STAGES,
        )

        phase = "INPUT"
        raw = _INPUT.read_bytes()
        if not 0 < len(raw) <= _MAX_INPUT_BYTES:
            raise ValueError("build input size is invalid")
        expected = json.loads(raw)
        if type(expected) is not dict:
            raise ValueError("build input is invalid")
        python = expected["python"]
        phase = "RUNTIME_IDENTITY"
        if (
            _SANDBOX_ID.fullmatch(os.environ.get("MODAL_SANDBOX_ID", "")) is None
            or type(os.environ.get("MODAL_IMAGE_ID")) is not str
            or not os.environ["MODAL_IMAGE_ID"].startswith("im-")
            or sys.implementation.name != python["implementation"]
            or platform.python_version() != python["version"]
            or sys.executable != python["executable"]
        ):
            raise ValueError("runtime identity differs")
        phase = "INTERPRETER"
        physical = Path(sys.executable).resolve(strict=True)
        digest = hashlib.sha256()
        with physical.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != python["executable_digest"]:
            raise ValueError("interpreter differs")
        phase = "PACKAGE_LOCATIONS"
        root = str(Path(sys.executable).parent.parent)
        paths = sysconfig.get_paths(
            scheme="venv", vars={"base": root, "platbase": root}
        )
        if any(paths[key] != python[key] for key in ("purelib", "platlib")):
            raise ValueError("package locations differ")
        phase = "INSTALLED_INVENTORY"
        measured = inspect_installed_runtime(expected)
        return {
            "schema_version": "synaptic-modal-build-capture/v1",
            "image_id": os.environ["MODAL_IMAGE_ID"],
            "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measured": measured,
        }
    except BaseException as error:
        inventory_stage = None
        if (
            phase == "INSTALLED_INVENTORY"
            and type(error) is PackagedInstalledRuntimeInspectionError
        ):
            if (
                type(error.stage) is str
                and error.stage in INSTALLED_RUNTIME_INSPECTION_STAGES
            ):
                inventory_stage = error.stage
        raise _InspectionFailure(phase, inventory_stage) from None


def main() -> int:
    try:
        report = inspect()
        raw = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(raw) > _MAX_OUTPUT_BYTES:
            raise ValueError("runtime capture is too large")
        sys.stdout.buffer.write(raw + b"\n")
        return 0
    except BaseException as error:
        phase = (
            error.phase
            if type(error) is _InspectionFailure
            and type(error.phase) is str
            and error.phase in INSPECTOR_FAILURE_PHASES
            else "OUTPUT"
        )
        stage = None
        if type(error) is _InspectionFailure and phase == "INSTALLED_INVENTORY":
            try:
                from tuner.runtime.packaged_training_worker import (
                    INSTALLED_RUNTIME_INSPECTION_STAGES,
                )

                if (
                    type(error.inventory_stage) is str
                    and error.inventory_stage in INSTALLED_RUNTIME_INSPECTION_STAGES
                ):
                    stage = error.inventory_stage
            except BaseException:
                pass
        failure = {
            "schema_version": "synaptic-modal-build-inspection-failure/v1",
            "phase": phase,
            "inventory_stage": stage,
        }
        try:
            sys.stdout.buffer.write(
                json.dumps(
                    failure, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )
        except BaseException:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
