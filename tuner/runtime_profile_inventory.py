"""Capture a runtime-profile distribution inventory from a local container image.

The capture runs one small, credential-free probe inside the image with the
network disabled and Docker's implicit pull turned off. The probe reports every
installed distribution's metadata name and version plus a fixed set of platform
facts. The host side normalizes, sorts and validates the result with the same
parser ``tuner.runtime_profiles`` uses, and writes canonical bytes, so the file
is accepted by ``load_runtime_profile`` exactly as written.

The probe never imports ``torch``: the CUDA build is read from
``torch/version.py`` without importing the package.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from packaging.utils import canonicalize_name

from tuner.runtime_profiles import (
    INVENTORY_SCHEMA,
    INVENTORY_VERSION_FACTS,
    RuntimeProfileError,
    parse_runtime_inventory,
)


PROBE_SCHEMA = "synaptic-runtime-inventory-probe/v1"
PROBE_TIMEOUT_SECONDS = 300
PROBE_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
# Platform facts recorded next to the version facts in every inventory.
PLATFORM_FACTS = (
    "architecture",
    "cuda_build",
    "libc",
    "os",
    "python_implementation",
    "python_version",
)
_IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_ABSOLUTE_EXECUTABLE = re.compile(r"^/[A-Za-z0-9_./+-]+$")

# Executed as ``python -I -c PROBE_SOURCE`` inside the image. Keep it
# dependency-free and side-effect free: it reads metadata and prints JSON.
PROBE_SOURCE = """\
import importlib.metadata as metadata
import importlib.util
import json
import platform
import runpy
import sys
from pathlib import Path

distributions = []
for distribution in metadata.distributions():
    name = distribution.metadata["Name"]
    version = distribution.version
    location = str(getattr(distribution, "_path", ""))
    distributions.append({"name": name, "version": version, "location": location})

cuda_build = None
spec = importlib.util.find_spec("torch")
if spec is not None and spec.submodule_search_locations:
    version_file = Path(list(spec.submodule_search_locations)[0]) / "version.py"
    if version_file.is_file():
        cuda_build = runpy.run_path(str(version_file)).get("cuda")

libc_name, libc_version = platform.libc_ver()
runtime = {
    "architecture": platform.machine(),
    "cuda_build": cuda_build,
    "libc": f"{libc_name} {libc_version}".strip(),
    "os": platform.system(),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
}
json.dump(
    {"schema_version": "synaptic-runtime-inventory-probe/v1",
     "executable": sys.executable,
     "distributions": distributions,
     "runtime": runtime},
    sys.stdout, sort_keys=True, separators=(",", ":"),
)
"""


class InventoryCaptureError(ValueError):
    """Raised when probe output cannot become a valid runtime inventory."""


@dataclass(frozen=True)
class CapturedInventory:
    payload: bytes
    sha256: str
    distribution_count: int
    runtime: Mapping[str, Any]


def probe_command(
    *, docker: str, image: str, python_executable: str
) -> list[str]:
    """Docker argv that runs the probe without network access or pulls."""

    if not image or any(character.isspace() for character in image):
        raise InventoryCaptureError("image to run must be a non-empty reference")
    if not _ABSOLUTE_EXECUTABLE.fullmatch(python_executable) or ".." in python_executable.split("/"):
        raise InventoryCaptureError("python executable must be an absolute path")
    return [
        docker, "run", "--rm", "--pull", "never", "--network", "none",
        "--platform", "linux/amd64", "--entrypoint", python_executable,
        image, "-I", "-c", PROBE_SOURCE,
    ]


def _probe_document(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > PROBE_MAX_OUTPUT_BYTES:
        raise InventoryCaptureError("probe output is empty or too large")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryCaptureError("probe output is not JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "executable", "distributions", "runtime"}
        or document["schema_version"] != PROBE_SCHEMA
        or not isinstance(document["distributions"], list)
        or not isinstance(document["runtime"], dict)
    ):
        raise InventoryCaptureError("probe output does not match its schema")
    return document


def build_inventory(
    probe_output: bytes, *, image: str, python_executable: str
) -> CapturedInventory:
    """Turn raw probe output into canonical, loader-valid inventory bytes."""

    if not _IMMUTABLE_IMAGE.fullmatch(image):
        raise InventoryCaptureError("inventory image must be an immutable sha256 reference")
    document = _probe_document(probe_output)
    if document["executable"] != python_executable:
        raise InventoryCaptureError(
            f"probe ran {document['executable']!r}, expected {python_executable!r}"
        )
    seen: dict[str, dict[str, str]] = {}
    for item in document["distributions"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "version", "location"}
            or not all(isinstance(item[key], str) for key in ("name", "version", "location"))
            or not item["name"].strip()
            or not item["version"].strip()
        ):
            raise InventoryCaptureError("probe reported a distribution without a name or version")
        normalized = canonicalize_name(item["name"])
        if normalized in seen:
            # Two metadata directories for one project mean the image has a
            # shadowed installation; fix the image rather than pick one.
            raise InventoryCaptureError(
                f"distribution {normalized!r} is installed more than once: "
                f"{seen[normalized]['location']} and {item['location']}"
            )
        seen[normalized] = item
    distributions = [
        {"name": seen[key]["name"], "version": seen[key]["version"]}
        for key in sorted(seen, key=lambda key: (key, seen[key]["version"]))
    ]
    versions = {key: value["version"] for key, value in seen.items()}
    runtime_in = document["runtime"]
    if set(runtime_in) != set(PLATFORM_FACTS):
        raise InventoryCaptureError("probe runtime facts do not match the fixed fact set")
    runtime: dict[str, Any] = {}
    for fact in PLATFORM_FACTS:
        value = runtime_in[fact]
        if not isinstance(value, str) or not value:
            raise InventoryCaptureError(f"probe could not determine runtime fact {fact}")
        runtime[fact] = value
    for fact, package in INVENTORY_VERSION_FACTS.items():
        version = versions.get(package)
        if version is None:
            raise InventoryCaptureError(
                f"runtime profiles require distribution {package!r} for fact {fact}"
            )
        runtime[fact] = version
    inventory = {
        "distributions": distributions,
        "image": image,
        "runtime": dict(sorted(runtime.items())),
        "schema_version": INVENTORY_SCHEMA,
    }
    payload = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    try:
        _inventory, count = parse_runtime_inventory(payload, image=image)
    except RuntimeProfileError as exc:
        raise InventoryCaptureError(f"captured inventory is invalid: {exc}") from exc
    return CapturedInventory(
        payload=payload,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        distribution_count=count,
        runtime=inventory["runtime"],
    )


def overlay_probe(
    base_output: bytes, derived_output: bytes, *, declared: Mapping[str, str]
) -> tuple[bytes, dict[str, tuple[str, str]]]:
    """Restrict a derived image's probe to the distributions of its base.

    ``declared`` maps each packaged-runtime bootstrap distribution to its pinned
    version. Bootstrap wheels that replace a base distribution (an ML stack
    change) stay in the result at their new version; purely additive ones (the
    packaged-runtime closure) are dropped, so the inventory has the same scope
    as one captured from the base image alone. Fails closed when the derived
    image removed a base distribution, added anything undeclared, or installed
    a declared wheel at a different version.

    Returns the restricted probe output and ``{name: (base, derived)}`` for
    every replaced distribution whose version changed.
    """

    base, derived = _probe_document(base_output), _probe_document(derived_output)
    if base["executable"] != derived["executable"]:
        raise InventoryCaptureError("base and derived probes ran different interpreters")
    pins = {canonicalize_name(name): version for name, version in declared.items()}

    def by_name(document: dict[str, Any]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for item in document["distributions"]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise InventoryCaptureError("probe reported a distribution without a name")
            normalized = canonicalize_name(item["name"])
            if normalized in result:
                raise InventoryCaptureError(f"distribution {normalized!r} is installed more than once")
            result[normalized] = item
        return result

    base_items, derived_items = by_name(base), by_name(derived)
    removed = sorted(set(base_items) - set(derived_items))
    if removed:
        raise InventoryCaptureError("derived image removed base distributions: " + ", ".join(removed))
    undeclared = sorted(set(derived_items) - set(base_items) - set(pins))
    if undeclared:
        raise InventoryCaptureError(
            "derived image added undeclared distributions: " + ", ".join(undeclared)
        )
    for name, version in sorted(pins.items()):
        installed = derived_items.get(name)
        if installed is None or installed["version"] != version:
            raise InventoryCaptureError(
                f"bootstrap {name} is pinned to {version} but the image has "
                f"{None if installed is None else installed['version']}"
            )
    changed = {
        name: (base_items[name]["version"], derived_items[name]["version"])
        for name in sorted(base_items)
        if base_items[name]["version"] != derived_items[name]["version"]
    }
    undeclared_changes = sorted(set(changed) - set(pins))
    if undeclared_changes:
        raise InventoryCaptureError(
            "derived image changed undeclared base distributions: " + ", ".join(undeclared_changes)
        )
    restricted = dict(derived)
    restricted["distributions"] = [derived_items[name] for name in sorted(base_items)]
    return json.dumps(restricted, sort_keys=True).encode("utf-8"), changed


def capture_dockerfile(*, base_image: str, python_executable: str) -> str:
    """Base plus the bootstrap wheels, installed and gated like a packaged build.

    The install command and ``pip check`` before/after gate match
    ``tuner.cloud.derived_training_image.render_dockerfile`` and the Modal
    build. The synaptic-tuner wheel is not installed: it is additive and
    source-specific, so it never appears in a runtime-profile inventory.
    """

    if not _ABSOLUTE_EXECUTABLE.fullmatch(python_executable):
        raise InventoryCaptureError("python executable must be an absolute path")
    python = python_executable
    return (
        f"FROM {base_image}\n"
        "COPY wheels/ /opt/synaptic-inventory-capture/\n"
        "RUN set -eu; \\\n"
        f"    before=\"$({python} -I -m pip check 2>&1)\" || test \"$?\" -eq 1; \\\n"
        f"    {python} -I -m pip install --no-index --no-deps --require-hashes --no-cache-dir "
        "-r /opt/synaptic-inventory-capture/requirements.txt; \\\n"
        f"    after=\"$({python} -I -m pip check 2>&1)\" || test \"$?\" -eq 1; \\\n"
        "    test \"$before\" = \"$after\"\n"
    )


def diff_inventories(expected: bytes, actual: bytes) -> list[str]:
    """Human-readable differences between two inventory documents."""

    left, right = json.loads(expected), json.loads(actual)
    lines: list[str] = []
    for key in ("schema_version", "image"):
        if left.get(key) != right.get(key):
            lines.append(f"{key}: {left.get(key)!r} -> {right.get(key)!r}")
    before = {canonicalize_name(d["name"]): d for d in left.get("distributions", [])}
    after = {canonicalize_name(d["name"]): d for d in right.get("distributions", [])}
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old != new:
            lines.append(
                f"distribution {name}: "
                f"{None if old is None else old['name'] + ' ' + old['version']} -> "
                f"{None if new is None else new['name'] + ' ' + new['version']}"
            )
    old_runtime, new_runtime = left.get("runtime", {}), right.get("runtime", {})
    for fact in sorted(set(old_runtime) | set(new_runtime)):
        if old_runtime.get(fact) != new_runtime.get(fact):
            lines.append(f"runtime {fact}: {old_runtime.get(fact)!r} -> {new_runtime.get(fact)!r}")
    return lines


__all__ = [
    "CapturedInventory",
    "InventoryCaptureError",
    "PLATFORM_FACTS",
    "PROBE_SCHEMA",
    "PROBE_SOURCE",
    "build_inventory",
    "capture_dockerfile",
    "diff_inventories",
    "overlay_probe",
    "probe_command",
]
