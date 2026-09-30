"""Declarative, provider-neutral runtime profiles for local training recipes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from packaging.utils import canonicalize_name


PROFILE_SCHEMA = "syntunia-runtime-profile/v1"
INVENTORY_SCHEMA = "syntunia-python-distribution-inventory/v1"

_PROFILE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class RuntimeProfileError(ValueError):
    """Raised when a named runtime profile or its inventory is invalid."""


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    path: Path
    profile_sha256: str
    image: str
    model_revisions: Mapping[str, tuple[str, ...]]
    methods: tuple[str, ...]
    inventory_path: Path
    inventory_sha256: str
    runtime_facts: Mapping[str, Any]
    distribution_count: int

    def resolve(
        self, *, model: str, model_revision: str, method: str
    ) -> "RuntimeProfile":
        revisions = self.model_revisions.get(model)
        if revisions is None:
            raise RuntimeProfileError(
                f"Runtime profile {self.name!r} does not support model {model!r}"
            )
        if not model_revision or model_revision not in revisions:
            raise RuntimeProfileError(
                f"Runtime profile {self.name!r} does not support model revision "
                f"{model_revision!r} for {model!r}"
            )
        if method not in self.methods:
            raise RuntimeProfileError(
                f"Runtime profile {self.name!r} does not support method {method!r}"
            )
        return self

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": PROFILE_SCHEMA,
            "profile_path": str(self.path),
            "profile_sha256": self.profile_sha256,
            "image": self.image,
            "inventory_path": str(self.inventory_path),
            "inventory_sha256": self.inventory_sha256,
            "distribution_count": self.distribution_count,
            "runtime_facts": dict(self.runtime_facts),
        }


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeProfileError(
            f"{label} must contain exactly: {', '.join(sorted(keys))}"
        )
    return value


def _nonempty_unique_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeProfileError(f"{label} must be a non-empty list")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise RuntimeProfileError(f"{label} must contain non-empty strings")
    if len(set(items)) != len(items):
        raise RuntimeProfileError(f"{label} must not contain duplicates")
    return items


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_stable_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    """Read one descriptor and reject symlinks and identity/content races."""

    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeProfileError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or _file_identity(opened_before) != _file_identity(before)
        ):
            raise RuntimeProfileError(f"{label} identity changed before open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise RuntimeProfileError(f"{label} exceeds its size limit")
        opened_after = os.fstat(descriptor)
        after = os.stat(path, follow_symlinks=False)
        identity = _file_identity(opened_before)
        if (
            _file_identity(opened_after) != identity
            or _file_identity(after) != identity
            or not stat.S_ISREG(after.st_mode)
        ):
            raise RuntimeProfileError(f"{label} identity changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_inventory(
    path: Path, *, expected_sha256: str, image: str
) -> tuple[dict[str, Any], int]:
    payload_bytes = _read_stable_regular_file(
        path, maximum=4 * 1024 * 1024, label="Runtime profile inventory"
    )
    actual_sha256 = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeProfileError("Runtime profile inventory digest mismatch")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeProfileError(
            "Runtime profile inventory is not valid JSON"
        ) from exc
    inventory = _exact_mapping(
        payload,
        {"schema_version", "image", "distributions", "runtime"},
        "runtime inventory",
    )
    if inventory["schema_version"] != INVENTORY_SCHEMA:
        raise RuntimeProfileError("Unsupported runtime inventory schema")
    if inventory["image"] != image:
        raise RuntimeProfileError(
            "Runtime inventory image does not match profile image"
        )
    distributions = inventory["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise RuntimeProfileError("Runtime inventory distributions must be non-empty")
    parsed: list[tuple[str, str, str]] = []
    normalized_names: set[str] = set()
    for item in distributions:
        entry = _exact_mapping(item, {"name", "version"}, "runtime distribution")
        name, version = entry["name"], entry["version"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
        ):
            raise RuntimeProfileError(
                "Runtime distribution name and version must be non-empty strings"
            )
        normalized = canonicalize_name(name)
        if normalized in normalized_names:
            raise RuntimeProfileError(
                "Runtime inventory contains colliding normalized distribution names"
            )
        normalized_names.add(normalized)
        parsed.append((normalized, version, name))
    if parsed != sorted(parsed, key=lambda item: (item[0], item[1])):
        raise RuntimeProfileError(
            "Runtime inventory distributions must be sorted by normalized name"
        )
    runtime = inventory["runtime"]
    if not isinstance(runtime, dict) or not runtime:
        raise RuntimeProfileError("Runtime inventory facts must be a non-empty object")
    if any(not isinstance(key, str) or not key for key in runtime):
        raise RuntimeProfileError(
            "Runtime inventory fact names must be non-empty strings"
        )
    try:
        json.dumps(runtime, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeProfileError(
            "Runtime inventory facts must be finite JSON values"
        ) from exc
    versions = {normalized: version for normalized, version, _name in parsed}
    fact_packages = {
        "torch_version": "torch",
        "transformers_version": "transformers",
        "trl_version": "trl",
        "unsloth_version": "unsloth",
        "unsloth_zoo_version": "unsloth-zoo",
    }
    for fact, package in fact_packages.items():
        value = runtime.get(fact)
        if not isinstance(value, str) or versions.get(package) != value:
            raise RuntimeProfileError(
                f"Runtime fact {fact} does not match distribution {package}"
            )
    return inventory, len(distributions)


def load_runtime_profile(name: str, profiles_dir: Path) -> RuntimeProfile:
    """Load ``<profiles_dir>/<name>.yaml`` and verify its bound inventory."""

    if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
        raise RuntimeProfileError("Runtime profile name is invalid")
    root = profiles_dir.resolve(strict=True)
    profile_candidate = root / f"{name}.yaml"
    if profile_candidate.parent != root:
        raise RuntimeProfileError("Runtime profile escapes the profile directory")
    profile_bytes = _read_stable_regular_file(
        profile_candidate, maximum=1024 * 1024, label="Runtime profile"
    )
    profile_path = profile_candidate.resolve(strict=True)
    if profile_path.parent != root:
        raise RuntimeProfileError("Runtime profile escapes the profile directory")
    try:
        raw = yaml.safe_load(profile_bytes)
    except yaml.YAMLError as exc:
        raise RuntimeProfileError("Runtime profile is not valid YAML") from exc
    profile = _exact_mapping(
        raw,
        {"schema_version", "name", "compatibility", "runtime"},
        "runtime profile",
    )
    if profile["schema_version"] != PROFILE_SCHEMA or profile["name"] != name:
        raise RuntimeProfileError("Runtime profile schema or name mismatch")
    compatibility = _exact_mapping(
        profile["compatibility"],
        {"models", "methods"},
        "runtime profile compatibility",
    )
    raw_models = compatibility["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise RuntimeProfileError("compatibility.models must be a non-empty list")
    model_revisions: dict[str, tuple[str, ...]] = {}
    for raw_model in raw_models:
        binding = _exact_mapping(
            raw_model, {"name", "revisions"}, "runtime model compatibility"
        )
        model_name = binding["name"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise RuntimeProfileError("runtime model name must be a non-empty string")
        if model_name in model_revisions:
            raise RuntimeProfileError("runtime model compatibility contains duplicates")
        model_revisions[model_name] = _nonempty_unique_strings(
            binding["revisions"], "runtime model revisions"
        )
    methods = _nonempty_unique_strings(
        compatibility["methods"], "compatibility.methods"
    )
    if methods != ("sft",):
        raise RuntimeProfileError("Runtime profile schema v1 supports only method sft")
    runtime = _exact_mapping(
        profile["runtime"], {"image", "inventory"}, "runtime profile runtime"
    )
    image = runtime["image"]
    if not isinstance(image, str) or not _IMMUTABLE_IMAGE.fullmatch(image):
        raise RuntimeProfileError(
            "Runtime profile image must be an immutable sha256 reference"
        )
    inventory_ref = _exact_mapping(
        runtime["inventory"], {"path", "sha256"}, "runtime profile inventory reference"
    )
    inventory_name, inventory_sha256 = inventory_ref["path"], inventory_ref["sha256"]
    if (
        not isinstance(inventory_name, str)
        or not inventory_name
        or Path(inventory_name).name != inventory_name
    ):
        raise RuntimeProfileError("Runtime inventory path must be a sibling filename")
    if not isinstance(inventory_sha256, str) or not _SHA256.fullmatch(inventory_sha256):
        raise RuntimeProfileError("Runtime inventory sha256 is invalid")
    inventory_candidate = root / inventory_name
    inventory_path = inventory_candidate.resolve(strict=True)
    if inventory_path.parent != root:
        raise RuntimeProfileError("Runtime inventory escapes the profile directory")
    inventory, distribution_count = _load_inventory(
        inventory_candidate, expected_sha256=inventory_sha256, image=image
    )
    return RuntimeProfile(
        name=name,
        path=profile_path,
        profile_sha256="sha256:" + hashlib.sha256(profile_bytes).hexdigest(),
        image=image,
        model_revisions=model_revisions,
        methods=methods,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha256,
        runtime_facts=inventory["runtime"],
        distribution_count=distribution_count,
    )
