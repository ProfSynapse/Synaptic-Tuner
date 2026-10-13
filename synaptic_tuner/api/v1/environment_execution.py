"""Effect-free, host-resolved environment execution contracts.

Schema validity and a resolved binding are input admission, not authority to
create an environment. Adapters separately enforce authorization and custody.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Protocol

from ._contract import contract_digest, digest_text, exact_fields


ENVIRONMENT_BINDING_SCHEMA = "synaptic-environment-binding/v1"
ENVIRONMENT_DATA_PLAN_SCHEMA = "synaptic-dataset-plan/v2"
MAX_DEPENDENCY_BYTES = 4 * 1024 * 1024
MAX_DEPENDENCIES = 128
MAX_DEPENDENCY_TOTAL_BYTES = 16 * 1024 * 1024
_REF = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")


def logical_ref(value: object, name: str) -> str:
    if type(value) is not str or _REF.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded logical reference")
    return value


def _positive(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an exact positive bounded integer")
    return value


class EnvironmentMode(str, Enum):
    EXECUTE = "execute"
    SIMULATE = "simulate"


@dataclass(frozen=True, slots=True)
class EnvironmentSelectionV1:
    profile_ref: str
    mode: EnvironmentMode

    def __post_init__(self) -> None:
        logical_ref(self.profile_ref, "profile_ref")
        if type(self.mode) is not EnvironmentMode:
            raise TypeError("mode must be exact EnvironmentMode")

    def to_dict(self) -> dict[str, object]:
        return {"profile_ref": self.profile_ref, "mode": self.mode.value}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EnvironmentSelectionV1:
        value = exact_fields(value, frozenset({"profile_ref", "mode"}), "environment_selection")
        if type(value["mode"]) is not str:
            raise TypeError("mode must be a string")
        return cls(value["profile_ref"], EnvironmentMode(value["mode"]))


@dataclass(frozen=True, slots=True)
class EnvironmentBoundsV1:
    max_rows: int
    max_turns: int
    max_tool_calls: int
    max_fixture_files: int
    max_fixture_bytes: int
    max_output_bytes: int
    max_snapshot_bytes: int
    max_seconds: int

    def __post_init__(self) -> None:
        for field in fields(self):
            maximum = 3600 if field.name == "max_seconds" else (
                16 * 1024 * 1024 if field.name.endswith("_bytes") else 10000
            )
            _positive(getattr(self, field.name), field.name, maximum)

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EnvironmentBoundsV1:
        value = exact_fields(value, frozenset(field.name for field in fields(cls)), "environment_bounds")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EnvironmentDependencyV1:
    role: str
    ref: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        logical_ref(self.role, "role")
        logical_ref(self.ref, "ref")
        digest_text(self.sha256, "sha256")
        _positive(self.size_bytes, "size_bytes", MAX_DEPENDENCY_BYTES)

    def to_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EnvironmentDependencyV1:
        value = exact_fields(value, frozenset(field.name for field in fields(cls)), "environment_dependency")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class EnvironmentBindingV1:
    schema_version: str
    selection: EnvironmentSelectionV1
    backend_ref: str
    backend_revision: str
    scope_ref: str
    bounds: EnvironmentBoundsV1
    allowed_actions: tuple[str, ...]
    dependencies: tuple[EnvironmentDependencyV1, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != ENVIRONMENT_BINDING_SCHEMA:
            raise ValueError("unsupported environment binding schema")
        if type(self.selection) is not EnvironmentSelectionV1 or type(self.bounds) is not EnvironmentBoundsV1:
            raise TypeError("exact selection and bounds required")
        for name in ("backend_ref", "scope_ref"):
            logical_ref(getattr(self, name), name)
        digest_text(self.backend_revision, "backend_revision")
        if type(self.allowed_actions) is not tuple or not 1 <= len(self.allowed_actions) <= 128:
            raise ValueError("allowed_actions must be a bounded nonempty tuple")
        for action in self.allowed_actions:
            logical_ref(action, "action")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("actions must be unique")
        object.__setattr__(self, "allowed_actions", tuple(sorted(self.allowed_actions)))
        if type(self.dependencies) is not tuple or not 1 <= len(self.dependencies) <= MAX_DEPENDENCIES:
            raise ValueError("dependencies must be a bounded nonempty tuple")
        if any(type(item) is not EnvironmentDependencyV1 for item in self.dependencies):
            raise TypeError("exact dependencies required")
        refs = [item.ref for item in self.dependencies]
        if len(set(refs)) != len(refs):
            raise ValueError("dependency references must be unique")
        if sum(item.size_bytes for item in self.dependencies) > MAX_DEPENDENCY_TOTAL_BYTES:
            raise ValueError("dependency aggregate exceeds bound")
        object.__setattr__(self, "dependencies", tuple(sorted(self.dependencies, key=lambda item: item.ref)))
        if len(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")) > 8192:
            raise ValueError("environment binding exceeds document byte bound")

    @property
    def binding_digest(self) -> str:
        return contract_digest(ENVIRONMENT_BINDING_SCHEMA, self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "selection": self.selection.to_dict(),
            "backend_ref": self.backend_ref, "backend_revision": self.backend_revision,
            "scope_ref": self.scope_ref, "bounds": self.bounds.to_dict(),
            "allowed_actions": list(self.allowed_actions),
            "dependencies": [item.to_dict() for item in self.dependencies],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EnvironmentBindingV1:
        value = exact_fields(value, frozenset(field.name for field in fields(cls)), "environment_binding")
        if type(value["allowed_actions"]) is not list or type(value["dependencies"]) is not list:
            raise TypeError("actions and dependencies must be exact arrays")
        return cls(
            value["schema_version"], EnvironmentSelectionV1.from_dict(value["selection"]),
            value["backend_ref"], value["backend_revision"], value["scope_ref"],
            EnvironmentBoundsV1.from_dict(value["bounds"]), tuple(value["allowed_actions"]),
            tuple(EnvironmentDependencyV1.from_dict(item) for item in value["dependencies"]),
        )


class EnvironmentExecutionPort(Protocol):
    """Trusted effect-free host resolution; C1 never calls an execution method."""

    def resolve(self, selection: EnvironmentSelectionV1, *, project_ref: str) -> EnvironmentBindingV1: ...

    def read_dependency(self, ref: str, *, maximum_bytes: int) -> bytes: ...
