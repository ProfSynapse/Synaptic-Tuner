"""Provider-neutral artifact roles and exact inventory verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath


_ROLE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _role(value: str) -> str:
    if not isinstance(value, str) or _ROLE.fullmatch(value) is None:
        raise ValueError("artifact role must be lowercase snake_case")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    role: str
    minimum: int = 1
    maximum: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        for name in ("minimum", "maximum"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.maximum < self.minimum:
            raise ValueError("artifact maximum cannot be below minimum")


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    schema_version: str
    requirements: tuple[ArtifactRequirement, ...]
    allow_unlisted_roles: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("artifact contract schema_version is required")
        requirements = tuple(self.requirements)
        if not requirements or any(
            not isinstance(item, ArtifactRequirement) for item in requirements
        ):
            raise TypeError("requirements must contain ArtifactRequirement values")
        roles = tuple(item.role for item in requirements)
        if len(roles) != len(set(roles)):
            raise ValueError("artifact contract roles must be unique")
        if not isinstance(self.allow_unlisted_roles, bool):
            raise TypeError("allow_unlisted_roles must be a boolean")
        object.__setattr__(self, "requirements", requirements)


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    role: str
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        if not isinstance(self.path, str) or not self.path or "\\" in self.path:
            raise ValueError("artifact path must be a relative POSIX path")
        path = PurePosixPath(self.path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact path must be contained and normalized")
        if path.as_posix() != self.path:
            raise ValueError("artifact path must use canonical POSIX syntax")
        if not isinstance(self.sha256, str) or _DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("artifact sha256 must be a lowercase digest")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("artifact size must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ArtifactInventory:
    entries: tuple[ArtifactEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(item, ArtifactEntry) for item in entries):
            raise TypeError("entries must contain ArtifactEntry values")
        paths = tuple(item.path for item in entries)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact inventory paths must be unique")
        object.__setattr__(self, "entries", entries)

    def for_role(self, role: str) -> tuple[ArtifactEntry, ...]:
        selected = _role(role)
        return tuple(item for item in self.entries if item.role == selected)


@dataclass(frozen=True, slots=True)
class InventoryVerification:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactIntegrity:
    artifact: ArtifactEntry
    valid: bool
    actual_size: int | None
    actual_sha256: str | None
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactEntry):
            raise TypeError("artifact must be an ArtifactEntry")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a boolean")
        if self.actual_size is not None and (
            not isinstance(self.actual_size, int)
            or isinstance(self.actual_size, bool)
            or self.actual_size < 0
        ):
            raise ValueError("actual_size must be a non-negative integer")
        if self.actual_sha256 is not None and (
            not isinstance(self.actual_sha256, str)
            or _DIGEST.fullmatch(self.actual_sha256) is None
        ):
            raise ValueError("actual_sha256 must be a lowercase digest")
        errors = tuple(self.errors)
        if self.valid == bool(errors):
            raise ValueError("integrity validity must agree with its errors")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True, slots=True)
class IntegrityVerification:
    valid: bool
    artifacts: tuple[ArtifactIntegrity, ...]

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactIntegrity) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactIntegrity values")
        if self.valid != all(item.valid for item in artifacts):
            raise ValueError("integrity validity must agree with artifact results")
        object.__setattr__(self, "artifacts", artifacts)


def verify_inventory(
    contract: ArtifactContract,
    inventory: ArtifactInventory,
) -> InventoryVerification:
    if not isinstance(contract, ArtifactContract):
        raise TypeError("contract must be an ArtifactContract")
    if not isinstance(inventory, ArtifactInventory):
        raise TypeError("inventory must be an ArtifactInventory")
    errors: list[str] = []
    expected = {item.role: item for item in contract.requirements}
    actual_roles = {item.role for item in inventory.entries}
    if not contract.allow_unlisted_roles:
        for role in sorted(actual_roles - set(expected)):
            errors.append(f"unexpected artifact role: {role}")
    for role, requirement in expected.items():
        count = len(inventory.for_role(role))
        if count < requirement.minimum or count > requirement.maximum:
            errors.append(
                f"artifact role {role} has {count} entries; expected "
                f"{requirement.minimum}..{requirement.maximum}"
            )
    return InventoryVerification(not errors, tuple(errors))


__all__ = [
    "ArtifactContract",
    "ArtifactEntry",
    "ArtifactIntegrity",
    "ArtifactInventory",
    "ArtifactRequirement",
    "IntegrityVerification",
    "InventoryVerification",
    "verify_inventory",
]
