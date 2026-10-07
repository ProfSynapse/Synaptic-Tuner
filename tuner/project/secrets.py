"""Opaque secret identifiers and the execution-time resolution boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from .errors import SecretReferenceError, SecretUnavailableError

SecretProvider = Literal["env", "provider_secret", "credential_helper"]
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:token|secret|password|api_key|credential)(?:$|_)", re.IGNORECASE
)


@dataclass(frozen=True)
class SecretRef:
    """A serializable identifier. It never contains the referenced value."""

    provider: SecretProvider
    name: str

    def __post_init__(self) -> None:
        if self.provider not in {"env", "provider_secret", "credential_helper"}:
            raise SecretReferenceError(f"Unsupported secret provider: {self.provider!r}")
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise SecretReferenceError("Secret names must be non-empty, stable identifiers")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SecretRef":
        unknown = set(value) - {"provider", "name"}
        if unknown:
            raise SecretReferenceError(
                "SecretRef accepts only provider and name",
                details={"unknown_fields": sorted(unknown)},
            )
        provider = value.get("provider")
        name = value.get("name")
        if not isinstance(provider, str) or not isinstance(name, str):
            raise SecretReferenceError("SecretRef provider and name must be strings")
        return cls(provider=provider, name=name)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "name": self.name}

    def __repr__(self) -> str:
        return f"SecretRef(provider={self.provider!r}, name={self.name!r})"


SecretResolver = Callable[[str], str | None]


def resolve_secret(
    reference: SecretRef,
    *,
    environment: Mapping[str, str] | None = None,
    provider_secret: SecretResolver | None = None,
    credential_helper: SecretResolver | None = None,
) -> str:
    """Resolve a secret only at execution time without adding it to any model."""

    if reference.provider == "env":
        value = (environment or {}).get(reference.name)
    elif reference.provider == "provider_secret":
        value = provider_secret(reference.name) if provider_secret else None
    else:
        value = credential_helper(reference.name) if credential_helper else None
    if not value:
        raise SecretUnavailableError(
            f"Secret reference {reference.name!r} is unavailable",
            details={"provider": reference.provider, "name": reference.name},
        )
    return value


def redact_secrets(value: object) -> object:
    """Convert SecretRefs to identifiers while recursively preserving structure."""

    if isinstance(value, SecretRef):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def reject_literal_secrets(value: object, *, path: str = "") -> None:
    """Fail when a conventionally sensitive field contains a literal value."""

    if isinstance(value, SecretRef):
        return
    if isinstance(value, dict):
        if set(value) == {"provider", "name"}:
            SecretRef.from_dict(value)
            return
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if _SENSITIVE_KEY_RE.search(str(key)) and not isinstance(item, SecretRef):
                if not (isinstance(item, dict) and set(item) == {"provider", "name"}):
                    raise SecretReferenceError(
                        f"Sensitive field {key_path!r} must contain a SecretRef",
                        details={"path": key_path},
                    )
            reject_literal_secrets(item, path=key_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_literal_secrets(item, path=f"{path}[{index}]")
