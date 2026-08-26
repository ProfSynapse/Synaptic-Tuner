"""Method recipe contracts and canonical workload values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from synaptic_tuner.api.v1.training import CanonicalDocument
from tuner.project.execution_source import ExecutionSourceV1


WORKLOAD_FINGERPRINT_DOMAIN = b"synaptic-training-workload/v1\0"
MAX_WORKLOAD_BYTES = 256 * 1024


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode a JSON object in the one accepted byte representation."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON root must be a mapping")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("workload must contain only finite JSON values") from exc
    if not encoded or len(encoded) > MAX_WORKLOAD_BYTES:
        raise ValueError(f"canonical workload exceeds {MAX_WORKLOAD_BYTES} bytes")
    return encoded


@dataclass(frozen=True, slots=True)
class CompiledWorkload:
    """Immutable canonical bytes crossing the engine-runtime boundary."""

    method: str
    schema_version: str
    entrypoint: str
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes):
            raise TypeError("canonical_bytes must be bytes")
        if not self.method or not self.schema_version or not self.entrypoint:
            raise ValueError("workload identity fields are required")
        try:
            document = json.loads(self.canonical_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("canonical workload bytes are invalid JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("canonical workload must be a JSON object")
        if canonical_json_bytes(document) != self.canonical_bytes:
            raise ValueError("workload bytes are not canonically encoded")
        expected = (self.schema_version, self.method, self.entrypoint)
        actual = (
            document.get("schema_version"),
            document.get("method"),
            document.get("entrypoint"),
        )
        if actual != expected:
            raise ValueError("workload identity does not match its canonical document")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            WORKLOAD_FINGERPRINT_DOMAIN + self.canonical_bytes
        ).hexdigest()

    @property
    def document(self) -> dict[str, object]:
        value = json.loads(self.canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise TypeError("canonical workload must decode to an object")
        return value


@runtime_checkable
class TrainingRecipe(Protocol):
    method: str

    def compile(
        self,
        *,
        resolved_config: CanonicalDocument,
        execution_source: ExecutionSourceV1,
    ) -> CompiledWorkload: ...


class RecipeAlreadyRegistered(RuntimeError):
    pass


class RecipeNotRegistered(RuntimeError):
    pass


class RecipeRegistry:
    """Small method registry with no provider or persistence dependencies."""

    def __init__(self) -> None:
        self._recipes: dict[str, TrainingRecipe] = {}

    def register(self, recipe: TrainingRecipe) -> None:
        if not isinstance(recipe, TrainingRecipe):
            raise TypeError("recipe must implement TrainingRecipe")
        method = recipe.method.strip().lower()
        if not method:
            raise ValueError("recipe method is required")
        if method in self._recipes:
            raise RecipeAlreadyRegistered(f"recipe is already registered: {method}")
        self._recipes[method] = recipe

    def resolve(self, method: str) -> TrainingRecipe:
        try:
            return self._recipes[method.strip().lower()]
        except (AttributeError, KeyError) as exc:
            raise RecipeNotRegistered(f"training method is not registered: {method}") from exc

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._recipes))


__all__ = [
    "CompiledWorkload",
    "MAX_WORKLOAD_BYTES",
    "RecipeAlreadyRegistered",
    "RecipeNotRegistered",
    "RecipeRegistry",
    "TrainingRecipe",
    "WORKLOAD_FINGERPRINT_DOMAIN",
    "canonical_json_bytes",
]
