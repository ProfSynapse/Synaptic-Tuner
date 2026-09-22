"""Method recipe contracts and canonical workload values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from tuner.training.contracts import CanonicalDocument
from tuner.project.execution_source import ExecutionSourceV1


WORKLOAD_FINGERPRINT_DOMAIN = b"synaptic-training-workload/v1\0"
MAX_WORKLOAD_BYTES = 256 * 1024


def selected_execution_mode(config: CanonicalDocument, *, required: bool = True) -> str:
    """Select explicitly, except at the historical Git-only API boundary."""
    value = config.to_dict()
    if "execution" not in value and not required:
        return "developer_integration"
    execution = value.get("execution")
    if (type(execution) is not dict or set(execution) != {"mode"}
            or type(execution["mode"]) is not str
            or execution["mode"] not in {"packaged_runtime", "developer_integration"}):
        raise ValueError("execution requires exactly one explicit supported mode")
    return execution["mode"]


def compile_execution_workload(*, resolved_config, execution_source, recipes,
                               require_mode: bool = False) -> "CompiledWorkload":
    if type(execution_source) is not ExecutionSourceV1:
        raise TypeError("developer compilation requires exact ExecutionSourceV1")
    mode = selected_execution_mode(resolved_config, required=require_mode)
    if mode != "developer_integration":
        raise ValueError("developer compilation requires developer_integration mode")
    config = resolved_config.to_dict()
    if "execution" in config and set(config) & {
        "runtime_release", "provider_runtime_binding", "execution_binding",
        "packaged_execution_binding",
    }:
        raise ValueError("selected developer configuration has mixed execution fields")
    method = config.get("method")
    if type(method) is not str or not method.strip():
        raise ValueError("resolved config requires a method")
    return recipes.resolve(method).compile(
        resolved_config=resolved_config, execution_source=execution_source,
    )


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
