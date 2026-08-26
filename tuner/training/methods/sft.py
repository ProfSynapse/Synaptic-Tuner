"""Strict compiler for the versioned provider-neutral SFT workload."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping

from synaptic_tuner.api.v1.training import CanonicalDocument
from tuner.project.execution_source import ExecutionSourceV1
from tuner.runtime.artifacts import ArtifactContract, ArtifactRequirement
from tuner.training.recipes import (
    CompiledWorkload,
    TrainingRecipe,
    canonical_json_bytes,
)


SFT_CONFIG_SCHEMA = "synaptic-sft-config/v1"
SFT_WORKLOAD_SCHEMA = "synaptic-sft-workload/v1"
SFT_ENTRYPOINT = "Trainers/sft/runtime_v1.py"
SFT_RUNTIME_REQUIREMENTS_SCHEMA = "synaptic-sft-runtime-requirements/v1"
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

SFT_ARTIFACT_CONTRACT = ArtifactContract(
    schema_version="synaptic-sft-artifacts/v1",
    requirements=tuple(
        ArtifactRequirement(role)
        for role in (
            "workload_record",
            "training_lineage",
            "training_metrics",
            "final_model",
            "tokenizer",
        )
    ),
)


def _runtime_requirements() -> dict[str, object]:
    return {
        "schema_version": SFT_RUNTIME_REQUIREMENTS_SCHEMA,
        "python": {
            "implementation": "cpython",
            "minimum_version": "3.11",
            "maximum_version_exclusive": "3.14",
        },
        "isolation": {"no_user_site": True, "safe_path": True},
        "allowed_environment": [
            "COMSPEC", "CUDA_VISIBLE_DEVICES", "LANG", "LC_ALL",
            "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "PATH", "PATHEXT",
            "PYTHONIOENCODING", "SystemRoot", "WINDIR", "PYTHONNOUSERSITE",
            "PYTHONSAFEPATH", "PYTHONPATH", "HF_HOME", "TRANSFORMERS_CACHE",
            "WANDB_DISABLED", "SYNAPTIC_ENGINE_ROOT", "SYNAPTIC_PROJECT_ROOT",
            "SYNAPTIC_ARTIFACT_ROOT", "SYNAPTIC_STATE_ROOT",
            "SYNAPTIC_TRACKING_ROOT", "SYNAPTIC_CACHE_ROOT", "SYNAPTIC_TMP_ROOT",
        ],
        "trainer_projection_schema": "synaptic-sft-trainer-projection/v1",
        "artifact_formats": {
            "model": ["peft-safetensors", "full-safetensors"],
            "tokenizer": "tokenizer-json",
        },
    }


def _text(value: object, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or len(value) > maximum or value != value.strip():
        raise ValueError(f"{name} must be bounded nonblank text")
    if any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _revision(value: object, name: str) -> str:
    revision = _text(value, name, maximum=64)
    if _REVISION.fullmatch(revision) is None:
        raise ValueError(f"{name} must be an exact lowercase 40- or 64-hex revision")
    return revision


def _resource(value: object, name: str, required: tuple[str, ...]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    _text(result["ref"], f"{name}.ref")
    _revision(result["revision"], f"{name}.revision")
    digest = result.get("content_digest")
    if digest is not None and (
        not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None
    ):
        raise ValueError(f"{name}.content_digest must be a lowercase SHA-256 digest")
    return result


def compile_sft_workload(
    *,
    resolved_config: CanonicalDocument,
    execution_source: ExecutionSourceV1,
) -> CompiledWorkload:
    if not isinstance(resolved_config, CanonicalDocument):
        raise TypeError("resolved_config must be a CanonicalDocument")
    if not isinstance(execution_source, ExecutionSourceV1):
        raise TypeError("execution_source must be an ExecutionSourceV1")
    config = resolved_config.to_dict()
    if config.get("schema_version") != SFT_CONFIG_SCHEMA:
        raise ValueError(f"resolved SFT config must use {SFT_CONFIG_SCHEMA}")
    if config.get("method") != "sft":
        raise ValueError("resolved SFT config must declare method 'sft'")
    model = _resource(
        config.get("model"),
        "model",
        ("ref", "revision", "tokenizer_revision", "load_in_4bit"),
    )
    _revision(model["tokenizer_revision"], "model.tokenizer_revision")
    if not isinstance(model["load_in_4bit"], bool):
        raise TypeError("model.load_in_4bit must be a boolean")
    dataset = _resource(config.get("dataset"), "dataset", ("ref", "revision"))
    method_config = config.get("sft")
    if not isinstance(method_config, Mapping):
        raise TypeError("sft must be a mapping")
    config_bytes = resolved_config.canonical_json.encode("utf-8")
    config_revision = hashlib.sha256(config_bytes).hexdigest()
    requirements = [
        {
            "role": item.role,
            "minimum": item.minimum,
            "maximum": item.maximum,
        }
        for item in SFT_ARTIFACT_CONTRACT.requirements
    ]
    document: dict[str, object] = {
        "schema_version": SFT_WORKLOAD_SCHEMA,
        "method": "sft",
        "entrypoint": SFT_ENTRYPOINT,
        "execution_source": execution_source.to_dict(),
        "configuration": {
            "revision": config_revision,
            "document": config,
        },
        "identities": {"model": model, "dataset": dataset},
        "runtime_requirements": _runtime_requirements(),
        "artifacts": {
            "schema_version": SFT_ARTIFACT_CONTRACT.schema_version,
            "requirements": requirements,
        },
    }
    return CompiledWorkload(
        method="sft",
        schema_version=SFT_WORKLOAD_SCHEMA,
        entrypoint=SFT_ENTRYPOINT,
        canonical_bytes=canonical_json_bytes(document),
    )


@dataclass(frozen=True, slots=True)
class SFTRecipe(TrainingRecipe):
    method: str = "sft"

    def compile(
        self,
        *,
        resolved_config: CanonicalDocument,
        execution_source: ExecutionSourceV1,
    ) -> CompiledWorkload:
        return compile_sft_workload(
            resolved_config=resolved_config,
            execution_source=execution_source,
        )


__all__ = [
    "SFT_ARTIFACT_CONTRACT",
    "SFT_CONFIG_SCHEMA",
    "SFT_ENTRYPOINT",
    "SFT_RUNTIME_REQUIREMENTS_SCHEMA",
    "SFT_WORKLOAD_SCHEMA",
    "SFTRecipe",
    "compile_sft_workload",
]
