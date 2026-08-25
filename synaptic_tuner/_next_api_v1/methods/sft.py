"""Strict, pure compiler for the closed SFT workload contract."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .._compiled_plan import (
    FixtureCompiledTrainingPlan, ProductionCompilationUnavailable,
    ProductionCompiledTrainingPlan, _compile_fixture_plan,
)
from .._workloads import (
    CanonicalWorkload, FixtureVerified, LiveVerified, RemoteDatasetBinding,
    SFT_ENTRYPOINT, SFT_WORKLOAD_SCHEMA, canonical_sft_workload, require_live_verified,
    _digest, _repository, _revision,
)

ARTIFACT_ROLES = (
    "workload_record", "training_lineage", "training_metrics", "final_adapter", "tokenizer"
)


def canonical_decimal(value: object, name: str, *, positive: bool = False,
                      maximum: Decimal | None = None) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise TypeError(f"{name} must be a decimal-compatible scalar")
    if isinstance(value, float):
        value = str(value)
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not number.is_finite() or (positive and number <= 0) or (not positive and number < 0):
        raise ValueError(f"{name} is outside its accepted range")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    if number == 0:
        return "0"
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered.startswith("+"):
        rendered = rendered[1:]
    return rendered


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class SFTSpec:
    max_steps: int | None = None
    epochs: object | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: object = "0.0002"
    max_seq_length: int = 2048
    seed: int = 3407
    warmup_ratio: object = "0"
    weight_decay: object = "0"
    adapter_rank: int = 16
    adapter_alpha: int = 16
    adapter_dropout: object = "0"
    record_format: str = "conversations"
    loss_scope: str = "assistant_messages"
    prompt_render: str = "full_conversation"
    optimizer: str = "adamw_8bit"
    scheduler: str = "linear"
    max_grad_norm: object = "1"
    compute_dtype: str = "bf16"
    model_quantization: str = "none"
    gradient_checkpointing: str = "unsloth"
    logging_steps: int = 1
    checkpoint_strategy: str = "none"
    checkpoint_steps: int | None = None
    checkpoint_limit: int = 0
    packing: bool = False
    adapter_initialization: str = "default"
    use_rslora: bool = False
    use_dora: bool = False
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.max_steps is None) == (self.epochs is None):
            raise ValueError("exactly one of max_steps or epochs is required")
        if self.max_steps is not None:
            _positive_int(self.max_steps, "max_steps")
        if self.epochs is not None:
            object.__setattr__(self, "epochs", canonical_decimal(self.epochs, "epochs", positive=True))
        for name in ("per_device_train_batch_size", "gradient_accumulation_steps",
                     "max_seq_length", "adapter_rank", "adapter_alpha", "logging_steps"):
            _positive_int(getattr(self, name), name)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, positive, maximum in (
            ("learning_rate", True, None), ("warmup_ratio", False, Decimal(1)),
            ("weight_decay", False, None), ("adapter_dropout", False, Decimal(1)),
            ("max_grad_norm", True, None),
        ):
            object.__setattr__(self, name, canonical_decimal(
                getattr(self, name), name, positive=positive, maximum=maximum
            ))
        constants = {
            "record_format": "conversations", "prompt_render": "full_conversation",
            "optimizer": "adamw_8bit", "scheduler": "linear", "compute_dtype": "bf16",
            "model_quantization": "none", "gradient_checkpointing": "unsloth",
            "adapter_initialization": "default",
        }
        for name, expected in constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")
        if self.loss_scope not in {"assistant_messages", "full_sequence"}:
            raise ValueError("loss_scope is not supported")
        if self.packing is not False or self.use_rslora is not False or self.use_dora is not False:
            raise ValueError("packing, rsLoRA, and DoRA are prohibited")
        if tuple(self.target_modules):
            raise ValueError("caller target_modules are prohibited")
        if self.checkpoint_strategy not in {"none", "steps"}:
            raise ValueError("checkpoint_strategy is not supported")
        if self.checkpoint_strategy == "steps":
            if self.checkpoint_steps is None:
                raise ValueError("checkpoint_steps is required for steps strategy")
            _positive_int(self.checkpoint_steps, "checkpoint_steps")
            _positive_int(self.checkpoint_limit, "checkpoint_limit")
        elif self.checkpoint_steps is not None or self.checkpoint_limit != 0:
            raise ValueError("none checkpoint strategy requires null steps and zero limit")


@dataclass(frozen=True, slots=True)
class FixtureSFTInputs:
    provenance: FixtureVerified
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    dataset: RemoteDatasetBinding
    engine_commit: str
    source_digest: str
    dependency_lock_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, FixtureVerified):
            raise TypeError("fixture inputs require FixtureVerified provenance")
        object.__setattr__(self, "model_repository", _repository(self.model_repository, "model.repository"))
        object.__setattr__(self, "model_revision", _revision(self.model_revision, "model.revision"))
        object.__setattr__(self, "tokenizer_revision", _revision(self.tokenizer_revision, "tokenizer_revision"))
        if not isinstance(self.dataset, RemoteDatasetBinding):
            raise TypeError("dataset must be RemoteDatasetBinding")
        object.__setattr__(self, "engine_commit", _revision(self.engine_commit, "engine_commit"))
        object.__setattr__(self, "source_digest", _digest(self.source_digest, "source_digest"))
        object.__setattr__(self, "dependency_lock_digest", _digest(self.dependency_lock_digest, "dependency_lock_digest"))


def _compile(spec: SFTSpec, resolved: Any, *, live: bool) -> CanonicalWorkload:
    if not isinstance(spec, SFTSpec):
        raise TypeError("spec must be SFTSpec")
    if live:
        resolved = require_live_verified(resolved)
        profile = resolved.target_profile
        provenance = {
            "class": "live_verified",
            "release_manifest_digest": resolved.release_manifest_digest,
            "resolution_attestation_digest": resolved.resolution_attestation_digest,
        }
        profile_id = profile.profile_id
        profile_digest = profile.profile_digest
        targets = profile.target_modules
    else:
        if not isinstance(resolved, FixtureSFTInputs):
            raise TypeError("fixture compilation requires FixtureSFTInputs")
        fixture = resolved.provenance
        provenance = {
            "class": "fixture_verified",
            "fixture_manifest_digest": fixture.fixture_manifest_digest,
        }
        profile_id = fixture.target_profile_id
        profile_digest = fixture.target_profile_digest
        targets = fixture.target_modules
    duration = ({"max_steps": spec.max_steps} if spec.max_steps is not None
                else {"epochs": spec.epochs})
    document = {
        "adapter": {
            "alpha": spec.adapter_alpha, "bias": "none", "dropout": spec.adapter_dropout,
            "initialization": "default", "rank": spec.adapter_rank,
            "target_modules": list(targets), "target_profile_digest": profile_digest,
            "target_profile_id": profile_id, "type": "lora", "use_dora": False,
            "use_rslora": False,
        },
        "artifacts": {"required_roles": list(ARTIFACT_ROLES)},
        "checkpoints": {"limit": spec.checkpoint_limit, "resume": None,
                        "steps": spec.checkpoint_steps, "strategy": spec.checkpoint_strategy},
        "code": {"engine_commit": resolved.engine_commit, "source_digest": resolved.source_digest},
        "dataset": {
            "file_selector": resolved.dataset.file_selector, "record_format": "conversations",
            "repository": resolved.dataset.repository, "revision": resolved.dataset.revision,
            "split": resolved.dataset.split,
        },
        "entrypoint": SFT_ENTRYPOINT,
        "kind": "sft",
        "logging": {"external_reporting": "none", "structured_metrics_steps": spec.logging_steps},
        "model": {
            "repository": resolved.model_repository, "revision": resolved.model_revision,
            "tokenizer_revision": resolved.tokenizer_revision, "trust_remote_code": False,
            "weights_format": "safetensors",
        },
        "objective": {
            "chat_template": "tokenizer_embedded_required", "loss_scope": spec.loss_scope,
            "masking_contract": "synaptic.sft-mask/conversation-prefix-v1",
            "max_seq_length": spec.max_seq_length, "packing": False,
            "prompt_render": "full_conversation", "truncation": "right",
        },
        "optimization": {
            "batch_size": spec.per_device_train_batch_size, "dtype": "bf16",
            "duration": duration, "gradient_accumulation_steps": spec.gradient_accumulation_steps,
            "gradient_checkpointing": "unsloth", "learning_rate": spec.learning_rate,
            "max_grad_norm": spec.max_grad_norm, "model_quantization": "none",
            "optimizer": "adamw_8bit", "scheduler": "linear", "seed": spec.seed,
            "warmup_ratio": spec.warmup_ratio, "weight_decay": spec.weight_decay,
        },
        "provenance": provenance,
        "runtime": {"dependency_lock_digest": resolved.dependency_lock_digest},
        "schema_version": SFT_WORKLOAD_SCHEMA,
    }
    workload = canonical_sft_workload(document)
    if live:
        raise ProductionCompilationUnavailable("production compilation is unavailable")
    request = {
        name: getattr(spec, name) for name in spec.__dataclass_fields__
    }
    resolution = {
        "provenance": {"fixture_manifest_digest": resolved.provenance.fixture_manifest_digest},
        "model_repository": resolved.model_repository, "model_revision": resolved.model_revision,
        "tokenizer_revision": resolved.tokenizer_revision, "dataset": {
            "repository": resolved.dataset.repository, "revision": resolved.dataset.revision,
            "split": resolved.dataset.split, "file_selector": resolved.dataset.file_selector,
            "record_format": resolved.dataset.record_format,
        }, "engine_commit": resolved.engine_commit, "source_digest": resolved.source_digest,
        "dependency_lock_digest": resolved.dependency_lock_digest,
        "target_profile_id": resolved.provenance.target_profile_id,
        "target_profile_digest": resolved.provenance.target_profile_digest,
        "target_modules": list(resolved.provenance.target_modules),
    }
    return _compile_fixture_plan(request, resolution, workload)


def compile_live_sft(spec: SFTSpec, resolved: LiveVerified) -> ProductionCompiledTrainingPlan:
    return _compile(spec, resolved, live=True)


def compile_fixture_sft(spec: SFTSpec, resolved: FixtureSFTInputs) -> FixtureCompiledTrainingPlan:
    return _compile(spec, resolved, live=False)


__all__ = [
    "ARTIFACT_ROLES", "FixtureSFTInputs", "SFTSpec", "canonical_decimal",
    "compile_fixture_sft", "compile_live_sft",
]
