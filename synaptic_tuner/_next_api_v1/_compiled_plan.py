"""Sealed compiler receipts and compiled-plan authority boundaries."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ._workloads import CanonicalWorkload, decode_sft_workload

_FIXTURE_PLAN_SEAL = object()
_RECEIPT_SEAL = object()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _domain_digest(domain: bytes, value: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + value).hexdigest()


def _bindings(document: Mapping[str, Any], workload_digest: str) -> dict[str, str]:
    return {
        "schema_version": document["schema_version"],
        "entrypoint": document["entrypoint"],
        "workload_digest": workload_digest,
        "source_digest": document["code"]["source_digest"],
        "dependency_lock_digest": document["runtime"]["dependency_lock_digest"],
        "engine_commit": document["code"]["engine_commit"],
        "model_repository": document["model"]["repository"],
        "model_revision": document["model"]["revision"],
        "tokenizer_revision": document["model"]["tokenizer_revision"],
        "dataset_repository": document["dataset"]["repository"],
        "dataset_revision": document["dataset"]["revision"],
        "dataset_split": document["dataset"]["split"],
        "dataset_file_selector": document["dataset"]["file_selector"],
        "target_profile_id": document["adapter"]["target_profile_id"],
        "target_profile_digest": document["adapter"]["target_profile_digest"],
    }


@dataclass(frozen=True, slots=True, init=False)
class CompilerReceipt:
    canonical_request: bytes
    request_digest: str
    canonical_resolution: bytes
    resolution_digest: str
    workload_digest: str
    plan_projection: bytes
    plan_digest: str
    duplicate_bindings: tuple[tuple[str, str], ...]

    def __init__(self, canonical_request: bytes, canonical_resolution: bytes,
                 workload: CanonicalWorkload, plan_projection: bytes,
                 duplicate_bindings: tuple[tuple[str, str], ...], *,
                 _seal: object = None) -> None:
        if _seal is not _RECEIPT_SEAL:
            raise TypeError("compiler receipts are compiler-created values")
        if decode_sft_workload(workload.canonical_bytes).digest != workload.digest:
            raise ValueError("workload is not authoritative")
        for name, value in (("request", canonical_request), ("resolution", canonical_resolution),
                            ("projection", plan_projection)):
            if not isinstance(value, bytes) or not value:
                raise TypeError(f"canonical {name} must be bytes")
            decoded = json.loads(value.decode("utf-8"))
            if _canonical_bytes(decoded) != value:
                raise ValueError(f"canonical {name} bytes are not canonical JSON")
        expected = tuple(sorted(_bindings(workload.document, workload.digest).items()))
        if duplicate_bindings != expected:
            raise ValueError("compiler duplicate bindings do not match workload")
        projection = json.loads(plan_projection.decode("utf-8"))
        for key, value in expected:
            if projection.get(key) != value:
                raise ValueError(f"plan projection does not bind {key}")
        object.__setattr__(self, "canonical_request", canonical_request)
        object.__setattr__(self, "request_digest", _domain_digest(b"synaptic.sft-request/v1", canonical_request))
        object.__setattr__(self, "canonical_resolution", canonical_resolution)
        object.__setattr__(self, "resolution_digest", _domain_digest(b"synaptic.sft-resolution/v1", canonical_resolution))
        object.__setattr__(self, "workload_digest", workload.digest)
        object.__setattr__(self, "plan_projection", plan_projection)
        object.__setattr__(self, "plan_digest", _domain_digest(b"synaptic.sft-plan-projection/v1", plan_projection))
        object.__setattr__(self, "duplicate_bindings", expected)


@dataclass(frozen=True, slots=True, init=False)
class FixtureCompiledTrainingPlan:
    workload: CanonicalWorkload
    receipt: CompilerReceipt
    _seal: object

    def __init__(self, workload: CanonicalWorkload, receipt: CompilerReceipt, *,
                 _seal: object = None) -> None:
        if _seal is not _FIXTURE_PLAN_SEAL:
            raise TypeError("fixture compiled plans are compiler-created values")
        if not isinstance(receipt, CompilerReceipt) or receipt.workload_digest != workload.digest:
            raise ValueError("compiler receipt does not bind workload")
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "_seal", _FIXTURE_PLAN_SEAL)

    @property
    def fingerprint(self) -> str:
        return self.receipt.plan_digest

    def __repr__(self) -> str:
        return f"FixtureCompiledTrainingPlan(plan_digest={self.fingerprint!r})"


class ProductionCompilationUnavailable(RuntimeError):
    code = "production_compilation_unavailable"


class ProductionCompiledTrainingPlan:
    """Future resolver result; no concrete construction path exists offline."""

    __slots__ = ()

    def __init__(self, *_: object, **__: object) -> None:
        raise ProductionCompilationUnavailable("production compilation is unavailable")


def _verify_fixture_inputs(request: Mapping[str, Any], resolution: Mapping[str, Any],
                           workload: CanonicalWorkload) -> None:
    d = workload.document
    duration = d["optimization"]["duration"]
    expected_request = {
        "max_steps": duration.get("max_steps"), "epochs": duration.get("epochs"),
        "per_device_train_batch_size": d["optimization"]["batch_size"],
        "gradient_accumulation_steps": d["optimization"]["gradient_accumulation_steps"],
        "learning_rate": d["optimization"]["learning_rate"],
        "max_seq_length": d["objective"]["max_seq_length"], "seed": d["optimization"]["seed"],
        "warmup_ratio": d["optimization"]["warmup_ratio"],
        "weight_decay": d["optimization"]["weight_decay"],
        "adapter_rank": d["adapter"]["rank"], "adapter_alpha": d["adapter"]["alpha"],
        "adapter_dropout": d["adapter"]["dropout"], "record_format": d["dataset"]["record_format"],
        "loss_scope": d["objective"]["loss_scope"], "prompt_render": d["objective"]["prompt_render"],
        "optimizer": d["optimization"]["optimizer"], "scheduler": d["optimization"]["scheduler"],
        "max_grad_norm": d["optimization"]["max_grad_norm"],
        "compute_dtype": d["optimization"]["dtype"],
        "model_quantization": d["optimization"]["model_quantization"],
        "gradient_checkpointing": d["optimization"]["gradient_checkpointing"],
        "logging_steps": d["logging"]["structured_metrics_steps"],
        "checkpoint_strategy": d["checkpoints"]["strategy"],
        "checkpoint_steps": d["checkpoints"]["steps"], "checkpoint_limit": d["checkpoints"]["limit"],
        "packing": d["objective"]["packing"],
        "adapter_initialization": d["adapter"]["initialization"],
        "use_rslora": d["adapter"]["use_rslora"], "use_dora": d["adapter"]["use_dora"],
        "target_modules": [],
    }
    p = d["provenance"]
    expected_resolution = {
        "provenance": {"fixture_manifest_digest": p["fixture_manifest_digest"]},
        "model_repository": d["model"]["repository"], "model_revision": d["model"]["revision"],
        "tokenizer_revision": d["model"]["tokenizer_revision"],
        "dataset": dict(d["dataset"]), "engine_commit": d["code"]["engine_commit"],
        "source_digest": d["code"]["source_digest"],
        "dependency_lock_digest": d["runtime"]["dependency_lock_digest"],
        "target_profile_id": d["adapter"]["target_profile_id"],
        "target_profile_digest": d["adapter"]["target_profile_digest"],
        "target_modules": list(d["adapter"]["target_modules"]),
    }
    if _canonical_bytes(request) != _canonical_bytes(expected_request):
        raise ValueError("canonical request does not exactly compile to workload")
    if _canonical_bytes(resolution) != _canonical_bytes(expected_resolution):
        raise ValueError("canonical resolution does not exactly compile to workload")

def _compile_fixture_plan(request: Mapping[str, Any], resolution: Mapping[str, Any],
                          workload: CanonicalWorkload) -> FixtureCompiledTrainingPlan:
    _verify_fixture_inputs(request, resolution, workload)
    request_bytes = _canonical_bytes(request)
    resolution_bytes = _canonical_bytes(resolution)
    bindings = tuple(sorted(_bindings(workload.document, workload.digest).items()))
    projection = _canonical_bytes(dict(bindings))
    receipt = CompilerReceipt(request_bytes, resolution_bytes, workload, projection, bindings,
                              _seal=_RECEIPT_SEAL)
    return FixtureCompiledTrainingPlan(workload, receipt, _seal=_FIXTURE_PLAN_SEAL)


__all__ = [
    "CompilerReceipt", "FixtureCompiledTrainingPlan", "ProductionCompilationUnavailable",
    "ProductionCompiledTrainingPlan",
]
