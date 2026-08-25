"""Frozen private-staging provider-neutral training API v1 contract gates."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import synaptic_tuner._next_api_v1 as next_api_v1
from synaptic_tuner._next_api_v1 import training
from synaptic_tuner._next_api_v1.execution import (
    ArtifactRef,
    ArtifactVerificationState,
    AuthorizationRequirement,
    RunRef,
    RunState,
    RunStatus,
)
from synaptic_tuner._next_api_v1.training import (
    AdapterSpec,
    ArtifactPolicy,
    DatasetSpec,
    ExecutionSpec,
    ModelSpec,
    ResolvedTrainingRequest,
    TrainingMethod,
    TrainingOutcome,
    TrainingParameters,
    TrainingPlan,
    TrainingPreflight,
    TrainingRequest,
    TrainingSubmission,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
IMAGE_DIGEST = f"sha256:{'c' * 64}"
MODEL_REVISION = "1" * 40
TOKENIZER_REVISION = "2" * 40
DATASET_REVISION = "3" * 40


def _model(*, resolved: bool = True) -> ModelSpec:
    return ModelSpec(
        identifier="org/model",
        revision=MODEL_REVISION if resolved else None,
        tokenizer_revision=TOKENIZER_REVISION if resolved else None,
    )


def _dataset(*, resolved: bool = True) -> DatasetSpec:
    return DatasetSpec(
        identifier="org/dataset",
        revision=DATASET_REVISION if resolved else None,
        file="train.jsonl",
    )


def _execution(*, resolved: bool = True) -> ExecutionSpec:
    return ExecutionSpec(
        provider="modal",
        accelerator="L4",
        runtime_image="registry.example/trainer@" + IMAGE_DIGEST if resolved else None,
        runtime_image_digest=IMAGE_DIGEST if resolved else None,
        dependency_lock_digest=HEX_B if resolved else None,
        allowed_secret_refs=("secret://hf/token",),
    )


def _request(*, resolved_execution: bool = True) -> TrainingRequest:
    return TrainingRequest(
        method=TrainingMethod.SFT,
        model=_model(resolved=False),
        dataset=_dataset(resolved=False),
        parameters=TrainingParameters(max_steps=5),
        adapter=AdapterSpec(mode="lora", rank=16, alpha=32),
        execution=_execution(resolved=resolved_execution),
    )


def _plan(**overrides) -> TrainingPlan:
    values = {
        "method": TrainingMethod.SFT,
        "model": _model(),
        "dataset": _dataset(),
        "parameters": TrainingParameters(max_steps=5),
        "adapter": AdapterSpec(mode="lora", rank=16, alpha=32),
        "execution": _execution(),
        "artifacts": ArtifactPolicy(),
        "source_digest": HEX_A,
        "workload_digest": HEX_B,
        "artifact_slot_ref": "artifact://runs/exclusive-slot",
        "quote_ref": None,
        "authorization": (
            AuthorizationRequirement(operation="training.start", paid_effect=False),
        ),
    }
    values.update(overrides)
    return TrainingPlan(**values)


def test_training_public_vocabulary_contains_no_cloud_training_compatibility() -> None:
    assert not any(name.startswith("CloudTraining") for name in training.__all__)
    assert not any(name.startswith("CloudTraining") for name in next_api_v1.__all__)
    assert not any(name.startswith("CloudTraining") for name in dir(next_api_v1))


def test_training_contract_module_imports_only_stdlib_and_execution_contracts() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= {"__future__", "dataclasses", "enum", "hashlib", "json", "math", "re", "typing"}


def test_training_request_is_frozen_and_serializes_stably_to_json() -> None:
    request = _request()
    first = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert request.to_dict()["schema_version"] == "synaptic.training/v1"
    with pytest.raises(FrozenInstanceError):
        request.kind = "job"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "full", "target_modules": ("q_proj",)},
        {"mode": "full", "dropout": 0.1},
        {"mode": "full", "init": "gaussian"},
        {"mode": "full", "use_dora": True},
    ],
)
def test_full_adapter_rejects_every_lora_only_field(kwargs) -> None:
    with pytest.raises(ValueError, match="LoRA options"):
        AdapterSpec(**kwargs)


def test_resolved_request_requires_resolved_image_and_dependency_lock() -> None:
    request = _request(resolved_execution=False)
    with pytest.raises(ValueError, match="runtime image|dependency lock|resolved execution"):
        ResolvedTrainingRequest(
            request=request,
            model=_model(),
            dataset=_dataset(),
            source_digest=HEX_A,
            resolved_at="2026-08-25T16:00:00Z",
        )


@pytest.mark.parametrize("field", ["runtime_image", "runtime_image_digest", "dependency_lock_digest"])
def test_plan_requires_every_resolved_runtime_binding(field: str) -> None:
    execution_values = _execution().__dict__ if hasattr(_execution(), "__dict__") else {
        "provider": "modal",
        "accelerator": "L4",
        "runtime_image": "registry.example/trainer@" + IMAGE_DIGEST,
        "runtime_image_digest": IMAGE_DIGEST,
        "dependency_lock_digest": HEX_B,
        "allowed_secret_refs": ("secret://hf/token",),
    }
    execution_values = dict(execution_values)
    execution_values[field] = None
    with pytest.raises(ValueError, match="runtime image|dependency lock|resolved execution"):
        _plan(execution=ExecutionSpec(**execution_values))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ResolvedTrainingRequest(
            request=_request(), model=_model(), dataset=_dataset(),
            source_digest="not-a-sha256", resolved_at="2026-08-25T16:00:00Z",
        ),
        lambda: _plan(workload_digest="not-a-sha256"),
        lambda: _plan(execution=ExecutionSpec(
            provider="modal", runtime_image="image", runtime_image_digest="sha256:short",
            dependency_lock_digest=HEX_B,
        )),
    ],
)
def test_canonical_digest_fields_require_exact_sha256(constructor) -> None:
    with pytest.raises(ValueError, match="SHA-256|sha256"):
        constructor()


@pytest.mark.parametrize("container", ["preflight", "plan"])
def test_authorization_tuples_validate_every_element(container: str) -> None:
    with pytest.raises(TypeError, match="AuthorizationRequirement"):
        if container == "preflight":
            TrainingPreflight(
                provider="modal",
                ready=True,
                checked_at="2026-08-25T16:00:00Z",
                authorization=("not-a-requirement",),  # type: ignore[arg-type]
            )
        else:
            _plan(authorization=("not-a-requirement",))


def test_plan_fingerprint_is_canonical_stable_sha256_and_domain_separated() -> None:
    first = _plan()
    second = _plan()
    assert first.fingerprint == second.fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", first.fingerprint)
    bare = hashlib.sha256(
        json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first.fingerprint != bare, "plan fingerprints must include a domain separator"


def test_submission_and_outcome_reject_invalid_nested_contract_types() -> None:
    with pytest.raises(TypeError, match="RunRef"):
        TrainingSubmission(
            run="provider-job-id",  # type: ignore[arg-type]
            plan_fingerprint=HEX_A,
            submitted_at="2026-08-25T16:00:00Z",
        )

    run = RunRef("run-001", "project://alpha")
    submission = TrainingSubmission(run, HEX_A, "2026-08-25T16:00:00Z")
    status = RunStatus(
        run, RunState.SUCCEEDED, ArtifactVerificationState.VERIFIED,
        "2026-08-25T16:01:00Z",
    )
    with pytest.raises(TypeError, match="ArtifactRef"):
        TrainingOutcome(submission, status, artifacts=("artifact",))  # type: ignore[arg-type]

    artifact = ArtifactRef("artifact-1", run, "model", ArtifactVerificationState.VERIFIED)
    outcome = TrainingOutcome(submission, status, artifacts=(artifact,))
    assert outcome.success




@pytest.mark.parametrize(
    "target, revision",
    [
        ("model", "main"),
        ("model", "latest"),
        ("model", "v1.2.3"),
        ("model", "A" * 40),
        ("tokenizer", "release-candidate"),
        ("tokenizer", "abc123"),
        ("dataset", "master"),
        ("dataset", "latest"),
    ],
)
def test_resolved_request_rejects_mutable_or_noncanonical_revisions(target, revision):
    model = _model()
    dataset = _dataset()
    if target == "model":
        model = replace(model, revision=revision)
    elif target == "tokenizer":
        model = replace(model, tokenizer_revision=revision)
    else:
        dataset = replace(dataset, revision=revision)
    with pytest.raises(ValueError, match="immutable|revision|commit"):
        ResolvedTrainingRequest(
            request=_request(), model=model, dataset=dataset,
            source_digest=HEX_A, resolved_at="2026-08-25T16:00:00Z",
        )


@pytest.mark.parametrize("target", ["model", "tokenizer", "dataset"])
def test_plan_rejects_mutable_revisions(target):
    model = _model()
    dataset = _dataset()
    if target == "model":
        model = replace(model, revision="main")
    elif target == "tokenizer":
        model = replace(model, tokenizer_revision="latest")
    else:
        dataset = replace(dataset, revision="release-v1")
    with pytest.raises(ValueError, match="immutable|revision|commit"):
        _plan(model=model, dataset=dataset)


@pytest.mark.parametrize(
    "runtime_image, declared_digest",
    [
        ("registry.example/trainer:latest", IMAGE_DIGEST),
        ("registry.example/trainer@" + IMAGE_DIGEST, "sha256:" + "d" * 64),
        ("registry.example/trainer@sha256:" + "C" * 64, "sha256:" + "C" * 64),
        ("registry.example/trainer@sha256:short", "sha256:short"),
        ("registry.example/trainer@" + IMAGE_DIGEST + ":suffix", IMAGE_DIGEST),
    ],
)
def test_plan_rejects_unpinned_or_mismatched_runtime_images(runtime_image, declared_digest):
    with pytest.raises(ValueError, match="runtime image|digest|sha256"):
        execution = ExecutionSpec(
            provider="modal", runtime_image=runtime_image,
            runtime_image_digest=declared_digest, dependency_lock_digest=HEX_B,
        )
        _plan(execution=execution)


def test_resolved_request_rejects_runtime_image_digest_mismatch():
    execution = ExecutionSpec(
        provider="modal",
        runtime_image="registry.example/trainer@" + IMAGE_DIGEST,
        runtime_image_digest="sha256:" + "d" * 64,
        dependency_lock_digest=HEX_B,
    )
    request = replace(_request(), execution=execution)
    with pytest.raises(ValueError, match="runtime image|digest|sha256"):
        ResolvedTrainingRequest(
            request=request, model=_model(), dataset=_dataset(),
            source_digest=HEX_A, resolved_at="2026-08-25T16:00:00Z",
        )
