from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tuner.cloud import hf_training_smoke_artifacts as artifacts
from tuner.cloud.hf_training_smoke_artifacts import (
    ArtifactExpectation, TrainingSmokeArtifactError, build_inventory, verify_artifact_tree,
)
from tuner.cloud.hf_training_smoke_contract import (
    RUNTIME_PYTHON_IMPLEMENTATION,
    RUNTIME_PYTHON_VERSION,
)


IDENTITY = "a" * 64
RUNTIME_PACKAGES = (("torch", "2.9.0"),)
RUNTIME_SIGNATURES = tuple(sorted({
    "TrainerCallback.on_optimizer_step": "(self, args, state, control, **kwargs)",
    "safetensors.safe_open": "(filename, framework, device='cpu')",
    "torch.load": "(f, *, weights_only=None)",
    "unsloth.import": "GPU_RUNTIME_REQUIRED",
}.items()))


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"))


def _raw_artifact(root: Path, relative: str, content: bytes) -> None:
    (root / relative).write_bytes(content)
    if relative != "inventory.json":
        _json(root / "inventory.json", build_inventory(root))


def _tree(root: Path, *, delta: float = 0.5) -> ArtifactExpectation:
    source = b'{"locked":true}\n'
    (root / "source-lock.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "source-lock.json").write_bytes(source)
    _json(root / "exclusive-sentinel.json", {
        "schema_version": "synaptic-hf-training-exclusive-sentinel/v1",
        "artifact_slot": "7" * 64,
    })
    for directory in (root / "checkpoint-1", root / "final_model"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(b"safe-adapter")
        _json(directory / "adapter_config.json", {"r": 8})
    _json(root / "checkpoint-1/trainer_state.json", {"global_step": 1})
    (root / "checkpoint-1/optimizer.pt").write_bytes(b"opaque-hashed-only")
    (root / "checkpoint-1/scheduler.pt").write_bytes(b"opaque-hashed-only")
    _json(root / "final_model/tokenizer_config.json", {"model_max_length": 2048})
    _json(root / "training_lineage.json", {
        "model_revision": "a" * 40, "dataset_sha256": "d" * 64,
        "max_steps": 1, "gradient_accumulation_steps": 1,
    })
    _json(root / "step-evidence.json", {
        "schema_version": "synaptic-protected-sft-evidence/v1", "global_step": 1,
        "optimizer_boundaries": 1, "step_one_loss": 1.0, "delta_l2": delta,
        "scheduler_last_epoch": 1, "optimizer_steps": [1],
        "pre_step_identity": "b" * 64, "post_step_identity": "c" * 64,
        "changed_tensor_count": 1,
        "serialized_adapter_identity": IDENTITY,
    })
    _json(root / "result.json", {
        "schema_version": "synaptic-hf-training-job-result/v1", "status": "COMPLETED",
        "publication": False, "model_revision": "a" * 40, "dataset_sha256": "d" * 64,
        "workload_sha256": "e" * 64, "adapter_identity": IDENTITY,
        "artifact_slot": "7" * 64, "runtime_lock_id": "f" * 64,
        "runtime": {
            "python_implementation": RUNTIME_PYTHON_IMPLEMENTATION,
            "python": RUNTIME_PYTHON_VERSION, "packages": dict(RUNTIME_PACKAGES),
            "signatures": dict(RUNTIME_SIGNATURES),
            "gpu_observed_signatures": {"FastLanguageModel.from_pretrained": "(model_name, *, revision=None)"},
            "cuda": {
                "available": True, "device_count": 1, "device_name": "NVIDIA A10G",
                "compute_capability": [8, 6], "total_memory": 24 * 1024**3,
            },
        },
    })
    _json(root / "manifest.json", {
        "schema_version": "synaptic-hf-training-manifest/v1", "status": "COMPLETED",
        "publication": False, "artifact_slot": "7" * 64, "workload_sha256": "e" * 64,
    })
    _json(root / "inventory.json", build_inventory(root))
    return ArtifactExpectation(
        source_lock_sha256=hashlib.sha256(source).hexdigest(), workload_sha256="e" * 64,
        model_revision="a" * 40, dataset_sha256="d" * 64, artifact_slot="7" * 64,
        runtime_lock_id="f" * 64,
        runtime_python_implementation=RUNTIME_PYTHON_IMPLEMENTATION,
        runtime_python=RUNTIME_PYTHON_VERSION,
        runtime_packages=RUNTIME_PACKAGES, runtime_signatures=RUNTIME_SIGNATURES,
    )


def test_verifies_exact_closed_tree_without_deserializing_optimizer(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    result = verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)
    assert set(result) == {
        "status", "inventory_sha256", "adapter_identity", "global_step", "optimizer_proof",
    }
    assert result["status"] == "VERIFIED"
    assert result["global_step"] == 1
    assert result["optimizer_proof"] == {
        "optimizer_boundaries": 1, "global_step": 1, "optimizer_step": 1,
        "scheduler_step": 1, "loss": 1.0, "max_steps": 1,
        "gradient_accumulation_steps": 1,
        "pre_adapter_sha256": "b" * 64,
        "post_adapter_sha256": IDENTITY,
        "checkpoint_adapter_sha256": IDENTITY,
        "final_adapter_sha256": IDENTITY,
        "trainable_weight_delta": 0.5,
    }


def test_optimizer_proof_comes_from_verified_snapshot(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    result = verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)
    evidence_path = tmp_path / "step-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["step_one_loss"] = 99.0
    evidence["delta_l2"] = 88.0
    evidence["pre_step_identity"] = "d" * 64
    _json(evidence_path, evidence)
    assert result["optimizer_proof"]["loss"] == 1.0
    assert result["optimizer_proof"]["trainable_weight_delta"] == 0.5
    assert result["optimizer_proof"]["pre_adapter_sha256"] == "b" * 64


def test_rejects_serialized_adapter_drift_before_projecting_optimizer_proof(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    evidence_path = tmp_path / "step-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["serialized_adapter_identity"] = "d" * 64
    _json(evidence_path, evidence)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match="Adapter identity evidence"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    ("relative", "field", "value"),
    [
        ("step-evidence.json", "global_step", True),
        ("step-evidence.json", "optimizer_boundaries", True),
        ("step-evidence.json", "step_one_loss", True),
        ("step-evidence.json", "delta_l2", True),
        ("step-evidence.json", "scheduler_last_epoch", True),
        ("step-evidence.json", "optimizer_steps", [True]),
        ("checkpoint-1/trainer_state.json", "global_step", True),
        ("training_lineage.json", "max_steps", True),
        ("training_lineage.json", "gradient_accumulation_steps", True),
    ],
)
def test_rejects_boolean_numeric_step_evidence(
    tmp_path: Path, relative: str, field: str, value: object,
) -> None:
    expectation = _tree(tmp_path)
    path = tmp_path / relative
    document = json.loads(path.read_text(encoding="ascii"))
    document[field] = value
    _json(path, document)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_unexpected_object_before_semantic_reads(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    (tmp_path / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TrainingSmokeArtifactError, match="exact closed inventory"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_zero_delta_even_with_completed_provider_artifacts(tmp_path: Path) -> None:
    expectation = _tree(tmp_path, delta=0.0)
    with pytest.raises(TrainingSmokeArtifactError, match="one finite update"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_tamper_after_inventory_capture(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _json(tmp_path / "result.json", {"status": "COMPLETED"})
    with pytest.raises(TrainingSmokeArtifactError, match="inventory"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_linked_artifact(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    target = tmp_path / "outside"
    target.write_bytes(b"x")
    link = tmp_path / "result.json"
    link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(TrainingSmokeArtifactError, match="link"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_name", "NVIDIA A10G-SPOOF"),
        ("compute_capability", [8, 9]),
        ("total_memory", 23_068_671_999),
        ("device_count", True),
    ],
)
def test_rejects_hostile_cuda_evidence(tmp_path: Path, field: str, value: object) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    result["runtime"]["cuda"][field] = value
    _json(result_path, result)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match="approved A10G CUDA"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    "mutation",
    [lambda cuda: cuda.update({"extra": "hostile"}), lambda cuda: cuda.pop("total_memory")],
)
def test_rejects_open_or_incomplete_cuda_evidence(tmp_path: Path, mutation) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    mutation(result["runtime"]["cuda"])
    _json(result_path, result)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match="approved A10G CUDA"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize("total_memory", [23_068_672_000, 25_769_803_776])
def test_accepts_inclusive_a10g_memory_bounds(tmp_path: Path, total_memory: int) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    result["runtime"]["cuda"]["total_memory"] = total_memory
    _json(result_path, result)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    assert verify_artifact_tree(
        tmp_path, expectation, tensor_identity=lambda path: IDENTITY
    )["status"] == "VERIFIED"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runtime: runtime["signatures"].update({"unsloth.import": "captured-on-cpu"}),
        lambda runtime: runtime["gpu_observed_signatures"].update({"extra": "()"}),
        lambda runtime: runtime["gpu_observed_signatures"].clear(),
        lambda runtime: runtime["gpu_observed_signatures"].update(
            {"FastLanguageModel.from_pretrained": "bad\nline"}
        ),
    ],
)
def test_rejects_hostile_runtime_signature_evidence(tmp_path: Path, mutation) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    mutation(result["runtime"])
    _json(result_path, result)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match="runtime signatures|GPU signature"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    [
        ("result.json", lambda value: value.update({"extra": True}), "result semantics"),
        ("result.json", lambda value: value.pop("artifact_slot"), "result semantics"),
        ("manifest.json", lambda value: value.update({"extra": True}), "manifest semantics"),
        ("manifest.json", lambda value: value.pop("artifact_slot"), "manifest semantics"),
        ("exclusive-sentinel.json", lambda value: value.update({"extra": True}), "sentinel semantics"),
        ("exclusive-sentinel.json", lambda value: value.pop("schema_version"), "sentinel semantics"),
    ],
)
def test_rejects_open_or_incomplete_top_level_evidence(
    tmp_path: Path, filename: str, mutation, message: str
) -> None:
    expectation = _tree(tmp_path)
    path = tmp_path / filename
    value = json.loads(path.read_text(encoding="ascii"))
    mutation(value)
    _json(path, value)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match=message):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runtime: runtime.update({"extra": True}), "runtime evidence is not closed"),
        (lambda runtime: runtime.pop("python_implementation"), "runtime evidence is not closed"),
        (lambda runtime: runtime.pop("python"), "runtime evidence is not closed"),
        (
            lambda runtime: runtime.update({"python_implementation": "PyPy"}),
            "implementation drifted",
        ),
        (
            lambda runtime: runtime.update({"python_implementation": True}),
            "implementation drifted",
        ),
        (lambda runtime: runtime.update({"python": "3.12.7"}), "Python drifted"),
        (lambda runtime: runtime.update({"python": 31114}), "Python drifted"),
        (lambda runtime: runtime["packages"].update({"extra": "1"}), "packages drifted"),
        (lambda runtime: runtime["packages"].pop("torch"), "packages drifted"),
        (lambda runtime: runtime["packages"].update({"torch": "2.9.1"}), "packages drifted"),
        (lambda runtime: runtime["signatures"].update({"extra": "()"}), "signatures drifted"),
        (lambda runtime: runtime["signatures"].pop("torch.load"), "signatures drifted"),
        (lambda runtime: runtime["signatures"].update({"torch.load": "(drifted)"}), "signatures drifted"),
    ],
)
def test_rejects_open_missing_or_drifted_authenticated_runtime_evidence(
    tmp_path: Path, mutation, message: str
) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    mutation(result["runtime"])
    _json(result_path, result)
    _json(tmp_path / "inventory.json", build_inventory(tmp_path))
    with pytest.raises(TrainingSmokeArtifactError, match=message):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_python_implementation", "PyPy", "Python implementation"),
        ("runtime_python_implementation", True, "Python implementation"),
        ("runtime_python", "3.12.7", "runtime Python"),
        ("runtime_python", 31114, "runtime Python"),
    ],
)
def test_expectation_requires_exact_child_python_identity(
    tmp_path: Path, field: str, value: object, message: str,
) -> None:
    expectation = _tree(tmp_path)
    values = dict(vars(expectation))
    values[field] = value
    with pytest.raises(TrainingSmokeArtifactError, match=message):
        ArtifactExpectation(**values)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (b'"status":"COMPLETED"', b'"status":"COMPLETED","status":"COMPLETED"'),
        (b'"runtime":{', b'"runtime":{"nested":0,"nested":0,'),
        (
            b'"unsloth.import":"GPU_RUNTIME_REQUIRED"',
            b'"unsloth.import":"GPU_RUNTIME_REQUIRED","unsloth.import":"GPU_RUNTIME_REQUIRED"',
        ),
        (b'"device_count":1', b'"device_count":1,"device_count":1'),
        (
            b'"artifact_slot":"' + b"7" * 64 + b'"',
            b'"artifact_slot":"' + b"7" * 64 + b'","artifact_slot":"' + b"7" * 64 + b'"',
        ),
        (
            b'"adapter_identity":"' + b"a" * 64 + b'"',
            b'"adapter_identity":"' + b"a" * 64 + b'","adapter_identity":"' + b"a" * 64 + b'"',
        ),
    ],
)
def test_rejects_raw_duplicate_keys_at_all_runtime_levels(
    tmp_path: Path, needle: bytes, replacement: bytes
) -> None:
    expectation = _tree(tmp_path)
    result_path = tmp_path / "result.json"
    raw = result_path.read_bytes()
    assert raw.count(needle) == 1
    _raw_artifact(tmp_path, "result.json", raw.replace(needle, replacement))
    with pytest.raises(TrainingSmokeArtifactError, match="duplicate object key"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_nested_duplicate_in_supporting_json_before_semantics(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _raw_artifact(tmp_path, "final_model/adapter_config.json", b'{"nested":{"r":8,"r":8}}\n')
    with pytest.raises(TrainingSmokeArtifactError, match="duplicate object key"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity", b"1e999"])
def test_rejects_raw_nonfinite_numbers(tmp_path: Path, literal: bytes) -> None:
    expectation = _tree(tmp_path)
    raw = (tmp_path / "step-evidence.json").read_bytes()
    assert raw.count(b'"step_one_loss":1.0') == 1
    _raw_artifact(
        tmp_path, "step-evidence.json",
        raw.replace(b'"step_one_loss":1.0', b'"step_one_loss":' + literal),
    )
    with pytest.raises(TrainingSmokeArtifactError, match="non-finite"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_json_beyond_closed_nesting_bound(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _raw_artifact(
        tmp_path, "final_model/tokenizer_config.json",
        b"[" * 65 + b"0" + b"]" * 65,
    )
    with pytest.raises(TrainingSmokeArtifactError, match="nesting bound"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_extreme_json_decoder_depth(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _raw_artifact(
        tmp_path, "checkpoint-1/adapter_config.json",
        b"[" * 2000 + b"0" + b"]" * 2000,
    )
    with pytest.raises(TrainingSmokeArtifactError, match="JSON is invalid|nesting bound"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_normalizes_decoder_recursion_error(monkeypatch, tmp_path: Path) -> None:
    expectation = _tree(tmp_path)

    def recurse(*args, **kwargs):
        raise RecursionError("private decoder detail")

    monkeypatch.setattr(artifacts.json, "loads", recurse)
    try:
        with pytest.raises(TrainingSmokeArtifactError, match="JSON is invalid"):
            verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)
    finally:
        monkeypatch.undo()


def test_rejects_oversized_numeric_literal(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _raw_artifact(
        tmp_path, "checkpoint-1/adapter_config.json",
        b'{"r":' + b"9" * 129 + b"}\n",
    )
    with pytest.raises(TrainingSmokeArtifactError, match="oversized numeric literal"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)


def test_rejects_invalid_utf8_in_any_json_artifact(tmp_path: Path) -> None:
    expectation = _tree(tmp_path)
    _raw_artifact(tmp_path, "final_model/adapter_config.json", b'{"r":"\xff"}\n')
    with pytest.raises(TrainingSmokeArtifactError, match="JSON is invalid"):
        verify_artifact_tree(tmp_path, expectation, tensor_identity=lambda path: IDENTITY)
