from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import struct
import tarfile
import tempfile

import pytest

from tests.runtime.test_native_mlx_child import inputs
from tuner.runtime.native_files import NativeTree
from tuner.runtime.native_mlx_artifacts import adapter_tensors, verify_adapter_delta
from tuner.runtime.native_mlx_input import NativeModelSnapshot
from tuner.runtime.native_mlx_material import canonical, strict_json
from tuner.runtime.native_mlx_publication import NativeArtifactBudget, verify_native_publication
from tuner.training.native_mlx_compilation import compile_native_sft


def tensor_archive(value, *, layers=range(2), targets=("q_proj", "v_proj")):
    headers, payload = {}, bytearray()
    for layer in layers:
        for target in targets:
            for suffix in ("lora_a", "lora_b"):
                name = f"model.layers.{layer}.self_attn.{target}.{suffix}"
                start = len(payload)
                payload.extend(struct.pack("<4f", value, 0, 0, 0))
                headers[name] = {"dtype": "F32", "shape": [1, 4],
                                 "data_offsets": [start, len(payload)]}
    header = canonical(headers)
    return struct.pack("<Q", len(header)) + header + payload


@pytest.fixture
def publication():
    mask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="native-publication-test-", dir=Path.home()) as path:
            root = Path(path)
            output, token_root, model_root = root / "output", root / "tokenizer", root / "model"
            output.mkdir(mode=0o700)
            token_root.mkdir(mode=0o700)
            model_root.mkdir(mode=0o700)
            (model_root / "config.json").write_bytes(canonical({"num_hidden_layers": 2}))
            (token_root / "tokenizer.json").write_bytes(b'{"fixture":true}')
            (token_root / "tokenizer_config.json").write_bytes(b'{"fixture":true}')
            with NativeTree(token_root) as tree:
                tokenizer_snapshot = NativeModelSnapshot("configured/model", "c" * 40, str(token_root), tree.inventory())
            with NativeTree(model_root) as tree:
                model_snapshot = NativeModelSnapshot("configured/model", "c" * 40, str(model_root), tree.inventory())
            projection = compile_native_sft(*inputs())
            initial, final = tensor_archive(0), tensor_archive(.25)
            shapes = tuple((t.name, t.shape) for t in adapter_tensors(initial))
            delta = verify_adapter_delta(initial, final, expected_shapes=shapes)
            binding = {
                "schema": "synaptic-native-tokenization-binding/v1",
                "prepared_digest": "e" * 64, "source": projection.source.to_dict(),
                "tokenizer_snapshot_digest": tokenizer_snapshot.digest,
                "tokenization": projection.tokenization.to_dict(),
                "workload_digest": projection.workload.fingerprint,
                "runtime_digest": projection.profile.runtime_digest,
            }
            result = {
                "schema": "synaptic-native-mlx-sft-result/v1", "status": "CANDIDATE_WORKER_COMPLETE",
                "input_binding_digest": hashlib.sha256(canonical(binding)).hexdigest(),
                "workload_digest": projection.workload.fingerprint,
                "runtime_digest": projection.profile.runtime_digest,
                "model_snapshot_digest": model_snapshot.digest,
                "lora": asdict(projection.lora), "optimizer": asdict(projection.optimizer),
                "completed_updates": 3, "adapter_format": "mlx-lm-lora-safetensors",
                "parameter_delta": delta,
                "final_validation": {
                    "adapter_sha256": hashlib.sha256(final).hexdigest(),
                    "phase": "reloaded_saved_final_adapter", "loss": 1.25,
                    "rows": 2, "supervised_tokens": 10,
                },
            }
            config = {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": {
                "rank": 4, "scale": 8.0, "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            }}
            # Preserve exact numeric projection (canonical distinguishes 0/0.0).
            config["lora_parameters"]["dropout"] = projection.lora.dropout
            files = {
                "initial-adapters.safetensors": initial, "adapters.safetensors": final,
                "0000003_adapters.safetensors": final, "input_binding.json": canonical(binding),
                "native_result.json": canonical(result), "adapter_config.json": canonical(config),
                "workload.json": projection.workload.canonical_bytes,
            }
            for name, raw in files.items():
                (output / name).write_bytes(raw)
            yield dict(output=output, workload=projection.workload.canonical_bytes, model=model_snapshot,
                       tokenizer=tokenizer_snapshot,
                       tokenizer_members=("tokenizer.json", "tokenizer_config.json"),
                       budget=NativeArtifactBudget(1024 * 1024, 4 * 1024 * 1024))
    finally:
        os.umask(mask)


def test_native_roles_have_verified_bytes_and_deterministic_portable_archives(publication):
    artifacts = verify_native_publication(**publication)
    assert artifacts == verify_native_publication(**publication)
    assert tuple(item.descriptor.role for item in artifacts) == (
        "final_model", "tokenizer", "training_lineage", "training_metrics", "workload_record",
    )
    for item in artifacts:
        assert item.descriptor.sha256 == hashlib.sha256(item.content).hexdigest()
        assert item.descriptor.size_bytes == len(item.content)
    model = artifacts[0]
    with tarfile.open(fileobj=io.BytesIO(model.content)) as archive:
        assert archive.getnames() == [
            "0000003_adapters.safetensors", "adapter_config.json",
            "adapters.safetensors", "native_model.json",
        ]
        manifest = strict_json(archive.extractfile("native_model.json").read())
        assert manifest["format"] == "mlx-lm-lora-safetensors"
        assert manifest["revision"] == "c" * 40
        assert all(member.isfile() and member.mtime == 0 for member in archive)


@pytest.mark.parametrize("field,value", [
    ("completed_updates", 2), ("completed_updates", True),
    ("model_snapshot_digest", "0" * 64), ("input_binding_digest", "0" * 64),
    ("runtime_digest", "0" * 64), ("adapter_format", "peft"),
    ("final_validation", {"phase": "upstream-final-in-loop"}),
])
def test_result_claims_cannot_override_saved_output_contract(publication, field, value):
    path = publication["output"] / "native_result.json"
    result = strict_json(path.read_bytes())
    result[field] = value
    path.write_bytes(canonical(result))
    with pytest.raises(ValueError):
        verify_native_publication(**publication)


def test_changed_tensor_payload_and_extra_file_reject(publication):
    path = publication["output"] / "adapters.safetensors"
    path.write_bytes(tensor_archive(.5))
    with pytest.raises(ValueError, match="delta"):
        verify_native_publication(**publication)
    path.write_bytes(tensor_archive(.25))
    (publication["output"] / "unclaimed.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_native_publication(**publication)


def test_tokenizer_bytes_and_per_role_budget_are_enforced(publication):
    with pytest.raises(ValueError, match="budget"):
        verify_native_publication(**{**publication, "budget": NativeArtifactBudget(4096, 32768)})
    tokenizer = publication["tokenizer"]
    (Path(tokenizer.root) / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot"):
        verify_native_publication(**publication)


def test_tokenizer_selection_must_cover_complete_declared_snapshot(publication):
    with pytest.raises(ValueError, match="projection inputs"):
        verify_native_publication(**{**publication, "tokenizer_members": ("tokenizer.json",)})


@pytest.mark.parametrize("layers,targets", [
    (range(4), ("q_proj",)),
    (range(1, 3), ("q_proj", "v_proj")),
])
def test_adapter_target_and_last_layer_set_cannot_be_substituted(publication, layers, targets):
    before = tensor_archive(0, layers=layers, targets=targets)
    after = tensor_archive(.25, layers=layers, targets=targets)
    output = publication["output"]
    (output / "initial-adapters.safetensors").write_bytes(before)
    (output / "adapters.safetensors").write_bytes(after)
    (output / "0000003_adapters.safetensors").write_bytes(after)
    result_path = output / "native_result.json"
    result = strict_json(result_path.read_bytes())
    result["parameter_delta"] = verify_adapter_delta(
        before, after, expected_shapes=tuple((t.name, t.shape) for t in adapter_tensors(before)),
    )
    result["final_validation"]["adapter_sha256"] = hashlib.sha256(after).hexdigest()
    result_path.write_bytes(canonical(result))
    with pytest.raises(ValueError, match="target layout"):
        verify_native_publication(**publication)


def test_bound_tokenization_source_cannot_be_replaced_with_same_result(publication):
    path = publication["output"] / "input_binding.json"
    binding = strict_json(path.read_bytes())
    binding["source"]["content_digest"] = "f" * 64
    path.write_bytes(canonical(binding))
    result_path = publication["output"] / "native_result.json"
    result = strict_json(result_path.read_bytes())
    result["input_binding_digest"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_path.write_bytes(canonical(result))
    with pytest.raises(ValueError, match="input binding"):
        verify_native_publication(**publication)


def test_retained_checkpoint_must_also_contain_finite_weights(publication):
    (publication["output"] / "0000003_adapters.safetensors").write_bytes(tensor_archive(float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        verify_native_publication(**publication)
