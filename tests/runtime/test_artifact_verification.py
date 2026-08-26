from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace

import pytest

from synaptic_tuner.api.v1.training import CanonicalDocument
from tuner.runtime.artifacts import ArtifactEntry, ArtifactInventory, verify_inventory
from tuner.runtime.dispatch import ProcessResult
from tuner.runtime.verification import (
    MAX_SEMANTIC_ARTIFACT_BYTES,
    VerificationService,
    VerificationStatus,
    WorkloadBindingVerifier,
)
from tuner.training.methods.sft import SFT_ARTIFACT_CONTRACT, compile_sft_workload
from tests.training.test_sft_compilation import _execution_source


def _entry(role: str, path: str, content: bytes) -> ArtifactEntry:
    return ArtifactEntry(role, path, hashlib.sha256(content).hexdigest(), len(content))


def _tar(values: dict[str, bytes]) -> bytes:
    return _tar_pairs(tuple(values.items()))


def _tar_pairs(values: tuple[tuple[str, bytes], ...]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, content in values:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _safetensors(
    payload: bytes = b"\x01\x00\x00\x00",
    *,
    offsets: tuple[int, int] = (0, 4),
    name: str = "weight",
) -> bytes:
    header = json.dumps(
        {name: {"dtype": "F32", "shape": [1], "data_offsets": list(offsets)}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return len(header).to_bytes(8, "little") + header + payload


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _runtime_workload():
    source = _execution_source()
    roots = {
        "engine": "/workspace/engine", "project": "/workspace/project",
        **{name: f"/workspace/run/{name}" for name in ("artifacts", "state", "tracking", "cache", "tmp")},
    }
    environment = dict(source.environment)
    replacements = {
        "PYTHONPATH": roots["engine"], "SYNAPTIC_ENGINE_ROOT": roots["engine"],
        "SYNAPTIC_PROJECT_ROOT": roots["project"], "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"],
        "SYNAPTIC_STATE_ROOT": roots["state"], "SYNAPTIC_TRACKING_ROOT": roots["tracking"],
        "SYNAPTIC_CACHE_ROOT": roots["cache"], "SYNAPTIC_TMP_ROOT": roots["tmp"],
        "HF_HOME": roots["cache"] + "/huggingface",
        "TRANSFORMERS_CACHE": roots["cache"] + "/transformers",
    }
    environment.update(replacements)
    source = replace(
        source, roots=roots, writable_capability_root="/workspace/run",
        environment=environment,
        python_executable="/usr/bin/python", python_version="3.12.3",
    )
    config = CanonicalDocument.from_mapping(
        {
            "schema_version": "synaptic-sft-config/v1", "method": "sft",
            "model": {"ref": "example/model", "revision": "c" * 40, "tokenizer_revision": "c" * 40, "load_in_4bit": False},
            "dataset": {"ref": "project://data/train.jsonl", "revision": "9" * 40, "content_digest": "d" * 64, "format": "configured/v1"},
            "sft": {
                "max_steps": 1, "batch_size": 2, "gradient_accumulation_steps": 4,
                "learning_rate": "0.0002", "max_seq_length": 1024, "seed": 7,
                "save_steps": 1, "save_total_limit": 1, "lora_rank": 8,
                "lora_alpha": 16, "lora_dropout": "0.0",
                "lora_target_modules": ["q_proj", "v_proj"], "use_dora": False,
                "use_rslora": False, "init_lora_weights": True, "split_dataset": False,
            },
        }
    )
    return compile_sft_workload(resolved_config=config, execution_source=source)


def _lineage(workload) -> bytes:
    config = workload.document["configuration"]["document"]
    roots = workload.document["execution_source"]["runtime"]["roots"]
    root = roots["engine"]
    project_root = roots["project"]
    run_dir = roots["state"] + "/runtime-v1-trainer/output/runtime-v1"
    dataset_path = f"{project_root}/data/train.jsonl"
    argv = [
        "/usr/bin/python", f"{root}/Trainers/sft/train_sft.py", "--model-name", "example/model",
        "--model-revision", "c" * 40, "--anonymous-model", "--model-cache-dir", roots["cache"] + "/model",
        "--local-file", dataset_path, "--output-root", roots["state"] + "/runtime-v1-trainer/output",
        "--run-timestamp", "runtime-v1", "--no-dashboard", "--quiet",
        "--runtime-v1-workload-fingerprint", workload.fingerprint,
        "--runtime-v1-configuration-revision", workload.document["configuration"]["revision"],
        "--runtime-v1-tokenizer-revision", "c" * 40,
        "--runtime-v1-dataset-revision", "9" * 40,
        "--runtime-v1-dataset-digest", "d" * 64,
        "--batch-size", "2",
        "--gradient-accumulation", "4", "--learning-rate", "0.0002", "--max-steps", "1",
        "--max-seq-length", "1024", "--seed", "7", "--save-steps", "1",
        "--save-total-limit", "1", "--lora-r", "8", "--lora-alpha", "16",
        "--lora-dropout", "0.0", "--lora-target-modules", "q_proj,v_proj",
        "--init-lora-weights", "true", "--no-load-in-4bit",
    ]
    environment = {
        "PATH": "/usr/local/bin",
        "SYNAPTIC_ENGINE_ROOT": root, "SYNAPTIC_PROJECT_ROOT": project_root,
        "SYNAPTIC_ARTIFACT_ROOT": roots["artifacts"], "SYNAPTIC_STATE_ROOT": roots["state"],
        "SYNAPTIC_TRACKING_ROOT": roots["tracking"], "SYNAPTIC_CACHE_ROOT": roots["cache"],
        "SYNAPTIC_TMP_ROOT": roots["tmp"], "SYNAPTIC_WORKLOAD_FINGERPRINT": workload.fingerprint,
        "PYTHONPATH": root + ":" + root + "/Trainers/sft",
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "HF_HOME": roots["cache"] + "/huggingface", "TRANSFORMERS_CACHE": roots["cache"] + "/transformers",
        "WANDB_DISABLED": "true",
    }
    execution = {
        "schema_version": "synaptic-sft-execution-evidence/v1",
        "workload_fingerprint": workload.fingerprint,
        "configuration_revision": workload.document["configuration"]["revision"],
        "model": {key: config["model"][key] for key in ("ref", "revision", "tokenizer_revision", "load_in_4bit")},
        "dataset": {"ref": config["dataset"]["ref"], "resolved_path": dataset_path, "revision": config["dataset"]["revision"], "content_digest": config["dataset"]["content_digest"]},
        "sft": config["sft"], "argv": argv, "environment": environment, "cwd": roots["tmp"],
        "outputs": {"run_dir": run_dir, "final_model_dir": f"{run_dir}/final_model", "tokenizer_dir": f"{run_dir}/final_model", "lineage_path": f"{run_dir}/training_lineage.json"},
        "result": {"exit_code": 0, "status": "completed"},
    }
    projection = {
        "schema_version": "synaptic-sft-trainer-projection/v1",
        "workload_fingerprint": workload.fingerprint,
        "configuration_revision": workload.document["configuration"]["revision"],
        "model": {key: config["model"][key] for key in ("ref", "revision", "tokenizer_revision", "load_in_4bit")},
        "dataset": {"resolved_path": dataset_path, "revision": config["dataset"]["revision"], "content_digest": config["dataset"]["content_digest"]},
        "training": {"batch_size": 2, "gradient_accumulation_steps": 4, "learning_rate": 0.0002, "max_steps": 1, "num_epochs": 1, "max_seq_length": 1024, "seed": 7, "save_steps": 1, "save_total_limit": 1, "split_dataset": False},
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"], "use_dora": False, "use_rslora": False, "init_lora_weights": True},
        "outputs": {"run_dir": run_dir, "final_model_dir": f"{run_dir}/final_model"},
        "status": "completed",
    }
    trainer = {
        "training_type": "SFT", "run_directory": run_dir,
        "model": {"base_model": "example/model", "load_in_4bit": False},
        "dataset": {"source": dataset_path}, "runtime": {"status": "completed"},
        "training": {"batch_size": 2, "gradient_accumulation_steps": 4, "learning_rate": 0.0002, "max_steps": 1, "max_seq_length": 1024, "seed": 7},
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "synaptic_runtime_projection": projection,
    }
    wrapper = {
        "schema_version": "synaptic-sft-training-lineage/v1", "workload_fingerprint": workload.fingerprint,
        "execution_source": workload.document["execution_source"], "configuration_revision": workload.document["configuration"]["revision"],
        "identities": workload.document["identities"], "trainer_exit_code": 0,
        "execution_evidence": execution, "execution_evidence_sha256": hashlib.sha256(_canonical(execution)).hexdigest(),
        "trainer_lineage": trainer,
    }
    return _canonical(wrapper)


def _fixture():
    workload = _runtime_workload()
    values = {
        "workload.json": workload.canonical_bytes,
        "training_lineage.json": _lineage(workload),
        "training_metrics.json": b"{}",
        "final_model.tar": _tar(
            {
                "adapter_config.json": b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
                "adapter_model.safetensors": _safetensors(),
            }
        ),
        "tokenizer.tar": _tar(
            {
                "tokenizer_config.json": b'{"tokenizer_class":"Fixture"}',
                "tokenizer.json": b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1.0"}',
            }
        ),
    }
    roles = (
        "workload_record",
        "training_lineage",
        "training_metrics",
        "final_model",
        "tokenizer",
    )
    inventory = ArtifactInventory(
        tuple(_entry(role, path, values[path]) for role, path in zip(roles, values))
    )

    class Reader:
        def __init__(self):
            self.calls = []

        def read_bytes(self, artifact, *, maximum):
            self.calls.append((artifact.path, maximum))
            value = values[artifact.path]
            assert len(value) <= maximum
            return value

    return workload, inventory, Reader(), values


def _verify_current(workload, inventory, reader, values):
    entries = tuple(
        _entry(item.role, item.path, values[item.path]) for item in inventory.entries
    )
    return VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory(entries),
        reader=reader,
    )


def test_inventory_rejects_missing_and_unexpected_roles() -> None:
    _, inventory, _, _ = _fixture()
    entries = inventory.entries[1:] + (
        ArtifactEntry("debug_dump", "debug.txt", "0" * 64, 0),
    )
    result = verify_inventory(SFT_ARTIFACT_CONTRACT, ArtifactInventory(entries))
    assert not result.valid
    assert any("workload_record" in error for error in result.errors)
    assert any("unexpected" in error for error in result.errors)


def test_provider_completion_is_not_success_without_semantic_verification() -> None:
    workload, inventory, reader, _ = _fixture()
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=inventory,
        reader=reader,
    )
    assert report.status is VerificationStatus.VERIFIED
    assert report.success
    assert [path for path, _ in reader.calls] == [
        item.path for item in inventory.entries
    ]
    assert len(reader.calls) == len(inventory.entries)
    assert {check.code for check in report.semantic_checks} >= {
        "final_model_semantic",
        "tokenizer_semantic",
        "model_tokenizer_disjoint",
    }


@pytest.mark.parametrize("attack", ("mixed-family", "base-drift", "unsupported-weight"))
def test_provider_rejects_nonexclusive_or_unbound_model_archives(attack: str) -> None:
    workload, inventory, reader, values = _fixture()
    members = {
        "adapter_config.json": b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
        "adapter_model.safetensors": _safetensors(),
    }
    if attack == "mixed-family":
        members["config.json"] = b'{"model_type":"fixture"}'
        members["model.safetensors"] = _safetensors()
    elif attack == "base-drift":
        members["adapter_config.json"] = b'{"base_model_name_or_path":"attacker/model","peft_type":"LORA"}'
    else:
        members["evil.bin"] = b"unsupported"
    values["final_model.tar"] = _tar(members)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID
    assert {item.code: item.passed for item in report.semantic_checks}["final_model_semantic"] is False


def test_provider_accepts_complete_sharded_peft_archive_with_exact_index() -> None:
    workload, inventory, reader, values = _fixture()
    values["final_model.tar"] = _tar(
        {
            "adapter_config.json": b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
            "adapter_model-00001-of-00002.safetensors": _safetensors(name="weight_a"),
            "adapter_model-00002-of-00002.safetensors": _safetensors(payload=b"\x02\x00\x00\x00", name="weight_b"),
            "adapter_model.safetensors.index.json": b'{"metadata":{},"weight_map":{"weight_a":"adapter_model-00001-of-00002.safetensors","weight_b":"adapter_model-00002-of-00002.safetensors"}}',
        }
    )
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.VERIFIED


def test_provider_accepts_pretty_printed_qwen2_fast_tokenizer_sidecars() -> None:
    workload, inventory, reader, values = _fixture()
    values["tokenizer.tar"] = _tar(
        {
            "tokenizer_config.json": json.dumps({"tokenizer_class": "Qwen2TokenizerFast"}, indent=2).encode(),
            "tokenizer.json": json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {"hello": 0}}}, indent=2).encode(),
            "vocab.json": json.dumps({"hello": 0}, indent=2).encode(),
            "merges.txt": b"#version: 0.2\nh e\n",
            "added_tokens.json": json.dumps({"<|im_start|>": 1}, indent=2).encode(),
            "special_tokens_map.json": json.dumps({"eos_token": "<|im_end|>"}, indent=2).encode(),
            "chat_template.jinja": b"{% for message in messages %}{{ message.content }}{% endfor %}",
        }
    )
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.VERIFIED


@pytest.mark.parametrize("argv", ([], ["/usr/bin/python", 7], ["/usr/bin/python"]))
def test_provider_maps_malformed_argv_to_invalid(argv: list[object]) -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["execution_evidence"]["argv"] = argv
    wrapper["execution_evidence_sha256"] = hashlib.sha256(_canonical(wrapper["execution_evidence"])).hexdigest()
    values["training_lineage.json"] = _canonical(wrapper)
    assert _verify_current(workload, inventory, reader, values).status is VerificationStatus.INVALID


def test_provider_projection_comparison_rejects_boolean_numeric_alias() -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["trainer_lineage"]["synaptic_runtime_projection"]["model"]["load_in_4bit"] = 0
    values["training_lineage.json"] = _canonical(wrapper)
    assert _verify_current(workload, inventory, reader, values).status is VerificationStatus.INVALID


@pytest.mark.parametrize("attack", ("mixed-unsharded", "orphan-index", "opposite-index", "metadata-bool", "metadata-extra"))
def test_provider_rejects_inexact_shard_and_index_contracts(attack: str) -> None:
    workload, inventory, reader, values = _fixture()
    members = {
        "adapter_config.json": b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
        "adapter_model-00001-of-00001.safetensors": _safetensors(),
        "adapter_model.safetensors.index.json": b'{"metadata":{},"weight_map":{"weight":"adapter_model-00001-of-00001.safetensors"}}',
    }
    if attack == "mixed-unsharded":
        members["adapter_model.safetensors"] = _safetensors()
    elif attack == "orphan-index":
        del members["adapter_model-00001-of-00001.safetensors"]
        members["adapter_model.safetensors"] = _safetensors()
    elif attack == "opposite-index":
        members["model.safetensors.index.json"] = b'{"metadata":{},"weight_map":{}}'
    elif attack == "metadata-bool":
        members["adapter_model.safetensors.index.json"] = b'{"metadata":{"total_size":true},"weight_map":{"weight":"adapter_model-00001-of-00001.safetensors"}}'
    else:
        members["adapter_model.safetensors.index.json"] = b'{"metadata":{"unexpected":1},"weight_map":{"weight":"adapter_model-00001-of-00001.safetensors"}}'
    values["final_model.tar"] = _tar(members)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "revision", "0" * 40),
        ("model", "tokenizer_revision", "0" * 40),
        ("dataset", "revision", "0" * 40),
        ("dataset", "content_digest", "0" * 64),
        ("dataset", "resolved_path", "/workspace/engine/data/other.jsonl"),
        ("sft", "learning_rate", "0.9"),
        ("outputs", "final_model_dir", "/workspace/state/other"),
        ("result", "status", "failed"),
    ),
)
def test_provider_rejects_execution_evidence_contradictions(
    section: str, field: str, value: object
) -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["execution_evidence"][section][field] = value
    wrapper["execution_evidence_sha256"] = hashlib.sha256(
        _canonical(wrapper["execution_evidence"])
    ).hexdigest()
    values["training_lineage.json"] = _canonical(wrapper)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID
    assert {item.code: item.passed for item in report.semantic_checks}[
        "lineage_binds_workload"
    ] is False


@pytest.mark.parametrize("field", ("argv", "environment"))
def test_provider_rejects_invocation_or_environment_drift(field: str) -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    evidence = wrapper["execution_evidence"]
    if field == "argv":
        evidence["argv"][3] = "attacker/model"
    else:
        evidence["environment"]["PYTHONPATH"] = "/attacker"
    wrapper["execution_evidence_sha256"] = hashlib.sha256(_canonical(evidence)).hexdigest()
    values["training_lineage.json"] = _canonical(wrapper)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


def test_provider_maps_malformed_complete_evidence_to_invalid() -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["execution_evidence"]["outputs"] = "not-an-object"
    wrapper["execution_evidence_sha256"] = hashlib.sha256(
        _canonical(wrapper["execution_evidence"])
    ).hexdigest()
    values["training_lineage.json"] = _canonical(wrapper)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "base_model", "attacker/model"),
        ("model", "load_in_4bit", 0),
        ("dataset", "source", "/other.jsonl"),
        ("training", "batch_size", 99),
        ("training", "batch_size", 2.0),
        ("training", "max_steps", True),
        ("lora", "rank", 99),
        ("lora", "alpha", 16.0),
        ("runtime", "status", "failed"),
    ),
)
def test_provider_rejects_contradictory_trainer_lineage(
    section: str, field: str, value: object
) -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["trainer_lineage"][section][field] = value
    values["training_lineage.json"] = _canonical(wrapper)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


def test_provider_rejects_boolean_wrapper_exit_code() -> None:
    workload, inventory, reader, values = _fixture()
    wrapper = json.loads(values["training_lineage.json"])
    wrapper["trainer_exit_code"] = False
    values["training_lineage.json"] = _canonical(wrapper)
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


@pytest.mark.parametrize("attack", ("noncanonical", "duplicate", "nonfinite"))
def test_provider_rejects_non_strict_lineage_wrapper(attack: str) -> None:
    workload, inventory, reader, values = _fixture()
    raw = values["training_lineage.json"]
    if attack == "noncanonical":
        raw = json.dumps(json.loads(raw), indent=2).encode()
    elif attack == "duplicate":
        raw = b'{"schema_version":"duplicate",' + raw[1:]
    else:
        raw = raw.replace(b'"trainer_exit_code":0', b'"trainer_exit_code":NaN')
    values["training_lineage.json"] = raw
    report = _verify_current(workload, inventory, reader, values)
    assert report.status is VerificationStatus.INVALID


def test_identical_placeholder_archives_are_semantically_invalid() -> None:
    workload, inventory, reader, values = _fixture()
    placeholder = _tar({"inventory.json": b"{}"})
    values["final_model.tar"] = placeholder
    values["tokenizer.tar"] = placeholder
    entries = tuple(
        _entry(item.role, item.path, values[item.path]) for item in inventory.entries
    )
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory(entries),
        reader=reader,
    )
    checks = {item.code: item.passed for item in report.semantic_checks}
    assert report.status is VerificationStatus.INVALID
    assert checks["final_model_semantic"] is False
    assert checks["tokenizer_semantic"] is False
    assert checks["model_tokenizer_disjoint"] is False


def test_archive_semantics_reject_link_members_without_extracting() -> None:
    workload, inventory, reader, values = _fixture()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        config = tarfile.TarInfo("adapter_config.json")
        config.size = 2
        archive.addfile(config, io.BytesIO(b"{}"))
        link = tarfile.TarInfo("adapter_model.safetensors")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)
    values["final_model.tar"] = output.getvalue()
    entries = tuple(
        _entry(item.role, item.path, values[item.path]) for item in inventory.entries
    )
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory(entries),
        reader=reader,
    )
    checks = {item.code: item.passed for item in report.semantic_checks}
    assert report.status is VerificationStatus.INVALID
    assert checks["final_model_semantic"] is False


@pytest.mark.parametrize(
    "attack",
    (
        "model-one-byte",
        "model-zeros",
        "model-zero-tensor",
        "model-offsets",
        "model-length",
        "model-fake-config",
        "model-duplicate-config",
        "model-nonfinite-config",
        "tokenizer-one-byte",
        "tokenizer-zeros",
        "tokenizer-fake-config",
        "tokenizer-duplicate-config",
        "tokenizer-nonfinite-config",
        "tokenizer-empty-vocab",
    ),
)
def test_provider_rejects_fake_model_and_tokenizer_contents(attack: str) -> None:
    workload, inventory, reader, values = _fixture()
    if attack.startswith("model"):
        config = b'{"base_model_name_or_path":"example/model","peft_type":"LORA"}'
        payload = _safetensors()
        if attack == "model-one-byte":
            payload = b"x"
        elif attack == "model-zeros":
            payload = b"\x00" * 32
        elif attack == "model-zero-tensor":
            payload = _safetensors(payload=b"\x00" * 4)
        elif attack == "model-offsets":
            payload = _safetensors(offsets=(1, 5))
        elif attack == "model-length":
            payload += b"trailing"
        elif attack == "model-fake-config":
            config = b"{}"
        elif attack == "model-duplicate-config":
            config = b'{"peft_type":"LORA","peft_type":"LORA","base_model_name_or_path":"example/model"}'
        elif attack == "model-nonfinite-config":
            config = b'{"peft_type":"LORA","base_model_name_or_path":"example/model","x":NaN}'
        values["final_model.tar"] = _tar(
            {"adapter_config.json": config, "adapter_model.safetensors": payload}
        )
        failed_code = "final_model_semantic"
    else:
        config = b'{"tokenizer_class":"Fixture"}'
        payload = b'{"model":{"type":"BPE","vocab":{"x":0}},"version":"1.0"}'
        if attack == "tokenizer-one-byte":
            payload = b"x"
        elif attack == "tokenizer-zeros":
            payload = b"\x00" * 32
        elif attack == "tokenizer-fake-config":
            config = b"{}"
        elif attack == "tokenizer-duplicate-config":
            config = b'{"tokenizer_class":"A","tokenizer_class":"B"}'
        elif attack == "tokenizer-nonfinite-config":
            config = b'{"tokenizer_class":"A","x":Infinity}'
        elif attack == "tokenizer-empty-vocab":
            payload = b'{"model":{"type":"BPE","vocab":{}},"version":"1.0"}'
        values["tokenizer.tar"] = _tar(
            {"tokenizer_config.json": config, "tokenizer.json": payload}
        )
        failed_code = "tokenizer_semantic"
    report = _verify_current(workload, inventory, reader, values)
    checks = {item.code: item.passed for item in report.semantic_checks}
    assert report.status is VerificationStatus.INVALID
    assert checks[failed_code] is False


@pytest.mark.parametrize(
    "attacked",
    (
        _tar_pairs(
            (
                ("adapter_config.json", b"{}"),
                ("adapter_config.json", b"{}"),
                ("adapter_model.safetensors", b"model"),
            )
        ),
        _tar_pairs(
            (
                ("../adapter_config.json", b"{}"),
                ("adapter_model.safetensors", b"model"),
            )
        ),
        _tar_pairs(
            (
                ("adapter_config.json", b"{}"),
                ("adapter_model.safetensors", b"model"),
                ("unlisted.bin", b"substitution"),
            )
        ),
    ),
    ids=("duplicate", "traversal", "unlisted"),
)
def test_archive_semantics_reject_duplicate_noncanonical_and_unlisted_members(
    attacked: bytes,
) -> None:
    workload, inventory, reader, values = _fixture()
    values["final_model.tar"] = attacked
    entries = tuple(
        _entry(item.role, item.path, values[item.path]) for item in inventory.entries
    )
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory(entries),
        reader=reader,
    )
    checks = {item.code: item.passed for item in report.semantic_checks}
    assert report.status is VerificationStatus.INVALID
    assert checks["final_model_semantic"] is False


def test_declared_digest_and_size_mismatch_reproduce_as_invalid() -> None:
    workload, inventory, reader, _ = _fixture()
    attacked = replace(
        inventory.entries[0],
        sha256="0" * 64,
        size=inventory.entries[0].size + 999,
    )
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory((attacked, *inventory.entries[1:])),
        reader=reader,
    )

    assert report.status is VerificationStatus.INVALID
    assert not report.success
    assert report.semantic_checks == ()
    assert report.integrity.artifacts[0].errors == (
        "artifact_size_mismatch",
        "artifact_digest_mismatch",
    )


def test_each_declared_artifact_is_authenticated_not_only_semantic_roles() -> None:
    workload, inventory, reader, values = _fixture()
    metrics = inventory.for_role("training_metrics")[0]
    values[metrics.path] = b"[]"
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=inventory,
        reader=reader,
    )

    assert report.status is VerificationStatus.INVALID
    metrics_result = next(
        item for item in report.integrity.artifacts if item.artifact.role == "training_metrics"
    )
    assert metrics_result.errors == ("artifact_digest_mismatch",)


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    (
        ({"sha256": "0" * 64}, "artifact_digest_mismatch"),
        ({"size_delta": 1}, "artifact_size_mismatch"),
    ),
)
def test_digest_and_size_are_independently_authenticated(changes, expected_error) -> None:
    workload, inventory, reader, _ = _fixture()
    original = inventory.entries[0]
    attacked = replace(
        original,
        sha256=changes.get("sha256", original.sha256),
        size=original.size + changes.get("size_delta", 0),
    )
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory((attacked, *inventory.entries[1:])),
        reader=reader,
    )

    assert report.status is VerificationStatus.INVALID
    assert report.integrity.artifacts[0].errors == (expected_error,)


def test_semantics_use_authenticated_bytes_without_a_second_source_read() -> None:
    workload, inventory, _, values = _fixture()

    class MutatingReader:
        def __init__(self):
            self.calls = {}

        def read_bytes(self, artifact, *, maximum):
            self.calls[artifact.path] = self.calls.get(artifact.path, 0) + 1
            content = values[artifact.path]
            values[artifact.path] = b"changed-after-authentication"
            return content

    reader = MutatingReader()
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=inventory,
        reader=reader,
    )

    assert report.status is VerificationStatus.VERIFIED
    assert set(reader.calls.values()) == {1}
    assert set(reader.calls) == {item.path for item in inventory.entries}


@pytest.mark.parametrize(
    ("role", "path", "failed_code"),
    (
        ("workload_record", "workload.json", "workload_record_exact"),
        ("training_lineage", "training_lineage.json", "lineage_binds_workload"),
    ),
)
def test_semantic_artifact_over_256k_is_structured_invalid(
    role: str,
    path: str,
    failed_code: str,
) -> None:
    workload, inventory, reader, values = _fixture()
    oversized = b"x" * (MAX_SEMANTIC_ARTIFACT_BYTES + 1)
    values[path] = oversized
    replacement = _entry(role, path, oversized)
    entries = tuple(
        replacement if item.role == role else item for item in inventory.entries
    )

    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=ArtifactInventory(entries),
        reader=reader,
    )

    assert report.integrity.valid
    assert report.status is VerificationStatus.INVALID
    checks = {item.code: item.passed for item in report.semantic_checks}
    assert checks[failed_code] is False


def test_semantic_drift_invalidates_a_completed_provider_job() -> None:
    workload, inventory, _, values = _fixture()
    values["training_lineage.json"] = json.dumps(
        {"workload_fingerprint": "0" * 64}
    ).encode()

    class DriftedReader:
        def read_bytes(self, artifact, *, maximum):
            return values[artifact.path]

    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=True,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=inventory,
        reader=DriftedReader(),
    )
    assert report.status is VerificationStatus.INVALID
    assert not report.success


def test_nonterminal_provider_cannot_be_verified() -> None:
    workload, inventory, reader, _ = _fixture()
    report = VerificationService(WorkloadBindingVerifier()).verify(
        provider_completed=False,
        process=ProcessResult(0),
        workload=workload,
        contract=SFT_ARTIFACT_CONTRACT,
        inventory=inventory,
        reader=reader,
    )
    assert report.status is VerificationStatus.INCONCLUSIVE
    assert not report.success
