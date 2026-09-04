from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

import Trainers.sft.runtime_v1 as runtime_v1
from synaptic_tuner.api.v1.training import CanonicalDocument
from Trainers.sft.runtime_v1 import (
    MAX_WORKLOAD_BYTES,
    RuntimeV1Error,
    TrainerEvidence,
    TrainerFailed,
    execute_runtime,
    read_authenticated_workload_file,
    read_bounded_workload,
    _canonical_document,
)
from tuner.project.execution_source import (
    AuthenticatedSourceEvidenceV1,
    ExecutionSourceV1,
)
from tuner.project.source_bundle import SourceLock
from tuner.training.methods.sft import compile_sft_workload
from tuner.runtime.verification import (
    _validate_embedded_trainer_lineage,
    _validate_execution_evidence,
)


_REPO = Path(__file__).parents[3]


def _read_fifo_transport_child(path_text: str, control_root_text: str, result) -> None:
    try:
        read_authenticated_workload_file(
            Path(path_text),
            control_root=Path(control_root_text),
            expected_byte_count=1,
            expected_sha256="f" * 64,
            expected_fingerprint="e" * 64,
        )
    except RuntimeV1Error:
        result.put("rejected")
    else:
        result.put("accepted")


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


def _fixture(
    tmp_path: Path,
    *,
    dataset_ref: str = "project://data/train.jsonl",
    mode: str = "dual_clone",
    redirected_capability: bool = False,
    extra_environment: dict[str, str] | None = None,
):
    if mode != "dual_clone":
        raise ValueError(
            "runtime v1 test fixture supports only the finalized dual-clone mode"
        )
    project = (tmp_path / "host-project").resolve()
    engine = (tmp_path / "training-engine").resolve()
    manifest_source = (
        _REPO / "tuner" / "runtime" / "manifests" / "offline-sft-worker-v1.json"
    )
    manifest_document = json.loads(manifest_source.read_text(encoding="utf-8"))
    for member in manifest_document["members"]:
        relative = Path(*member["path"].split("/"))
        destination = engine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / relative, destination)
    control = tmp_path / "control"
    control.mkdir()
    closure_manifest = control / "offline-sft-worker-v1.json"
    shutil.copy2(manifest_source, closure_manifest)
    engine_file = engine / "Trainers" / "sft" / "runtime_v1.py"
    dataset = project / "data" / "train.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b'{"messages":[]}\n')
    capability_root = tmp_path / "capabilities"
    if redirected_capability:
        provider_volume = tmp_path / "provider-volume"
        provider_volume.mkdir()
        try:
            capability_root.symlink_to(provider_volume, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable")
    roots = {
        name: capability_root / name
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    }
    for path in roots.values():
        path.mkdir(parents=True)
    locked_roots = {
        "engine": str(engine),
        "project": str(project),
        **{name: str(path) for name, path in roots.items()},
    }
    project_source = {
        "url": "https://github.com/example/product.git",
        "commit": "a" * 40,
        "dirty": False,
        "pushed": True,
    }
    engine_source = {
        "url": "https://github.com/example/training-engine.git",
        "commit": "b" * 40,
        "dirty": False,
        "pushed": True,
    }
    engine_source.update(
        {
            "submodule_path": "vendor/training-engine",
            "gitlink_commit": "b" * 40,
        }
    )
    source_lock = SourceLock.from_dict(
        {
            "schema_version": "synaptic-source-lock/v1",
            "run_id": "runtime-v1-test",
            "created_at": "2026-08-25T12:00:00Z",
            "mode": "superproject",
            "sources": {"project": project_source, "engine": engine_source},
            "project": {
                "manifest_uri": "project://synaptic.yaml",
                "manifest_sha256": "1" * 64,
                "engine_requires": "training-engine==1",
            },
            "configuration": {
                "resolved_uri": "project://resolved-config.json",
                "resolved_sha256": "2" * 64,
                "documents": [],
            },
            "plugins": [],
            "inputs": [],
            "runtime": {},
            "outputs": {},
        }
    )
    config = CanonicalDocument.from_mapping(
        {
            "schema_version": "synaptic-sft-config/v1",
            "method": "sft",
            "model": {
                "ref": "example/model",
                "revision": "c" * 40,
                "tokenizer_revision": "c" * 40,
                "load_in_4bit": False,
            },
            "dataset": {
                "ref": dataset_ref,
                "revision": "a" * 40,
                "content_digest": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "format": "configured/project-row-adapter-v1",
            },
            "sft": {
                "max_steps": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": "0.0002",
                "max_seq_length": 1024,
                "seed": 7,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": "0.0",
                "lora_target_modules": ["q_proj", "v_proj"],
                "use_dora": False,
                "use_rslora": False,
                "init_lora_weights": True,
                "split_dataset": False,
                "save_steps": 1,
                "save_total_limit": 1,
            },
        }
    )
    model_snapshot = (
        roots["cache"]
        / "model"
        / "models--example--model"
        / "snapshots"
        / ("c" * 40)
    )
    model_snapshot.mkdir(parents=True)
    (model_snapshot / "config.json").write_text("{}", encoding="utf-8")
    planned_environment = {
        "PATH": "fixture-path",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONPATH": locked_roots["engine"],
        "SYNAPTIC_ENGINE_ROOT": locked_roots["engine"],
        "SYNAPTIC_PROJECT_ROOT": locked_roots["project"],
        "SYNAPTIC_ARTIFACT_ROOT": locked_roots["artifacts"],
        "SYNAPTIC_STATE_ROOT": locked_roots["state"],
        "SYNAPTIC_TRACKING_ROOT": locked_roots["tracking"],
        "SYNAPTIC_CACHE_ROOT": locked_roots["cache"],
        "SYNAPTIC_TMP_ROOT": locked_roots["tmp"],
        "HF_HOME": locked_roots["cache"] + "/huggingface",
        "TRANSFORMERS_CACHE": locked_roots["cache"] + "/transformers",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SYNAPTIC_MODEL_SNAPSHOT": str(model_snapshot),
        "WANDB_DISABLED": "true",
    }
    if extra_environment is not None:
        planned_environment.update(extra_environment)
    execution_source = ExecutionSourceV1(
        run_id=source_lock.run_id,
        created_at=source_lock.created_at,
        project_source=source_lock.project_source,
        engine_source=source_lock.engine_source,
        engine_submodule_path="vendor/training-engine",
        source_evidence=AuthenticatedSourceEvidenceV1(
            project_url=project_source["url"],
            project_commit=project_source["commit"],
            engine_url=engine_source["url"],
            engine_commit=engine_source["commit"],
            engine_submodule_path="vendor/training-engine",
            gitlink_commit="b" * 40,
            source_lock_binding=source_lock.binding,
            issuer_ref="test-verifier",
            evidence_ref="test-proof",
            audience_ref="project/run-1",
            challenge_nonce="source-nonce",
            verified_at="2026-08-25T12:01:00Z",
            expires_at="2026-08-25T12:10:00Z",
            key_ref="source-key",
            tag_base64="dGFn",
            attestation_digest="8" * 64,
        ),
        deployment_member_sha256="7" * 64,
        roots=locked_roots,
        writable_capability_root=str(capability_root),
        python_implementation="cpython",
        python_version=".".join(str(part) for part in sys.version_info[:3]),
        python_executable=str(Path(sys.executable).resolve()),
        python_executable_digest="6" * 64,
        environment=planned_environment,
        secret_requirements_digest="5" * 64,
        provider_runtime_requirements_digest="4" * 64,
    )
    workload = compile_sft_workload(
        resolved_config=config,
        execution_source=execution_source,
    )
    environment = dict(planned_environment)
    environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] = str(closure_manifest)
    environment["SYNAPTIC_WORKER_CLOSURE_DIGEST"] = manifest_document[
        "closure_digest"
    ]
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = workload.fingerprint
    return workload, environment, engine_file, roots, dataset


class FakeRunner:
    def __init__(self, *, exit_code: int = 0, metrics=None) -> None:
        self.exit_code = exit_code
        self.metrics = {"loss": 0.25, "steps": 1} if metrics is None else metrics
        self.calls = []

    def run(self, invocation):
        self.calls.append(invocation)
        model = invocation.final_model_dir
        model.mkdir(parents=True)
        (model / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"example/model","peft_type":"LORA"}',
            encoding="utf-8",
        )
        (model / "adapter_model.safetensors").write_bytes(_safetensors())
        (model / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"Fixture"}', encoding="utf-8"
        )
        (model / "tokenizer.json").write_text(
            '{"model":{"type":"BPE","vocab":{"x":0}},"version":"1.0"}',
            encoding="utf-8",
        )
        (model / "training_args.bin").write_bytes(b"known trainer byproduct")
        dataset = invocation.argv[invocation.argv.index("--local-file") + 1]
        projection = dict(invocation.expected_projection)
        return TrainerEvidence(
            self.exit_code,
            model,
            invocation.tokenizer_dir,
            {
                "training_type": "SFT",
                "run_directory": str(invocation.run_dir),
                "model": {"base_model": "example/model", "load_in_4bit": False},
                "dataset": {"source": dataset},
                "training": {
                    "batch_size": 2,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 0.0002,
                    "max_steps": 1,
                    "max_seq_length": 1024,
                    "seed": 7,
                },
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
                "runtime": {"status": "completed"},
                "synaptic_runtime_projection": projection,
            },
            projection,
            self.metrics,
        )


def test_trainer_evidence_rejects_boolean_exit_code() -> None:
    with pytest.raises(TypeError, match="exact integer"):
        TrainerEvidence(False, Path("model"), Path("tokenizer"), {}, {}, {})


def test_runtime_invokes_fixed_non_shell_trainer_and_emits_exact_roles(
    tmp_path: Path, monkeypatch
) -> None:
    workload, environment, engine_file, roots, dataset = _fixture(tmp_path)
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    runner = FakeRunner()

    result = execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=runner,
        engine_file=engine_file,
    )

    invocation = runner.calls[0]
    assert invocation.cwd == roots["tmp"]
    assert invocation.argv[:4] == (
        sys.executable,
        "-I",
        str(
            Path(environment["SYNAPTIC_ENGINE_ROOT"])
            / "tuner/runtime/offline_sft_worker.py"
        ),
        "--",
    )
    assert "--local-file" in invocation.argv
    assert invocation.argv[invocation.argv.index("--local-file") + 1] == str(dataset)
    assert "--max-steps" in invocation.argv
    assert invocation.argv[invocation.argv.index("--model-snapshot") + 1] == (
        environment["SYNAPTIC_MODEL_SNAPSHOT"]
    )
    assert "--no-load-in-4bit" in invocation.argv
    assert not any(";" in item or "&&" in item for item in invocation.argv)
    assert "HF_TOKEN" not in dict(invocation.environment)
    assert "PYTHONPATH" not in dict(invocation.environment)
    assert dict(invocation.environment)["PYTHONNOUSERSITE"] == "1"
    assert dict(invocation.environment)["PYTHONSAFEPATH"] == "1"
    assert dict(invocation.environment)["HF_HUB_OFFLINE"] == "1"
    assert dict(invocation.environment)["TRANSFORMERS_OFFLINE"] == "1"
    assert dict(invocation.environment)["SYNAPTIC_WORKER_CLOSURE_MANIFEST"] == (
        environment["SYNAPTIC_WORKER_CLOSURE_MANIFEST"]
    )
    assert dict(invocation.environment)["SYNAPTIC_WORKER_CLOSURE_DIGEST"] == (
        environment["SYNAPTIC_WORKER_CLOSURE_DIGEST"]
    )
    assert type(invocation.expected_projection["training"]["num_epochs"]) is int
    assert {item["role"] for item in result.artifacts} == {
        "workload_record",
        "training_lineage",
        "training_metrics",
        "final_model",
        "tokenizer",
    }
    assert {path.name for path in roots["artifacts"].iterdir()} == {
        "workload.json",
        "training_lineage.json",
        "training_metrics.json",
        "final_model.tar",
        "tokenizer.tar",
    }
    lineage = json.loads((roots["artifacts"] / "training_lineage.json").read_text())
    assert lineage["workload_fingerprint"] == workload.fingerprint
    execution = lineage["execution_evidence"]
    assert execution["argv"] == list(invocation.argv)
    assert execution["environment"] == dict(invocation.environment)
    assert execution["model"]["revision"] == "c" * 40
    assert execution["model"]["tokenizer_revision"] == "c" * 40
    assert (
        execution["dataset"]["content_digest"]
        == hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    assert execution["sft"] == workload.document["configuration"]["document"]["sft"]
    assert (
        lineage["execution_evidence_sha256"]
        == hashlib.sha256(
            json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert result.inventory_path.parent == roots["state"]
    assert (roots["artifacts"] / "final_model.tar").read_bytes() != (
        roots["artifacts"] / "tokenizer.tar"
    ).read_bytes()


def test_runtime_rejects_missing_local_model_snapshot(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    snapshot = Path(environment["SYNAPTIC_MODEL_SNAPSHOT"])
    shutil.rmtree(snapshot)

    with pytest.raises(RuntimeV1Error, match="snapshot is unavailable"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_rejects_foreign_model_snapshot_binding(tmp_path: Path) -> None:
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)
    environment["SYNAPTIC_MODEL_SNAPSHOT"] = str(
        roots["cache"] / "model" / "foreign" / ("c" * 40)
    )

    with pytest.raises(RuntimeV1Error, match="canonical binding"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


@pytest.mark.parametrize("name", ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
def test_runtime_requires_exact_offline_model_controls(
    tmp_path: Path, name: str
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    environment[name] = "0"

    with pytest.raises(RuntimeV1Error, match="offline model controls"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_rejects_redirected_model_snapshot_member(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    snapshot = Path(environment["SYNAPTIC_MODEL_SNAPSHOT"])
    (snapshot / "config.json").unlink()
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"model")
    try:
        (snapshot / "model.bin").symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(RuntimeV1Error, match="redirect"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_accepts_only_the_finalized_dual_clone_topology(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    result = execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=FakeRunner(),
        engine_file=engine_file,
    )
    assert result.workload_fingerprint == workload.fingerprint


def test_runtime_replaces_hostile_python_environment(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    environment.update(
        {
            "PYTHONPATH": "C:/hostile/imports",
            "PYTHONHOME": "C:/hostile/home",
            "HF_TOKEN": "must-not-pass",
        }
    )
    runner = FakeRunner()
    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=runner,
        engine_file=engine_file,
    )
    child = dict(runner.calls[0].environment)
    assert "PYTHONPATH" not in child
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONSAFEPATH"] == "1"
    assert "PYTHONHOME" not in child
    assert "HF_TOKEN" not in child


def test_runtime_routes_qwen2_fast_tokenizer_sidecars_to_tokenizer_only(
    tmp_path: Path,
) -> None:
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)

    class Qwen2FastOutput(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            root = invocation.final_model_dir
            (root / "vocab.json").write_text(
                json.dumps({"hello": 0}, indent=2), encoding="utf-8"
            )
            (root / "merges.txt").write_text("#version: 0.2\nh e\n", encoding="utf-8")
            (root / "added_tokens.json").write_text(
                json.dumps({"<|im_start|>": 1}, indent=2), encoding="utf-8"
            )
            (root / "special_tokens_map.json").write_text(
                json.dumps({"eos_token": "<|im_end|>"}, indent=2), encoding="utf-8"
            )
            (root / "chat_template.jinja").write_text(
                "{% for message in messages %}{{ message.content }}{% endfor %}",
                encoding="utf-8",
            )
            return evidence

    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=Qwen2FastOutput(),
        engine_file=engine_file,
    )
    with tarfile.open(roots["artifacts"] / "final_model.tar") as archive:
        model_names = set(archive.getnames())
    with tarfile.open(roots["artifacts"] / "tokenizer.tar") as archive:
        tokenizer_names = set(archive.getnames())
    assert {
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    } <= tokenizer_names
    assert model_names.isdisjoint(tokenizer_names)


@pytest.mark.parametrize(
    ("name", "content"),
    (
        ("vocab.json", '{"token":true}'),
        ("merges.txt", "not-a-pair"),
        ("special_tokens_map.json", '{"x":NaN}'),
    ),
)
def test_runtime_rejects_malformed_optional_tokenizer_sidecars(
    tmp_path: Path, name: str, content: str
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class BadSidecar(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            (invocation.final_model_dir / name).write_text(content, encoding="utf-8")
            return evidence

    with pytest.raises(RuntimeV1Error):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=BadSidecar(),
            engine_file=engine_file,
        )


def test_portable_runtime_requirements_enforce_implementation_and_version(
    monkeypatch,
) -> None:
    requirements = {
        "schema_version": "synaptic-sft-runtime-requirements/v1",
        "python": {
            "implementation": "cpython",
            "minimum_version": "3.12",
            "maximum_version_exclusive": "3.14",
        },
        "isolation": {"no_user_site": True, "safe_path": True},
        "allowed_environment": ["PATH"],
        "trainer_projection_schema": "synaptic-sft-trainer-projection/v1",
        "artifact_formats": {
            "model": ["peft-safetensors", "full-safetensors"],
            "tokenizer": "tokenizer-json",
        },
    }
    runtime_v1._validate_portable_runtime_requirements(requirements)
    monkeypatch.setattr(runtime_v1.sys, "version_info", (3, 12, 0))
    runtime_v1._validate_portable_runtime_requirements(requirements)
    monkeypatch.setattr(runtime_v1.sys, "version_info", (3, 11, 99))
    with pytest.raises(RuntimeV1Error, match="outside"):
        runtime_v1._validate_portable_runtime_requirements(requirements)
    monkeypatch.setattr(runtime_v1.sys, "version_info", (3, 14, 0))
    with pytest.raises(RuntimeV1Error, match="outside"):
        runtime_v1._validate_portable_runtime_requirements(requirements)
    monkeypatch.setitem(requirements["python"], "implementation", "pypy")
    with pytest.raises(RuntimeV1Error, match="implementation"):
        runtime_v1._validate_portable_runtime_requirements(requirements)


_REDIRECTED_CACHE_ENVIRONMENT = {
    "HOME": "/tmp/home",
    "XDG_CACHE_HOME": "/tmp/xdg",
    "TORCH_HOME": "/tmp/torch",
    "TRITON_CACHE_DIR": "/tmp/triton",
}


def test_runtime_admits_the_redirected_cache_environment(tmp_path: Path) -> None:
    """B-9-R1: the four cache-root keys pass the allowlist subset gate.

    The container runs as a foreign uid with an unwritable ``HOME``, so the Host
    redirects the torch/triton/XDG caches under ``/tmp``. That dispatch
    environment only reaches the trainer if every key is inside
    ``allowed_environment`` (``tuner/training/methods/sft.py``), which
    ``execute_runtime`` enforces as a subset check. This drives that gate and
    then confirms the keys survive into the child process environment, which is
    the whole point of admitting them.
    """
    workload, environment, engine_file, _, _ = _fixture(
        tmp_path, extra_environment=dict(_REDIRECTED_CACHE_ENVIRONMENT)
    )
    runner = FakeRunner()

    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=runner,
        engine_file=engine_file,
    )

    child_environment = dict(runner.calls[0].environment)
    for name, value in _REDIRECTED_CACHE_ENVIRONMENT.items():
        assert child_environment[name] == value


def test_runtime_rejects_a_dispatch_environment_key_outside_the_allowlist(
    tmp_path: Path,
) -> None:
    """The gate above is live: an unadmitted key still fails closed.

    ``TMPDIR`` is deliberately excluded from ``allowed_environment`` because
    ``/tmp`` is already writable, so it is the exact counter-case for the test
    above. Same fixture, one extra key, opposite outcome.
    """
    workload, environment, engine_file, _, _ = _fixture(
        tmp_path,
        extra_environment={**_REDIRECTED_CACHE_ENVIRONMENT, "TMPDIR": "/tmp/other"},
    )

    with pytest.raises(RuntimeV1Error, match="violates portable requirements"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_admits_the_container_user_name_environment(tmp_path: Path) -> None:
    """B-16 E3: ``USER`` passes the subset gate and reaches the child.

    The prepared path runs the container as ``--user 1000:1000``, a uid the
    pinned image's password database does not name, so ``getpass.getuser()``
    deep inside the unsloth import chain falls through to ``pwd.getpwuid`` and
    raises. ``getpass`` reads ``USER`` from the environment before it consults
    ``pwd``, so the Host binds it to a literal name. That binding only survives
    into the trainer's child process if ``allowed_environment`` admits the key,
    which ``build_trainer_invocation`` enforces as a subset check.

    This is the positive direction of that gate. Its counter-test is
    ``test_runtime_rejects_a_dispatch_environment_key_outside_the_allowlist``
    above: same fixture, one unadmitted key, opposite outcome. Without that
    pair a blanket allowlist would satisfy this assertion.

    Neither test proves B-16 is closed. The failing import is a third-party
    chain inside a pinned image and only run 11 can prove it passes; these prove
    the gate will let the remedy through.
    """
    workload, environment, engine_file, _, _ = _fixture(
        tmp_path, extra_environment={"USER": "synaptic"}
    )
    runner = FakeRunner()

    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=runner,
        engine_file=engine_file,
    )

    assert dict(runner.calls[0].environment)["USER"] == "synaptic"


def test_runtime_rejects_unrecognized_or_empty_model_output(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class MissingModelConfig(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            (invocation.final_model_dir / "adapter_config.json").unlink()
            return evidence

    with pytest.raises(RuntimeV1Error, match="exactly one model family"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=MissingModelConfig(),
            engine_file=engine_file,
        )


@pytest.mark.parametrize(
    "attack",
    (
        "mixed-family",
        "base-drift",
        "snapshot-path",
        "unknown-weight",
        "incomplete-shards",
        "mixed-shard-unsharded",
        "orphan-index",
        "opposite-index",
        "metadata-bool",
    ),
)
def test_runtime_rejects_ambiguous_or_incomplete_model_family(
    tmp_path: Path, attack: str
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class HostileModel(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            root = invocation.final_model_dir
            if attack == "mixed-family":
                (root / "config.json").write_text(
                    '{"model_type":"fixture"}', encoding="utf-8"
                )
                (root / "model.safetensors").write_bytes(_safetensors())
            elif attack == "base-drift":
                (root / "adapter_config.json").write_text(
                    '{"base_model_name_or_path":"attacker/model","peft_type":"LORA"}',
                    encoding="utf-8",
                )
            elif attack == "snapshot-path":
                # The exact shape B-2 produces in the field: peft derives
                # base_model_name_or_path from the loaded model's _name_or_path,
                # which the SFT model loader asserts is the offline snapshot
                # directory, so an unstamped run writes a PATH here. base-drift
                # only proves the check rejects a different repo id; this proves
                # it rejects a path. The FakeRunner writes an already-correct
                # "example/model", so without this case the suite never sees one.
                (root / "adapter_config.json").write_text(
                    '{"base_model_name_or_path":'
                    '"/artifacts/cache/model/models--HuggingFaceTB--SmolLM2-135M-Instruct'
                    '/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac",'
                    '"peft_type":"LORA"}',
                    encoding="utf-8",
                )
            elif attack == "unknown-weight":
                (root / "evil.bin").write_bytes(b"not a supported model artifact")
            elif attack == "incomplete-shards":
                (root / "adapter_model.safetensors").unlink()
                (root / "adapter_model-00001-of-00002.safetensors").write_bytes(
                    _safetensors()
                )
                (root / "adapter_model.safetensors.index.json").write_text(
                    '{"metadata":{},"weight_map":{"weight":"adapter_model-00001-of-00002.safetensors"}}',
                    encoding="utf-8",
                )
            elif attack == "mixed-shard-unsharded":
                (root / "adapter_model-00001-of-00001.safetensors").write_bytes(
                    _safetensors()
                )
                (root / "adapter_model.safetensors.index.json").write_text(
                    '{"metadata":{},"weight_map":{"weight":"adapter_model-00001-of-00001.safetensors"}}',
                    encoding="utf-8",
                )
            elif attack == "orphan-index":
                (root / "adapter_model.safetensors.index.json").write_text(
                    '{"metadata":{},"weight_map":{}}', encoding="utf-8"
                )
            elif attack == "opposite-index":
                (root / "model.safetensors.index.json").write_text(
                    '{"metadata":{},"weight_map":{}}', encoding="utf-8"
                )
            else:
                (root / "adapter_model.safetensors").unlink()
                (root / "adapter_model-00001-of-00001.safetensors").write_bytes(
                    _safetensors()
                )
                (root / "adapter_model.safetensors.index.json").write_text(
                    '{"metadata":{"total_size":true},"weight_map":{"weight":"adapter_model-00001-of-00001.safetensors"}}',
                    encoding="utf-8",
                )
            return evidence

    with pytest.raises(RuntimeV1Error):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=HostileModel(),
            engine_file=engine_file,
        )


@pytest.mark.parametrize("family", ("full", "sharded-peft"))
def test_runtime_accepts_each_supported_exact_model_family(
    tmp_path: Path, family: str
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class SupportedModel(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            root = invocation.final_model_dir
            if family == "full":
                (root / "adapter_config.json").unlink()
                (root / "adapter_model.safetensors").unlink()
                (root / "config.json").write_text(
                    '{"model_type":"fixture"}', encoding="utf-8"
                )
                (root / "model.safetensors").write_bytes(_safetensors())
            else:
                (root / "adapter_model.safetensors").unlink()
                (root / "adapter_model-00001-of-00002.safetensors").write_bytes(
                    _safetensors(name="weight_a")
                )
                (root / "adapter_model-00002-of-00002.safetensors").write_bytes(
                    _safetensors(payload=b"\x02\x00\x00\x00", name="weight_b")
                )
                (root / "adapter_model.safetensors.index.json").write_text(
                    '{"metadata":{},"weight_map":{"weight_a":"adapter_model-00001-of-00002.safetensors","weight_b":"adapter_model-00002-of-00002.safetensors"}}',
                    encoding="utf-8",
                )
            return evidence

    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=SupportedModel(),
        engine_file=engine_file,
    )


@pytest.mark.parametrize(
    ("filename", "content"),
    (
        ("adapter_config.json", b"{}"),
        (
            "adapter_config.json",
            b'{"peft_type":"LORA","peft_type":"LORA","base_model_name_or_path":"example/model"}',
        ),
        (
            "adapter_config.json",
            b'{"peft_type":"LORA","base_model_name_or_path":"example/model","x":NaN}',
        ),
        ("adapter_model.safetensors", b"x"),
        ("adapter_model.safetensors", b"\x00" * 32),
        ("adapter_model.safetensors", _safetensors(payload=b"\x00" * 4)),
        ("adapter_model.safetensors", _safetensors(offsets=(1, 5))),
        ("adapter_model.safetensors", _safetensors() + b"trailing"),
    ),
    ids=(
        "fake-config",
        "duplicate-config",
        "nonfinite-config",
        "one-byte",
        "zeros",
        "zero-tensor",
        "offsets",
        "length",
    ),
)
def test_runtime_rejects_fake_model_contents(
    tmp_path: Path, filename: str, content: bytes
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class CorruptModel(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            (invocation.final_model_dir / filename).write_bytes(content)
            return evidence

    with pytest.raises(RuntimeV1Error):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=CorruptModel(),
            engine_file=engine_file,
        )


@pytest.mark.parametrize(
    ("filename", "content"),
    (
        ("tokenizer_config.json", b"{}"),
        (
            "tokenizer_config.json",
            b'{"tokenizer_class":"A","tokenizer_class":"B"}',
        ),
        ("tokenizer_config.json", b'{"tokenizer_class":"A","x":Infinity}'),
        ("tokenizer.json", b"x"),
        ("tokenizer.json", b"\x00" * 32),
        ("tokenizer.json", b'{"model":{"type":"BPE","vocab":{}},"version":"1.0"}'),
    ),
    ids=(
        "fake-config",
        "duplicate-config",
        "nonfinite-config",
        "one-byte",
        "zeros",
        "empty-vocab",
    ),
)
def test_runtime_rejects_fake_tokenizer_contents(
    tmp_path: Path, filename: str, content: bytes
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class CorruptTokenizer(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            (invocation.tokenizer_dir / filename).write_bytes(content)
            return evidence

    with pytest.raises(RuntimeV1Error):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=CorruptTokenizer(),
            engine_file=engine_file,
        )


def test_runtime_rejects_lineage_not_bound_to_invocation(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class DriftedLineage(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            projection = {**evidence.projection, "status": "drifted"}
            return TrainerEvidence(
                evidence.exit_code,
                evidence.final_model_dir,
                evidence.tokenizer_dir,
                {**evidence.lineage, "synaptic_runtime_projection": projection},
                projection,
                evidence.metrics,
            )

    with pytest.raises(RuntimeV1Error, match="lineage"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=DriftedLineage(),
            engine_file=engine_file,
        )


def test_runtime_projection_comparison_is_json_type_strict(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class BooleanNumericAlias(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            projection = json.loads(json.dumps(evidence.projection))
            projection["model"]["load_in_4bit"] = 0
            lineage = {**evidence.lineage, "synaptic_runtime_projection": projection}
            return TrainerEvidence(
                evidence.exit_code,
                evidence.final_model_dir,
                evidence.tokenizer_dir,
                lineage,
                projection,
                evidence.metrics,
            )

    with pytest.raises(RuntimeV1Error, match="bind"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=BooleanNumericAlias(),
            engine_file=engine_file,
        )


def test_runtime_rejects_unexpected_child_directory(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class NestedOutput(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            nested = invocation.final_model_dir / "checkpoint-copy"
            nested.mkdir()
            (nested / "ignored.bin").write_bytes(b"must not be ignored")
            return evidence

    with pytest.raises(RuntimeV1Error, match="nested"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=NestedOutput(),
            engine_file=engine_file,
        )


def test_runtime_detects_artifact_mutation_during_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class WrongDigest:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(runtime_v1.hashlib, "file_digest", lambda *args: WrongDigest())
    with pytest.raises(RuntimeV1Error, match="changed during archival"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_rejects_environment_root_not_bound_by_workload(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    environment["SYNAPTIC_ARTIFACT_ROOT"] = str((tmp_path / "other").resolve())
    Path(environment["SYNAPTIC_ARTIFACT_ROOT"]).mkdir()

    with pytest.raises(RuntimeV1Error, match="locked runtime root"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )


def test_runtime_accepts_one_locked_provider_capability_redirect(
    tmp_path: Path,
) -> None:
    workload, environment, engine_file, roots, _ = _fixture(
        tmp_path, redirected_capability=True
    )
    runner = FakeRunner()

    result = execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=runner,
        engine_file=engine_file,
    )

    assert result.workload_fingerprint == workload.fingerprint
    assert runner.calls[0].cwd == (tmp_path / "provider-volume" / "tmp").resolve()
    lineage = json.loads(
        (roots["artifacts"] / "training_lineage.json").read_text(encoding="utf-8")
    )
    execution = lineage["execution_evidence"]
    assert _validate_execution_evidence(
        execution, workload,
        closure_digest=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_DIGEST"],
        closure_manifest_path=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_MANIFEST"],
    )
    assert _validate_embedded_trainer_lineage(
        lineage["trainer_lineage"], workload, execution
    )


def _relocated_runtime_lineage(tmp_path: Path):
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)
    execute_runtime(
        workload.canonical_bytes,
        environment=environment,
        runner=FakeRunner(),
        engine_file=engine_file,
    )
    lineage = json.loads(
        (roots["artifacts"] / "training_lineage.json").read_text(encoding="utf-8")
    )
    locked = workload.document["execution_source"]["runtime"]["roots"]
    observed_boundary = (tmp_path / "provider-observed" / "runtime-v1-test").resolve()
    replacements = {
        locked[name]
        .replace("\\", "/"): str(observed_boundary / name)
        .replace("\\", "/")
        for name in ("artifacts", "state", "tracking", "cache", "tmp")
    }

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for old, new in replacements.items():
                if normalized == old or normalized.startswith(old + "/"):
                    return new + normalized[len(old) :]
            return normalized
        return value

    return workload, relocate(deepcopy(lineage))


def test_semantic_verifier_accepts_one_consistent_provider_root_relocation(
    tmp_path: Path,
) -> None:
    workload, lineage = _relocated_runtime_lineage(tmp_path)
    execution = lineage["execution_evidence"]
    assert _validate_execution_evidence(
        execution, workload,
        closure_digest=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_DIGEST"],
        closure_manifest_path=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_MANIFEST"],
    )
    assert _validate_embedded_trainer_lineage(
        lineage["trainer_lineage"], workload, execution
    )


def test_semantic_verifier_rejects_inconsistent_provider_root_relocation(
    tmp_path: Path,
) -> None:
    workload, lineage = _relocated_runtime_lineage(tmp_path)
    execution = lineage["execution_evidence"]
    execution["environment"]["SYNAPTIC_CACHE_ROOT"] = str(
        (tmp_path / "different-provider-root" / "cache").resolve()
    ).replace("\\", "/")
    assert not _validate_execution_evidence(
        execution, workload,
        closure_digest=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_DIGEST"],
        closure_manifest_path=execution["environment"]["SYNAPTIC_WORKER_CLOSURE_MANIFEST"],
    )


def test_runtime_rejects_redirect_below_locked_capability_boundary(
    tmp_path: Path,
) -> None:
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)
    shutil.rmtree(roots["cache"])
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    try:
        roots["cache"].symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(RuntimeV1Error, match="redirected component") as failure:
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )
    assert failure.value.diagnostic_code == "runtime_workload_roots_rejected"


def test_runtime_rejects_stale_dispatcher_fingerprint(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    environment["SYNAPTIC_WORKLOAD_FINGERPRINT"] = "0" * 64
    with pytest.raises(RuntimeV1Error, match="fingerprint") as failure:
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )
    assert failure.value.diagnostic_code == "runtime_workload_fingerprint_rejected"


def test_runtime_rejects_project_dataset_traversal(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(
        tmp_path, dataset_ref="project://../outside.jsonl"
    )
    with pytest.raises(RuntimeV1Error, match="escapes") as failure:
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )
    assert failure.value.diagnostic_code == "runtime_invocation_rejected"


def test_runtime_propagates_trainer_failure_without_artifacts(tmp_path: Path) -> None:
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)
    with pytest.raises(TrainerFailed) as failure:
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(exit_code=17),
            engine_file=engine_file,
        )
    assert failure.value.diagnostic_code == "runtime_trainer_failed"
    assert not tuple(roots["artifacts"].iterdir())


def test_runtime_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    workload, environment, engine_file, roots, _ = _fixture(tmp_path)
    with pytest.raises(RuntimeV1Error, match="finite"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FakeRunner(metrics={"loss": float("nan")}),
            engine_file=engine_file,
        )
    assert not tuple(roots["artifacts"].iterdir())


def test_bounded_stdin_rejects_one_byte_over_limit() -> None:
    with pytest.raises(RuntimeV1Error, match="byte bound"):
        read_bounded_workload(io.BytesIO(b"x" * (MAX_WORKLOAD_BYTES + 1)))


def test_bounded_stdin_accepts_exact_limit_and_rejects_trailing_byte() -> None:
    exact = b"x" * MAX_WORKLOAD_BYTES
    assert read_bounded_workload(io.BytesIO(exact)) == exact
    with pytest.raises(RuntimeV1Error, match="byte bound"):
        read_bounded_workload(io.BytesIO(exact + b"!"))


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_accepts_the_exact_sealed_workload(
    tmp_path: Path,
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    path.write_bytes(workload.canonical_bytes)

    assert (
        read_authenticated_workload_file(
            path,
            control_root=control_root,
            expected_byte_count=len(workload.canonical_bytes),
            expected_sha256=hashlib.sha256(workload.canonical_bytes).hexdigest(),
            expected_fingerprint=workload.fingerprint,
        )
        == workload.canonical_bytes
    )


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_runtime_parser_consumes_only_the_complete_file_transport_binding(
    tmp_path: Path,
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    path.write_bytes(workload.canonical_bytes)
    args = runtime_v1.build_parser().parse_args(
        [
            "--canonical-workload-file",
            str(path),
            "--canonical-workload-control-root",
            str(control_root),
            "--canonical-workload-byte-count",
            str(len(workload.canonical_bytes)),
            "--canonical-workload-sha256",
            hashlib.sha256(workload.canonical_bytes).hexdigest(),
            "--canonical-workload-fingerprint",
            workload.fingerprint,
        ]
    )

    assert runtime_v1._read_workload_input(args) == workload.canonical_bytes


def test_runtime_rejects_incomplete_or_mixed_file_transport_flags() -> None:
    incomplete = runtime_v1.build_parser().parse_args(
        ["--canonical-workload-file", "/control/workload.json"]
    )
    with pytest.raises(RuntimeV1Error, match="complete binding"):
        runtime_v1._read_workload_input(incomplete)

    mixed = runtime_v1.build_parser().parse_args(
        [
            "--canonical-workload-stdin",
            "--canonical-workload-sha256",
            "f" * 64,
        ]
    )
    with pytest.raises(RuntimeV1Error, match="cannot include file bindings"):
        runtime_v1._read_workload_input(mixed)


@pytest.mark.parametrize("binding", ("size", "digest", "fingerprint"))
@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_rejects_a_foreign_binding(
    tmp_path: Path, binding: str
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    path.write_bytes(workload.canonical_bytes)
    values = {
        "expected_byte_count": len(workload.canonical_bytes),
        "expected_sha256": hashlib.sha256(workload.canonical_bytes).hexdigest(),
        "expected_fingerprint": workload.fingerprint,
    }
    values[
        {
            "size": "expected_byte_count",
            "digest": "expected_sha256",
            "fingerprint": "expected_fingerprint",
        }[binding]
    ] = (
        len(workload.canonical_bytes) - 1 if binding == "size" else "f" * 64
    )

    with pytest.raises(RuntimeV1Error, match="binding"):
        read_authenticated_workload_file(path, control_root=control_root, **values)


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_rejects_a_redirected_member(
    tmp_path: Path,
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(workload.canonical_bytes)
    path = control_root / "workload.json"
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(RuntimeV1Error, match="descriptor traversal"):
        read_authenticated_workload_file(
            path,
            control_root=control_root,
            expected_byte_count=len(workload.canonical_bytes),
            expected_sha256=hashlib.sha256(workload.canonical_bytes).hexdigest(),
            expected_fingerprint=workload.fingerprint,
        )


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_rejects_fifo_member_promptly(
    tmp_path: Path,
) -> None:
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    os.mkfifo(path)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(
        target=_read_fifo_transport_child,
        args=(str(path), str(control_root), result),
    )
    process.start()
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        pytest.fail("FIFO workload member blocked instead of failing promptly")
    assert process.exitcode == 0
    assert result.get(timeout=1) == "rejected"


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_rejects_identity_change_during_read(
    tmp_path: Path, monkeypatch
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    path.write_bytes(workload.canonical_bytes)
    original = runtime_v1._file_identity
    calls = 0

    def changed_identity(value):
        nonlocal calls
        calls += 1
        identity = original(value)
        return identity if calls == 1 else (*identity[:-1], identity[-1] + 1)

    monkeypatch.setattr(runtime_v1, "_file_identity", changed_identity)
    with pytest.raises(RuntimeV1Error, match="changed while"):
        read_authenticated_workload_file(
            path,
            control_root=control_root,
            expected_byte_count=len(workload.canonical_bytes),
            expected_sha256=hashlib.sha256(workload.canonical_bytes).hexdigest(),
            expected_fingerprint=workload.fingerprint,
        )


def test_authenticated_file_transport_fails_closed_without_posix_primitives(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runtime_v1.os, "name", "nt")
    with pytest.raises(RuntimeV1Error, match="POSIX descriptor traversal"):
        read_authenticated_workload_file(
            tmp_path / "control" / "workload.json",
            control_root=tmp_path / "control",
            expected_byte_count=1,
            expected_sha256="f" * 64,
            expected_fingerprint="e" * 64,
        )


def test_authenticated_file_transport_rejects_noncanonical_paths_before_open() -> None:
    with pytest.raises(RuntimeV1Error, match="canonical and absolute"):
        read_authenticated_workload_file(
            Path("/control/../control/workload.json"),
            control_root=Path("/control/../control"),
            expected_byte_count=1,
            expected_sha256="f" * 64,
            expected_fingerprint="e" * 64,
        )


@pytest.mark.skipif(os.name != "posix", reason="sealed file transport is POSIX-only")
def test_authenticated_file_transport_rejects_ancestry_identity_swap(
    tmp_path: Path, monkeypatch
) -> None:
    workload, _, _, _, _ = _fixture(tmp_path)
    control_root = (tmp_path / "control").resolve()
    control_root.mkdir()
    path = control_root / "workload.json"
    path.write_bytes(workload.canonical_bytes)
    original = runtime_v1._open_directory_chain
    calls = 0

    def changed_chain(root):
        nonlocal calls
        calls += 1
        descriptors, identities = original(root)
        if calls == 2:
            identities = (*identities[:-1], (identities[-1][0], identities[-1][1] + 1))
        return descriptors, identities

    monkeypatch.setattr(runtime_v1, "_open_directory_chain", changed_chain)
    with pytest.raises(RuntimeV1Error, match="ancestry changed"):
        read_authenticated_workload_file(
            path,
            control_root=control_root,
            expected_byte_count=len(workload.canonical_bytes),
            expected_sha256=hashlib.sha256(workload.canonical_bytes).hexdigest(),
            expected_fingerprint=workload.fingerprint,
        )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"x":1e10000}',
        b'{"x":-1e10000}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'\xef\xbb\xbf{"x":1}',
        b'{"x":1}trailing',
    ),
)
def test_strict_canonical_parser_closes_hostile_numbers_and_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(RuntimeV1Error):
        _canonical_document(payload)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation unavailable")
def test_runtime_rejects_special_artifact_entry(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)

    class FifoRunner(FakeRunner):
        def run(self, invocation):
            evidence = super().run(invocation)
            os.mkfifo(invocation.final_model_dir / "hostile-fifo")
            return evidence

    with pytest.raises(RuntimeV1Error, match="special"):
        execute_runtime(
            workload.canonical_bytes,
            environment=environment,
            runner=FifoRunner(),
            engine_file=engine_file,
        )


def test_noncanonical_and_duplicate_json_are_rejected(tmp_path: Path) -> None:
    workload, environment, engine_file, _, _ = _fixture(tmp_path)
    noncanonical = json.dumps(workload.document, indent=2).encode()
    with pytest.raises(RuntimeV1Error, match="canonically"):
        execute_runtime(
            noncanonical,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )
    duplicate = b'{"schema_version":"x","schema_version":"y"}'
    with pytest.raises(RuntimeV1Error, match="strict JSON"):
        execute_runtime(
            duplicate,
            environment=environment,
            runner=FakeRunner(),
            engine_file=engine_file,
        )
