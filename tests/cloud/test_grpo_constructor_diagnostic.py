from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts.grpo_constructor_probe_child import diagnose
from tuner.cloud.grpo_constructor_diagnostic import (
    ADMISSION_SCHEMA, _file_digest, _copy_model_snapshot,
    validate_probe_admission, derive_probe_input,
    diagnose_derived_grpo_constructor,
)
from tuner.cloud.derived_training_image import DerivedTrainingImageError
from tuner.cloud.hf_training_image_lock import CommandResult, CommandSpec, TrainingImageLockError, subprocess_runner


def _admission():
    digests = dict(source_sha256="1" * 64, input_sha256="2" * 64,
        recipe_sha256="3" * 64, dataset_sha256="4" * 64,
        plan_ready_sha256="5" * 64, profile_sha256="6" * 64,
        runtime_release_digest="7" * 64, image_config_digest="sha256:" + "8" * 64,
        host_helper_sha256="a" * 64, cli_sha256="b" * 64,
        model_ref="model/example", model_revision="c" * 40)
    admission = {"schema_version": ADMISSION_SCHEMA, "source_commit": "a" * 40,
        "probe_source_sha256": digests["source_sha256"],
        **{key: value for key, value in digests.items() if key != "source_sha256"},
        "model_inventory": [{"path": "config.json", "size_bytes": 2, "sha256": "9" * 64}],
        "gpu_device": 0}
    return admission, digests


def test_admission_rejects_each_source_input_runtime_and_image_mutation():
    admission, digests = _admission()
    validate_probe_admission(admission, **digests)
    for key in ("source_commit", "probe_source_sha256", "host_helper_sha256", "cli_sha256", "input_sha256", "recipe_sha256",
                "dataset_sha256", "plan_ready_sha256", "profile_sha256",
                "runtime_release_digest", "image_config_digest", "model_ref", "model_revision", "gpu_device"):
        changed = dict(admission)
        changed[key] = "b" * 40 if key == "source_commit" else (16 if key == "gpu_device" else "0" * 64)
        if key == "source_commit":
            # Any valid commit may be reviewed; the independent git check binds it.
            continue
        with pytest.raises(DerivedTrainingImageError):
            validate_probe_admission(changed, **digests)


def test_file_digest_rejects_symlink_and_mutation(tmp_path: Path):
    source = tmp_path / "model"
    source.write_bytes(b"model")
    assert _file_digest(source, 5) == (5, __import__("hashlib").sha256(b"model").hexdigest())
    source.write_bytes(b"model2")
    with pytest.raises(ValueError):
        _file_digest(source, 5)
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        _file_digest(link, 10)


def test_private_model_copy_rechecks_bytes_and_rejects_changed_source(tmp_path: Path):
    import hashlib
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    (source / "nested").mkdir(mode=0o700)
    (source / "nested" / "weights").write_bytes(b"pinned")
    member = {"path": "nested/weights", "size_bytes": 6,
              "sha256": hashlib.sha256(b"pinned").hexdigest()}
    destination = tmp_path / "copy"
    _copy_model_snapshot(source, destination, [member])
    assert (destination / "nested" / "weights").read_bytes() == b"pinned"
    assert (destination / "nested" / "weights").stat().st_mode & 0o222 == 0
    (source / "nested" / "weights").write_bytes(b"tamper")
    with pytest.raises(DerivedTrainingImageError):
        _copy_model_snapshot(source, tmp_path / "second", [member])


def test_input_derives_from_existing_compiler_and_exact_four_rows(tmp_path: Path):
    import hashlib
    import yaml
    from synaptic_tuner.api.v1.training_input import TrainingInputV1
    from synaptic_tuner.api.v1.training_sources import PreparedTrainingInputIdentity
    from tuner.training.grpo_compilation import compile_grpo_input, grpo_prepared_semantic_digest
    root = Path(__file__).resolve().parents[2]
    recipe = root / "examples/grpo_captured_candidate/recipe.yaml"
    dataset = root / "examples/grpo_captured_candidate/rows.jsonl"
    data = yaml.safe_load(recipe.read_bytes())
    raw = dataset.read_bytes()
    content = hashlib.sha256(raw).hexdigest()
    semantic = grpo_prepared_semantic_digest(content_digest=content, size_bytes=len(raw),
        row_count=4, prompt_column="prompt")
    prepared = PreparedTrainingInputIdentity(ref="prepared://sha256/" + semantic,
        revision=semantic, content_digest=content, size_bytes=len(raw), format="synaptic-grpo-prompts/v1")
    training = TrainingInputV1.from_dict(data["training_input"])
    compiled = compile_grpo_input(training, prepared)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"data": {"training_input": training.to_dict(),
        "prepared_dataset": prepared.to_dict(), "core_workload_digest": compiled.fingerprint}}))
    input_raw = derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=dataset)
    projected = json.loads(input_raw)
    assert len(projected["rows"]) == 4
    assert projected["config"] == compiled.document["config"]
    assert projected["qualification_policy"] == data["limits"]["qualification_evidence"]
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(raw.replace(b"blue", b"green", 1))
    with pytest.raises(DerivedTrainingImageError):
        derive_probe_input(recipe_path=recipe, plan_ready_path=plan, dataset_path=changed)


def test_constructor_probe_uses_actual_preparation_shape_without_train(monkeypatch, tmp_path):
    from Trainers.grpo import runtime_v1 as runtime
    from Trainers.grpo.src import data_loader, rewards

    calls = []
    class Trainer:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def train(self):
            raise AssertionError("training must not run")
    class ModelLoader:
        @staticmethod
        def load_model_and_tokenizer(*args, **kwargs):
            calls.append(("load", kwargs))
            return object(), object(), False
        @staticmethod
        def apply_lora_adapters(**kwargs):
            calls.append(("lora", kwargs))
            return object()
    monkeypatch.setattr(runtime, "build_grpo_config", lambda *a, **kw: object())
    monkeypatch.setattr(runtime, "_QualificationMeasurements", lambda *a, **kw: SimpleNamespace(
        reward_wrapper=lambda reward: reward, begin=lambda *a: None,
        end=lambda *a: None, log=lambda *a: None))
    monkeypatch.setattr(data_loader, "dataset_from_rows", lambda rows, dataset_class: rows)
    monkeypatch.setattr(data_loader, "format_dataset_for_grpo", lambda data, *a, **kw: data)
    monkeypatch.setattr(rewards, "build_combined_reward_function", lambda *a, **kw: (lambda *x: 1, {}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=type("TrainerCallback", (), {})))
    payload = {"schema_version": "synaptic-grpo-constructor-probe-input/v1",
        "config": {"model": {"max_seq_length": 256}, "lora": {}, "dataset": {"prompt_column": "prompt"},
                   "training": {}, "rewards": {}},
        "rows": [{"prompt": str(i)} for i in range(4)],
        "reward_documents": {}, "reward_schema": {}, "qualification_policy": {}}
    deps = lambda: {"trainer_class": Trainer, "config_class": object, "bf16_supported": False,
                    "model_loader": ModelLoader, "dataset_class": object}
    result = diagnose(payload, model_root=tmp_path, output_root=tmp_path, dependencies=deps)
    assert (result["status"], result["stage"], result["category"]) == ("CONSTRUCTED", "CONSTRUCTOR", "NONE")
    assert set(calls[-1]) == {"model", "processing_class", "reward_funcs", "args", "train_dataset", "callbacks"}
    assert len(calls[-1]["train_dataset"]) == 4
    assert len(calls[-1]["callbacks"]) == 2
    assert calls[0][1]["offline"] is True
    class RejectTrainer(Trainer):
        def __init__(self, **kwargs):
            raise TypeError("private constructor detail must not leak")
    rejected = diagnose(payload, model_root=tmp_path, output_root=tmp_path,
        dependencies=lambda: {**deps(), "trainer_class": RejectTrainer})
    assert (rejected["status"], rejected["stage"], rejected["category"]) == ("REJECTED", "CONSTRUCTOR", "TYPE")
    assert "private" not in str(rejected)


@pytest.mark.parametrize("mode", ["hang", "flood"])
def test_owned_runner_terminates_real_hung_or_flooding_process(tmp_path: Path, mode: str):
    pid_file = tmp_path / "pid"
    action = "time.sleep(30)" if mode == "hang" else "while True: os.write(1, b'x' * 65536)"
    program = "import os,time\nopen(" + repr(str(pid_file)) + ", 'w').write(str(os.getpid()))\n" + action
    command = CommandSpec((sys.executable, "-c", program), dict(os.environ), 1, 1024)
    with pytest.raises(TrainingImageLockError):
        subprocess_runner(command)
    for _ in range(100):
        if pid_file.exists():
            break
        time.sleep(0.01)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("start_timeout", [False, True])
def test_probe_owns_container_terminal_and_cleanup(monkeypatch, tmp_path: Path, start_timeout: bool):
    import tuner.cloud.grpo_constructor_diagnostic as probe
    image_id = "sha256:" + "b" * 64
    container_id = "c" * 64
    profile = SimpleNamespace(canonical_sha256="sha256:" + "d" * 64,
        packaged_runtime={"python": {"purelib": "/opt/conda/lib/python3.11/site-packages"}})
    release = SimpleNamespace(manifest_digest="e" * 64, python_executable="/opt/conda/bin/python3")
    image = {"image_config_digest": image_id}
    admission = {"source_commit": "f" * 40, "probe_source_sha256": "1" * 64,
                 "gpu_device": 0, "model_inventory": []}
    monkeypatch.setattr(probe, "_admit", lambda **kw: (profile, release, image, admission, b"pass\n", b"{}\n"))
    monkeypatch.setattr(probe, "_validate_docker_inputs", lambda *a: SimpleNamespace(
        executable=Path("/usr/bin/docker"), config_directory=tmp_path))
    monkeypatch.setattr(probe, "_parse_inspect", lambda raw: (image_id, (),
        {"ai.synapticlabs.derived-training-profile": profile.canonical_sha256}))
    commands = []
    created = {}
    child = {"schema_version": "synaptic-grpo-constructor-probe-result/v1", "status": "REJECTED",
        "stage": "CONSTRUCTOR", "category": "TYPE", "class_digest": "2" * 64,
        "signature_digest": "3" * 64, "training_executed": False, "generation_executed": False}
    def run(_authority, _runner, spec):
        argv = spec.argv
        commands.append(argv)
        if "create" in argv:
            created["nonce"] = argv[argv.index("--name") + 1].removeprefix("synaptic-grpo-constructor-")
            return CommandResult((container_id + "\n").encode(), b"")
        if "start" in argv:
            if start_timeout:
                raise TrainingImageLockError("OPERATION_TIMEOUT")
            return CommandResult((container_id + "\n").encode(), b"")
        if "wait" in argv:
            return CommandResult(b"0\n", b"")
        if "container" in argv and "inspect" in argv:
            if container_id in argv:
                return CommandResult((json.dumps(container_id) + "\nfalse\n0\n" +
                    json.dumps({"ai.synapticlabs.probe-nonce": created["nonce"]}) + "\n").encode(), b"")
            nonce = argv[-1].removeprefix("synaptic-grpo-constructor-")
            return CommandResult((json.dumps(container_id) + "\n" +
                json.dumps({"ai.synapticlabs.probe-nonce": nonce}) + "\n").encode(), b"")
        if "logs" in argv:
            return CommandResult(probe._canonical_bytes(child), b"")
        if "rm" in argv:
            return CommandResult((container_id + "\n").encode(), b"")
        return CommandResult(b"", b"")
    monkeypatch.setattr(probe, "_run_authorized", run)
    output = tmp_path / "result.json"
    params = dict(admission_path=tmp_path / "admission.json", plan_ready_path=tmp_path / "plan.json",
        snapshot_root=tmp_path / "snapshot", recipe_path=tmp_path / "recipe.yaml",
        dataset_path=tmp_path / "data.jsonl",
        release_path=tmp_path / "release.json", capture_path=tmp_path / "capture.json",
        profile_path=tmp_path / "profile.yaml", build_receipt_path=tmp_path / "build.json",
        candidate_path=tmp_path / "candidate.json", image_verification_path=tmp_path / "image.json",
        docker=Path("/usr/bin/docker"), docker_config=tmp_path, output=output, execute=True)
    params["admission_path"].write_bytes(b"{}\n")
    if start_timeout:
        with pytest.raises(DerivedTrainingImageError, match="PROBE_COMMAND_TIMEOUT"):
            diagnose_derived_grpo_constructor(**params)
        assert not output.exists()
    else:
        report = diagnose_derived_grpo_constructor(**params)
        assert report["child"] == child
        assert json.loads(output.read_bytes()) == report
    assert output.with_suffix(".claim.json").exists()
    assert any("rm" in command and container_id in command for command in commands)
    create = next(command for command in commands if "create" in command)
    assert "--pull=never" in create
    assert "--network=none" in create
    assert "--read-only" in create
    assert "--gpus" in create
    assert "--memory=8g" in create


def test_cli_exposes_grpo_diagnostic_as_release_subcommand(monkeypatch, tmp_path: Path, capsys):
    from scripts import qualify_derived_training_image as cli
    import tuner.cloud.grpo_constructor_diagnostic as probe
    received = []
    monkeypatch.setattr(probe, "diagnose_derived_grpo_constructor",
        lambda **kwargs: received.append(kwargs) or {"status": "PLAN_ONLY"})
    flags = []
    for name in ("admission", "plan-ready", "snapshot", "recipe", "dataset",
                 "release", "config", "build-receipt", "candidate", "image-verification",
                 "final-runtime-capture", "docker", "docker-config", "output"):
        flags.extend(("--" + name, str(tmp_path / name)))
    assert cli.main(["release", "diagnose-grpo-constructor", *flags]) == 0
    assert received[0]["execute"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
